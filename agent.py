"""The agent loop: the AI decides, the tools act, the results go back to the AI.

    goal --> AI --> "call tool X" --> run X --> result --> AI --> ... --> "done" + summary
"""

import json
import os
import time
from pathlib import Path

import openai

from prompts import SYSTEM_PROMPT, adapt_messages, recipe_messages, request_messages, tool_schemas, wish_messages

MODEL = "gpt-4o-mini"
MAX_STEPS = 80      # a full week is ~10 slots x 3-4 calls; this leaves room for retries
MAX_NUDGES = 2      # times we push back when the AI stops with slots still empty
API_RETRIES = 3     # for network hiccups and temporary server errors

# Cost cap per run. Prices are gpt-4o-mini's list prices in dollars per million
# tokens; check https://openai.com/api/pricing if they change. The estimate
# ignores cache discounts, so the real cost is the same or lower.
PRICE_PER_M_INPUT = 0.15
PRICE_PER_M_OUTPUT = 0.60
MAX_COST_USD = 0.20

# Tools that only read: calling one again with the same arguments, while the plan
# hasn't changed, would return the same answer.
READ_ONLY_TOOLS = {"get_pantry", "get_plan_status", "find_meal_options",
                   "search_recipes", "check_missing", "check_kosher", "find_in_plan"}

# The whole conversation is resent on every step, so old, long tool results are
# the main cost. Only the latest few read results are kept in full; older ones
# (e.g. the options for a slot that's already filled) are replaced by a short note.
KEEP_RECENT_READS = 3
STALE_NOTE = json.dumps({"note": "older result removed to save space; call the tool again if you need it"})
# Never shrunk: the pantry is short and the AI needs it until the end. Without
# it, the AI once "remembered" pantry items that didn't exist in its summary.
PINNED_TOOLS = {"get_pantry"}


def estimate_cost(input_tokens, output_tokens):
    """Dollars for this many tokens, at the list prices above."""
    return (input_tokens * PRICE_PER_M_INPUT + output_tokens * PRICE_PER_M_OUTPUT) / 1_000_000


class AgentError(Exception):
    """A problem talking to OpenAI. The message is shown to the user."""


ENV_FILE = Path(__file__).parent / ".env"


def load_env_file(path=ENV_FILE):
    """Read KEY=value lines from the .env file (see .env.example) into the environment.
    A variable that is already set wins, and a missing file is fine."""
    if not Path(path).exists():
        return
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        key, sep, value = line.partition("=")
        if sep and key.strip() and not key.strip().startswith("#"):
            os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def make_client():
    """Create the OpenAI client. The key is the OPENAI_API_KEY environment variable,
    or the line OPENAI_API_KEY=... in the .env file. It's never written in the code."""
    load_env_file()
    try:
        return openai.OpenAI()
    except openai.OpenAIError:
        raise AgentError(
            "לא נמצא מפתח API. יש להעתיק את הקובץ .env.example לקובץ בשם .env ולכתוב בו את המפתח, "
            "או לשמור אותו במשתנה הסביבה OPENAI_API_KEY (ראו README)."
        )


def _call_openai(client, **request):
    """One call to the AI, retrying only errors that may pass by themselves
    (network, server busy). Anything else becomes an AgentError the user can act on."""
    for attempt in range(1, API_RETRIES + 1):
        try:
            return client.chat.completions.create(model=MODEL, **request)
        except openai.AuthenticationError:
            raise AgentError("מפתח ה-API לא תקין. כדאי ליצור מפתח חדש באתר של OpenAI.")
        except openai.RateLimitError as e:
            if "insufficient_quota" in str(e):
                raise AgentError("נגמר הקרדיט בחשבון OpenAI. אפשר לטעון קרדיט בעמוד ה-Billing.")
            error = e
        except (openai.APIConnectionError, openai.APITimeoutError) as e:
            error = e
        except openai.APIStatusError as e:
            if e.status_code < 500:
                raise AgentError(f"שגיאה מ-OpenAI ({e.status_code}): {e.message}")
            error = e
        if attempt < API_RETRIES:
            time.sleep(2 ** attempt)  # 2s, then 4s
    raise AgentError(f"אין חיבור ל-OpenAI אחרי {API_RETRIES} ניסיונות. כדאי לבדוק את האינטרנט. ({error})")


def _ask_model(client, messages, tools):
    """One step of the agent: the AI either calls a tool or answers in words."""
    return _call_openai(
        client,
        messages=messages,
        tools=tools,
        parallel_tool_calls=False,  # one action at a time, so each decision sees the last result
        temperature=0.2,            # steadier choices from run to run
    )


# How the AI may name things in a wish's rule, in our names.
MEAL_NAMES = {"lunch": "lunch", "dinner": "dinner", "צהריים": "lunch", "ערב": "dinner", "בערב": "dinner"}
KIND_NAMES = {"meat": "meat", "dairy": "dairy", "parve": "parve", "בשרי": "meat", "חלבי": "dairy", "פרווה": "parve"}
# Units a new ingredient may have (the same ones the pantry offers).
NEW_INGREDIENT_UNITS = ("יחידה", "גרם", 'מ"ל', "צרור")
# Words that say a meal in a request ("בצהריים", "צהרים", "בערב", "ארוחת ערב", "בלילה").
MEAL_WORDS = {"lunch": ("צהר",), "dinner": ("ערב", "לילה")}
# A new recipe's name can't start with these: they describe a meal, they don't name a dish.
GENERIC_NAME_WORDS = ("ארוחה", "ארוחת", "מנה", "משהו", "אוכל")


def _names(values, names, everything):
    """The values the AI gave, in our names. Nothing given: everything. Given but not
    understood: None, so an unclear answer never becomes a wider rule than asked."""
    if not values:
        return list(everything)
    values = values if isinstance(values, list) else [values]
    found = list(dict.fromkeys(names[v] for v in (str(x).strip().lower() for x in values) if v in names))
    return found if len(found) == len(values) else None


def _words(values):
    """The AI's words as a clean list: trimmed, no empties, no repeats."""
    values = values if isinstance(values, list) else [values] if values else []
    return list(dict.fromkeys(str(w).strip() for w in values if str(w).strip()))


def _clean_rule(answer, note, days):
    """The AI's rule, checked and in our names; None if it's unclear or limits nothing."""
    rule = {"note": note,
            "days": _names(answer.get("days"), {d: d for d in days} | {f"יום {d}": d for d in days}, days),
            "meals": _names(answer.get("meals"), MEAL_NAMES, ("lunch", "dinner")),
            "allowed": _names(answer.get("allowed"), KIND_NAMES, ()),
            "must_have": _words(answer.get("must_have")),
            "must_not_have": _words(answer.get("must_not_have"))}
    if None in rule.values():
        return None              # unclear: rather no rule than a wider one than asked
    if len(rule["allowed"]) == 3:
        rule["allowed"] = []     # allows every kind: no limit on kinds
    if not (rule["allowed"] or rule["must_have"] or rule["must_not_have"]):
        return None
    return {k: v for k, v in rule.items() if v}


def interpret_request(request, days, today_name, client=None):
    """One short AI call: is a change request a named dish for a day, or a number of
    people? Then the program does it exactly (the AI only understands the words).
    Returns {"kind": "dish", "dish", "dish_type", "day", "meal", "with_dish"} (day may be
    None when the request points to a dish on the menu instead), {"kind": "guests", "day",
    "meal", "people"} or {"kind": "other"} (then the full agent handles the request)."""
    response = _call_openai(client or make_client(), messages=request_messages(request, days, today_name),
                            temperature=0, response_format={"type": "json_object"})
    try:
        answer = json.loads(response.choices[0].message.content or "{}")
    except json.JSONDecodeError:
        return {"kind": "other"}
    if not isinstance(answer, dict):
        return {"kind": "other"}
    day = answer.get("day") if answer.get("day") in days else None
    meal = MEAL_NAMES.get(str(answer.get("meal") or "").strip())
    if meal and not any(w in request for w in MEAL_WORDS[meal]):
        meal = None  # the AI guessed a meal the user didn't say: then the user is asked
    with_dish = str(answer.get("with_dish") or "").strip() or None
    instead_of = str(answer.get("instead_of") or "").strip() or None
    if answer.get("kind") == "dish" and str(answer.get("dish") or "").strip() and (day or with_dish or instead_of):
        return {"kind": "dish", "dish": str(answer["dish"]).strip(), "day": day, "meal": meal,
                "with_dish": with_dish, "instead_of": instead_of,
                "add": answer.get("add") is True and not instead_of,  # "instead of" always takes a place
                "dish_type": answer.get("dish_type") if answer.get("dish_type") in ("main", "side") else None}
    note = str(answer.get("note") or "").strip()
    dish = str(answer.get("dish") or "").strip() or None
    if answer.get("kind") == "note" and note and (dish or day):
        return {"kind": "note", "note": note, "dish": dish, "day": day, "meal": meal}
    people = answer.get("people")
    if day is None:
        return {"kind": "other"}
    if answer.get("kind") == "guests" and isinstance(people, int) and not isinstance(people, bool) and people > 0:
        return {"kind": "guests", "day": answer["day"], "meal": meal, "people": people}
    kind = KIND_NAMES.get(str(answer.get("meal_kind") or "").strip())
    if answer.get("kind") == "meal_kind" and kind:  # "a meat meal on Wednesday": no dish named
        return {"kind": "meal_kind", "meal_kind": kind, "day": day, "meal": meal,
                "new": answer.get("new") is True}  # "one that isn't in the recipe book"
    return {"kind": "other"}


def adapt_recipe(name, ingredients, steps, change, client=None):
    """A short AI call: the recipe rewritten with the change the user asked for, so the
    change is part of the steps, not a note beside them. Checked here: no step may get
    lost, and only listed ingredients can be taken out. Returns {"steps", "added",
    "removed"}, or None if the answer can't be trusted (then the change stays a note)."""
    response = _call_openai(client or make_client(), temperature=0, response_format={"type": "json_object"},
                            messages=adapt_messages(name, ingredients, steps, change))
    try:
        answer = json.loads(response.choices[0].message.content or "{}")
    except json.JSONDecodeError:
        return None
    if not isinstance(answer, dict):
        return None
    new_steps = [str(s).strip() for s in answer.get("steps") or [] if str(s).strip()]
    if len(new_steps) < len(steps):  # a step got lost
        return None
    names = [i.rsplit(" ", 2)[0] for i in ingredients]  # "בצל 1 יח'" -> "בצל"
    removed = [str(r).strip() for r in answer.get("removed") or [] if str(r).strip()]
    removed = [next((n for n in names if n in r or r in n), None) for r in removed]
    if None in removed:
        return None
    return {"steps": new_steps, "added": _words(answer.get("added")), "removed": removed}


def create_recipe(name, dish_type, recipes_data, client=None, kosher=None):
    """A short AI call for a dish the recipe book doesn't have ("ירקות מאודים"): a simple
    recipe, from the known ingredients where they fit. Everything the AI wrote is checked
    here; what can't be trusted (kosher, fish) is worked out from the ingredients.
    Basics (oil, salt, paprika...) are dropped from the ingredients: they're always at home.
    An ingredient the program doesn't know yet is kept if it has a known unit; it's
    returned in "new_units" ({name: unit}) to be added to the catalog.
    kosher: the kind of dish asked for ("a new dairy meal"); a recipe of another kind is refused.
    Returns the recipe without an id, or None if it couldn't be made."""
    units = recipes_data["ingredient_units"]
    staples = recipes_data.get("staples", [])
    response = _call_openai(client or make_client(), temperature=0.3, response_format={"type": "json_object"},
                            messages=recipe_messages(name, dish_type, units, staples, kosher,
                                                     known=[r["name"] for r in recipes_data["recipes"]
                                                            if kosher and r["kosher"] == kosher]))
    try:
        answer = json.loads(response.choices[0].message.content or "{}")
    except json.JSONDecodeError:
        return None
    if not isinstance(answer, dict) or not str(answer.get("name") or "").strip():
        return None
    if str(answer["name"]).split()[0] in GENERIC_NAME_WORDS:
        return None  # "ארוחה בשרית" describes a meal, it isn't a dish's name in a recipe book
    ingredients, new_units = [], {}
    for item in answer.get("ingredients") or []:
        if not isinstance(item, dict) or not str(item.get("name") or "").strip():
            return None
        ing = str(item["name"]).strip()
        if any(s in ing or ing in s for s in staples) and ing not in units:
            continue  # a basic: always at home, mentioned in the steps
        if ing not in units:
            unit = str(item.get("unit") or "").strip()
            if unit not in NEW_INGREDIENT_UNITS:
                return None  # an unknown ingredient with no usable unit: rather no recipe than a wrong one
            new_units[ing] = unit
        qty = item.get("quantity")
        if isinstance(qty, bool) or not isinstance(qty, (int, float)) or qty <= 0:
            return None
        ingredients.append({"name": ing, "quantity": qty})
    steps = [str(s).strip() for s in answer.get("steps") or [] if str(s).strip()]
    if not ingredients or not steps:
        return None
    names = [i["name"] for i in ingredients]
    meat = any(w in n for n in names for w in ("בשר", "עוף", "פרגיות"))
    dairy = any(w in n for n in names for w in ("גבינ", "חלב", "חמאה", "שמנת", "קוטג", "יוגורט"))
    if meat and dairy:
        return None  # not kosher
    if kosher and kosher != ("meat" if meat else "dairy" if dairy else "parve"):
        return None  # asked for a dairy meal, and the recipe isn't one
    meals =[m for m in answer.get("meals") or [] if m in ("lunch", "dinner")] or ["lunch", "dinner"]
    minutes = answer.get("prep_minutes")
    return {"name": str(answer["name"]).strip(),
            "dish_type": dish_type or (answer.get("dish_type") if answer.get("dish_type") in ("main", "side") else "side"),
            "kosher": "meat" if meat else "dairy" if dairy else "parve",
            "contains_fish": any("דג" in n for n in names),
            "meals": meals,
            "prep_minutes": minutes if isinstance(minutes, int) and 0 < minutes < 600 else 30,
            "servings": 4, "ingredients": ingredients, "steps": steps, "new_units": new_units}


def interpret_wish(note, days, client=None):
    """One short AI call when a wish is saved: is it a rule about what goes where?
    Returns (rule or None, what was understood, in Hebrew). The rule is checked in
    code, so a bad answer can't put nonsense into the preferences."""
    response = _call_openai(client or make_client(), messages=wish_messages(note, days),
                            temperature=0, response_format={"type": "json_object"})
    try:
        answer = json.loads(response.choices[0].message.content or "{}")
    except json.JSONDecodeError:
        return None, ""
    if not isinstance(answer, dict):  # asked for an object; anything else is not understood
        return None, ""
    understood = str(answer.get("understood") or "")
    rule = answer.get("rule")
    return (_clean_rule(rule, note, days) if isinstance(rule, dict) else None), understood


def _run_tool(planner, allowed, name, raw_args):
    """Run one tool the AI asked for. Never raises: any problem becomes an error
    the AI can read and react to."""
    if name not in allowed:
        return {"error": f"unknown tool '{name}'"}
    try:
        args = json.loads(raw_args or "{}")
    except json.JSONDecodeError:
        return {"error": "arguments were not valid JSON"}
    if not isinstance(args, dict):
        return {"error": "arguments must be a JSON object"}
    try:
        return getattr(planner, name)(**args)
    except TypeError as e:  # wrong or missing argument names
        return {"error": f"bad arguments for {name}: {e}"}
    except Exception as e:  # a bug in a tool must not crash the whole run
        return {"error": f"internal error in {name}: {type(e).__name__}: {e}"}


def _describe(name, args, result):
    """One short log line per step, so you can follow the agent's decisions. When the AI
    gave a reason for a change, it's shown under it: the "why" of the decision."""
    args = args if isinstance(args, dict) else {}
    reason = str(args.get("reason") or "").strip()
    shown = ", ".join(f"{k}={v}" for k, v in args.items() if k != "reason")
    if "note" in result:
        outcome = "(repeat, nothing new)"
    elif "error" in result:
        outcome = f"✗ {result['error']}"
    elif name == "add_meal_to_plan":
        outcome = f"✓ {result['main']} + {result['side']}  (slots left: {result['slots_left']})"
    elif name == "change_side":
        outcome = f"✓ {result['main']}: {result['side_before']} -> {result['side']}"
    elif name == "change_people":
        outcome = f"✓ kept {result['kept']}, {result['people_before']} -> {result['people']} people"
    elif name == "search_recipes":
        outcome = f"{result['count']} found"
    elif name == "find_meal_options":
        outcome = f"{result['count']} valid mains"
    elif name == "check_missing":
        outcome = f"missing {result['missing_count']} (limit {result['max_missing_allowed']})"
    elif name == "check_kosher":
        outcome = "kosher ✓" if result["ok"] else f"not kosher: {result['problems']}"
    elif name == "find_in_plan":
        outcome = f"in {result['count']} planned dishes"
    elif name == "get_plan_status":
        outcome = f"{len(result['planned'])} planned, {len(result['empty_slots'])} empty"
    else:
        outcome = "ok"
    why = f"\n      למה: {reason}" if reason and "error" not in result and "note" not in result else ""
    return f"{name}({shown}) -> {outcome}{why}"


def _summary_from_facts(client, messages, planner):
    """The plan is complete: the AI writes its summary again, given the final plan as facts.
    A summary written from memory once said a day "couldn't be planned" when it was. One
    short call, no tools. Returns (summary or None if it failed, the call's usage or None)."""
    planned = [f"{p['day']} {p['meal']}: {p['main']} + {p['side']}" for p in planner.get_plan_status()["planned"]]
    check = ("Final check before your summary. These are the facts: EVERY slot is planned:\n"
             + "\n".join(planned)
             + "\nWrite your summary again (same rules: Hebrew, at most 5 lines, 2-3 specific decisions and why). "
               "Don't say any meal couldn't be planned.")
    try:
        response = _call_openai(client, messages=messages + [{"role": "user", "content": check}], temperature=0.2)
    except AgentError:
        return None, None
    return (response.choices[0].message.content or "").strip() or None, response.usage


def run_agent(planner, goal, client=None, require_done=False, log=print):
    """Run the loop until the AI finishes, or MAX_STEPS is reached.

    require_done: when planning a week, the AI isn't allowed to stop while slots
    are still empty without being asked once more (up to MAX_NUDGES times).
    Returns {"finished", "summary", "steps", "tokens", "cost", "reason"}.
    """
    client = client or make_client()
    tools = tool_schemas(planner.prefs)
    allowed = {t["function"]["name"] for t in tools}
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": goal},
    ]
    failed_calls = set()  # (tool, arguments, state) that already returned an error
    seen_reads = set()    # (tool, arguments, plan) for read-only calls already answered
    read_results = []     # (tool message, its seen_reads key), oldest first
    nudges = 0
    input_tokens = output_tokens = 0

    def outcome(finished, summary, step, reason):
        """The run's result, with what it cost so far."""
        return {"finished": finished, "summary": summary, "steps": step,
                "tokens": input_tokens + output_tokens,
                "cost": estimate_cost(input_tokens, output_tokens), "reason": reason}

    for step in range(1, MAX_STEPS + 1):
        # --- Stop condition: the run is getting too expensive ---
        if estimate_cost(input_tokens, output_tokens) >= MAX_COST_USD:
            return outcome(False, None, step - 1, f"נעצר בתקרת העלות (${MAX_COST_USD:.2f})")

        # Shrink old read results. Their keys are forgotten too, so asking again
        # gets a real answer rather than "use the earlier result".
        while len(read_results) > KEEP_RECENT_READS:
            old_message, old_key = read_results.pop(0)
            old_message["content"] = STALE_NOTE
            seen_reads.discard(old_key)

        response = _ask_model(client, messages, tools)
        if response.usage:
            input_tokens += response.usage.prompt_tokens
            output_tokens += response.usage.completion_tokens
        message = response.choices[0].message

        # --- Stop condition: the AI answers in words instead of calling a tool ---
        if not message.tool_calls:
            messages.append({"role": "assistant", "content": message.content or ""})
            empty = planner.empty_slots()
            if require_done and empty and nudges < MAX_NUDGES:
                nudges += 1
                log(f"[{step}] the AI stopped with {len(empty)} empty slots; asking it to continue")
                messages.append({"role": "user", "content": (
                    f"These slots are still empty: {empty}. Keep planning them. "
                    f"If a slot truly can't be filled, say which one and why."
                )})
                continue
            incomplete = bool(require_done and empty)
            summary = message.content
            if require_done and not incomplete:
                rewritten, usage = _summary_from_facts(client, messages, planner)
                summary = rewritten or summary
                if usage:  # counted in the run's cost like every other call
                    input_tokens += usage.prompt_tokens
                    output_tokens += usage.completion_tokens
            return outcome(not incomplete, summary, step,
                           "חלק מהארוחות לא תוכננו" if incomplete else None)

        # --- Otherwise: run the tool(s) the AI chose, and give it the results ---
        messages.append({
            "role": "assistant",
            "content": message.content,
            "tool_calls": [c.model_dump() for c in message.tool_calls],
        })
        for call in message.tool_calls:
            name, raw_args = call.function.name, call.function.arguments
            plan_state = json.dumps(planner.plan, sort_keys=True)
            # A failed call is blocked only while nothing changed: after the AI declares
            # its slots or changes the plan, the same call may now work.
            key = (name, raw_args, plan_state, repr(sorted(planner.change_targets or [])))
            read_key = (name, raw_args, plan_state)
            if key in failed_calls:
                result = {"error": "you already tried exactly this and it failed",
                          "hint": "Change your choice (another recipe, side, or slot) instead of repeating it."}
            elif name in READ_ONLY_TOOLS and read_key in seen_reads:
                # Short answer instead of the full result again: saves tokens and
                # tells the AI it is going in circles.
                result = {"note": "same call as before and nothing changed since: use that earlier result and decide"}
            else:
                result = _run_tool(planner, allowed, name, raw_args)
                if "error" in result:
                    failed_calls.add(key)
                elif name in READ_ONLY_TOOLS:
                    seen_reads.add(read_key)
            try:
                args = json.loads(raw_args or "{}")
            except json.JSONDecodeError:
                args = {}
            log(f"[{step}] {_describe(name, args, result)}")
            tool_message = {
                "role": "tool",
                "tool_call_id": call.id,
                "content": json.dumps(result, ensure_ascii=False, default=str),
            }
            messages.append(tool_message)
            if (name in READ_ONLY_TOOLS and name not in PINNED_TOOLS
                    and "error" not in result and "note" not in result):
                read_results.append((tool_message, read_key))

    # --- Stop condition: out of steps ---
    return outcome(False, None, MAX_STEPS, f"נעצר אחרי {MAX_STEPS} צעדים בלי לסיים")
