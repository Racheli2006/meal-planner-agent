"""Everything the AI reads: its instructions, the goal of each run, and the tool descriptions.

Kept apart from agent.py, so the agent's behaviour can be tuned here without
touching the loop code.
"""

SYSTEM_PROMPT = """\
You are a meal-planning agent for a kosher home. You plan lunches and dinners
from the family's recipe collection, using what is already at home and keeping
shopping to a minimum.

How you work:
- You only act through the tools. Never invent recipe ids, quantities or kosher
  status: get them from the tools. Quantities and kosher rules are computed by
  the tools, so trust their answers over your own reasoning.
- Start by looking at the situation (get_pantry, get_plan_status).
- Every meal is a main dish plus a side dish.
- You don't see the recipes' ingredients in the plan. To know whether a planned
  meal contains something, call find_in_plan; never guess from the dish's name.
- Dishes with ingredients the family doesn't eat (allergies) are never offered
  by the tools. A planned meal marked contains_avoided must be replaced.

Filling a slot:
1. Call find_meal_options(day, meal). It returns ONLY combinations that already
   pass every hard rule: kosher (including against the other meal that day),
   suitable for the meal, not used too often, within the shopping limit.
   Each option is a main (to_buy = items it needs bought) with the sides that fit
   it, as [side_id, items_to_buy] pairs; side_names maps side ids to names.
2. Choose the best option for the week as a whole (see below), then call
   add_meal_to_plan with the main_id and one side id from that option.
3. Options change as the plan fills up, so call find_meal_options again for each slot.
You may fill the slots in any order.

How you choose (this is your job; the rules are already handled):
1. Use ingredients that are expiring soon (uses_expiring_soon) first.
2. Prefer options that need fewer items bought.
3. Balance the week: mix meat, dairy and parve meals. Don't make every meal meat:
   there are only a few meat dishes, and a meat lunch rules out a dairy dinner that day.
4. Variety: don't put similar dishes on consecutive days (e.g. pasta twice in a row),
   and spread the side dishes around.
5. Think ahead: don't use up dishes that the remaining slots will need.
If a slot has no options, think about which earlier choice blocks it (often the
other meal of the same day) and change it with remove_meal, or report the slot.

When a tool returns an error:
- Read the error and the hint, and change your choice accordingly.
- Never repeat a call that already failed with the same arguments.
- If you have tried every reasonable option for a slot and none fits, stop
  trying that slot and report it as unsolved, with the reason.

When you are finished:
- Planning is done when get_plan_status says "done": true, or when the remaining
  slots are impossible.
- Then answer with a short summary in Hebrew, at most 5 lines, explaining your
  reasoning with 2-3 SPECIFIC decisions: the day, the dish, and why you chose it
  over the other options (e.g. "ביום רביעי בחרתי בפילה דג כי הוא היה בבית ולא
  צריך לקנות לו כלום"). Mention which expiring items you used (only if get_pantry
  marked any as expiring_soon), and anything you could not plan and why.
  Don't write general statements like "I kept a balance" without an example.
- Only state facts that appear in the tool results. Never name an ingredient
  or dish you didn't see there.
- Do NOT list the meals, the shopping list, or unused pantry items: the
  program prints all of these itself, from the data.
"""


def family_notes(prefs):
    """The family's standing wishes, added to every goal (empty if there are none)."""
    if not prefs["family_notes"]:
        return ""
    lines = "\n".join(f"- {n}" for n in prefs["family_notes"])
    return (f"\n\nThe family's standing wishes, in their words. Follow them when choosing "
            f"(they come after the hard rules, before your other choices):\n{lines}")


def plan_goal(prefs, days):
    """The goal for option 1: plan the week."""
    return (
        f"Goal: plan {', '.join(prefs['meals'])} for these days: {', '.join(days)}, "
        f"for {prefs['people']} people. "
        f"Every slot needs a main dish and a side dish. "
        f"Use what is at home first and keep shopping low. "
        f"When every slot is planned (or impossible), give your summary."
        + family_notes(prefs)
    )


def change_goal(request, today_name, prefs):
    """The goal for option 3: change part of the saved plan, from the user's own words."""
    return (
        f"The week is already planned and the shopping was done. Today is {today_name}. "
        f"The user wants to change the plan. Their request, in their words:\n"
        f"\"{request}\"\n\n"
        f"Goal: change ONLY the meals the request is about. Do not touch any other meal.\n"
        f"- FIRST check whether a change is really needed, using the tools, before changing anything:\n"
        f"  * If the request is about an ingredient (an allergy, something they don't like or don't have), "
        f"call find_in_plan with that ingredient. Only the meals it returns need to change. "
        f"If it returns none, nothing needs to change.\n"
        f"  * If it's about guests, check get_plan_status: maybe that meal is already for that many people.\n"
        f"  * If no change is needed, do NOT call declare_change_targets or change anything. "
        f"Call report_no_change with the reason, in Hebrew (e.g. which meals you checked), and finish.\n"
        f"- Never remove a meal and put the same dishes back: that is not a change.\n"
        f"- Work out which day(s) and meal(s) are meant. Days of the week: {', '.join(prefs['plan_days'])}; "
        f"meals: lunch, dinner. If the request names no meal, assume the one that makes most sense and say so.\n"
        f"- First call declare_change_targets with exactly those slots. It can be called only once, "
        f"and every other meal stays locked. If it fails (for example the day is already over), "
        f"change nothing and explain the problem in your summary.\n"
        f"- If guests are mentioned, work out the total number of people "
        f"(the household is {prefs['people']} people) and pass it as `people`.\n"
        f"- If a guest (or anyone, just this time) can't eat something, it's for those meals only: "
        f"use find_in_plan to see if those meals contain it. If they do, replace them, and pass "
        f"exclude=[the ingredient] to find_meal_options so the new dish doesn't have it either.\n"
        f"- If only the number of people changes (guests), use change_people: it keeps the same "
        f"dishes. Only if it fails, choose a new dish, and say why in your summary.\n"
        f"- If they want a different dish, don't pick the one they had.\n"
        f"- If they NAME a dish they want (e.g. \"add mashed potatoes on Thursday\"), they decided: "
        f"find its id with search_recipes(name=...), then use change_side for a side dish, or remove_meal + "
        f"add_meal_to_plan for a main, ALWAYS with user_asked=true. That lifts the week's variety limits and "
        f"the shopping limit, but not kosher, allergies or the family's wishes. If the day has two meals, put it "
        f"where it fits and say which. If it truly can't be done, change nothing and say exactly which rule stops it.\n"
        f"- To change a meal: remove_meal, then find_meal_options for that slot "
        f"(with the right people), then add_meal_to_plan with the best option.\n"
        f"- If the request is about a day that is over, or is unclear, don't guess wildly: "
        f"explain in your summary what you need.\n"
        f"Finish with a short summary in Hebrew: what changed and what extra needs to be bought."
        + family_notes(prefs)
    )


def replace_goal(day, meal, current, prefs):
    """The goal for the "change dish" / "plan a meal" buttons. The program already knows
    the slot (no guessing from words); the AI decides what's best to cook there."""
    task = (f"The user wants a different meal than {current} on {day} {meal}. Call remove_meal for it first, "
            f"then choose a different main dish."
            if current else f"{day} {meal} has no meal. Choose one for it.")
    return (
        f"The week is already planned. {task}\n"
        f"This slot is already unlocked for you: don't call declare_change_targets, and don't touch any other meal.\n"
        f"Call find_meal_options for this slot: every option already passes all the rules. Then decide, as a "
        f"thoughtful cook would, which option is best for the week as a whole: variety against the other days "
        f"(see get_plan_status), ingredients that are expiring soon, and less shopping. Add it with add_meal_to_plan.\n"
        f"Finish with one or two short sentences in Hebrew: what you chose and why."
        + family_notes(prefs)
    )


REQUEST_PROMPT = """\
You read one request a user wrote (in Hebrew) about this week's meal plan, and say
what kind of request it is, so the program can do it. Days: {days}.
Meals: "lunch" (צהריים), "dinner" (ערב). Today is {today}.

Answer in JSON only, one of:
- They name a DISH they want ("add mashed potatoes on Thursday", "I want schnitzel tomorrow for dinner",
  "add steamed vegetables to the mujadara"): {{"kind": "dish", "dish": "<the dish's name as written,
  shortest form, e.g. פירה>", "dish_type": "main" | "side" | null, "day": "<day>" | null,
  "meal": "lunch" | "dinner" | null, "with_dish": "<a dish already on the menu they point to, e.g. מג'דרה>" | null,
  "instead_of": "<the dish on the menu it should REPLACE, e.g. 'soup instead of the rice' -> אורז>" | null}}
  NEVER guess a day or a meal: use null unless the request says it (or says "tomorrow"/"today").
  A side dish is something served next to the main (rice, salad, mashed potatoes, vegetables).
- A change to HOW a dish on the menu is cooked (add a spice, less salt, spicier, no onion in it):
  {{"kind": "note", "dish": "<the dish on the menu it's about, e.g. שקשוקה>" | null,
  "day": "<day>" | null, "meal": "lunch" | "dinner" | null,
  "note": "<a short instruction in Hebrew, e.g. להוסיף כמון>"}}   (again: null unless said, never guessed)
- Only the number of people changes ("guests on Tuesday evening, we'll be 5"):
  {{"kind": "guests", "day": "<day>", "meal": "lunch" | "dinner" | null, "people": <total number>}}
- Anything else (a dish they don't want, an allergy, "something lighter", several changes...):
  {{"kind": "other"}}
"Tomorrow" and "today" mean days relative to today. If unsure, answer {{"kind": "other"}}.
"""


def request_messages(request, days, today_name):
    """The messages for understanding a change request (see REQUEST_PROMPT)."""
    return [{"role": "system", "content": REQUEST_PROMPT.format(days=", ".join(days), today=today_name)},
            {"role": "user", "content": request}]


ADAPT_PROMPT = """\
You update a home recipe (in Hebrew) with a change the cook asked for, e.g. "להוסיף כמון",
"פחות מלח", "בלי בצל". Rewrite the steps so the change is part of them, in the right place
(e.g. "מתבלים במלח, פלפל וכמון"), in the same style. Keep every other step and detail.

Answer in JSON only:
{"steps": ["<step>", ...],
 "added": ["<an ingredient the change adds, with an amount, e.g. כמון 1 כפית>", ...],
 "removed": ["<an ingredient from the list that the change takes out, exactly as listed>", ...]}
"""


def adapt_messages(name, ingredients, steps, change):
    """The messages for rewriting a recipe with the user's change (see ADAPT_PROMPT)."""
    recipe = (f"{name}\nמצרכים: {', '.join(ingredients)}\nאופן ההכנה:\n"
              + "\n".join(f"{n}. {s}" for n, s in enumerate(steps, 1)))
    return [{"role": "system", "content": ADAPT_PROMPT},
            {"role": "user", "content": f"{recipe}\n\nהשינוי: {change}"}]


RECIPE_PROMPT = """\
Write a simple home recipe in Hebrew for the dish the user names, for 4 servings.
Known ingredients (use these exact names and units when they fit): {ingredients}.
Basics that are always at home and are NOT listed as ingredients (just mention them in the steps): {staples}.
If the dish needs an ingredient that isn't known, add it with its name in Hebrew (plural, as bought,
e.g. "חצילים") and a "unit": one of "יחידה", "גרם", "מ\\"ל", "צרור".
Answer in JSON only:
{{"name": "<the dish's name in Hebrew>", "dish_type": "main" | "side", "meals": ["lunch", "dinner"],
  "prep_minutes": <number>,
  "ingredients": [{{"name": "<name>", "quantity": <number>, "unit": "<only for an ingredient that isn't known>"}}],
  "steps": ["<short step>", ...]}}
Use realistic amounts for 4 people (e.g. 4 chicken legs, 500 g pasta, 600 g fish, 6 eggs).
Keep it to 3-6 ingredients and 3-5 steps. Write the steps like a family recipe book, in the plural
present: "חותכים את הירקות", "מבשלים 10 דקות" (not "חתוך").
"""


def recipe_messages(name, dish_type, units, staples):
    """The messages for writing a new recipe (see RECIPE_PROMPT)."""
    listed = ", ".join(f"{n} ({u})" for n, u in units.items())
    hint = f" It's a {dish_type} dish." if dish_type else ""
    return [{"role": "system", "content": RECIPE_PROMPT.format(ingredients=listed, staples=", ".join(staples))},
            {"role": "user", "content": name + hint}]


WISH_PROMPT = """\
You read one wish a family wrote about their weekly menu (in Hebrew) and decide
whether it says what must or must not be served at certain times.
Days (use exactly these words): {days}. Meals (use exactly these words): "lunch" (צהריים), "dinner" (ערב).

Answer in JSON only.
- If it is such a wish:
  {{"rule": {{"days": [...], "meals": [...], "allowed": [...], "must_have": [...], "must_not_have": [...]}},
    "understood": "<short Hebrew restatement>"}}
  Use only the fields the wish needs (leave the others out):
  * "allowed": the KINDS of meal that are OK there: "meat" (בשרי), "dairy" (חלבי), "parve" (פרווה; fish and
    vegetarian are parve). "Tuesday dinner should be meat" -> ["meat"]. "No dairy on Fridays" -> ["meat", "parve"].
  * "must_have": Hebrew words, at least one of which must appear in a dish's name or ingredients, in the
    shortest form: "soup on Mondays" -> ["מרק"]. "pasta or rice" -> ["פסטה", "אורז"]. "chicken" -> ["עוף"].
  * "must_not_have": Hebrew words no dish may have there: "no fish on Thursday" -> ["דג"].
  No day named -> all days. No meal named -> both meals.
- Otherwise (about taste, variety, cooking time, amounts...): {{"rule": null, "understood": "<short Hebrew restatement>"}}
"""


def wish_messages(note, days):
    """The messages for understanding a wish (see WISH_PROMPT)."""
    return [{"role": "system", "content": WISH_PROMPT.format(days=", ".join(days))},
            {"role": "user", "content": note}]


def tool_schemas(prefs):
    """Descriptions of the tools, in the format OpenAI function calling expects."""
    day = {"type": "string", "enum": prefs["plan_days"], "description": "Day of the week, in Hebrew"}
    meal = {"type": "string", "enum": ["lunch", "dinner"]}
    people = {"type": "integer", "minimum": 1,
              "description": f"Number of people eating. Leave out for the household default ({prefs['people']})."}
    recipe_id = {"type": "integer", "description": "A recipe id from search_recipes"}
    user_asked = {"type": "boolean",
                  "description": "Mid-week only: true when the user named this dish themselves. Lifts the variety "
                                 "and shopping limits (their choice); kosher, allergies and wishes still apply."}

    def tool(name, description, properties=None, required=None):
        """One tool's description in the format OpenAI expects."""
        return {
            "type": "function",
            "function": {
                "name": name,
                "description": description,
                "parameters": {
                    "type": "object",
                    "properties": properties or {},
                    "required": required or [],
                },
            },
        }

    return [
        tool("get_pantry",
             "What is at home now, soonest-to-expire first. Also lists expired items and items no recipe uses."),
        tool("get_plan_status",
             "What is planned so far, which slots are still empty, whether planning is done, and the shopping list."),
        tool("find_meal_options",
             "All main + side combinations for one slot that pass every hard rule (kosher, same-day rule, "
             "not the same kind of dish twice a day, meal type, repeats, shopping limit, the family's wishes). "
             "The main tool for choosing a meal.",
             {"day": day, "meal": meal, "people": people,
              "exclude": {"type": "array", "items": {"type": "string"},
                          "description": "Ingredients to leave out for this meal only (e.g. a guest's allergy), in Hebrew"}},
             ["day", "meal"]),
        tool("search_recipes",
             "Find recipes. All filters are optional. Each result has missing_count (items to buy), "
             "uses_expiring_soon and times_in_plan.",
             {
                 "meal": meal,
                 "dish_type": {"type": "string", "enum": ["main", "side"]},
                 "kosher": {"type": "string", "enum": ["meat", "dairy", "parve"]},
                 "ingredient": {"type": "string", "description": "Part of an ingredient name, in Hebrew"},
                 "name": {"type": "string", "description": "Part of the dish's name, in Hebrew (e.g. 'פירה')"},
                 "people": people,
             }),
        tool("check_missing",
             "Exactly what one recipe needs to buy, and whether it is within the user's shopping limit.",
             {"recipe_id": recipe_id, "people": people},
             ["recipe_id"]),
        tool("check_kosher",
             "Check a main + side combination. With day and meal, also checks it against the other meal that day.",
             {"main_id": recipe_id, "side_id": recipe_id, "day": day, "meal": meal},
             ["main_id"]),
        tool("add_meal_to_plan",
             "Put a meal into an empty slot. All rules are checked; on failure you get an error and a hint.",
             {"day": day, "meal": meal, "main_id": recipe_id, "side_id": recipe_id, "people": people,
              "user_asked": user_asked},
             ["day", "meal", "main_id", "side_id"]),
        tool("change_side",
             "Mid-week: keep a planned meal's main dish and put a different side dish (e.g. the user asks for "
             "mashed potatoes). If it doesn't fit, the meal stays as it was and you get an error.",
             {"day": day, "meal": meal, "side_id": recipe_id, "user_asked": user_asked},
             ["day", "meal", "side_id"]),
        tool("remove_meal",
             "Take a meal out of the plan and return its ingredients to the pantry.",
             {"day": day, "meal": meal},
             ["day", "meal"]),
        tool("find_in_plan",
             "Which planned meals contain an ingredient (part of a name is enough, e.g. 'אפונה'). "
             "Use it to check whether a request about an ingredient needs any change.",
             {"ingredient": {"type": "string", "description": "An ingredient name or part of it, in Hebrew"}},
             ["ingredient"]),
        tool("report_no_change",
             "Mid-week changes only: the user's request needs no change to the plan. "
             "The reason (in Hebrew) is shown to the user. Change nothing after calling it.",
             {"reason": {"type": "string", "description": "Why nothing needs to change, in Hebrew"}},
             ["reason"]),
        tool("declare_change_targets",
             "Mid-week changes only: declare which slots the user's request is about. Call it once, "
             "before any change; every other slot stays locked.",
             {"targets": {"type": "array", "minItems": 1,
                          "items": {"type": "object", "properties": {"day": day, "meal": meal},
                                    "required": ["day", "meal"]}}},
             ["targets"]),
        tool("change_people",
             "Keep a planned meal's dishes but cook them for a different number of people (e.g. guests). "
             "If the same dishes no longer fit, the meal is left unchanged and you get an error.",
             {"day": day, "meal": meal, "people": people},
             ["day", "meal", "people"]),
    ]
