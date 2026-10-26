"""The meal planner in the browser: the week's menu, the pantry and the shopping
list, with everything you update during the week.

    py web.py                  opens http://localhost:8765 in the browser
    py web.py --date 2026-10-27    pretend today is another date (for testing)

It runs only on this computer. The AI is used only for planning and for changing
a dish; everything else (pantry, shopping, guests, a meal you didn't cook) just
updates the data files.
"""

import argparse
import copy
import json
import sys
import threading
import webbrowser
from datetime import date
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import agent
import household
import main
import tools
from prompts import change_goal, plan_goal, replace_goal

PAGE = Path(__file__).parent / "web" / "index.html"
LOCK = threading.Lock()  # one change at a time: the data files have a single writer
TODAY = None             # set by --date; None means the real date


def today():
    """The app's date: the real one, or the one given with --date."""
    return TODAY or date.today()


# ---------- What the page shows ----------

def meal_view(planner, day, meal):
    """One meal as the page shows it: dishes, kind, people, what's left to buy, and the recipes."""
    entry = planner.plan.get(day, {}).get(meal)
    view = {"meal": meal, "label": main.MEAL_NAMES[meal]}
    if entry is None:
        return {**view, "empty": True}
    ids = [entry["main"]] + ([entry["side"]] if entry["side"] else [])
    return {**view,
            "main": planner.recipes[entry["main"]]["name"],
            "side": planner.recipes[entry["side"]]["name"] if entry["side"] else None,
            "kosher": planner.meal_kosher_type(ids),
            "people": entry["people"],
            "to_buy": [main.amount(q, planner.units[n]) + " " + n for n, q in entry["to_buy"].items()],
            "recipes": [recipe_view(planner, entry, i) for i in ids],
            "notes": [n["text"] for n in entry.get("notes", [])]}


def recipe_view(planner, entry, dish_id):
    """A dish's recipe as this meal will cook it: ingredients scaled to the meal's people,
    and the changes the user asked for already in the steps and the ingredients."""
    r = planner.recipes[dish_id]
    steps, added, removed = household.meal_recipe(planner, entry, dish_id)
    notes = [n["text"] for n in entry.get("notes", []) if n["dish"] == dish_id]
    return {"name": r["name"], "minutes": r["prep_minutes"], "servings": r["servings"],
            "ingredients": [{"text": f"{i['name']} {main.amount(i['quantity'], i['unit'])}",
                             "removed": i["name"] in removed}
                            for i in planner.scaled_ingredients(r, entry["people"])]
                           + [{"text": a, "added": True} for a in added],
            "steps": steps,
            # The changes asked for; when the recipe couldn't be rewritten, they're shown as steps.
            "notes": notes, "rewritten": str(dish_id) in entry.get("custom", {})}


def state():
    """Everything the page needs, worked out from the data files."""
    now = today()
    recipes = tools.load_recipes()
    prefs = tools.load_preferences()
    this_week = tools.week_start_for(now)
    saved = tools.load_plan(recipes) if tools.PLAN_FILE.exists() else None
    items = household.read_pantry()

    if saved and saved["week_start"] == this_week:
        mode = "week"
        planner = main.load_saved_planner(now)
    else:
        mode = "new_week" if saved and saved["week_start"] < this_week else "no_plan"
        _, planner = main.load_planner(now)

    result = {
        "mode": mode,
        "today": now.isoformat(),
        "today_name": tools.hebrew_weekday(now),
        "week_start": f"{this_week:%d/%m/%Y}",
        "people": prefs["people"],
        "settings": {"people": prefs["people"], "plan_days": prefs["plan_days"], "meals": prefs["meals"],
                     "all_days": SETTING_DAYS, "max_people": tools.MAX_PEOPLE},
        "avoid": prefs["avoid_ingredients"],
        # Each wish with what the AI understood, and whether the code checks it as a rule.
        "family_notes": [{"note": n, "understood": (prefs.get("understood") or {}).get(n, ""),
                          "rule": any(r["note"] == n for r in prefs["slot_rules"])}
                         for n in prefs["family_notes"]],
        "to_fix": [],
        "units": recipes["ingredient_units"],
        "unit_choices": household.UNIT_CHOICES,
        "days": [],
        "shopping": [],
        "notes": {},
    }

    if mode == "new_week":
        result["last_week"] = f"{saved['week_start']:%d/%m}"
        result["proposed"] = household.pantry_after_week(items, saved["plan"])
        return result

    for day in prefs["plan_days"]:
        result["days"].append({
            "name": day,
            "date": f"{planner.day_date(day):%d/%m}",
            "past": planner.is_past(day),
            "today": planner.day_date(day) == now,
            "meals": [meal_view(planner, day, m) for m in prefs["meals"]],
        })
    result["shopping"] = planner.shopping_list() if mode == "week" else []
    if mode == "week":
        result["to_fix"] = [{**s, "meal": main.MEAL_NAMES[s["meal"]]} for s in household.slots_to_fix(planner)]
    pantry = planner.get_pantry()
    result["notes"] = {"expired": pantry["expired_not_used"], "unused": pantry["not_in_any_recipe"]}

    free = {name: item["quantity"] for name, item in planner.available.items()}
    result["pantry"] = [{
        **item,
        "free": free.get(item["name"], 0),
        "expired": item["name"] in planner.expired,
        "in_recipes": item["name"] in planner.used_in_recipes,
    } for item in items]
    return result


# ---------- Actions ----------

def run_ai(planner, goal, require_done=False):
    """Run the agent on a goal, save what it did, and keep its steps for the page."""
    log = []
    result = agent.run_agent(planner, goal, require_done=require_done, log=log.append)
    main.save_outputs(planner)
    return result, log


def ai_answer(result, log, message=None):
    """The agent's run as a message for the page: what changed, its summary, and its steps."""
    lines = [message] if message else []
    if result["summary"]:
        lines.append(result["summary"])
    if not result["finished"]:
        lines.append(f"⚠️ האגנט לא השלים את המשימה: {result['reason']}")
    return {"message": "\n\n".join(lines), "log": log}


def do_plan(body):
    """Plan the week with the agent; the days that are over keep their meals."""
    recipes, planner = main.load_planner(today())
    saved = main.saved_plan_this_week(recipes, today())
    if saved:
        household.keep_past_days(planner, saved)
    days = [d for d in planner.prefs["plan_days"] if not planner.is_past(d)]
    if not days:
        raise tools.DataError("כל ימי התכנון של השבוע כבר עברו.")
    result, log = run_ai(planner, plan_goal(planner.prefs, days), require_done=True)
    # Whatever the AI couldn't fit to a wish, the program fits (relaxing if it must).
    stuck = household.fix_menu(planner, fill_empty=True)
    main.save_outputs(planner)
    answer = ai_answer(result, log)
    answer["message"] += stuck_text(stuck)
    return answer


def do_change(body):
    """A request in the user's words. A short AI call first says what it is: a dish named
    for a day, or a number of people. Those the program does exactly. Anything else goes
    to the agent, which works out the steps itself."""
    request = str(body.get("request") or "").strip()
    if not request:
        raise tools.DataError("צריך לכתוב מה לשנות.")
    planner = main.load_saved_planner(today())
    intent = agent.interpret_request(request, planner.prefs["plan_days"], tools.hebrew_weekday(today()))
    if intent["kind"] == "dish":
        return put_named_dish(planner, intent)
    if intent["kind"] == "note":
        return put_meal_note(planner, intent)
    if intent["kind"] == "guests" and intent["meal"]:
        return do_guests({"day": intent["day"], "meal": intent["meal"], "people": intent["people"]})

    before = {(d, m): dict(e) for d, meals in planner.plan.items() for m, e in meals.items()}
    goal = change_goal(request, tools.hebrew_weekday(today()), planner.prefs)
    result, log = run_ai(planner, goal)
    result["summary"] = main.change_summary(planner, before, result["summary"])
    return ai_answer(result, log, main.changes_text(planner, before))


def put_named_dish(planner, intent):
    """'Add mashed potatoes on Thursday' / 'add steamed vegetables to the mujadara': exactly
    that dish, or the reason it can't be. A dish the recipe book doesn't have: the user
    chooses between a new recipe for it and a similar dish (buttons on the page)."""
    name, meal = intent["dish"], intent.get("meal")
    day, role = intent.get("day"), intent.get("role")
    if intent.get("instead_of"):  # "soup instead of the rice": in the rice's place, the main stays
        found_at = household.place_of(planner, intent["instead_of"], day, meal)
        if not found_at:
            raise tools.DataError(f"לא מצאתי בתפריט של הימים הבאים את \"{intent['instead_of']}\".")
        day, meal = found_at
        entry = planner.plan[day][meal]
        side_name = planner.recipes[entry["side"]]["name"] if entry["side"] is not None else ""
        role = "side" if side_name and tools.word_in(intent["instead_of"], side_name) else "main"
    if intent.get("with_dish"):  # "to the mujadara": the meal that has it
        found_at = household.place_of(planner, intent["with_dish"], day, meal)
        if not found_at:
            raise tools.DataError(f"לא מצאתי בתפריט של הימים הבאים את \"{intent['with_dish']}\".")
        day, meal = found_at
    if planner.is_past(day):
        raise tools.DataError(f"יום {day} כבר עבר, אז אי אפשר לשנות אותו.")
    if household.find_dish(planner, name) is None:
        target = {"day": day, "meal": meal}
        where = f"ל{main.MEAL_NAMES[meal]} ביום {day}" if meal else f"ליום {day}"
        choices = [{"label": f"ליצור מתכון חדש: {name}", "path": "/api/put-dish",
                    "body": {**target, "dish": name, "create": True, "dish_type": intent.get("dish_type")}}]
        choices += [{"label": other, "path": "/api/put-dish", "body": {**target, "dish": other}}
                    for other in household.dishes_that_fit(planner, day, meal, name, intent.get("dish_type"))]
        others = "או לבחור מנה אחרת שמתאימה " + where + ":" if len(choices) > 1 else "(אין כרגע מנה אחרת שמתאימה " + where + ")."
        return {"message": f"אין במתכונים מנה בשם \"{name}\". אפשר ליצור לה מתכון חדש, {others}",
                "choices": choices}
    before = {(d, m): dict(e) for d, meals in planner.plan.items() for m, e in meals.items()}
    # A name that fits several recipes ("מרק"): the user picks among those that can go there.
    named = household.dishes_named(planner, name)
    if len(named) > 1 and named[0]["name"] != name:
        target = {"day": day, "meal": meal, "role": role}
        usage = planner.dish_usage()  # dishes not on the menu yet this week come first
        fits = [r["name"] for r in sorted(named, key=lambda r: usage[r["id"]] > 0)
                if household.put_dish(copy.deepcopy(planner), day, meal, r["name"], role)[0]]
        if len(fits) > 1:
            return {"message": f"יש כמה מנות בשם \"{name}\" שמתאימות. איזו לשים?",
                    "choices": [{"label": n, "path": "/api/put-dish", "body": {**target, "dish": n}} for n in fits]}
        if fits:
            name = fits[0]
    ok, meal, found = household.put_dish(planner, day, meal, name, role)
    if not ok:
        raise tools.DataError(f"לא הצלחתי לשים \"{name}\" ביום {day}: {'; '.join(found)}.")
    if not main.changed_slots(planner, before) and before == {(d, m): dict(e) for d, meals in planner.plan.items()
                                                              for m, e in meals.items()}:
        return {"message": f"{found['name']} כבר מתוכנן ביום {day} ב{main.MEAL_NAMES[meal]}."}
    main.save_outputs(planner)
    answer = with_review(f"שמתי {found['name']} ביום {day} ב{main.MEAL_NAMES[meal]}, כמו שביקשת.\n\n"
                         f"{main.changes_text(planner, before)}")
    # A new main: does the side still go with it? (Crepes with white rice...) The user decides.
    entry = main.load_saved_planner(today()).plan.get(day, {}).get(meal)
    if (role or found["dish_type"]) == "main" and entry and entry["side"] is not None:
        side = planner.recipes[entry["side"]]["name"]
        others = household.sides_that_fit(main.load_saved_planner(today()), day, meal)
        answer["message"] += f"\n\nהתוספת עכשיו היא {side}. להשאיר אותה, או להחליף לתוספת שמתאימה יותר?"
        answer["choices"] = [{"label": f"להשאיר את {side}", "dismiss": f"בסדר, {side} נשאר."}] + [
            {"label": other, "path": "/api/put-dish", "body": {"dish": other, "day": day, "meal": meal}}
            for other in others]
    return answer


def put_meal_note(planner, intent):
    """'Add cumin to the shakshuka': a note on that meal, shown with its recipe. The user can
    also keep it in the recipe for next times (a button)."""
    day, meal, dish = intent.get("day"), intent.get("meal"), intent.get("dish")
    if dish:
        everywhere = household.slots_with(planner, dish)
        if day is None and len(everywhere) > 1:
            # The dish is on the menu more than once and no day was said: the user picks.
            note = intent["note"]
            choices = [{"label": f"יום {d} ב{main.MEAL_NAMES[m]}", "path": "/api/meal-note",
                        "body": {"places": [[d, m]], "dish": dish, "note": note}} for d, m in everywhere]
            choices.append({"label": "בכל הפעמים השבוע", "path": "/api/meal-note",
                            "body": {"places": [list(p) for p in everywhere], "dish": dish, "note": note}})
            return {"message": f"{dish} מופיע השבוע {len(everywhere)} פעמים. לאיזו ארוחה להוסיף \"{note}\"?",
                    "choices": choices}
        place = household.place_of(planner, dish, day, meal)
        if not place:
            raise tools.DataError(f"לא מצאתי בתפריט של הימים הבאים את \"{dish}\".")
        day, meal = place
    elif meal is None:
        raise tools.DataError(f"לאיזו ארוחה ביום {day}? אפשר לכתוב למשל \"ביום {day} בערב\" או את שם המנה.")
    entry = planner.plan.get(day, {}).get(meal)
    if entry is None or planner.is_past(day):
        raise tools.DataError("אפשר להוסיף הערה רק לארוחה מתוכננת ביום שעוד לא עבר.")
    # The dish it's about: the one named, or the main.
    dish_id = next((i for i in (entry["main"], entry["side"])
                    if i is not None and dish and tools.word_in(dish, planner.recipes[i]["name"])), entry["main"])
    # The AI writes the change into this meal's recipe; if that fails, it's shown as a last step.
    view = recipe_view(planner, entry, dish_id)
    try:
        adapted = agent.adapt_recipe(view["name"], [i["text"] for i in view["ingredients"] if not i.get("removed")],
                                     view["steps"], intent["note"])
    except agent.AgentError:
        adapted = None
    household.add_meal_note(planner, day, meal, dish_id, intent["note"], adapted)
    main.save_outputs(planner)
    name = planner.recipes[dish_id]["name"]
    done = "המתכון של הארוחה הזאת עודכן" if adapted else "זה מופיע כצעד אחרון במתכון של הארוחה הזאת"
    saved = household.meal_recipe(planner, planner.plan[day][meal], dish_id)
    return {"message": f"📝 {name} ביום {day} ב{main.MEAL_NAMES[meal]}: {intent['note']}. {done}.",
            "choices": [{"label": f"לשמור גם במתכון של {name} לפעמים הבאות", "path": "/api/recipe-note",
                         "body": {"recipe": dish_id, "note": intent["note"],
                                  "adapted": {"steps": saved[0], "added": saved[1], "removed": saved[2]}
                                  if adapted else None}}]}


def do_meal_note(body):
    """The user's pick after "which meal?": the note on one meal, or on every one listed."""
    places, dish, note = body.get("places") or [], str(body.get("dish") or ""), str(body.get("note") or "").strip()
    if not places or not note:
        raise tools.DataError("הבקשה לא תקינה.")
    planner = main.load_saved_planner(today())
    answer = None
    for day, meal in places:
        answer = put_meal_note(planner, {"note": note, "dish": dish, "day": day, "meal": meal})
    if len(places) > 1:
        answer["message"] = f"📝 הוספתי \"{note}\" ל{dish} בכל {len(places)} הפעמים השבוע."
    return answer


def do_recipe_note(body):
    """Keep a change in the recipe book for next times (the button after a meal note)."""
    recipe_id, note = body.get("recipe"), str(body.get("note") or "").strip()
    if not isinstance(recipe_id, int) or not note:
        raise tools.DataError("הבקשה לא תקינה.")
    adapted = body.get("adapted")
    if adapted is not None and not (isinstance(adapted, dict) and isinstance(adapted.get("steps"), list)
                                    and all(isinstance(s, str) and s.strip() for s in adapted["steps"])):
        raise tools.DataError("הבקשה לא תקינה.")
    if adapted:
        adapted = {"steps": adapted["steps"], "added": [str(a) for a in adapted.get("added") or []],
                   "removed": [str(r) for r in adapted.get("removed") or []]}
    name = household.add_recipe_step(recipe_id, note, adapted=adapted)
    return {"message": f"נשמר במתכון של {name}: {note}. מעכשיו זה יופיע בכל פעם שמבשלים אותו."}


def do_put_dish(body):
    """The user's choice after "not in the recipe book": a similar dish, or a new recipe
    the AI writes (checked in code, then saved in the recipe book)."""
    name = str(body.get("dish") or "").strip()
    day, meal = body.get("day"), body.get("meal")
    if not name or meal not in (None, *main.MEAL_NAMES):
        raise tools.DataError("הבקשה לא תקינה.")
    if body.get("create"):
        recipes = tools.load_recipes()
        recipe = agent.create_recipe(name, body.get("dish_type"), recipes)
        if recipe is None:
            raise tools.DataError(f"לא הצלחתי לכתוב מתכון ל\"{name}\" מהמצרכים שהתוכנית מכירה.")
        household.add_recipe(recipe)
        name = recipe["name"]
    planner = main.load_saved_planner(today())  # loaded again: the recipe book may be new
    role = body.get("role") if body.get("role") in ("main", "side") else None
    answer = put_named_dish(planner, {"dish": name, "day": day, "meal": meal, "role": role})
    if body.get("create"):
        answer["message"] = f"הוספתי למתכונים: {name}.\n\n{answer['message']}"
    return answer


def slot(body):
    """The day and meal a request is about (checked)."""
    day, meal = body.get("day"), body.get("meal")
    if meal not in main.MEAL_NAMES:
        raise tools.DataError("ארוחה לא מוכרת.")
    return day, meal


def do_replace(body):
    """'I feel like something else' (or 'plan a meal' for an empty slot). The program
    names the slot, so nothing depends on reading a dish's name from words; the AI
    decides what's best to cook there. If it doesn't manage, the program picks."""
    day, meal = slot(body)
    planner = main.load_saved_planner(today())
    if day not in planner.prefs["plan_days"] or planner.is_past(day):
        raise tools.DataError("אי אפשר לשנות יום שכבר עבר.")
    entry = planner.plan.get(day, {}).get(meal)
    old_main = entry and entry["main"]
    current = entry and planner.recipes[old_main]["name"]
    before = {(d, m): dict(e) for d, meals in planner.plan.items() for m, e in meals.items()}
    planner.change_targets = {(day, meal)}  # only this slot may change
    try:
        result, log = run_ai(planner, replace_goal(day, meal, current, planner.prefs))
        answer = ai_answer(result, log)
        fallback = "האגנט לא השלים את ההחלפה, אז בחרתי מנה מתאימה בעצמי."
    except agent.AgentError as e:
        answer = {"log": []}
        fallback = f"לא הצלחתי לפנות לאגנט ({e}), אז בחרתי מנה מתאימה בעצמי."
    now = planner.plan.get(day, {}).get(meal)
    if not now or now["main"] == old_main:
        # The AI's own words would describe a change it didn't make, so they're replaced.
        if household.replace_meal(planner, day, meal, not_main=old_main):
            answer["message"] = fallback
        else:
            answer["message"] = "אין במתכונים מנה אחרת שמתאימה לכל הכללים בארוחה הזאת, אז השארתי אותה."
        main.save_outputs(planner)
    answer["message"] = f"{main.changes_text(planner, before)}\n\n{answer['message']}"
    return answer


def do_guests(body):
    """A different number of people for one meal, without the AI."""
    day, meal = slot(body)
    people = body.get("people")
    if isinstance(people, bool) or not isinstance(people, int) or not 1 <= people <= tools.MAX_PEOPLE:
        raise tools.DataError(f"מספר האנשים צריך להיות מספר שלם בין 1 ל-{tools.MAX_PEOPLE}.")
    planner = main.load_saved_planner(today())
    entry = planner.plan.get(day, {}).get(meal)
    if entry is None or planner.is_past(day):
        raise tools.DataError("אפשר לעדכן אורחים רק בארוחה מתוכננת ביום שעוד לא עבר.")
    if entry["people"] == people:
        return {"message": f"הארוחה כבר מתוכננת ל-{people} אנשים."}
    result = household.set_people(planner, day, meal, people)
    if "error" in result:
        raise tools.DataError(f"אין במתכונים מנה שמתאימה ל-{people} אנשים בארוחה הזאת, אז השארתי אותה כמו שהיא.")
    main.save_outputs(planner)
    extra = ", ".join(f"{i['name']} {main.amount(i['quantity'], i['unit'])}" for i in result["to_buy"])
    what = (f"המנות הקודמות לא מתאימות ל-{people} אנשים, אז בחרתי במקומן: {result['kept']}."
            if result.get("new_dishes") else f"עודכן ל-{people} אנשים.")
    return with_review(what + (f" צריך לקנות: {extra}." if extra else ""))


def do_skip(body):
    """'Not cooked': the meal comes out and its ingredients are free again."""
    day, meal = slot(body)
    planner = main.load_saved_planner(today())
    household.skip_meal(planner, day, meal)
    main.save_outputs(planner)
    return {"message": "הארוחה הוסרה, והמצרכים שלה חזרו למלאי."}


def do_bought(body):
    """Items on the shopping list were bought: they move into the pantry."""
    names = body.get("names")  # None: the whole list
    planner = main.load_saved_planner(today())
    bought = household.record_shopping(planner, set(names) if names is not None else None)
    main.save_outputs(planner)
    return {"message": f"נוספו למלאי {len(bought)} מוצרים." if bought else "לא סומן שום מוצר."}


def units():
    """Each known ingredient's unit (grams, units...)."""
    return tools.load_recipes()["ingredient_units"]


def match_plan_to_pantry():
    """The pantry changed: recount what this week's meals take from home and what's
    missing, so the shopping list always matches. No AI. Returns a line for the user."""
    recipes = tools.load_recipes()
    if not main.saved_plan_this_week(recipes, today()):
        return ""
    planner = main.load_saved_planner(today())
    if household.reallocate(planner, tools.load_pantry(recipes["ingredient_units"])):
        main.save_outputs(planner)
        return " רשימת הקניות עודכנה לפי המלאי."
    return ""


def stuck_text(stuck):
    """Lines for the meals that nothing fits, with the reasons."""
    return "".join(f"\n\n⚠️ יום {s['day']} {main.MEAL_NAMES[s['meal']]}: {'; '.join(s['reasons'])}. "
                   f"אין במתכונים אף מנה שמתאימה לכל הכללים, אז השארתי אותה כמו שהיא." for s in stuck)


def with_review(message):
    """After any change to the data, the program fits the menu to it (household.fix_menu):
    every meal still to come that breaks a rule or a wish is replaced, in code, so it
    always happens and costs nothing. `message` says what was saved."""
    if not main.saved_plan_this_week(tools.load_recipes(), today()):
        return {"message": message}
    planner = main.load_saved_planner(today())
    before = {(d, m): dict(e) for d, meals in planner.plan.items() for m, e in meals.items()}
    stuck = household.fix_menu(planner)
    if not main.changed_slots(planner, before):
        ok = "" if stuck else "\n\n✅ התפריט כבר מתאים לכל הנתונים, לא היה צריך לשנות כלום."
        return {"message": message + ok + stuck_text(stuck)}
    main.save_outputs(planner)
    return {"message": f"{message}\n\nהתאמתי את התפריט:\n{main.changes_text(planner, before)}{stuck_text(stuck)}"}


def do_pantry(body):
    """Save the pantry the user edited, then fit the shopping list and the menu to it."""
    household.write_pantry(body.get("items") or [], units(), today())
    return with_review("המלאי נשמר." + match_plan_to_pantry())


def do_purchase(body):
    """Add a new purchase to the pantry, then fit the menu to it."""
    plan = tools.load_plan(tools.load_recipes())["plan"] if tools.PLAN_FILE.exists() else {}
    household.add_to_pantry(body.get("items") or [], units(), today(), used=household.reserved_in_plan(plan))
    return with_review("הקנייה נוספה למלאי." + match_plan_to_pantry())


def save_preferences(**changes):
    """Save the important updates in preferences.json, keeping everything else.
    Checked by loading it back; if it's refused, the old file is put back."""
    path = tools.DATA_DIR / "preferences.json"
    prefs = tools.load_json(path)
    prefs.update(changes)
    # Written next to it and checked first; only then it replaces the file, in one step.
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(prefs, ensure_ascii=False, indent=2), encoding="utf-8")
    try:
        tools.load_preferences(tmp)
    except tools.DataError:
        tmp.unlink()
        raise
    tmp.replace(path)


def texts(values):
    """The user's list of texts, trimmed, without empties or repeats."""
    return list(dict.fromkeys(str(v).strip() for v in values or [] if str(v).strip()))  # no duplicates


def do_avoid(body):
    """Save the ingredients the family never eats; the review replaces meals that have them."""
    save_preferences(avoid_ingredients=texts(body.get("items")))
    return with_review("נשמר.")


PENDING_WISHES = {}  # a new wish waiting for the user to choose between it and one it contradicts


def do_family_notes(body):
    """Save the wishes. A new wish is understood once, by a short AI call: if it's about
    what goes where, it becomes a rule the code checks from now on. If it contradicts a
    saved wish, nothing is saved and the user is asked which one to keep."""
    prefs = tools.load_preferences()
    notes = texts(body.get("items"))
    rules = [r for r in prefs["slot_rules"] if r["note"] in notes]
    understood = dict(prefs.get("understood") or {})
    _, planner = main.load_planner(today())
    warning = ""
    for note in notes:
        if note in understood:  # already understood when it was saved
            continue
        try:
            rule, understood[note] = PENDING_WISHES.pop(note, None) or agent.interpret_wish(note, prefs["plan_days"])
        except agent.AgentError as e:
            warning = f"\n\n⚠️ לא הצלחתי לפנות לאגנט כדי להבין את ההעדפה, אז היא נשמרה כהערה כללית: {e}"
            continue
        if rule:
            planner.prefs = {**planner.prefs, "slot_rules": rules}
            clash = household.conflicting_wishes(planner, rule)
            if clash:
                PENDING_WISHES[note] = (rule, understood[note])
                old = ", ".join(f"\"{c}\"" for c in clash)
                return {"message": f"ההעדפה החדשה \"{note}\" סותרת את {old}: אין מנה שעונה על שתיהן. באיזו לבחור?",
                        "conflict": {"new": note, "old": clash}}
            rules.append(rule)
    understood = {n: u for n, u in understood.items() if n in notes}
    save_preferences(family_notes=notes, slot_rules=rules, understood=understood)
    return with_review("נשמר." + warning)


SETTING_DAYS = tools.WEEKDAYS[:5]  # Sunday to Thursday: on Friday the program plans the coming week


def do_settings(body):
    """The household: how many people, which days and meals to plan. Meals still to come
    that were for the old number of people are updated (other dishes if they must)."""
    people = body.get("people")
    days = [d for d in SETTING_DAYS if d in (body.get("plan_days") or [])]
    meals = [m for m in tools.MEAL_TYPES if m in (body.get("meals") or [])]
    if isinstance(people, bool) or not isinstance(people, int) or not 1 <= people <= tools.MAX_PEOPLE:
        raise tools.DataError(f"מספר האנשים בבית צריך להיות מספר שלם בין 1 ל-{tools.MAX_PEOPLE}.")
    if not days or not meals:
        raise tools.DataError("צריך לבחור לפחות יום אחד וארוחה אחת.")
    old = tools.load_preferences()["people"]
    save_preferences(people=people, plan_days=days, meals=meals)

    lines = ["ההגדרות נשמרו."]
    if people != old and main.saved_plan_this_week(tools.load_recipes(), today()):
        planner = main.load_saved_planner(today())
        stuck = []
        for day in planner.prefs["plan_days"]:
            for meal in planner.prefs["meals"]:
                entry = planner.plan.get(day, {}).get(meal)
                if entry and not planner.is_past(day) and entry["people"] == old:
                    if "error" in household.set_people(planner, day, meal, people):
                        stuck.append(f"יום {day} {main.MEAL_NAMES[meal]}")
        main.save_outputs(planner)
        lines.append(f"הארוחות שנשארו השבוע עודכנו ל-{people} אנשים.")
        if stuck:
            lines.append("אין מנה שמתאימה למספר הזה ב: " + ", ".join(stuck) + ". שם השארתי כמו שהיה.")
    if main.saved_plan_this_week(tools.load_recipes(), today()) and main.load_saved_planner(today()).empty_slots():
        lines.append("יש ימים או ארוחות חדשים בלי תפריט: אפשר ללחוץ שם על 'לתכנן ארוחה'.")
    return with_review(" ".join(lines))


def do_new_week(body):
    """Start a new week from the pantry the user confirmed; last week's plan is archived."""
    household.start_new_week(body.get("items") or [], units(), today())
    return {"message": "המלאי לשבוע החדש נשמר. עכשיו אפשר לתכנן את השבוע."}


ACTIONS = {
    "/api/plan": do_plan, "/api/change": do_change, "/api/replace": do_replace,
    "/api/guests": do_guests, "/api/skip": do_skip, "/api/bought": do_bought,
    "/api/pantry": do_pantry, "/api/purchase": do_purchase, "/api/new-week": do_new_week,
    "/api/avoid": do_avoid, "/api/family-notes": do_family_notes, "/api/settings": do_settings,
    "/api/put-dish": do_put_dish, "/api/recipe-note": do_recipe_note, "/api/meal-note": do_meal_note,
}


# ---------- The web server ----------

class Handler(BaseHTTPRequestHandler):
    """The web server: the page itself, the state, and one POST route per action (ACTIONS)."""
    def send(self, status, body, content_type="application/json; charset=utf-8"):
        """Answer with JSON (or the page's HTML)."""
        data = body if isinstance(body, bytes) else json.dumps(body, ensure_ascii=False, default=str).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def answer(self, work):
        """Run one request; problems come back as a message the page can show."""
        try:
            with LOCK:
                body = work()
                body["state"] = state()
            self.send(200, body)
        except (tools.DataError, agent.AgentError) as e:
            self.send(400, {"error": str(e)})
        except Exception as e:  # a bug must not kill the server
            self.send(500, {"error": f"שגיאה פנימית: {type(e).__name__}: {e}"})

    def do_GET(self):
        """The page, or the current state."""
        if self.path in ("/", "/index.html"):
            self.send(200, PAGE.read_bytes(), "text/html; charset=utf-8")
        elif self.path == "/api/state":
            self.answer(dict)
        else:
            self.send(404, {"error": "not found"})

    def do_POST(self):
        """An action: read its JSON body and run it (see ACTIONS)."""
        # The body is read first, always: answering before it's read can make the
        # browser's connection drop (seen on Windows).
        try:
            length = int(self.headers.get("Content-Length") or 0)
            raw = self.rfile.read(length)
        except ValueError:
            return self.send(400, {"error": "הבקשה לא תקינה"})
        action = ACTIONS.get(self.path)
        if action is None:
            return self.send(404, {"error": "not found"})
        try:
            body = json.loads(raw or b"{}")
        except (ValueError, json.JSONDecodeError):
            return self.send(400, {"error": "הבקשה לא תקינה"})
        self.answer(lambda: action(body if isinstance(body, dict) else {}))

    def log_message(self, format, *args):
        """Keep the terminal quiet; the agent's steps are shown on the page."""
        pass  # keep the terminal quiet; the AI's steps are shown on the page


def run(argv=None):
    """Start the server (and the browser), until Ctrl+C."""
    global TODAY
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="Meal planner in the browser")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--date", type=date.fromisoformat, help="pretend today is this date (YYYY-MM-DD)")
    parser.add_argument("--no-browser", action="store_true", help="don't open the browser")
    args = parser.parse_args(argv)
    TODAY = args.date

    try:
        server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    except OSError:
        print(f"❌ הפורט {args.port} תפוס. אולי האפליקציה כבר פתוחה? אפשר גם: py web.py --port 8766")
        return 1
    url = f"http://localhost:{args.port}"
    print(f"🍽️  מתכנן הארוחות פתוח בכתובת {url}")
    print("   לסגירה: Ctrl+C בחלון הזה")
    if not args.no_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nנסגר.")
    return 0


if __name__ == "__main__":
    sys.exit(run())
