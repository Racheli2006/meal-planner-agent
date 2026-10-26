"""Weekly kosher meal planner: the program you run.

    py main.py                     menu
    py main.py plan                plan the week (uses the AI)
    py main.py today               what's cooking today (no AI: just reads the saved plan)
    py main.py change "בשלישי בצהריים יהיו 5 אנשים"     change a day (uses the AI)

    --date 2026-10-27   pretend today is another date (for testing and demos)
"""

import argparse
import sys
from collections import Counter
from datetime import date

import agent
import household
import tools
from prompts import change_goal, plan_goal

MEAL_NAMES = {"lunch": "צהריים", "dinner": "ערב"}
KOSHER_NAMES = {"meat": "בשרי", "dairy": "חלבי", "parve": "פרווה"}
UNIT_NAMES = {"יחידה": "יח'", "צרור": "צרור"}  # "5 יח'" reads better than "5 יחידה"


def amount(quantity, unit):
    return f"{quantity} {UNIT_NAMES.get(unit, unit)}"


# ---------- Building the planner from the data files ----------

def load_planner(today):
    recipes = tools.load_recipes()
    pantry = tools.load_pantry(recipes["ingredient_units"])
    prefs = tools.load_preferences()
    return recipes, tools.MealPlanner(recipes, pantry, prefs, today=today)


def load_saved_planner(today):
    """The planner with this week's saved plan loaded, or a DataError explaining why not."""
    recipes, planner = load_planner(today)
    saved = tools.load_plan(recipes)
    this_week = tools.week_start_for(today)
    if saved["week_start"] != this_week:
        raise tools.DataError(
            f"התפריט השמור הוא לשבוע שמתחיל ב-{saved['week_start']:%d/%m}, "
            f"ולא לשבוע הנוכחי ({this_week:%d/%m}). כדאי לתכנן שבוע חדש (אפשרות 1)."
        )
    planner.restore(saved)
    return planner


# ---------- Text output (printed, and saved to output/menu.txt) ----------

def menu_text(planner):
    status = planner.get_plan_status()
    lines = [f"תפריט לשבוע שמתחיל ב-{planner.week_start:%d/%m/%Y}", ""]
    for day in planner.prefs["plan_days"]:
        lines.append(f"יום {day} ({planner.day_date(day):%d/%m})")
        for meal in planner.prefs["meals"]:
            item = next((p for p in status["planned"] if p["day"] == day and p["meal"] == meal), None)
            if item is None:
                lines.append(f"   {MEAL_NAMES[meal]}: —")
                continue
            guests = f", {item['people']} אנשים" if item["people"] != planner.prefs["people"] else ""
            lines.append(f"   {MEAL_NAMES[meal]}: {item['main']} + {item['side']}"
                         f"  ({KOSHER_NAMES[item['meal_type']]}{guests})")
        lines.append("")
    return "\n".join(lines)


def shopping_text(items, title="רשימת קניות"):
    if not items:
        return f"{title}: אין צורך לקנות כלום 🎉"
    return "\n".join([f"{title}:"] + [f"   ☐ {i['name']}: {amount(i['quantity'], i['unit'])}" for i in items])


def day_text(planner, day):
    meals = planner.describe_day(day)
    if not meals:
        return f"אין ארוחות מתוכננות ליום {day}."
    lines = [f"🍽️  מה מבשלים היום, יום {day}", ""]
    for m in meals:
        lines.append(f"── {MEAL_NAMES[m['meal']]} ({m['people']} אנשים) ──")
        for dish in m["dishes"]:
            lines.append(f"{dish['name']}  (כ-{dish['prep_minutes']} דקות)")
            lines.append("   מצרכים: " + ", ".join(f"{i['name']} {amount(i['quantity'], i['unit'])}" for i in dish["ingredients"]))
            # The ingredients are scaled; the steps' text isn't (it's the recipe as written).
            if dish["servings"] != m["people"]:
                lines.append(f"   (המצרכים מותאמים ל-{m['people']} אנשים. כמויות שכתובות בשלבים, כמו קמח, שמן ומים, "
                             f"הן למתכון המלא ל-{dish['servings']}.)")
            for n, step in enumerate(dish["steps"], 1):
                lines.append(f"   {n}. {step}")
            lines.append("")
    return "\n".join(lines)


def changed_slots(planner, before):
    """(day, meal, old, new) for every slot whose dishes or number of people changed.
    A meal taken out and put back the same is not a change, even if its shopping moved."""
    def what(entry):
        return entry and (entry["main"], entry["side"], entry["people"])

    after = {(d, m): e for d, meals in planner.plan.items() for m, e in meals.items()}
    return [(day, meal, before.get((day, meal)), after.get((day, meal)))
            for day in planner.prefs["plan_days"] for meal in planner.prefs["meals"]
            if what(before.get((day, meal))) != what(after.get((day, meal)))]


def changes_text(planner, before):
    """What really changed in the plan, from the data (not from the AI's summary)."""
    def describe(entry):
        if entry is None:
            return "ריק"
        side = f" + {planner.recipes[entry['side']]['name']}" if entry["side"] else ""
        return f"{planner.recipes[entry['main']]['name']}{side} ({entry['people']} אנשים)"

    lines = []
    for day, meal, old, new in changed_slots(planner, before):
        # Words, not an arrow: an arrow's direction flips between RTL and LTR displays.
        lines.append(f"   יום {day}, {MEAL_NAMES[meal]}:")
        lines.append(f"      לפני: {describe(old)}")
        lines.append(f"      אחרי: {describe(new)}")
    if not lines:
        return "✏️  מה השתנה בפועל: לא בוצע שום שינוי בתפריט."
    return "\n".join(["✏️  מה השתנה בפועל:"] + lines)


def change_summary(planner, before, summary):
    """The AI's summary, unless the plan didn't change: then only the reason it
    reported. A free summary could claim a change that never happened."""
    if changed_slots(planner, before):
        return summary
    if planner.no_change_reason:
        return f"✅ אין צורך בשינוי: {planner.no_change_reason}"
    return "האגנט לא שינה את התפריט. אם ציפית לשינוי, אפשר לנסח את הבקשה אחרת."


def pantry_notes_text(planner):
    """Facts from the data, not from the AI: expired items, and items no recipe uses."""
    pantry = planner.get_pantry()
    lines = []
    if pantry["expired_not_used"]:
        lines.append("🗑️  פג תוקף (לא נעשה בהם שימוש): " + ", ".join(pantry["expired_not_used"]))
    if pantry["not_in_any_recipe"]:
        lines.append("💡 יש בבית מוצרים שאף מתכון לא משתמש בהם: " + ", ".join(pantry["not_in_any_recipe"])
                     + ". כדאי להוסיף מתכונים שמשתמשים בהם.")
    return "\n".join(lines)


def save_outputs(planner):
    """Save the plan (for later runs) and a readable copy of the menu and shopping list."""
    tools.save_plan(planner)
    text = menu_text(planner) + "\n" + shopping_text(planner.shopping_list()) + "\n"
    menu_file().write_text(text, encoding="utf-8")


def menu_file():
    return tools.PLAN_FILE.parent / "menu.txt"


def report(result):
    """Print the AI's summary and how the run ended."""
    print()
    if result["summary"]:
        print("💬 סיכום האגנט:")
        print(result["summary"])
    if not result["finished"]:
        print(f"\n⚠️  האגנט לא השלים את המשימה: {result['reason']}")
    print(f"\n(צעדים: {result['steps']}, טוקנים: {result['tokens']:,}, "
          f"עלות משוערת: {result['cost'] * 100:.1f} סנט)")


# ---------- The three actions ----------

def saved_plan_this_week(recipes, today):
    """This week's saved plan, or None if there isn't one."""
    if not tools.PLAN_FILE.exists():
        return None
    saved = tools.load_plan(recipes)
    return saved if saved["week_start"] == tools.week_start_for(today) else None


def plan_week(today):
    recipes = tools.load_recipes()
    if tools.PLAN_FILE.exists():
        last = tools.load_plan(recipes)
        if last["week_start"] < tools.week_start_for(today):
            # A new week: start from what's left of the last one, as the web app does
            # (there it's shown first so it can be corrected).
            left = household.pantry_after_week(household.read_pantry(), last["plan"])
            household.start_new_week(left, recipes["ingredient_units"], today)
            print(f"📦 המלאי עודכן לפי מה שנשאר מהשבוע של {last['week_start']:%d/%m}, "
                  f"והתפריט הקודם נשמר בארכיון.\n")
    recipes, planner = load_planner(today)
    saved = saved_plan_this_week(recipes, today)
    if saved:
        household.keep_past_days(planner, saved)
    days = [d for d in planner.prefs["plan_days"] if not planner.is_past(d)]
    print(f"🗓️  מתכנן את השבוע שמתחיל ב-{planner.week_start:%d/%m}: {', '.join(days)}\n")

    result = agent.run_agent(planner, plan_goal(planner.prefs, days), require_done=True)
    # Whatever the AI couldn't fit to a family wish, the program fits (relaxing if it must).
    stuck = household.fix_menu(planner, fill_empty=True)

    save_outputs(planner)
    print("\n" + menu_text(planner))
    print(shopping_text(planner.shopping_list()))
    notes = pantry_notes_text(planner)
    if notes:
        print("\n" + notes)
    for s in stuck:
        print(f"\n⚠️  יום {s['day']} {MEAL_NAMES[s['meal']]}: {'; '.join(s['reasons'])}. אין מנה שמתאימה לכל הכללים.")
    report(result)
    print(f"\n📁 נשמר ב: {menu_file()}")


def show_today(today):
    day = tools.hebrew_weekday(today)
    prefs = tools.load_preferences()
    if day not in prefs["plan_days"]:
        print(f"היום יום {day}, והוא לא חלק מהתכנון ({', '.join(prefs['plan_days'])}).")
        return
    planner = load_saved_planner(today)
    print(day_text(planner, day))


def change_day(today, request):
    planner = load_saved_planner(today)
    if not request:
        request = input("מה לשנות? (למשל: ביום שלישי בצהריים יהיו 5 אנשים)\n> ").strip()
    if not request:
        print("לא נכתבה בקשה, אז לא שיניתי כלום.")
        return

    # In the terminal there's no way to tick off the shopping list, so we assume it
    # was bought and add it to the pantry. (The web app asks instead.) If the AI run
    # then fails, the pantry is put back: the plan isn't saved, so nothing was bought.
    pantry_before = household.pantry_file().read_text(encoding="utf-8")
    if household.record_shopping(planner):
        print("🛒 הנחתי שרשימת הקניות כבר נקנתה, והוספתי אותה למלאי.\n")

    before = {(day, meal): dict(entry) for day, meals in planner.plan.items() for meal, entry in meals.items()}
    goal = change_goal(request, tools.hebrew_weekday(today), planner.prefs)
    try:
        result = agent.run_agent(planner, goal)
    except BaseException:  # also Ctrl+C
        household.pantry_file().write_text(pantry_before, encoding="utf-8")
        raise
    result["summary"] = change_summary(planner, before, result["summary"])

    # Sunday's list was already bought, so the extra shopping is exactly what the
    # new or changed meals still need to buy (their to_buy). Comparing list totals
    # instead would be wrong: a removed meal's bought items would be counted twice.
    extra = Counter()
    for day, meals in planner.plan.items():
        for meal, entry in meals.items():
            if before.get((day, meal)) != entry:
                extra.update(entry["to_buy"])
    save_outputs(planner)
    print("\n" + menu_text(planner))
    print(changes_text(planner, before) + "\n")
    print(shopping_text([{"name": n, "quantity": q, "unit": planner.units[n]} for n, q in sorted(extra.items())],
                        title="קניות נוספות בגלל השינוי"))
    report(result)


# ---------- Entry point ----------

def choose_from_menu():
    print("מה תרצי לעשות?")
    print("  1. לתכנן שבוע חדש")
    print("  2. מה מבשלים היום?")
    print("  3. לשנות יום בתפריט")
    while True:
        choice = input("> ").strip()
        if choice in ("1", "2", "3"):
            return {"1": "plan", "2": "today", "3": "change"}[choice]
        print("נא לבחור 1, 2 או 3")


def main(argv=None):
    # Hebrew and emoji must print even when the output is redirected to a file.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(description="Weekly kosher meal-planner agent")
    parser.add_argument("action", nargs="?", choices=["plan", "today", "change"])
    parser.add_argument("request", nargs="?", help="for 'change': what to change, in your own words")
    parser.add_argument("--date", type=date.fromisoformat, default=date.today(),
                        help="pretend today is this date (YYYY-MM-DD)")
    args = parser.parse_args(argv)

    try:
        action = args.action or choose_from_menu()
        if action == "plan":
            plan_week(args.date)
        elif action == "today":
            show_today(args.date)
        else:
            change_day(args.date, args.request)
    except (tools.DataError, agent.AgentError) as e:
        print(f"\n❌ {e}")
        return 1
    except (KeyboardInterrupt, EOFError):
        print("\nבוטל.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
