"""Keeping the home's data up to date between AI runs: shopping that was done,
new purchases, meals that weren't cooked, guests, and moving on to a new week.

Nothing here calls the AI. The quantities follow one rule:
- pantry.json is what came into the house this week: what was there when the
  week started, plus everything bought since.
- Each planned meal reserves what it takes from that, so what's free now is the
  pantry minus the reservations (MealPlanner.restore works that out).
- A meal's to_buy is what still has to be bought for it. Marking it bought adds
  it to the pantry and to the meal's reservation.
"""

import copy
import json
from collections import Counter
from datetime import date
from pathlib import Path

import tools



def pantry_file():
    """Where the pantry is kept (looked up each time, so tests can point it elsewhere)."""
    return tools.DATA_DIR / "pantry.json"


# Units offered for an item no recipe uses (a recipe's item always gets the recipe's unit).
UNIT_CHOICES = ("יחידה", "גרם", 'מ"ל', "צרור")


# ---------- The pantry file ----------

def read_pantry(path=None):
    """The pantry items as written in the file: [{name, quantity, unit, expires}]."""
    return tools.load_json(path or pantry_file()).get("items", [])


def _clean(item, units):
    """One item as the user typed it, made consistent: trimmed name, the recipes' unit, no empty date."""
    name = str(item.get("name") or "").strip()
    if not name:
        raise tools.DataError("יש מוצר בלי שם")
    qty = item.get("quantity")
    if isinstance(qty, bool) or not isinstance(qty, (int, float)) or qty < 0:
        raise tools.DataError(f"כמות לא תקינה למוצר '{name}': {qty}")
    if isinstance(qty, float) and qty.is_integer():
        qty = int(qty)
    unit = units.get(name) or item.get("unit")
    if not unit:
        raise tools.DataError(f"חסרה יחידת מידה למוצר '{name}'")
    return {"name": name, "quantity": qty, "unit": unit, "expires": item.get("expires") or None}


def write_pantry(items, units, today, path=None):
    """Save the pantry. It's checked with the same rules as loading it, so a bad
    entry is refused with a clear message and the file is left as it was.
    Items at 0 are dropped: they're finished."""
    items = [i for i in (_clean(i, units) for i in items) if i["quantity"] > 0]
    path = Path(path or pantry_file())
    tmp = path.with_suffix(".tmp")
    data = {"last_updated": today.isoformat(), "items": items}
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    try:
        tools.load_pantry(units, tmp)
    except tools.DataError:
        tmp.unlink()
        raise
    tmp.replace(path)
    return items


def add_to_pantry(additions, units, today, path=None, used=None):
    """Add bought items to the pantry, on top of what's there.

    When both the old and the new item have a date, the sooner one is kept, so
    planning uses up the older one in time. An expired item is replaced, not added to.
    used: {name: quantity} the week's meals took from the pantry (plan reservations).
    The pantry counts everything that came in this week, so a replaced item keeps what
    meals already took from it; otherwise the new stock would look smaller than it is.
    """
    used = used or {}
    items = read_pantry(path)
    by_name = {i["name"]: i for i in items}
    for add in additions:
        add = _clean(add, units)
        item = by_name.get(add["name"])
        if item is None:
            item = by_name[add["name"]] = add
            items.append(item)
        elif item.get("expires") and date.fromisoformat(item["expires"]) < today:
            item.update(quantity=add["quantity"] + used.get(add["name"], 0), expires=add["expires"])
        else:
            item["quantity"] += add["quantity"]
            dates = [d for d in (item.get("expires"), add["expires"]) if d]
            item["expires"] = min(dates) if dates else None  # ISO dates sort as text
    return write_pantry(items, units, today, path)


# ---------- During the week ----------

def record_shopping(planner, names=None, path=None):
    """Mark items of the shopping list as bought (names=None: the whole list).

    They go into the pantry and into the reservation of the meals they were
    bought for, so the free amounts at home don't change. Returns what was bought.
    The caller saves the plan.
    """
    used = reserved_in_plan(planner.plan)  # before the bought items join the reservations
    bought = Counter()
    for meals in planner.plan.values():
        for entry in meals.values():
            for name in list(entry["to_buy"]):
                if names is None or name in names:
                    qty = entry["to_buy"].pop(name)
                    entry["reserved"][name] = entry["reserved"].get(name, 0) + qty
                    bought[name] += qty
    if bought:
        add_to_pantry([{"name": n, "quantity": q} for n, q in bought.items()],
                      planner.units, planner.today, path, used)
    return bought


def reserved_in_plan(plan):
    """{name: quantity} that all the plan's meals take from the pantry."""
    used = Counter()
    for meals in plan.values():
        for entry in meals.values():
            used.update(entry["reserved"])
    return used


def skip_meal(planner, day, meal):
    """A planned meal wasn't cooked: take it out, and what it reserved is free again.

    Unlike the AI's remove_meal, this works on a day that's over too, so the
    end-of-week leftovers are right. The caller saves the plan.
    """
    entry = planner.plan.get(day, {}).pop(meal, None)
    if entry is None:
        raise tools.DataError(f"אין ארוחה מתוכננת ביום {day} בארוחה הזאת")
    for name, qty in entry["reserved"].items():
        item = planner.available.setdefault(
            name, {"quantity": 0, "unit": planner.units.get(name, ""), "expires": None}
        )
        item["quantity"] += qty
    planner.cover_shopping_from_home()
    return entry


def set_people(planner, day, meal, people):
    """A different number of people (guests), without the AI. Same dishes if they still
    fit; otherwise the program picks other dishes that fit that many people.
    Returns the tool's answer, plus "new_dishes": True when the dishes changed, or
    {"error"} when nothing fits (then nothing changed)."""
    planner.change_targets = {(day, meal)}
    result = planner.change_people(day, meal, people)
    if result.get("ok") or "don't work" not in result.get("error", ""):
        return result
    if _change(planner, day, meal, strong=has_wish(planner, day, meal), won=set(), people=people):
        entry = planner.plan[day][meal]
        return {"ok": True, "new_dishes": True,
                "kept": f"{planner.recipes[entry['main']]['name']} + {planner.recipes[entry['side']]['name']}",
                "to_buy": [{"name": n, "quantity": q, "unit": planner.units[n]} for n, q in entry["to_buy"].items()]}
    return result


def reallocate(planner, pantry):
    """After the pantry changed: work out again, from the pantry as it is now, what
    each meal still to come takes from home and what it needs bought.

    `pantry` is tools.load_pantry's result. Days that are over keep what they took
    (it was eaten). Meals are done in order, so the earlier one gets the stock first.
    Returns how many meals changed. The caller saves the plan.
    """
    available = {n: dict(i) for n, i in pantry.items()
                 if i["expires"] is None or i["expires"] >= planner.today}
    future = []
    for day in planner.prefs["plan_days"]:
        for meal in planner.prefs["meals"]:
            entry = planner.plan.get(day, {}).get(meal)
            if entry is None:
                continue
            if planner.is_past(day):
                for name, qty in entry["reserved"].items():
                    if name in available:
                        available[name]["quantity"] = max(0, available[name]["quantity"] - qty)
            else:
                future.append((day, entry))
    planner.available = available

    changed = 0
    for day, entry in future:
        dishes = [planner.recipes[i] for i in (entry["main"], entry["side"]) if i is not None]
        reserved, to_buy, _ = planner.allocate(dishes, entry["people"], planner.day_date(day))
        for name, qty in reserved.items():
            available[name]["quantity"] -= qty
        if (dict(reserved), dict(to_buy)) != (entry["reserved"], entry["to_buy"]):
            changed += 1
        entry["reserved"], entry["to_buy"] = dict(reserved), dict(to_buy)
    return changed


def has_wish(planner, day, meal):
    """Does one of the family's wishes (slot_rules) say what goes in this slot?"""
    return any(day in r["days"] and meal in r["meals"] for r in planner.prefs.get("slot_rules", []))


def slots_to_fix(planner, keep=()):
    """Meals still to come that break a rule now, each with the reasons (in Hebrew).

    Only rules that hold right now count, so taking a wish away never changes the menu:
    - a dish has something the family never eats;
    - the meal goes against one of the family's wishes;
    - the same kind of dish (or the same side) twice in one day;
    - a dish used more often in the week than allowed (after one was moved to fit a wish).
    Of two meals that clash, the one without a wish is the one to change.
    Slots with a wish come first, so they get the dishes they need.
    """
    found = {}
    hard_reasons = {}
    family = set()  # slots a family rule (a wish, something they never eat) says must change
    hard = set()    # slots that break a fixed rule (kosher across the day, repeats)

    def flag(day, meal, reason, kind):
        """Mark a meal still to come as needing a change, with the reason."""
        if not planner.is_past(day):
            found.setdefault((day, meal), []).append(reason)
            kind.add((day, meal))
            if kind is hard:
                hard_reasons.setdefault((day, meal), []).append(reason)

    for item in planner.get_plan_status()["planned"]:
        if item.get("contains_avoided"):
            flag(item["day"], item["meal"], f"מכילה {', '.join(item['contains_avoided'])}, שהמשפחה לא אוכלת", family)
        for wish in item.get("breaks_wish", []):
            flag(item["day"], item["meal"], f"לא מתאימה להעדפה \"{wish}\"", family)

    # Only the days and meals still in the settings (a day taken out of plan_days is ignored).
    plan = {day: {meal: e for meal, e in meals.items() if meal in planner.prefs["meals"]}
            for day, meals in planner.plan.items() if day in planner.prefs["plan_days"]}

    def chosen(slot):
        """The user named this meal's dishes themselves (add_meal_to_plan user_asked)."""
        return plan.get(slot[0], {}).get(slot[1], {}).get("chosen", False)

    def the_one_to_change(slots):
        """Of slots holding the same thing, the ones to change, least important first. The one
        kept is a day that's over (it can't change), or else one the user chose or wished for."""
        return sorted(slots, key=lambda s: (planner.is_past(s[0]), s in keep, chosen(s), has_wish(planner, *s)))[:-1]

    for day, meals in plan.items():
        if "lunch" in meals and "dinner" in meals:
            dinner = meals["dinner"]
            # Lunch is kept when it's a tie: it's sooner, and may already be cooking.
            if planner.same_day_clash(day, "dinner", dinner["main"], dinner["side"]):
                [slot] = the_one_to_change([(day, "dinner"), (day, "lunch")])
                flag(*slot, "אותו סוג מנה או אותה תוספת כמו בארוחה השנייה באותו יום", hard)
            if not planner.check_kosher(dinner["main"], dinner["side"], day, "dinner").get("ok", True):
                [slot] = the_one_to_change([(day, "dinner"), (day, "lunch")])
                flag(*slot, "ארוחה חלבית אחרי ארוחה בשרית באותו יום", hard)

    uses = {}
    for day, meals in plan.items():
        for meal, entry in meals.items():
            uses.setdefault(("main", entry["main"]), []).append((day, meal))
            if entry["side"] is not None:
                uses.setdefault(("side", entry["side"]), []).append((day, meal))
    for (kind, dish_id), slots in uses.items():
        if kind == "main":
            allowed = len(slots) if planner.prefs["repeat_main_in_week"] else 1
        else:
            allowed = planner.prefs["max_uses_per_side_in_week"]
        # A dish the user asked for is theirs: it never counts as the one to change.
        # A main they asked for still shouldn't repeat elsewhere (schnitzel two days running),
        # so its other places change; a side elsewhere is left as it was.
        picked = sum(1 for s in slots if chosen(s))
        if picked:
            allowed = picked if kind == "main" else allowed + picked
        if len(slots) > allowed:
            for slot in [s for s in the_one_to_change(slots) if not chosen(s)][:len(slots) - allowed]:
                flag(*slot, f"{planner.recipes[dish_id]['name']} כבר מופיע השבוע ביום אחר", hard)

    result = [{"day": d, "meal": m, "dish": planner.recipes[planner.plan[d][m]["main"]]["name"], "reasons": r,
               "family": (d, m) in family, "hard_reasons": hard_reasons.get((d, m), [])}
              for (d, m), r in found.items()]
    days = planner.prefs["plan_days"]
    result.sort(key=lambda s: (not s["family"], not has_wish(planner, s["day"], s["meal"]),
                               days.index(s["day"]), s["meal"] != "lunch"))
    return result


def conflicting_wishes(planner, rule):
    """The saved wishes that can't hold together with this new one: in a slot both are
    about, no main + side in the recipes satisfies both (e.g. "fish" and "meat" on the
    same Thursday lunch). Then the user is asked which one to keep."""
    def combos(meal, rules, day):
        """(kind, family, side) of every main + side that this meal could have under these wishes."""
        saved = planner.prefs
        planner.prefs = {**saved, "slot_rules": rules}
        try:
            return {(planner.meal_kosher_type([m["id"], s["id"]]), planner.dish_family(m), s["id"])
                    for m in planner.recipes.values() if m["dish_type"] == "main" and meal in m["meals"]
                    for s in planner.recipes.values() if s["dish_type"] == "side" and meal in s["meals"]
                    if planner.check_kosher(m["id"], s["id"])["ok"]
                    and not planner.broken_wishes(day, meal, [m["id"], s["id"]])}
        finally:
            planner.prefs = saved

    def day_works(lunch, dinner):
        """A lunch and a dinner that can be on the same day: kosher, not the same kind or side."""
        no_dairy_after_meat = planner.prefs["no_dairy_after_meat_same_day"]
        return any(not (no_dairy_after_meat and lk == "meat" and dk == "dairy") and lf != df and ls != ds
                   for lk, lf, ls in lunch for dk, df, ds in dinner)

    found = []
    for other in planner.prefs.get("slot_rules", []):
        for day in (d for d in rule["days"] if d in other["days"]):
            both = [rule, other]
            lunch = combos("lunch", [r for r in both if "lunch" in r["meals"]], day)
            dinner = combos("dinner", [r for r in both if "dinner" in r["meals"]], day)
            if not lunch or not dinner or not day_works(lunch, dinner):
                found.append(other["note"])
                break
    return found


def _best_choice(planner, options, keep_main=None):
    """The program's pick among options that already pass every rule: the same main as
    before if it still fits (so a repeated side changes only the side), then dishes not
    used yet this week, then the least shopping, then using what expires soon."""
    usage = planner.dish_usage()
    best = None
    for o in options:
        for side_id, side_missing in o["sides"]:
            score = (o["main_id"] != keep_main, usage[o["main_id"]] > 0, usage[side_id], o["to_buy"] + side_missing,
                     -len(o.get("uses_expiring_soon", [])))
            if best is None or score < best[0]:
                best = (score, o["main_id"], side_id)
    return best and best[1:]


def _refill(planner, day, meal, people, not_main=None, strong=False):
    """Put the best meal that fits every rule and wish into this slot, relaxing step by
    step when nothing fits: first any amount of shopping; then, when a family rule needs
    this slot changed (strong), a dish from another day or like the other meal that day
    (see _change). Returns how far it had to relax (0-3), or None: then nothing changed."""
    before = copy.deepcopy((planner.plan, planner.available))
    targets = planner.change_targets
    planner.change_targets = {(day, meal)}
    old = planner.plan.get(day, {}).get(meal)
    try:
        if old:
            skip_meal(planner, day, meal)
        for relax in (0, 1, 2, 3) if strong else (0, 1):
            planner.relax = relax
            options = planner.find_meal_options(day, meal, people=people).get("options", [])
            if not_main is not None:  # the user asked for something else
                options = [o for o in options if o["main_id"] != not_main]
            choice = _best_choice(planner, options, keep_main=None if not_main else old and old["main"])
            if choice and planner.add_meal_to_plan(day, meal, *choice, people=people).get("ok"):
                return relax
        planner.plan, planner.available = before
        return None
    finally:
        planner.relax = 0
        planner.change_targets = targets


def _people(planner, day, meal):
    """How many people a slot is for: its meal's number, or the household's."""
    entry = planner.plan.get(day, {}).get(meal)
    return entry["people"] if entry else planner.prefs["people"]


def _hard_problems(planner):
    """Every fixed rule the menu breaks now, as (day, reason): kosher across the day, the
    same kind twice a day, repeats in the week."""
    return {(s["day"], r) for s in slots_to_fix(planner) for r in s["hard_reasons"]}


def _change(planner, day, meal, strong, won, not_main=None, people=None):
    """Refill one slot. If that took a dish from another meal, or pushed the other meal
    of the day aside (relax 2-3), those meals are planned again right away. If in the
    end the menu breaks a fixed rule it didn't break before (say, a dairy dinner after a
    meat lunch), the whole change is undone. True if the slot changed.

    won: slots that got their way like this; they're kept when meals clash.
    """
    before = copy.deepcopy((planner.plan, planner.available))
    problems_before = _hard_problems(planner)
    relax = _refill(planner, day, meal, people or _people(planner, day, meal), not_main, strong)
    if relax is None:
        return False
    if relax >= 2:
        won.add((day, meal))
        for _ in range(4 * len(planner.prefs["plan_days"]) * len(planner.prefs["meals"])):
            pushed = [s for s in slots_to_fix(planner, keep=won)
                      if s["hard_reasons"] and (s["day"], s["meal"]) not in won]
            if not pushed:
                break
            slot = pushed[0]
            if _refill(planner, slot["day"], slot["meal"], _people(planner, slot["day"], slot["meal"])) is None:
                break
    if _hard_problems(planner) - problems_before:
        planner.plan, planner.available = before
        won.discard((day, meal))
        return False
    return True


def replace_meal(planner, day, meal, not_main=None):
    """The program's own pick of a different main dish for this slot (or a meal for an
    empty one), when the AI didn't manage. not_main: the main the user wants to replace
    (the AI may already have taken it out). True if it changed; otherwise nothing changes."""
    entry = planner.plan.get(day, {}).get(meal)
    not_main = not_main or (entry and entry["main"])
    return _change(planner, day, meal, strong=has_wish(planner, day, meal), won=set(), not_main=not_main)


def fix_menu(planner, fill_empty=False):
    """Make the meals still to come fit every rule and wish, in code (no AI), so the menu
    always follows the family's wishes. One meal at a time: fixing one can settle another.

    fill_empty: also fill empty slots that have a wish (after planning a week).
    Returns the slots nothing fits, with the reasons. The caller saves the plan.
    """
    stuck = {}
    won = set()  # slots that got their way by pushing another meal aside: they stay
    for _ in range(4 * len(planner.prefs["plan_days"]) * len(planner.prefs["meals"])):
        todo = [s for s in slots_to_fix(planner, keep=won) if (s["day"], s["meal"]) not in stuck]
        if fill_empty:
            todo += [{"day": e["day"], "meal": e["meal"], "dish": None, "reasons": ["אין ארוחה, ויש לה העדפה"],
                      "family": True}
                     for e in planner.empty_slots()
                     if has_wish(planner, e["day"], e["meal"]) and (e["day"], e["meal"]) not in stuck]
        if not todo:
            break
        slot = todo[0]
        strong = slot["family"] or has_wish(planner, slot["day"], slot["meal"])
        if not _change(planner, slot["day"], slot["meal"], strong, won):
            stuck[(slot["day"], slot["meal"])] = slot
    return list(stuck.values())


# Why a tool refused, in words for the user (the tools answer in English, for the AI).
WHY_NOT = [("already over", "היום הזה כבר עבר"),
           ("not kosher", "זה לא יוצא כשר עם שאר הארוחה או עם הארוחה השנייה באותו יום"),
           ("doesn't eat", "יש במנה מצרך שהמשפחה לא אוכלת"),
           ("family's wish", "זה סותר העדפה קבועה של המשפחה"),
           ("other meal of", "יוצא אותו סוג מנה או אותה תוספת פעמיים באותו יום"),
           ("not suitable for", "המנה לא מתאימה לארוחה הזאת"),
           ("already has this side", "זו כבר התוספת בארוחה הזאת")]


def why_not(error):
    """A tool's refusal (English, for the AI) in words for the user."""
    return next((hebrew for key, hebrew in WHY_NOT if key in error), "המנה לא מתאימה לכללים של הארוחה")


def dishes_named(planner, name):
    """Every recipe this name can mean ("מרק": all the soups), the closest first."""
    found = [r for r in planner.recipes.values() if tools.word_in(name, r["name"])]
    found.sort(key=lambda r: (r["name"] != name, not r["name"].startswith(name), len(r["name"])))
    return found


def find_dish(planner, name):
    """The recipe the user means by this name ("פירה"): an exact name first, then the shortest that has it."""
    found = dishes_named(planner, name)
    return found[0] if found else None


def slots_with(planner, name):
    """Meals still to come that have a dish by this name ("the mujadara"): [(day, meal)]."""
    return [(day, meal) for day in planner.prefs["plan_days"] if not planner.is_past(day)
            for meal in planner.prefs["meals"]
            if (e := planner.plan.get(day, {}).get(meal))
            and any(tools.word_in(name, planner.recipes[i]["name"]) for i in (e["main"], e["side"]) if i is not None)]


def place_of(planner, name, day=None, meal=None):
    """Where the dish named is on the menu: the given day and meal if it's there, else the
    given day (a guessed meal is less sure than the dish's name), else anywhere. Or None."""
    places = slots_with(planner, name)
    for fits in (lambda d, m: day in (None, d) and meal in (None, m), lambda d, m: day in (None, d)):
        found = [p for p in places if fits(*p)]
        if found:
            return found[0]
    return None


def dishes_that_fit(planner, day, meal, name, dish_type=None, count=3):
    """Other recipes to offer instead of a name the recipe book doesn't have: of the same
    kind (a main for a main, a side for a side), that really can go in this meal now (each
    one is tried on a copy of the plan), not used yet this week first. A shared word in
    the name puts one first ("ירקות מאודים" -> "סלט ירקות קצוץ")."""
    dish_type = dish_type or "main"
    words = [w for w in name.split() if len(w) > 1]
    usage = planner.dish_usage()
    there = {i for m in ([meal] if meal else planner.prefs["meals"])
             for i in (planner.plan.get(day, {}).get(m) or {"main": None, "side": None}).values()
             if isinstance(i, int)}  # what the meal already has isn't "another" dish
    fits = []
    for r in planner.recipes.values():
        if r["dish_type"] != dish_type or (meal and meal not in r["meals"]) or r["id"] in there:
            continue
        trial = copy.deepcopy(planner)
        if put_dish(trial, day, meal, r["name"])[0]:
            fits.append(r)
    fits.sort(key=lambda r: (-sum(tools.word_in(w, r["name"]) for w in words), usage[r["id"]] > 0, r["id"]))
    return [r["name"] for r in fits[:count]]


def sides_that_fit(planner, day, meal, count=3):
    """Sides that can go with the meal's main now (each tried on a copy), for "keep the
    side?" after the main changed: not used yet this week first."""
    entry = planner.plan.get(day, {}).get(meal)
    if entry is None:
        return []
    usage = planner.dish_usage()
    fits = []
    for r in planner.recipes.values():
        if r["dish_type"] != "side" or r["id"] == entry["side"] or meal not in r["meals"]:
            continue
        trial = copy.deepcopy(planner)
        trial.change_targets = {(day, meal)}
        if trial.change_side(day, meal, r["id"]).get("ok"):
            fits.append(r)
    fits.sort(key=lambda r: (usage[r["id"]] > 0, r["id"]))
    return [r["name"] for r in fits[:count]]


def meal_recipe(planner, entry, dish_id):
    """A dish's recipe as it is for this meal: the recipe book's, with the changes the user
    asked for in this meal ("להוסיף כמון") already in it. Returns (steps, added, removed)."""
    recipe = planner.recipes[dish_id]
    custom = entry.get("custom", {}).get(str(dish_id))
    if custom:
        return custom["steps"], custom["added"], custom["removed"]
    return recipe["steps"], list(recipe.get("extras", [])), []


def add_meal_note(planner, day, meal, dish_id, text, adapted=None):
    """A change the user asked for in one meal's dish ("להוסיף כמון"). adapted: the recipe
    rewritten with it (agent.adapt_recipe), which is what the meal's recipe shows from now
    on; without it, the change is shown as a last step. Only this meal: the recipe book
    doesn't change. The caller saves the plan."""
    entry = planner.plan.get(day, {}).get(meal)
    if entry is None or dish_id not in (entry["main"], entry["side"]):
        raise tools.DataError("המנה הזאת לא בארוחה")
    notes = entry.setdefault("notes", [])
    if {"dish": dish_id, "text": text} not in notes:
        notes.append({"dish": dish_id, "text": text})
    if adapted:
        steps, added, removed = meal_recipe(planner, entry, dish_id)
        entry.setdefault("custom", {})[str(dish_id)] = {
            "steps": adapted["steps"],
            "added": list(dict.fromkeys(added + adapted["added"])),
            "removed": list(dict.fromkeys(removed + adapted["removed"]))}


def add_recipe_step(recipe_id, text, path=None, adapted=None):
    """Keep a change in the recipe itself, for every time it's cooked. adapted: the meal's
    rewritten recipe (its steps, added and removed ingredients); without it, a last step."""
    path = Path(path or tools.DATA_DIR / "recipes.json")
    data = tools.load_json(path)
    recipe = next((r for r in data["recipes"] if r["id"] == recipe_id), None)
    if recipe is None:
        raise tools.DataError("המתכון לא נמצא")
    if adapted:
        recipe["steps"] = adapted["steps"]
        recipe["extras"] = list(dict.fromkeys(recipe.get("extras", []) + adapted["added"]))
        recipe["ingredients"] = [i for i in recipe["ingredients"] if i["name"] not in adapted["removed"]] \
            or recipe["ingredients"]  # a recipe keeps at least one ingredient
    elif text not in recipe["steps"]:
        recipe["steps"].append(text)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    try:
        tools.load_recipes(tmp)
    except tools.DataError:
        tmp.unlink()
        raise
    tmp.replace(path)
    return recipe["name"]


def add_recipe(recipe, path=None, pantry_path=None):
    """Add a new recipe to the recipe book. It's checked with the same rules as loading
    the book; if anything is wrong, the book is left as it was. Returns its id.

    recipe["new_units"]: ingredients the catalog doesn't know yet, {name: unit}. They're
    added to it; one that's already in the pantry gets the pantry's unit, so the two match.
    """
    path = Path(path or tools.DATA_DIR / "recipes.json")
    data = tools.load_json(path)
    if any(r["name"] == recipe["name"] for r in data["recipes"]):
        raise tools.DataError(f"כבר יש מתכון בשם \"{recipe['name']}\"")
    recipe = dict(recipe)
    in_pantry = {i["name"]: i.get("unit") for i in read_pantry(pantry_path)}
    for name, unit in recipe.pop("new_units", {}).items():
        data["ingredient_units"].setdefault(name, in_pantry.get(name) or unit)
    recipe = {"id": max(r["id"] for r in data["recipes"]) + 1, **recipe}
    data["recipes"].append(recipe)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    try:
        tools.load_recipes(tmp)
    except tools.DataError:
        tmp.unlink()
        raise
    tmp.replace(path)
    return recipe["id"]


def put_dish(planner, day, meal, name, role=None):
    """The user named a dish for a day ("add mashed potatoes on Thursday"): put exactly that
    dish there. It's their choice, so the week's variety and shopping limits don't apply;
    kosher, allergies and the family's wishes do. With no meal named, the day's meals are
    tried in order. Returns (True, meal, recipe) or (False, None, the reasons in Hebrew).

    role: "main" or "side" when the user said what it replaces ("soup instead of the
    rice"): it goes in that place even if it's usually the other kind, and the meal's other
    dish stays. Without it, a recipe goes where its kind goes. The caller saves the plan."""
    recipe = find_dish(planner, name)
    if recipe is None:
        return False, None, [f"לא מצאתי במתכונים מנה בשם \"{name}\""]
    if day not in planner.prefs["plan_days"]:
        return False, None, [f"יום {day} לא בימים שמתכננים"]
    reasons = []
    for m in [meal] if meal else planner.prefs["meals"]:
        entry = planner.plan.get(day, {}).get(m)
        if entry and recipe["id"] in (entry["main"], entry["side"]):
            return True, m, recipe  # it's already there
        if m not in recipe["meals"]:
            reasons.append(f"{recipe['name']} לא מתאימה ל{'צהריים' if m == 'lunch' else 'ערב'}")
            continue
        before = copy.deepcopy((planner.plan, planner.available))
        targets, planner.change_targets = planner.change_targets, {(day, m)}
        place = role or recipe["dish_type"]
        try:
            if place == "side" and entry:
                result = planner.change_side(day, m, recipe["id"], user_asked=True)
            else:
                result = _put_with_partner(planner, day, m, recipe, entry, place)
        finally:
            planner.change_targets = targets
        if result.get("ok"):
            return True, m, recipe
        planner.plan, planner.available = before
        reasons.append(why_not(result.get("error", "")))
    return False, None, list(dict.fromkeys(reasons))


def _put_with_partner(planner, day, meal, recipe, entry, place=None):
    """Put a named dish in its place (place: "main" or "side", its own kind by default) with
    the best dish to go with it: the one there now if it still fits, else the least used one."""
    people = entry["people"] if entry else planner.prefs["people"]
    if entry:
        planner.remove_meal(day, meal)
    other_type = "side" if (place or recipe["dish_type"]) == "main" else "main"
    kept = entry and (entry["side"] if other_type == "side" else entry["main"])
    usage = planner.dish_usage()
    partners = sorted((r for r in planner.recipes.values() if r["dish_type"] == other_type and meal in r["meals"]),
                      key=lambda r: (r["id"] != kept, usage[r["id"]]))
    result = {"error": "not suitable for"}
    for partner in partners:
        main_id, side_id = (recipe["id"], partner["id"]) if other_type == "side" else (partner["id"], recipe["id"])
        # The partner wasn't named, so it isn't repeated against the week's variety.
        if partner["id"] != kept and usage[partner["id"]] and other_type == "main":
            continue
        result = planner.add_meal_to_plan(day, meal, main_id, side_id, people, user_asked=True)
        if result.get("ok"):
            return result
    return result


def keep_past_days(planner, saved):
    """Replanning mid-week: the days that are over keep their meals (they were eaten)."""
    for day, meals in saved["plan"].items():
        if planner.is_past(day):
            planner.plan[day] = meals
            for entry in meals.values():
                for name, qty in entry["reserved"].items():
                    if name in planner.available:
                        item = planner.available[name]
                        item["quantity"] = max(0, item["quantity"] - qty)


# ---------- A new week ----------

def pantry_after_week(items, plan):
    """What's left at home when the week is over: the pantry minus what every
    meal still in the plan took (meals that weren't cooked were taken out)."""
    used = Counter()
    for meals in plan.values():
        for entry in meals.values():
            used.update(entry["reserved"])
    left = []
    for item in items:
        qty = max(0, item["quantity"] - used.get(item["name"], 0))
        if qty:
            left.append({**item, "quantity": qty})
    return left


def start_new_week(items, units, today, pantry_path=None, plan_path=None, archive_dir=None):
    """Save the pantry the user confirmed, and move last week's plan to the archive."""
    write_pantry(items, units, today, pantry_path)
    plan_path = Path(plan_path or tools.PLAN_FILE)
    if plan_path.exists():
        week = tools.load_json(plan_path).get("week_start", "unknown")
        archive_dir = Path(archive_dir or plan_path.parent / "archive")
        archive_dir.mkdir(parents=True, exist_ok=True)
        plan_path.replace(archive_dir / f"plan-{week}.json")
