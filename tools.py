"""Tools the meal-planner agent can call, and the data they work on."""

import copy
import json
import math
from collections import Counter
from datetime import date, timedelta
from pathlib import Path

# Looked up when a file is read, not fixed in default arguments, so tests can point them elsewhere.
DATA_DIR = Path(__file__).parent / "data"
PLAN_FILE = Path(__file__).parent / "output" / "plan.json"

# Hebrew weekdays, Sunday first, as in Israel.
WEEKDAYS = ["ראשון", "שני", "שלישי", "רביעי", "חמישי", "שישי", "שבת"]

KOSHER_TYPES = ("meat", "dairy", "parve")
DISH_TYPES = ("main", "side")
MEAL_TYPES = ("lunch", "dinner")

MAX_PEOPLE = 30  # more than this at one meal is almost surely a typo

# Units you can't buy a fraction of: half a recipe of 3 eggs is 2 eggs, not 1.5.
COUNTABLE_UNITS = ("יחידה", "צרור")


def _stem(word):
    """A Hebrew word without its usual ending, so singular and plural meet:
    ביצה/ביצים -> ביצ, עגבנייה/עגבניות -> עגבנ, אגוז/אגוזים -> אגוז."""
    for ending in ("ים", "ות", "ה", "ת"):
        if word.endswith(ending) and len(word) - len(ending) >= 2:
            word = word[:-len(ending)]
            break
    while word.endswith("י") and len(word) > 2:
        word = word[:-1]
    return word


def word_in(word, text):
    """Is this word (or part of one) in the text, in singular or plural? ("ביצה" is in "ביצים".)"""
    word = word.strip()
    if word in text:
        return True
    stem = _stem(word)
    return len(stem) >= 2 and any(_stem(token).startswith(stem) for token in text.split())


class DataError(Exception):
    """A data file is missing or malformed. The message is shown to the user."""


# ---------- Loading data files ----------

def load_json(path):
    """Read a JSON file, turning low-level errors into a clear DataError."""
    path = Path(path)
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        raise DataError(f"הקובץ לא נמצא: {path}")
    except json.JSONDecodeError as e:
        raise DataError(f"שגיאת כתיבה בקובץ {path.name}, שורה {e.lineno}: {e.msg}")


def load_recipes(path=None):
    """Read the recipe book and check every recipe: valid kinds and meals, known ingredients."""
    path = path or DATA_DIR / "recipes.json"
    data = load_json(path)
    for key in ("ingredient_units", "recipes"):
        if key not in data:
            raise DataError(f"{Path(path).name}: חסר החלק '{key}'")

    units = data["ingredient_units"]
    for r in data["recipes"]:
        name = r.get("name", f"מתכון {r.get('id')}")
        if r.get("kosher") not in KOSHER_TYPES:
            raise DataError(f"מתכון '{name}': סוג כשרות לא תקין '{r.get('kosher')}'")
        if r.get("dish_type") not in DISH_TYPES:
            raise DataError(f"מתכון '{name}': סוג מנה לא תקין '{r.get('dish_type')}'")
        if not r.get("meals") or not set(r["meals"]) <= set(MEAL_TYPES):
            raise DataError(f"מתכון '{name}': ארוחות לא תקינות {r.get('meals')}")
        for ing in r.get("ingredients", []):
            if ing.get("name") not in units:
                raise DataError(f"מתכון '{name}': מצרך לא מוכר '{ing.get('name')}'")
    return data


def load_pantry(units, path=None):
    """Return the pantry as {name: {"quantity", "unit", "expires"}}.

    `units` is the recipe catalog, used to catch unit mismatches
    (e.g. milk written in liters in the pantry but in ml in recipes).
    """
    data = load_json(path or DATA_DIR / "pantry.json")
    pantry = {}
    for item in data.get("items", []):
        name = item.get("name")
        qty = item.get("quantity")
        if not name:
            raise DataError("מלאי: יש פריט בלי שם")
        if name in pantry:
            raise DataError(f"מלאי: המצרך '{name}' מופיע פעמיים")
        if not isinstance(qty, (int, float)) or qty < 0:
            raise DataError(f"מלאי: כמות לא תקינה למצרך '{name}': {qty}")
        if name in units and item.get("unit") != units[name]:
            raise DataError(
                f"מלאי: '{name}' כתוב ביחידה '{item.get('unit')}', "
                f"אבל במתכונים הוא ב'{units[name]}'"
            )

        expires = item.get("expires")
        if expires is not None:
            try:
                expires = date.fromisoformat(expires)
            except (TypeError, ValueError):
                raise DataError(f"מלאי: תאריך לא תקין למצרך '{name}': {expires}")

        pantry[name] = {"quantity": qty, "unit": item.get("unit", ""), "expires": expires}
    return pantry


PREFERENCE_DEFAULTS = {
    "plan_days": ["ראשון", "שני", "שלישי", "רביעי", "חמישי"],
    "meals": ["lunch", "dinner"],
    "people": 4,
    "side_dish_required": True,
    "no_dairy_after_meat_same_day": True,
    "max_missing_per_dish": 2,
    "max_missing_per_dish_with_guests": 4,
    "repeat_main_in_week": False,
    "max_uses_per_side_in_week": 2,
    "expiring_soon_days": 7,
    # Allergies and things the family doesn't eat. A recipe with any of these
    # (also as part of a name: "אפונה" rules out "אפונה יבשה") is never planned.
    "avoid_ingredients": [],
    # The family's standing wishes, in their own words (e.g. "no fish on Thursdays").
    # The AI reads them on every run.
    "family_notes": [],
    # The wishes that are about what goes where, as the AI understood them when they
    # were saved. These ARE checked in code, like the kosher rules:
    # {"note": the wish's text, "days": [...], "meals": [...], and any of:
    #  "allowed": kinds that are OK there (["meat"]),
    #  "must_have": the meal has a dish with one of these words in its name or ingredients (["מרק"]),
    #  "must_not_have": no dish with any of these words (["דג"])}
    "slot_rules": [],
}


def load_preferences(path=None):
    """Read preferences; any setting the user left out falls back to a default."""
    prefs = {**PREFERENCE_DEFAULTS, **load_json(path or DATA_DIR / "preferences.json")}

    if not isinstance(prefs["people"], int) or prefs["people"] < 1:
        raise DataError(f"העדפות: מספר אנשים לא תקין: {prefs['people']}")
    for key in ("max_missing_per_dish", "max_missing_per_dish_with_guests"):
        if not isinstance(prefs[key], int) or prefs[key] < 0:
            raise DataError(f"העדפות: מספר חוסרים לא תקין ב-{key}: {prefs[key]}")
    if not prefs["plan_days"]:
        raise DataError("העדפות: לא הוגדרו ימים לתכנון")
    unknown_days = [d for d in prefs["plan_days"] if d not in WEEKDAYS]
    if unknown_days:
        raise DataError(f"העדפות: ימים לא מוכרים: {unknown_days}")
    if not prefs["meals"] or not set(prefs["meals"]) <= set(MEAL_TYPES):
        raise DataError(f"העדפות: ארוחות לא תקינות: {prefs['meals']}")
    for key in ("avoid_ingredients", "family_notes"):
        value = prefs[key]
        if not isinstance(value, list) or not all(isinstance(a, str) and a.strip() for a in value):
            raise DataError(f"העדפות: {key} צריך להיות רשימה של טקסטים: {value}")
    def words(value):
        return isinstance(value, list) and all(isinstance(w, str) and w.strip() for w in value)

    for rule in prefs["slot_rules"]:
        if not (isinstance(rule, dict) and rule.get("note") in prefs["family_notes"]
                and rule.get("days") and set(rule["days"]) <= set(WEEKDAYS)
                and rule.get("meals") and set(rule["meals"]) <= set(MEAL_TYPES)
                and set(rule.get("allowed", [])) <= set(KOSHER_TYPES)
                and words(rule.get("must_have", [])) and words(rule.get("must_not_have", []))
                and (rule.get("allowed") or rule.get("must_have") or rule.get("must_not_have"))):
            raise DataError(f"העדפות: כלל לא תקין ב-slot_rules: {rule}")
    return prefs


# ---------- Dates of the planned week ----------

def hebrew_weekday(d):
    """Name of the day in Hebrew. Python counts Monday=0, we count Sunday=0."""
    return WEEKDAYS[(d.weekday() + 1) % 7]


def week_start_for(today):
    """The Sunday of the week being planned.

    Sunday to Thursday: this week's Sunday. Friday and Saturday: the week is
    over, so we plan the coming one.
    """
    index = WEEKDAYS.index(hebrew_weekday(today))
    if index >= WEEKDAYS.index("שישי"):
        return today + timedelta(days=7 - index)
    return today - timedelta(days=index)


# ---------- The saved plan file ----------

def save_plan(planner, path=None):
    """Write the plan to disk. Written to a temp file first, then renamed, so a crash
    mid-write can't leave a half-written plan.json behind."""
    path = Path(path or PLAN_FILE)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "week_start": planner.week_start.isoformat(),
        "created": planner.today.isoformat(),
        "plan": planner.plan,
    }
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    tmp.replace(path)


def load_plan(recipes_data, path=None):
    """Read a saved plan and check that every recipe in it still exists."""
    path = Path(path or PLAN_FILE)
    if not path.exists():
        raise DataError("עדיין אין תפריט שמור. קודם צריך לתכנן שבוע (אפשרות 1).")
    data = load_json(path)
    try:
        week_start = date.fromisoformat(data["week_start"])
        plan = data["plan"]
    except (KeyError, TypeError, ValueError):
        raise DataError(f"קובץ התפריט {path.name} פגום. כדאי לתכנן שבוע מחדש.")

    ids = {r["id"] for r in recipes_data["recipes"]}
    for day, meals in plan.items():
        for meal, entry in meals.items():
            for dish in (entry.get("main"), entry.get("side")):
                if dish is not None and dish not in ids:
                    raise DataError(f"בתפריט השמור ({day}) יש מתכון {dish} שכבר לא קיים במאגר. כדאי לתכנן מחדש.")
    return {"week_start": week_start, "plan": plan}


# ---------- The planner: shared state that all tools read and update ----------

class MealPlanner:
    """Holds the data and the plan being built. Each public method is one agent tool."""

    def __init__(self, recipes_data, pantry, prefs, today=None):
        """The week being planned: the recipes, what's free at home, the preferences, and the plan so far."""
        self.units = recipes_data["ingredient_units"]
        self.recipes = {r["id"]: r for r in recipes_data["recipes"]}
        self.prefs = prefs
        self.today = today or date.today()
        self.week_start = week_start_for(self.today)

        # Expired items are set aside: never cooked with, but reported to the user.
        self.expired = []
        self.available = {}
        for name, item in pantry.items():
            if item["expires"] is not None and item["expires"] < self.today:
                self.expired.append(name)
            else:
                self.available[name] = dict(item)

        self.used_in_recipes = {
            ing["name"] for r in self.recipes.values() for ing in r["ingredients"]
        }

        # The plan being built: {day: {meal: {"main": id, "side": id or None, ...}}}
        self.plan = {}

        # True when changing a saved plan mid-week: only the declared slots may change.
        self.mid_week = False

        # When changing mid-week: the slots the user's request is about, declared
        # once by the AI. Any other slot is locked. None = not declared yet.
        self.change_targets = None

        # Only when the program fits a meal to a family wish and nothing fits otherwise
        # (household.fix_menu): 1 = any amount of shopping, 2 = also a dish used on
        # another day, 3 = also whatever the other meal that day is (same kind, or a
        # dairy dinner after meat). That other day or meal is then planned again.
        # Never set for the AI.
        self.relax = 0

        # Set when the AI decides the request needs no change (report_no_change).
        self.no_change_reason = None

    def restore(self, saved):
        """Continue from a saved plan (for changing a day mid-week).

        The saved meals' ingredients are taken out of the pantry again. A meal's
        to_buy is what hasn't been bought yet: bought items are added to the pantry
        and to the meal's reserved (see household.record_shopping).
        """
        self.week_start = saved["week_start"]
        self.plan = saved["plan"]
        self.mid_week = True
        for meals in self.plan.values():
            for entry in meals.values():
                for name, qty in entry["reserved"].items():
                    if name in self.available:
                        # max(0, ...): if the pantry file was edited since, don't go negative.
                        item = self.available[name]
                        item["quantity"] = max(0, item["quantity"] - qty)

    def day_date(self, day):
        """The date of a weekday in the planned week."""
        return self.week_start + timedelta(days=WEEKDAYS.index(day))

    def is_past(self, day):
        """True for days of the planned week that are already over (today is not past)."""
        return self.day_date(day) < self.today

    def days_left(self, name):
        """Days until the item expires, or None if it has no expiry date."""
        expires = self.available[name]["expires"]
        return None if expires is None else (expires - self.today).days

    # ----- Tool: get_pantry -----

    def get_pantry(self):
        """What is still available at home, soonest-to-expire first."""
        soon = self.prefs["expiring_soon_days"]
        items = []
        for name, item in self.available.items():
            if item["quantity"] <= 0:
                continue
            days = self.days_left(name)
            items.append({
                "name": name,
                "quantity": item["quantity"],
                "unit": item["unit"],
                "days_left": days,
                "expiring_soon": days is not None and days <= soon,
                "used_in_recipes": name in self.used_in_recipes,
            })
        # Items with no expiry date go last.
        items.sort(key=lambda i: (i["days_left"] is None, i["days_left"] or 0))

        return {
            "today": self.today.isoformat(),
            "items": items,
            "expired_not_used": self.expired,
            "not_in_any_recipe": [i["name"] for i in items if not i["used_in_recipes"]],
        }

    # ----- Quantities: scale a recipe to the number of people -----

    def scaled_ingredients(self, recipe, people=None):
        """Recipe ingredients scaled from the recipe's servings to `people` (default: prefs)."""
        factor = (people or self.prefs["people"]) / recipe["servings"]
        result = []
        for ing in recipe["ingredients"]:
            unit = self.units[ing["name"]]
            qty = ing["quantity"] * factor
            qty = math.ceil(qty) if unit in COUNTABLE_UNITS else max(1, round(qty))
            result.append({"name": ing["name"], "quantity": qty, "unit": unit})
        return result

    def _missing(self, recipe, people=None):
        """Ingredients the home doesn't have enough of, with how much to buy."""
        missing = []
        for ing in self.scaled_ingredients(recipe, people):
            have = self.available.get(ing["name"], {}).get("quantity", 0)
            if have < ing["quantity"]:
                missing.append({
                    "name": ing["name"],
                    "need": ing["quantity"],
                    "have": have,
                    "to_buy": ing["quantity"] - have,
                    "unit": ing["unit"],
                })
        return missing

    def avoided_ingredients(self, recipe, extra=()):
        """The recipe's ingredients the family doesn't eat (allergies), if any.
        `extra`: more to avoid just this time (a guest's allergy)."""
        avoid = [a.strip() for a in [*self.prefs.get("avoid_ingredients", []), *extra] if str(a).strip()]
        return [i["name"] for i in recipe["ingredients"] if any(word_in(a, i["name"]) for a in avoid)]

    def _has(self, recipe, word):
        """Does the dish have this word in its name or ingredients? ("מרק" finds every soup.)"""
        return (word_in(word, recipe["name"]) or any(word_in(word, i["name"]) for i in recipe["ingredients"])
                or (word.startswith("דג") and recipe["contains_fish"]))

    def broken_wishes(self, day, meal, dish_ids):
        """The family's wishes (slot_rules) that this main + side would break in this slot."""
        dishes = [self.recipes[i] for i in dish_ids if i is not None]
        meal_type = self.meal_kosher_type([r["id"] for r in dishes])
        broken = []
        for rule in self.prefs.get("slot_rules", []):
            if day not in rule["days"] or meal not in rule["meals"]:
                continue
            if rule.get("allowed") and meal_type not in rule["allowed"]:
                broken.append(rule["note"])
            elif rule.get("must_have") and not any(self._has(r, w) for r in dishes for w in rule["must_have"]):
                broken.append(rule["note"])
            elif any(self._has(r, w) for r in dishes for w in rule.get("must_not_have", [])):
                broken.append(rule["note"])
        return broken

    def dish_family(self, recipe):
        """What kind of dish it is, to avoid the same kind twice a day: fish, or the first
        word of the name ("מרק עוף" and "מרק ירקות" are both soups)."""
        if recipe["contains_fish"]:
            return "דג"
        word = recipe["name"].split()[0]
        return {"פשטידת": "פשטידה", "חביתת": "חביתה"}.get(word, word)

    def _families(self, main_id, side_id):
        """The kinds of dish in a meal: its main's, and its side's when that is a main dish
        served as a side (soup instead of the rice)."""
        ids = [main_id] + ([side_id] if side_id is not None and self.recipes[side_id]["dish_type"] == "main" else [])
        return {self.dish_family(self.recipes[i]): self.recipes[i]["name"] for i in ids}

    def same_day_clash(self, day, meal, main_id, side_id):
        """The other meal that day has the same kind of main, or the same side: why, or None."""
        other = self.plan.get(day, {}).get("dinner" if meal == "lunch" else "lunch")
        if other is None:
            return None
        theirs = self._families(other["main"], other["side"])
        same = [name for family, name in theirs.items() if family in self._families(main_id, side_id)]
        if same:
            return f"the other meal of {day} is already {same[0]}: not the same kind of dish twice a day"
        if side_id is not None and side_id == other["side"]:
            return f"the other meal of {day} already has {self.recipes[side_id]['name']} as its side"
        return None

    def _uses_expiring(self, recipe):
        """The recipe's ingredients that expire soon (so the AI can prefer it)."""
        soon = self.prefs["expiring_soon_days"]
        names = []
        for ing in recipe["ingredients"]:
            if ing["name"] in self.available:
                days = self.days_left(ing["name"])
                if days is not None and days <= soon:
                    names.append(ing["name"])
        return names

    # ----- Tool: search_recipes -----

    def search_recipes(self, meal=None, dish_type=None, kosher=None, ingredient=None, people=None, name=None):
        """Recipes matching all given filters, fewest missing ingredients first."""
        if meal is not None and meal not in MEAL_TYPES:
            return {"error": f"meal must be one of {MEAL_TYPES}"}
        if dish_type is not None and dish_type not in DISH_TYPES:
            return {"error": f"dish_type must be one of {DISH_TYPES}"}
        if kosher is not None and kosher not in KOSHER_TYPES:
            return {"error": f"kosher must be one of {KOSHER_TYPES}"}
        error = self._people_error(people)
        if error:
            return {"error": error}

        usage = self.dish_usage()
        found = []
        for r in self.recipes.values():
            if meal and meal not in r["meals"]:
                continue
            if dish_type and r["dish_type"] != dish_type:
                continue
            if kosher and r["kosher"] != kosher:
                continue
            # Partial match, so "עוף" finds both "חזה עוף" and "כרעי עוף".
            if ingredient and not any(ingredient in i["name"] for i in r["ingredients"]):
                continue
            # By name, so a dish the user names ("פירה") is found.
            if name and not word_in(str(name), r["name"]):
                continue
            if self.avoided_ingredients(r):
                continue
            # Kept short on purpose: this goes back to the AI on every search, and
            # every field costs tokens. Fields already fixed by a filter are left out.
            entry = {"id": r["id"], "name": r["name"]}
            if not dish_type:
                entry["dish_type"] = r["dish_type"]
            if not kosher:
                entry["kosher"] = r["kosher"]
            if r["contains_fish"]:
                entry["fish"] = True
            if not meal:
                entry["meals"] = r["meals"]
            entry["missing_count"] = len(self._missing(r, people))
            if self._uses_expiring(r):
                entry["uses_expiring_soon"] = self._uses_expiring(r)
            entry["times_in_plan"] = usage[r["id"]]
            found.append(entry)
        found.sort(key=lambda r: r["missing_count"])

        result = {"count": len(found), "recipes": found}
        if not found:
            # An empty result is feedback too: say which filter is the likely problem.
            if dish_type == "side" and kosher == "meat":
                result["hint"] = ("There are no meat side dishes. Sides that go with a meat main are parve: "
                                  "search sides with kosher='parve' or with no kosher filter.")
            else:
                result["hint"] = "Nothing matches all these filters. Try again with fewer filters (e.g. drop kosher or ingredient)."
        return result

    # ----- Tool: check_missing -----

    def check_missing(self, recipe_id, people=None):
        """Full shopping detail for one recipe, and whether it fits the user's limit."""
        recipe = self.recipes.get(recipe_id)
        if recipe is None:
            return {"error": f"no recipe with id {recipe_id}"}
        error = self._people_error(people)
        if error:
            return {"error": error}

        missing = self._missing(recipe, people)
        missing_names = {m["name"] for m in missing}
        limit = self._missing_limit(people)
        return {
            "id": recipe_id,
            "name": recipe["name"],
            "people": people or self.prefs["people"],
            "from_home": [i for i in self.scaled_ingredients(recipe, people) if i["name"] not in missing_names],
            "missing": missing,
            "missing_count": len(missing),
            "max_missing_allowed": limit,
            "within_limit": len(missing) <= limit,
            "uses_expiring_soon": self._uses_expiring(recipe),
        }

    # ----- Kosher rules (fixed in code, not a user preference) -----

    def meal_kosher_type(self, dish_ids):
        """meat if any dish is meat, dairy if any is dairy, otherwise parve."""
        types = {self.recipes[i]["kosher"] for i in dish_ids}
        if "meat" in types:
            return "meat"
        if "dairy" in types:
            return "dairy"
        return "parve"

    # ----- Tool: check_kosher -----

    def check_kosher(self, main_id, side_id=None, day=None, meal=None):
        """Is main + side kosher together, and (if day/meal given) with the rest of that day?"""
        dish_ids = [i for i in (main_id, side_id) if i is not None]
        unknown = [i for i in dish_ids if i not in self.recipes]
        if unknown:
            return {"error": f"no recipe with id {unknown}"}

        # Each problem comes with a hint, so the AI knows what to change next time.
        problems, hints = [], []
        types = {self.recipes[i]["kosher"] for i in dish_ids}
        has_fish = any(self.recipes[i]["contains_fish"] for i in dish_ids)

        if {"meat", "dairy"} <= types:
            problems.append("meat and dairy in the same meal")
            hints.append("Pair this main with a parve side.")
        if has_fish and "meat" in types:
            problems.append("fish and meat in the same meal")
            hints.append("Don't serve fish with meat: pick a non-meat main or a different side.")

        meal_type = self.meal_kosher_type(dish_ids)

        if day is not None and meal is not None:
            if day not in self.prefs["plan_days"]:
                return {"error": f"day must be one of {self.prefs['plan_days']}"}
            if meal not in self.prefs["meals"]:
                return {"error": f"meal must be one of {self.prefs['meals']}"}

            # After a meat lunch, no dairy dinner the same day (the 6-hour wait).
            if self.prefs["no_dairy_after_meat_same_day"]:
                other_meal = "dinner" if meal == "lunch" else "lunch"
                other = self.plan.get(day, {}).get(other_meal)
                if other:
                    other_type = self.meal_kosher_type([other["main"], other["side"]] if other["side"] else [other["main"]])
                    lunch_type, dinner_type = (meal_type, other_type) if meal == "lunch" else (other_type, meal_type)
                    if lunch_type == "meat" and dinner_type == "dairy":
                        problems.append(f"{day}: dairy dinner after a meat lunch")
                        if meal == "dinner":
                            hints.append("This dinner can't be dairy: pick a meat or parve main "
                                         "(search with kosher='meat' or 'parve') and a parve side.")
                        else:
                            hints.append("This lunch can't be meat because dinner is dairy: "
                                         "pick a dairy or parve main (search with kosher='dairy' or 'parve').")

        return {"ok": not problems, "meal_type": meal_type, "problems": problems, "hints": hints}

    # ----- The plan: helpers -----

    def _slot_error(self, day, meal):
        """Why this day/meal can't be planned (not a planned day or meal, or over), or None."""
        if day not in self.prefs["plan_days"]:
            return f"day must be one of {self.prefs['plan_days']}"
        if meal not in self.prefs["meals"]:
            return f"meal must be one of {self.prefs['meals']}"
        if self.is_past(day):
            return f"{day} ({self.day_date(day).isoformat()}) is already over; it can't be changed"
        return None

    def _locked_error(self, day, meal):
        """Mid-week, only the slots declared for this request may change."""
        if not self.mid_week:
            return None
        if self.change_targets is None:
            return {"error": "declare the slots you are changing first",
                    "hint": "Call declare_change_targets with the day/meal pairs the user's request is about."}
        if (day, meal) not in self.change_targets:
            return {"error": f"{day} {meal} is not part of the user's request, so it is locked",
                    "hint": "Change only the slots you declared. If the request can't be done, explain why to the user."}
        return None

    # ----- Tool: declare_change_targets -----

    def declare_change_targets(self, targets):
        """Mid-week: say which slots the user's request is about. Allowed once.

        The lock is set even if a slot is invalid (e.g. a day that's over), so
        a failed request can't turn into changing some other day instead.
        """
        if not self.mid_week:
            return {"error": "only used when changing a saved plan mid-week"}
        if self.change_targets is not None:
            return {"error": "the slots were already declared and can't be changed",
                    "hint": "Work with the declared slots, or explain to the user what you can't do."}
        if not isinstance(targets, list) or not targets:
            return {"error": "targets must be a non-empty list of {day, meal}"}

        self.change_targets = set()
        problems = []
        for t in targets:
            day, meal = (t.get("day"), t.get("meal")) if isinstance(t, dict) else (None, None)
            error = self._slot_error(day, meal)
            if error:
                problems.append(error)
            else:
                self.change_targets.add((day, meal))
        if problems:
            self.change_targets = set()
            return {"error": "; ".join(problems),
                    "hint": "Nothing can be changed for this request. Don't change other meals: "
                            "explain the problem to the user in your summary."}
        return {"ok": True, "targets": [{"day": d, "meal": m} for d, m in sorted(self.change_targets)]}

    # ----- Tool: report_no_change -----

    def report_no_change(self, reason):
        """Mid-week: the request needs no change. The reason is what the user sees,
        instead of a free summary that might claim a change that didn't happen."""
        if not self.mid_week:
            return {"error": "only used when changing a saved plan mid-week"}
        reason = str(reason or "").strip()
        if not reason:
            return {"error": "reason is required: explain to the user, in Hebrew, why nothing needs to change"}
        self.no_change_reason = reason
        return {"ok": True, "hint": "Don't change anything now. Finish with a one-line answer."}

    def _people_error(self, people):
        """Why a number of people isn't valid, or None."""
        # Checked with `is None`, not `or`: people=0 must be an error, not "use the default".
        if people is not None and (isinstance(people, bool) or not isinstance(people, int) or not 1 <= people <= MAX_PEOPLE):
            return f"people must be a whole number from 1 to {MAX_PEOPLE}, got {people}"
        return None

    def _missing_limit(self, people, day=None, meal=None):
        """How many items the user agrees to buy per dish; more when cooking for guests,
        and in a slot with a family wish (a soup on Mondays is worth a bit more shopping)."""
        if self.relax >= 1:
            return 99
        limit = self.prefs["max_missing_per_dish"]
        if people is not None and people > self.prefs["people"]:
            limit = self.prefs["max_missing_per_dish_with_guests"]
        if day is not None and any(day in r["days"] and meal in r["meals"] for r in self.prefs.get("slot_rules", [])):
            limit = max(limit, self.prefs["max_missing_per_dish_with_guests"])
        return limit

    def dish_usage(self):
        """How many times each recipe id appears in the plan."""
        used = Counter()
        for meals in self.plan.values():
            for entry in meals.values():
                used[entry["main"]] += 1
                if entry["side"] is not None:
                    used[entry["side"]] += 1
        return used

    def empty_slots(self):
        """Slots still to fill. Days already over don't count, so planning on a
        Tuesday fills Tuesday to Thursday only."""
        return [
            {"day": d, "meal": m}
            for d in self.prefs["plan_days"]
            for m in self.prefs["meals"]
            if m not in self.plan.get(d, {}) and not self.is_past(d)
        ]

    def describe_day(self, day):
        """Full details of one day's meals, for the 'what's cooking today' screen."""
        result = []
        for meal in self.prefs["meals"]:
            entry = self.plan.get(day, {}).get(meal)
            if entry is None:
                continue
            dishes = []
            for dish_id in (entry["main"], entry["side"]):
                if dish_id is None:
                    continue
                r = self.recipes[dish_id]
                dishes.append({
                    "name": r["name"],
                    "servings": r["servings"],
                    "prep_minutes": r["prep_minutes"],
                    "ingredients": self.scaled_ingredients(r, entry["people"]),
                    "steps": r["steps"],
                })
            result.append({"meal": meal, "people": entry["people"], "dishes": dishes})
        return result

    def shopping_list(self):
        """Everything the plan needs to buy, summed over all meals."""
        total = Counter()
        for meals in self.plan.values():
            for entry in meals.values():
                total.update(entry["to_buy"])
        return [{"name": n, "quantity": q, "unit": self.units[n]} for n, q in sorted(total.items())]

    def allocate(self, dishes, people, on=None):
        """Work out what each dish takes from home and what must be bought.

        Done dish by dish on a scratch copy, so if main and side both need onions
        the second dish sees what's left. on: the meal's date; what expires before
        it can't be used for it. Changes nothing: returns
        (reserved, to_buy, number of items to buy per dish).
        """
        left = {name: item["quantity"] for name, item in self.available.items()
                if on is None or item["expires"] is None or item["expires"] >= on}
        reserved, to_buy = Counter(), Counter()
        missing_per_dish = []
        for r in dishes:
            missing_here = 0
            for ing in self.scaled_ingredients(r, people):
                take = min(left.get(ing["name"], 0), ing["quantity"])
                if take:
                    left[ing["name"]] -= take
                    reserved[ing["name"]] += take
                if ing["quantity"] > take:
                    to_buy[ing["name"]] += ing["quantity"] - take
                    missing_here += 1
            missing_per_dish.append(missing_here)
        return reserved, to_buy, missing_per_dish

    # ----- Tool: find_meal_options -----

    def find_meal_options(self, day, meal, people=None, exclude=None):
        """Every main + side that passes ALL the hard rules for this slot.

        The hard rules (kosher, also against the other meal that day; suitable
        for the meal; not used too often; within the shopping limit) are checked
        here in code. What's left for the AI is the judgement: which valid
        option is best for the week as a whole.

        exclude: ingredients to leave out for this meal only (e.g. a guest's allergy).
        """
        if exclude is not None and (not isinstance(exclude, list) or not all(isinstance(e, str) for e in exclude)):
            return {"error": "exclude must be a list of ingredient names"}
        exclude = exclude or []
        error = self._slot_error(day, meal)
        if error:
            return {"error": error}
        if meal in self.plan.get(day, {}):
            return {"error": f"{day} {meal} is already planned",
                    "hint": "Pick an empty slot from get_plan_status, or remove_meal first to replace it."}
        error = self._people_error(people)
        if error:
            return {"error": error}
        if people is None:
            people = self.prefs["people"]

        usage = self.dish_usage()
        limit = self._missing_limit(people, day, meal)
        repeats_ok = self.relax >= 2
        mains = [r for r in self.recipes.values()
                 if r["dish_type"] == "main" and meal in r["meals"] and not self.avoided_ingredients(r, exclude)
                 and (self.prefs["repeat_main_in_week"] or usage[r["id"]] == 0 or repeats_ok)]
        sides = [r for r in self.recipes.values()
                 if r["dish_type"] == "side" and meal in r["meals"] and not self.avoided_ingredients(r, exclude)
                 and (usage[r["id"]] < self.prefs["max_uses_per_side_in_week"] or repeats_ok)]

        # Compact on purpose (this answer is resent to the AI on every later step):
        # side names appear once in `side_names`; each option lists its sides as
        # [side_id, items_to_buy] pairs.
        options = []
        side_names = {}
        for m in mains:
            fitting_sides = []
            for s in sides:
                # At relax 3 the other meal of the day doesn't count (it's planned again after).
                if not self.check_kosher(m["id"], s["id"], *((day, meal) if self.relax < 3 else ()))["ok"]:
                    continue
                if self.broken_wishes(day, meal, [m["id"], s["id"]]):
                    continue
                if self.relax < 3 and self.same_day_clash(day, meal, m["id"], s["id"]):
                    continue
                _, _, (main_missing, side_missing) = self.allocate([m, s], people, self.day_date(day))
                if main_missing <= limit and side_missing <= limit:
                    fitting_sides.append([s["id"], side_missing])
                    side_names[s["id"]] = s["name"]
            if not fitting_sides:
                continue
            option = {"main_id": m["id"], "main": m["name"], "kosher": m["kosher"],
                      "to_buy": len(self._missing(m, people)), "sides": fitting_sides}
            if self._uses_expiring(m):
                option["uses_expiring_soon"] = self._uses_expiring(m)
            options.append(option)
        options.sort(key=lambda o: o["to_buy"])

        result = {"day": day, "meal": meal, "people": people, "count": len(options),
                  "side_names": side_names, "options": options}
        if not options:
            result["hint"] = ("No valid option for this slot. Consider changing the other meal that day "
                              "(remove_meal), or an earlier meal that used up the dishes this slot needs. "
                              "If nothing helps, report this slot as unsolvable.")
        return result

    # ----- Tool: add_meal_to_plan -----

    def add_meal_to_plan(self, day, meal, main_id, side_id=None, people=None, user_asked=False):
        """Put a meal in the plan. Every rule is checked again here, so a bad choice
        by the AI is rejected with a reason instead of slipping into the plan.

        user_asked (mid-week only): the user named these dishes, so the week's variety
        limits (repeats) and the shopping limit don't apply: it's their choice. Kosher,
        what the family never eats, and the family's wishes still do. The meal is marked
        "chosen", so the program never changes it later just for variety."""
        user_asked = bool(user_asked) and self.mid_week
        # Every rejection says what went wrong AND what to try instead (the hint),
        # so the AI's next attempt is more focused, not a blind retry.
        error = self._slot_error(day, meal)
        if error:
            return {"error": error}
        locked = self._locked_error(day, meal)
        if locked:
            return locked
        if meal in self.plan.get(day, {}):
            return {"error": f"{day} {meal} is already planned",
                    "hint": "Choose an empty slot from get_plan_status, or call remove_meal first to replace it."}

        error = self._people_error(people)
        if error:
            return {"error": error}
        if people is None:
            people = self.prefs["people"]

        main = self.recipes.get(main_id)
        # A dish the user named can be served in either place (soup instead of the rice).
        if main is None or (main["dish_type"] != "main" and not user_asked):
            return {"error": f"{main_id} is not a main dish id",
                    "hint": "Use an id from search_recipes with dish_type='main'."}
        if side_id is None and self.prefs["side_dish_required"]:
            return {"error": "a side dish is required for every meal",
                    "hint": "Pass side_id: an id from search_recipes with dish_type='side'."}
        side = self.recipes.get(side_id) if side_id is not None else None
        if side_id is not None and (side is None or (side["dish_type"] != "side" and not user_asked)):
            return {"error": f"{side_id} is not a side dish id",
                    "hint": "Use an id from search_recipes with dish_type='side'."}
        dishes = [r for r in (main, side) if r is not None]

        for r in dishes:
            if meal not in r["meals"]:
                return {"error": f"'{r['name']}' is not suitable for {meal}",
                        "hint": f"Search with meal='{meal}' to get only dishes that fit."}
            if self.avoided_ingredients(r):
                return {"error": f"'{r['name']}' contains {self.avoided_ingredients(r)}, which the family doesn't eat",
                        "hint": "Use find_meal_options for this slot: it never lists these dishes."}

        usage = self.dish_usage()
        repeats_ok = self.relax >= 2 or user_asked
        asked_hint = " If the user asked for this dish by name, call again with user_asked=true."
        if not self.prefs["repeat_main_in_week"] and usage[main_id] > 0 and not repeats_ok:
            return {"error": f"'{main['name']}' is already in the plan this week",
                    "hint": "Use find_meal_options for this slot: it lists only mains not used yet." + asked_hint}
        max_side = self.prefs["max_uses_per_side_in_week"]
        if side and usage[side_id] >= max_side and not repeats_ok:
            return {"error": f"'{side['name']}' was already used {usage[side_id]} times this week (the maximum)",
                    "hint": "Use find_meal_options for this slot: it lists only sides still allowed." + asked_hint}

        kosher = self.check_kosher(main_id, side_id, *((day, meal) if self.relax < 3 else ()))
        if not kosher["ok"]:
            return {"error": "not kosher: " + "; ".join(kosher["problems"]),
                    "hint": " ".join(kosher["hints"])}
        broken = self.broken_wishes(day, meal, [main_id, side_id])
        if broken:
            return {"error": f"this meal goes against the family's wish: {broken}",
                    "hint": "Use find_meal_options for this slot: it lists only meals that fit the family's wishes."}
        clash = self.relax < 3 and self.same_day_clash(day, meal, main_id, side_id)
        if clash:
            return {"error": clash, "hint": "Use find_meal_options for this slot: it never repeats the other meal of the day."}

        reserved, to_buy, missing_per_dish = self.allocate(dishes, people, self.day_date(day))
        limit = 99 if user_asked else self._missing_limit(people, day, meal)
        for r, missing_here in zip(dishes, missing_per_dish):
            if missing_here > limit:
                return {"error": f"'{r['name']}' needs {missing_here} items bought (the limit is {limit})",
                        "hint": f"Use find_meal_options(day, meal, people={people}): it lists only dishes within the limit."
                                + asked_hint}

        for name, qty in reserved.items():
            self.available[name]["quantity"] -= qty
        self.plan.setdefault(day, {})[meal] = {
            "main": main_id,
            "side": side_id,
            "people": people,
            "reserved": dict(reserved),
            "to_buy": dict(to_buy),
        }
        if user_asked:
            self.plan[day][meal]["chosen"] = True
        return {
            "ok": True,
            "day": day,
            "meal": meal,
            "main": main["name"],
            "side": side["name"] if side else None,
            "meal_type": kosher["meal_type"],
            "people": people,
            "to_buy": [{"name": n, "quantity": q, "unit": self.units[n]} for n, q in to_buy.items()],
            "slots_left": len(self.empty_slots()),
        }

    # ----- Tool: change_side -----

    def change_side(self, day, meal, side_id, user_asked=False):
        """Keep the main dish and its people; put a different side (e.g. "add mashed
        potatoes on Thursday"). If the new side doesn't fit, the meal stays as it was."""
        error = self._slot_error(day, meal) or self._locked_error(day, meal)
        if error:
            return error if isinstance(error, dict) else {"error": error}
        old = self.plan.get(day, {}).get(meal)
        if old is None:
            return {"error": f"nothing is planned for {day} {meal}",
                    "hint": "Plan the meal with find_meal_options and add_meal_to_plan instead."}
        if old["side"] == side_id:
            return {"error": f"{day} {meal} already has this side"}
        before = copy.deepcopy((self.plan, self.available))
        self.remove_meal(day, meal)
        result = self.add_meal_to_plan(day, meal, old["main"], side_id, old["people"],
                                       user_asked=user_asked or old.get("chosen", False))
        if result.get("ok"):
            self._keep_notes(day, meal, old)
            return {**result, "side_before": self.recipes[old["side"]]["name"] if old["side"] else None}
        self.plan, self.available = before
        return result

    # ----- Tool: remove_meal -----

    def remove_meal(self, day, meal):
        """Take a meal out of the plan and give its ingredients back."""
        error = self._slot_error(day, meal)
        if error:
            return {"error": error}
        locked = self._locked_error(day, meal)
        if locked:
            return locked
        entry = self.plan.get(day, {}).pop(meal, None)
        if entry is None:
            return {"error": f"nothing is planned for {day} {meal}"}

        # Only what it reserved comes back: its to_buy was never bought.
        for name, qty in entry["reserved"].items():
            item = self.available.setdefault(
                name, {"quantity": 0, "unit": self.units[name], "expires": None}
            )
            item["quantity"] += qty

        self.cover_shopping_from_home()
        return {"ok": True, "removed": self.recipes[entry["main"]]["name"],
                "slots_left": len(self.empty_slots())}

    # ----- Tool: change_people -----

    def change_people(self, day, meal, people):
        """Same dishes, different number of people (e.g. guests are coming).

        If the same dishes don't fit any more (say, too much to buy), the meal is
        put back exactly as it was and the AI is told to choose a new dish.
        """
        error = self._slot_error(day, meal) or self._people_error(people)
        if error:
            return {"error": error}
        locked = self._locked_error(day, meal)
        if locked:
            return locked
        if people is None:
            return {"error": "people is required"}
        old = self.plan.get(day, {}).get(meal)
        if old is None:
            return {"error": f"nothing is planned for {day} {meal}"}
        if old["people"] == people:
            return {"error": f"{day} {meal} is already planned for {people} people"}

        before = copy.deepcopy((self.plan, self.available))
        self.remove_meal(day, meal)
        result = self.add_meal_to_plan(day, meal, old["main"], old["side"], people, user_asked=old.get("chosen", False))
        if result.get("ok"):
            self._keep_notes(day, meal, old)
            return {"ok": True, "day": day, "meal": meal, "kept": f"{result['main']} + {result['side']}",
                    "people_before": old["people"], "people": people, "to_buy": result["to_buy"]}

        # Put everything back exactly as it was (adding it again could fail too), then report.
        self.plan, self.available = before
        return {"error": f"the same dishes don't work for {people} people: {result['error']}",
                "hint": "Choose a different dish: remove_meal, then find_meal_options with the new people, "
                        "then add_meal_to_plan."}

    def _keep_notes(self, day, meal, old):
        """The user's changes to the dishes that are still in the meal stay with it."""
        entry = self.plan[day][meal]
        kept = (entry["main"], entry["side"])
        notes = [n for n in old.get("notes", []) if n["dish"] in kept]
        if notes:
            entry["notes"] = notes
        custom = {k: v for k, v in old.get("custom", {}).items() if int(k) in kept}
        if custom:
            entry["custom"] = custom

    def cover_shopping_from_home(self):
        """After ingredients come back home, use them to shrink other meals' shopping.

        Example: Sunday's shakshuka took the last tomatoes, so Monday's salad had to
        buy 2. If the shakshuka is removed, its tomatoes are home again, and
        Monday shouldn't still tell you to buy tomatoes.
        """
        for meals in self.plan.values():
            for entry in meals.values():
                for name in list(entry["to_buy"]):
                    have = self.available.get(name, {}).get("quantity", 0)
                    take = min(have, entry["to_buy"][name])
                    if take:
                        self.available[name]["quantity"] -= take
                        entry["reserved"][name] = entry["reserved"].get(name, 0) + take
                        entry["to_buy"][name] -= take
                        if entry["to_buy"][name] == 0:
                            del entry["to_buy"][name]

    # ----- Tool: find_in_plan -----

    def find_in_plan(self, ingredient):
        """Which planned meals contain an ingredient (part of a name is enough).

        So a request like "we're allergic to peas" is checked against the real
        recipes before anything changes: if no meal has peas, nothing needs to.
        """
        ingredient = str(ingredient or "").strip()
        if not ingredient:
            return {"error": "ingredient is required"}
        found = []
        for day in self.prefs["plan_days"]:
            for meal in self.prefs["meals"]:
                entry = self.plan.get(day, {}).get(meal)
                if entry is None:
                    continue
                for dish_id in (entry["main"], entry["side"]):
                    if dish_id is None:
                        continue
                    r = self.recipes[dish_id]
                    names = [i["name"] for i in r["ingredients"] if ingredient in i["name"]]
                    if names:
                        found.append({"day": day, "meal": meal, "dish": r["name"], "ingredients": names,
                                      "day_is_over": self.is_past(day)})
        return {"ingredient": ingredient, "count": len(found), "meals": found}

    # ----- Tool: get_plan_status -----

    def get_plan_status(self):
        """What's planned so far, what's still empty, and the shopping list."""
        planned = []
        for day in self.prefs["plan_days"]:
            for meal in self.prefs["meals"]:
                entry = self.plan.get(day, {}).get(meal)
                if entry is None:
                    continue
                ids = [entry["main"]] + ([entry["side"]] if entry["side"] else [])
                item = {
                    "day": day,
                    "meal": meal,
                    "main": self.recipes[entry["main"]]["name"],
                    "side": self.recipes[entry["side"]]["name"] if entry["side"] else None,
                    "meal_type": self.meal_kosher_type(ids),
                    "people": entry["people"],
                    "to_buy_items": len(entry["to_buy"]),
                }
                # Planned before the allergy was added to the preferences.
                avoided = list(dict.fromkeys(n for i in ids for n in self.avoided_ingredients(self.recipes[i])))
                if avoided:
                    item["contains_avoided"] = avoided
                broken = self.broken_wishes(day, meal, ids)
                if broken:
                    item["breaks_wish"] = broken
                planned.append(item)
        empty = self.empty_slots()
        return {
            "planned": planned,
            "empty_slots": empty,
            "done": not empty,
            "shopping_list": self.shopping_list(),
        }
