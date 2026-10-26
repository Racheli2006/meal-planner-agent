"""Tests for the agent's tools. Run from the project folder:  py -m unittest -v

The tests use small made-up data, not the real files in data/, so they keep
passing when the pantry is updated every week.
"""

import json
import sys
import unittest
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import tools
from tests import temp_dir

SUNDAY = date(2026, 10, 25)

UNITS = {"ביצים": "יחידה", "עגבניות": "יחידה", "בשר טחון": "גרם",
         "גבינה לבנה": "גרם", "אורז": "גרם", "פילה דג": "גרם"}


def recipe(id, name, dish_type, kosher, ingredients, meals=("lunch", "dinner"), fish=False):
    return {"id": id, "name": name, "dish_type": dish_type, "kosher": kosher,
            "contains_fish": fish, "meals": list(meals), "prep_minutes": 20, "servings": 4,
            "ingredients": [{"name": n, "quantity": q} for n, q in ingredients], "steps": ["..."]}


RECIPES = {
    "ingredient_units": UNITS,
    "recipes": [
        recipe(1, "קציצות", "main", "meat", [("בשר טחון", 500), ("ביצים", 1)]),
        recipe(2, "לביבות גבינה", "main", "dairy", [("גבינה לבנה", 500), ("ביצים", 2)]),
        recipe(3, "שקשוקה", "main", "parve", [("ביצים", 6), ("עגבניות", 5)]),
        recipe(4, "דג בתנור", "main", "parve", [("פילה דג", 600)], fish=True),
        recipe(10, "אורז", "side", "parve", [("אורז", 300)]),
        recipe(11, "סלט חלבי", "side", "dairy", [("גבינה לבנה", 100)]),
        recipe(12, "סלט עגבניות", "side", "parve", [("עגבניות", 2)]),
    ],
}


def make_planner(today=SUNDAY, eggs=5, **pref_overrides):
    pantry = {
        "ביצים": {"quantity": eggs, "unit": "יחידה", "expires": date(2026, 11, 12)},
        "עגבניות": {"quantity": 5, "unit": "יחידה", "expires": date(2026, 10, 28)},
        "גבינה לבנה": {"quantity": 250, "unit": "גרם", "expires": date(2026, 10, 20)},  # expired
        "אורז": {"quantity": 1000, "unit": "גרם", "expires": None},
    }
    prefs = {**tools.PREFERENCE_DEFAULTS, "people": 2, "max_missing_per_dish": 1,
             "max_missing_per_dish_with_guests": 2, **pref_overrides}
    return tools.MealPlanner(RECIPES, pantry, prefs, today=today)


class LoadingTests(unittest.TestCase):
    def write(self, content):
        path = temp_dir() / "file.json"
        path.write_text(json.dumps(content, ensure_ascii=False), encoding="utf-8")
        return path

    def test_missing_file(self):
        with self.assertRaises(tools.DataError):
            tools.load_json(temp_dir() / "nope.json")

    def test_bad_date(self):
        path = self.write({"items": [{"name": "ביצים", "quantity": 1, "unit": "יחידה", "expires": "31/12"}]})
        with self.assertRaises(tools.DataError):
            tools.load_pantry(UNITS, path)

    def test_unit_mismatch(self):
        path = self.write({"items": [{"name": "אורז", "quantity": 1, "unit": "ק\"ג", "expires": None}]})
        with self.assertRaises(tools.DataError):
            tools.load_pantry(UNITS, path)

    def test_missing_preferences_get_defaults(self):
        prefs = tools.load_preferences(self.write({"people": 3}))
        self.assertEqual(prefs["people"], 3)
        self.assertEqual(prefs["meals"], tools.PREFERENCE_DEFAULTS["meals"])


class PantryAndQuantityTests(unittest.TestCase):
    def test_expired_item_is_not_used(self):
        p = make_planner()
        self.assertIn("גבינה לבנה", p.get_pantry()["expired_not_used"])
        self.assertNotIn("גבינה לבנה", p.available)

    def test_pantry_sorted_soonest_first(self):
        names = [i["name"] for i in make_planner().get_pantry()["items"]]
        self.assertEqual(names[0], "עגבניות")   # expires in 3 days
        self.assertEqual(names[-1], "אורז")     # no expiry date

    def test_countable_units_round_up(self):
        p = make_planner()
        eggs = [i for i in p.scaled_ingredients(p.recipes[2]) if i["name"] == "ביצים"][0]
        self.assertEqual(eggs["quantity"], 1)   # 2 eggs for 4 -> 1 for 2
        p = make_planner(people=3)
        eggs = [i for i in p.scaled_ingredients(p.recipes[1]) if i["name"] == "ביצים"][0]
        self.assertEqual(eggs["quantity"], 1)   # 0.75 egg -> 1, never 0

    def test_empty_search_explains_meat_sides(self):
        result = make_planner().search_recipes(dish_type="side", kosher="meat")
        self.assertEqual(result["count"], 0)
        self.assertIn("parve", result["hint"])

    def test_check_missing(self):
        result = make_planner().check_missing(1)  # meatballs: no meat at home
        self.assertEqual([m["name"] for m in result["missing"]], ["בשר טחון"])
        self.assertTrue(result["within_limit"])


class KosherTests(unittest.TestCase):
    def test_meat_with_dairy_side(self):
        result = make_planner().check_kosher(1, 11)
        self.assertFalse(result["ok"])
        self.assertTrue(result["hints"])

    def test_parve_with_dairy_is_dairy(self):
        self.assertEqual(make_planner().check_kosher(3, 11)["meal_type"], "dairy")

    def test_no_dairy_dinner_after_meat_lunch(self):
        p = make_planner()
        p.add_meal_to_plan("ראשון", "lunch", 1, 10)
        self.assertFalse(p.check_kosher(2, 10, "ראשון", "dinner")["ok"])
        self.assertTrue(p.check_kosher(3, 10, "ראשון", "dinner")["ok"])


class PlanTests(unittest.TestCase):
    def test_add_reserves_ingredients(self):
        p = make_planner()
        self.assertTrue(p.add_meal_to_plan("ראשון", "dinner", 3, 10)["ok"])  # shakshuka: 3 eggs
        self.assertEqual(p.available["ביצים"]["quantity"], 2)

    def test_rejected_meal_changes_nothing(self):
        p = make_planner(eggs=0, max_missing_per_dish=0)  # shakshuka is missing eggs
        rice_before = p.available["אורז"]["quantity"]
        result = p.add_meal_to_plan("ראשון", "dinner", 3, 10)
        self.assertIn("error", result)
        self.assertIn("hint", result)
        self.assertEqual(p.available["אורז"]["quantity"], rice_before)  # the side took nothing
        self.assertEqual(p.plan, {})

    def test_main_not_repeated(self):
        p = make_planner()
        p.add_meal_to_plan("ראשון", "lunch", 1, 10)
        self.assertIn("error", p.add_meal_to_plan("שני", "lunch", 1, 10))

    def test_remove_gives_back_and_shrinks_shopping(self):
        p = make_planner(eggs=3, max_missing_per_dish=2)
        p.add_meal_to_plan("ראשון", "dinner", 3, 10)  # shakshuka takes all 3 eggs
        p.add_meal_to_plan("שני", "lunch", 1, 10)     # meatballs must buy 1 egg
        self.assertEqual(p.plan["שני"]["lunch"]["to_buy"].get("ביצים"), 1)

        p.remove_meal("ראשון", "dinner")              # 3 eggs come back home
        self.assertNotIn("ביצים", p.plan["שני"]["lunch"]["to_buy"])
        self.assertEqual(p.available["ביצים"]["quantity"], 2)

    def test_guests_get_higher_limit(self):
        p = make_planner()
        self.assertEqual(p.check_missing(1)["max_missing_allowed"], 1)
        self.assertEqual(p.check_missing(1, people=5)["max_missing_allowed"], 2)

    def test_people_zero_is_an_error(self):
        self.assertIn("error", make_planner().add_meal_to_plan("ראשון", "lunch", 1, 10, people=0))

    def test_past_day_cannot_be_changed(self):
        p = make_planner(today=date(2026, 10, 27))  # Tuesday
        self.assertIn("error", p.add_meal_to_plan("שני", "lunch", 1, 10))
        self.assertNotIn({"day": "ראשון", "meal": "lunch"}, p.empty_slots())

    def test_options_pass_all_hard_rules(self):
        p = make_planner(max_missing_per_dish=2)
        p.add_meal_to_plan("ראשון", "lunch", 1, 10)  # meat lunch
        options = p.find_meal_options("ראשון", "dinner")["options"]
        mains = {o["main_id"] for o in options}
        self.assertNotIn(1, mains)   # already used this week
        self.assertNotIn(2, mains)   # dairy dinner after a meat lunch
        self.assertIn(3, mains)      # shakshuka (parve) is fine
        shakshuka = next(o for o in options if o["main_id"] == 3)
        self.assertNotIn(11, [side_id for side_id, _ in shakshuka["sides"]])  # dairy side would make it a dairy dinner

    def test_no_options_gives_hint(self):
        p = make_planner(max_missing_per_dish=0, eggs=0)
        result = p.find_meal_options("ראשון", "lunch")
        self.assertEqual(result["count"], 0)
        self.assertIn("hint", result)

    def test_avoided_ingredient_is_never_planned(self):
        p = make_planner(max_missing_per_dish=2, avoid_ingredients=["בשר"])  # part of "בשר טחון"
        mains = {o["main_id"] for o in p.find_meal_options("ראשון", "lunch")["options"]}
        self.assertNotIn(1, mains)
        self.assertIn("doesn't eat", p.add_meal_to_plan("ראשון", "lunch", 1, 10)["error"])
        self.assertNotIn(1, [r["id"] for r in p.search_recipes()["recipes"]])

    def test_guest_allergy_excludes_only_this_meal(self):
        p = make_planner(max_missing_per_dish=2)
        lunch = {o["main_id"] for o in p.find_meal_options("ראשון", "lunch", exclude=["ביצים"])["options"]}
        dinner = {o["main_id"] for o in p.find_meal_options("ראשון", "dinner")["options"]}
        self.assertNotIn(3, lunch)   # shakshuka has eggs
        self.assertIn(3, dinner)     # the rest of the week isn't affected
        self.assertIn("error", p.find_meal_options("ראשון", "lunch", exclude="ביצים"))

    def test_wish_rule_is_enforced(self):
        rule = {"note": "בראשון בערב בשרי", "days": ["ראשון"], "meals": ["dinner"], "allowed": ["meat"]}
        p = make_planner(max_missing_per_dish=2, family_notes=[rule["note"]], slot_rules=[rule])
        options = p.find_meal_options("ראשון", "dinner")["options"]
        self.assertEqual({o["main_id"] for o in options}, {1})               # only the meatballs
        self.assertIn("wish", p.add_meal_to_plan("ראשון", "dinner", 3, 10)["error"])
        self.assertIn(3, {o["main_id"] for o in p.find_meal_options("ראשון", "lunch")["options"]})  # lunch is free

    def test_wish_for_a_kind_of_dish(self):
        rule = {"note": "בראשון בצהריים משהו עם ביצים", "days": ["ראשון"], "meals": ["lunch"], "must_have": ["ביצים"]}
        p = make_planner(max_missing_per_dish=2, family_notes=[rule["note"]], slot_rules=[rule])
        mains = {o["main_id"] for o in p.find_meal_options("ראשון", "lunch")["options"]}
        self.assertEqual(mains, {1, 2, 3})                         # the dishes with eggs; not the fish
        self.assertIn("wish", p.add_meal_to_plan("ראשון", "lunch", 4, 10)["error"])

    def test_wish_slot_allows_more_shopping(self):
        rule = {"note": "בראשון בצהריים בשר", "days": ["ראשון"], "meals": ["lunch"], "must_have": ["בשר"]}
        p = make_planner(max_missing_per_dish=0, max_missing_per_dish_with_guests=2,
                         family_notes=[rule["note"]], slot_rules=[rule])
        self.assertEqual({o["main_id"] for o in p.find_meal_options("ראשון", "lunch")["options"]}, {1})
        dinner = {o["main_id"] for o in p.find_meal_options("ראשון", "dinner")["options"]}
        self.assertNotIn(1, dinner)  # no wish there: the normal limit, so no meat to buy

    def test_wish_against_a_dish(self):
        rule = {"note": "בלי דגים בראשון", "days": ["ראשון"], "meals": ["lunch", "dinner"], "must_not_have": ["דגים"]}
        p = make_planner(max_missing_per_dish=2, family_notes=[rule["note"]], slot_rules=[rule])
        self.assertNotIn(4, {o["main_id"] for o in p.find_meal_options("ראשון", "dinner")["options"]})
        self.assertIn(4, {o["main_id"] for o in p.find_meal_options("שני", "dinner")["options"]})

    def test_family_notes_reach_the_ai(self):
        from prompts import change_goal, plan_goal
        prefs = {**tools.PREFERENCE_DEFAULTS, "family_notes": ["הילדים לא אוהבים פטריות"]}
        self.assertIn("הילדים לא אוהבים פטריות", plan_goal(prefs, ["ראשון"]))
        self.assertIn("הילדים לא אוהבים פטריות", change_goal("x", "ראשון", prefs))
        self.assertNotIn("standing wishes", plan_goal(tools.PREFERENCE_DEFAULTS, ["ראשון"]))

    def test_meal_planned_before_the_allergy_is_flagged(self):
        p = make_planner(max_missing_per_dish=2)
        p.add_meal_to_plan("ראשון", "lunch", 1, 10)
        p.prefs["avoid_ingredients"] = ["בשר"]
        self.assertEqual(p.get_plan_status()["planned"][0]["contains_avoided"], ["בשר טחון"])

    def test_find_in_plan(self):
        p = make_planner(max_missing_per_dish=2)
        p.add_meal_to_plan("ראשון", "lunch", 1, 10)   # meatballs: meat and eggs
        p.add_meal_to_plan("שני", "dinner", 4, 10)    # fish
        found = p.find_in_plan("ביצים")
        self.assertEqual([(m["day"], m["dish"]) for m in found["meals"]], [("ראשון", "קציצות")])
        self.assertEqual(p.find_in_plan("אפונה")["count"], 0)  # not in any meal: nothing to change

    def test_done_when_all_slots_filled(self):
        p = make_planner(plan_days=["ראשון"], max_missing_per_dish=2)
        p.add_meal_to_plan("ראשון", "lunch", 1, 10)
        self.assertFalse(p.get_plan_status()["done"])
        p.add_meal_to_plan("ראשון", "dinner", 4, 12)  # not the rice again the same day
        self.assertTrue(p.get_plan_status()["done"])


class SavedPlanTests(unittest.TestCase):
    def test_save_and_restore_mid_week(self):
        p = make_planner()
        p.add_meal_to_plan("שלישי", "lunch", 1, 10)  # buys 250 g meat
        path = temp_dir() / "plan.json"
        tools.save_plan(p, path)

        q = make_planner(today=date(2026, 10, 26))
        q.restore(tools.load_plan(RECIPES, path))
        self.assertEqual(q.plan["שלישי"]["lunch"]["main"], 1)
        q.declare_change_targets([{"day": "שלישי", "meal": "lunch"}])
        q.remove_meal("שלישי", "lunch")
        # The meat was on the shopping list but never marked bought, so it isn't home.
        self.assertEqual(q.available.get("בשר טחון", {}).get("quantity", 0), 0)
        self.assertEqual(q.shopping_list(), [])

    def test_change_people_keeps_dishes(self):
        p = make_planner(max_missing_per_dish=1, max_missing_per_dish_with_guests=2)
        p.add_meal_to_plan("ראשון", "dinner", 3, 10)   # shakshuka for 2: 3 eggs
        result = p.change_people("ראשון", "dinner", 4)  # for 4: 6 eggs, 5 at home
        self.assertTrue(result["ok"])
        self.assertEqual(p.plan["ראשון"]["dinner"]["main"], 3)
        self.assertEqual(p.plan["ראשון"]["dinner"]["people"], 4)
        self.assertEqual(p.plan["ראשון"]["dinner"]["to_buy"].get("ביצים"), 1)

    def test_change_people_failure_restores_meal(self):
        p = make_planner(max_missing_per_dish=1, max_missing_per_dish_with_guests=0)
        p.add_meal_to_plan("ראשון", "lunch", 1, 10)    # meatballs: meat must be bought
        before = dict(p.plan["ראשון"]["lunch"])
        result = p.change_people("ראשון", "lunch", 6)  # guests allow 0 items to buy: can't
        self.assertIn("error", result)
        self.assertIn("hint", result)
        self.assertEqual(p.plan["ראשון"]["lunch"], before)

    def mid_week_planner(self):
        """A plan made on Sunday, reopened on Monday for a change."""
        settings = {"max_missing_per_dish": 2, "max_uses_per_side_in_week": 5}
        p = make_planner(**settings)
        for day, meal, main, side in [("ראשון", "dinner", 3, 10), ("שלישי", "lunch", 1, 10), ("שלישי", "dinner", 4, 12)]:
            self.assertTrue(p.add_meal_to_plan(day, meal, main, side)["ok"])
        path = temp_dir() / "plan.json"
        tools.save_plan(p, path)
        q = make_planner(today=date(2026, 10, 26), **settings)
        q.restore(tools.load_plan(RECIPES, path))
        return q

    def test_mid_week_change_needs_declared_slots(self):
        q = self.mid_week_planner()
        self.assertIn("declare", q.remove_meal("שלישי", "dinner")["error"])
        q.declare_change_targets([{"day": "שלישי", "meal": "dinner"}])
        self.assertTrue(q.remove_meal("שלישי", "dinner")["ok"])
        self.assertIn("locked", q.remove_meal("שלישי", "lunch")["error"])   # not declared

    def test_targets_can_be_declared_once(self):
        q = self.mid_week_planner()
        q.declare_change_targets([{"day": "שלישי", "meal": "dinner"}])
        self.assertIn("error", q.declare_change_targets([{"day": "שלישי", "meal": "lunch"}]))

    def test_past_day_request_locks_everything(self):
        q = self.mid_week_planner()
        self.assertIn("over", q.declare_change_targets([{"day": "ראשון", "meal": "dinner"}])["error"])
        # The AI can't "make up for it" by changing another day:
        self.assertIn("locked", q.remove_meal("שלישי", "dinner")["error"])
        self.assertIn("error", q.declare_change_targets([{"day": "שלישי", "meal": "dinner"}]))

    def test_no_saved_plan(self):
        with self.assertRaises(tools.DataError):
            tools.load_plan(RECIPES, temp_dir() / "plan.json")

    def test_week_start(self):
        self.assertEqual(tools.week_start_for(date(2026, 10, 28)), SUNDAY)          # Wednesday
        self.assertEqual(tools.week_start_for(date(2026, 10, 30)), date(2026, 11, 1))  # Friday


if __name__ == "__main__":
    unittest.main()
