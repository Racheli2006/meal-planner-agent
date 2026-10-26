"""Tests for updating the home's data during the week (no AI). Run:  py -m unittest -v"""

import json
import unittest
from datetime import date
from pathlib import Path

import household
import tools
from tests.test_tools import RECIPES, SUNDAY, UNITS, make_planner
from tests import temp_dir

MONDAY = date(2026, 10, 26)


def pantry_file(items):
    path = temp_dir() / "pantry.json"
    path.write_text(json.dumps({"items": items}, ensure_ascii=False), encoding="utf-8")
    return path


def reopen(planner, pantry_path, today=MONDAY):
    """Save the plan and load it again the way the app does on every request."""
    plan_path = pantry_path.parent / "plan.json"
    tools.save_plan(planner, plan_path)
    pantry = tools.load_pantry(UNITS, pantry_path)
    q = tools.MealPlanner(RECIPES, pantry, planner.prefs, today=today)
    q.restore(tools.load_plan(RECIPES, plan_path))
    return q


EGGS_AND_RICE = [{"name": "ביצים", "quantity": 5, "unit": "יחידה", "expires": "2026-11-12"},
                 {"name": "אורז", "quantity": 1000, "unit": "גרם", "expires": None}]


def planner_from(path, today=SUNDAY, **prefs):
    p = make_planner(today=today, **prefs)
    p.available = {n: dict(i) for n, i in tools.load_pantry(UNITS, path).items()}
    return p


class PantryFileTests(unittest.TestCase):
    def test_write_drops_finished_items_and_uses_recipe_units(self):
        path = pantry_file([])
        items = household.write_pantry(
            [{"name": " ביצים ", "quantity": 3, "unit": "גרם"},
             {"name": "אורז", "quantity": 0},
             {"name": "חומץ", "quantity": 500, "unit": 'מ"ל', "expires": ""}], UNITS, SUNDAY, path)
        self.assertEqual([i["name"] for i in items], ["ביצים", "חומץ"])
        self.assertEqual(items[0]["unit"], "יחידה")  # the recipes' unit wins
        self.assertIsNone(items[1]["expires"])

    def test_bad_item_is_refused_and_file_kept(self):
        path = pantry_file(EGGS_AND_RICE)
        for bad in ({"name": "ביצים", "quantity": -1}, {"name": "", "quantity": 1},
                    {"name": "חומץ", "quantity": 1}, {"name": "ביצים", "quantity": 1, "expires": "מחר"}):
            with self.assertRaises(tools.DataError):
                household.write_pantry([bad], UNITS, SUNDAY, path)
        self.assertEqual(len(household.read_pantry(path)), 2)

    def test_purchase_adds_and_keeps_sooner_date(self):
        path = pantry_file(EGGS_AND_RICE)
        household.add_to_pantry([{"name": "ביצים", "quantity": 12, "expires": "2026-11-30"},
                                 {"name": "עגבניות", "quantity": 4}], UNITS, SUNDAY, path)
        items = {i["name"]: i for i in household.read_pantry(path)}
        self.assertEqual(items["ביצים"]["quantity"], 17)
        self.assertEqual(items["ביצים"]["expires"], "2026-11-12")
        self.assertEqual(items["עגבניות"]["unit"], "יחידה")

    def test_purchase_replaces_expired_item(self):
        path = pantry_file([{"name": "גבינה לבנה", "quantity": 250, "unit": "גרם", "expires": "2026-10-20"}])
        household.add_to_pantry([{"name": "גבינה לבנה", "quantity": 500, "expires": "2026-11-05"}], UNITS, SUNDAY, path)
        item = household.read_pantry(path)[0]
        self.assertEqual((item["quantity"], item["expires"]), (500, "2026-11-05"))


class WeekTests(unittest.TestCase):
    def setUp(self):
        self.path = pantry_file(EGGS_AND_RICE)
        self.p = planner_from(self.path, max_missing_per_dish=2)
        self.assertTrue(self.p.add_meal_to_plan("שלישי", "lunch", 1, 10)["ok"])  # meatballs: buy 250 g meat

    def test_bought_items_move_to_pantry_and_meal(self):
        bought = household.record_shopping(self.p, path=self.path)
        self.assertEqual(bought, {"בשר טחון": 250})
        self.assertEqual(self.p.shopping_list(), [])
        q = reopen(self.p, self.path)
        self.assertEqual(q.available["בשר טחון"]["quantity"], 0)  # bought, and saved for the meatballs

        # Not cooking it after all: the bought meat is free at home.
        q.declare_change_targets([{"day": "שלישי", "meal": "lunch"}])
        q.remove_meal("שלישי", "lunch")
        self.assertEqual(q.available["בשר טחון"]["quantity"], 250)

    def test_bought_less_than_the_list(self):
        # Meatballs need 250 g meat; only 100 g were in the shop.
        bought = household.record_shopping(self.p, path=self.path, amounts={"בשר טחון": 100})
        self.assertEqual(bought, {"בשר טחון": 100})
        self.assertEqual(self.p.plan["שלישי"]["lunch"]["to_buy"], {"בשר טחון": 150})   # the rest stays on the list
        self.assertEqual(self.p.plan["שלישי"]["lunch"]["reserved"]["בשר טחון"], 100)
        self.assertEqual({i["name"]: i["quantity"] for i in household.read_pantry(self.path)}["בשר טחון"], 100)

    def test_bought_more_than_the_list(self):
        household.record_shopping(self.p, path=self.path, amounts={"בשר טחון": 500})
        self.assertEqual(self.p.shopping_list(), [])
        q = reopen(self.p, self.path)
        self.assertEqual(q.available["בשר טחון"]["quantity"], 250)  # the extra is free at home

    def test_less_goes_to_the_soonest_meal(self):
        # No eggs at home: Monday's shakshuka needs 3, Tuesday's meatballs 1. Only 2 were bought.
        path = pantry_file([{"name": "אורז", "quantity": 1000, "unit": "גרם", "expires": None},
                            {"name": "עגבניות", "quantity": 5, "unit": "יחידה", "expires": None}])
        p = planner_from(path, max_missing_per_dish=2)
        self.assertTrue(p.add_meal_to_plan("שלישי", "lunch", 1, 10)["ok"])
        self.assertTrue(p.add_meal_to_plan("שני", "dinner", 3, 10)["ok"])
        household.record_shopping(p, path=path, amounts={"ביצים": 2})
        self.assertEqual(p.plan["שני"]["dinner"]["to_buy"]["ביצים"], 1)    # Monday got the 2 eggs
        self.assertEqual(p.plan["שלישי"]["lunch"]["to_buy"]["ביצים"], 1)   # Tuesday still needs its 1

    def test_only_ticked_items_are_bought(self):
        self.assertTrue(self.p.add_meal_to_plan("שלישי", "dinner", 3, 12)["ok"])  # shakshuka: buys tomatoes
        household.record_shopping(self.p, {"עגבניות"}, path=self.path)
        self.assertEqual([i["name"] for i in self.p.shopping_list()], ["בשר טחון"])

    def test_skip_works_on_a_past_day_and_frees_ingredients(self):
        household.record_shopping(self.p, path=self.path)
        q = reopen(self.p, self.path, today=date(2026, 10, 29))  # Thursday: Tuesday is over
        household.skip_meal(q, "שלישי", "lunch")
        self.assertNotIn("lunch", q.plan.get("שלישי", {}))
        self.assertEqual(q.available["בשר טחון"]["quantity"], 250)
        with self.assertRaises(tools.DataError):
            household.skip_meal(q, "שלישי", "lunch")

    def test_guests_without_ai(self):
        q = reopen(self.p, self.path)
        result = household.set_people(q, "שלישי", "lunch", 4)
        self.assertTrue(result["ok"])
        self.assertEqual(q.plan["שלישי"]["lunch"]["people"], 4)

    def test_less_in_pantry_shows_up_in_shopping(self):
        q = reopen(self.p, self.path)
        self.assertEqual(q.plan["שלישי"]["lunch"]["to_buy"], {"בשר טחון": 250})
        household.write_pantry([{"name": "ביצים", "quantity": 0}, {"name": "אורז", "quantity": 100}],
                               UNITS, MONDAY, self.path)
        changed = household.reallocate(q, tools.load_pantry(UNITS, self.path))
        self.assertEqual(changed, 1)
        self.assertEqual(q.plan["שלישי"]["lunch"]["to_buy"], {"בשר טחון": 250, "ביצים": 1, "אורז": 50})
        self.assertEqual(q.plan["שלישי"]["lunch"]["reserved"], {"אורז": 100})

    def test_bought_meat_is_used_after_recount(self):
        household.add_to_pantry([{"name": "בשר טחון", "quantity": 500}], UNITS, MONDAY, self.path)
        q = reopen(self.p, self.path)
        household.reallocate(q, tools.load_pantry(UNITS, self.path))
        self.assertEqual(q.shopping_list(), [])
        self.assertEqual(q.available["בשר טחון"]["quantity"], 250)  # the other half is free

    def test_recount_leaves_past_days_alone(self):
        q = reopen(self.p, self.path, today=date(2026, 10, 29))  # Tuesday is over
        before = dict(q.plan["שלישי"]["lunch"])
        household.write_pantry([], UNITS, MONDAY, self.path)
        self.assertEqual(household.reallocate(q, tools.load_pantry(UNITS, self.path)), 0)
        self.assertEqual(q.plan["שלישי"]["lunch"], before)

    def test_leftovers_for_next_week(self):
        household.record_shopping(self.p, path=self.path)
        left = {i["name"]: i["quantity"] for i in household.pantry_after_week(household.read_pantry(self.path), self.p.plan)}
        # Meatballs for 2 took 1 egg and all 250 g of meat; the rice side took 150 g.
        self.assertEqual(left, {"ביצים": 4, "אורז": 850})

    def test_new_week_archives_the_plan(self):
        plan_path = self.path.parent / "plan.json"
        tools.save_plan(self.p, plan_path)
        archive = self.path.parent / "archive"
        household.start_new_week([{"name": "ביצים", "quantity": 4}], UNITS, date(2026, 11, 1),
                                 self.path, plan_path, archive)
        self.assertFalse(plan_path.exists())
        self.assertTrue((archive / "plan-2026-10-25.json").exists())
        self.assertEqual(household.read_pantry(self.path)[0]["quantity"], 4)


class SlotsToFixTests(unittest.TestCase):
    def setUp(self):
        self.path = pantry_file(EGGS_AND_RICE)
        p = planner_from(self.path, max_missing_per_dish=2)
        p.add_meal_to_plan("שלישי", "dinner", 3, 10)  # shakshuka: parve
        self.q = reopen(p, self.path)

    def test_everything_fits(self):
        self.assertEqual(household.slots_to_fix(self.q), [])

    def test_meal_against_a_wish(self):
        rule = {"note": "בשלישי בערב בשרי", "days": ["שלישי"], "meals": ["dinner"], "allowed": ["meat"]}
        self.q.prefs.update(family_notes=[rule["note"]], slot_rules=[rule])
        [slot] = household.slots_to_fix(self.q)
        self.assertEqual((slot["day"], slot["meal"]), ("שלישי", "dinner"))
        self.assertIn("בשלישי בערב בשרי", slot["reasons"][0])

    def test_forbidden_ingredient(self):
        self.q.prefs["avoid_ingredients"] = ["עגבניות"]
        self.q.prefs["max_missing_per_dish"] = 0  # more shopping than allowed is not a reason to change
        [slot] = household.slots_to_fix(self.q)
        self.assertEqual(len(slot["reasons"]), 1)

    def wish(self, words, **extra):
        rule = {"note": "בשלישי בערב משהו עם " + words[0], "days": ["שלישי"], "meals": ["dinner"],
                "must_have": words, **extra}
        self.q.prefs.update(family_notes=[rule["note"]], slot_rules=[rule])

    def test_menu_follows_a_new_wish(self):
        self.wish(["בשר"])
        self.assertEqual(household.fix_menu(self.q), [])
        self.assertEqual(self.q.plan["שלישי"]["dinner"]["main"], 1)  # the meatballs

    def test_more_shopping_when_that_is_what_it_takes(self):
        self.wish(["בשר"])
        self.q.prefs["max_missing_per_dish"] = self.q.prefs["max_missing_per_dish_with_guests"] = 0
        self.assertEqual(household.fix_menu(self.q), [])   # the meat is bought anyway
        self.assertEqual(self.q.plan["שלישי"]["dinner"]["main"], 1)

    def test_dish_moves_from_another_day(self):
        # The only fish dish is on Monday; the wish wants it on Tuesday.
        q = reopen(self.q, self.path)
        q.change_targets = {("שני", "lunch")}
        self.assertTrue(q.add_meal_to_plan("שני", "lunch", 4, 10)["ok"])
        self.q = q
        self.wish(["דג"])
        self.assertEqual(household.fix_menu(self.q), [])
        self.assertEqual(self.q.plan["שלישי"]["dinner"]["main"], 4)
        self.assertNotEqual(self.q.plan["שני"]["lunch"]["main"], 4)  # Monday got another dish

    def test_wish_wins_over_the_other_meal_of_the_day(self):
        # Fish is wanted for Tuesday lunch, but the only fish dish is Tuesday dinner:
        # the fish goes to lunch, and dinner (no wish) gets something else.
        q = self.q
        q.change_targets = {("שלישי", "dinner"), ("שלישי", "lunch")}
        q.remove_meal("שלישי", "dinner")
        self.assertTrue(q.add_meal_to_plan("שלישי", "lunch", 3, 10)["ok"])   # shakshuka lunch
        self.assertTrue(q.add_meal_to_plan("שלישי", "dinner", 4, 12)["ok"])  # fish dinner
        rule = {"note": "בשלישי בצהריים דג", "days": ["שלישי"], "meals": ["lunch"], "must_have": ["דג"]}
        q.prefs.update(family_notes=[rule["note"]], slot_rules=[rule])
        self.assertEqual(household.fix_menu(q), [])
        self.assertEqual(q.plan["שלישי"]["lunch"]["main"], 4)
        self.assertNotEqual(q.plan["שלישי"]["dinner"]["main"], 4)

    def test_meat_wish_at_lunch_changes_a_dairy_dinner(self):
        # Wanted: meat for Tuesday lunch. But dinner is dairy, and no dairy after meat:
        # lunch gets the meat, and dinner (no wish) becomes something that isn't dairy.
        q = self.q
        q.change_targets = {("שלישי", "dinner"), ("שלישי", "lunch")}
        q.remove_meal("שלישי", "dinner")
        self.assertTrue(q.add_meal_to_plan("שלישי", "lunch", 3, 10)["ok"])   # shakshuka lunch
        self.assertTrue(q.add_meal_to_plan("שלישי", "dinner", 2, 12)["ok"])  # cheese latkes dinner
        rule = {"note": "בשלישי בצהריים בשרי", "days": ["שלישי"], "meals": ["lunch"], "allowed": ["meat"]}
        q.prefs.update(family_notes=[rule["note"]], slot_rules=[rule])
        self.assertEqual(household.fix_menu(q), [])
        self.assertEqual(q.plan["שלישי"]["lunch"]["main"], 1)
        dinner = q.plan["שלישי"]["dinner"]
        self.assertNotEqual(q.meal_kosher_type([dinner["main"], dinner["side"]]), "dairy")
        self.assertEqual(household.slots_to_fix(q), [])

    def test_contradicting_wishes_are_found(self):
        fish = {"note": "בשלישי בצהריים דג", "days": ["שלישי"], "meals": ["lunch"], "must_have": ["דג"]}
        meat = {"note": "בשלישי בצהריים בשרי", "days": ["שלישי"], "meals": ["lunch", "dinner"], "allowed": ["meat"]}
        rice = {"note": "בשלישי בצהריים אורז", "days": ["שלישי"], "meals": ["lunch"], "must_have": ["אורז"]}
        wed = {"note": "ברביעי בשרי", "days": ["רביעי"], "meals": ["lunch"], "allowed": ["meat"]}
        self.q.prefs.update(family_notes=[fish["note"], wed["note"]], slot_rules=[fish, wed])
        self.assertEqual(household.conflicting_wishes(self.q, meat), [fish["note"]])  # fish can't be meat
        self.assertEqual(household.conflicting_wishes(self.q, rice), [])              # fish with rice is fine
        # Different meals of one day: meat lunch and dairy dinner can't both be.
        lunch_meat = {"note": "ברביעי בצהריים בשרי", "days": ["רביעי"], "meals": ["lunch"], "allowed": ["meat"]}
        dinner_dairy = {"note": "ברביעי בערב חלבי", "days": ["רביעי"], "meals": ["dinner"], "allowed": ["dairy"]}
        dinner_parve = {"note": "ברביעי בערב פרווה", "days": ["רביעי"], "meals": ["dinner"], "allowed": ["parve"]}
        self.q.prefs.update(family_notes=[lunch_meat["note"]], slot_rules=[lunch_meat])
        self.assertEqual(household.conflicting_wishes(self.q, dinner_dairy), [lunch_meat["note"]])
        self.assertEqual(household.conflicting_wishes(self.q, dinner_parve), [])

    def test_a_change_that_cant_be_finished_is_undone(self):
        # Four mains only: meat for Tuesday lunch means the dairy dinner must go, and here
        # nothing else can take its place. Then nothing changes, and the menu stays kosher.
        q = reopen(planner_from(self.path, max_missing_per_dish=2), self.path)
        q.change_targets = None
        q.mid_week = False
        for day, meal, main, side in [("שני", "lunch", 1, 10), ("שני", "dinner", 3, 12),
                                      ("שלישי", "lunch", 4, 10), ("שלישי", "dinner", 2, 12)]:
            self.assertTrue(q.add_meal_to_plan(day, meal, main, side)["ok"])
        before = {(d, m): e["main"] for d, meals in q.plan.items() for m, e in meals.items()}
        rule = {"note": "בשלישי בצהריים בשרי", "days": ["שלישי"], "meals": ["lunch"], "allowed": ["meat"]}
        q.prefs.update(family_notes=[rule["note"]], slot_rules=[rule])
        [slot] = household.fix_menu(q)
        self.assertEqual((slot["day"], slot["meal"]), ("שלישי", "lunch"))
        self.assertEqual({(d, m): e["main"] for d, meals in q.plan.items() for m, e in meals.items()}, before)
        self.assertTrue(all(s["family"] for s in household.slots_to_fix(q)))  # only the wish is unmet

    def test_repeated_side_changes_only_the_side(self):
        q = self.q
        q.change_targets = {("שלישי", "lunch")}
        self.assertTrue(q.add_meal_to_plan("שלישי", "lunch", 1, 12)["ok"])  # meatballs + tomato salad
        q.plan["שלישי"]["dinner"]["side"] = 12                              # the same salad at dinner
        self.assertEqual(household.fix_menu(q), [])
        self.assertEqual(q.plan["שלישי"]["dinner"]["main"], 3)              # still the shakshuka
        self.assertNotEqual(q.plan["שלישי"]["dinner"]["side"], 12)

    def test_replace_picks_a_different_main(self):
        self.assertTrue(household.replace_meal(self.q, "שלישי", "dinner"))
        self.assertNotEqual(self.q.plan["שלישי"]["dinner"]["main"], 3)  # not the shakshuka again

    def test_replace_fills_an_empty_slot(self):
        self.assertTrue(household.replace_meal(self.q, "רביעי", "lunch"))
        self.assertIn("lunch", self.q.plan["רביעי"])

    def test_replace_keeps_the_meal_when_nothing_else_fits(self):
        self.wish(["ביצים"], allowed=["parve"])  # only the shakshuka fits Tuesday dinner
        self.assertFalse(household.replace_meal(self.q, "שלישי", "dinner"))
        self.assertEqual(self.q.plan["שלישי"]["dinner"]["main"], 3)

    def test_impossible_wish_leaves_the_meal(self):
        self.wish(["שוקולד"])
        [slot] = household.fix_menu(self.q)
        self.assertEqual((slot["day"], slot["meal"]), ("שלישי", "dinner"))
        self.assertEqual(self.q.plan["שלישי"]["dinner"]["main"], 3)  # still the shakshuka

    def test_removing_a_wish_changes_nothing(self):
        self.wish(["בשר"])
        household.fix_menu(self.q)
        self.q.prefs.update(family_notes=[], slot_rules=[])
        self.assertEqual(household.slots_to_fix(self.q), [])

    def test_same_kind_twice_a_day(self):
        q = self.q
        q.plan["שלישי"]["lunch"] = dict(q.plan["שלישי"]["dinner"], main=4, reserved={}, to_buy={})  # fish + rice
        q.plan["שלישי"]["dinner"]["side"] = 10                                                     # rice again
        [slot] = household.slots_to_fix(q)
        self.assertEqual(slot["meal"], "dinner")
        self.assertEqual(household.fix_menu(q), [])
        self.assertNotEqual(q.plan["שלישי"]["dinner"]["side"], q.plan["שלישי"]["lunch"]["side"])

    def test_past_days_are_left_alone(self):
        self.q.prefs["avoid_ingredients"] = ["עגבניות"]
        self.q.today = date(2026, 10, 29)
        self.assertEqual(household.slots_to_fix(self.q), [])


class ChangeSummaryTests(unittest.TestCase):
    """The answer after a change request must match what really happened to the plan."""

    def setUp(self):
        import main
        self.main = main
        self.path = pantry_file(EGGS_AND_RICE)
        p = planner_from(self.path, max_missing_per_dish=2)
        p.add_meal_to_plan("שלישי", "lunch", 1, 10)
        self.q = reopen(p, self.path)
        self.before = {(d, m): dict(e) for d, meals in self.q.plan.items() for m, e in meals.items()}

    def test_no_change_hides_a_summary_that_claims_one(self):
        text = self.main.change_summary(self.q, self.before, "שיניתי את הקציצות לקציצות")
        self.assertNotIn("שיניתי", text)

    def test_reported_reason_is_shown(self):
        self.assertTrue(self.q.report_no_change("אין אפונה באף ארוחה")["ok"])
        text = self.main.change_summary(self.q, self.before, "שיניתי משהו")
        self.assertEqual(text, "✅ אין צורך בשינוי: אין אפונה באף ארוחה")

    def test_same_meal_put_back_is_not_a_change(self):
        self.q.declare_change_targets([{"day": "שלישי", "meal": "lunch"}])
        self.q.remove_meal("שלישי", "lunch")
        self.q.add_meal_to_plan("שלישי", "lunch", 1, 10)
        self.assertEqual(self.main.changed_slots(self.q, self.before), [])

    def test_real_change_keeps_the_summary(self):
        self.q.declare_change_targets([{"day": "שלישי", "meal": "lunch"}])
        self.q.remove_meal("שלישי", "lunch")
        self.q.add_meal_to_plan("שלישי", "lunch", 4, 10)
        self.assertEqual(self.main.change_summary(self.q, self.before, "החלפתי לדג"), "החלפתי לדג")


class EdgeCaseTests(unittest.TestCase):
    """One test per bug found in review, so none of them comes back."""

    def setUp(self):
        self.path = pantry_file(EGGS_AND_RICE)

    def test_fixing_a_wish_never_breaks_a_fixed_rule(self):
        # Every wish kind on every slot, with and without forbidden ingredients: whatever
        # the fixer does, it never adds a broken fixed rule (kosher across the day, repeats).
        wishes = [("allowed", ["meat"]), ("allowed", ["dairy"]), ("allowed", ["parve"]),
                  ("must_have", ["דג"]), ("must_have", ["ביצים"]), ("must_not_have", ["בשר"])]
        for avoid in ([], ["בשר"], ["גבינה", "דג"]):
            for day in ("שני", "שלישי"):
                for meal in ("lunch", "dinner"):
                    for key, words in wishes:
                        p = planner_from(self.path, max_missing_per_dish=2, max_uses_per_side_in_week=5)
                        for d, m, main, side in [("שני", "lunch", 1, 10), ("שני", "dinner", 3, 12),
                                                 ("שלישי", "lunch", 4, 10), ("שלישי", "dinner", 2, 12)]:
                            self.assertTrue(p.add_meal_to_plan(d, m, main, side)["ok"])
                        rule = {"note": "x", "days": [day], "meals": [meal], key: words}
                        p.prefs.update(avoid_ingredients=avoid, family_notes=["x"], slot_rules=[rule])
                        before = household._hard_problems(p)
                        household.fix_menu(p)
                        with self.subTest(avoid=avoid, day=day, meal=meal, wish=(key, words)):
                            self.assertLessEqual(household._hard_problems(p), before)

    def test_stock_that_expires_before_the_meal_is_not_counted(self):
        path = pantry_file([{"name": "גבינה לבנה", "quantity": 500, "unit": "גרם", "expires": "2026-10-26"},
                            *EGGS_AND_RICE])
        p = planner_from(path, max_missing_per_dish=2)  # today is Sunday 25/10
        self.assertTrue(p.add_meal_to_plan("רביעי", "dinner", 2, 12)["ok"])  # cheese latkes on Wednesday
        self.assertIn("גבינה לבנה", p.plan["רביעי"]["dinner"]["to_buy"])     # the cheese is gone by then

    def test_buying_again_after_the_old_stock_expired(self):
        path = pantry_file([{"name": "ביצים", "quantity": 6, "unit": "יחידה", "expires": "2026-10-26"},
                            {"name": "אורז", "quantity": 1000, "unit": "גרם", "expires": None}])
        p = planner_from(path, max_missing_per_dish=2)
        self.assertTrue(p.add_meal_to_plan("ראשון", "dinner", 3, 10)["ok"])  # shakshuka takes 3 eggs
        q = reopen(p, path, today=date(2026, 10, 27))                        # Tuesday: the eggs expired
        household.add_to_pantry([{"name": "ביצים", "quantity": 10}], UNITS, q.today, path,
                                used=household.reserved_in_plan(q.plan))
        q = reopen(q, path, today=date(2026, 10, 27))
        self.assertEqual(q.available["ביצים"]["quantity"], 10)
        left = {i["name"]: i["quantity"] for i in household.pantry_after_week(household.read_pantry(path), q.plan)}
        self.assertEqual(left["ביצים"], 10)

    def test_guests_that_dont_fit_keep_the_meal(self):
        p = planner_from(self.path, max_missing_per_dish=0, max_missing_per_dish_with_guests=0)
        p.relax = 1  # placed by the fixer with extra shopping, like a wish slot
        self.assertTrue(p.add_meal_to_plan("שני", "lunch", 1, 10)["ok"])
        p.relax = 0
        q = reopen(p, self.path)
        before = dict(q.plan["שני"]["lunch"])
        q.change_targets = {("שני", "lunch")}
        self.assertIn("error", q.change_people("שני", "lunch", 4))
        self.assertEqual(q.plan["שני"]["lunch"], before)  # still there, as it was

    def test_guests_get_other_dishes_when_these_dont_fit(self):
        p = planner_from(self.path, max_missing_per_dish=2, max_missing_per_dish_with_guests=2)
        self.assertTrue(p.add_meal_to_plan("שני", "dinner", 3, 10)["ok"])  # shakshuka: tomatoes to buy
        q = reopen(p, self.path)
        q.prefs["max_missing_per_dish_with_guests"] = 1
        result = household.set_people(q, "שני", "dinner", 6)
        if result.get("ok"):
            self.assertEqual(q.plan["שני"]["dinner"]["people"], 6)
        else:
            self.assertEqual(q.plan["שני"]["dinner"]["people"], 2)  # nothing fits: unchanged

    def test_replace_after_the_ai_took_the_meal_out(self):
        p = planner_from(self.path, max_missing_per_dish=2)
        self.assertTrue(p.add_meal_to_plan("שני", "dinner", 3, 10)["ok"])
        q = reopen(p, self.path)
        q.change_targets = {("שני", "dinner")}
        q.remove_meal("שני", "dinner")  # the AI did this, then stopped
        if household.replace_meal(q, "שני", "dinner", not_main=3):
            self.assertNotEqual(q.plan["שני"]["dinner"]["main"], 3)

    def test_a_day_taken_out_of_the_settings(self):
        p = planner_from(self.path, max_missing_per_dish=2)
        self.assertTrue(p.add_meal_to_plan("שלישי", "lunch", 1, 10)["ok"])
        self.assertTrue(p.add_meal_to_plan("שלישי", "dinner", 3, 12)["ok"])
        p.prefs["plan_days"] = ["ראשון", "שני"]
        self.assertEqual(household.slots_to_fix(p), [])  # no crash, and nothing to fix

    def test_dish_the_user_named_beats_the_variety_limit(self):
        p = planner_from(self.path, max_missing_per_dish=2, max_uses_per_side_in_week=1)
        self.assertTrue(p.add_meal_to_plan("שני", "lunch", 1, 12)["ok"])   # tomato salad, its one use
        self.assertTrue(p.add_meal_to_plan("שלישי", "dinner", 3, 10)["ok"])
        q = reopen(p, self.path)
        q.change_targets = {("שלישי", "dinner")}
        self.assertIn("user_asked", q.change_side("שלישי", "dinner", 12)["hint"])  # over the limit
        self.assertEqual(q.plan["שלישי"]["dinner"]["side"], 10)                    # unchanged
        self.assertTrue(q.change_side("שלישי", "dinner", 12, user_asked=True)["ok"])
        self.assertEqual((q.plan["שלישי"]["dinner"]["main"], q.plan["שלישי"]["dinner"]["side"]), (3, 12))
        self.assertTrue(q.plan["שלישי"]["dinner"]["chosen"])
        # The program doesn't undo it for variety later.
        self.assertEqual(household.slots_to_fix(q), [])
        self.assertEqual(household.fix_menu(q), [])
        self.assertEqual(q.plan["שלישי"]["dinner"]["side"], 12)

    def test_user_choice_still_kosher(self):
        p = planner_from(self.path, max_missing_per_dish=2)
        self.assertTrue(p.add_meal_to_plan("שני", "lunch", 1, 10)["ok"])  # meatballs: meat
        q = reopen(p, self.path)
        q.change_targets = {("שני", "lunch")}
        self.assertIn("not kosher", q.change_side("שני", "lunch", 11, user_asked=True)["error"])  # dairy salad

    def test_search_by_name(self):
        p = planner_from(self.path)
        self.assertEqual({r["id"] for r in p.search_recipes(name="סלט")["recipes"]}, {11, 12})

    def test_forbidden_word_in_singular(self):
        p = planner_from(self.path, avoid_ingredients=["ביצה"])
        self.assertEqual(p.avoided_ingredients(p.recipes[3]), ["ביצים"])  # shakshuka is out


if __name__ == "__main__":
    unittest.main()
