"""Tests for the browser app's actions (web.py), on made-up data in a temporary folder,
with a fake AI. No API key, internet or cost, and the real data files are never touched.
Run:  py -m unittest -v
"""

import contextlib
import io
import json
import threading
import unittest
import urllib.error
import urllib.request
from datetime import date
from http.server import ThreadingHTTPServer
from pathlib import Path
from unittest.mock import patch

import agent
import household
import tools
import web
from tests.test_agent import FakeClient, reply, tool_call
from tests.test_tools import RECIPES, recipe
from tests import temp_dir

MONDAY = date(2026, 10, 26)
# The tools' test recipes, plus a few more, so a meal usually has somewhere to go.
WEB_RECIPES = {**RECIPES, "recipes": RECIPES["recipes"] + [
    recipe(5, "חביתה", "main", "parve", [("ביצים", 2)]),
    recipe(6, "בשר בתנור", "main", "meat", [("בשר טחון", 400)]),
    recipe(13, "סלט ביצים", "side", "parve", [("ביצים", 1)]),
]}
PREFS = {"plan_days": ["ראשון", "שני", "שלישי"], "meals": ["lunch", "dinner"], "people": 2,
         "max_missing_per_dish": 2, "max_missing_per_dish_with_guests": 3}
PANTRY = [{"name": "ביצים", "quantity": 6, "unit": "יחידה", "expires": None},
          {"name": "עגבניות", "quantity": 5, "unit": "יחידה", "expires": None},
          {"name": "אורז", "quantity": 1000, "unit": "גרם", "expires": None},
          {"name": "גבינה לבנה", "quantity": 500, "unit": "גרם", "expires": None},
          {"name": "פילה דג", "quantity": 600, "unit": "גרם", "expires": None}]

# What the fake AI does when asked to plan: four meals, then it says it's done.
PLAN = [("שני", "lunch", 1, 10), ("שני", "dinner", 3, 12), ("שלישי", "lunch", 4, 10), ("שלישי", "dinner", 2, 12)]


def add(day, meal, main_id, side_id):
    return reply(calls=[tool_call("add_meal_to_plan", {"day": day, "meal": meal, "main_id": main_id, "side_id": side_id})])


class WebTestCase(unittest.TestCase):
    def setUp(self):
        root = temp_dir()
        (root / "data").mkdir()
        (root / "output").mkdir()
        for name, content in [("recipes", WEB_RECIPES), ("preferences", PREFS), ("pantry", {"items": PANTRY})]:
            (root / "data" / f"{name}.json").write_text(json.dumps(content, ensure_ascii=False), encoding="utf-8")
        self.root = root
        for p in [patch.object(tools, "DATA_DIR", root / "data"),
                  patch.object(tools, "PLAN_FILE", root / "output" / "plan.json"),
                  patch.object(web, "TODAY", MONDAY),
                  # Never the real AI in tests: a fake one that answers nothing useful, unless a test sets replies.
                  patch.object(agent, "make_client", lambda: FakeClient([]))]:
            p.start()
            self.addCleanup(p.stop)
        web.PENDING_WISHES.clear()

    def ai(self, replies):
        """The fake AI's replies for the next AI run."""
        return patch.object(agent, "make_client", lambda: FakeClient(replies))

    def plan(self):
        with self.ai([add(*meal) for meal in PLAN]):
            return web.do_plan({})

    def kind(self, day, meal):
        """meat / dairy / parve of a saved meal."""
        entry = tools.load_json(tools.PLAN_FILE)["plan"][day][meal]
        kinds = {r["id"]: r["kosher"] for r in WEB_RECIPES["recipes"]}
        found = {kinds[entry["main"]], kinds[entry["side"]]}
        return "meat" if "meat" in found else "dairy" if "dairy" in found else "parve"

    def menu(self):
        """{(day, meal): main dish id} of the saved plan."""
        plan = tools.load_json(tools.PLAN_FILE)["plan"]
        return {(d, m): e["main"] for d, meals in plan.items() for m, e in meals.items()}

    def wish(self, notes, rules):
        """Save wishes; the fake AI understands each one as the rule given for it."""
        def understand(note, days, client=None):
            rule = rules.get(note)
            return ({"note": note, **rule} if rule else None), note
        with patch.object(agent, "interpret_wish", understand):
            return web.do_family_notes({"items": notes})


class PlanningTests(WebTestCase):
    def test_no_plan_yet(self):
        self.assertEqual(web.state()["mode"], "no_plan")

    def test_plan_the_week(self):
        answer = self.plan()
        self.assertIn("סיימתי", answer["message"])
        state = web.state()
        self.assertEqual(state["mode"], "week")
        self.assertEqual(self.menu(), {(d, m): main for d, m, main, _ in PLAN})
        self.assertTrue((self.root / "output" / "menu.txt").exists())

    def test_a_wish_the_ai_missed_is_still_met(self):
        self.wish(["בשני בערב דג"], {"בשני בערב דג": {"days": ["שני"], "meals": ["dinner"], "must_have": ["דג"]}})
        with self.ai([add("שני", "lunch", 1, 10)]):  # the AI plans one meal and stops
            web.do_plan({})
        self.assertEqual(self.menu()[("שני", "dinner")], 4)  # the program filled the wish slot


class DuringTheWeekTests(WebTestCase):
    def setUp(self):
        super().setUp()
        self.plan()

    def test_recipe_is_scaled_to_the_meal(self):
        def recipe_of(day, meal):
            [m] = [m for d in web.state()["days"] if d["name"] == day for m in d["meals"] if m["meal"] == meal]
            return m["recipes"][0]
        shakshuka = recipe_of("שני", "dinner")
        self.assertEqual(shakshuka["name"], "שקשוקה")
        self.assertIn({"text": "ביצים 3 יח'", "removed": False}, shakshuka["ingredients"])  # 6 eggs for 4, so 3 for 2
        self.assertTrue(shakshuka["steps"])
        web.do_guests({"day": "שני", "meal": "dinner", "people": 4})
        self.assertIn({"text": "ביצים 6 יח'", "removed": False}, recipe_of("שני", "dinner")["ingredients"])

    def test_guests(self):
        answer = web.do_guests({"day": "שני", "meal": "dinner", "people": 4})
        self.assertIn("4 אנשים", answer["message"])
        self.assertEqual(tools.load_json(tools.PLAN_FILE)["plan"]["שני"]["dinner"]["people"], 4)

    def test_not_cooked(self):
        web.do_skip({"day": "שני", "meal": "lunch"})
        self.assertNotIn(("שני", "lunch"), self.menu())
        with self.assertRaises(tools.DataError):
            web.do_skip({"day": "שני", "meal": "lunch"})

    def test_bought_goes_to_the_pantry(self):
        shopping = web.state()["shopping"]
        self.assertTrue(shopping)
        web.do_bought({})
        self.assertEqual(web.state()["shopping"], [])
        names = {i["name"] for i in household.read_pantry()}
        self.assertTrue({s["name"] for s in shopping} <= names)

    def test_bought_a_different_amount(self):
        item = max(web.state()["shopping"], key=lambda s: s["quantity"])  # e.g. meat in grams
        self.assertGreater(item["quantity"], 1)
        half = item["quantity"] // 2
        answer = web.do_bought({"items": [{"name": item["name"], "quantity": half}]})
        self.assertIn("עוד צריך לקנות", answer["message"])
        left = {s["name"]: s["quantity"] for s in web.state()["shopping"]}
        self.assertEqual(left[item["name"]], item["quantity"] - half)
        answer = web.do_bought({"items": [{"name": item["name"], "quantity": item["quantity"] * 3}]})
        self.assertIn("נשארים פנויים במלאי", answer["message"])
        self.assertNotIn(item["name"], {s["name"] for s in web.state()["shopping"]})
        for bad in (0, -1, "2", True):
            with self.subTest(quantity=bad), self.assertRaises(tools.DataError):
                web.do_bought({"items": [{"name": item["name"], "quantity": bad}]})

    def test_bad_pantry_is_refused_and_kept(self):
        before = household.read_pantry()
        with self.assertRaises(tools.DataError):
            web.do_pantry({"items": [{"name": "ביצים", "quantity": "הרבה"}]})
        self.assertEqual(household.read_pantry(), before)

    def test_change_that_isnt_needed(self):
        with self.ai([reply(calls=[tool_call("report_no_change", {"reason": "אין אפונה באף ארוחה"})]),
                      reply("שיניתי את הכול")]):  # a summary that claims a change
            answer = web.do_change({"request": "יש אלרגיה לאפונה"})
        self.assertIn("אין צורך בשינוי: אין אפונה באף ארוחה", answer["message"])
        self.assertNotIn("שיניתי את הכול", answer["message"])


class EdgeCaseTests(WebTestCase):
    """Things found in review, through the app's own actions."""

    def setUp(self):
        super().setUp()
        self.plan()

    def test_guest_messages_are_clear(self):
        for people in (0, -2, 31, "4", True):
            with self.subTest(people=people), self.assertRaises(tools.DataError) as e:
                web.do_guests({"day": "שני", "meal": "dinner", "people": people})
            self.assertIn("בין 1 ל-30", str(e.exception))
        with self.assertRaises(tools.DataError) as e:
            web.do_guests({"day": "ראשון", "meal": "lunch", "people": 4})  # Sunday is over
        self.assertIn("שעוד לא עבר", str(e.exception))
        self.assertIn("כבר מתוכננת ל-2", web.do_guests({"day": "שני", "meal": "dinner", "people": 2})["message"])

    def test_less_in_the_pantry_says_the_list_changed(self):
        answer = web.do_pantry({"items": [i for i in PANTRY if i["name"] != "ביצים"]})
        self.assertIn("רשימת הקניות עודכנה", answer["message"])

    def test_replace_after_the_ai_took_the_meal_out_and_stopped(self):
        with self.ai([reply(calls=[tool_call("remove_meal", {"day": "שני", "meal": "dinner"})]), reply("זהו")]):
            answer = web.do_replace({"day": "שני", "meal": "dinner"})
        self.assertNotEqual(self.menu().get(("שני", "dinner")), 3)  # not the shakshuka it had
        self.assertIn(("שני", "dinner"), self.menu())
        self.assertNotIn("לא בוצע שום שינוי", answer["message"])

    def test_a_day_taken_out_of_the_settings(self):
        web.do_settings({"people": 2, "plan_days": ["ראשון", "שני"], "meals": ["lunch", "dinner"]})
        self.assertEqual([d["name"] for d in web.state()["days"]], ["ראשון", "שני"])  # no crash

    def test_more_people_at_home(self):
        answer = web.do_settings({"people": 4, "plan_days": PREFS["plan_days"], "meals": ["lunch", "dinner"]})
        self.assertIn("עודכנו ל-4", answer["message"])
        plan = tools.load_json(tools.PLAN_FILE)["plan"]
        self.assertEqual({e["people"] for meals in plan.values() for e in meals.values() if meals is not plan.get("ראשון")}, {4})
        self.assertEqual(tools.load_preferences()["people"], 4)

    def test_bad_settings(self):
        for body in ({"people": 0, "plan_days": ["שני"], "meals": ["lunch"]},
                     {"people": 2, "plan_days": [], "meals": ["lunch"]},
                     {"people": 2, "plan_days": ["שבת"], "meals": ["lunch"]}):
            with self.subTest(body=body), self.assertRaises(tools.DataError):
                web.do_settings(body)
        self.assertEqual(tools.load_preferences()["plan_days"], PREFS["plan_days"])  # unchanged

    def test_new_recipe_from_the_ai_is_checked(self):
        def written(answer):
            client = FakeClient([reply(json.dumps(answer, ensure_ascii=False))])
            return agent.create_recipe("משהו", "side", {**WEB_RECIPES, "staples": ["מלח", "שמן"]}, client=client)
        good = {"name": "אורז עם ביצה", "ingredients": [{"name": "אורז", "quantity": 300}, {"name": "ביצים", "quantity": 2}],
                "steps": ["מבשלים"], "prep_minutes": 25}
        recipe = written(good)
        self.assertEqual((recipe["kosher"], recipe["dish_type"], recipe["servings"]), ("parve", "side", 4))
        self.assertIsNone(written({**good, "ingredients": [{"name": "שוקולד", "quantity": 1}]}))  # unknown
        self.assertIsNone(written({**good, "ingredients": [{"name": "בשר טחון", "quantity": 1},
                                                           {"name": "גבינה לבנה", "quantity": 1}]}))  # not kosher
        self.assertEqual(written({**good, "ingredients": [{"name": "בשר טחון", "quantity": 500}]})["kosher"], "meat")
        self.assertIsNone(written({"name": None}))
        # Basics are dropped (always at home); a new ingredient with a known unit is kept.
        with_new = written({**good, "ingredients": [{"name": "שמן", "quantity": 20},
                                                    {"name": "חצילים", "quantity": 2, "unit": "יחידה"}]})
        self.assertEqual([i["name"] for i in with_new["ingredients"]], ["חצילים"])
        self.assertEqual(with_new["new_units"], {"חצילים": "יחידה"})
        self.assertIsNone(written({**good, "ingredients": [{"name": "חצילים", "quantity": 2, "unit": "קופסה"}]}))

    def test_new_ingredient_joins_the_catalog_with_the_pantry_unit(self):
        household.write_pantry([*PANTRY, {"name": "חצילים", "quantity": 3, "unit": "יחידה"}],
                               WEB_RECIPES["ingredient_units"], MONDAY)
        recipe = {"name": "סלט חצילים", "dish_type": "side", "kosher": "parve", "contains_fish": False,
                  "meals": ["lunch", "dinner"], "prep_minutes": 30, "servings": 4,
                  "ingredients": [{"name": "חצילים", "quantity": 2}], "steps": ["קולים את החצילים"],
                  "new_units": {"חצילים": "גרם"}}  # the AI's unit differs from the pantry's
        household.add_recipe(recipe)
        self.assertEqual(tools.load_recipes()["ingredient_units"]["חצילים"], "יחידה")
        tools.load_pantry(tools.load_recipes()["ingredient_units"])  # and the pantry still loads

    def test_wish_answer_that_isnt_an_object(self):
        for content in ('["x"]', '"x"', "null"):
            with self.subTest(content=content):
                rule, understood = agent.interpret_wish("משהו", PREFS["plan_days"], client=FakeClient([reply(content)]))
                self.assertIsNone(rule)


class NamedDishTests(WebTestCase):
    """'Add mashed potatoes on Thursday': the AI only says what was asked; the program does it."""

    def setUp(self):
        super().setUp()
        self.plan()

    def ask(self, intent):
        with patch.object(agent, "interpret_request", lambda *a, **k: intent):
            return web.do_change({"request": "..."})

    def test_side_by_name(self):
        # (Over the week's limit is in test_household: test_dish_the_user_named_beats_the_variety_limit.)
        answer = self.ask({"kind": "dish", "dish": "סלט ביצים", "day": "שלישי", "meal": "lunch"})
        plan = tools.load_json(tools.PLAN_FILE)["plan"]
        self.assertEqual(plan["שלישי"]["lunch"]["side"], 13)
        self.assertEqual(plan["שלישי"]["lunch"]["main"], 4)  # the main stays
        self.assertTrue(plan["שלישי"]["lunch"]["chosen"])
        self.assertIn("כמו שביקשת", answer["message"])
        self.assertEqual(web.state()["to_fix"], [])           # and nothing undoes it

    def test_named_main_moves_its_other_place(self):
        # Fish (4) is Tuesday lunch; the user wants fish on Monday dinner: Tuesday gets something else.
        answer = self.ask({"kind": "dish", "dish": "דג בתנור", "day": "שני", "meal": "dinner"})
        menu = self.menu()
        self.assertEqual(menu[("שני", "dinner")], 4)
        self.assertNotEqual(menu[("שלישי", "lunch")], 4)
        self.assertIn("התאמתי את התפריט", answer["message"])

    def test_instead_of_the_side_keeps_the_main(self):
        # Monday lunch: meatballs (1) + rice (10). "An omelette instead of the rice": the
        # omelette (a main dish in the book) goes in the rice's place; the meatballs stay.
        answer = self.ask({"kind": "dish", "dish": "חביתה", "day": "שני", "meal": None, "instead_of": "אורז"})
        entry = tools.load_json(tools.PLAN_FILE)["plan"]["שני"]["lunch"]
        self.assertEqual((entry["main"], entry["side"]), (1, 5))
        self.assertNotIn("התוספת עכשיו", answer["message"])  # the main didn't change: no question

    def test_instead_of_the_main_keeps_the_side(self):
        self.ask({"kind": "dish", "dish": "בשר בתנור", "day": "שני", "meal": None, "instead_of": "קציצות"})
        entry = tools.load_json(tools.PLAN_FILE)["plan"]["שני"]["lunch"]
        self.assertEqual((entry["main"], entry["side"]), (6, 10))

    def test_a_name_that_fits_several_dishes_asks_which(self):
        answer = self.ask({"kind": "dish", "dish": "סלט", "day": "שלישי", "meal": "lunch", "instead_of": "אורז"})
        self.assertIn("כמה מנות", answer["message"])
        labels = [c["label"] for c in answer["choices"]]
        self.assertIn("סלט ביצים", labels)
        self.assertNotIn("סלט עגבניות", labels)  # it's Tuesday dinner's side: not twice a day
        web.do_put_dish(next(c for c in answer["choices"] if c["label"] == "סלט ביצים")["body"])
        self.assertEqual(tools.load_json(tools.PLAN_FILE)["plan"]["שלישי"]["lunch"]["side"], 13)

    def test_same_side_twice_a_day_says_why(self):
        with self.assertRaises(tools.DataError) as e:  # Tuesday dinner already has the tomato salad
            self.ask({"kind": "dish", "dish": "סלט עגבניות", "day": "שלישי", "meal": "lunch"})
        self.assertIn("פעמיים באותו יום", str(e.exception))

    def test_no_meal_named_tries_the_day(self):
        answer = self.ask({"kind": "dish", "dish": "חביתה", "day": "שלישי", "meal": None})
        self.assertIn(5, self.menu().values())
        self.assertIn("חביתה", answer["message"])

    def test_dish_that_cant_go_there_says_why(self):
        # Monday lunch is meatballs (meat): a dairy salad can't join it.
        with self.assertRaises(tools.DataError) as e:
            self.ask({"kind": "dish", "dish": "סלט חלבי", "day": "שני", "meal": "lunch"})
        self.assertIn("כשר", str(e.exception))

    def test_unknown_dish_offers_a_new_recipe_or_others_that_fit(self):
        answer = self.ask({"kind": "dish", "dish": "סלט פירות", "day": "שני", "meal": "dinner", "dish_type": "side"})
        self.assertIn("מנה אחרת שמתאימה לערב ביום שני", answer["message"])
        self.assertNotIn("דומה", answer["message"])
        labels = [c["label"] for c in answer["choices"]]
        self.assertEqual(labels[0], "ליצור מתכון חדש: סלט פירות")
        self.assertNotIn("סלט עגבניות", labels)  # already the side of that meal
        self.assertNotIn("סלט חלבי", labels)     # a dairy dinner after Monday's meat lunch isn't allowed
        # The user picks one of them:
        pick = next(c for c in answer["choices"] if c["label"] == "סלט ביצים")
        web.do_put_dish(pick["body"])
        self.assertEqual(tools.load_json(tools.PLAN_FILE)["plan"]["שני"]["dinner"]["side"], 13)

    def test_only_dishes_that_really_fit_are_offered(self):
        # Monday lunch is meatballs (meat): a dairy side can't be offered for it.
        answer = self.ask({"kind": "dish", "dish": "סלט פירות", "day": "שני", "meal": "lunch", "dish_type": "side"})
        labels = [c["label"] for c in answer["choices"]]
        self.assertNotIn("סלט חלבי", labels)
        self.assertTrue(all(r in [x["name"] for x in WEB_RECIPES["recipes"] if x["dish_type"] == "side"]
                            for r in labels[1:]))  # sides only

    def test_new_main_asks_about_the_side(self):
        # Monday dinner: shakshuka + tomato salad. The user wants an omelette instead.
        answer = self.ask({"kind": "dish", "dish": "חביתה", "day": "שני", "meal": "dinner"})
        self.assertIn("התוספת עכשיו היא סלט עגבניות", answer["message"])
        labels = [c["label"] for c in answer["choices"]]
        self.assertEqual(labels[0], "להשאיר את סלט עגבניות")
        self.assertTrue(answer["choices"][0]["dismiss"])
        self.assertGreater(len(labels), 1)                      # and other sides that fit
        web.do_put_dish(answer["choices"][1]["body"])           # the user switches the side
        entry = tools.load_json(tools.PLAN_FILE)["plan"]["שני"]["dinner"]
        self.assertEqual(entry["main"], 5)                      # the omelette stays
        self.assertNotEqual(entry["side"], 12)

    def test_new_recipe_is_written_checked_and_used(self):
        steamed = {"name": "ירקות מאודים", "dish_type": "side", "kosher": "parve", "contains_fish": False,
                   "meals": ["lunch", "dinner"], "prep_minutes": 20, "servings": 4,
                   "ingredients": [{"name": "עגבניות", "quantity": 4}], "steps": ["מאדים 15 דקות"]}
        with patch.object(agent, "create_recipe", lambda *a, **k: dict(steamed)):
            answer = web.do_put_dish({"dish": "ירקות מאודים", "day": "שלישי", "meal": "lunch", "create": True})
        self.assertIn("הוספתי למתכונים", answer["message"])
        recipes = tools.load_recipes()["recipes"]
        new = next(r for r in recipes if r["name"] == "ירקות מאודים")
        self.assertEqual(tools.load_json(tools.PLAN_FILE)["plan"]["שלישי"]["lunch"]["side"], new["id"])

    def test_dish_to_a_dish_on_the_menu(self):
        # "Add the egg salad to the shakshuka": the meal that has the shakshuka (Monday dinner).
        self.ask({"kind": "dish", "dish": "סלט ביצים", "day": None, "meal": None, "with_dish": "שקשוקה"})
        self.assertEqual(tools.load_json(tools.PLAN_FILE)["plan"]["שני"]["dinner"]["side"], 13)
        with self.assertRaises(tools.DataError):
            self.ask({"kind": "dish", "dish": "סלט ביצים", "day": None, "meal": None, "with_dish": "פיצה"})

    def test_guests_by_words(self):
        self.ask({"kind": "guests", "day": "שני", "meal": "dinner", "people": 5})
        self.assertEqual(tools.load_json(tools.PLAN_FILE)["plan"]["שני"]["dinner"]["people"], 5)

    def test_other_requests_go_to_the_agent(self):
        with self.ai([reply(calls=[tool_call("report_no_change", {"reason": "הכול בסדר"})])]):
            answer = self.ask({"kind": "other"})
        self.assertIn("הכול בסדר", answer["message"])


class MealNoteTests(WebTestCase):
    """'Add cumin to the shakshuka': a note on that meal, and optionally in the recipe."""

    def setUp(self):
        super().setUp()
        self.plan()

    def ask(self, intent):
        with patch.object(agent, "interpret_request", lambda *a, **k: intent):
            return web.do_change({"request": "..."})

    def meal(self, day, meal):
        return next(m for d in web.state()["days"] if d["name"] == day for m in d["meals"] if m["meal"] == meal)

    def test_note_on_the_dish_named(self):
        answer = self.ask({"kind": "note", "note": "להוסיף כמון", "dish": "שקשוקה", "day": None, "meal": None})
        self.assertIn("שקשוקה", answer["message"])
        m = self.meal("שני", "dinner")
        self.assertEqual(m["notes"], ["להוסיף כמון"])
        self.assertEqual(m["recipes"][0]["notes"], ["להוסיף כמון"])  # with the shakshuka's recipe
        self.assertEqual(m["recipes"][1]["notes"], [])                # not with the side
        # It's only this meal: the recipe book is the same.
        shakshuka = next(r for r in tools.load_recipes()["recipes"] if r["id"] == 3)
        self.assertNotIn("להוסיף כמון", shakshuka["steps"])
        # Unless the user keeps it in the recipe:
        web.do_recipe_note(answer["choices"][0]["body"])
        shakshuka = next(r for r in tools.load_recipes()["recipes"] if r["id"] == 3)
        self.assertEqual(shakshuka["steps"][-1], "להוסיף כמון")

    def test_change_is_written_into_the_recipe(self):
        rewritten = {"steps": ["מטגנים בצל", "מוסיפים עגבניות, מלח וכמון", "שוברים ביצים ומבשלים"],
                     "added": ["כמון 1 כפית"], "removed": []}
        with patch.object(agent, "adapt_recipe", lambda *a, **k: dict(rewritten)):
            answer = self.ask({"kind": "note", "note": "להוסיף כמון", "dish": "שקשוקה", "day": None, "meal": None})
        self.assertIn("המתכון של הארוחה הזאת עודכן", answer["message"])
        [shakshuka, _] = self.meal("שני", "dinner")["recipes"]
        self.assertEqual(shakshuka["steps"], rewritten["steps"])  # the change is in the steps
        self.assertTrue(shakshuka["rewritten"])
        self.assertIn({"text": "כמון 1 כפית", "added": True}, shakshuka["ingredients"])
        # Kept for next times: the recipe book gets the new steps and the added spice.
        web.do_recipe_note(answer["choices"][0]["body"])
        book = next(r for r in tools.load_recipes()["recipes"] if r["id"] == 3)
        self.assertEqual((book["steps"], book["extras"]), (rewritten["steps"], ["כמון 1 כפית"]))

    def test_rewrite_from_the_ai_is_checked(self):
        steps = ["א", "ב", "ג"]
        def adapted(answer):
            return agent.adapt_recipe("שקשוקה", ["ביצים 3 יח'", "בצל 1 יח'"], steps, "בלי בצל",
                                      client=FakeClient([reply(json.dumps(answer, ensure_ascii=False))]))
        self.assertEqual(adapted({"steps": steps, "removed": ["בצל"]})["removed"], ["בצל"])
        self.assertIsNone(adapted({"steps": ["א"]}))                                # a step got lost
        self.assertIsNone(adapted({"steps": steps, "removed": ["שוקולד"]}))         # not in the recipe
        self.assertIsNone(adapted(["not", "an", "object"]))

    def test_dish_on_the_menu_twice_asks_which(self):
        # The tomato salad is Monday and Tuesday dinner; no day was said.
        answer = self.ask({"kind": "note", "note": "להוסיף שמיר", "dish": "סלט עגבניות", "day": None, "meal": None})
        self.assertIn("פעמים", answer["message"])
        self.assertEqual([c["label"] for c in answer["choices"]], ["יום שני בערב", "יום שלישי בערב", "בכל הפעמים השבוע"])
        self.assertEqual(self.meal("שני", "dinner")["notes"], [])  # nothing yet
        web.do_meal_note(answer["choices"][-1]["body"])            # every time
        for day in ("שני", "שלישי"):
            [salad] = [r for r in self.meal(day, "dinner")["recipes"] if r["name"] == "סלט עגבניות"]
            self.assertEqual(salad["notes"], ["להוסיף שמיר"])

    def test_dish_name_wins_over_a_guessed_meal(self):
        # The shakshuka is Monday dinner; the request was understood as Monday lunch.
        self.ask({"kind": "note", "note": "להוסיף כמון", "dish": "שקשוקה", "day": "שני", "meal": "lunch"})
        self.assertEqual(self.meal("שני", "dinner")["notes"], ["להוסיף כמון"])

    def test_note_stays_when_the_side_or_people_change(self):
        self.ask({"kind": "note", "note": "פחות מלח", "dish": "שקשוקה", "day": "שני", "meal": None})
        web.do_guests({"day": "שני", "meal": "dinner", "people": 3})
        self.assertEqual(self.meal("שני", "dinner")["notes"], ["פחות מלח"])

    def test_note_needs_a_meal(self):
        with self.assertRaises(tools.DataError) as e:
            self.ask({"kind": "note", "note": "להוסיף כמון", "dish": None, "day": "שני", "meal": None})
        self.assertIn("לאיזו ארוחה", str(e.exception))
        with self.assertRaises(tools.DataError):
            self.ask({"kind": "note", "note": "להוסיף כמון", "dish": "פיצה", "day": None, "meal": None})


class TerminalTests(WebTestCase):
    def test_failed_change_doesnt_add_the_shopping_twice(self):
        import main
        self.plan()
        before = household.read_pantry()

        def no_key():
            raise agent.AgentError("לא נמצא מפתח API")
        for _ in range(2):
            with patch.object(agent, "make_client", no_key), self.assertRaises(agent.AgentError), \
                    contextlib.redirect_stdout(io.StringIO()):
                main.change_day(MONDAY, "ביום שני בערב משהו אחר")
        self.assertEqual(household.read_pantry(), before)

    def test_terminal_plan_starts_a_new_week_from_what_is_left(self):
        import main
        self.plan()
        with self.ai([reply("סיימתי")]), contextlib.redirect_stdout(io.StringIO()):
            main.plan_week(date(2026, 11, 1))
        self.assertTrue((self.root / "output" / "archive" / "plan-2026-10-25.json").exists())
        eggs = {i["name"]: i["quantity"] for i in household.read_pantry()}.get("ביצים", 0)
        self.assertLess(eggs, 6)  # what last week's meals took is gone


class ReplaceTests(WebTestCase):
    def setUp(self):
        super().setUp()
        self.plan()

    def test_ai_replaces(self):
        with self.ai([reply(calls=[tool_call("remove_meal", {"day": "שני", "meal": "dinner"})]),
                      add("שני", "dinner", 5, 13), reply("בחרתי בחביתה כי הביצים בבית")]):
            answer = web.do_replace({"day": "שני", "meal": "dinner"})
        self.assertEqual(self.menu()[("שני", "dinner")], 5)
        self.assertIn("הביצים בבית", answer["message"])

    def test_program_replaces_when_the_ai_doesnt(self):
        with self.ai([reply("לא מצאתי מנה בשם הזה")]):
            answer = web.do_replace({"day": "שני", "meal": "dinner"})
        self.assertNotEqual(self.menu()[("שני", "dinner")], 3)
        self.assertIn("בחרתי מנה מתאימה בעצמי", answer["message"])
        self.assertNotIn("לא מצאתי", answer["message"])

    def test_program_replaces_without_the_ai(self):
        def no_credit():
            raise agent.AgentError("נגמר הקרדיט")
        with patch.object(agent, "make_client", no_credit):
            answer = web.do_replace({"day": "שני", "meal": "dinner"})
        self.assertNotEqual(self.menu()[("שני", "dinner")], 3)
        self.assertIn("נגמר הקרדיט", answer["message"])

    def test_day_that_is_over(self):
        with self.assertRaises(tools.DataError):
            web.do_replace({"day": "ראשון", "meal": "lunch"})


class WishTests(WebTestCase):
    FISH = "בשלישי בצהריים דג"
    MEAT = "בשלישי בצהריים בשרי"
    RULES = {FISH: {"days": ["שלישי"], "meals": ["lunch"], "must_have": ["דג"]},
             MEAT: {"days": ["שלישי"], "meals": ["lunch"], "allowed": ["meat"]}}

    def setUp(self):
        super().setUp()
        self.plan()

    def test_menu_follows_a_new_wish(self):
        answer = self.wish([self.MEAT], self.RULES)
        self.assertEqual(self.kind("שלישי", "lunch"), "meat")
        self.assertNotEqual(self.kind("שלישי", "dinner"), "dairy")  # no dairy after meat
        self.assertIn("התאמתי את התפריט", answer["message"])
        self.assertEqual(web.state()["to_fix"], [])

    def test_contradicting_wish_asks_the_user(self):
        self.wish([self.FISH], self.RULES)
        answer = self.wish([self.FISH, self.MEAT], self.RULES)
        self.assertEqual(answer["conflict"], {"new": self.MEAT, "old": [self.FISH]})
        self.assertEqual(tools.load_preferences()["family_notes"], [self.FISH])  # nothing saved yet
        # The user chooses the new one: it's saved (without asking the AI again) and the menu follows.
        with patch.object(agent, "interpret_wish", side_effect=AssertionError("asked the AI again")):
            web.do_family_notes({"items": [self.MEAT]})
        self.assertEqual(tools.load_preferences()["family_notes"], [self.MEAT])
        self.assertEqual(self.kind("שלישי", "lunch"), "meat")

    def test_removing_a_wish_changes_nothing(self):
        self.wish([self.MEAT], self.RULES)
        before = self.menu()
        answer = web.do_family_notes({"items": []})
        self.assertEqual(self.menu(), before)
        self.assertIn("לא היה צריך לשנות", answer["message"])

    def test_wish_saved_even_without_the_ai(self):
        def no_credit(note, days, client=None):
            raise agent.AgentError("נגמר הקרדיט")
        with patch.object(agent, "interpret_wish", no_credit):
            answer = web.do_family_notes({"items": ["פחות פטריות"]})
        self.assertIn("נגמר הקרדיט", answer["message"])
        self.assertEqual(tools.load_preferences()["family_notes"], ["פחות פטריות"])

    def test_forbidden_ingredient_is_replaced(self):
        web.do_avoid({"items": ["בשר"]})
        self.assertNotEqual(self.menu()[("שני", "lunch")], 1)  # the meatballs are gone


class NewWeekTests(WebTestCase):
    def test_next_week_starts_from_what_is_left(self):
        self.plan()
        with patch.object(web, "TODAY", date(2026, 11, 1)):
            state = web.state()
            self.assertEqual(state["mode"], "new_week")
            left = {i["name"]: i["quantity"] for i in state["proposed"]}
            self.assertLess(left["ביצים"], 6)  # what the meals took is gone
            web.do_new_week({"items": state["proposed"]})
            self.assertEqual(web.state()["mode"], "no_plan")
        self.assertTrue((self.root / "output" / "archive" / "plan-2026-10-25.json").exists())


class ServerTests(WebTestCase):
    """The real HTTP server, on a free port: answers, and errors as messages."""

    def setUp(self):
        super().setUp()
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), web.Handler)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.addCleanup(self.server.server_close)
        self.addCleanup(self.server.shutdown)
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}"

    def request(self, path, body=None):
        data = body if isinstance(body, bytes) or body is None else json.dumps(body).encode()
        req = urllib.request.Request(self.url + path, data=data, headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req) as r:
                return r.status, json.loads(r.read())
        except urllib.error.HTTPError as e:
            with e:
                return e.code, json.loads(e.read())

    def test_page_and_state(self):
        with urllib.request.urlopen(self.url + "/") as r:
            self.assertIn("מתכנן הארוחות", r.read().decode("utf-8"))
        status, body = self.request("/api/state")
        self.assertEqual((status, body["state"]["mode"]), (200, "no_plan"))

    def test_errors_come_back_as_messages(self):
        self.assertEqual(self.request("/api/skip", b"not json")[0], 400)
        self.assertEqual(self.request("/api/nothing", {})[0], 404)
        status, body = self.request("/api/change", {"request": ""})
        self.assertEqual(status, 400)
        self.assertIn("צריך לכתוב", body["error"])


if __name__ == "__main__":
    unittest.main()
