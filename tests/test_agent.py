"""Tests for the agent loop, with a fake AI that follows a script.

No API key or internet needed, and no cost: the fake client returns the
replies we wrote, so we can check how the loop reacts to each situation.
"""

import json
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import agent
from tests.test_tools import make_planner
from tests import temp_dir


def tool_call(name, args, call_id="c1"):
    fn = SimpleNamespace(name=name, arguments=json.dumps(args, ensure_ascii=False))
    call = SimpleNamespace(id=call_id, type="function", function=fn)
    call.model_dump = lambda: {"id": call_id, "type": "function",
                               "function": {"name": fn.name, "arguments": fn.arguments}}
    return call


def reply(text=None, calls=None, input_tokens=10, output_tokens=5):
    message = SimpleNamespace(content=text, tool_calls=calls)
    usage = SimpleNamespace(prompt_tokens=input_tokens, completion_tokens=output_tokens)
    return SimpleNamespace(choices=[SimpleNamespace(message=message)], usage=usage)


class FakeClient:
    """Plays back scripted replies and remembers what the loop sent."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.sent = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    def create(self, **kwargs):
        self.sent.append(kwargs["messages"][:])
        return self.replies.pop(0) if self.replies else reply("סיימתי")


def run(planner, replies, **kwargs):
    client = FakeClient(replies)
    result = agent.run_agent(planner, "goal", client=client, log=lambda _: None, **kwargs)
    return result, client


def tool_results(client):
    """The tool results the loop sent back to the AI, in order."""
    return [json.loads(m["content"]) for m in client.sent[-1] if m["role"] == "tool"]


class AgentLoopTests(unittest.TestCase):
    def test_tool_result_goes_back_to_the_ai(self):
        p = make_planner()
        result, client = run(p, [
            reply(calls=[tool_call("add_meal_to_plan", {"day": "ראשון", "meal": "lunch", "main_id": 1, "side_id": 10})]),
            reply("תכננתי"),
        ])
        self.assertTrue(result["finished"])
        self.assertEqual(result["summary"], "תכננתי")
        self.assertEqual(p.plan["ראשון"]["lunch"]["main"], 1)     # the tool really ran
        self.assertTrue(tool_results(client)[0]["ok"])            # and the AI saw its result

    def test_bad_tool_calls_dont_crash(self):
        bad_json = tool_call("get_pantry", {})
        bad_json.function.arguments = "{not json"
        result, client = run(make_planner(), [
            reply(calls=[tool_call("delete_everything", {})]),
            reply(calls=[bad_json]),
            reply(calls=[tool_call("check_missing", {"wrong_name": 1})]),
            reply("done"),
        ])
        errors = [r["error"] for r in tool_results(client)]
        self.assertIn("unknown tool", errors[0])
        self.assertIn("not valid JSON", errors[1])
        self.assertIn("bad arguments", errors[2])
        self.assertTrue(result["finished"])

    def test_repeated_failed_call_is_blocked(self):
        not_kosher = {"day": "ראשון", "meal": "lunch", "main_id": 1, "side_id": 11}
        _, client = run(make_planner(), [
            reply(calls=[tool_call("add_meal_to_plan", not_kosher)]),
            reply(calls=[tool_call("add_meal_to_plan", not_kosher)]),
            reply("done"),
        ])
        first, second = tool_results(client)
        self.assertIn("not kosher", first["error"])
        self.assertIn("already tried", second["error"])

    def test_failed_call_may_be_retried_after_the_state_changed(self):
        # Mid-week: remove_meal fails before the slots are declared, and works after.
        p = make_planner(max_missing_per_dish=2)
        p.add_meal_to_plan("שני", "lunch", 1, 10)
        p.mid_week = True
        remove = {"day": "שני", "meal": "lunch"}
        _, client = run(p, [
            reply(calls=[tool_call("remove_meal", remove)]),
            reply(calls=[tool_call("declare_change_targets", {"targets": [remove]})]),
            reply(calls=[tool_call("remove_meal", remove)]),
            reply("done"),
        ])
        first, _, third = tool_results(client)
        self.assertIn("declare", first["error"])
        self.assertTrue(third["ok"])

    def test_nudged_when_stopping_with_empty_slots(self):
        result, client = run(make_planner(plan_days=["ראשון"]), [reply("זהו"), reply("זהו"), reply("זהו")],
                             require_done=True)
        nudges = [m for m in client.sent[-1] if m["role"] == "user" and "still empty" in m["content"]]
        self.assertEqual(len(nudges), agent.MAX_NUDGES)
        self.assertFalse(result["finished"])
        self.assertEqual(result["reason"], "חלק מהארוחות לא תוכננו")

    def test_stops_after_max_steps(self):
        endless = [reply(calls=[tool_call("get_pantry", {})]) for _ in range(10)]
        with patch.object(agent, "MAX_STEPS", 5):
            result, _ = run(make_planner(), endless)
        self.assertFalse(result["finished"])
        self.assertEqual(result["steps"], 5)

    def test_stops_at_cost_limit(self):
        # Each reply "costs" 1M input tokens = $0.15, so the second step crosses $0.20.
        pricey = [reply(calls=[tool_call("get_plan_status", {}, f"c{i}")], input_tokens=1_000_000) for i in range(5)]
        result, client = run(make_planner(), pricey)
        self.assertFalse(result["finished"])
        self.assertIn("תקרת העלות", result["reason"])
        self.assertEqual(len(client.sent), 2)   # it didn't keep calling the API

    def test_repeated_search_gets_short_note(self):
        search = {"meal": "lunch", "dish_type": "main"}
        _, client = run(make_planner(), [
            reply(calls=[tool_call("search_recipes", search)]),
            reply(calls=[tool_call("search_recipes", search)]),
            reply("done"),
        ])
        first, second = tool_results(client)
        self.assertIn("recipes", first)
        self.assertIn("note", second)

    def test_old_read_results_are_shrunk(self):
        searches = [reply(calls=[tool_call("search_recipes", {"meal": "lunch", "kosher": k}, f"c{i}")])
                    for i, k in enumerate(["meat", "dairy", "parve", "meat"])]
        with patch.object(agent, "KEEP_RECENT_READS", 2):
            _, client = run(make_planner(), searches + [reply("done")])
        contents = [m["content"] for m in client.sent[-1] if m["role"] == "tool"]
        self.assertEqual(contents[0], agent.STALE_NOTE)   # oldest result was shrunk
        self.assertIn("recipes", contents[-1])
        # The 4th call repeats the 1st, but the 1st was shrunk, so it ran for real.
        self.assertIn("recipes", json.loads(contents[3]))

    def test_same_search_after_plan_changed_runs_again(self):
        search = {"meal": "dinner", "dish_type": "main"}
        _, client = run(make_planner(), [
            reply(calls=[tool_call("search_recipes", search)]),
            reply(calls=[tool_call("add_meal_to_plan", {"day": "ראשון", "meal": "lunch", "main_id": 1, "side_id": 10})]),
            reply(calls=[tool_call("search_recipes", search)]),
            reply("done"),
        ])
        self.assertIn("recipes", tool_results(client)[2])  # a real answer, since the plan changed


class ReasonTests(unittest.TestCase):
    def test_the_reason_is_shown_under_the_step(self):
        log = []
        p = make_planner(max_missing_per_dish=2)
        agent.run_agent(p, "goal", log=log.append, client=FakeClient([
            reply(calls=[tool_call("add_meal_to_plan", {"day": "ראשון", "meal": "lunch", "main_id": 1, "side_id": 10,
                                                        "reason": "יש בבית אורז, וצריך לקנות רק בשר"})]),
            reply("סיימתי")]))
        self.assertIn("למה: יש בבית אורז, וצריך לקנות רק בשר", log[0])
        self.assertNotIn("reason=", log[0])  # not repeated among the arguments

    def test_reason_is_required_where_the_plan_changes(self):
        from prompts import tool_schemas
        required = {t["function"]["name"]: t["function"]["parameters"]["required"] for t in tool_schemas(p_prefs())}
        for name in ("add_meal_to_plan", "remove_meal", "change_side", "change_people"):
            self.assertIn("reason", required[name])


def p_prefs():
    return make_planner().prefs


class SummaryTests(unittest.TestCase):
    def test_summary_is_rewritten_from_the_final_plan(self):
        # The plan is complete, but the first summary says a day couldn't be planned.
        p = make_planner(plan_days=["ראשון"], max_missing_per_dish=2)
        result, client = run(p, [
            reply(calls=[tool_call("add_meal_to_plan", {"day": "ראשון", "meal": "lunch", "main_id": 1, "side_id": 10})]),
            reply(calls=[tool_call("add_meal_to_plan", {"day": "ראשון", "meal": "dinner", "main_id": 3, "side_id": 12})]),
            reply("לא הצלחתי לתכנן את ראשון בערב"),
            reply("ביום ראשון בחרתי בקציצות ובשקשוקה"),
        ], require_done=True)
        self.assertEqual(result["summary"], "ביום ראשון בחרתי בקציצות ובשקשוקה")
        facts = client.sent[-1][-1]["content"]
        self.assertIn("EVERY slot is planned", facts)
        self.assertIn("קציצות", facts)

    def test_incomplete_plan_keeps_its_own_summary(self):
        result, client = run(make_planner(plan_days=["ראשון"]), [reply("זהו")] * 3, require_done=True)
        self.assertFalse(result["finished"])
        self.assertEqual(result["summary"], "זהו")


class EnvFileTests(unittest.TestCase):
    """The API key from a .env file, so it's never in the code."""

    def write(self, text):
        from pathlib import Path
        path = temp_dir() / ".env"
        path.write_text(text, encoding="utf-8")
        return path

    def test_key_from_the_file(self):
        with patch.dict("os.environ", {}, clear=True):
            agent.load_env_file(self.write('# comment\nOPENAI_API_KEY="sk-from-file"\n\nnot a line\n'))
            import os
            self.assertEqual(os.environ["OPENAI_API_KEY"], "sk-from-file")

    def test_environment_variable_wins(self):
        import os
        with patch.dict("os.environ", {"OPENAI_API_KEY": "sk-from-env"}, clear=True):
            agent.load_env_file(self.write("OPENAI_API_KEY=sk-from-file\n"))
            self.assertEqual(os.environ["OPENAI_API_KEY"], "sk-from-env")

    def test_no_file_is_fine(self):
        agent.load_env_file("no-such-folder/.env")


class WishTests(unittest.TestCase):
    """Turning a wish in the family's words into a rule the code can check."""
    DAYS = ["ראשון", "שני", "שלישי", "רביעי", "חמישי"]

    def understand(self, answer):
        client = FakeClient([reply(json.dumps(answer, ensure_ascii=False))])
        return agent.interpret_wish("בשלישי בערב בשרי", self.DAYS, client=client)

    def test_rule(self):
        rule, understood = self.understand({"rule": {"days": ["שלישי"], "meals": ["dinner"], "allowed": ["meat"]},
                                            "understood": "יום שלישי, ערב: בשרי"})
        self.assertEqual(rule, {"note": "בשלישי בערב בשרי", "days": ["שלישי"], "meals": ["dinner"], "allowed": ["meat"]})
        self.assertEqual(understood, "יום שלישי, ערב: בשרי")

    def test_not_a_rule(self):
        self.assertIsNone(self.understand({"rule": None, "understood": "פחות פטריות"})[0])

    def test_dish_rule(self):
        rule, _ = self.understand({"rule": {"days": ["שני"], "meals": ["lunch"], "must_have": ["מרק"]}})
        self.assertEqual(rule, {"note": "בשלישי בערב בשרי", "days": ["שני"], "meals": ["lunch"], "must_have": ["מרק"]})

    def test_hebrew_names_are_understood(self):
        rule, _ = self.understand({"rule": {"days": ["יום שלישי"], "meals": ["ערב"], "allowed": ["בשרי"]}})
        self.assertEqual((rule["days"], rule["meals"], rule["allowed"]), (["שלישי"], ["dinner"], ["meat"]))

    def test_unclear_answer_is_not_widened(self):
        # "brunch" isn't a meal we know: rather no rule than one for both meals.
        self.assertIsNone(self.understand({"rule": {"days": ["שלישי"], "meals": ["brunch"], "allowed": ["meat"]}})[0])
        self.assertIsNone(self.understand({"rule": {"allowed": ["meat", "dairy", "parve"]}})[0])  # allows anything
        rule, _ = self.understand({"rule": {"allowed": ["meat"]}})  # no day or meal named: all of them
        self.assertEqual((rule["days"], rule["meals"]), (self.DAYS, ["lunch", "dinner"]))


if __name__ == "__main__":
    unittest.main()
