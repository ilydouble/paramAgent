import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/run_parammem_pilot.py"
spec = importlib.util.spec_from_file_location("parammem_pilot", SCRIPT)
pilot = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pilot)


class PilotTests(unittest.TestCase):
    def test_final_results_include_second_stage_recovery(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            first = [{"task_id": "a", "is_solved": False}, {"task_id": "b", "is_solved": True}]
            final = [{"task_id": "a", "is_solved": True, "stage2": True}, first[1]]
            (path / "first_stage_log.jsonl").write_text("\n".join(map(json.dumps, first)))
            with self.assertRaisesRegex(RuntimeError, "not completed"):
                pilot.stage_results(path)
            (path / "second_stage_log.jsonl").write_text("\n".join(map(json.dumps, final)))
            self.assertEqual(sum(r["is_solved"] for r in pilot.stage_results(path)[1]), 2)

    def test_humaneval_contract_accepts_correct_and_rejects_wrong(self):
        test = "def check(candidate):\n    assert candidate(2,3)==5\ncheck(add)"
        self.assertTrue(pilot.code_check("def add(a,b): return a+b", test)[0])
        self.assertFalse(pilot.code_check("def add(a,b): return a-b", test)[0])

    def test_candidate_timeout_is_terminated(self):
        ok, error = pilot.code_check("while True: pass", "", timeout=0.1)
        self.assertFalse(ok)
        self.assertEqual(error, "timeout")

    def test_qa_normalization_and_yes_no(self):
        self.assertEqual(pilot.qa_score("The Nile.", "Nile")["em"], 1)
        self.assertEqual(pilot.qa_score("New York", "New York City")["f1"], 0.8)
        self.assertEqual(pilot.qa_score("yes", "no")["f1"], 0)

    def test_memory_uses_training_contract_without_current_answer(self):
        for domain, marker in [("code", "FUNC_SIGNATURE:"), ("math", "PROBLEM:"), ("qa", "[Question]:")]:
            messages = pilot.memory_messages(domain, "frozen question")
            self.assertIn(marker, messages[1]["content"])
            self.assertTrue(messages[1]["content"].endswith("frozen question"))
            self.assertNotIn("Current response:", messages[1]["content"])

    def test_math_units_do_not_change_numeric_value(self):
        self.assertEqual(pilot.clean_math(r"428\text{ square centimeters}"), "428")
        self.assertEqual(pilot.clean_math(r"90^\circ"), "90")

    def test_journal_recovers_cached_calls_but_not_evaluations(self):
        with tempfile.TemporaryDirectory() as directory:
            journal = pilot.Journal(Path(directory))
            journal.write({"type": "actor", "key": "a", "response": {}})
            journal.write({"type": "evaluation", "result": {"success": False}})
            self.assertEqual(set(pilot.Journal(Path(directory)).cache), {"a"})


if __name__ == "__main__":
    unittest.main()
