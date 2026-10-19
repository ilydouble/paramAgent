import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
spec = importlib.util.spec_from_file_location("pilot_summary", SCRIPTS / "summarize_parammem_pilot.py")
summary = importlib.util.module_from_spec(spec)
spec.loader.exec_module(summary)


class SummaryTests(unittest.TestCase):
    def test_second_stage_gains_and_caught_errors(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for domain in ("code", "math", "qa"):
                for method in ("qwen", "parammem", "parammem-dpo"):
                    path = root / "results" / domain / method
                    path.mkdir(parents=True)
                    (path / "completed.json").write_text("{}")
                    first = [{"task_id": str(i), "is_solved": i > 0, "solution": "Paris", "answer": "Paris"} for i in range(10)]
                    final = [{**r, "stage2": True, "is_solved": method == "parammem-dpo" or r["is_solved"]} for r in first]
                    calls = [{"type": "evaluation", "task_id": str(i), "result": {"success": i > 0, "f1": float(i > 0)}} for i in range(10)]
                    for name, rows in (("first_stage_log", first), ("second_stage_log", final), ("calls", calls)):
                        (path / f"{name}.jsonl").write_text("\n".join(map(json.dumps, rows)))
            result = summary.summarize(root)
            row = next(r for r in result["groups"] if r["domain"] == "code" and r["method"] == "parammem-dpo")
            self.assertEqual((row["stage1"], row["final"]), (9, 10))
            pair = next(r for r in result["dpo_pairs"] if r["domain"] == "code" and r["stage"] == "final" and r["control"] == "parammem")
            self.assertEqual(pair["dpo_improved"], ["0"])
            (root / "code-qwen.log").write_text("Error in item 0: invalid execution")
            with self.assertRaisesRegex(RuntimeError, "Runner errors"):
                summary.summarize(root)


if __name__ == "__main__":
    unittest.main()
