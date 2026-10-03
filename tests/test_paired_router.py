import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from gain_router.config import ModelConfig
from gain_router.paired import collect, collect_pair, label, prepare_pool, score_pair, validate_record
from gain_router.schema import ModelCall, OfflineSupervision, TaskSample


def result(answer):
    return {"output": answer, "finish_reason": "stop", "quality_flags": [],
            "usage": {"total_tokens": 10}, "latency_seconds": 1}


class PairedRouterTests(unittest.TestCase):
    def setUp(self):
        self.actor = ModelConfig("http://localhost:8000/v1", "actor", "revision-a", .2, 4096)
        self.preference = ModelConfig("http://localhost:8001/v1", "preference", "revision-p", .4, 3072)
        self.task = TaskSample("qa:1", "g:1", "qa", "Capital of France?", "train", 42, "France is in Europe.")
        self.sup = OfflineSupervision("qa:1", "g:1", "qa", "Paris")

    def record(self, a="London", b="Paris", initial="London", rounds=2):
        self.calls = []

        def invoke(stage, role, model, system, user):
            self.calls.append((stage, role, user))
            output = initial if stage == "initial" else ("Consider the geography" if role == "preference"
                        else (b if stage.startswith("guided") else a))
            return ModelCall(role, model.model_id, model.revision, result(output))

        return collect_pair(self.task, self.actor, self.preference, invoke, rounds, "run", "attempt")

    def test_shared_initial_equal_actor_budget_and_guidance_boundary(self):
        record = self.record()
        self.assertEqual(len(self.calls), 7)
        self.assertEqual(sum(s == "initial" for s, _, _ in self.calls), 1)
        self.assertEqual([r for _, r, _ in self.calls], ["actor", "actor", "actor", "preference", "actor", "preference", "actor"])
        self.assertIn("Current response:\nLondon", self.calls[1][2])
        self.assertIn("Current response:\nLondon", self.calls[3][2])
        self.assertNotIn("Diversity guidance", self.calls[1][2])
        self.assertIn("Diversity guidance", self.calls[4][2])
        self.assertNotIn("gold", json.dumps(record["state"]))

    def test_four_outcomes_and_same_wrong_initial_different_route(self):
        for a, b, expected in (("London", "Paris", 1), ("Paris", "Paris", 0),
                                ("Paris", "London", 0), ("London", "London", 0)):
            row = score_pair(self.record(a, b), self.sup)
            self.assertEqual(row["label"], expected)
            self.assertEqual(row["branch_costs"]["actor_only"]["total_tokens"], 20)
            self.assertEqual(row["extra_guided_cost"]["total_tokens"], 20)
            self.assertNotIn("gold", row["features"])

    def test_final_answer_not_best_historical_answer(self):
        record = self.record("London", "London")
        record["guided"]["calls"][1]["result"]["output"] = "Paris"
        row = score_pair(record, self.sup)
        self.assertEqual(row["label"], 0)
        self.assertTrue(row["round_scores"]["guided"][0]["metrics"]["success"])
        self.assertFalse(row["round_scores"]["guided"][-1]["metrics"]["success"])

    def test_truncation_and_unknown_supervision_not_negative(self):
        record = self.record()
        record["guided"]["calls"][0]["result"]["finish_reason"] = "length"
        self.assertIsNone(score_pair(record, self.sup)["label"])
        wrong_sup = OfflineSupervision("different", "g:1", "qa", "Paris")
        with self.assertRaises(ValueError):
            score_pair(record, wrong_sup)

    def test_reject_mismatched_budget_roles_and_final(self):
        for mutation in (lambda r: r.update(rounds=3),
                         lambda r: r["actor_only"]["calls"][0].update(role="preference"),
                         lambda r: r["guided"].update(final_output="other")):
            record = self.record()
            mutation(record)
            with self.assertRaises(ValueError):
                validate_record(record)

    def test_resume_keeps_initial_and_completed_rounds_after_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            tasks = root / "tasks.jsonl"
            tasks.write_text(json.dumps(self.task.__dict__) + "\n")
            settings = root / "settings.json"
            settings.write_text(json.dumps({"actor": self.actor.__dict__,
                "preference_models": {"qa": self.preference.__dict__}, "rounds": 2,
                "seed": 42, "max_per_domain": 1, "split": "train"}))
            count = 0

            def fail(**kwargs):
                nonlocal count
                count += 1
                if count == 4:
                    raise RuntimeError("temporary failure")
                return result("London")

            with patch("gain_router.paired.call_actor", side_effect=fail):
                with self.assertRaises(RuntimeError):
                    collect(tasks, settings, root / "run", execute=True)
            with patch("gain_router.paired.call_actor", return_value=result("Paris")) as client:
                collect(tasks, settings, root / "run", execute=True)
                self.assertEqual(client.call_count, 4)
                # Same per-round actor seeds in both branches.
                record = json.loads((root / "run" / "trajectories.jsonl").read_text())
                self.assertEqual(record["state"]["initial"]["result"]["output"], "London")
            with patch("gain_router.paired.call_actor") as client:
                collect(tasks, settings, root / "run", execute=True)
                client.assert_not_called()
            supervision = root / "supervision.jsonl"
            supervision.write_text(json.dumps(self.sup.__dict__) + "\n")
            report = label(root / "run", supervision, root / "labels")
            self.assertEqual(report["positive"], 1)
            exported = json.loads((root / "labels" / "router_dataset.jsonl").read_text())
            self.assertNotIn("round_scores", exported)
            self.assertNotIn("gold", exported)
            with patch("gain_router.paired.call_actor") as client:
                collect(tasks, settings, root / "unused")
                client.assert_not_called()
                self.assertFalse((root / "unused").exists())

    def test_matched_actor_seeds_and_change_of_settings_refuses_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            tasks = root / "tasks.jsonl"
            tasks.write_text(json.dumps(self.task.__dict__) + "\n")
            settings = root / "settings.json"
            value = {"actor": self.actor.__dict__, "preference_models": {"qa": self.preference.__dict__},
                     "rounds": 1, "seed": 42, "max_per_domain": 1, "split": "train"}
            settings.write_text(json.dumps(value))
            with patch("gain_router.paired.call_actor", return_value=result("Paris")) as client:
                collect(tasks, settings, root / "run", execute=True)
                self.assertEqual(client.call_args_list[1].kwargs["seed"], client.call_args_list[3].kwargs["seed"])
            value["rounds"] = 2
            settings.write_text(json.dumps(value))
            with self.assertRaises(ValueError):
                collect(tasks, settings, root / "run", execute=True)

    def test_prepare_combines_old_partitions_and_separates_supervision(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data = {
                "dataset/code/train/code.json": [{"question": "write code", "func_sign": "def f():"}],
                "dataset/math/train/math.json": [{"problem": "calculate one"}],
                "dataset/multihop/train/sft.jsonl": [{"question": "Capital of France?"}],
                "dataset/code/test/humaneval.jsonl": [{"prompt": "other code"}],
                "samples_math_150/sample_math_150.jsonl": [{"problem": "other math"}],
                "dataset/multihop/test/hotpot_qa.jsonl": [{"question": "other QA"}],
            }
            for name, rows in data.items():
                p = root / name
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_text(json.dumps(rows) if p.suffix == ".json" else "".join(json.dumps(r) + "\n" for r in rows))
            code = [{"source_index": 0, "problem": "write code def f():", "tests": {"inputs": [[]], "outputs": [1], "fn_name": "f"}}]
            math = [{"source_index": 0, "problem": "calculate one", "gold": "1"}]
            qa = [{"source_index": 0, "problem": "Capital of France?", "gold": "Paris", "context": "France is in Europe"}]
            # Each old partition yields duplicates: combine then group/deduplicate.
            with patch("gain_router.datasets.load_code_rows", return_value=(code, {})) as c, \
                 patch("gain_router.datasets.load_math_rows", return_value=(math, {})), \
                 patch("gain_router.datasets.load_qa_rows", return_value=(qa, {})), \
                 patch("gain_router.datasets.find_qa_context_files", return_value=[]):
                audit = prepare_pool(root, root / "pool")
                self.assertEqual(c.call_count, 3)
            self.assertTrue(all(v["eligible_groups"] == 1 for v in audit.values()))
            tasks = [json.loads(l) for l in (root / "pool/tasks.jsonl").read_text().splitlines()]
            self.assertEqual(len(tasks), 3)
            self.assertTrue(all("gold" not in t and "tests" not in t for t in tasks))
            self.assertTrue((root / "pool/supervision.jsonl").exists())


if __name__ == "__main__":
    unittest.main()
