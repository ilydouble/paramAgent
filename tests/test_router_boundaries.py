"""Config, online/offline isolation, call/no-call labels and safe feature export."""
import contextlib
from dataclasses import asdict, replace
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from split_protocol import assign_exact_groups
from gain_router import actor, collect, offline
from gain_router.audit import audit_events, audit_trajectories
from gain_router.config import ExperimentConfig, LabelConfig, load_config
from gain_router.export import export_features
from gain_router.inputs import load_tasks, read_objects
from gain_router.schema import CollectionRecord, DecisionState, ModelCall, OfflineSupervision, StrategyTrace, TaskSample, serialize


def model_config(name="actor"):
    return {"endpoint": "http://127.0.0.1:8000/v1", "model_id": name, "revision": "test-revision",
            "temperature": 0.2, "max_tokens": 256, "enable_thinking": False}


def experiment():
    return {"version": "router_experiment_v1", "purpose": "pilot", "tasks_path": "tasks.jsonl",
            "split_manifest": "split.json", "split": "train", "output_dir": "run", "seed": 42,
            "max_per_domain": 2, "actor": model_config(), "preference_models": {"math": {**model_config("preference"), "weights_path": "sft", "adapter_path": "dpo"}},
            "policy_version": "preference_repair_single_v1", "retries": 0,
            "labels": {"version": "binary_success_net_gain_v1", "verifier_version": "legacy_deterministic_v1",
                       "lambda_tokens": 0.0, "lambda_latency": 0.0}}


def setup_inputs(root):
    tasks = [{"sample_id": f"math:{i}", "group_id": f"group:{i}", "domain": "math", "problem": f"Return one, item {i}",
              "split": "train", "split_seed": 42} for i in range(2)]
    records = [{"sample_id": t["sample_id"], "group_id": t["group_id"], "stratum": "math"} for t in tasks]
    _, manifest = assign_exact_groups(records, seed=42, targets={"math": {"train": 2, "val": 0, "test": 0}})
    (root / "tasks.jsonl").write_text("".join(json.dumps(t) + "\n" for t in tasks))
    (root / "split.json").write_text(json.dumps(manifest))
    # Small fake artifacts exercise the same provenance guard without model loading.
    from split_protocol import file_sha256
    for stage in ("sft", "dpo"):
        directory = root / stage
        directory.mkdir()
        (directory / "weights.bin").write_bytes(stage.encode())
        report = {"status": "complete", "stage": stage, "domain": "math", "split_seed": 42,
                  "assignment_sha256": manifest["assignment_sha256"],
                  "group_ids": {"train": [t["group_id"] for t in tasks], "val": []},
                  "artifacts": {"weights.bin": file_sha256(directory / "weights.bin")}}
        if stage == "dpo":
            report["base_artifacts"] = json.loads((root / "sft/training_split.json").read_text())["artifacts"]
        (directory / "training_split.json").write_text(json.dumps(report))
    (root / "supervision.jsonl").write_text("".join(json.dumps({"sample_id": t["sample_id"], "group_id": t["group_id"],
                                                               "domain": "math", "gold": "1"}) + "\n" for t in tasks))


def response(text="FINAL_ANSWER: 1"):
    return {"choices": [{"message": {"content": text}, "finish_reason": "stop"}],
            "usage": {"total_tokens": 100, "prompt_tokens": 90, "completion_tokens": 10}}


def call(role, output, flags=()):
    return ModelCall(role, role, "test", {"output": output, "finish_reason": "stop", "quality_flags": list(flags),
                                          "usage": {"total_tokens": 100}, "latency_seconds": 1.0,
                                          "confidence": {"mean_logprob": -0.5, "gold": "LEAK", "f1": 1.0},
                                          "raw_response": {"offline_gold": "LEAK"}})


def paired(initial="London", repaired="Paris", domain="qa", flags=()):
    task = TaskSample("a", "g", domain, "What city?", "train", 42, "Inference-visible evidence")
    state = DecisionState(task, call("actor", initial, flags))
    preference, repair = call("preference", "Guidance"), call("actor", repaired)
    return CollectionRecord("router_call_v1", "pilot", "r", "attempt", state,
                            StrategyTrace("retain_initial_v1", (), initial),
                            StrategyTrace("preference_repair_single_v1", (preference, repair), repaired))


class BoundaryTests(unittest.TestCase):
    def test_task_rejects_offline_fields(self):
        value = asdict(paired().state.task)
        for key in ("gold", "tests", "pitfalls", "decomposition", "guidance"):
            with self.subTest(key=key), self.assertRaises(ValueError):
                TaskSample.from_dict({**value, key: "secret"})

    def test_config_rejects_unknown_fields_formal_mode_and_test_split(self):
        for change in ({"typo": 1}, {"purpose": "formal"}, {"split": "test"}, {"policy_version": "unknown"}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                ExperimentConfig.from_dict({**experiment(), **change})

    def test_yaml_duplicate_keys_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.yaml"
            path.write_text("purpose: pilot\npurpose: formal\n")
            with self.assertRaises(ValueError):
                load_config(path)

    def test_model_endpoint_credentials_and_bad_sampling_rejected(self):
        for change in ({"endpoint": "http://name:secret@localhost/v1"}, {"max_tokens": 0},
                       {"top_p": float("nan")}, {"temperature": -1}, {"enable_thinking": "false"}):
            value = experiment()
            value["actor"].update(change)
            with self.subTest(change=change), self.assertRaises(ValueError):
                ExperimentConfig.from_dict(value)

    def test_relabeling_does_not_change_collection_identity(self):
        value = experiment()
        first = ExperimentConfig.from_dict(value)
        value["labels"]["lambda_tokens"] = 0.01
        second = ExperimentConfig.from_dict(value)
        self.assertEqual(first.collection_config(), second.collection_config())
        self.assertNotEqual(first.labels, second.labels)

    def test_manifest_mismatch_and_duplicate_inputs_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            setup_inputs(root)
            tasks = list(read_objects(root / "tasks.jsonl"))
            tasks[0]["split"] = "val"
            (root / "tasks.jsonl").write_text("".join(json.dumps(t) + "\n" for t in tasks))
            with self.assertRaises(ValueError):
                load_tasks(root / "tasks.jsonl", root / "split.json")
            tasks[0]["split"] = "train"
            (root / "tasks.jsonl").write_text(json.dumps(tasks[0]) + "\n" + json.dumps(tasks[0]) + "\n")
            with self.assertRaises(ValueError):
                load_tasks(root / "tasks.jsonl", root / "split.json")

    def test_whitelist_excludes_offline_and_future_information(self):
        record = paired()
        features = record.state.router_features()
        packed = json.dumps(features)
        self.assertNotIn("LEAK", packed)
        self.assertNotIn("Paris", packed)
        self.assertNotIn("Guidance", packed)
        self.assertNotIn("gold", packed)
        self.assertNotIn("f1", packed)
        self.assertEqual(features["initial_answer"], "London")

    def test_binary_labels_compare_call_to_retain_not_generic_repair(self):
        config = ExperimentConfig.from_dict(experiment()).labels
        supervision = OfflineSupervision("a", "g", "qa", gold="Paris")
        for initial, repaired, expected in (("London", "Paris", 1), ("Paris", "Paris", 0),
                                           ("Paris", "London", 0), ("London", "London", 0)):
            with self.subTest(initial=initial, repaired=repaired):
                result = offline.derive_label(paired(initial, repaired), supervision, config)
                self.assertEqual(result["label"], expected)
        # Calling is not worth a two-token-budget unit penalty, even if it repairs.
        costly = replace(config, lambda_tokens=0.01)
        self.assertEqual(offline.derive_label(paired(), supervision, costly)["label"], 0)

    def test_quality_and_ambiguous_math_are_excluded_not_negative_labels(self):
        config = ExperimentConfig.from_dict(experiment()).labels
        truncated = offline.derive_label(paired(flags=("truncated",)), OfflineSupervision("a", "g", "qa", gold="Paris"), config)
        self.assertIsNone(truncated["label"])
        self.assertIn("call_0:truncated", truncated["exclusion_reasons"])
        ambiguous = offline.derive_label(paired("2*sqrt(2)", "2*sqrt(2)", "math"),
                                         OfflineSupervision("a", "g", "math", gold=r"\sqrt{8}"), config)
        self.assertIsNone(ambiguous["label"])
        self.assertIn("ambiguous_math_equivalence", ambiguous["exclusion_reasons"])

    def test_supervision_identity_and_test_lengths_checked(self):
        with self.assertRaises(ValueError):
            OfflineSupervision.from_dict({"sample_id": "a", "group_id": "g", "domain": "code",
                                          "tests": {"fn_name": "f", "inputs": [[1]], "outputs": []}})
        with self.assertRaises(ValueError):
            offline.derive_label(paired(), OfflineSupervision("other", "g", "qa", gold="Paris"),
                                 ExperimentConfig.from_dict(experiment()).labels)

    def test_no_call_baseline_cannot_be_replaced_by_another_answer(self):
        value = json.loads(json.dumps(serialize(paired())))
        value["no_call"]["final_output"] = "Other answer"
        with self.assertRaises(ValueError):
            CollectionRecord.from_dict(value)

    def test_dry_run_and_execute_guard(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            setup_inputs(root)
            (root / "config.yaml").write_text(json.dumps(experiment()))
            with patch.object(actor, "post_json") as request, contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(collect.main(["--config", str(root / "config.yaml"), "--root", tmp]), 0)
            request.assert_not_called()
            self.assertFalse((root / "run").exists())

    def test_collection_rejects_wrong_model_partition_before_api(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            setup_inputs(root)
            config = ExperimentConfig.from_dict(experiment())
            path = root / "dpo/training_split.json"
            report = json.loads(path.read_text())
            report["split_seed"] = 123
            path.write_text(json.dumps(report))
            with patch.object(actor, "post_json") as request, self.assertRaises(ValueError):
                collect.run_collection(root, config)
            request.assert_not_called()
            self.assertFalse((root / "run").exists())

    def test_collection_rejects_different_sft_ancestry_before_api(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            setup_inputs(root)
            config = ExperimentConfig.from_dict(experiment())
            path = root / "dpo/training_split.json"
            report = json.loads(path.read_text())
            report["base_artifacts"] = {"weights.bin": "wrong-base-hash"}
            path.write_text(json.dumps(report))
            with patch.object(actor, "post_json") as request, self.assertRaisesRegex(ValueError, "different SFT"):
                collect.run_collection(root, config)
            request.assert_not_called()

    def test_collect_all_initial_states_and_resume_without_verifier(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            setup_inputs(root)
            config = ExperimentConfig.from_dict(experiment())
            with patch.object(actor, "post_json", return_value=response()) as request, \
                 patch.object(offline.verifiers, "verify", side_effect=AssertionError("gold access")), \
                 contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(collect.run_collection(root, config), 0)
                self.assertEqual(collect.run_collection(root, config), 0)
            # Both initial answers are correct; BOTH still have preference/repair calls.
            self.assertEqual(request.call_count, 6)
            records = list(read_objects(root / "run" / "trajectories.jsonl"))
            self.assertEqual(len(records), 2)
            self.assertTrue(all(len(r["call"]["calls"]) == 2 for r in records))
            self.assertNotIn("gold", json.dumps(records))
            self.assertNotIn("label", json.dumps(records))
            report = audit_trajectories(root / "run" / "trajectories.jsonl")
            self.assertEqual(report["total_calls"], 6)
            self.assertEqual(report["unlabeled_call_no_call_rows"], 2)
            events = audit_events(root / "run" / "events.jsonl")
            self.assertFalse(events["online_verification_expected"])
            self.assertNotIn("returned_stages_without_verification", events)

    def test_collect_label_export_end_to_end_and_immutable_versions(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            setup_inputs(root)
            config = ExperimentConfig.from_dict(experiment())
            with patch.object(actor, "post_json", return_value=response()), contextlib.redirect_stdout(io.StringIO()):
                collect.run_collection(root, config)
            with patch.object(actor, "post_json", side_effect=AssertionError("offline model call")):
                directory = offline.build_labels(root / "run", config, root / "supervision.jsonl")
                with self.assertRaises(FileExistsError):
                    offline.build_labels(root / "run", config, root / "supervision.jsonl")
                with self.assertRaises(ValueError):
                    export_features(root / "run", directory, root / "export", config)
                self.assertEqual(export_features(root / "run", directory, root / "export", config, allow_pilot=True), 2)
                revised = replace(config, labels=replace(config.labels, lambda_tokens=0.01))
                second = offline.build_labels(root / "run", revised, root / "supervision.jsonl")
                self.assertNotEqual(directory, second)
            features = list(read_objects(root / "export" / "router_features.jsonl"))
            self.assertTrue(all(row["label"] == 0 for row in features))
            self.assertNotIn("offline_rewards", json.dumps(features))
            self.assertNotIn("raw_response", json.dumps(features))
            self.assertNotIn("repair", json.dumps(features))

    def test_modified_labels_are_not_exported(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            setup_inputs(root)
            config = ExperimentConfig.from_dict(experiment())
            with patch.object(actor, "post_json", return_value=response()), contextlib.redirect_stdout(io.StringIO()):
                collect.run_collection(root, config)
            directory = offline.build_labels(root / "run", config, root / "supervision.jsonl")
            path = directory / "labels.jsonl"
            path.write_text(path.read_text().replace('"label": 0', '"label": 1'))
            with self.assertRaises(ValueError):
                export_features(root / "run", directory, root / "export", config, allow_pilot=True)

    def test_changed_collection_config_refuses_resume_before_api(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            setup_inputs(root)
            config = ExperimentConfig.from_dict(experiment())
            with patch.object(actor, "post_json", return_value=response()), contextlib.redirect_stdout(io.StringIO()):
                collect.run_collection(root, config)
            changed = replace(config, actor=replace(config.actor, max_tokens=300))
            with patch.object(actor, "post_json") as request, self.assertRaises(ValueError):
                collect.run_collection(root, changed)
            request.assert_not_called()

    def test_partial_collection_retains_initial_and_records_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            setup_inputs(root)
            config = replace(ExperimentConfig.from_dict(experiment()), max_per_domain=1)
            with patch.object(actor, "post_json", side_effect=[response(), RuntimeError("preference unavailable")]):
                self.assertEqual(collect.run_collection(root, config), 1)
            events = list(read_objects(root / "run" / "events.jsonl"))
            self.assertTrue(any(e["stage"] == "initial" and e["event"] == "response.received" for e in events))
            self.assertEqual(events[-1]["event"], "sample.failed")
            self.assertEqual(json.loads((root / "run" / "manifest.json").read_text())["status"], "partial")
            self.assertFalse((root / "run" / "trajectories.jsonl").exists())


if __name__ == "__main__":
    unittest.main()
