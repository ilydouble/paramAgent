import importlib.util
import json
from pathlib import Path
import tarfile
import tempfile
import unittest
from unittest.mock import patch

from gain_router.actor import call_actor
from gain_router.paired import collect, score_pair
from gain_router.schema import OfflineSupervision
from gain_router.traces import config_fingerprint, write_json_atomic

spec = importlib.util.spec_from_file_location("compact_paired_run", Path(__file__).resolve().parents[1] / "scripts/compact_paired_run.py")
migration = importlib.util.module_from_spec(spec)
spec.loader.exec_module(migration)


class CompactStorageTests(unittest.TestCase):
    def test_compact_actor_preserves_confidence_without_raw_or_candidate_tokens(self):
        raw = {"choices": [{"message": {"content": "Paris", "reasoning_content": "France"},
            "finish_reason": "stop", "logprobs": {"content": [
                {"logprob": -.2, "top_logprobs": [{"token": "unused", "logprob": -.1}]},
                {"logprob": -.4}]}}], "usage": {"completion_tokens": 2}}
        kwargs = dict(base_url="http://localhost/v1", model="stub", system="s", user="u",
            temperature=.2, max_tokens=10, seed=42, timeout=1, retries=0)
        events = []
        with patch("gain_router.actor.post_json", return_value=raw) as client:
            original = call_actor(**kwargs)
            compact = call_actor(**kwargs, compact=True, trace=lambda e, d: events.append((e, d)))
            self.assertEqual(client.call_args.args[1]["top_logprobs"], 0)
            self.assertTrue(client.call_args.args[1]["logprobs"])
        self.assertEqual(compact, {k: v for k, v in original.items()
            if k not in {"raw_response", "messages", "latency_seconds"}} | {"latency_seconds": compact["latency_seconds"]})
        self.assertEqual(compact["confidence"], original["confidence"])
        self.assertNotIn("raw_response", json.dumps(events))
        self.assertNotIn("unused", json.dumps(events))
        self.assertEqual(events[-1][1]["call"]["output_sha256"], migration.hashlib.sha256(b"Paris").hexdigest())

    def prepare_legacy(self, root):
        tasks = root / "tasks.jsonl"
        rows = [{"sample_id": f"qa:{i}", "group_id": f"g:{i}", "domain": "qa",
            "problem": "Capital of France?", "split": "train", "split_seed": 42, "context": "France"} for i in range(2)]
        tasks.write_text("".join(json.dumps(r) + chr(10) for r in rows))
        model = {"endpoint": "http://localhost/v1", "model_id": "stub", "revision": "revision",
            "temperature": .2, "max_tokens": 4096}
        settings = root / "settings.json"
        settings.write_text(json.dumps({"actor": model, "preference_models": {"qa": model},
            "rounds": 5, "seed": 42, "max_per_domain": 2, "split": "train"}))
        n = 0
        def fail(**kwargs):
            nonlocal n
            n += 1
            if n == 20:
                raise RuntimeError("interrupted")
            return {"output": "Paris", "reasoning_content": "France", "finish_reason": "stop",
                "confidence": {"available": True, "mean_logprob": -.2}, "usage": {"total_tokens": 10},
                "latency_seconds": 1, "seed": kwargs["seed"], "quality_flags": [],
                "raw_response": {"large_unused_tokens": "x" * 10000}, "messages": [{"content": "duplicate"}]}
        run = root / "run"
        with patch("gain_router.paired.call_actor", side_effect=fail), self.assertRaises(RuntimeError):
            collect(tasks, settings, run, execute=True)
        # Reconstruct a legacy disk representation with genuine duplicated raw
        # fields, including a partially completed second task.
        for p in (run / "calls").glob("*.json"):
            v = json.loads(p.read_text())
            v["call"]["result"].update(raw_response={"large_unused_tokens": "x" * 10000}, messages=[{"content": "duplicate"}])
            write_json_atomic(p, v)
        v = json.loads((run / "trajectories.jsonl").read_text())
        for c in [v["state"]["initial"], *v["actor_only"]["calls"], *v["guided"]["calls"]]:
            c["result"].update(raw_response={"large_unused_tokens": "x" * 10000}, messages=[{"content": "duplicate"}])
        (run / "trajectories.jsonl").write_text(json.dumps(v) + chr(10))
        metadata = json.loads((run / "run.json").read_text())
        metadata["config"].pop("storage_format")
        metadata["config"]["code_sha256"].pop("paired_storage.py")
        metadata["config"]["code_sha256"].update(migration.LEGACY_CODE)
        metadata["config_fingerprint"] = config_fingerprint(metadata["config"])
        write_json_atomic(run / "run.json", metadata)
        events = [{"run_id": metadata["run_id"], "config_fingerprint": metadata["config_fingerprint"],
            "event": "request.completed", "data": {"call": v["state"]["initial"]["result"]}},
            {"run_id": metadata["run_id"], "config_fingerprint": metadata["config_fingerprint"],
             "event": "response.received", "data": {"raw_response": {"choices": [], "huge": "x" * 10000}}}]
        (run / "events.jsonl").write_text("".join(json.dumps(e) + chr(10) for e in events))
        return run, tasks, settings

    def test_verified_backup_migration_preserves_labels_and_resumes_partial_task(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run, tasks, settings = self.prepare_legacy(root)
            old_bytes = {str(p.relative_to(run)): p.read_bytes() for p in run.rglob("*") if p.is_file() and p.name != ".writer.lock"}
            before = json.loads((run / "trajectories.jsonl").read_text())
            task = before["state"]["task"]
            sup = OfflineSupervision(task["sample_id"], task["group_id"], "qa", "Paris")
            score = score_pair(before, sup)
            backup = root / "backup.tar.gz"
            report = migration.migrate(run, tasks, settings, backup)
            self.assertEqual(report["counts"]["trajectories"], 1)
            self.assertEqual(report["counts"]["calls"], 19)
            self.assertEqual(report["counts"]["events"], 2)
            self.assertEqual(report["partial_task_call_counts"], [3])
            self.assertLess(report["after_bytes"], report["before_bytes"] / 4)
            with tarfile.open(backup, "r:gz") as archive:
                for name, expected in old_bytes.items():
                    self.assertEqual(archive.extractfile(name).read(), expected)
            after = json.loads((run / "trajectories.jsonl").read_text())
            self.assertEqual(before["run_id"], after["run_id"])
            self.assertEqual(score_pair(after, sup), score)
            self.assertNotIn("raw_response", (run / "events.jsonl").read_text())
            with patch("gain_router.paired.call_actor", return_value={"output": "Paris", "finish_reason": "stop"}) as client:
                collect(tasks, settings, run, execute=True)
                self.assertEqual(client.call_count, 13)
            with patch("gain_router.paired.call_actor") as client:
                collect(tasks, settings, run, execute=True)
                client.assert_not_called()

    def test_changed_settings_or_active_writer_refuses_migration_without_backup(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run, tasks, settings = self.prepare_legacy(root)
            metadata_bytes = (run / "run.json").read_bytes()
            values = json.loads(settings.read_text()); values["rounds"] = 4
            settings.write_text(json.dumps(values))
            with self.assertRaisesRegex(ValueError, "differ"):
                migration.migrate(run, tasks, settings, root / "backup.tar.gz")
            self.assertFalse((root / "backup.tar.gz").exists())
            self.assertEqual((run / "run.json").read_bytes(), metadata_bytes)
            with (run / ".writer.lock").open("a") as handle:
                migration.fcntl.flock(handle.fileno(), migration.fcntl.LOCK_EX | migration.fcntl.LOCK_NB)
                with self.assertRaises(BlockingIOError):
                    migration.migrate(run, tasks, settings, root / "backup.tar.gz")

    def test_incomplete_migration_blocks_collection(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run, tasks, settings = self.prepare_legacy(root)
            (run / ".storage-migration.pending.json").write_text("{}")
            with patch("gain_router.paired.call_actor") as client:
                with self.assertRaisesRegex(ValueError, "migration incomplete"):
                    collect(tasks, settings, run, execute=True)
                client.assert_not_called()


if __name__ == "__main__":
    unittest.main()
