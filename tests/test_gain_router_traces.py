"""CPU-only regression tests; no model service or external datasets required."""
import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from gain_router import actor, legacy
from gain_router.audit import audit_events, audit_trajectories
from gain_router.common import read_jsonl
from gain_router.traces import TraceRecorder


def response(output="FINAL_ANSWER: 1", finish="stop"):
    return {"choices": [{"message": {"content": output, "reasoning_content": "private reasoning"},
                         "finish_reason": finish}], "usage": {"completion_tokens": 3}}


class TraceTests(unittest.TestCase):
    def test_actor_keeps_raw_response_reasoning_and_quality_flags(self):
        events = []
        raw = response("", "length")
        with patch.object(actor, "post_json", return_value=raw):
            call = actor.call_actor(base_url="http://localhost/v1", model="stub",
                                    system="s", user="u", temperature=0.2,
                                    max_tokens=10, seed=1, timeout=1, retries=0,
                                    trace=lambda kind, data: events.append((kind, data)))
        self.assertEqual(call["raw_response"], raw)
        self.assertEqual(call["reasoning_content"], "private reasoning")
        self.assertEqual(call["quality_flags"], ["truncated", "empty_output"])
        self.assertEqual([e[0] for e in events], ["request.started", "response.received", "request.completed"])

    def test_actor_journals_each_retry(self):
        events = []
        with patch.object(actor, "post_json", side_effect=[RuntimeError("down"), response()]), \
             patch.object(actor.time, "sleep"):
            actor.call_actor(base_url="http://localhost/v1", model="stub", system="s", user="u",
                             temperature=0, max_tokens=10, seed=1, timeout=1, retries=1,
                             trace=lambda kind, data: events.append((kind, data)))
        self.assertEqual([e[0] for e in events],
                         ["request.started", "request.failed", "request.started", "response.received", "request.completed"])
        self.assertEqual(events[-1][1]["request_attempt"], 2)

    def test_journal_disk_error_does_not_repeat_model_call(self):
        def fail_after_response(kind, data):
            if kind == "request.completed":
                raise OSError("disk full")
        with patch.object(actor, "post_json", return_value=response()) as request:
            with self.assertRaises(OSError):
                actor.call_actor(base_url="http://localhost/v1", model="stub", system="s", user="u",
                                 temperature=0, max_tokens=10, seed=1, timeout=1, retries=2,
                                 trace=fail_after_response)
        self.assertEqual(request.call_count, 1)

    def test_config_changes_and_unversioned_outputs_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with TraceRecorder(root, {"model": "a"}) as recorder:
                run_id = recorder.run_id
            original = (root / "run.json").read_bytes()
            with TraceRecorder(root, {"model": "a"}) as recorder:
                self.assertEqual(recorder.run_id, run_id)
            with self.assertRaises(ValueError):
                TraceRecorder(root, {"model": "b"})
            self.assertEqual((root / "run.json").read_bytes(), original)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "trajectories.jsonl").write_text("{}\n")
            with self.assertRaises(ValueError):
                TraceRecorder(root, {})
            self.assertEqual((root / "trajectories.jsonl").read_text(), "{}\n")

    def test_single_writer_and_corrupt_tail_protection(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with TraceRecorder(root, {}):
                with self.assertRaises(BlockingIOError):
                    TraceRecorder(root, {})
            (root / "events.jsonl").write_text('{"event":')
            with self.assertRaises(ValueError):
                TraceRecorder(root, {})
            self.assertEqual((root / "events.jsonl").read_text(), '{"event":')

    def test_model_return_is_saved_before_verifier_crash(self):
        row = {"sample_id": "math:a", "group_id": "math:a", "domain": "math",
               "split": "train", "problem": "One?", "gold": "1"}
        args = legacy.build_parser().parse_args([])
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with TraceRecorder(root, {}) as recorder, \
                 patch.object(actor, "post_json", return_value=response()), \
                 patch.object(legacy, "verify", side_effect=ValueError("verifier broke")):
                with self.assertRaises(ValueError):
                    legacy.generate_one(row, args, recorder=recorder)
            events = list(read_jsonl(root / "events.jsonl"))
            self.assertEqual(events[-1]["event"], "request.completed")
            self.assertIn("raw_response", events[-1]["data"]["call"])
            report = audit_events(root / "events.jsonl")
            self.assertEqual(report["returned_stages_without_verification"], 1)
            self.assertEqual(report["sample_attempts_without_terminal_event"], 1)

    def test_seed_is_sample_based(self):
        self.assertEqual(legacy.sample_seed("a", 42), legacy.sample_seed("a", 42))
        self.assertNotEqual(legacy.sample_seed("a", 42), legacy.sample_seed("b", 42))
        self.assertNotEqual(legacy.sample_seed("a", 42), legacy.sample_seed("a", 43))

    def test_malformed_response_is_preserved_before_parse_failure(self):
        events = []
        with patch.object(actor, "post_json", return_value={"unexpected": "response"}), \
             self.assertRaises(RuntimeError):
            actor.call_actor(base_url="http://localhost/v1", model="stub", system="s", user="u",
                             temperature=0, max_tokens=10, seed=1, timeout=1, retries=0,
                             trace=lambda kind, data: events.append((kind, data)))
        self.assertEqual(events[1][0], "response.received")
        self.assertEqual(events[1][1]["raw_response"], {"unexpected": "response"})
        self.assertEqual(events[-1][0], "request.failed")

    def test_later_branch_failure_keeps_initial_call(self):
        row = {"sample_id": "math:a", "group_id": "math:a", "domain": "math",
               "split": "train", "problem": "One?", "gold": "1"}
        args = legacy.build_parser().parse_args(["--retries", "0"])
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with TraceRecorder(root, {}) as recorder, \
                 patch.object(actor, "post_json", side_effect=[response("FINAL_ANSWER: 2"), RuntimeError("down")]):
                with self.assertRaises(RuntimeError):
                    legacy.generate_one(row, args, recorder=recorder)
            events = list(read_jsonl(root / "events.jsonl"))
            self.assertTrue(any(e["stage"] == "initial" and e["event"] == "verification.completed" for e in events))
            self.assertEqual(events[-1]["stage"], "generic")
            self.assertEqual(events[-1]["event"], "request.failed")

    def test_dry_run_does_not_write_or_call_models(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "math.json").write_text(json.dumps([{"problem": "One?", "answer": "1"}]))
            with patch.object(actor, "post_json") as request, contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(legacy.main(["--root", tmp, "--domains", "math", "--math-file", "math.json",
                                              "--output-dir", "new-run", "--dry-run"]), 0)
            request.assert_not_called()
            self.assertFalse((root / "new-run").exists())

    def test_generation_requires_explicit_legacy_opt_in(self):
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as exc:
            legacy.main([])
        self.assertEqual(exc.exception.code, 2)

    def test_main_records_completed_rows_and_resumes_without_calls(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "math.json"
            source.write_text(json.dumps([{"problem": "One?", "answer": "1"}]))
            argv = ["--root", tmp, "--domains", "math", "--math-file", "math.json",
                    "--output-dir", "run", "--samples-per-domain", "1", "--allow-legacy-pilot"]
            row_split = legacy.load_math_rows(source, 42, "train")[0]
            if not row_split:
                for split in ("validation", "test"):
                    if legacy.load_math_rows(source, 42, split)[0]:
                        argv += ["--split", split]
                        break
            with patch.object(actor, "post_json", return_value=response()) as request, \
                 contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(legacy.main(argv), 0)
                self.assertEqual(legacy.main(argv), 0)
                self.assertEqual(request.call_count, 1)
            rows = list(read_jsonl(root / "run" / "trajectories.jsonl"))
            self.assertEqual(len(rows), 1)
            self.assertFalse(rows[0]["valid_for_training"])
            self.assertEqual(rows[0]["outcome"]["case_type"], "initial_success")
            self.assertEqual(audit_events(root / "run" / "events.jsonl")["sample_attempts_without_terminal_event"], 0)
            self.assertEqual(audit_trajectories(root / "run" / "trajectories.jsonl")["total_calls"], 1)

    def test_partial_run_has_nonzero_exit_and_failure_journal(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            row = {"sample_id": "math:a", "group_id": "math:a", "domain": "math",
                   "split": "train", "problem": "One?", "gold": "1"}
            args = legacy.build_parser().parse_args([])
            with patch.object(actor, "post_json", return_value=response()), \
                 patch.object(legacy, "verify", side_effect=ValueError("broken")), \
                 contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(legacy.run_samples(args, [row], {"source_sha256": {}}, root), 1)
            self.assertEqual(json.loads((root / "manifest.json").read_text())["status"], "partial")
            self.assertEqual(list(read_jsonl(root / "events.jsonl"))[-1]["event"], "sample.failed")
            self.assertFalse((root / "trajectories.jsonl").exists())

    def test_audit_counts_old_truncations_without_claiming_eligibility(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "trajectories.jsonl"
            row = {"sample_id": "a", "domain": "code", "outcome": {"label": 0, "case_type": "shared_failure"},
                   "initial": {"output": "reasoning", "finish_reason": "length",
                               "verifier": {"failure_type": "SyntaxError"}}}
            path.write_text(json.dumps(row) + "\n")
            report = audit_trajectories(path)
            self.assertEqual(report["truncation_rate"], 1.0)
            self.assertEqual(report["unknown_training_eligibility"], 1)
            self.assertEqual(report["explicit_training_eligible"], 0)


if __name__ == "__main__":
    unittest.main()
