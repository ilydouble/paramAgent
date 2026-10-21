import importlib.util
import ast
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch
from types import SimpleNamespace
from typing import Optional

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/run_parammem_pilot.py"
spec = importlib.util.spec_from_file_location("parammem_pilot", SCRIPT)
pilot = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pilot)


class PilotTests(unittest.TestCase):
    def test_explicit_small_frozen_sample_preserves_count_and_uniqueness_guards(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "input.jsonl"
            path.write_text(json.dumps({"task_id": "new-question"}) + "\n")
            self.assertEqual(len(pilot.read_frozen_rows(path, 1)), 1)
            with self.assertRaises(ValueError):
                pilot.read_frozen_rows(path)
            with self.assertRaises(ValueError):
                pilot.read_frozen_rows(path, 0)
            path.write_text(path.read_text() * 2)
            with self.assertRaises(ValueError):
                pilot.read_frozen_rows(path, 2)

    def budget_args(self, **overrides):
        return SimpleNamespace(**{"actor_url": "http://127.0.0.1:8000/v1", "actor_context_tokens": 16384,
                                  "math_answer_tokens": 8192, "context_reserve_tokens": 256, **overrides})

    def test_exact_budget_uses_server_tokenization_and_caches_counts(self):
        actor = pilot.LocalActor(self.budget_args(), None)
        messages = [SimpleNamespace(role="user", content="question")]
        with patch.object(pilot.urllib.request, "urlopen", return_value=io.StringIO(
                json.dumps({"count": 863, "max_model_len": 16384}))) as request:
            self.assertEqual(actor.completion_budget(messages, 8192), 8192)
            self.assertEqual(actor.completion_budget(messages, 4096), 4096)
            self.assertEqual(request.call_count, 1)
            self.assertEqual(request.call_args.args[0].full_url, "http://127.0.0.1:8000/tokenize")
            payload = json.loads(request.call_args.args[0].data)
            self.assertEqual(payload["messages"], [{"role": "user", "content": "question"}])
            self.assertEqual(payload["chat_template_kwargs"], {"enable_thinking": False})

    def test_long_prompt_reduces_output_and_invalid_context_fails_closed(self):
        messages = [{"role": "user", "content": "question"}]
        for metadata, expected in [({"count": 12000, "max_model_len": 16384}, 4128),
                                   ({"count": 16128, "max_model_len": 16384}, None),
                                   ({"count": 863, "max_model_len": 8192}, None),
                                   ({"count": -1, "max_model_len": 16384}, None)]:
            actor = pilot.LocalActor(self.budget_args(), None)
            with self.subTest(metadata=metadata), patch.object(pilot.urllib.request, "urlopen",
                    return_value=io.StringIO(json.dumps(metadata))):
                if expected is None:
                    with self.assertRaises(ValueError):
                        actor.completion_budget(messages, 8192)
                else:
                    self.assertEqual(actor.completion_budget(messages, 8192), expected)

    def test_actor_sends_expanded_budget_and_logs_actual_prompt_count(self):
        journal = SimpleNamespace(cache={}, context={}, write=Mock(),
                                  sample_request=lambda kind, payload: payload)
        actor = pilot.LocalActor(self.budget_args(seed=42), journal)
        responses = [io.StringIO(json.dumps({"count": 863, "max_model_len": 16384})),
                     io.StringIO(json.dumps({"choices": [{"index": 0, "message": {"content": "Answer: 4"},
                                                          "finish_reason": "stop"}]}))]
        with patch.object(pilot.urllib.request, "urlopen", side_effect=responses) as request:
            output = actor.generate_chat([SimpleNamespace(role="user", content="question")], max_tokens=8192)
            self.assertEqual(output, "Answer: 4")
            self.assertEqual(json.loads(request.call_args_list[1].args[0].data)["max_tokens"], 8192)
        budget = journal.write.call_args_list[0].args[0]
        self.assertEqual((budget["prompt_tokens"], budget["max_tokens"], budget["context_tokens"]),
                         (863, 8192, 16384))

    def test_budget_settings_are_explicit_and_validated(self):
        with patch.dict(pilot.os.environ, {}, clear=False):
            pilot.configure_generation_budget(self.budget_args())
            self.assertEqual(pilot.os.environ["VLLM_MAX_MODEL_LEN"], "16384")
            self.assertEqual(pilot.os.environ["MATH_MAX_ANSWER_TOKENS"], "8192")
            for overrides in ({"math_answer_tokens": 0}, {"context_reserve_tokens": -1},
                              {"actor_context_tokens": 8192}, {"actor_context_tokens": 0}):
                with self.subTest(overrides=overrides), self.assertRaises(ValueError):
                    pilot.configure_generation_budget(self.budget_args(**overrides))

    def test_math_helper_uses_runtime_cap_and_exact_actor_budget(self):
        tree = ast.parse((SCRIPT.parents[1] / "generators/generator_utils.py").read_text())
        function = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                        and node.name == "_answer_max_tokens")
        namespace = {"List": list, "Optional": Optional, "Message": object,
                     "MAX_ANSWER_TOKENS": 8192, "MAX_CONTEXT_TOKENS": 16384, "RESERVED_TOKENS": 256}
        exec(compile(ast.Module(body=[function], type_ignores=[]), "budget_helper", "exec"), namespace)
        budget = namespace["_answer_max_tokens"]
        messages = [SimpleNamespace(content="short")]
        self.assertEqual(budget(messages), 8192)
        namespace["MAX_ANSWER_TOKENS"] = 4096
        self.assertEqual(budget(messages), 4096)
        model = SimpleNamespace(completion_budget=lambda messages, cap: min(cap, 3000))
        self.assertEqual(budget(messages, model=model), 3000)
        with self.assertRaises(ValueError):
            budget([SimpleNamespace(content="x" * 50000)])
        runner = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                      and node.name == "generic_generate_mathqa_impl")
        calls = [node for node in ast.walk(runner) if isinstance(node, ast.Call)
                 and isinstance(node.func, ast.Name) and node.func.id == "_answer_max_tokens"]
        self.assertTrue(calls)
        self.assertTrue(all(any(keyword.arg == "model" for keyword in call.keywords) for call in calls))

    def test_explicit_dpo_checks_hash_and_exact_sft_base(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            adapter = root / "new-dpo"
            adapter.mkdir()
            weights = adapter / "adapter_model.safetensors"
            weights.write_bytes(b"newly trained adapter")
            config = adapter / "adapter_config.json"
            base = root / "models" / pilot.BASES["code"]
            config.write_text(json.dumps({"base_model_name_or_path": str(base)}))
            args = SimpleNamespace(domain="code", method="parammem-dpo", model_root=root,
                                   dpo_adapter_path=adapter, expected_dpo_sha256=pilot.file_digest(weights))
            self.assertEqual(pilot.resolve_dpo_adapter(args), (adapter.resolve(), args.expected_dpo_sha256))
            config.write_text(json.dumps({"base_model_name_or_path": str(root / "other" / base.name)}))
            with self.assertRaisesRegex(ValueError, "base mismatch"):
                pilot.resolve_dpo_adapter(args)
            config.write_text(json.dumps({"base_model_name_or_path": str(base)}))
            weights.write_bytes(b"different checkpoint")
            with self.assertRaisesRegex(ValueError, "SHA256"):
                pilot.resolve_dpo_adapter(args)

    def test_explicit_dpo_requires_hash_and_dpo_method(self):
        args = SimpleNamespace(domain="code", method="parammem-dpo", model_root=Path("/tmp"),
                               dpo_adapter_path=Path("/tmp/new-dpo"), expected_dpo_sha256=None)
        with self.assertRaisesRegex(ValueError, "together"):
            pilot.resolve_dpo_adapter(args)
        args.expected_dpo_sha256 = "expected"
        args.method = "parammem"
        with self.assertRaisesRegex(ValueError, "require method"):
            pilot.resolve_dpo_adapter(args)

    def test_repeated_samples_have_distinct_reproducible_seeds(self):
        with tempfile.TemporaryDirectory() as directory:
            journal = pilot.Journal(Path(directory))
            request = {"seed": 42, "prompt": "same question", "temperature": 1.0}
            self.assertEqual(journal.sample_request("memory", request)["seed"], 42)
            self.assertEqual(journal.sample_request("memory", request)["seed"], 43)
            self.assertEqual(journal.sample_request("actor", request)["seed"], 42)
            self.assertEqual(journal.sample_request("memory", {**request, "prompt": "other"})["seed"], 42)
            self.assertEqual(pilot.Journal(Path(directory)).sample_request("memory", request)["seed"], 42)

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

    def test_latex_pi_constant_is_not_an_unknown_variable(self):
        try:
            import sympy
            import antlr4
            from executors.Math_executor import _extract_final_answer
        except ImportError:
            self.skipTest("Requires the server Math evaluation dependencies")
        self.assertTrue(pilot.math_score(r"Answer: 256\pi", r"256 \pi")["success"])
        self.assertFalse(pilot.math_score(r"Answer: 12\pi", r"256 \pi")["success"])
        self.assertIsNone(pilot.math_score(r"Answer: x", "4")["success"])

    def test_journal_recovers_cached_calls_but_not_evaluations(self):
        with tempfile.TemporaryDirectory() as directory:
            journal = pilot.Journal(Path(directory))
            journal.write({"type": "actor", "key": "a", "response": {}})
            journal.write({"type": "evaluation", "result": {"success": False}})
            self.assertEqual(set(pilot.Journal(Path(directory)).cache), {"a"})

    def test_finish_reason_requires_exact_logged_response(self):
        with tempfile.TemporaryDirectory() as directory:
            journal = pilot.Journal(Path(directory))
            choice = {"message": {"content": "unfinished reasoning"}, "finish_reason": "length"}
            journal.cache["a"] = {"type": "actor", "response": {"choices": [choice]}}
            self.assertEqual(journal.actor_finish_reason("unfinished reasoning"), "length")
            self.assertIsNone(journal.actor_finish_reason("other answer"))
            journal.cache["b"] = {"type": "actor", "response": {"choices": [{**choice, "finish_reason": "stop"}]}}
            self.assertIsNone(journal.actor_finish_reason("unfinished reasoning"))

    def test_math_truncation_without_final_is_budget_failure(self):
        try:
            import sympy
            import antlr4
            from executors.Math_executor import _extract_final_answer
        except ImportError:
            self.skipTest("Requires the server Math evaluation dependencies")
        answer = "Let us reason carefully. The possible pairs are (N, P) or (P,"
        self.assertFalse(pilot.math_score(answer, "4", finish_reason="length")["success"])
        self.assertEqual(pilot.math_score(answer, "4", finish_reason="length")["status"],
                         "budget_exhausted_without_final")
        self.assertIsNone(pilot.math_score(answer, "4", finish_reason="stop")["success"])
        self.assertIsNone(pilot.math_score(answer, "bad gold", finish_reason="length")["success"])
        self.assertIsNone(pilot.math_score("Answer: x", "4", finish_reason="length")["success"])
        self.assertIsNone(pilot.math_score("x", "4", finish_reason="length")["success"])
        self.assertTrue(pilot.math_score("4", "4", finish_reason="length")["success"])


if __name__ == "__main__":
    unittest.main()
