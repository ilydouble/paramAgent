import ast
from contextlib import redirect_stdout
from copy import deepcopy
import io
import json
import os
from pathlib import Path
import pickle
import random
import re
import tempfile
from time import time
from types import SimpleNamespace
import unittest


ROOT = Path(__file__).resolve().parents[1]


def functions_from_source(relative, names, namespace):
    path = ROOT / relative
    tree = ast.parse(path.read_text())
    nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[future, *nodes], type_ignores=[])),
                 str(path), "exec"), namespace)
    return namespace


class MathExtractionTests(unittest.TestCase):
    def setUp(self):
        self.extract = functions_from_source(
            "executors/Math_executor.py",
            {"_extract_final_answer", "_clean_value", "_unwrap_boxed", "_find_boxed"},
            {"re": re})["_extract_final_answer"]

    def test_markdown_titles_accept_blank_lines_and_do_not_require_colon(self):
        for answer in ("### Answer\n8.6", "### Answer\n\n10.3", "## Final Answer:\n8.6",
                       "**Answer:**\n8.6", "**`Answer:`** 8.6", "**Answer**: 8.6"):
            with self.subTest(answer=answer):
                self.assertEqual(self.extract(answer), "10.3" if "10.3" in answer else "8.6")

    def test_last_answer_wins_across_format_variants(self):
        self.assertEqual(self.extract("**Answer:**\nThe value is $\\frac{3}{2}$.\n\nAnswer: 3/2"), "3/2")
        self.assertEqual(self.extract("**Answer:**\nEach meerkat gets 36 nights of sleep.\n\nAnswer: 36"), "36")
        self.assertEqual(self.extract("Answer: 1\n\n### Answer\n2"), "2")

    def test_no_answer_is_guessed_from_prose_or_quoted_instructions(self):
        for text in ('The instruction says "Answer: <final answer>".',
                     'We previously wrote Answer: 1 but that was wrong.',
                     'Answering requires more thought.', '### Answer\n\n', ''):
            with self.subTest(text=text):
                self.assertIsNone(self.extract(text))

    def test_nested_boxes_and_original_numeric_notation_are_preserved(self):
        self.assertEqual(self.extract(r'**`Answer:`** $\boxed{\frac{1}{36}}$'), r'\frac{1}{36}')
        self.assertEqual(self.extract(r'Work: \boxed{\frac{3}{2}}'), r'\frac{3}{2}')
        self.assertEqual(self.extract('Answer: x'), 'x')


class MathRunnerErrorTests(unittest.TestCase):
    def run_cases(self, failures):
        def evaluate(answer, gold, timeout):
            if answer in failures:
                raise ValueError("Unresolved math scoring")
            return True

        def write_rows(path, rows, append=False):
            with open(path, "a" if append else "w") as stream:
                for row in rows:
                    stream.write(json.dumps(row) + "\n")

        def read_rows(path):
            return [json.loads(line) for line in Path(path).read_text().splitlines() if line]

        namespace = functions_from_source("math/dot_Math_parametric_with_bank.py",
            {"run_dot", "_math_prompt_string"}, {
                "executor_factory": lambda **kwargs: SimpleNamespace(evaluate=evaluate),
                "generator_factory": lambda **kwargs: SimpleNamespace(func_impl=lambda question, *args, **kwargs: question),
                "model_factory": lambda name: SimpleNamespace(), "make_printv": lambda verbose: lambda *args: None,
                "resume_success_count_from_log": lambda path: 0,
                "enumerate_resume": lambda rows, path: enumerate(rows),
                "colored": lambda value, *args: value, "log_detail": lambda *args: None,
                "write_jsonl": write_rows, "read_jsonl": read_rows, "os": os, "pkl": pickle,
                "random": random, "time": time, "deepcopy": deepcopy,
                "get_openai_embedding": lambda values: [[1.0]],
                "gpt_usage": lambda **kwargs: {"cost": 0, "completion_tokens": 0, "prompt_tokens": 0},
            })
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            path = Path(directory) / "results.jsonl"
            namespace["run_dot"](
                dataset=[{"task_id": str(index), "problem": question, "answer": "gold"}
                         for index, question in enumerate(("first", "second"))],
                model_name="fake", language="math", max_iters=1, pass_at_k=1,
                log_path=str(path), verbose=False,
                Math_agent=SimpleNamespace(generate=lambda *args, **kwargs: "pitfalls"))
            self.assertTrue((Path(directory) / "mem_bank.pkl").exists())
            return read_rows(path), read_rows(Path(directory) / "first_stage_log.jsonl")

    def test_first_item_error_does_not_hide_original_error_or_stop_second_item(self):
        rows, first_stage = self.run_cases({"first"})
        self.assertEqual(len(rows), 2)
        self.assertIsNone(rows[0]["is_solved"])
        self.assertEqual(rows[0]["runtime_error"], "ValueError: Unresolved math scoring")
        self.assertGreaterEqual(rows[0]["runtime"], 0)
        self.assertTrue(rows[1]["is_solved"])
        self.assertEqual(first_stage, rows)

    def test_all_items_error_still_preserves_records_without_missing_bank_crash(self):
        rows, first_stage = self.run_cases({"first", "second"})
        self.assertEqual(len(rows), 2)
        self.assertTrue(all(row.get("runtime_error") for row in rows))
        self.assertTrue(all(row["is_solved"] is None for row in rows))
        self.assertEqual(first_stage, rows)


if __name__ == "__main__":
    unittest.main()
