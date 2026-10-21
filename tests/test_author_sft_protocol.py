import argparse
import ast
import hashlib
import json
from pathlib import Path
import re
import sys
import tempfile
import unittest
from unittest.mock import patch

from shared.parammem_prompts import AUTHOR_SOURCE_SHA256, ROOT, author_constants, memory_messages, tokenize_sft


def source_functions(path, names):
    tree = ast.parse(path.read_text())
    nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    namespace = {"argparse": argparse, "json": json, "re": re}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)
    return namespace


class Processor:
    def apply_chat_template(self, messages, **kwargs):
        self.messages, self.options = messages, kwargs
        return "rendered prompt"


class Tokenizer:
    pad_token_id = 0
    eos_token = "<EOS>"

    def __call__(self, text, **kwargs):
        return {"input_ids": [21, 0] if text == "guidance<EOS>" else [11, 12]}


class AuthorSFTTests(unittest.TestCase):
    def test_author_sources_are_pinned_without_importing_training_dependencies(self):
        for relative, expected in AUTHOR_SOURCE_SHA256.items():
            self.assertEqual(hashlib.sha256((ROOT / relative).read_bytes()).hexdigest(), expected)

    def test_training_and_inference_templates_keep_authors_distinct_demonstrations(self):
        for domain in ("code", "math"):
            train = memory_messages(domain, " question ", stage="train")
            inference = memory_messages(domain, " question ")
            self.assertNotIn("[Example]", train[0]["content"])
            self.assertIn("[Example]", inference[0]["content"])
            self.assertEqual(train[1], inference[1])
            self.assertTrue(train[1]["content"].endswith("question"))
        for stage in ("train", "inference"):
            system, examples = author_constants("qa", stage)
            messages = memory_messages("qa", "question", stage)
            self.assertEqual(messages[0]["content"], system)
            self.assertEqual(messages[1]["content"], f"Here are some examples:{examples}\n\n[Question]: question")
            self.assertIn("Turchin’s", examples)
            self.assertIn("question.  Use", system)

    def test_qwen_wrapper_is_explicit_without_changing_author_instruction_text(self):
        processor = Processor()
        tokenize_sft(processor, Tokenizer(), "math", "question", "guidance", 6)
        self.assertEqual(processor.messages, memory_messages("math", "question", "train"))
        self.assertEqual(processor.options, {"tokenize": False, "add_generation_prompt": True, "enable_thinking": False})

    def test_prompt_tokens_receive_loss_but_eos_padding_does_not(self):
        record = tokenize_sft(Processor(), Tokenizer(), "code", "question", "guidance", 6)
        self.assertEqual(record["input_ids"], [11, 12, 21, 0, 0, 0])
        self.assertEqual(record["labels"], [11, 12, 21, -100, -100, -100])
        self.assertEqual(record["attention_mask"], [1, 1, 1, 1, 0, 0])

    def test_dynamic_padding_and_author_length_truncation_keep_full_sequence_loss(self):
        record = tokenize_sft(Processor(), Tokenizer(), "qa", "question", "guidance", 3, True)
        self.assertEqual(record["input_ids"], [11, 12, 21])
        self.assertEqual(record["labels"], [11, 12, 21])
        self.assertEqual(record["length"], 3)
        with self.assertRaises(ValueError):
            tokenize_sft(Processor(), Tokenizer(), "qa", "question", "guidance", 0)

    def test_code_supervision_prefers_full_author_target_instead_of_pitfalls_only(self):
        functions = source_functions(ROOT / "code/LoRA_Qwen35_Code_4090.py", {"_normalize_func_sign", "load_code_examples"})
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "samples.json"
            samples = [
                {"func_sign": "def example():", "raw_texts": "full plural", "raw_text": "full singular", "pitfalls": "short"},
                {"func_sign": "def example():", "raw_text": "full singular", "pitfalls": "short"},
                {"func_sign": "def example():", "pitfalls": "short"},
            ]
            path.write_text(json.dumps(samples))
            self.assertEqual([row["answer"] for row in functions["load_code_examples"](str(path))],
                             ["full plural", "full singular", "short"])

    def test_training_defaults_match_author_launches_with_equivalent_global_batch(self):
        for domain, label in (("code", "Code"), ("math", "Math"), ("qa", "QA")):
            path = ROOT / domain / f"LoRA_Qwen35_{label}_4090.py"
            parse_args = source_functions(path, {"parse_args"})["parse_args"]
            with patch.object(sys, "argv", [str(path)]):
                args = parse_args()
            self.assertEqual((args.lora_r, args.lora_alpha, args.val_ratio, args.seed), (64, 32, 0.15, 1))
            self.assertEqual(args.num_epochs, 2 if domain == "math" else 3)
            self.assertEqual(args.lr, 1.5e-5 if domain == "math" else 3e-5)
            self.assertEqual(args.per_device_batch_size * args.grad_accum_steps, 128 if domain == "math" else 96)
            self.assertTrue(args.output_dir.endswith(f"{domain}-author-sft"))
            text = path.read_text()
            self.assertIn("tokenizer.pad_token = tokenizer.eos_token", text)
            self.assertIn("return tokenize_sft(", text)

    def test_qa_accepts_author_json_list_as_well_as_existing_jsonl(self):
        loader = source_functions(ROOT / "qa/LoRA_Qwen35_QA_4090.py", {"load_qa_examples"})["load_qa_examples"]
        rows = [{"question": "question", "decomposition": "guidance"}] * 2
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "qa.json"
            for text in (json.dumps(rows, indent=2), "\n".join(json.dumps(row) for row in rows)):
                path.write_text(text)
                self.assertEqual(loader(str(path)), rows)

    def test_actor_prompt_literals_match_the_pinned_author_version(self):
        expected = {
            "MATH_SIMPLE_CHAT_INSTRUCTION": "d6f033314e6a8afdec0c02fe494f8378ca0c4a127c0d902e2f8927d54538a6a3",
            "MATH_REFLEXION_CHAT_INSTRUCTION": "154319e835e0a918a8e82558c9be72c1351daae24e0c48696410ae3c90327f65",
            "MATH_REFLEXION_CHAT_INSTRUCTION_PARAMETRIC": "c090180919a3c58719d94a833142afd14a8180838a79796ece383dabe521e40c",
            "math_pitfalls_instruction": "2e732fdbc424d6c7c573d1c8e767d83986967a31e92ef8da5812d48b6a51af92",
        }
        observed = {}
        code_instructions = set()
        for relative in ("generators/MathQA_generate.py", "generators/generator_utils.py"):
            tree = ast.parse((ROOT / relative).read_text())
            for node in ast.walk(tree):
                if not isinstance(node, ast.Assign):
                    continue
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id in {*expected, "mistake_instructions"}:
                        value = ast.literal_eval(node.value)
                        digest = hashlib.sha256(value.encode()).hexdigest()
                        if target.id == "mistake_instructions":
                            code_instructions.add(digest)
                        else:
                            observed[target.id] = digest
        self.assertEqual(observed, expected)
        self.assertEqual(code_instructions, {"5cf3e9b9c0366936a880dc83a6368fd1df47c496173f54457fca0526a87c981b",
                                             "10b7945030616e202e72f3f3e7c20b4db93b4592ef8618c6f109f11b9f2eff2a"})

    def test_inference_keeps_full_generated_code_guidance(self):
        text = (ROOT / "code/LoRA_Qwen35_Code_4090_inference.py").read_text()
        self.assertNotIn("_keep_only_pitfalls", text)
        self.assertIn("generation.strip() for generation in gens", text)


if __name__ == "__main__":
    unittest.main()
