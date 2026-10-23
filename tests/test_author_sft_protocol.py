import argparse
import ast
import hashlib
import json
from pathlib import Path
import re
import sys
import tempfile
import unittest
from types import SimpleNamespace
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
    def test_preserve_eos_masks_only_actual_padding_in_all_domains(self):
        for domain in ("code", "math", "qa"):
            record = tokenize_sft(Processor(), Tokenizer(), domain, "question", "guidance", 6,
                                  preserve_eos=True)
            self.assertEqual(record["input_ids"], [11, 12, 21, 0, 0, 0])
            self.assertEqual(record["labels"], [11, 12, 21, 0, -100, -100])
            self.assertEqual(record["attention_mask"], [1, 1, 1, 1, 0, 0])

    def test_preserve_eos_dynamic_and_truncation_do_not_invent_terminal_tokens(self):
        for dynamic in (False, True):
            for limit, expected in ((3, [11, 12, 21]), (4, [11, 12, 21, 0])):
                record = tokenize_sft(Processor(), Tokenizer(), "qa", "question", "guidance", limit,
                                      dynamic_padding=dynamic, preserve_eos=True)
                self.assertEqual(record["labels"], expected)
                self.assertEqual(record["input_ids"], expected)
                if dynamic:
                    self.assertEqual(record["length"], limit)

    def test_preserve_eos_also_keeps_real_chat_turn_endings_in_prompt(self):
        tokenizer = lambda text, **kwargs: {"input_ids": [11, 0, 12] if text == "rendered prompt" else [21, 0]}
        tokenizer.pad_token_id = 0
        tokenizer.eos_token = "<EOS>"
        record = tokenize_sft(Processor(), tokenizer, "qa", "question", "guidance", 6, preserve_eos=True)
        self.assertEqual(record["labels"], [11, 0, 12, 21, 0, -100])

    def test_training_entrypoints_forward_opt_in_flag_and_dynamic_collators_keep_eos(self):
        for domain, label, question, answer in (("code", "Code", "func_sign", "answer"),
                                                ("math", "Math", "problem", "pitfalls"),
                                                ("qa", "QA", "question", "decomposition")):
            path = ROOT / domain / f"LoRA_Qwen35_{label}_4090.py"
            parse_args = source_functions(path, {"parse_args"})["parse_args"]
            with patch.object(sys, "argv", [str(path), "--preserve_eos", "--dynamic_padding"]):
                args = parse_args()
            self.assertTrue(args.preserve_eos)
            args.max_seq_len = 6
            tree = ast.parse(path.read_text())
            tokenize = next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)
                            and node.name == "tokenize")
            collator = next(node for node in tree.body if isinstance(node, ast.ClassDef)
                            and node.name == "DynamicCausalLMCollator")
            future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
            namespace = {"args": args, "processor": Processor(), "tokenizer": Tokenizer(),
                         "tokenize_sft": tokenize_sft,
                         "torch": SimpleNamespace(long="long", tensor=lambda values, **kwargs: values)}
            exec(compile(ast.fix_missing_locations(ast.Module(body=[future, tokenize, collator], type_ignores=[])),
                         str(path), "exec"), namespace)
            record = namespace["tokenize"]({question: "question", answer: "guidance"})
            self.assertEqual(record["labels"], [11, 12, 21, 0])
            short = {"input_ids": [21, 0], "attention_mask": [1, 1], "labels": [21, 0]}
            batch = namespace["DynamicCausalLMCollator"](0)([record, short])
            self.assertEqual(batch["labels"], [[11, 12, 21, 0, -100, -100, -100, -100],
                                               [21, 0, -100, -100, -100, -100, -100, -100]])
            self.assertEqual(batch["attention_mask"][1], [1, 1, 0, 0, 0, 0, 0, 0])

    def test_independent_inference_entrypoints_use_author_inference_messages(self):
        for domain, label in (("code", "Code"), ("math", "Math"), ("qa", "QA")):
            path = ROOT / domain / f"LoRA_Qwen35_{label}_4090_inference.py"
            tree = ast.parse(path.read_text())
            function = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                            and node.name == "format_prompt")
            future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
            namespace = {"memory_messages": memory_messages}
            exec(compile(ast.fix_missing_locations(ast.Module(body=[future, function], type_ignores=[])),
                         str(path), "exec"), namespace)
            processor = Processor()
            self.assertEqual(namespace["format_prompt"](processor, " question "), "rendered prompt")
            self.assertEqual(processor.messages, memory_messages(domain, "question", "inference"))
            self.assertEqual(processor.options,
                             {"tokenize": False, "add_generation_prompt": True, "enable_thinking": False})

    def test_actor_uses_restored_instructions_in_actual_messages(self):
        path = ROOT / "generators/generator_utils.py"
        tree = ast.parse(path.read_text())
        functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)
                     and node.name in {"generic_generate_mathqa_impl", "generic_generate_func_impl",
                                       "generic_generate_multihopqa_impl"}]
        future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
        namespace = {"Message": SimpleNamespace, "_answer_max_tokens": lambda *args, **kwargs: 8192,
                     "print_messages": lambda *args: None, "print_generated_func_body": lambda *args: None}
        exec(compile(ast.fix_missing_locations(ast.Module(body=[future, *functions], type_ignores=[])),
                     str(path), "exec"), namespace)
        model = SimpleNamespace(is_chat=True)
        requests = []
        model.generate_chat = lambda **kwargs: requests.append(kwargs) or "answer"
        namespace["generic_generate_func_impl"](
            func_sig="def example():", model=model, strategy="simple", prev_func_impl=None,
            feedback=None, self_reflection=None, num_comps=1, temperature=0.1,
            reflexion_chat_instruction="unused", reflexion_few_shot="unused",
            simple_chat_instruction="unused", reflexion_completion_instruction="unused",
            simple_completion_instruction="unused", code_block_instruction="code block",
            parse_code_block=lambda value: value, add_code_block=lambda value: value,
            mistake_insights="author guidance including flawed code")
        self.assertIn("several flawed implementation examples", requests[-1]["messages"][0].content)
        self.assertEqual(requests[-1]["messages"][1].content,
                         "author guidance including flawed code\n [func signature]: def example():")
        namespace["generic_generate_mathqa_impl"](
            question="question", model=model, strategy="simple", prev_answers=None,
            feedback=None, self_reflection=None, num_comps=1, temperature=0.1,
            simple_chat_instruction="unused", reflexion_chat_instruction="unused",
            simple_completion_instruction="unused", reflexion_completion_instruction="unused",
            mistake_insights="author pitfalls")
        messages = requests[-1]["messages"]
        self.assertEqual(hashlib.sha256(messages[0].content.encode()).hexdigest(),
                         "2e732fdbc424d6c7c573d1c8e767d83986967a31e92ef8da5812d48b6a51af92")
        self.assertEqual(messages[1].content, "[question]: question\n\n[mistake insights]:\nauthor pitfalls")
        namespace["generic_generate_multihopqa_impl"](
            question="question", context="supporting passages", model=model, strategy="simple",
            prev_answers=None, feedback=None, self_reflection=None, num_comps=1, temperature=0.1,
            simple_chat_instruction="unused", reflexion_chat_instruction="unused",
            simple_completion_instruction="unused", reflexion_completion_instruction="unused",
            question_decomposition="author decomposition")
        messages = requests[-1]["messages"]
        self.assertEqual(messages[0].content,
                         "You will be given a multi-hop question, with some context sentences, some of them are relevant to the question, some are not. "
                         "You will be provided with some extracted insights about the queston to facilitate your understanding of the question, and possibly some demonstrations from similar questions. "
                         "Using the information provided, write a short answer (a phrase or word) to the question. "
                         " Return **only** the answer—no other texts such as explanations ")
        self.assertEqual(messages[1].content,
                         "[question]: question\n\n[question intent decomposition]:\nauthor decomposition"
                         "\n\n[context]:\nsupporting passages")

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
            self.assertFalse(args.preserve_eos)
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
