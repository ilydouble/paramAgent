# python code/eval_pitfalls_v2.py \
#     --model Meta-Llama-3.1-8B-Instruct \
#     --api_base http://localhost:8000/v1 \
#     --input_file dataset/code/eval_results/apps_baseline_w.jsonl

from __future__ import annotations

import argparse
import json
import multiprocessing
import os
import re
import sys
sys.set_int_max_str_digits(0)  # allow huge integers in test cases
import time
import traceback
import types
from typing import Dict, List, Optional
import numpy as np

# ── Shim pyext so testing_util can be imported ──────────────────────────
class _RuntimeModule:
    @staticmethod
    def from_string(name: str, filename: str, source: str, globals_=None):
        mod = types.ModuleType(name)
        if globals_ is not None:
            mod.__dict__.update(globals_)
        code = compile(source, filename, "exec")
        exec(code, mod.__dict__)
        return mod

class _PyextShim:
    RuntimeModule = _RuntimeModule

sys.modules["pyext"] = _PyextShim()
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "apps", "eval"))
import testing_util as test_util

# ── API client ───────────────────────────────────────────────────────────
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_random_exponential

YUNWU_BASE_URL = "https://api.openlux.ai/v1"

DEFAULT_API_KEY = os.getenv(
    "OPENAI_API_KEY",
    "<REDACTED_HARDCODED_KEY>",
)

GEN_RETRIES = 3
MAX_CONSECUTIVE_API_FAILURES = 5
TIMEOUT = 10   # seconds per problem test execution

# ── System prompts ───────────────────────────────────────────────────────
SYSTEM_PROMPT_STANDARD = (
    "You are an expert Python programmer. You will be given a function signature "
    "and its docstring. Write a complete, correct Python implementation. "
    "Output ONLY the Python code in a markdown code block.\n\n"
    "```python\n"
    "def example():\n    return 42\n"
    "```"
)

SYSTEM_PROMPT_PITFALLS = (
    "You are an expert Python programmer. You will be given a function signature with its docstring, "
    "and a list of common pitfalls. "
    "Study the pitfalls carefully and write "
    "a CORRECT implementation that AVOIDS all the listed mistakes.\n\n"
    "Output ONLY the Python code in a markdown code block:\n"
    "```python\n"
    "def example():\n    return 42\n"
    "```"
)


class YunwuModel:
    """Thin wrapper around an OpenAI-compatible endpoint (yunwu.ai, local vLLM, ...)."""

    def __init__(self, model_name: str, api_key: str = DEFAULT_API_KEY,
                 base_url: str = YUNWU_BASE_URL):
        self.name = model_name
        self._client = OpenAI(base_url=base_url, api_key=api_key, timeout=120.0)

    @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(GEN_RETRIES))
    def generate(self, system: str, user: str, max_tokens: int = 2048,
                 temperature: float = 0.2) -> str:
        resp = self._client.chat.completions.create(
            model=self.name,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            max_tokens=max_tokens,
            temperature=temperature,
        )
        content = resp.choices[0].message.content
        if not content:
            print(f"  ⚠ API empty response (finish_reason={resp.choices[0].finish_reason})")
        return content or ""


class LocalModel:
    """Local HuggingFace model loaded from disk (e.g. Meta-Llama-3.1-8B-Instruct)."""

    def __init__(self, model_path: str, device_override: Optional[int] = None):
        import torch
        import transformers

        self.name = model_path
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        device_map = {"": device_override if device_override is not None else 0}

        print(f"Loading tokenizer from {model_path}...")
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(model_path)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"

        print(f"Loading model from {model_path}...")
        self.model = transformers.AutoModelForCausalLM.from_pretrained(
            model_path,
            dtype=torch.bfloat16,
            device_map=device_map,
        )
        self.model.eval()
        self.model.config.use_cache = True

        if hasattr(self.model, "hf_device_map"):
            for dev in self.model.hf_device_map.values():
                if isinstance(dev, (str, int)):
                    self._device = torch.device(dev if isinstance(dev, str) else f"cuda:{dev}")
                    break
            else:
                self._device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        elif hasattr(self.model, "device"):
            self._device = torch.device(self.model.device)
        else:
            self._device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        print(f"Model loaded on {self._device}")

    def generate(self, system: str, user: str, max_tokens: int = 2048,
                 temperature: float = 0.2) -> str:
        import torch

        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        inputs = self.tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=8192
        )
        inputs = {k: v.to(self._device) for k, v in inputs.items()}

        with torch.inference_mode():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=0.9,
                top_k=50,
                use_cache=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        # Only decode the NEWLY GENERATED tokens (after the input prompt)
        input_len = inputs["input_ids"].shape[1]
        new_tokens = outputs[0][input_len:]
        raw = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
        return raw.strip()


def create_model(args: argparse.Namespace):
    """Factory: return the right model based on CLI args."""
    if args.use_local:
        return LocalModel(args.local_model_path)
    else:
        return YunwuModel(args.model, args.api_key, args.api_base)


# ── Prompt construction ──────────────────────────────────────────────────
def _clean_func_sign(func_sign: str) -> str:
    func_sign_clean = re.sub(r"^```python\s*\n?", "", func_sign)
    func_sign_clean = re.sub(r"\n?```\s*$", "", func_sign_clean)
    return func_sign_clean.strip()


def _extract_func_name(func_sign: str) -> Optional[str]:
    m = re.search(r"def\s+(\w+)\s*\(", func_sign)
    return m.group(1) if m else None


def _correct_func_name(func_sign: str, target_name: str) -> str:
    return re.sub(r"def\s+\w+\s*\(", f"def {target_name}(", func_sign)


def build_prompt_standard(item: dict, fn_name: Optional[str] = None) -> str:
    """Prompt with function signature only (no question description)."""
    func_sign = item.get("func_sign", "").strip()

    func_sign_clean = _clean_func_sign(func_sign)
    if fn_name and func_sign_clean:
        sig_name = _extract_func_name(func_sign_clean)
        if sig_name and sig_name != fn_name:
            func_sign_clean = _correct_func_name(func_sign_clean, fn_name)

    prompt = func_sign_clean
    prompt += "\n\nWrite the complete Python solution.\n"
    return prompt


def build_prompt_with_pitfalls(item: dict, fn_name: Optional[str] = None) -> str:
    """Prompt with function signature + pitfalls (no question, no flawed impl)."""
    func_sign = item.get("func_sign", "").strip()
    pitfalls = item.get("pitfalls", "").strip()

    func_sign_clean = _clean_func_sign(func_sign)
    if fn_name and func_sign_clean:
        sig_name = _extract_func_name(func_sign_clean)
        if sig_name and sig_name != fn_name:
            func_sign_clean = _correct_func_name(func_sign_clean, fn_name)

    prompt = func_sign_clean

    if pitfalls:
        prompt += "\n\n--- Common Pitfalls & Warnings ---\n"
        prompt += f"\n{pitfalls}\n"
        prompt += "--- End of Pitfalls ---\n\n"

    prompt += "Write a correct Python solution that avoids the pitfalls listed above.\n"
    return prompt


# ── Code extraction ──────────────────────────────────────────────────────
def extract_code(raw: str, fn_name: Optional[str] = None) -> str:
    """Pull Python code from an LLM response — picks the LAST code block.

    If fn_name is provided, renames the top-level function to match it.
    """
    matches = list(re.finditer(r"```(?:python)?\s*\n?(.*?)```", raw, re.DOTALL))
    if matches:
        code = matches[-1].group(1).strip()
    else:
        code = re.sub(r"<\|.*?\|>", "", raw).strip()

    if fn_name and code:
        code = _fix_func_name_in_code(code, fn_name)

    return code


def _fix_func_name_in_code(code: str, target_name: str) -> str:
    """Replace the first 'def xxx(' in code with 'def target_name('."""
    m = re.search(r"\bdef\s+(\w+)\s*\(", code)
    if not m:
        return code
    current_name = m.group(1)
    if current_name == target_name:
        return code
    code = re.sub(r"\b" + re.escape(current_name) + r"\b", target_name, code)
    return code


# ── Code evaluation ──────────────────────────────────────────────────────
def _run_test_in_subprocess(problem: dict, generation: str, timeout: int, debug: bool, early_stop: bool) -> List:
    """Run test_util.run_test with a global timeout.

    Tries multiprocessing first; falls back to direct call when multiprocessing
    is unavailable (e.g. macOS spawn mode).
    """
    def _temp_run(problem, generation, debug, early_stop, result):
        try:
            result.append(test_util.run_test(problem=problem, test=generation,
                                             debug=debug, early_stop=early_stop))
        except Exception as e:
            if debug:
                print(f"Error in _temp_run: {e}")

    try:
        ctx = multiprocessing.get_context("fork")
    except (ValueError, AttributeError):
        ctx = multiprocessing

    try:
        manager = ctx.Manager()
        result = manager.list()
        p = ctx.Process(target=_temp_run, args=(problem, generation, debug, early_stop, result))
        p.start()
        p.join(timeout=timeout + 1)
        if p.is_alive():
            p.kill()
        if not result:
            result = [[-1] * 21]
        return result[0]
    except Exception as mp_err:
        # Fallback: run directly (testing_util.run_test has its own signal.alarm)
        if debug:
            print(f"Multiprocessing unavailable ({mp_err}), running test directly")
        try:
            return test_util.run_test(problem=problem, test=generation,
                                      debug=debug, early_stop=early_stop)
        except Exception as e:
            if debug:
                print(f"Direct run_test error: {e}")
            return [-2]


def evaluate_single(problem: dict, generation: str, timeout: int, debug: bool,
                    early_stop: bool = False) -> List:
    curr_res = [-2]
    try:
        curr_res = _run_test_in_subprocess(problem, generation=generation,
                                           timeout=timeout, debug=debug,
                                           early_stop=early_stop)
        fixed = []
        for e in curr_res:
            if isinstance(e, (np.ndarray,)):
                e = e.item(0)
            if isinstance(e, (np.bool_,)):
                e = bool(e)
            fixed.append(e)
        curr_res = fixed
    except Exception:
        if debug:
            traceback.print_exc()
    finally:
        assert isinstance(curr_res, list)
    return curr_res


def is_all_correct(results: List) -> bool:
    if not results:
        return False
    return all(r is True for r in results)


def get_actual_outputs(code: str, problem: dict, early_stop: bool = False) -> Optional[str]:
    """Re-run the generated code against test inputs and capture actual outputs.

    With early_stop=True, stops at the first failing case (mirroring the judge),
    so the diagnostic never runs more cases than the judge already did.
    """
    io = problem["input_output"]
    fn_name = io.get("fn_name")
    inputs_list = io.get("inputs", [])
    expected_list = io.get("outputs", [])

    # Compile/execute the generated module once, not once per test case.
    ns = {}
    try:
        exec(code, ns)
    except Exception as e:
        return f"  [0] module error: {type(e).__name__}: {e}"

    lines = []
    for idx, (inp, exp) in enumerate(zip(inputs_list, expected_list)):
        failed = False
        try:
            if fn_name:
                func = ns.get(fn_name)
                if func is None:
                    lines.append(f"  [{idx}] fn '{fn_name}' not found in code")
                    failed = True
                else:
                    actual = func(*inp)
                    ok = (actual == exp) or (
                        isinstance(exp, list) and len(exp) == 1 and actual == exp[0]
                    )
                    lines.append(
                        f"  [{idx}] in={inp} → got={actual} expected={exp} "
                        f"{'✓' if ok else '✗'}"
                    )
                    failed = not ok
            else:
                lines.append(f"  [{idx}] (stdin mode, expected={exp})")
                # stdin mode cannot be re-run here; never treated as a failure signal
        except Exception as e:
            lines.append(f"  [{idx}] error: {type(e).__name__}: {e}")
            failed = True
        if early_stop and failed:
            break

    return "\n".join(lines) if lines else None


# ── Utilities ────────────────────────────────────────────────────────────
def load_json_or_jsonl(path: str) -> list:
    with open(path, encoding="utf-8") as f:
        first = f.read(1)
        f.seek(0)
        if first == "[":
            return json.load(f)
        try:
            obj = json.load(f)
            return [obj]
        except json.JSONDecodeError:
            f.seek(0)
            items = []
            for line in f:
                line = line.strip()
                if line:
                    items.append(json.loads(line))
            return items


def load_processed_questions(right_path: str, wrong_path: str) -> set:
    """Return set of question texts already present in right/wrong output files."""
    seen = set()
    for p in (right_path, wrong_path):
        if os.path.exists(p):
            with open(p, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            rec = json.loads(line)
                            q = rec.get("question", "").strip()
                            if q:
                                seen.add(q)
                        except json.JSONDecodeError:
                            pass
    return seen


# ── Main ──────────────────────────────────────────────────────────────────
def run_evaluation(args: argparse.Namespace):
    """Load the model ONCE, then evaluate every requested input file in order.

    Multiple --input_file values are processed sequentially in this same
    process, so the local model (which alone fills most of a 24GB 4090) is
    loaded a single time instead of once per file.
    """
    model = create_model(args)
    if args.use_local:
        print(f"Model: {args.local_model_path}  (local, loaded once)")
    else:
        print(f"Model: {args.model}  |  endpoint: {args.api_base}")

    input_files = list(args.input_file) if args.input_file else [None]
    for input_file in input_files:
        run_single_file(args, model, input_file)


def run_single_file(args: argparse.Namespace, model, input_file):
    # 1. Load data (model is loaded once in run_evaluation, shared across files)
    apps_original_path = os.path.join(args.data_dir, "dataset", "code", "train", "apps_original.json")
    if input_file:
        apps_filename = os.path.basename(input_file)
        apps_path = os.path.abspath(input_file)
    else:
        apps_filename = "apps_original.json"
        apps_path = apps_original_path
    mapping_path = os.path.join(args.data_dir, "dataset", "code", "train", "apps_to_meta_mapping.json")
    meta_train_dir = os.path.join(args.data_dir, "dataset", "code", "meta", "train")

    for p in [apps_path, mapping_path, apps_original_path]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Required file not found: {p}")

    apps_data = load_json_or_jsonl(apps_path)
    mapping = load_json_or_jsonl(mapping_path)
    original_apps = load_json_or_jsonl(apps_original_path)
    if len(original_apps) != len(mapping):
        raise ValueError(
            f"apps_original.json ({len(original_apps)}) and apps_to_meta_mapping.json "
            f"({len(mapping)}) must have the same length"
        )

    # Test cases are matched by QUESTION TEXT, not by position. Input files such
    # as qwen_new/qwenN.json reorder the same question set, and positional
    # pairing silently attaches another problem's tests to each question.
    mapping_by_question: Dict[str, dict] = {}
    for item, entry in zip(original_apps, mapping):
        q = item.get("question", "").strip()
        if not q:
            continue
        if q in mapping_by_question and mapping_by_question[q]["meta_dir"] != entry["meta_dir"]:
            print(f"  ⚠ Duplicate question text maps to different meta_dirs; keeping the first")
        mapping_by_question.setdefault(q, entry)

    pos_aligned = sum(
        1
        for item, entry in zip(apps_data, mapping)
        if mapping_by_question.get(item.get("question", "").strip(), {}).get("meta_dir")
        == entry.get("meta_dir")
    )
    print(f"Positional alignment with apps_to_meta_mapping.json: {pos_aligned}/{len(apps_data)}")
    if len(apps_data) != len(mapping):
        print(f"  ⚠ Input has {len(apps_data)} items but mapping has {len(mapping)}; "
              f"items not found in apps_original.json will be skipped")

    # Range selection
    if args.test:
        apps_data = apps_data[:args.num_test]
        mapping = mapping[:args.num_test]
        print(f"Test mode — {len(apps_data)} samples")
    elif args.num_samples:
        apps_data = apps_data[:args.num_samples]
        mapping = mapping[:args.num_samples]
        print(f"Limited to {len(apps_data)} samples")

    # 2. Choose prompt mode
    if args.no_pitfalls:
        mode = "standard"
        system_prompt = SYSTEM_PROMPT_STANDARD
        prompt_builder = build_prompt_standard
        print("Prompt mode: STANDARD (--no_pitfalls)")
    else:
        mode = "pitfalls"
        system_prompt = SYSTEM_PROMPT_PITFALLS
        prompt_builder = build_prompt_with_pitfalls
        with_p = sum(1 for item in apps_data if item.get("pitfalls", "").strip())
        with_f = sum(1 for item in apps_data if item.get("flawed_impl", "").strip())
        print(f"Prompt mode: PITFALLS")
        print(f"  problems with pitfalls field:    {with_p}/{len(apps_data)}")
        print(f"  problems with flawed_impl field: {with_f}/{len(apps_data)}")

    # 3. Setup output files
    os.makedirs(args.output_dir, exist_ok=True)
    apps_basename = os.path.splitext(apps_filename)[0]
    right_path = os.path.join(args.output_dir, f"{apps_basename}_{mode}_r.jsonl")
    wrong_path = os.path.join(args.output_dir, f"{apps_basename}_{mode}_w.jsonl")
    detail_path = os.path.join(args.output_dir, f"{apps_basename}_{mode}_detail.txt")

    processed_qs = load_processed_questions(right_path, wrong_path)
    if processed_qs:
        print(f"Resuming: {len(processed_qs)} already processed")

    # 4. Evaluation loop
    num_right = 0
    num_wrong = 0
    skipped_no_io = 0
    skipped_no_tests = 0
    skipped_processed = 0
    skipped_no_match = 0
    consecutive_failures = 0

    from tqdm import tqdm

    for idx, item in enumerate(
        tqdm(apps_data, total=len(apps_data), desc="Evaluating")
    ):
        question_key = item.get("question", "").strip()

        # Resume check
        if question_key and question_key in processed_qs:
            skipped_processed += 1
            continue

        matched = mapping_by_question.get(question_key)
        if matched is None:
            skipped_no_match += 1
            if args.debug:
                tqdm.write(f"  ⚠ No test cases found for question: {question_key[:100]}")
            continue

        meta_dir = matched["meta_dir"]
        if meta_dir is None:
            skipped_no_io += 1
            continue

        # Load test cases
        io_path = os.path.join(meta_train_dir, meta_dir, "input_output.json")
        if not os.path.exists(io_path):
            skipped_no_io += 1
            continue

        with open(io_path, encoding="utf-8") as f:
            input_output = json.load(f)

        if not input_output.get("inputs") or len(input_output.get("inputs", [])) == 0:
            skipped_no_tests += 1
            continue

        problem = {"input_output": input_output}
        fn_name = input_output.get("fn_name")
        user_prompt = prompt_builder(item, fn_name=fn_name)

        if args.debug:
            tqdm.write(f"\n{'='*40}")
            tqdm.write(f"[{idx+1}] apps_idx={matched['apps_idx']} meta={meta_dir}")
            tqdm.write(f"Prompt tail (last 400 chars): ...{user_prompt[-400:]}")

        # ── Generate code with retry ──
        raw_output = ""
        is_api_failure = True
        for attempt in range(1, GEN_RETRIES + 1):
            try:
                raw_output = model.generate(
                    system=system_prompt,
                    user=user_prompt,
                    max_tokens=args.max_tokens,
                    temperature=args.temperature,
                )
                if raw_output and raw_output.strip():
                    is_api_failure = False
                    break
                tqdm.write(f"  ⚠ Empty response (attempt {attempt}/{GEN_RETRIES})")
            except Exception as e:
                tqdm.write(f"  ⚠ API error (attempt {attempt}/{GEN_RETRIES}): {type(e).__name__}: {e}")
            if attempt < GEN_RETRIES:
                time.sleep(2 ** attempt)

        if is_api_failure:
            consecutive_failures += 1
            tqdm.write(f"  ⛔ API failed ({consecutive_failures}/{MAX_CONSECUTIVE_API_FAILURES})")
            if consecutive_failures >= MAX_CONSECUTIVE_API_FAILURES:
                print(f"\nToo many consecutive API failures. Stopping.")
                break
            continue
        consecutive_failures = 0

        code = extract_code(raw_output, fn_name=fn_name)
        if not code:
            tqdm.write(f"  ⚠ Could not extract code from response")
            num_wrong += 1
            # Write the full original item to wrong jsonl
            with open(wrong_path, "a", encoding="utf-8") as wf:
                wf.write(json.dumps(item, ensure_ascii=False) + "\n")
            # Detail log
            with open(detail_path, "a", encoding="utf-8") as df:
                df.write(f"{'='*80}\n")
                df.write(f"[{idx+1}/{len(apps_data)}]  ✗ WRONG (no code extracted)\n")
                df.write(f"{'='*80}\n")
                df.write(f"apps_idx={matched['apps_idx']}  meta_dir={meta_dir}  mode={mode}\n")
                df.write(f"difficulty={item.get('difficulty','')}\n\n")
                df.write(f"--- Prompt (tail) ---\n{user_prompt[-1000:]}\n\n")
                df.write(f"--- Test Inputs ---\n{json.dumps(input_output.get('inputs',[]), indent=2)[:2000]}\n\n")
                df.write(f"--- Expected Outputs ---\n{json.dumps(input_output.get('outputs',[]), indent=2)[:2000]}\n\n")
                df.write(f"--- Raw LLM Output ---\n{raw_output[:2000]}\n\n")
                df.write(f"--- Test Results ---\n[-2]  (compile error: no code extracted)\n\n")
            continue

        if args.debug:
            tqdm.write(f"  Code ({len(code)} chars):\n{code[:250]}...")

        # ── Evaluate ──
        start = time.time()
        test_results = evaluate_single(problem, code, timeout=TIMEOUT, debug=args.debug,
                                       early_stop=True)
        elapsed = time.time() - start

        is_correct = is_all_correct(test_results)
        compile_err = -2 in test_results
        runtime_err = -1 in test_results

        # Write the full original item to right/wrong jsonl
        filtered_item = item
        if is_correct:
            num_right += 1
            with open(right_path, "a", encoding="utf-8") as rf:
                rf.write(json.dumps(filtered_item, ensure_ascii=False) + "\n")
            tqdm.write(f"  → RIGHT ✓ ({elapsed:.1f}s)")
        else:
            num_wrong += 1
            with open(wrong_path, "a", encoding="utf-8") as wf:
                wf.write(json.dumps(filtered_item, ensure_ascii=False) + "\n")
            status = "compile err" if compile_err else ("runtime err" if runtime_err else "wrong output")
            correct_count = sum(1 for r in test_results if r is True)
            total_count = len(test_results)
            tqdm.write(f"  → WRONG ✗ [{status}] ({correct_count}/{total_count} passed, {elapsed:.1f}s)")

        # Detail log — rich output in txt
        with open(detail_path, "a", encoding="utf-8") as df:
            df.write(f"{'='*80}\n")
            df.write(f"[{idx+1}/{len(apps_data)}]  {'✓ CORRECT' if is_correct else '✗ WRONG'}\n")
            df.write(f"{'='*80}\n")
            df.write(f"apps_idx={matched['apps_idx']}  meta_dir={meta_dir}  mode={mode}\n")
            df.write(f"difficulty={item.get('difficulty','')}\n\n")
            df.write(f"--- Question ---\n{item.get('question','')[:3000]}\n\n")
            df.write(f"--- Func Sign ---\n{item.get('func_sign','')[:500]}\n")
            df.write(f"  (test fn_name: {fn_name if fn_name else '(stdin mode)'})\n\n")
            df.write(f"--- Test Inputs ---\n{json.dumps(input_output.get('inputs',[]), indent=2)[:3000]}\n\n")
            df.write(f"--- Expected Outputs ---\n{json.dumps(input_output.get('outputs',[]), indent=2)[:3000]}\n\n")
            df.write(f"--- Generated Code ---\n{code[:3000]}\n\n")
            df.write(f"--- Raw LLM Output ---\n{raw_output[:3000]}\n\n")
            df.write(f"--- Test Results ---\n{test_results}\n")
            if not is_correct and not compile_err and not runtime_err:
                actual_diag = get_actual_outputs(code, problem, early_stop=True)
                if actual_diag:
                    df.write(f"\n--- Expected vs Actual ---\n{actual_diag}\n")
            df.write(f"--- Runtime ---\n{elapsed:.1f}s\n\n")

    # 6. Summary
    total_eval = num_right + num_wrong
    acc = num_right / total_eval if total_eval > 0 else 0.0
    model_label = args.local_model_path if args.use_local else args.model
    print(f"\n{'='*60}")
    print(f"Evaluation Complete  (mode: {mode})")
    print(f"{'='*60}")
    print(f"Model:              {model_label}  ({'local' if args.use_local else 'api'})")
    print(f"Right (passed):     {num_right}  → {right_path}")
    print(f"Wrong (failed):     {num_wrong}  → {wrong_path}")
    print(f"Total evaluated:    {total_eval}")
    print(f"Skipped (processed): {skipped_processed}")
    print(f"Skipped (no match):  {skipped_no_match}")
    print(f"Skipped (no IO):    {skipped_no_io}")
    print(f"Skipped (0 tests):  {skipped_no_tests}")
    print(f"Accuracy:           {acc:.4f} ({num_right}/{total_eval})")
    print(f"Detail log:         {detail_path}")
    print(f"{'='*60}")


# ── CLI ───────────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate LLM on APPS with pitfalls (via API or local model)"
    )
    parser.add_argument("--model", default="llama-3.3-70b-instruct", help="Model name for api.openlux.ai/v1")
    parser.add_argument("--api_key", default=DEFAULT_API_KEY, help="API key")
    parser.add_argument("--use_local", action="store_true",
                        help="Use local model instead of API")
    parser.add_argument("--local_model_path", default="models/Meta-Llama-3.1-8B-Instruct",
                        help="Path to local HuggingFace model")
    parser.add_argument("--data_dir",
                        default=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        help="Project root (auto-detected)")
    parser.add_argument("--output_dir", default="dataset/code/eval_results/new", help="Output directory")
    parser.add_argument("--input_file", default=None, nargs="+",
                        help="One or more input JSON/JSONL files to evaluate in order "
                             "(the model is loaded once; default: apps_original.json)")
    parser.add_argument("--api_base", default=YUNWU_BASE_URL,
                        help="OpenAI-compatible base URL; point it at a local vLLM server "
                             "(e.g. http://localhost:8000/v1) to share one loaded model across runs")
    parser.add_argument("--max_tokens", type=int, default=4096)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--no_pitfalls", action="store_true",
                        help="Use standard prompt (baseline comparison)")
    parser.add_argument("--test", action="store_true", help="Test on first N samples")
    parser.add_argument("--num_test", type=int, default=10, help="Number of test samples")
    parser.add_argument("--num_samples", type=int, default=None, help="Limit total samples")
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_evaluation(args)
