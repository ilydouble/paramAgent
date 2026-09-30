# python math/eval_pitfalls_v2.py \
#     --model Meta-Llama-3.1-8B-Instruct \
#     --api_base http://localhost:8000/v1 \
#     --dataset_path dataset/code/eval_results/apps_baseline_w.jsonl

# python qa/eval_pitfalls_v2.py --dataset_path <file> --use_local \
#     --local_model_path models/Meta-Llama-3.1-8B-Instruct


# python qa/eval_pitfalls_v2.py --dataset_path <file> --model gpt-4o --api_base https://yunwu.ai/v1

import os
import sys
import json
import argparse
import re
import time
from typing import Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from generators.model import ModelBase, Message
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_random_exponential

YUNWU_BASE_URL = "https://yunwu.ai/v1"

# ── Judge model (semantic answer comparison) ─────────────────────────────
# Served through the openai-labs relay (same endpoint as executors/
# Math_executor.py). Hardcoded on purpose — no env indirection.
JUDGE_BASE_URL = "https://www.openai-labs.com/v1"
JUDGE_API_KEY = "<REDACTED_HARDCODED_KEY>"   # ← paste your openai-labs key here

# ──────────────── Fixed pitfalls prompt (with Answer: marker) ────────────────

FIXED_PITFALLS_SYSTEM_PROMPT = (
    "You are an AI agent specialised in solving mathematics problems. "
    "A user will provide a single question, along with some potential "
    "mistakes and pitfalls about the question to help you avoid common errors.\n\n"
    "❖ Directly output only the final answer, no reasoning steps.\n"
    "❖ Prefix the result with **`Answer:`** on its own line.\n"
    "❖ The final answer should be simplified to its simplest form, "
    "e.g., 25, 2516_8, \\frac{1}{36}, etc.\n"
    "❖ Do **not** include explanations, steps, or citations."
)

# ────────────────────── Model Wrapper ──────────────────────

class YunwuModel(ModelBase):
    def __init__(self, model_name: str, api_key: str = None,
                 base_url: str = YUNWU_BASE_URL):
        self.name = model_name
        self.is_chat = True
        self._client = OpenAI(
            base_url=base_url,
            api_key=api_key or os.getenv("OPENAI_API_KEY", "<REDACTED_HARDCODED_KEY>"),
            timeout=60.0,  # 60s timeout to avoid hanging
        )

    @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(3))
    def generate_chat(
        self, messages: list, max_tokens: int = 1024,
        temperature: float = 0.2, num_comps: int = 1,
    ) -> str:
        converted = [{"role": m.role, "content": m.content} for m in messages]
        response = self._client.chat.completions.create(
            model=self.name, messages=converted,
            max_tokens=max_tokens, temperature=temperature, n=num_comps,
            # extra_body={"enable_thinking": True},  
        )
        if num_comps == 1:
            content = response.choices[0].message.content
            if not content:
                print(f"  ⚠ API returned empty content (finish_reason={response.choices[0].finish_reason})")
            return content or ""
        return [c.message.content or "" for c in response.choices]


class LocalModel:
    """Local HuggingFace model loaded from disk (e.g. models/Meta-Llama-3.1-8B-Instruct).

    Mirrors the YunwuModel.generate_chat interface so the caller stays
    identical whether it hits the API or a local checkpoint.
    """

    def __init__(self, model_path: str, device_override: Optional[int] = None):
        import torch
        import transformers

        self.name = model_path
        self.is_chat = True
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

    def generate_chat(
        self, messages: list, max_tokens: int = 1024,
        temperature: float = 0.2, num_comps: int = 1,
    ) -> str:
        import torch

        converted = [{"role": m.role, "content": m.content} for m in messages]
        prompt = self.tokenizer.apply_chat_template(
            converted, tokenize=False, add_generation_prompt=True
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


def create_model(args: argparse.Namespace, api_key: str) -> "ModelBase":
    """Factory: return the right model based on CLI args."""
    if args.use_local:
        return LocalModel(args.local_model_path)
    else:
        return YunwuModel(args.model, api_key, args.api_base)


# ────────────────────── Answer Extraction ──────────────────────

def _clean_extracted(ans: str) -> str:
    """Strip leftover markdown artifacts around the extracted answer."""
    ans = ans.strip()
    # Remove leading markdown bold/italic markers (** or *)
    ans = re.sub(r'^\*{1,2}\s*', '', ans)
    # Remove leading/trailing backticks with optional bold markers
    ans = re.sub(r'^`+\*{0,2}\s*', '', ans)
    ans = re.sub(r'\s*\*{0,2}`+$', '', ans)
    # Remove trailing markdown bold/italic markers
    ans = re.sub(r'\s*\*{1,2}$', '', ans)
    # Remove leading/trailing LaTeX math delimiters
    ans = re.sub(r'^\\\(\s*', '', ans)
    ans = re.sub(r'\s*\\\)$', '', ans)
    ans = re.sub(r'^\$\s*', '', ans)
    ans = re.sub(r'\s*\$$', '', ans)
    return ans.strip()


def extract_final_answer(text: str) -> str:
    patterns = [
        r'\*\*\s*`?\s*Answer\s*`?\s*\*\*\s*:\s*(.+?)(?:\n|$)',   # **Answer**: or **`Answer`**:
        r'\*\*\s*`?\s*Answer\s*`?\s*:\s*\*\*\s*(.+?)(?:\n|$)',   # **Answer:** or **`Answer`:**
        r'\*\*Answer\*\*\s*:\s*(.+?)(?:\n|$)',                    # **Answer**:
        r'Answer\s*:\s*(.+?)(?:\n|$)',
        r'\*\*answer\*\*\s*:\s*(.+?)(?:\n|$)',
        r'answer\s*:\s*(.+?)(?:\n|$)',
        r'\\boxed\{([^}]+)\}',                  # \boxed{1}
        r'(?:^|\n)\s*(?:Therefore|Thus|So|Hence|The\s+final\s+answer\s+is)[,:]\s*(.+?)(?:\n|$)',
    ]
    for pattern in patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        if matches:
            return _clean_extracted(matches[-1])
    lines = [l.strip() for l in text.strip().split('\n') if l.strip()]
    # Skip lines that are just LaTeX delimiters
    for line in reversed(lines):
        if line not in ('\]', '\[', '$$', '$', '\)', '\(', '```'):
            return _clean_extracted(line)
    return _clean_extracted(lines[-1]) if lines else text.strip()


def normalize_answer(ans: str) -> str:
    ans = ans.strip()
    # Remove LaTeX wrappers like $...$ or \(...\)
    ans = re.sub(r'^\$|\$$|^\\\(|\\\)$', '', ans)
    # Convert \frac{a}{b} → a/b first (before other LaTeX stripping)
    ans = re.sub(r'\\frac\{([^}]*)\}\{([^}]*)\}', r'\1/\2', ans)
    # Normalize \text{...} to just its content
    ans = re.sub(r'\\text\{([^}]*)\}', r'\1', ans)
    # Remove LaTeX formatting commands like \mathrm, \mathbf, etc. (but not \frac)
    ans = re.sub(r'\\(?:mathrm|mathbf|mathit|mathsf|mathtt|textrm|textsf|texttt|bm|cal|scr|frak|bb)\{([^}]*)\}', r'\1', ans)
    # Remove spaces around math operators for consistent comparison
    ans = re.sub(r'\s*([+\-])\s*', r'\1', ans)
    # Collapse remaining whitespace
    ans = re.sub(r'\s+', ' ', ans)
    # Remove trailing punctuation
    ans = ans.rstrip('.,;:')
    return ans.strip()


def _strip_units(s: str) -> str:
    """Extract just the numeric/math part, dropping trailing unit words."""
    # Remove trailing English unit words (feet, miles, dollars, etc.)
    s = re.sub(r'\s+(feet|foot|inches|inch|miles|mile|meters|meter|dollars?|cents?|hours?|minutes?|seconds?|per\s+hour|mph|km/h)\s*$', '', s, flags=re.IGNORECASE)
    return s.strip()


def llm_compare(predicted: str, golden: str, api_key: str) -> bool:
    prompt = (
        f"Determine if the following two mathematical answers are equivalent.\n\n"
        f"Answer 1 (predicted): {predicted}\n"
        f"Answer 2 (golden truth): {golden}\n\n"
        f'Respond only with "Yes" or "No".'
    )
    try:
        client = OpenAI(base_url=JUDGE_BASE_URL, api_key=JUDGE_API_KEY, timeout=60.0)
        response = client.chat.completions.create(
            model="gpt-4o-mini", messages=[{"role": "user", "content": prompt}],
            temperature=0.0, max_tokens=8,
        )
        result = response.choices[0].message.content.strip().lower()
        print("  executor eval:", result)
        return result.startswith("yes")
    except Exception as e:
        print(f"  executor eval: exception ({e})")
        return False


def check_answer(raw_output: str, golden_truth: str, api_key: str) -> tuple:
    extracted = extract_final_answer(raw_output)

    # Guard against empty output (generation failed or unparseable)
    if not extracted or not extracted.strip():
        print("  executor eval: empty answer")
        return False, "(empty)"

    norm_extracted = normalize_answer(extracted)
    norm_golden = normalize_answer(golden_truth)

    # Level 1: exact match after normalization
    if norm_extracted == norm_golden:
        print("  executor eval: exact match")
        return True, extracted

    # Level 1b: exact match after stripping units
    stripped_extracted = _strip_units(norm_extracted)
    stripped_golden = _strip_units(norm_golden)
    if stripped_extracted == stripped_golden:
        print("  executor eval: exact match (unit-stripped)")
        return True, extracted

    # Level 2: LLM comparison
    is_match = llm_compare(extracted, golden_truth, api_key)
    return is_match, extracted


# ────────────────────── Generation ──────────────────────

def generate_with_pitfalls(problem: str, pitfalls: str, model: YunwuModel) -> str:
    user_block = f"[question]: {problem}\n\n[mistake insights]:\n{pitfalls}"
    messages = [
        Message(role="system", content=FIXED_PITFALLS_SYSTEM_PROMPT),
        Message(role="user", content=user_block),
    ]
    return model.generate_chat(messages=messages, max_tokens=2048, temperature=0.2)


# ────────────────────── Main ──────────────────────

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, default="generated_reflections/new/math_r0=0_code.jsonl")
    parser.add_argument("--model", type=str, default="llama-3.1-8b")
    parser.add_argument("--api_key", type=str, default=None)
    parser.add_argument("--api_base", type=str, default=YUNWU_BASE_URL,
                        help="OpenAI-compatible base URL; point it at a local vLLM server "
                             "(e.g. http://localhost:8000/v1) to share one loaded model across runs")
    parser.add_argument("--use_local", action="store_true",
                        help="Use local HuggingFace model instead of the yunwu.ai API")
    parser.add_argument("--local_model_path", type=str, default="models/Meta-Llama-3.1-8B-Instruct",
                        help="Path to a local HF checkpoint when --use_local is set")
    parser.add_argument("--num_samples", type=int, default=None)
    parser.add_argument("--output_dir", type=str, default="dataset/math/eval_results/new")
    return parser.parse_args()


def load_dataset(path: str) -> list:
    """Load .json (single array) or .jsonl (line-delimited)."""
    with open(path, encoding="utf-8") as f:
        first = f.read(1)
        f.seek(0)
        if first == '[':
            return json.load(f)
        # Could be a single JSON object or JSONL – try single first, fall back to JSONL
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


def main():
    args = get_args()
    api_key = args.api_key or os.getenv("OPENAI_API_KEY", "<REDACTED_HARDCODED_KEY>")

    dataset = load_dataset(args.dataset_path)
    print(f"Loaded {len(dataset)} examples from {args.dataset_path}")

    if args.num_samples and args.num_samples > 0:
        dataset = dataset[:args.num_samples]
        print(f"Limited to first {len(dataset)} samples")

    model = create_model(args, api_key)
    if args.use_local:
        print(f"Using local model: {args.local_model_path}")
    else:
        print(f"Using API model: {args.model}")

    basename = os.path.splitext(os.path.basename(args.dataset_path))[0]
    os.makedirs(args.output_dir, exist_ok=True)
    right_path = os.path.join(args.output_dir, f"{basename}_r.jsonl")
    wrong_path = os.path.join(args.output_dir, f"{basename}_w.jsonl")
    detail_path = os.path.join(args.output_dir, f"{basename}_detail.txt")

    n = len(dataset)

    # ── Resume support: skip already-processed items ──
    processed_ids = set()
    num_right = 0
    num_wrong = 0
    for p in (right_path, wrong_path):
        if os.path.exists(p):
            with open(p, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            rec = json.loads(line)
                            if "unique_id" in rec:
                                processed_ids.add(rec["unique_id"])
                        except json.JSONDecodeError:
                            pass
    # Count existing results
    if os.path.exists(right_path):
        num_right = sum(1 for _ in open(right_path, encoding="utf-8"))
    if os.path.exists(wrong_path):
        num_wrong = sum(1 for _ in open(wrong_path, encoding="utf-8"))
    if processed_ids:
        print(f"Resuming: {num_right} right + {num_wrong} wrong = {num_right + num_wrong} already processed, will skip them")

    for i, item in enumerate(dataset):
        # Skip already-processed items
        uid = item.get("unique_id", "")
        if uid and uid in processed_ids:
            print(f"\n[{i+1}/{n}] SKIP (already processed): {item.get('problem', '')[:60]}...")
            # Count from existing output to maintain acc
            continue

        problem = item["problem"]
        golden = item["answer"]
        pitfalls = item.get("pitfalls", "")

        print(f"\n[{i+1}/{n}] {problem[:80]}...")

        start_time = time.time()

        try:
            raw_answer = generate_with_pitfalls(problem, pitfalls, model)
        except Exception as e:
            print(f"  ❌ generation error: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
            raw_answer = ""

        is_correct, extracted = check_answer(raw_answer, golden, api_key)

        # Skip unstable/empty outputs — don't write to right/wrong, so
        # the item will be re-processed on the next run (resume handles it).
        if not raw_answer or not raw_answer.strip() or extracted == "(empty)":
            elapsed = time.time() - start_time
            print(f"  ⚠ unstable/empty output, skip (will retry on resume)")
            with open(detail_path, "a", encoding="utf-8") as df:
                df.write(f"{'=' * 80}\n")
                df.write(f"[{i+1}/{n}]  ⚠ SKIPPED (empty output)\n")
                df.write(f"{'=' * 80}\n\n")
                df.write(f"--- Question ---\n{problem}\n\n")
                df.write(f"--- Golden Answer ---\n{golden}\n\n")
                df.write(f"--- Extracted Answer ---\n{extracted or '(empty)'}\n\n")
                df.write(f"--- Model Full Output ---\n{raw_answer or '(empty)'}\n\n")
                df.write(f"--- Runtime ---\n{elapsed:.1f}s\n\n")
            continue

        elapsed = time.time() - start_time

        # Save original item (without eval fields) to right/wrong jsonl
        if is_correct:
            num_right += 1
            with open(right_path, "a", encoding="utf-8") as rf:
                rf.write(json.dumps(item, ensure_ascii=False) + "\n")
            print(f"  → RIGHT (pitfalls useful)")
            verdict = "✓ CORRECT"
        else:
            num_wrong += 1
            with open(wrong_path, "a", encoding="utf-8") as wf:
                wf.write(json.dumps(item, ensure_ascii=False) + "\n")
            print(f"  → WRONG (for DPO negative)")
            verdict = "✗ WRONG"

        acc = num_right / (i + 1)
        print(f"  golden:      {golden}")
        print(f"  extracted:   {extracted[:100]}")
        print(f"  running acc: {acc:.4f} ({num_right}/{i+1})")

        # Append human-readable detail log
        with open(detail_path, "a", encoding="utf-8") as df:
            df.write(f"{'=' * 80}\n")
            df.write(f"[{i+1}/{n}]  {verdict}\n")
            df.write(f"{'=' * 80}\n\n")
            df.write(f"--- Question ---\n{problem}\n\n")
            df.write(f"--- Golden Answer ---\n{golden}\n\n")
            df.write(f"--- Extracted Answer ---\n{extracted or '(empty)'}\n\n")
            df.write(f"--- Model Full Output ---\n{raw_answer or '(empty)'}\n\n")
            df.write(f"--- Runtime ---\n{elapsed:.1f}s\n\n")

    print(f"\n{'=' * 60}")
    print(f"Done! right={num_right}, wrong={num_wrong}, total={n}, acc={num_right/n:.4f}")
    print(f"Right cases  → {right_path}")
    print(f"Wrong cases  → {wrong_path}")
    print(f"Detail log   → {detail_path}")


if __name__ == "__main__":
    main()
