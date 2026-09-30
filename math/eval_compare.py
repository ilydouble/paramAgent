#!/usr/bin/env python
"""
Evaluate model on math problems WITHOUT pitfalls (baseline).

Only gives the question — no pitfalls/hints. Outputs go to:
  - {basename}_baseline_r.jsonl
  - {basename}_baseline_w.jsonl
  - {basename}_baseline_detail.txt

Usage:
  source /root/autodl-tmp/miniconda3/etc/profile.d/conda.sh && conda activate param
  python math/eval_compare.py --model qwen3.5-27b --num_samples 100
"""

import os
import sys
import json
import argparse
import re
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from generators.model import ModelBase, Message
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_random_exponential

YUNWU_BASE_URL = "https://yunwu.ai/v1"

# ──────────────── System prompt (NO pitfalls) ────────────────

BASELINE_SYSTEM_PROMPT = (
    "You are an AI agent specialised in solving mathematics problems.\n\n"
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
            timeout=60.0,
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


# ────────────────────── Answer Extraction ──────────────────────

def _clean_extracted(ans: str) -> str:
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
        r'\\boxed\{([^}]+)\}',
        r'(?:^|\n)\s*(?:Therefore|Thus|So|Hence|The\s+final\s+answer\s+is)[,:]\s*(.+?)(?:\n|$)',
    ]
    for pattern in patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        if matches:
            return _clean_extracted(matches[-1])
    lines = [l.strip() for l in text.strip().split('\n') if l.strip()]
    for line in reversed(lines):
        if line not in ('\]', '\[', '$$', '$', '\)', '\(', '```'):
            return _clean_extracted(line)
    return _clean_extracted(lines[-1]) if lines else text.strip()


def normalize_answer(ans: str) -> str:
    ans = ans.strip()
    ans = re.sub(r'^\$|\$$|^\\\(|\\\)$', '', ans)
    ans = re.sub(r'\\frac\{([^}]*)\}\{([^}]*)\}', r'\1/\2', ans)
    ans = re.sub(r'\\text\{([^}]*)\}', r'\1', ans)
    ans = re.sub(r'\\(?:mathrm|mathbf|mathit|mathsf|mathtt|textrm|textsf|texttt|bm|cal|scr|frak|bb)\{([^}]*)\}', r'\1', ans)
    ans = re.sub(r'\s*([+\-])\s*', r'\1', ans)
    ans = re.sub(r'\s+', ' ', ans)
    ans = ans.rstrip('.,;:')
    return ans.strip()


def _strip_units(s: str) -> str:
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
        client = OpenAI(base_url=YUNWU_BASE_URL, api_key=api_key, timeout=60.0)
        response = client.chat.completions.create(
            model="gpt-4o-2024-11-20", messages=[{"role": "user", "content": prompt}],
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

    if not extracted or not extracted.strip():
        print("  executor eval: empty answer")
        return False, "(empty)"

    norm_extracted = normalize_answer(extracted)
    norm_golden = normalize_answer(golden_truth)

    if norm_extracted == norm_golden:
        print("  executor eval: exact match")
        return True, extracted

    stripped_extracted = _strip_units(norm_extracted)
    stripped_golden = _strip_units(norm_golden)
    if stripped_extracted == stripped_golden:
        print("  executor eval: exact match (unit-stripped)")
        return True, extracted

    is_match = llm_compare(extracted, golden_truth, api_key)
    return is_match, extracted


# ────────────────────── Generation ──────────────────────

def generate_baseline(problem: str, model: YunwuModel) -> str:
    """Only give the question, no pitfalls."""
    messages = [
        Message(role="system", content=BASELINE_SYSTEM_PROMPT),
        Message(role="user", content=f"[question]: {problem}"),
    ]
    return model.generate_chat(messages=messages, max_tokens=2048, temperature=0.2)


# ────────────────────── Main ──────────────────────

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, default="dataset/math/train/math.json")
    parser.add_argument("--model", type=str, default="llama-3.1-8b")
    parser.add_argument("--api_key", type=str, default=None)
    parser.add_argument("--api_base", type=str, default=YUNWU_BASE_URL,
                        help="OpenAI-compatible base URL; point it at a local vLLM server "
                             "(e.g. http://localhost:8000/v1) to share one loaded model across runs")
    parser.add_argument("--num_samples", type=int, default=None)
    parser.add_argument("--output_dir", type=str, default="dataset/math/eval_results")
    return parser.parse_args()


def load_dataset(path: str) -> list:
    """Load .json (single array) or .jsonl (line-delimited)."""
    with open(path, encoding="utf-8") as f:
        first = f.read(1)
        f.seek(0)
        if first == '[':
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


def main():
    args = get_args()
    api_key = args.api_key or os.getenv("OPENAI_API_KEY", "<REDACTED_HARDCODED_KEY>")

    dataset = load_dataset(args.dataset_path)
    print(f"Loaded {len(dataset)} examples from {args.dataset_path}")

    if args.num_samples and args.num_samples > 0:
        dataset = dataset[:args.num_samples]
        print(f"Limited to first {len(dataset)} samples")

    model = YunwuModel(args.model, api_key, args.api_base)

    basename = os.path.splitext(os.path.basename(args.dataset_path))[0]
    right_path = os.path.join(args.output_dir, f"{basename}_baseline_r.jsonl")
    wrong_path = os.path.join(args.output_dir, f"{basename}_baseline_w.jsonl")
    detail_path = os.path.join(args.output_dir, f"{basename}_baseline_detail.txt")

    n = len(dataset)

    # ── Resume support ──
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
    if os.path.exists(right_path):
        num_right = sum(1 for _ in open(right_path, encoding="utf-8"))
    if os.path.exists(wrong_path):
        num_wrong = sum(1 for _ in open(wrong_path, encoding="utf-8"))
    if processed_ids:
        print(f"Resuming: {num_right} right + {num_wrong} wrong = {num_right + num_wrong} already processed, will skip them")

    for i, item in enumerate(dataset):
        uid = item.get("unique_id", "")
        if uid and uid in processed_ids:
            print(f"\n[{i+1}/{n}] SKIP (already processed): {item.get('problem', '')[:60]}...")
            continue

        problem = item["problem"]
        golden = item["answer"]

        print(f"\n[{i+1}/{n}] {problem[:80]}...")

        start_time = time.time()

        try:
            raw_answer = generate_baseline(problem, model)
        except Exception as e:
            print(f"  ❌ generation error: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
            raw_answer = ""

        is_correct, extracted = check_answer(raw_answer, golden, api_key)

        elapsed = time.time() - start_time

        if is_correct:
            num_right += 1
            with open(right_path, "a", encoding="utf-8") as rf:
                rf.write(json.dumps(item, ensure_ascii=False) + "\n")
            print(f"  → RIGHT")
            verdict = "✓ CORRECT"
        else:
            num_wrong += 1
            with open(wrong_path, "a", encoding="utf-8") as wf:
                wf.write(json.dumps(item, ensure_ascii=False) + "\n")
            print(f"  → WRONG")
            verdict = "✗ WRONG"

        acc = num_right / (i + 1)
        print(f"  golden:      {golden}")
        print(f"  extracted:   {extracted[:100]}")
        print(f"  running acc: {acc:.4f} ({num_right}/{i+1})")

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

