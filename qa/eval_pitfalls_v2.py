# python qa/eval_pitfalls_v2.py \
#     --model Meta-Llama-3.1-8B-Instruct \
#     --api_base http://localhost:8000/v1 \
#     --dataset_path generated_reflections/new/qa_r0=0_code.jsonl

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


JUDGE_BASE_URL = "https://www.openai-labs.com/v1"
JUDGE_API_KEY = "<REDACTED_HARDCODED_KEY>"   # ← paste your openai-labs key here

# ── Retry configuration for API failures (timeout / rate-limit / empty response) ──
GENERATION_MAX_RETRIES = 3        # max retries per sample
MAX_CONSECUTIVE_API_FAILURES = 5  # stop entire eval if this many consecutive failures

FIXED_PITFALLS_SYSTEM_PROMPT = (
    "You are an AI agent specialised in answering *multi‑hop* factual questions "
    "(e.g. HotpotQA). A user will provide a single question, along with some decomposition and "
    "reasoning steps to help you arrive at the correct answer.\n\n"
    "❖ Respond with **only one short answer phrase** that "
    "correctly answers the question.\n"
    "❖ Do **not** include other texts such as explanations and citations."
)

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
    """Factory: return the right model based on CLI args.

    - --use_local        -> load a HuggingFace checkpoint from disk
    - --api_base <vllm>  -> point the OpenAI-compatible client at a local vLLM
                            server (e.g. http://localhost:8000/v1)
    - otherwise          -> the yunwu.ai relay
    """
    if args.use_local:
        return LocalModel(args.local_model_path)
    else:
        return YunwuModel(args.model, api_key, args.api_base)


def _clean_extracted(ans: str) -> str:
    ans = ans.strip()
    ans = re.sub(r'^\*{1,2}\s*', '', ans)
    ans = re.sub(r'^`+\*{0,2}\s*', '', ans)
    ans = re.sub(r'\s*\*{0,2}`+$', '', ans)
    ans = re.sub(r'\s*\*{1,2}$', '', ans)
    return ans.strip()


def extract_final_answer(text: str) -> str:
    patterns = [
        r'\*\*\s*`?\s*Answer\s*`?\s*\*\*\s*:\s*(.+?)(?:\n|$)',
        r'\*\*\s*`?\s*Answer\s*`?\s*:\s*\*\*\s*(.+?)(?:\n|$)',
        r'\*\*Answer\*\*\s*:\s*(.+?)(?:\n|$)',
        r'Answer\s*:\s*(.+?)(?:\n|$)',
        r'\*\*answer\*\*\s*:\s*(.+?)(?:\n|$)',
        r'answer\s*:\s*(.+?)(?:\n|$)',
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


def normalize_qa_answer(ans: str) -> str:
    ans = ans.strip().lower()
    ans = re.sub(r'\s+', ' ', ans)
    ans = ans.rstrip('.,;:!?')
    return ans


def llm_compare(predicted: str, golden: str, api_key: str) -> bool:
    prompt = (
        f"Determine if the following two answers to a multi-hop question are equivalent.\n\n"
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

    if not extracted or not extracted.strip():
        print("  executor eval: empty answer")
        return False, "(empty)"

    # 1. exact match (case-insensitive)
    if extracted.strip().lower() == golden_truth.strip().lower():
        print("  executor eval: exact match")
        return True, extracted

    # 2. normalized match (punctuation/whitespace-insensitive)
    norm_extracted = normalize_qa_answer(extracted)
    norm_golden = normalize_qa_answer(golden_truth)

    if norm_extracted == norm_golden:
        print("  executor eval: normalized match")
        return True, extracted

    # Substring containment is disabled: unified judge chain is
    # exact -> normalized -> GPT, no substring step.
    # if norm_extracted in norm_golden or norm_golden in norm_extracted:
    #     print("  executor eval: substring match")
    #     return True, extracted

    # 3. GPT semantic fallback
    is_match = llm_compare(extracted, golden_truth, api_key)
    return is_match, extracted


def generate_with_pitfalls(question: str, decomposition: str, model) -> str:
    user_block = f"[question]: {question}\n\n[decomposition insights]:\n{decomposition}"
    messages = [
        Message(role="system", content=FIXED_PITFALLS_SYSTEM_PROMPT),
        Message(role="user", content=user_block),
    ]
    return model.generate_chat(messages=messages, max_tokens=1024, temperature=0.2)


def generate_with_retry(generate_fn, *args, **kwargs):
    """
    Call a generation function with retries for API failures (timeout, rate limit, empty response).
    Returns (output, is_api_failure) tuple.
    """
    for attempt in range(1, GENERATION_MAX_RETRIES + 1):
        try:
            result = generate_fn(*args, **kwargs)
            if result and result.strip():
                return result, False
            else:
                print(f"  ⚠ Empty response from API (attempt {attempt}/{GENERATION_MAX_RETRIES})")
        except Exception as e:
            print(f"  ⚠ API call failed (attempt {attempt}/{GENERATION_MAX_RETRIES}): {type(e).__name__}: {e}")
        if attempt < GENERATION_MAX_RETRIES:
            wait = 2 ** attempt
            print(f"  ↻ Retrying in {wait}s...")
            time.sleep(wait)
    print(f"  ❌ All {GENERATION_MAX_RETRIES} retries exhausted — API unavailable")
    return "", True


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, default="dataset/multihop/train/qa.jsonl")
    parser.add_argument("--model", type=str, default="llama-3.1-8b")
    parser.add_argument("--api_key", type=str, default=None)
    parser.add_argument("--api_base", type=str, default=YUNWU_BASE_URL,
                        help="OpenAI-compatible base URL; point it at a local vLLM server "
                             "(e.g. http://localhost:8000/v1) to share one loaded model across runs")
    parser.add_argument("--use_local", action="store_true",
                        help="Use local HuggingFace model instead of the API")
    parser.add_argument("--local_model_path", type=str, default="models/Meta-Llama-3.1-8B-Instruct",
                        help="Path to a local HF checkpoint when --use_local is set")
    parser.add_argument("--num_samples", type=int, default=None)
    parser.add_argument("--output_dir", type=str, default="dataset/multihop/eval_results/new")
    return parser.parse_args()


def load_dataset(path: str) -> list:
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

    model = create_model(args, api_key)
    if args.use_local:
        print(f"Using local model: {args.local_model_path}")
    else:
        print(f"Using API model: {args.model} @ {args.api_base}")

    basename = os.path.splitext(os.path.basename(args.dataset_path))[0]
    os.makedirs(args.output_dir, exist_ok=True)
    right_path = os.path.join(args.output_dir, f"{basename}_r.jsonl")
    wrong_path = os.path.join(args.output_dir, f"{basename}_w.jsonl")
    detail_path = os.path.join(args.output_dir, f"{basename}_detail.txt")

    n = len(dataset)

    processed_keys = set()
    for p in (right_path, wrong_path):
        if os.path.exists(p):
            with open(p, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            rec = json.loads(line)
                            key = rec.get("unique_id") or rec.get("question", "")
                            if key:
                                processed_keys.add(key)
                        except json.JSONDecodeError:
                            pass
    num_right = 0
    num_wrong = 0
    if processed_keys:
        print(f"Resuming: {len(processed_keys)} already processed, will skip them")

    consecutive_failures = 0

    for i, item in enumerate(dataset):
        uid = item.get("unique_id", "") or item.get("question", "")
        if uid and uid in processed_keys:
            print(f"\n[{i+1}/{n}] SKIP (already processed): {item.get('question', '')[:60]}...")
            continue

        question = item["question"]
        decomposition = item.get("decomposition", "")
        golden = item.get("answer", "")

        if not golden:
            print(f"\n[{i+1}/{n}] SKIP (no answer): {question[:60]}...")
            continue

        print(f"\n[{i+1}/{n}] {question[:80]}...")

        start_time = time.time()

        raw_answer, is_api_failure = generate_with_retry(generate_with_pitfalls, question, decomposition, model)
        if is_api_failure:
            consecutive_failures += 1
            print(f"  ⛔ Skipping sample (API failure, consecutive: {consecutive_failures}/{MAX_CONSECUTIVE_API_FAILURES})")
            if consecutive_failures >= MAX_CONSECUTIVE_API_FAILURES:
                print(f"\n⛔ Too many consecutive API failures ({MAX_CONSECUTIVE_API_FAILURES}). Stopping evaluation.")
                break
            continue
        consecutive_failures = 0

        is_correct, extracted = check_answer(raw_answer, golden, api_key)

        elapsed = time.time() - start_time

        if is_correct:
            num_right += 1
            with open(right_path, "a", encoding="utf-8") as rf:
                rf.write(json.dumps(item, ensure_ascii=False) + "\n")
            print(f"  → RIGHT (decomposition useful)")
            verdict = "✓ CORRECT"
        else:
            num_wrong += 1
            with open(wrong_path, "a", encoding="utf-8") as wf:
                wf.write(json.dumps(item, ensure_ascii=False) + "\n")
            print(f"  → WRONG (for DPO negative)")
            verdict = "✗ WRONG"

        total_eval = num_right + num_wrong
        acc = num_right / total_eval if total_eval > 0 else 0.0
        print(f"  golden:      {golden}")
        print(f"  extracted:   {extracted[:100]}")
        print(f"  running acc: {acc:.4f} ({num_right}/{total_eval})")

        with open(detail_path, "a", encoding="utf-8") as df:
            df.write(f"{'=' * 80}\n")
            df.write(f"[{i+1}/{n}]  {verdict}\n")
            df.write(f"{'=' * 80}\n\n")
            df.write(f"--- Question ---\n{question}\n\n")
            df.write(f"--- Golden Answer ---\n{golden}\n\n")
            # df.write(f"--- Decomposition ---\n{decomposition}\n\n")
            df.write(f"--- Extracted Answer ---\n{extracted or '(empty)'}\n\n")
            df.write(f"--- Model Full Output ---\n{raw_answer or '(empty)'}\n\n")
            df.write(f"--- Runtime ---\n{elapsed:.1f}s\n\n")

    total_eval = num_right + num_wrong
    final_acc = num_right / total_eval if total_eval > 0 else 0.0
    print(f"\n{'=' * 60}")
    print(f"Done! right={num_right}, wrong={num_wrong}, total={total_eval}, acc={final_acc:.4f}")
    print(f"Right cases  → {right_path}")
    print(f"Wrong cases  → {wrong_path}")
    print(f"Detail log   → {detail_path}")


if __name__ == "__main__":
    main()
