#!/usr/bin/env python
"""
Generate pitfalls & flawed implementations for dataset/code/train/code.json
using the merged Qwen3.5-2B-code model.

Uses the LoRA-merged bf16 model (./models/Qwen3.5-2B-code-merged) for fast inference.
The original base model (./models/Qwen3.5-2B) is untouched.

Outputs 3 JSON files (qwen1.json, qwen2.json, qwen3.json), each containing the
original dataset with 'pitfalls', 'flawed_impl', and 'raw_text' fields replaced
by model-generated content. Each version uses a different temperature.

Example:
  CUDA_VISIBLE_DEVICES=0 python code/gen_code_pitfalls_qwen35.py \
    --merged_model ./models/Qwen3.5-2B-code-merged \
    --input_json dataset/code/train/code.json \
    --output_dir dataset/code/train \
    --batch_size 8
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import time
from typing import Any, Iterable, List, Tuple

import jsonlines
import torch
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)

SYSTEM_PROMPT = (
    "You are an AI assistant for Python coding. Given a function signature and docstring, "
    "use your knowledge to propose potential pitfalls for the implementation, "
    "and list the possible pitfalls, and generate up to 6 flawed implementations "
    "specific to the function signature that cover as many pitfalls as possible. "
    "Use <Pitfalls> and <Flawed Implementations> before pitfalls and implementations."
)

FEW_SHOT = """Example:
[Function Signature]:
def has_close_elements(numbers: List[float], threshold: float) -> bool:
    \"\"\"Check if any two numbers in the list are closer than the threshold.\"\"\"\n
<Pitfalls>:
1. **Empty or Single-Element Lists** must return `False`, not `True`.
2. **Duplicate Values** must be compared (difference 0), so never drop duplicates.
3. Always use **absolute difference** (`abs(a - b)`), not raw subtraction.
4. Use the correct **strictness** (`< threshold`, not `<=`).
5. Ensure you don't **exit too early**—check all distinct pairs.

[Flawed Implementations]:

```python
def has_close_elements_v1(numbers: List[float], threshold: float) -> bool:
    # BUG: returns True for empty or single-element lists
    if len(numbers) < 2:
        return True
    for i in range(len(numbers)-1):
        for j in range(i+1, len(numbers)):
            if abs(numbers[i] - numbers[j]) < threshold:
                return True
    return False

def has_close_elements_v2(numbers: List[float], threshold: float) -> bool:
    # BUG: removes duplicates, so identical values never compared
    numbers = sorted(set(numbers))
    for i in range(len(numbers)-1):
        if abs(numbers[i+1] - numbers[i]) < threshold:
            return True
    return False

def has_close_elements_v3(numbers: List[float], threshold: float) -> bool:
    # BUG: uses raw subtraction instead of abs()
    for i in range(len(numbers)-1):
        for j in range(i+1, len(numbers)):
            if (numbers[i] - numbers[j]) < threshold:
                return True
    return False

def has_close_elements_v4(numbers: List[float], threshold: float) -> bool:
    # BUG: uses <= instead of <, misclassifies exactly-threshold pairs
    for i in range(len(numbers)-1):
        for j in range(i+1, len(numbers)):
            if abs(numbers[i] - numbers[j]) <= threshold:
                return True
    return False

def has_close_elements_v5(numbers: List[float], threshold: float) -> bool:
    # BUG: breaks out of outer loop too soon
    for i in range(len(numbers)-1):
        for j in range(i+1, len(numbers)):
            if abs(numbers[i] - numbers[j]) < threshold:
                return True
            break   # <-- this break prevents checking all j for each i
    return False
```"""

IM_START = "<|im_start|>"
IM_END = "<|im_end|>"
ASSISTANT_TAG = f"{IM_START}assistant\n"

VERSION_TEMPS = [0.1, 0.5, 1.0]


def _normalize_func_sign_for_prompt(item: dict) -> str:
    func_sign = str(item.get("func_sign", "") or item.get("prompt", ""))
    docstring = str(item.get("docstring", ""))
    func_sign = re.sub(r"^```python\s*\n?", "", func_sign)
    func_sign = re.sub(r"\n?```\s*$", "", func_sign)
    func_sign = func_sign.strip()
    has_docstring_in_sign = '"""' in func_sign or "'''" in func_sign
    if docstring and not has_docstring_in_sign:
        if not func_sign.endswith("\n"):
            func_sign += "\n"
        func_sign += f'    """{docstring}"""\n'
    return func_sign


def format_prompt(tokenizer: AutoTokenizer, item: dict) -> str:
    func_sign = _normalize_func_sign_for_prompt(item)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": f"{FEW_SHOT}\n\n[Function Signature]:\n{func_sign.strip()}",
        },
    ]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )


def extract_assistant_reply(text: str) -> str:
    if ASSISTANT_TAG in text:
        reply = text.split(ASSISTANT_TAG, 1)[-1]
    else:
        reply = text

    reply = reply.split(IM_END, 1)[0] if IM_END in reply else reply
    reply = re.sub(r"<\|im_start\|>|<\|im_end\|>|<\|endoftext\|>", "", reply).strip()

    think_match = re.match(r"^think\s*\n(.*?)\n/think\s*\n?", reply, re.DOTALL)
    if think_match:
        reply = reply[think_match.end():]

    return reply.strip()


def split_raw_text(raw_text: str) -> Tuple[str, str]:
    """Extract pitfalls and flawed_impl from model output.

    Splits on <Flawed Implementations> (or variants) and then strips any preamble
    before <Pitfalls> (or variants) so that pitfalls starts directly with the content.
    """
    pitfalls = ""
    flawed_impl = ""

    # ── 1. Split off the flawed implementations section ──
    flawed_markers = [
        "<Flawed Implementations>",
        "[Flawed Implementations]:",
        "[Flawed Implementations]",
        "Flawed Implementations:",
        "Flawed Implementations\n",
        "# Flawed Implementations",
        "## Flawed Implementations",
    ]
    flawed_idx = -1
    matched_flawed_marker = ""
    for marker in flawed_markers:
        idx = raw_text.find(marker)
        if idx != -1:
            flawed_idx = idx
            matched_flawed_marker = marker
            break

    if flawed_idx != -1:
        pitfalls_section = raw_text[:flawed_idx]
        flawed_impl = raw_text[flawed_idx + len(matched_flawed_marker):].strip()
        if flawed_impl and not flawed_impl.startswith(":"):
            flawed_impl = ":\n\n" + flawed_impl
    else:
        pitfalls_section = raw_text

    # ── 2. Strip preamble before the pitfalls marker ──
    pitfall_markers = [
        "<Pitfalls>",
        "[Pitfalls]:",
        "[Pitfalls]",
        "Pitfalls:",        # fallback: angle brackets stripped by old bug
        "# Pitfalls",
        "## Pitfalls",
        "### Pitfalls",
    ]
    pitfall_idx = -1
    matched_pitfall_marker = ""
    for marker in pitfall_markers:
        idx = pitfalls_section.find(marker)
        if idx != -1:
            pitfall_idx = idx
            matched_pitfall_marker = marker
            break

    if pitfall_idx != -1:
        pitfalls = pitfalls_section[pitfall_idx + len(matched_pitfall_marker):].strip()
        # Remove leading colon if present
        pitfalls = re.sub(r"^:\s*", "", pitfalls)
    else:
        pitfalls = pitfalls_section.strip()

    return pitfalls, flawed_impl


def load_model(
    merged_model: str,
    device_override: int | None = None,
) -> Tuple[AutoTokenizer, Any, torch.device]:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    device_map = {"": device_override if device_override is not None else 0}

    tokenizer = AutoTokenizer.from_pretrained(merged_model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        merged_model,
        torch_dtype=torch.bfloat16,
        device_map=device_map,
    )
    model.eval()
    input_device = infer_input_device(model)
    model.config.use_cache = True
    return tokenizer, model, input_device


def infer_input_device(model: torch.nn.Module) -> torch.device:
    if hasattr(model, "hf_device_map"):
        for dev in model.hf_device_map.values():
            if isinstance(dev, str):
                return torch.device(dev)
            if isinstance(dev, int):
                return torch.device(f"cuda:{dev}")
    if hasattr(model, "device"):
        return torch.device(model.device)
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def get_sample_id(item: dict, prompt_key: str) -> str:
    if "unique_id" in item:
        return str(item["unique_id"])
    if "task_id" in item:
        return str(item["task_id"])
    if "id" in item:
        return str(item["id"])
    func_sign = str(item.get("func_sign", "") or item.get("prompt", ""))
    entry_point = str(item.get("entry_point", ""))
    question = str(item.get("question", ""))
    raw = f"{func_sign}|||{question}|||{entry_point}"
    if raw.strip("|||"):
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]
    raise KeyError(f"Cannot generate sample ID: missing keys in {list(item.keys())}")


def load_completed_ids(jsonl_path: str, prompt_key: str) -> set:
    completed = set()
    if not os.path.exists(jsonl_path):
        return completed
    try:
        with jsonlines.open(jsonl_path, mode="r") as reader:
            for item in reader:
                try:
                    completed.add(get_sample_id(item, prompt_key))
                except (KeyError, jsonlines.InvalidLineError):
                    continue
    except Exception as e:
        print(f"Warning: error reading {jsonl_path}: {e}. Treating as empty.")
        return set()
    return completed


def filter_pending(data: List[dict], completed: set, prompt_key: str) -> List[dict]:
    pending = []
    for item in data:
        try:
            if get_sample_id(item, prompt_key) not in completed:
                pending.append(item)
        except KeyError:
            pending.append(item)
    return pending


def jsonl_to_json(jsonl_path: str, json_path: str) -> None:
    items = []
    with jsonlines.open(jsonl_path, mode="r") as reader:
        for item in reader:
            items.append(item)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(items, f, ensure_ascii=False, indent=2)
    print(f"Converted {len(items)} records: {jsonl_path} -> {json_path}")


def chunked(seq: List[dict], n: int) -> Iterable[List[dict]]:
    for i in range(0, len(seq), n):
        yield seq[i : i + n]


def generate_batch(
    model: Any,
    tokenizer: AutoTokenizer,
    prompts: List[str],
    input_device: torch.device,
    max_prompt_len: int,
    max_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
    repetition_penalty: float,
) -> List[str]:
    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_prompt_len,
    )
    inputs = {k: v.to(input_device) for k, v in inputs.items()}

    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            use_cache=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    decoded = tokenizer.batch_decode(outputs, skip_special_tokens=False)
    return [extract_assistant_reply(text) for text in decoded]


def run_inference(
    data: List[dict],
    tokenizer: AutoTokenizer,
    model: Any,
    input_device: torch.device,
    args: argparse.Namespace,
    temperature: float,
    jsonl_path: str,
    json_path: str,
) -> None:
    completed_ids = load_completed_ids(jsonl_path, args.prompt_key)
    if completed_ids:
        original_count = len(data)
        data = filter_pending(data, completed_ids, args.prompt_key)
        print(f"  Resuming: {len(completed_ids)}/{original_count} done, {len(data)} remaining")
    if not data:
        print(f"  All samples already completed. Converting JSONL -> JSON.")
        jsonl_to_json(jsonl_path, json_path)
        return

    num_batches = (len(data) + args.batch_size - 1) // args.batch_size
    total_done = len(completed_ids)
    start = time.time()

    with jsonlines.open(jsonl_path, mode="a") as writer:
        for chunk in tqdm(chunked(data, args.batch_size), total=num_batches, desc=f"temp={temperature}"):
            prompts = [
                format_prompt(tokenizer, item)
                for item in chunk
            ]

            generated = generate_batch(
                model=model,
                tokenizer=tokenizer,
                prompts=prompts,
                input_device=input_device,
                max_prompt_len=args.max_prompt_len,
                max_tokens=args.max_tokens,
                temperature=temperature,
                top_p=args.top_p,
                top_k=args.top_k,
                repetition_penalty=args.repetition_penalty,
            )

            for item, gen_text in zip(chunk, generated):
                out_item = dict(item)
                pitfalls, flawed_impl = split_raw_text(gen_text)
                out_item["pitfalls"] = pitfalls
                out_item["flawed_impl"] = flawed_impl
                if "raw_texts" in item:
                    out_item["raw_texts"] = gen_text
                else:
                    out_item["raw_text"] = gen_text
                writer.write(out_item)
                total_done += 1

            writer._fp.flush()
            os.fsync(writer._fp.fileno())

            if total_done % 500 == 0:
                elapsed = time.time() - start
                speed = total_done / elapsed
                print(f"  Progress: {total_done}/{total_done + (len(data) - (total_done - len(completed_ids)))}, speed: {speed:.1f} samples/s")

    elapsed = time.time() - start
    print(f"  Wrote {total_done} samples to {jsonl_path} ({elapsed:.1f}s)")
    jsonl_to_json(jsonl_path, json_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Generate code pitfalls with merged Qwen3.5-2B-code")
    parser.add_argument("--merged_model", default="./models/Qwen3.5-2B-code-merged2")
    parser.add_argument("--input_json", default="dataset/code/train/code.json")
    parser.add_argument("--output_dir", default="dataset/code/train")
    parser.add_argument("--prompt_key", default="func_sign")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_prompt_len", type=int, default=2048)
    parser.add_argument("--max_tokens", type=int, default=812)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--repetition_penalty", type=float, default=1.05)
    parser.add_argument("--test", action="store_true", help="Test mode: only run on 4 samples")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    with open(args.input_json, encoding="utf-8") as f:
        data = json.load(f)

    if args.test:
        data = data[:4]
        print(f"Test mode — using first {len(data)} samples")

    os.makedirs(args.output_dir, exist_ok=True)

    tokenizer, model, input_device = load_model(merged_model=args.merged_model)

    print(f"Model loaded on {input_device} | batch_size={args.batch_size} | max_tokens={args.max_tokens}")

    for version_idx, temperature in enumerate(VERSION_TEMPS, start=1):
        jsonl_path = os.path.join(args.output_dir, f"qwen{version_idx}.jsonl")
        json_path = os.path.join(args.output_dir, f"qwen{version_idx}.json")
        print(f"\n=== Version {version_idx}: temperature={temperature} ===")
        run_inference(
            data=data,
            tokenizer=tokenizer,
            model=model,
            input_device=input_device,
            args=args,
            temperature=temperature,
            jsonl_path=jsonl_path,
            json_path=json_path,
        )

    print(f"\nAll 3 versions saved to {args.output_dir}/qwen1.json, qwen2.json, qwen3.json")


if __name__ == "__main__":
    main()
