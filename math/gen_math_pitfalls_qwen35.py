#!/usr/bin/env python
"""
Generate pitfalls for dataset/math/math.json using the merged Qwen3.5-2B-math model.

Uses the LoRA-merged bf16 model (./models/Qwen3.5-2B-math-merged) for fast inference.
The original base model (./models/Qwen3.5-2B) is untouched.

Outputs 3 JSON files (qwen1.json, qwen2.json, qwen3.json), each containing the
original dataset with a 'pitfalls' field replaced by model-generated pitfalls.
Each version uses a different temperature to produce diverse outputs.

Example:
  CUDA_VISIBLE_DEVICES=0 python math/gen_math_pitfalls_qwen35.py \
    --merged_model ./models/Qwen3.5-2B-math-merged \
    --input_json dataset/math/math.json \
    --output_dir dataset/math \
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
    AutoModelForMultimodalLM,
    AutoTokenizer,
)

SYSTEM_PROMPT = (
    "You are a mathematics education assistant. Given a math problem, identify "
    "and clearly explain common mistakes and pitfalls students may encounter."
)

IM_START = "<|im_start|>"
IM_END = "<|im_end|>"
ASSISTANT_TAG = f"{IM_START}assistant\n"

VERSION_TEMPS = [0.1, 0.5, 1.0]


def format_prompt(tokenizer: AutoTokenizer, problem: str) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"PROBLEM:\n{problem.strip()}"},
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

    fence_pos = reply.rfind("```")
    if fence_pos != -1:
        reply = reply[: fence_pos + 3]

    return reply.strip()


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

    model = AutoModelForMultimodalLM.from_pretrained(
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
    if prompt_key in item:
        return hashlib.sha256(item[prompt_key].encode("utf-8")).hexdigest()[:16]
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
            prompts = [format_prompt(tokenizer, item[args.prompt_key]) for item in chunk]

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
                out_item["pitfalls"] = gen_text
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
    parser = argparse.ArgumentParser("Generate math pitfalls with merged Qwen3.5-2B-math")
    parser.add_argument("--merged_model", default="./models/Qwen3.5-2B-math-merged")
    parser.add_argument("--input_json", default="dataset/math/math.json")
    parser.add_argument("--output_dir", default="dataset/math")
    parser.add_argument("--prompt_key", default="problem")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_prompt_len", type=int, default=1536)
    parser.add_argument("--max_tokens", type=int, default=900)
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
