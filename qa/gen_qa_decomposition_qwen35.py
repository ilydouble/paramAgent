#!/usr/bin/env python
"""
Generate decomposition for dataset/multihop/train/qa.jsonl using the merged
Qwen3.5-2B-qa model.

Uses the LoRA-merged bf16 model (./models/Qwen3.5-2B-qa-merged) for fast inference.
The original base model (./models/Qwen3.5-2B) is untouched.

Outputs 3 JSONL files (qwen1.jsonl, qwen2.jsonl, qwen3.jsonl), each containing the
original dataset with a 'decomposition' field replaced by model-generated decomposition.
Each version uses a different temperature to produce diverse outputs.
Streaming write with checkpoint/resume support.

Example:
  CUDA_VISIBLE_DEVICES=0 python qa/gen_qa_decomposition_qwen35.py \
    --merged_model ./models/Qwen3.5-2B-qa-merged \
    --input_jsonl dataset/multihop/train/qa.jsonl \
    --output_dir dataset/multihop/train \
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
from transformers import AutoModelForCausalLM, AutoTokenizer

SYSTEM_PROMPT = (
    "You will be given a question. Use your knowledge to extract the key point, "
    "the underlying intent, and possible inference patterns needed to answer the question."
)

PRE_INSIGHT_FEWSHOT = """
<Example 1>
q: Anatoly Maltsev and Valentin Turchin were both from Russia, which of the two is known for his work as a mathematician?

Question Parsing and Intent Extraction
Intent:
--------------------------------------------------------------------------------
🔍 Key Components
1. Entity A:
- **Anatoly Maltsev** — mathematician and logician known for contributions in mathematical logic and abstract algebra
2. Entity B:
- **Valentin Turchin** — computer scientist and philosopher known for work in cybernetics and philosophy of science
3. Implied Relationship:
- Comparative inquiry: which individual is more closely associated with the domain of mathematics
4. Answer Type Expected:
- Person name (e.g., "Anatoly Maltsev")
5. Reasoning Type:
- Comparative factual reasoning
6. Required Background:
- Biographical knowledge or retrieved professional profiles
--------------------------------------------------------------------------------
🧠 Inference Trace
1. Retrieve factual data about Maltsev and Turchin's academic domains.
2. Classify Maltsev as a mathematician based on core contributions to mathematical logic.
3. Classify Turchin as mainly working in cybernetics and philosophy.
4. Eliminate Turchin as primary mathematician.
5. Conclude Maltsev is the individual known for mathematics.
--------------------------------------------------------------------------------
📝 Disambiguation Note
- Nationality (Russia) does not help differentiate them.

<Example 2>
The Last Girl on Earth was the third concert tour by Barbadian recording artist Rihanna, the tour visited Europe, Asia, North America and Australia to support her fourth studio album, which 2009 , and fourth studio album by Barbadian singer Rihanna, and released on November 20, 2009 by Def Jam Recordings and SRP Records?

### Question Parsing and Intent Extraction

**Intent:**
--------------------------------------------------------------------------------
🔍 **Key Components**
1. **Entity A**:
   - *The Last Girl on Earth* — Rihanna's third concert tour, associated with promoting a studio album

2. **Entity B**:
   - *Fourth studio album* by Rihanna — referenced multiple times, released in 2009

3. **Key Relationship / Constraint**:
   - Identify the name of Rihanna's **fourth studio album**, which was released on **November 20, 2009**, and **supported by** her third concert tour, *The Last Girl on Earth*

4. **Answer Type Expected**:
   - Album title (e.g., *Rated R*)

5. **Reasoning Type**:
   - Factual entity retrieval based on event-album association and release date

6. **Required Background**:
   - Rihanna's discography: album release dates and which albums were promoted during which concert tours

--------------------------------------------------------------------------------

🧠 **Inference Trace**
- Determine the name of Rihanna's **fourth studio album**
- Confirm that this album was released on **November 20, 2009**
- Verify that this album was the basis for the **"The Last Girl on Earth"** tour
- Conclude that the album is **Rated R**

--------------------------------------------------------------------------------

📝 **Disambiguation Note**
- Although the question includes fragmented/redundant phrasing, the focus is clear: determine the album that matches both the **release date** and **tour association**
"""

IM_START = "<|im_start|>"
IM_END = "<|im_end|>"
ASSISTANT_TAG = f"{IM_START}assistant\n"

VERSION_TEMPS = [0.1, 0.5, 1.0]


def format_prompt(tokenizer: AutoTokenizer, question: str) -> str:
    user_content = (
        f"Here are some examples:{PRE_INSIGHT_FEWSHOT}\n\n"
        f"[Question]: {question.strip()}"
    )
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
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


def chunked(seq: List[dict], n: int) -> Iterable[List[dict]]:
    for i in range(0, len(seq), n):
        yield seq[i : i + n]


def generate_batch(
    model: Any,
    tokenizer: AutoTokenizer,
    prompts: List[str],
    input_device: torch.device,
    max_prompt_len: int,
    max_new_tokens: int,
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
            max_new_tokens=max_new_tokens,
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
    output_jsonl: str,
) -> None:
    completed_ids = load_completed_ids(output_jsonl, args.prompt_key)
    if completed_ids:
        original_count = len(data)
        data = filter_pending(data, completed_ids, args.prompt_key)
        print(f"  Resuming: {len(completed_ids)}/{original_count} done, {len(data)} remaining")
    if not data:
        print(f"  All samples already completed. Nothing to do.")
        return

    num_batches = (len(data) + args.batch_size - 1) // args.batch_size
    total_done = len(completed_ids)
    start = time.time()

    with jsonlines.open(output_jsonl, mode="a") as writer:
        for chunk in tqdm(chunked(data, args.batch_size), total=num_batches, desc=f"temp={temperature}"):
            prompts = [format_prompt(tokenizer, item[args.prompt_key]) for item in chunk]

            generated = generate_batch(
                model=model,
                tokenizer=tokenizer,
                prompts=prompts,
                input_device=input_device,
                max_prompt_len=args.max_prompt_len,
                max_new_tokens=args.max_new_tokens,
                temperature=temperature,
                top_p=args.top_p,
                top_k=args.top_k,
                repetition_penalty=args.repetition_penalty,
            )

            for item, gen_text in zip(chunk, generated):
                out_item = dict(item)
                out_item["decomposition"] = gen_text
                writer.write(out_item)
                total_done += 1

            writer._fp.flush()
            os.fsync(writer._fp.fileno())

            if total_done % 500 == 0:
                elapsed = time.time() - start
                speed = total_done / elapsed
                print(f"  Progress: {total_done} samples, speed: {speed:.1f} samples/s")

    elapsed = time.time() - start
    print(f"  Wrote {total_done} samples to {output_jsonl} ({elapsed:.1f}s, {total_done/elapsed:.1f} samples/s)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Generate QA decomposition with merged Qwen3.5-2B-qa")
    parser.add_argument("--merged_model", default="./models/Qwen3.5-2B-qa-merged")
    parser.add_argument("--input_jsonl", default="dataset/multihop/train/qa.jsonl")
    parser.add_argument("--output_dir", default="dataset/multihop/train")
    parser.add_argument("--prompt_key", default="question")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_prompt_len", type=int, default=1536)
    parser.add_argument("--max_new_tokens", type=int, default=600)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--repetition_penalty", type=float, default=1.05)
    parser.add_argument("--test", action="store_true", help="Test mode: only run on 4 samples")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    with jsonlines.open(args.input_jsonl, mode="r") as reader:
        data = list(reader)

    if args.test:
        data = data[:4]
        print(f"Test mode — using first {len(data)} samples")

    os.makedirs(args.output_dir, exist_ok=True)

    tokenizer, model, input_device = load_model(merged_model=args.merged_model)

    print(f"Model loaded on {input_device} | batch_size={args.batch_size} | max_new_tokens={args.max_new_tokens}")

    for version_idx, temperature in enumerate(VERSION_TEMPS, start=1):
        output_jsonl = os.path.join(args.output_dir, f"qwen{version_idx}.jsonl")
        print(f"\n=== Version {version_idx}: temperature={temperature} ===")
        run_inference(
            data=data,
            tokenizer=tokenizer,
            model=model,
            input_device=input_device,
            args=args,
            temperature=temperature,
            output_jsonl=output_jsonl,
        )

    print(f"\nAll 3 versions saved to {args.output_dir}/qwen1.jsonl, qwen2.jsonl, qwen3.jsonl")


if __name__ == "__main__":
    main()
