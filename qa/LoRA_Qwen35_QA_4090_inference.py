#!/usr/bin/env python
"""Single-GPU inference for the LoRA-fine-tuned Qwen3.5-2B QA decomposition model.

- 4-bit load fits on a single RTX 4090 (24 GB).
- Uses Qwen's native chat template (not Llama [INST] format).
- Checkpoint/resume: skips already-completed samples on re-run.

Example (with LoRA adapter):
  CUDA_VISIBLE_DEVICES=0 python qa/LoRA_Qwen35_QA_4090_inference.py \
    --base_model ./models/Qwen3.5-2B \
    --lora_path ./lora-qwen3.5-2b-qa \
    --input_jsonl benchmarks/hotpotQA_testset.jsonl \
    --output_jsonl generated_reflections/qa/qa_decomposition_qwen35_2b.jsonl \
    --prompt_key question --batch_size 2

Example (with merged model):
  CUDA_VISIBLE_DEVICES=0 python qa/LoRA_Qwen35_QA_4090_inference.py \
    --base_model ./models/Qwen3.5-2B-qa-merged \
    --input_jsonl dataset/multihop/meta/2w/2w-train.jsonl \
    --output_jsonl dataset/multihop/meta/2w/qa_inference_test.jsonl \
    --prompt_key question --test --batch_size 1
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from typing import Any, Iterable, List, Tuple

import torch
import jsonlines
from peft import PeftModel
from tqdm import tqdm
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForMultimodalLM,
    AutoProcessor,
    BitsAndBytesConfig,
)

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


def get_sample_id(item: dict, prompt_key: str) -> str:
    if "task_id" in item:
        return str(item["task_id"])
    if "id" in item:
        return str(item["id"])
    if prompt_key in item:
        return hashlib.sha256(item[prompt_key].encode("utf-8")).hexdigest()[:16]
    raise KeyError(
        f"Cannot generate sample ID: missing '{prompt_key}', 'task_id', or 'id' "
        f"in item: {list(item.keys())}"
    )


def load_completed_samples(path: str, prompt_key: str) -> set:
    completed = set()
    if not os.path.exists(path):
        return completed
    try:
        with jsonlines.open(path, mode="r") as reader:
            for item in reader:
                try:
                    completed.add(get_sample_id(item, prompt_key))
                except (KeyError, jsonlines.InvalidLineError):
                    continue
    except Exception as e:
        print(f"Warning: Error reading {path}: {e}. Treating as empty checkpoint.")
        return set()
    return completed


def filter_pending_samples(data: List[dict], completed: set, prompt_key: str) -> List[dict]:
    pending = []
    for item in data:
        try:
            if get_sample_id(item, prompt_key) not in completed:
                pending.append(item)
        except KeyError:
            pending.append(item)
    return pending


def format_prompt(processor: AutoProcessor, question: str) -> str:
    user_content = (
        f"Here are some examples:{PRE_INSIGHT_FEWSHOT}\n\n"
        f"[Question]: {question.strip()}"
    )
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]
    return processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)


def read_json(path: str) -> List[dict]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    raise ValueError(f"Expected a JSON list in {path}")


def read_jsonl(path: str) -> List[dict]:
    items = []
    with jsonlines.open(path) as reader:
        for item in reader:
            items.append(item)
    return items


def read_input(path: str) -> List[dict]:
    if path.endswith(".jsonl"):
        return read_jsonl(path)
    elif path.endswith(".json"):
        return read_json(path)
    raise ValueError(f"Unsupported file format: {path}. Use .json or .jsonl")


def chunked(seq: List[dict], n: int) -> Iterable[List[dict]]:
    for i in range(0, len(seq), n):
        yield seq[i : i + n]


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


def load_model(
    base_model: str,
    lora_path: str | None = None,
    device_override: int | None = None,
) -> Tuple[Any, Any, torch.device]:
    """Load a model, optionally with a LoRA adapter.

    If lora_path is None, base_model is treated as a standalone (e.g. merged) model.
    """
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    device_map = {"": device_override if device_override is not None else 0}

    is_gptq = False
    quantize_cfg_path = os.path.join(base_model, "quantize_config.json")
    if os.path.isfile(quantize_cfg_path):
        is_gptq = True
        print(f"Detected GPTQ model at {base_model} — skipping BitsAndBytes quantization.")

    if not is_gptq:
        bnb_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
    else:
        bnb_cfg = None

    processor = AutoProcessor.from_pretrained(base_model)
    # When loading a merged model without processor_config.json, AutoProcessor
    # may return a bare tokenizer. In that case use it directly.
    if hasattr(processor, "tokenizer"):
        tokenizer = processor.tokenizer
    else:
        tokenizer = processor
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    config = AutoConfig.from_pretrained(base_model, trust_remote_code=True)
    # Merged models use Qwen3_5TextConfig; base models use Qwen3_5Config (multimodal)
    if "TextConfig" in type(config).__name__:
        AutoModelClass = AutoModelForCausalLM
    else:
        AutoModelClass = AutoModelForMultimodalLM

    if is_gptq:
        # GPTQ model — load directly, no BitsAndBytes needed
        model = AutoModelClass.from_pretrained(
            base_model,
            device_map=device_map,
        )
    else:
        model = AutoModelClass.from_pretrained(
            base_model,
            quantization_config=bnb_cfg,
            dtype=torch.bfloat16,
            device_map=device_map,
        )

    if lora_path is not None:
        model = PeftModel.from_pretrained(model, lora_path, torch_dtype=torch.bfloat16)

    model.eval()
    input_device = infer_input_device(model)
    model.config.use_cache = True
    return processor, model, input_device


def extract_assistant_reply(text: str) -> str:
    if ASSISTANT_TAG in text:
        reply = text.split(ASSISTANT_TAG, 1)[-1]
    else:
        reply = text

    reply = reply.split(IM_END, 1)[0] if IM_END in reply else reply
    reply = re.sub(rf"{IM_START}|{IM_END}|<\|endoftext\|>", "", reply).strip()

    think_match = re.match(r"^think\s*\n(.*?)\n/think\s*\n?", reply, re.DOTALL)
    if think_match:
        reply = reply[think_match.end():]

    fence_pos = reply.rfind("```")
    if fence_pos != -1:
        reply = reply[: fence_pos + 3]

    return reply.strip()


def generate_batch(
    model: PeftModel,
    processor: AutoProcessor,
    prompts: List[str],
    input_device: torch.device,
    max_prompt_len: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
    repetition_penalty: float,
) -> List[str]:
    tokenizer = processor.tokenizer if hasattr(processor, "tokenizer") else processor
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


def process_data(
    data: List[dict],
    args: argparse.Namespace,
    processor: AutoProcessor,
    model: PeftModel,
    input_device: torch.device,
    output_path: str,
    resume_mode: bool = False,
) -> None:
    num_batches = (len(data) + args.batch_size - 1) // args.batch_size
    file_mode = "a" if resume_mode else "w"

    with jsonlines.open(output_path, mode=file_mode) as writer:
        for chunk in tqdm(chunked(data, args.batch_size), total=num_batches, desc="batches"):
            prompts: List[str] = []
            owners: List[int] = []
            for idx, item in enumerate(chunk):
                if args.prompt_key not in item:
                    raise KeyError(f"Missing key '{args.prompt_key}' in sample: {list(item.keys())}")
                prompt_text = format_prompt(processor, item[args.prompt_key])
                for _ in range(args.num_versions):
                    prompts.append(prompt_text)
                    owners.append(idx)

            outputs = generate_batch(
                model=model,
                processor=processor,
                prompts=prompts,
                input_device=input_device,
                max_prompt_len=args.max_prompt_len,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
                repetition_penalty=args.repetition_penalty,
            )

            grouped: List[List[str]] = [[] for _ in chunk]
            for owner_idx, text in zip(owners, outputs):
                grouped[owner_idx].append(text)

            for item, gens in zip(chunk, grouped):
                item[args.output_key] = gens if args.num_versions > 1 else gens[0]
                # Compatibility alias: downstream dot_QA consumers read `insight`,
                # not `high_temp_insight`. Mirror the Llama3 script behavior.
                if args.output_key == "high_temp_insight" and "insight" not in item:
                    item["insight"] = gens[0] if gens else ""
                writer.write(item)

            writer._fp.flush()
            os.fsync(writer._fp.fileno())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("LoRA Qwen3.5-2B QA decomposition inference")
    parser.add_argument("--base_model", default="./models/Qwen3.5-2B")
    parser.add_argument("--lora_path", default=None, help="Path to LoRA adapter weights (omit to use merged model)")
    parser.add_argument("--input_jsonl", required=True, help="Input JSON/JSONL with a question field")
    parser.add_argument("--output_jsonl", required=True, help="Where to write generations")
    parser.add_argument("--prompt_key", default="question", help="Key containing the question text")
    parser.add_argument("--output_key", default="decomposition", help="Key to store model output")
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--num_versions", type=int, default=1, help="Samples per prompt")
    parser.add_argument("--max_prompt_len", type=int, default=1536)
    parser.add_argument("--max_new_tokens", type=int, default=600)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--repetition_penalty", type=float, default=1.05)
    parser.add_argument("--test", action="store_true", help="Test mode: only run on 4 samples")
    parser.add_argument("--force_restart", action="store_true", help="Ignore checkpoints and restart")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    data = read_input(args.input_jsonl)
    if args.test:
        data = data[:4]
        print(f"Test mode — using first {len(data)} samples")
    os.makedirs(os.path.dirname(args.output_jsonl) or ".", exist_ok=True)

    completed_ids = set()
    if not args.force_restart and os.path.exists(args.output_jsonl):
        completed_ids = load_completed_samples(args.output_jsonl, args.prompt_key)
        if completed_ids:
            original_count = len(data)
            data = filter_pending_samples(data, completed_ids, args.prompt_key)
            print(
                f"Resuming: {len(completed_ids)}/{original_count} completed. "
                f"Remaining: {len(data)}"
            )

    if not data:
        print("All samples already completed. Nothing to do.")
        return

    processor, model, input_device = load_model(
        base_model=args.base_model,
        lora_path=args.lora_path,
    )

    print(
        f"Loaded {len(data)} samples | batch_size={args.batch_size}, "
        f"num_versions={args.num_versions}, device={input_device}"
    )
    process_data(
        data=data,
        args=args,
        processor=processor,
        model=model,
        input_device=input_device,
        output_path=args.output_jsonl,
        resume_mode=bool(completed_ids),
    )
    print(f"Done. Wrote results to {args.output_jsonl}")


if __name__ == "__main__":
    main()
