#!/usr/bin/env python
"""Single-GPU QLoRA fine-tuning of Qwen/Qwen3.5-2B for QA decomposition.

This script follows the same pattern as math/LoRA_Qwen35_Math_4090.py but
adapted for the QA domain using dataset/multihop/multihop_decomposition_all.jsonl.
Qwen3.5 is a multimodal model, so it is loaded with AutoModelForMultimodalLM;
the dataset itself only supplies text and does not train the vision encoder.
"""

from __future__ import annotations

import argparse
import inspect
import json
import subprocess
from pathlib import Path
from typing import Any

import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from training_splits import add_split_arguments, example_identity, frozen_datasets, record_training_completion

import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    set_seed,
    AutoModelForMultimodalLM,
    AutoProcessor,
    BitsAndBytesConfig,
    DefaultDataCollator,
    Trainer,
    TrainingArguments,
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


class DynamicCausalLMCollator:
    """Pad only to the longest example in a batch (rounded for Tensor Cores)."""

    def __init__(self, pad_token_id: int, pad_to_multiple_of: int = 8) -> None:
        self.pad_token_id = pad_token_id
        self.pad_to_multiple_of = pad_to_multiple_of

    def __call__(self, features: list[dict[str, list[int]]]) -> dict[str, torch.Tensor]:
        max_length = max(len(feature["input_ids"]) for feature in features)
        if self.pad_to_multiple_of:
            max_length = ((max_length + self.pad_to_multiple_of - 1) // self.pad_to_multiple_of) * self.pad_to_multiple_of
        batch = {"input_ids": [], "attention_mask": [], "labels": []}
        for feature in features:
            padding = max_length - len(feature["input_ids"])
            batch["input_ids"].append(feature["input_ids"] + [self.pad_token_id] * padding)
            batch["attention_mask"].append(feature["attention_mask"] + [0] * padding)
            batch["labels"].append(feature["labels"] + [-100] * padding)
        return {name: torch.tensor(values, dtype=torch.long) for name, values in batch.items()}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="QLoRA fine-tune Qwen3.5-2B on QA decomposition")
    parser.add_argument("--dataset_path", default="dataset/multihop/multihop_decomposition_all.jsonl")
    parser.add_argument("--output_dir", default="./lora-qwen3.5-2b/lora-qwen3.5-2b-qa")
    parser.add_argument("--base_model", default="Qwen/Qwen3.5-2B")
    parser.add_argument("--num_epochs", type=float, default=3)
    parser.add_argument("--per_device_batch_size", type=int, default=2)
    parser.add_argument("--grad_accum_steps", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--max_seq_len", type=int, default=2048)
    parser.add_argument(
        "--dynamic_padding",
        action="store_true",
        help="Experimental: pad each batch to its longest sequence.",
    )
    parser.add_argument("--lora_r", type=int, default=32)
    parser.add_argument("--lora_alpha", type=int, default=64)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--bf16", action="store_true", default=True)
    parser.add_argument("--no_bf16", dest="bf16", action="store_false")
    parser.add_argument("--tf32", action="store_true", default=True)
    parser.add_argument("--no_tf32", dest="tf32", action="store_false")
    parser.add_argument("--save_steps", type=int, default=200)
    parser.add_argument("--resume_from_checkpoint", default=None, help="Path to a checkpoint directory to resume training from")
    parser.add_argument("--seed", type=int, default=42)
    add_split_arguments(parser)
    return parser.parse_args()


def load_qa_examples(path: str) -> list[dict[str, str]]:
    examples = []
    with open(path, encoding="utf-8") as file:
        for index, line in enumerate(file):
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            if not isinstance(item, dict):
                raise ValueError(f"Line {index} must be a JSON object.")
            if not item.get("question") or not item.get("decomposition"):
                raise ValueError(f"Line {index} must contain non-empty 'question' and 'decomposition'.")
            examples.append({**example_identity(item, item.get("question", item.get("prompt", ""))), "question": str(item["question"]), "decomposition": str(item["decomposition"])})
    return examples


def cuda_diagnostic() -> str:
    """Return actionable version information when PyTorch cannot start CUDA."""
    try:
        driver = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip().splitlines()[0]
    except (OSError, subprocess.CalledProcessError, IndexError):
        driver = "unavailable"
    return f"NVIDIA driver={driver}; PyTorch CUDA build={torch.version.cuda}"


def select_lora_modules(model: torch.nn.Module) -> list[str]:
    """Choose Qwen language-model projection names actually present in this version."""
    candidates = {
        "q_proj", "k_proj", "v_proj", "o_proj", "in_proj", "out_proj",
        "gate_proj", "up_proj", "down_proj", "x_proj", "dt_proj",
    }
    found = set()
    for name, module in model.named_modules():
        if "visual" in name or "vision" in name or not isinstance(module, torch.nn.Linear):
            continue
        leaf = name.rsplit(".", 1)[-1]
        if leaf in candidates:
            found.add(leaf)
    if not found:
        raise RuntimeError(
            "No supported Qwen projection modules were found. Install the current "
            "Transformers main branch as described in requirements-qwen35.txt."
        )
    return sorted(found)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    split = frozen_datasets(load_qa_examples(args.dataset_path), args, stage="sft", domain="qa")
    if not torch.cuda.is_available():
        raise RuntimeError(
            "PyTorch cannot initialize CUDA. " + cuda_diagnostic() + ". "
            "For a CUDA 12.6 driver, run math/setup_qwen35_4090_cuda126.sh "
            "in this Python environment, then rerun training."
        )
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("This configuration requires BF16; an RTX 4090 supports it.")

    torch.backends.cuda.matmul.allow_tf32 = args.tf32
    processor = AutoProcessor.from_pretrained(args.base_model)
    tokenizer = processor.tokenizer
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForMultimodalLM.from_pretrained(
        args.base_model,
        quantization_config=bnb_config,
        dtype=torch.bfloat16,
        device_map={"": 0},
    )
    model = prepare_model_for_kbit_training(model)
    targets = select_lora_modules(model)
    print(f"LoRA target modules: {targets}")
    model = get_peft_model(
        model,
        LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bias="none",
            target_modules=targets,
            task_type="CAUSAL_LM",
        ),
    )
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model.config.use_cache = False
    model.print_trainable_parameters()

    def tokenize(example: dict[str, Any]) -> dict[str, list[int]]:
        user_content = (
            f"Here are some examples:{PRE_INSIGHT_FEWSHOT}\n\n"
            f"[Question]: {example['question']}"
        )
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]
        prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        answer = example["decomposition"].strip() + tokenizer.eos_token
        prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        answer_ids = tokenizer(answer, add_special_tokens=False)["input_ids"]
        input_ids = (prompt_ids + answer_ids)[: args.max_seq_len]
        labels = ([-100] * len(prompt_ids) + answer_ids)[: args.max_seq_len]
        attention_mask = [1] * len(input_ids)
        result = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }
        if args.dynamic_padding:
            result["length"] = len(input_ids)
        else:
            pad = args.max_seq_len - len(input_ids)
            if pad > 0:
                result["input_ids"] += [tokenizer.pad_token_id] * pad
                result["attention_mask"] += [0] * pad
                result["labels"] += [-100] * pad
        return result

    train_data = split["train"].map(tokenize, remove_columns=split["train"].column_names)
    eval_data = split["val"].map(tokenize, remove_columns=split["val"].column_names)

    warmup_steps = int(args.warmup_ratio * len(train_data) / (args.per_device_batch_size * args.grad_accum_steps))
    print(f"Computed warmup_steps={warmup_steps} from warmup_ratio={args.warmup_ratio}")

    training_kwargs: dict[str, Any] = {
        "output_dir": args.output_dir,
        "seed": args.seed,
        "data_seed": args.seed,
        "num_train_epochs": args.num_epochs,
        "per_device_train_batch_size": args.per_device_batch_size,
        "per_device_eval_batch_size": args.per_device_batch_size,
        "gradient_accumulation_steps": args.grad_accum_steps,
        "learning_rate": args.lr,
        "lr_scheduler_type": "cosine",
        "warmup_steps": warmup_steps,
        "bf16": args.bf16,
        "tf32": args.tf32,
        "optim": "paged_adamw_8bit",
        "logging_steps": 5,
        "save_strategy": "steps",
        "save_steps": args.save_steps,
        "eval_strategy": "steps",
        "eval_steps": args.save_steps,
        "save_total_limit": 3,
        "load_best_model_at_end": True,
        "metric_for_best_model": "eval_loss",
        "greater_is_better": False,
        "gradient_checkpointing": True,
        "report_to": "none",
        "remove_unused_columns": False,
    }
    if args.dynamic_padding:
        training_kwargs["group_by_length"] = True
        training_kwargs["length_column_name"] = "length"
    accepted = inspect.signature(TrainingArguments.__init__).parameters
    skipped = sorted(set(training_kwargs) - set(accepted))
    if skipped:
        print(f"TrainingArguments does not support {skipped}; continuing without them.")
    training_args = TrainingArguments(**{key: value for key, value in training_kwargs.items() if key in accepted})
    trainer_kwargs: dict[str, Any] = {
        "model": model,
        "args": training_args,
        "train_dataset": train_data,
        "eval_dataset": eval_data,
        "data_collator": (
            DynamicCausalLMCollator(tokenizer.pad_token_id)
            if args.dynamic_padding else DefaultDataCollator()
        ),
    }
    trainer_parameters = inspect.signature(Trainer.__init__).parameters
    if "processing_class" in trainer_parameters:
        trainer_kwargs["processing_class"] = tokenizer
    elif "tokenizer" in trainer_parameters:
        trainer_kwargs["tokenizer"] = tokenizer
    trainer = Trainer(**trainer_kwargs)
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    record_training_completion(args)
    with open(Path(args.output_dir) / "loss_history.json", "w", encoding="utf-8") as file:
        json.dump(trainer.state.log_history, file, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
