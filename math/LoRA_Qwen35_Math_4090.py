#!/usr/bin/env python
"""Single-GPU QLoRA fine-tuning of Qwen/Qwen3.5-2B for math pitfalls.

This script intentionally does not replace the Llama training scripts.  It is
for the text-only ``dataset/math/math.json`` data format used by this project.
Qwen3.5 is a multimodal model, so it is loaded with AutoModelForMultimodalLM;
the dataset itself only supplies text and does not train the vision encoder.
"""

from __future__ import annotations

import argparse
import inspect
import math
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from shared.parammem_prompts import author_constants, tokenize_sft
from shared.qwen_sft import qwen_lora_plan, save_qwen_lora_report

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForMultimodalLM,
    AutoProcessor,
    BitsAndBytesConfig,
    DefaultDataCollator,
    Trainer,
    TrainingArguments,
)


SYSTEM_PROMPT, _ = author_constants("math", "train")


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
    parser = argparse.ArgumentParser(description="QLoRA fine-tune Qwen3.5-2B on math pitfalls")
    parser.add_argument("--dataset_path", default="dataset/math/math.json")
    parser.add_argument("--output_dir", default="./lora-qwen3.5-2b/lora-qwen3.5-2b-math-author-sft")
    parser.add_argument("--base_model", default="Qwen/Qwen3.5-2B")
    parser.add_argument("--num_epochs", type=float, default=2)
    parser.add_argument("--per_device_batch_size", type=int, default=1)
    parser.add_argument("--grad_accum_steps", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1.5e-5)
    parser.add_argument("--max_seq_len", type=int, default=2048)
    parser.add_argument("--preserve_eos", action="store_true",
                        help="Qwen revision: supervise real EOS tokens; mask only appended padding.")
    parser.add_argument("--qwen_adapted", action="store_true",
                        help="Fresh-run Qwen3.5 language projection coverage, including EOS supervision.")
    parser.add_argument("--qwen_include_mlp", action="store_true",
                        help="Optional attention-plus-MLP comparison; requires --qwen_adapted.")
    parser.add_argument(
        "--dynamic_padding",
        action="store_true",
        help="Experimental: pad each batch to its longest sequence. Disabled by default for Qwen3.5 fallback-kernel stability.",
    )
    parser.add_argument("--lora_r", type=int, default=64)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--bf16", action="store_true", default=True)
    parser.add_argument("--no_bf16", dest="bf16", action="store_false")
    parser.add_argument("--tf32", action="store_true", default=True)
    parser.add_argument("--no_tf32", dest="tf32", action="store_false")
    parser.add_argument("--save_steps", type=int, default=200)
    parser.add_argument("--resume_from_checkpoint", default=None, help="Path to a checkpoint directory to resume training from")
    parser.add_argument("--val_ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()
    if args.qwen_include_mlp and not args.qwen_adapted:
        parser.error("--qwen_include_mlp requires --qwen_adapted")
    if args.qwen_adapted:
        args.preserve_eos = True
        if args.output_dir == parser.get_default("output_dir"):
            suffix = "-qwen-adapted-mlp-sft" if args.qwen_include_mlp else "-qwen-adapted-sft"
            args.output_dir = args.output_dir.replace("-author-sft", suffix)
        if args.resume_from_checkpoint or Path(args.output_dir).exists():
            parser.error("--qwen_adapted requires a new output directory and no resume checkpoint")
    return args


def load_examples(path: str) -> list[dict[str, str]]:
    with open(path, encoding="utf-8") as file:
        raw = json.load(file)
    if not isinstance(raw, list):
        raise ValueError("The dataset top level must be a JSON list.")

    examples = []
    for index, item in enumerate(raw):
        if not isinstance(item, dict) or not item.get("problem") or not item.get("pitfalls"):
            raise ValueError(f"Dataset entry {index} must contain non-empty 'problem' and 'pitfalls'.")
        examples.append({"problem": str(item["problem"]), "pitfalls": str(item["pitfalls"])})
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
    """Choose Qwen language-model projection names actually present in this version.

    Qwen3.5 has both Gated DeltaNet and attention layers.  Selecting by the
    module names discovered from the loaded checkpoint keeps this script usable
    across compatible Transformers releases and avoids adapting visual modules.
    """
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
    qwen_plan = qwen_lora_plan(model, include_mlp=args.qwen_include_mlp) if args.qwen_adapted else None
    targets = qwen_plan["target_regex"] if qwen_plan else select_lora_modules(model)
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
    if qwen_plan:
        save_qwen_lora_report(model, qwen_plan, args.output_dir)
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model.config.use_cache = False
    model.print_trainable_parameters()

    data = Dataset.from_list(load_examples(args.dataset_path)).shuffle(seed=args.seed)
    split = data.train_test_split(test_size=args.val_ratio, seed=args.seed)

    def tokenize(example: dict[str, Any]) -> dict[str, list[int]]:
        return tokenize_sft(processor, tokenizer, "math", example["problem"],
                            example["pitfalls"], args.max_seq_len, args.dynamic_padding,
                            preserve_eos=args.preserve_eos)

    train_data = split["train"].map(tokenize, remove_columns=split["train"].column_names)
    eval_data = split["test"].map(tokenize, remove_columns=split["test"].column_names)

    # Transformers releases shipped by cloud images do not all expose exactly
    # the same TrainingArguments keywords (notably warmup_ratio and the name of
    # the evaluation strategy). Keep the QLoRA settings while filtering only
    # unsupported optional keywords instead of failing before training starts.
    warmup_steps = math.ceil(args.warmup_ratio * math.ceil(len(train_data) / (args.per_device_batch_size * args.grad_accum_steps)) * args.num_epochs)
    print(f"Computed warmup_steps={warmup_steps} from warmup_ratio={args.warmup_ratio}")

    training_kwargs: dict[str, Any] = {
        "output_dir": args.output_dir,
        "num_train_epochs": args.num_epochs,
        "per_device_train_batch_size": args.per_device_batch_size,
        "per_device_eval_batch_size": args.per_device_batch_size,
        "gradient_accumulation_steps": args.grad_accum_steps,
        "learning_rate": args.lr,
        "lr_scheduler_type": "cosine",
        "warmup_steps": warmup_steps,
        "bf16": args.bf16,
        "tf32": args.tf32,
        "optim": "adamw_torch",
        "logging_steps": 2,
        "save_strategy": "epoch",
        "save_steps": args.save_steps,
        "eval_strategy": "epoch",
        "eval_steps": args.save_steps,
        "save_total_limit": 4,
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
    # New releases call this processing_class; older releases call it tokenizer.
    if "processing_class" in trainer_parameters:
        trainer_kwargs["processing_class"] = tokenizer
    elif "tokenizer" in trainer_parameters:
        trainer_kwargs["tokenizer"] = tokenizer
    trainer = Trainer(**trainer_kwargs)
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    with open(Path(args.output_dir) / "loss_history.json", "w", encoding="utf-8") as file:
        json.dump(trainer.state.log_history, file, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
