#!/usr/bin/env python
"""Single-GPU QLoRA DPO training for Qwen3.5 code pitfalls.

The policy starts from the merged code SFT checkpoint.  A new LoRA adapter is
trained with DPO, while the same merged SFT checkpoint with that adapter
disabled is used as the frozen reference policy.  This is standard PEFT DPO
and avoids loading a second copy of the model on a 24 GB RTX 4090.

The preference pairs come from ``dataset/code/train/dpo.jsonl`` with the
project schema (``func_sign`` / ``chosen_pitfalls`` / ``rejected_pitfalls``).
The model sees the function signature as input (the same input the code SFT
and inference scripts use) and is steered toward ``chosen_pitfalls`` over
``rejected_pitfalls``.  The prompt matches the code SFT and inference scripts
exactly (system prompt plus ``FUNC_SIGNATURE:`` marker, thinking disabled),
so the preference signal aligns with how the merged checkpoint is used
downstream.
"""

from __future__ import annotations

import argparse
import inspect
import math
import json
from pathlib import Path
from typing import Any

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoProcessor, BitsAndBytesConfig
from trl import DPOConfig, DPOTrainer
# Reuse the code SFT prompt/cleanup helpers so this stage stays aligned with SFT.
from LoRA_Qwen35_Code_4090 import SYSTEM_PROMPT, _normalize_func_sign


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DPO Qwen3.5-2B code pitfalls on one RTX 4090")
    parser.add_argument("--dataset_path", default="dataset/code/train/dpo.jsonl")
    parser.add_argument("--base_model", default="./models/Qwen3.5-2B-code-merged3")
    parser.add_argument("--output_dir", default="./lora-qwen3.5-2b-dpo/lora-qwen3.5-2b-code-dpo")
    parser.add_argument("--num_epochs", type=float, default=1.0)
    parser.add_argument("--per_device_batch_size", type=int, default=1)
    parser.add_argument("--grad_accum_steps", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=3e-6)
    parser.add_argument("--beta", type=float, default=0.3)
    parser.add_argument("--warmup_ratio", type=float, default=0.05)
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--max_length", type=int, default=3072)
    parser.add_argument("--max_prompt_length", type=int, default=1024)
    parser.add_argument("--max_chars", type=int, default=3000,
                        help="Drop preference pairs whose chosen or rejected completion "
                             "exceeds this many characters (outlier/noise filtering). "
                             "Set to a large value to disable.")
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--save_strategy", choices=("epoch", "steps"), default="steps")
    parser.add_argument("--save_steps", type=int, default=10,
                        help="Only used when --save_strategy steps.")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_preferences(path: str, max_chars: int = 3000) -> list[dict[str, str]]:
    """Convert the code DPO JSONL schema to TRL's prompt/chosen/rejected schema.

    The code preference pairs are keyed by the function signature: the model
    reads ``func_sign`` (the same input the code SFT and inference scripts
    use, cleaned of markdown fences and the noisy ``[Function Signature]:``
    prefix) and is steered toward ``chosen_pitfalls`` over ``rejected_pitfalls``.

    Parameters
    ----------
    path : str
        Path to the JSONL file.
    max_chars : int
        Skip pairs where either the chosen or rejected completion exceeds this
        many characters.  The DPO dataset has a few anomalously long entries
        (6000-12000 characters) that are likely noise (e.g. flawed impl
        appended to chosen).  The default 3000 covers ~97% of the data.
    """
    preferences: list[dict[str, str]] = []
    skipped = 0
    with open(path, encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            required = ("func_sign", "chosen_pitfalls", "rejected_pitfalls")
            missing = [key for key in required if not isinstance(row.get(key), str) or not row[key].strip()]
            if missing:
                raise ValueError(f"{path}:{line_number} is missing non-empty fields: {', '.join(missing)}")
            chosen, rejected = row["chosen_pitfalls"].strip(), row["rejected_pitfalls"].strip()
            if chosen == rejected:
                raise ValueError(f"{path}:{line_number} has identical chosen and rejected completions.")
            if len(chosen) > max_chars or len(rejected) > max_chars:
                skipped += 1
                print(f"  Skipping line {line_number}: chosen={len(chosen)} chars / "
                      f"rejected={len(rejected)} chars exceeds max_chars={max_chars}")
                continue
            preferences.append({
                "question": _normalize_func_sign(row),
                "chosen": chosen,
                "rejected": rejected,
            })
    if skipped:
        print(f"  Filtered out {skipped} anomalously long pairs, keeping {len(preferences)}.")
    if len(preferences) < 2:
        raise ValueError("DPO requires at least two valid preference pairs after filtering.")
    return preferences


def select_lora_modules(model: torch.nn.Module) -> list[str]:
    candidates = {
        "q_proj", "k_proj", "v_proj", "o_proj", "in_proj", "out_proj",
        "gate_proj", "up_proj", "down_proj", "x_proj", "dt_proj",
    }
    found = {
        name.rsplit(".", 1)[-1]
        for name, module in model.named_modules()
        if isinstance(module, torch.nn.Linear)
        and "vision" not in name
        and "visual" not in name
        and name.rsplit(".", 1)[-1] in candidates
    }
    if not found:
        raise RuntimeError("Could not find Qwen language projection modules for LoRA.")
    return sorted(found)


def apply_chat_template_no_thinking(processor: Any, messages: list[dict[str, str]]) -> str:
    """Apply Qwen's chat template with thinking mode disabled.

    The code inference script builds prompts with ``enable_thinking=False``;
    matching that here keeps the DPO prompt distribution identical to the one
    the merged checkpoint sees downstream.  Older template implementations may
    not accept the keyword, in which case the plain template (used by the SFT
    script) is a safe fallback.
    """
    try:
        return processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
    except TypeError:
        return processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this QLoRA DPO script.")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("This configuration requires BF16 (RTX 4090 is supported).")

    torch.backends.cuda.matmul.allow_tf32 = True
    processor = AutoProcessor.from_pretrained(args.base_model)
    # A merged model may ship without processor_config.json, in which case
    # AutoProcessor returns a bare tokenizer; use it directly.
    tokenizer = getattr(processor, "tokenizer", processor)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        quantization_config=quantization,
        dtype=torch.bfloat16,
        device_map={"": 0},
    )
    model = prepare_model_for_kbit_training(model)
    target_modules = select_lora_modules(model)
    print(f"DPO LoRA target modules: {target_modules}")
    model = get_peft_model(
        model,
        LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bias="none",
            target_modules=target_modules,
            task_type="CAUSAL_LM",
        ),
    )
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model.config.use_cache = False
    model.print_trainable_parameters()

    raw_data = Dataset.from_list(load_preferences(args.dataset_path, max_chars=args.max_chars)).shuffle(seed=args.seed)
    split = raw_data.train_test_split(test_size=args.val_ratio, seed=args.seed)

    def format_pair(example: dict[str, Any]) -> dict[str, str]:
        # Exact code-SFT prompt distribution: same system prompt and
        # FUNC_SIGNATURE marker, no few-shot, thinking disabled. The longer
        # sequence budget preserves the chosen/rejected pitfalls completions.
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"FUNC_SIGNATURE:\n{example['question']}"},
        ]
        prompt = apply_chat_template_no_thinking(processor, messages)
        return {
            "prompt": prompt,
            "chosen": example["chosen"] + tokenizer.eos_token,
            "rejected": example["rejected"] + tokenizer.eos_token,
        }

    train_data = split["train"].map(format_pair, remove_columns=split["train"].column_names)
    eval_data = split["test"].map(format_pair, remove_columns=split["test"].column_names)

    dpo_kwargs: dict[str, Any] = {
        "output_dir": args.output_dir,
        "num_train_epochs": args.num_epochs,
        "per_device_train_batch_size": args.per_device_batch_size,
        "per_device_eval_batch_size": args.per_device_batch_size,
        "gradient_accumulation_steps": args.grad_accum_steps,
        "learning_rate": args.learning_rate,
        "lr_scheduler_type": "cosine",
        "warmup_ratio": args.warmup_ratio,
        "beta": args.beta,
        "max_length": args.max_length,
        "max_prompt_length": args.max_prompt_length,
        "truncation_mode": "keep_end",
        "padding_free": False,
        "bf16": True,
        "tf32": True,
        "optim": "paged_adamw_8bit",
        "gradient_checkpointing": True,
        "logging_steps": 5,
        "save_strategy": args.save_strategy,
        "save_steps": args.save_steps if args.save_strategy == "steps" else 500,
        # None keeps every checkpoint (a LoRA adapter is ~40 MB, so this is
        # cheap and lets us pick the best step by downstream eval instead of
        # by eval_loss, which saturates quickly in DPO).
        "save_total_limit": None,
        "load_best_model_at_end": True,
        "metric_for_best_model": "eval_loss",
        "greater_is_better": False,
        "report_to": "none",
        "remove_unused_columns": False,
    }
    accepted = inspect.signature(DPOConfig.__init__).parameters
    evaluation_key = "eval_strategy" if "eval_strategy" in accepted else "evaluation_strategy"
    dpo_kwargs[evaluation_key] = args.save_strategy
    if args.save_strategy == "steps":
        dpo_kwargs["eval_steps"] = args.save_steps
    if "warmup_ratio" not in accepted and "warmup_steps" in accepted:
        estimated_steps = math.ceil(len(train_data) / (args.per_device_batch_size * args.grad_accum_steps)) * math.ceil(args.num_epochs)
        dpo_kwargs["warmup_steps"] = max(1, round(estimated_steps * args.warmup_ratio))
    skipped = sorted(set(dpo_kwargs) - set(accepted))
    if skipped:
        print(f"DPOConfig does not support {skipped}; continuing without them.")
    dpo_args = DPOConfig(**{key: value for key, value in dpo_kwargs.items() if key in accepted})
    # ref_model=None is intentional: TRL disables this newly attached PEFT
    # adapter to obtain the frozen merged-SFT reference policy.
    trainer_kwargs: dict[str, Any] = {
        "model": model,
        "ref_model": None,
        "args": dpo_args,
        "train_dataset": train_data,
        "eval_dataset": eval_data,
    }
    trainer_parameters = inspect.signature(DPOTrainer.__init__).parameters
    if "processing_class" in trainer_parameters:
        trainer_kwargs["processing_class"] = tokenizer
    elif "tokenizer" in trainer_parameters:
        trainer_kwargs["tokenizer"] = tokenizer
    trainer = DPOTrainer(**trainer_kwargs)
    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    with open(Path(args.output_dir) / "dpo_log_history.json", "w", encoding="utf-8") as file:
        json.dump(trainer.state.log_history, file, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
