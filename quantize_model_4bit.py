#!/usr/bin/env python3
"""
General-purpose GPTQ 4-bit quantization for any BF16 merged model.

Quantizes once, then inference loads the 4-bit weights directly — no re-quantization.

Usage:
    # QA model (calibrate on QA questions)
    CUDA_VISIBLE_DEVICES=0 python quantize_model_4bit.py \
      --model_path models/Qwen3.5-2B-qa-merged \
      --calib_path dataset/multihop/test/hotpot_.jsonl \
      --calib_key question

    # Code model (calibrate on code snippets)
    CUDA_VISIBLE_DEVICES=0 python quantize_model_4bit.py \
      --model_path models/Qwen3.5-2B-code-merged \
      --calib_path dataset/code/test/code_tests.jsonl \
      --calib_key prompt

    # Math model (calibrate on math problems)
    CUDA_VISIBLE_DEVICES=0 python quantize_model_4bit.py \
      --model_path models/Qwen3.5-2B-math-merged \
      --calib_path dataset/math/test/math_test.jsonl \
      --calib_key problem

    # With custom output path and calibration size
    CUDA_VISIBLE_DEVICES=0 python quantize_model_4bit.py \
      --model_path models/Qwen3.5-2B-qa-merged \
      --output_path my_models/qa-4bit \
      --calib_path dataset/multihop/test/hotpot_.jsonl \
      --calib_key question \
      --calib_size 256
"""

from __future__ import annotations

import argparse
import json
import os
import shutil

import torch
from optimum.gptq import GPTQQuantizer
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_calibration_texts(
    path: str,
    key: str | None,
    size: int,
) -> list[str]:
    """Load up to `size` text samples from a JSON or JSONL file.

    If the file is a JSON list, each element is used directly (if str) or
    indexed by ``key`` (if dict).  If the file is JSONL, each line is a dict
    indexed by ``key``.  If ``key`` is None, we look for a common textual
    field (``"question"``, ``"prompt"``, ``"problem"``, ``"text"``, ``"code"``)
    as a fallback.
    """
    texts: list[str] = []
    ext = os.path.splitext(path)[1].lower()

    if ext == ".jsonl":
        with open(path) as f:
            for line in f:
                if len(texts) >= size:
                    break
                item = json.loads(line)
                if isinstance(item, str):
                    texts.append(item)
                elif isinstance(item, dict):
                    val = _extract(item, key)
                    if val is not None:
                        texts.append(val)
    elif ext == ".json":
        with open(path) as f:
            data = json.load(f)
        if isinstance(data, list):
            for item in data:
                if len(texts) >= size:
                    break
                if isinstance(item, str):
                    texts.append(item)
                elif isinstance(item, dict):
                    val = _extract(item, key)
                    if val is not None:
                        texts.append(val)
        elif isinstance(data, dict):
            # Try to find a list field
            for k in ("questions", "prompts", "problems", "data", "examples"):
                if k in data and isinstance(data[k], list):
                    for item in data[k]:
                        if len(texts) >= size:
                            break
                        val = _extract(item, key) if isinstance(item, dict) else item
                        if val is not None:
                            texts.append(val)
                    break
    else:
        raise ValueError(f"Unsupported calibration file format: {ext}")

    if not texts:
        raise ValueError(
            f"No calibration texts found in {path} (key={key!r}). "
            "Use --calib_key to specify the correct field."
        )

    print(f"  Loaded {len(texts)} calibration samples from {path} (key={key!r})")
    return texts


_FALLBACK_KEYS = ("question", "prompt", "problem", "text", "code", "input", "content")


def _extract(item: dict, key: str | None) -> str | None:
    if key is not None:
        val = item.get(key)
        return str(val) if val is not None else None
    # Auto-detect a textual field
    for k in _FALLBACK_KEYS:
        if k in item and isinstance(item[k], str) and len(item[k]) > 20:
            return item[k]
    return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="GPTQ 4-bit quantization for any BF16 merged model.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python quantize_model_4bit.py --model_path models/Qwen3.5-2B-qa-merged \\\n"
            "      --calib_path dataset/multihop/test/hotpot_.jsonl --calib_key question\n"
        ),
    )
    parser.add_argument(
        "--model_path",
        required=True,
        help="Path to the BF16 merged model directory.",
    )
    parser.add_argument(
        "--output_path",
        default=None,
        help="Output directory for the 4-bit GPTQ model. "
        "Defaults to <model_path>-4bit-gptq.",
    )
    parser.add_argument(
        "--calib_path",
        required=True,
        help="Path to calibration data (.json or .jsonl).",
    )
    parser.add_argument(
        "--calib_key",
        default=None,
        help="Field name in each JSON entry containing the calibration text. "
        "If omitted, auto-detects from common names (question, prompt, problem, text, code, ...).",
    )
    parser.add_argument(
        "--calib_size",
        type=int,
        default=128,
        help="Number of calibration samples to use (default: 128).",
    )
    parser.add_argument(
        "--group_size",
        type=int,
        default=128,
        help="GPTQ group size (default: 128).",
    )
    parser.add_argument(
        "--desc_act",
        action="store_true",
        help="Enable desc_act (slightly better quality, slower inference). Off by default.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    output_path = args.output_path or f"{args.model_path.rstrip('/')}-4bit-gptq"
    os.makedirs(output_path, exist_ok=True)

    # ---- 1. Load tokenizer ----
    print(f"Loading tokenizer from {args.model_path} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ---- 2. Load calibration texts ----
    print(f"Loading calibration samples (size={args.calib_size}) ...")
    calib_texts = load_calibration_texts(
        path=args.calib_path,
        key=args.calib_key,
        size=args.calib_size,
    )

    # ---- 3. Tokenize ----
    print("Tokenizing calibration data ...")
    calib_encodings = tokenizer(
        calib_texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512,
    )
    calib_dataset = [calib_encodings]

    # ---- 4. Load BF16 model ----
    print(f"Loading BF16 model from {args.model_path} ...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    # ---- 5. Quantize ----
    print("Running GPTQ quantization (this may take a few minutes) ...")
    quantizer = GPTQQuantizer(
        bits=4,
        group_size=args.group_size,
        damp_percent=0.01,
        desc_act=args.desc_act,
        sym=True,
        block_name_to_quantize="model.layers",
        module_name_prepended=False,
    )
    model = quantizer.quantize_model(model, calib_dataset)

    # ---- 6. Save ----
    print(f"Saving 4-bit GPTQ model to {output_path} ...")
    model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)

    # Copy chat template if present
    src_jinja = os.path.join(args.model_path, "chat_template.jinja")
    if os.path.exists(src_jinja):
        shutil.copy2(src_jinja, os.path.join(output_path, "chat_template.jinja"))

    # Print size
    total_bytes = sum(
        os.path.getsize(os.path.join(dp, f))
        for dp, _dn, fn in os.walk(output_path)
        for f in fn
    )
    print(f"Done! 4-bit model saved to {output_path} ({total_bytes / 1024 / 1024 / 1024:.2f} GB)")


if __name__ == "__main__":
    main()