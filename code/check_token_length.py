#!/usr/bin/env python
"""Check the token-length distribution of the combined code dataset.

Computes the token count for every example using the same prompt template and
tokenizer as LoRA_Qwen35_Code_4090.py, then reports how many examples exceed
max_seq_len (default 2048) and binning statistics.

Usage:
    python code/check_token_length.py [--max_seq_len 2048] [--dataset_path dataset/code/train/code.json] [--model Qwen/Qwen3.5-2B]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

from transformers import AutoProcessor

SYSTEM_PROMPT = (
    "You are a coding education assistant. Given a Python function signature, "
    "identify and clearly explain common pitfalls and mistakes that may occur "
    "when implementing or using this function."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Token-length distribution checker for code dataset")
    parser.add_argument("--max_seq_len", type=int, default=2048, help="Maximum sequence length (default 2048)")
    parser.add_argument("--dataset_path", default="dataset/code/train/code.json", help="Path to the combined JSON dataset")
    parser.add_argument("--model", default="Qwen/Qwen3.5-2B", help="Tokenizer model name")
    parser.add_argument("--top_n", type=int, default=10, help="Show N longest examples")
    return parser.parse_args()


def _normalize_func_sign(item: dict) -> str:
    func_sign = str(item.get("func_sign", ""))
    docstring = str(item.get("docstring", ""))
    func_sign = re.sub(r"^```python\s*\n?", "", func_sign)
    func_sign = re.sub(r"\n?```\s*$", "", func_sign)
    func_sign = re.sub(r"^\[Function Signature\]:\s*\n?", "", func_sign)
    func_sign = func_sign.strip()
    has_docstring_in_sign = '"""' in func_sign or "'''" in func_sign
    if docstring and not has_docstring_in_sign:
        if not func_sign.endswith("\n"):
            func_sign += "\n"
        func_sign += f'    """{docstring}"""\n'
    return func_sign


def load_code_examples(path: str) -> list[dict[str, str]]:
    with open(path, encoding="utf-8") as file:
        raw = json.load(file)
    if not isinstance(raw, list):
        raise ValueError("The dataset top level must be a JSON list.")

    examples = []
    for index, item in enumerate(raw):
        if not isinstance(item, dict):
            raise ValueError(f"Dataset entry {index} must be a dict.")
        if not (item.get("func_sign") or item.get("prompt")):
            raise ValueError(f"Dataset entry {index} must contain non-empty 'func_sign' or 'prompt'.")
        func_sign = _normalize_func_sign(item)
        if "raw_texts" in item:
            answer = str(item["raw_texts"])
        elif "raw_text" in item:
            answer = str(item["raw_text"])
        else:
            pitfalls = str(item.get("pitfalls", "")).lstrip(":\n ")
            flawed_impl = str(item.get("flawed_impl", "")).lstrip(":\n ")
            if not pitfalls and not flawed_impl:
                raise ValueError(f"Dataset entry {index} must have 'raw_text', 'raw_texts', or 'pitfalls'+'flawed_impl'.")
            answer = pitfalls + "\n" + flawed_impl
        examples.append({"func_sign": func_sign, "answer": answer, "_index": index})
    return examples


def insert(sequence: list[int], position: int, value: int) -> list[int]:
    """Insert a value at position safely (clamp position to valid range)."""
    position = max(0, min(position, len(sequence)))
    return sequence[:position] + [value] + sequence[position:]


def main() -> None:
    args = parse_args()

    print(f"Loading tokenizer from {args.model} ...")
    processor = AutoProcessor.from_pretrained(args.model)
    tokenizer = processor.tokenizer
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    print(f"Loading dataset from {args.dataset_path} ...")
    examples = load_code_examples(args.dataset_path)
    print(f"Total examples: {len(examples)}")

    # ---- Tokenize every example and record lengths --------------------------------
    lengths: list[int] = []
    truncated_count = 0
    total_tokens = 0
    longest_examples: list[tuple[int, int, int]] = []  # (total_len, prompt_len, answer_len)

    for example in examples:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"FUNC_SIGNATURE:\n{example['func_sign']}"},
        ]
        prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        answer = example["answer"].strip() + tokenizer.eos_token
        prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        answer_ids = tokenizer(answer, add_special_tokens=False)["input_ids"]
        full_length = len(prompt_ids) + len(answer_ids)
        total_tokens += full_length
        lengths.append(full_length)

        if full_length > args.max_seq_len:
            truncated_count += 1
            longest_examples.append((full_length, len(prompt_ids), len(answer_ids)))

    # ---- Sort so worst offenders appear first ------------------------------------
    longest_examples.sort(key=lambda x: -x[0])

    # ---- Statistics --------------------------------------------------------------
    lengths_sorted = sorted(lengths)
    n = len(lengths_sorted)

    print()
    print("=" * 70)
    print("TOKEN-LENGTH STATISTICS")
    print("=" * 70)
    print(f"  Examples                : {n:,}")
    print(f"  Max sequence length     : {args.max_seq_len:,}")
    print(f"  Total tokens (all data) : {total_tokens:,}")
    print()
    print(f"  Min token count         : {lengths_sorted[0]:,}")
    print(f"  Max token count         : {lengths_sorted[-1]:,}")
    print(f"  Mean token count        : {sum(lengths_sorted) / n:,.1f}")
    print(f"  Median token count      : {lengths_sorted[n // 2]:,}")
    p90 = lengths_sorted[int(n * 0.90)]
    p95 = lengths_sorted[int(n * 0.95)]
    p99 = lengths_sorted[int(n * 0.99)]
    print(f"  90th percentile         : {p90:,}")
    print(f"  95th percentile         : {p95:,}")
    print(f"  99th percentile         : {p99:,}")
    print()
    print(f"  Exceeding {args.max_seq_len:>5,} tokens : {truncated_count:,}  "
          f"({truncated_count / n * 100:.2f}%)")

    # ---- Binning -----------------------------------------------------------------
    bins = [512, 1024, 1536, 2048, 2560, 3072, 4096, 6144, 8192]
    bin_counter = Counter()
    overflow_bin = 0
    for length in lengths:
        placed = False
        for b in bins:
            if length <= b:
                bin_counter[b] += 1
                placed = True
                break
        if not placed:
            overflow_bin += 1
    print()
    print("BINNING (cumulative ≤):")
    cumulative = 0
    for b in bins:
        cumulative += bin_counter[b]
        bar = "█" * (bin_counter[b] // max(1, n // 80))
        print(f"  ≤ {b:>5,}: {bin_counter[b]:>6,}  ({cumulative / n * 100:5.1f}%)  {bar}")
    if overflow_bin:
        cumulative += overflow_bin
        print(f"  > {bins[-1]:>5,}: {overflow_bin:>6,}  ({cumulative / n * 100:5.1f}%)")

    # ---- Longest examples --------------------------------------------------------
    if longest_examples and args.top_n > 0:
        print()
        print("=" * 70)
        print(f"TOP {min(args.top_n, len(longest_examples))} LONGEST EXAMPLES (exceeding {args.max_seq_len} tokens)")
        print("=" * 70)
        for rank, (total, prompt_len, answer_len) in enumerate(longest_examples[: args.top_n], 1):
            over_by = total - args.max_seq_len
            print(f"  #{rank}: total={total:,}  (prompt={prompt_len:,}  answer={answer_len:,}  "
                  f"over_by={over_by:,})")


if __name__ == "__main__":
    main()
