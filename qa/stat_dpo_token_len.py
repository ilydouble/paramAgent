#!/usr/bin/env python
"""Check DPO token lengths against max_length / max_prompt_length.

Replicates format_pair() from LoRA_Qwen35_QA_DPO_4090.py exactly, so the
numbers match what TRL will see. Run on the training machine from repo root:

    python qa/stat_dpo_token_len.py \
        --dataset_path dataset/multihop/train/dpo.jsonl \
        --base_model ./models/Qwen3.5-2B-qa-merged \
        --max_length 3072 --max_prompt_length 2048
"""

import argparse
import json
import statistics

from transformers import AutoProcessor

# Same import mechanism as LoRA_Qwen35_QA_DPO_4090.py (qa/ is on sys.path)
from LoRA_Qwen35_QA_4090 import PRE_INSIGHT_FEWSHOT, SYSTEM_PROMPT


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", default="dataset/multihop/train/dpo.jsonl")
    parser.add_argument("--base_model", default="./models/Qwen3.5-2B-qa-merged")
    parser.add_argument("--max_length", type=int, default=3072)
    parser.add_argument("--max_prompt_length", type=int, default=2048)
    args = parser.parse_args()

    processor = AutoProcessor.from_pretrained(args.base_model, trust_remote_code=True)
    tokenizer = getattr(processor, "tokenizer", processor)


    rows = []
    with open(args.dataset_path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))

    records = []  # (prompt_len, chosen_total, rejected_total)
    for row in rows:
        # == format_pair() in LoRA_Qwen35_QA_DPO_4090.py, kept identical ==
        user_content = (
            f"Here are some examples:{PRE_INSIGHT_FEWSHOT}\n\n"
            f"[Question]: {row['question']}"
        )
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]
        prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        prompt_len = len(tokenizer(prompt, add_special_tokens=False)["input_ids"])
        chosen_len = len(tokenizer(row["chosen_decomposition"].strip() + tokenizer.eos_token,
                                   add_special_tokens=False)["input_ids"])
        rejected_len = len(tokenizer(row["rejected_decomposition"].strip() + tokenizer.eos_token,
                                     add_special_tokens=False)["input_ids"])
        records.append((prompt_len, prompt_len + chosen_len, prompt_len + rejected_len))

    n = len(records)
    prompts = sorted(r[0] for r in records)
    chosen = sorted(r[1] for r in records)
    rejected = sorted(r[2] for r in records)

    def pct(sorted_lens, p):
        return sorted_lens[min(int(n * p / 100), n - 1)]

    print(f"样本数: {n}")
    print(f"prompt 长度: min={prompts[0]}  P90={pct(prompts, 90)}  "
          f"P99={pct(prompts, 99)}  max={prompts[-1]}")
    print(f"prompt+chosen: min={chosen[0]}  P90={pct(chosen, 90)}  "
          f"P99={pct(chosen, 99)}  max={chosen[-1]}")
    print(f"prompt+rejected: min={rejected[0]}  P90={pct(rejected, 90)}  "
          f"P99={pct(rejected, 99)}  max={rejected[-1]}")

    over_prompt = sum(1 for r in records if r[0] > args.max_prompt_length)
    over_chosen = sum(1 for r in records if r[1] > args.max_length)
    over_rejected = sum(1 for r in records if r[2] > args.max_length)
    over_any = sum(1 for r in records
                   if r[0] > args.max_prompt_length or r[1] > args.max_length or r[2] > args.max_length)

    print(f"\n超过 max_prompt_length={args.max_prompt_length} 的 prompt: {over_prompt} ({over_prompt/n*100:.1f}%)")
    print(f"超过 max_length={args.max_length} 的 chosen: {over_chosen} ({over_chosen/n*100:.1f}%)")
    print(f"超过 max_length={args.max_length} 的 rejected: {over_rejected} ({over_rejected/n*100:.1f}%)")
    print(f"任一超限（真正会被截断）的样本: {over_any} ({over_any/n*100:.1f}%)")

    # 最长的几条，方便人工查看截断会切到哪里
    worst = sorted(range(n), key=lambda i: -max(records[i]))[:5]
    print("\n最长的 5 条 (idx, prompt, +chosen, +rejected):")
    for i in worst:
        print(f"  #{i}: prompt={records[i][0]}  +chosen={records[i][1]}  +rejected={records[i][2]}")


if __name__ == "__main__":
    main()
