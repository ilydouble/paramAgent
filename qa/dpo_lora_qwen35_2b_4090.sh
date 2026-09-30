#!/usr/bin/env bash
# DPO on the merged QA-SFT Qwen3.5-2B checkpoint, for a single RTX 4090.
set -euo pipefail

cd "$(dirname "$0")/.."
export PYTHONPATH=.
export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false

python qa/LoRA_Qwen35_QA_DPO_4090.py \
  --dataset_path dataset/multihop/train/dpo.jsonl \
  --base_model ./models/Qwen3.5-2B-qa-merged \
  --output_dir ./lora-qwen3.5-2b-dpo/lora-qwen3.5-2b-qa-dpo3 \
  --num_epochs 1 \
  --per_device_batch_size 1 \
  --grad_accum_steps 16 \
  --learning_rate 2e-6 \
  --beta 0.1 \
  --warmup_ratio 0.05 \
  --lora_r 16 \
  --lora_alpha 32 \
  --lora_dropout 0.05 \
  --max_length 3072 \
  --max_prompt_length 2048 \
  --val_ratio 0.05 \
  --save_strategy steps \
  --save_steps 25 \
  --seed 42
