#!/usr/bin/env bash
# DPO on the merged math-SFT Qwen3.5-2B checkpoint, for a single RTX 4090.
set -euo pipefail

cd "$(dirname "$0")/.."
export PYTHONPATH=.
export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false

python math/LoRA_Qwen35_Math_DPO_4090.py \
  --dataset_path dataset/math/train/dpo.jsonl \
  --base_model ./models/Qwen3.5-2B-math-merged2 \
  --output_dir ./lora-qwen3.5-2b-dpo/lora-qwen3.5-2b-math-dpo3 \
  --num_epochs 1 \
  --per_device_batch_size 1 \
  --grad_accum_steps 16 \
  --learning_rate 3e-6 \
  --beta 0.3 \
  --warmup_ratio 0.05 \
  --lora_r 16 \
  --lora_alpha 32 \
  --lora_dropout 0.05 \
  --max_length 3072 \
  --max_prompt_length 2048 \
  --val_ratio 0.10 \
  --save_strategy steps \
  --save_steps 10 \
  --seed 42


# 若出现明显遗忘或验证集变差，可降到lr 3e-6