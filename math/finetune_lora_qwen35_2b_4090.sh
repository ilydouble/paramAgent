#!/usr/bin/env bash
# Single RTX 4090 QLoRA training for Qwen/Qwen3.5-2B on math pitfalls.
# To resume from a checkpoint, add: --resume_from_checkpoint ./lora-qwen3.5-2b/lora-qwen3.5-2b-math/checkpoint-XXX
set -euo pipefail

cd "$(dirname "$0")/.."
export PYTHONPATH=.
export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false

python math/LoRA_Qwen35_Math_4090.py \
  --dataset_path dataset/math/train/math.json \
  --output_dir ./lora-qwen3.5-2b/lora-qwen3.5-2b-math2 \
  --base_model ./models/Qwen3.5-2B \
  --num_epochs 2 \
  --per_device_batch_size 2 \
  --grad_accum_steps 8 \
  --lr 2e-4 \
  --lora_r 32 \
  --lora_alpha 64 \
  --lora_dropout 0.05 \
  --warmup_ratio 0.03 \
  --save_steps 200 \
  --max_seq_len 2048 \
  --val_ratio 0.05 \
  --seed 42
