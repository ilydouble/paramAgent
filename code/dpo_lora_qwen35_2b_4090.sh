#!/usr/bin/env bash
# DPO on the merged code-SFT Qwen3.5-2B checkpoint, for a single RTX 4090.
# The model takes func_sign as input and is trained to output chosen_pitfalls
# over rejected_pitfalls from dataset/code/train/dpo.jsonl.
#
# Tuned for the code DPO dataset (530 samples, short func_sign inputs):
#   - lr 1e-6            (math DPO uses 3e-6; only 1/4 the data -> more conservative)
#   - max_prompt_length 1024 (func_sign avg ~50 tokens; 2048 wastes memory)
#   - warmup_ratio 0.03  (only ~33 steps/epoch at grad_accum 16)
#   - max_chars 3000     (drop ~10 anomalously long/noisy pairs >3000 chars)
# If you see forgetting or degraded val loss, try lr 5e-7.
set -euo pipefail

cd "$(dirname "$0")/.."
export PYTHONPATH=.
export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false

python code/LoRA_Qwen35_Code_DPO_4090.py \
  --dataset_path dataset/code/train/dpo.jsonl \
  --base_model ./models/Qwen3.5-2B-code-merged3 \
  --output_dir ./lora-qwen3.5-2b-dpo/lora-qwen3.5-2b-code-dpo30 \
  --num_epochs 1 \
  --per_device_batch_size 1 \
  --grad_accum_steps 16 \
  --learning_rate 1e-6 \
  --beta 0.3 \
  --warmup_ratio 0.03 \
  --lora_r 16 \
  --lora_alpha 32 \
  --lora_dropout 0.05 \
  --max_length 3072 \
  --max_prompt_length 1024 \
  --max_chars 3000 \
  --val_ratio 0.10 \
  --save_strategy steps \
  --save_steps 10 \
  --seed 42


# 若出现明显遗忘或验证集变差，可降到 lr 5e-7  或升 --beta 0.5
