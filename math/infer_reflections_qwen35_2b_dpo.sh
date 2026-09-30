#!/bin/bash
set -e

# 使用 DPO LoRA 模型生成 per-sample QA insights。
#
# 加载 base model (models/Qwen3.5-2B-qa-merged) +
# DPO LoRA adapter (lora-qwen3.5-2b-dpo/lora-qwen3.5-2b-qa-dpo)
# 在本地 GPU 上推理，无需调用外部 API。
#
# NOTE: 与 SFT 版本的 infer_reflections_qwen35_2b.sh 相比，
#       这里多了 --lora_path 参数，base_model 使用未 GPTQ 的 merged 模型
#       （因为 4-bit 量化 + LoRA adapter 更灵活，可同时用于推理和微调）。

cd "$(dirname "$0")/.."
export PYTHONPATH=.

BASE_MODEL="lora-qwen3.5-2b/lora-qwen3.5-2b-math2"
LORA_PATH="lora-qwen3.5-2b-dpo/lora-qwen3.5-2b-math-dpo3"
INPUT_JSONL="dataset/new/math_baseline_w_sample80_r0=0.jsonl"
OUTPUT_DIR="generated_reflections/new"
OUTPUT_JSONL="$OUTPUT_DIR/math_r0=0_math.jsonl"

# 生成参数与 SFT 版本保持一致
TEMPERATURE=0.2
MAX_NEW_TOKENS=1024

mkdir -p "$OUTPUT_DIR"

python math/LoRA_Qwen35_Math_4090_inference.py \
  --base_model "$BASE_MODEL" \
  --input_jsonl "$INPUT_JSONL" \
  --lora_path "$LORA_PATH" \
  --output_jsonl "$OUTPUT_JSONL" \
  --prompt_key problem \
  --output_key pitfalls_high_temp \
  --temperature "$TEMPERATURE" \
  --max_new_tokens "$MAX_NEW_TOKENS" \
  --num_versions 1 \
  --batch_size 1