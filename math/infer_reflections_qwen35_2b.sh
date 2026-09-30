#!/bin/bash
set -e

# This uses the merged Qwen3.5-2B-math model to generate per-sample math pitfalls.
# Mirrors math/infer_reflections_llama3_8b.sh but targets the Qwen3.5-2B merged model.
#
# NOTE: this calls math/LoRA_Qwen35_Math_4090_inference.py, NOT the Llama3 script.
#       The --lora_path is omitted because we use a merged model.

cd "$(dirname "$0")/.."
export PYTHONPATH=.

BASE_MODEL="models/Qwen3.5-2B"
LORA_PATH="lora-qwen3.5-2b/lora-qwen3.5-2b-math2"
INPUT_JSONL="dataset/math/test/OlymMATH-EN-HARD.jsonl"
OUTPUT_DIR="generated_reflections/sft/math"
OUTPUT_JSONL="$OUTPUT_DIR/olymhard_pitfalls_qwen35_2b—tem1.0.jsonl"

# High temperature for diverse generation (matching the code/qa Qwen35 scripts).
# NOTE: math Llama3 uses default 0.2, but for pitfalls_high_temp with 8 variants
# we use 0.8 for diversity.
TEMPERATURE=1.0
# Default 600 is too small — math pitfalls median ≈ 763 tokens, p95 ≈ 864 tokens.
# Use 1024 to avoid truncation (covers 100 % of qwen1 pitfalls).
MAX_NEW_TOKENS=1024

mkdir -p "$OUTPUT_DIR"

python math/LoRA_Qwen35_Math_4090_inference.py \
  --base_model "$BASE_MODEL" \
  --lora_path "$LORA_PATH" \
  --input_jsonl "$INPUT_JSONL" \
  --output_jsonl "$OUTPUT_JSONL" \
  --prompt_key problem \
  --output_key pitfalls_high_temp \
  --temperature "$TEMPERATURE" \
  --max_new_tokens "$MAX_NEW_TOKENS" \
  --num_versions 8 \
  --batch_size 1