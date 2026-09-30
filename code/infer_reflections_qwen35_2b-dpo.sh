#!/bin/bash
set -e

# This uses the merged Qwen3.5-2B-code model to generate per-sample code pitfalls.
# Mirrors code/infer_reflections_llama3_8b.sh but targets the Qwen3.5-2B merged model.
#
# NOTE: this calls code/LoRA_Qwen35_Code_4090_inference.py, NOT the Llama3 script.
#       The --lora_path is omitted because we use a merged model.

cd "$(dirname "$0")/.."
export PYTHONPATH=.

BASE_MODEL="models/Qwen3.5-2B-code-merged3"
LORA_PATH="lora-qwen3.5-2b-dpo/lora-qwen3.5-2b-code-dpo30"
INPUT_JSONL="dataset/new/code_baseline_w_samole48_r0=0.jsonl"
OUTPUT_DIR="generated_reflections/new"
OUTPUT_JSONL="$OUTPUT_DIR/code_r0=0_code.jsonl"

# Matching code Llama3 convention: low temperature for accuracy,
# 600 tokens for pitfalls-only output (no flawed implementations).
# humaneval.jsonl uses 'prompt' key (func_sign); mbpp.jsonl uses 'text' (question).
# Adjust --prompt_key when switching input files.
TEMPERATURE=0.2
MAX_NEW_TOKENS=1000

mkdir -p "$OUTPUT_DIR"

python code/LoRA_Qwen35_Code_4090_inference.py \
  --base_model "$BASE_MODEL" \
  --lora_path "$LORA_PATH" \
  --input_jsonl "$INPUT_JSONL" \
  --output_jsonl "$OUTPUT_JSONL" \
  --prompt_key func_sign \
  --output_key high_temp_pitfall \
  --temperature "$TEMPERATURE" \
  --max_new_tokens "$MAX_NEW_TOKENS" \
  --num_versions 1 \
  --batch_size 1