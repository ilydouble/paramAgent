#!/bin/bash
set -e

# This uses the merged Qwen3.5-2B-code model to generate per-sample code pitfalls.
# Mirrors code/infer_reflections_llama3_8b.sh but targets the Qwen3.5-2B merged model.
#
# NOTE: this calls code/LoRA_Qwen35_Code_4090_inference.py, NOT the Llama3 script.
#       The --lora_path is omitted because we use a merged model.

cd "$(dirname "$0")/.."
export PYTHONPATH=.

BASE_MODEL="models/Qwen3.5-2B"
LORA_PATH="lora-qwen3.5-2b/lora-qwen3.5-2b-code3"
INPUT_JSONL="benchmarks/mbpp-py.jsonl"
OUTPUT_DIR="generated_reflections/sft/code"
OUTPUT_JSONL="$OUTPUT_DIR/mbpp-py_pitfalls_qwen35_2b-tem0.2.jsonl"

# Matching code Llama3 convention: low temperature for accuracy,
# 600 tokens for pitfalls-only output (no flawed implementations).
# humaneval.jsonl uses 'prompt' key (func_sign); mbpp.jsonl uses 'text' (question).
# Adjust --prompt_key when switching input files.
TEMPERATURE=0.2
MAX_NEW_TOKENS=600

mkdir -p "$OUTPUT_DIR"

python code/LoRA_Qwen35_Code_4090_inference.py \
  --base_model "$BASE_MODEL" \
  --lora_path "$LORA_PATH" \
  --input_jsonl "$INPUT_JSONL" \
  --output_jsonl "$OUTPUT_JSONL" \
  --prompt_key prompt \
  --output_key high_temp_pitfall \
  --temperature "$TEMPERATURE" \
  --max_new_tokens "$MAX_NEW_TOKENS" \
  --num_versions 1 \
  --batch_size 1