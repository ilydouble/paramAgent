#!/bin/bash
set -e

# This uses the merged Qwen3.5-2B-qa model to generate per-sample QA insights.
# Mirrors qa/infer_reflections_llama3_8b.sh but targets the Qwen3.5-2B merged model,
# and keeps the same sampling behavior (temperature / max_new_tokens) as the Llama3 run.
#
# NOTE: this calls qa/LoRA_Qwen35_QA_4090_inference.py, NOT the Llama3 script.
#       Argument names differ: use --input_jsonl / --output_jsonl (not --input_json).

cd "$(dirname "$0")/.."
export PYTHONPATH=.

BASE_MODEL="models/Qwen3.5-2B-qa-merged"
INPUT_JSONL="dataset/multihop/test/musique_100.jsonl"
OUTPUT_DIR="generated_reflections/sft/qa"
OUTPUT_JSONL="$OUTPUT_DIR/mus_insights_qwen35_2b-tem0.2.jsonl"

# Keep the same generation behavior as the original Llama3 run.
TEMPERATURE=0.2
MAX_NEW_TOKENS=1000

mkdir -p "$OUTPUT_DIR"

python qa/LoRA_Qwen35_QA_4090_inference.py \
  --base_model "$BASE_MODEL" \
  --input_jsonl "$INPUT_JSONL" \
  --output_jsonl "$OUTPUT_JSONL" \
  --prompt_key question \
  --output_key high_temp_insight \
  --temperature "$TEMPERATURE" \
  --max_new_tokens "$MAX_NEW_TOKENS" \
  --num_versions 1 \
  --batch_size 1
```
