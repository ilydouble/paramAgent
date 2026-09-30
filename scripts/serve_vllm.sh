#!/bin/bash
# scripts/serve_vllm.sh
#
# Start a persistent vLLM inference server for Meta-Llama-3.1-8B-Instruct.
# Run this ONCE on the 4090 server; the model stays loaded in VRAM.
# All scripts (eval_pitfalls_v2.py, main.py via model_factory, etc.) can
# then connect via --api_base http://localhost:8000/v1.
#
# Usage:
#   bash scripts/serve_vllm.sh
#   bash scripts/serve_vllm.sh models/Qwen3.5-9B 8000      # model + port
#
# Verify:
#   curl http://localhost:8000/v1/models
#
# (Optional) With LoRA support:
#   bash scripts/serve_vllm.sh --enable-lora --lora-modules code-pitfall=/path/to/lora

set -euo pipefail

# Positional args: [model_path] [port], both optional. They are consumed here
# and shifted off, so "$@" below carries only genuine extra vllm flags —
# otherwise the model path and port would reach `vllm serve` a second time and
# be rejected as "unrecognized arguments".
#
# A leading '-' marks an extra vllm flag rather than a positional, so
# `bash scripts/serve_vllm.sh --enable-lora` still works.
MODEL_DIR=""
PORT=""
if [ $# -ge 1 ] && [ "${1#-}" = "$1" ]; then MODEL_DIR="$1"; shift; fi
if [ $# -ge 1 ] && [ "${1#-}" = "$1" ]; then PORT="$1"; shift; fi
MODEL_DIR="${MODEL_DIR:-models/Meta-Llama-3.1-8B-Instruct}"
PORT="${PORT:-8000}"

# Check that the model directory exists
if [ ! -d "$MODEL_DIR" ]; then
    echo "ERROR: Model directory not found: $MODEL_DIR"
    echo "Usage: $0 [model_path] [port] [extra vllm flags...]"
    echo "  (default model: models/Meta-Llama-3.1-8B-Instruct)"
    echo "  (default port:  8000)"
    exit 1
fi

echo "Starting vLLM server..."
echo "  Model: $MODEL_DIR"
echo "  Port:  $PORT"
echo "  dtype: bfloat16 | max_model_len: 16384 | gpu_memory_utilization: 0.9"
echo ""
echo "The server will remain running. Press Ctrl+C to stop."
echo "Connect scripts via: --api_base http://localhost:$PORT/v1"
echo ""

# Derive a clean model name (last path component) so clients can use
# --model Meta-Llama-3.1-8B-Instruct instead of the full model path.
CLEAN_NAME="$(basename "$MODEL_DIR")"

vllm serve "$MODEL_DIR" \
    --dtype bfloat16 \
    --max-model-len 16384 \
    --gpu-memory-utilization 0.9 \
    --served-model-name "$CLEAN_NAME" \
    --host 0.0.0.0 \
    --port "$PORT" \
    "$@"