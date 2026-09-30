#!/bin/bash
# code/run_simple_code.sh — simple strategy (code domain, MBPP).
#
# Actor selection (defaults to the local vLLM server):
#   export ACTOR_MODEL="vllm:Meta-Llama-3.1-8B-Instruct"   # local vLLM server
#   export ACTOR_MODEL="deepseek:deepseek-v4-flash-0731"   # aiaaa.cc relay
#   ACTOR_MODEL="vllm:Qwen3.5-4B"
#
# Example — run the same benchmark with DeepSeek as the actor:
#   ACTOR_MODEL="relay:qwen3.8-27b" bash code/run_simple_code.sh


set -e

cd "$(dirname "$0")/.."
export PYTHONPATH=.

# Load relay credentials (RELAY_BASE_URL, RELAY_API_KEY) if a .env exists.
if [ -f .env ]; then
  set -a
  . ./.env
  set +a
fi
export RELAY_BASE_URL="${RELAY_BASE_URL:-https://aiaaa.cc/v1}"
export RELAY_API_KEY="${RELAY_API_KEY_CODE:-$RELAY_API_KEY}"

MODEL="${ACTOR_MODEL:-vllm:Meta-Llama-3.1-8B-Instruct}"
echo "Actor model: $MODEL"


ARGS=(
  --run_name "simple_mbpp_qwen3.5-4b"
  --root_dir "results/code/simple/"
  --dataset_path "dataset/code/test/mbpp-py.jsonl"
  --strategy simple
  --language py
  --model "$MODEL"
  --pass_at_k 1
  --max_iters 1
  --verbose
)
python code/main.py "${ARGS[@]}"
