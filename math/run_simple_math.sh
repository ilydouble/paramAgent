# Actor selection (defaults to the local vLLM server):
#   export ACTOR_MODEL="vllm:Meta-Llama-3.1-8B-Instruct"   # local vLLM server
#   export ACTOR_MODEL="deepseek:deepseek-v4-flash-0731"   # aiaaa.cc relay
#   ACTOR_MODEL="vllm:Qwen3.5-4B"

#
# Example — run the same benchmark with DeepSeek as the actor:
#   ACTOR_MODEL="deepseek:deepseek-v4-flash-0731" bash math/run_simple_math.sh

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
export RELAY_API_KEY="${RELAY_API_KEY_MATH:-$RELAY_API_KEY}"

MODEL="${ACTOR_MODEL:-vllm:Meta-Llama-3.1-8B-Instruct}"
echo "Actor model: $MODEL"


export VLLM_REPETITION_PENALTY=1.05



ARGS=(
  --run_name "simple_math_qwen3.5-4b"
  --root_dir "results/math/simple2/"
  --dataset_path "samples_math_150/sample_math_150.jsonl"
  --strategy simple
  --language py
  --model "$MODEL"
  --pass_at_k 1
  --max_iters 1
  --verbose
  --judge_api_key "<REDACTED_HARDCODED_KEY>"
)
python math/mainMath.py "${ARGS[@]}"
