# Actor selection (defaults to the local vLLM server):
#   ACTOR_MODEL="vllm:Meta-Llama-3.1-8B-Instruct"   # local vLLM server
#   ACTOR_MODEL="dee#pseek:deepseek-v4-flash-0731"   # aiaaa.cc relay
   ACTOR_MODEL="vllm:Qwen3.5-4B"
#
# Example — run DoT-Bank with DeepSeek as the actor:
#   ACTOR_MODEL="deepseek:deepseek-v4-flash-0731" bash math/run_bank_math.sh

set -e

cd "$(dirname "$0")/.."
export PYTHONPATH=.

# Route text-embedding-3-small (memory-bank retrieval, both passes) via the relay.
export OPENAI_BASE_URL="https://www.openai-labs.com/v1"
export OPENAI_API_KEY="<REDACTED_HARDCODED_KEY>"

# Load relay credentials (RELAY_BASE_URL, RELAY_API_KEY) if a .env exists.
if [ -f .env ]; then
  set -a
  . ./.env
  set +a
fi
export RELAY_BASE_URL="${RELAY_BASE_URL:-https://aiaaa.cc/v1}"
export RELAY_API_KEY="${RELAY_API_KEY_MATH:-$RELAY_API_KEY}"


export VLLM_REPETITION_PENALTY=1.05


MODEL="${ACTOR_MODEL:-vllm:Meta-Llama-3.1-8B-Instruct}"
echo "Actor model: $MODEL"
DATASET_PATH="samples_math_150/sample_math_150.jsonl"

ARGS=(
  --run_name "dot_bank_math_qwen3.5-4b"
  --root_dir "results/math/dot_bank2/"
  --dataset_path "$DATASET_PATH"
  --strategy dot_bank
  --language py
  --model "$MODEL"
  --pass_at_k 1
  --max_iters 6
  --device cuda:0
  --verbose
)
python math/mainMath.py "${ARGS[@]}"
