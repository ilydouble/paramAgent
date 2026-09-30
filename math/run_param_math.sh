# Actor selection (defaults to the local vLLM server):
#   ACTOR_MODEL="vllm:Meta-Llama-3.1-8B-Instruct"   # local vLLM server
#   ACTOR_MODEL="deepseek:deepseek-v4-flash-0731"   # aiaaa.cc relay
#   ACTOR_MODEL="vllm:Qwen3.5-4B"
#
# Example — run ParamAgent with DeepSeek as the actor:
#   ACTOR_MODEL="deepseek:deepseek-v4-flash-0731" bash math/run_param_math.sh

set -e

cd "$(dirname "$0")/.."
export PYTHONPATH=.

# Route text-embedding-3-small (memory-bank retrieval, second pass) via the relay.
export OPENAI_BASE_URL="https://www.openai-labs.com/v1"
export OPENAI_API_KEY="<REDACTED_HARDCODED_KEY>"

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
INSIGHT_JSONL="samples_math_150/sample_math_150_sft.jsonl"

ARGS=(
  --run_name "sft_math_qwen3.5-4b"
  --root_dir "results/math/sft2/"
  --dataset_path "$DATASET_PATH"
  --strategy dot
  --language py
  --model "$MODEL"
  --pass_at_k 1
  --max_iters 6
  --use_expert_module
  --insight_json_path "$INSIGHT_JSONL"
  --device cuda:0
  --verbose
)
python math/mainMath_param.py "${ARGS[@]}"
