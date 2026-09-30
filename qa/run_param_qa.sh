# Actor selection (defaults to the local vLLM server):
#   ACTOR_MODEL="vllm:Meta-Llama-3.1-8B-Instruct"   # local vLLM server
#   ACTOR_MODEL="deepseek:deepseek-v4-flash-0731"   # aiaaa.cc relay
#   ACTOR_MODEL="vllm:Qwen3.5-4B"
#
# Example — run ParamAgent with DeepSeek as the actor:
#   ACTOR_MODEL="deepseek:deepseek-v4-flash-0731" bash qa/run_param_qa.sh
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
export RELAY_API_KEY="${RELAY_API_KEY_QA:-$RELAY_API_KEY}"


MODEL="${ACTOR_MODEL:-vllm:Meta-Llama-3.1-8B-Instruct}"
echo "Actor model: $MODEL"
DATASET_PATH="dataset/multihop/test/hotpot_qa.jsonl"
INSIGHT_JSONL="generated_reflections/merged/sft/qa/hotpot_insights_qwen35_2b-merged.jsonl"

ARGS=(
  --run_name "sft_hotpot_qwen3.5-4b"
  --root_dir "results/qa/sft/"
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
python qa/mainQA_parametric.py "${ARGS[@]}"
