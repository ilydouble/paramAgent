# Actor selection (defaults to the local vLLM server):
#   export ACTOR_MODEL="vllm:Meta-Llama-3.1-8B-Instruct"   # local vLLM server
#   export ACTOR_MODEL="deepseek:deepseek-v4-flash-0731"   # aiaaa.cc relay
#   ACTOR_MODEL="vllm:Qwen3.5-4B"

#
# Example — run the same benchmark with DeepSeek as the actor:
#   ACTOR_MODEL="deepseek:deepseek-v4-flash-0731" bash qa/run_simple_qa.sh


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
export RELAY_API_KEY="${RELAY_API_KEY_QA:-$RELAY_API_KEY}"

MODEL="${ACTOR_MODEL:-vllm:Meta-Llama-3.1-8B-Instruct}"
echo "Actor model: $MODEL"

# 传入给llm的是“prompt”  各自领域的simple.py中规定的
# 续传是按行数（位置）判断的，不是按内容去重  （utils.py）

ARGS=(
  --run_name "simple_mus_qwen3.5-4b"
  --root_dir "results/qa/simple/"
  --dataset_path "dataset/multihop/test/musique_100.jsonl"
  --strategy simple
  --language py
  --model "$MODEL"
  --pass_at_k 1
  --max_iters 1
  --verbose
  --judge_api_key "<REDACTED_HARDCODED_KEY>"
)
python qa/mainQA.py "${ARGS[@]}"
