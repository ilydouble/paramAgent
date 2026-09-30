# 生成训练数据集

## 生成全部数据集
python qa/gen_multihop_decomposition_data.py

## 只生成某个数据集
python qa/gen_multihop_decomposition_data.py --datasets mus

## 测试模式（每个数据集只跑 5 条）
python qa/gen_multihop_decomposition_data.py --limit 5

## 强制 MuSiQue 也用 API 生成
python qa/gen_multihop_decomposition_data.py --mus_use_api


# 训练模型
python qa/stat_token_len.py    # 确定max_length
bash qa/finetune_lora_qwen35_2b_4090.sh

# 生成loss图像
python qa/plot_loss.py --log_dir lora-qwen3.5-2b/lora-qwen3.5-2b-qa/loss_history.json

# 合并模型权重
python qa/merge_lora_qwen35_2b_code.py

# 模型推理测试


# QA Domain 五个策略流程

入口脚本 `qa/run_qa.sh`，通过 `qa/mainQA.py` 调度，统一使用 `MultiHopQAGenerator`（`generators/MiltuhopQA_generate.py`）生成答案，`QAExecutor`（`executors/`）评估正确性。

---

## 1. simple

**文件**: `qa/simple_QA.py` → `run_simple`

**流程**:
1. 对每条数据，调用 `gen.func_impl(question, context, model, "simple")` 生成答案
2. 调用 `exe.evaluate(answer, golden_answer)` 判断对错
3. `pass_at_k=1`，`max_iters=1`，只做一次，无迭代改进

**Prompt**:
- System: `MQ_SIMPLE_CHAT_INSTRUCTION`（`generators/MiltuhopQA_generate.py:22`）
- User: `[question]: {question}\n\n[context]:\n{context}`
- 若有 `question_decomposition`（预生成的insight），则 System 改为 `question_parsing_instruction`（`generators/generator_utils.py:97`），User 额外加入 `[question intent decomposition]:\n{decomposition}`

---

## 2. reflexion

**文件**: `qa/reflexion_QA.py` → `run_reflexion`

**流程**:
1. 用 `gen.func_impl(..., "simple")` 生成初始答案
2. 若答错，进入迭代循环（最多 `max_iters=8` 次）：
   - 调用 `gen.self_reflection(question, answer, context, feedback, model)` 生成自我反思
   - 调用 `gen.func_impl(..., "reflexion", prev_answers, feedback, self_reflection)` 生成新答案
   - 评估新答案，若对则退出
3. 每轮只生成一条反思，逐条改进

**Prompt**:
- 初始答案同 simple
- 反思 System: `generators/generator_utils.py:245`（"You are a self-reflection assistant..."）
- 改进 System: `MQ_REFLEXION_CHAT_INSTRUCTION`（`generators/MiltuhopQA_generate.py:30`）
- 改进 User: `[question] + [previous answer] + [why wrong] + [feedback] + [context]`

---

## 3. dot (Diversity of Thought)

**文件**: `qa/dot_QA.py` → `run_dot`

**流程**:
1. 用 `gen.func_impl(..., "simple")` 生成初始答案
2. 若答错，进入迭代循环（最多 `max_iters=8` 次）：
   - 调用 `gen.self_reflection_diverse(question, answer, context, feedback, model, diverse_reflections)` 一次性生成多条多样性反思（用 `\n\n` 分割）
   - 过滤掉过短反思，只取第一条（`ref_id=0`）来改进
   - 调用 `gen.func_impl(..., "reflexion", prev_answers, feedback, self_reflection)` 生成新答案
   - 评估新答案，若对则退出
3. 与 reflexion 的区别：每轮生成多条反思（而非一条），但只使用第一条

**Prompt**:
- 多样性反思 System: `generators/generator_utils.py:305`（"produce no more than 5 new concise reflections..."）
- 其余同 reflexion

---

## 4. dot_bank

**文件**: `qa/dot_bank_QA.py` → `run_dot_bank`

**流程**（两阶段）:

**第一阶段**（与 dot 类似，但增加 memory bank）:
1. 生成初始答案，若对则将 trajectory 存入 `positive_trajectories`
2. 若答错，生成多样性反思，对每条反思（最多2条）生成改进答案
3. 按正确性加权采样选择最终答案（正确=1.0, 错误=0.1）
4. 将 trajectory（含 question/context/reflection embedding）存入 memory bank（正确→positive, 错误→negative）
5. 保存第一阶段日志，收集失败问题

**第二阶段**（对失败问题使用 memory bank）:
1. 用 OpenAI embedding 检索 memory bank 中最相似的 positive trajectory
2. 生成初始答案（同 simple）
3. 若答错，生成多样性反思时，再按 reflection embedding 检索相似 trajectory
4. 若检索到含 `prev_answer` 的 trajectory，构造 few-shot example 注入 `gen.func_impl(..., fewshot_example=QA_FEW_SHOT)`
5. 评估改进答案，更新 memory bank

**Prompt**:
- Few-shot example 格式（`dot_bank_QA.py:436`）:
  ```
  Example:
  [Question]: {question}
  [Previous Answer]: {prev_answer}
  [Reflection]: {reflection}
  [Improved Answer]: {gen_answer}
  ```
- 其余同 dot/reflexion

---

## 5. paramagent

**文件**: `qa/mainQA_parametric.py` → `dot_QA_parametric_with_bank.py` → `run_dot`

**流程**:
- 与 dot_bank 两阶段流程基本相同，但核心区别是**注入了参数化的 question decomposition（insight）**：
1. 从 `insight_json_path` 加载预生成的 insight，或用 `QADecomposer`（LoRA微调模型）实时生成
2. 在 `gen.func_impl(..., question_decomposition=refined_insights)` 中注入 insight
3. 生成多样性反思时使用 `gen.self_reflection_diverse_parametric(..., insights)`，反思也基于 insight
4. 改进答案时使用 `MQ_REFLEXION_CHAT_INSTRUCTION_PARAMETRIC`（`generators/MiltuhopQA_generate.py:40`），User 中额外包含 `[question intent decomposition]`

**Prompt**:
- 带 insight 的初始 System: `question_parsing_instruction`（`generators/generator_utils.py:97`）
- 带 insight 的反思 System: `generators/generator_utils.py:377`（"...extracted insights for the question..."）
- 带 insight 的改进 System: `MQ_REFLEXION_CHAT_INSTRUCTION_PARAMETRIC`（`generators/MiltuhopQA_generate.py:40`）
- 带 insight 的改进 User: `[question] + [question intent decomposition] + [previous answer] + [why wrong] + [feedback] + [context]`

---

## Simple 策略的 Prompt 位置

- **System prompt**: `generators/MiltuhopQA_generate.py:22` — `MQ_SIMPLE_CHAT_INSTRUCTION`
- **User prompt 构造**: `generators/generator_utils.py:119-127` — 无 insight 时为 `[question] + [context]`
- **带 insight 时的 System**: `generators/generator_utils.py:97` — `question_parsing_instruction`
- **带 insight 时的 User**: `generators/generator_utils.py:108-114` — `[question] + [question intent decomposition] + [context]`
