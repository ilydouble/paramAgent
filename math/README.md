## 用sft微调后的qwen生成数据
python math/gen_math_pitfalls_qwen35.py --test
python math/gen_math_pitfalls_qwen35.py

python math/eval_pitfalls_v2.py; python math/eval_pitfalls_v2.py --dataset_path dataset/math/qwen2.jsonl;python math/eval_pitfalls_v2.py --dataset_path dataset/math/qwen3.jsonl

## 五种策略流程详解
run_math.sh 依次运行 5 种策略，复杂度逐步递增：

### 1. simple — 单次直答
simple_Math.py

Problem → LLM 生成答案 → evaluate → 结束
    max_iters=1, 不重试
    可以选配 use_parsing，从预生成的 pitfalls json 中读取 mistake insights 注入提示词
    基线方法，最快最便宜
### 2. reflexion — 反思链
reflexion_Math.py

Problem → 生成答案 → evaluate
  ├─ 正确 → 结束
  └─ 错误 → self_reflection(题目+错误答案+feedback)
              → 用反思指导重新生成 → evaluate
              → 重复 max_iters=5 次
单链反思：每次只生成 1 个反思，基于它改 1 次
反思内容：「为什么错了 + 怎么修正」
### 3. dot (Diverse of Thought) — 多样反思
dot_Math.py

Problem → 生成答案 → evaluate
  ├─ 正确 → 结束
  └─ 错误 → self_reflection_diverse → 生成 N 个不同反思
              ├─ 反思1 → 指导生成新答案1 → evaluate
              ├─ 反思2 → 指导生成新答案2 → evaluate
              └─ ...
              选最优的 → 下一轮 (max_iters=5)
与 reflexion 的区别：多条反思并行 (self_reflection_diverse)，最多尝试 2 条
每个反思从不同角度诊断错误，增加找到正确路径的概率
### 4. dot_bank — 多样反思 + 记忆库
dot_bank_math.py

两轮 (Two-pass)：
Pass 1 — 同 dot，但对每道题：
    成功 → 存入 memory_bank["positive_trajectories"]（含 embedding）
    失败 → 记入 failed_problems
Pass 2 — 对失败题：
    当前Problem → embedding检索 → 从memory_bank找最相似的已解决题目
    → 作为 few-shot exemplar 拼接：「类似题+解法 → 请解新题」
    → 生成答案 → evaluate
    ├─ 正确 → 结束
    └─ 错误 → 生成多条反思
                → 每条反思再做 reflection-conditioned 检索
                → 找到类似反思对应的成功修复案例
                → 拼接 few-shot reflexion 示例
                → 指导重新生成
核心思想：记忆增强——让模型看到「类似题目是怎么解对的」
### 5. paramagent (ParamAgent) — 多样反思 + 记忆库 + Pitfalls
mainMath_param.py → dot_Math_parametric_with_bank.py

与 dot_bank 结构相同 (Two-pass + Memory Bank)，但每一步都注入 pitfalls：
    mistake_insights = MathPitfallAgent.generate(problem)  # 或从预生成 json 读取
    ↓
    Problem + mistake_insights → 生成答案 → evaluate
    ├─ 正确 → 存 memory_bank (含 insights)
    └─ 错误 → 换一组 pitfalls (高温采样 diversity)
                → self_reflection_diverse_parametric(题目, 答案, feedback, 已有反思, insights)
                → 多条反思并行尝试
关键差异：在 simple 生成、reflexion 生成、self_reflection 三个阶段全部传入 mistake_insights
pitfalls 来源：预生成的 json 文件 (pitfalls + pitfalls_high_temp) 或 LoRA 微调的 MathPitfallAgent
初始尝试时用低温 pitfalls (保守)，后续迭代换高温 pitfalls (探索)
### 总结对比
策略	反思	多样性	记忆库	Pitfalls	Passes
simple	✗	✗	✗	可选	1
reflexion	✓ 单链	✗	✗	✗	1
dot	✓	✓ 多头	✗	✗	1
dot_bank	✓	✓	✓	✗	2
paramagent	✓	✓	✓	✓	2

