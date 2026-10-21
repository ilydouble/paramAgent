# 三组小样本实现诊断

`scripts/run_parammem_pilot.py` 在相同的冻结测试题上运行 Qwen3.5-4B、
ParamMem（SFT 2B）和 ParamMem-DPO（同一 SFT 底座加 DPO adapter）。
每域10题。它直接调用现有三域 `*_parametric_with_bank.run_dot`，保留
领域生成提示、反思候选、成功早停及第二阶段跨题检索流程，不调用 paired 造数器。

## 对照范围

- `qwen` 仍运行相同的反思与检索流程，记忆模块返回空字符串。首次作答可从逐题轨迹单列，
  不把该组的多轮最终结果叫作零样本 Base。
- `parammem` 加载经核查的领域 SFT 合并模型。
- `parammem-dpo` 加载相同 SFT 模型和对应 DPO adapter，核对底座名称与登记的 adapter 哈希。
- 记忆输入采用固定作者 8b36cdd 的批量推理模板：Code 函数签名、Math 题目及各自示例，QA 问题及原始两个示例。
  作者训练模板与推理模板分开保留，来源哈希由 `shared/parammem_prompts.py` 校验。
  不使用 paired 的 task + current response 通用批评提示。
- 沿用领域 runner 的采样温度；同一请求首个样本种子42，重复采样依次43、44等，记忆输出上限3072。
  记忆采样仅在隔离的 PyTorch RNG 上设置种子，不重置 runner 的 Python/NumPy 搜索状态。
  所有 actor 原始请求/响应、记忆输出、截断标志和评分记录保存在 `calls.jsonl`。
- 小型 MiniLM 在 CPU 上做512-token截断、attention-mask均值池化和L2归一化。
  三组使用同一模型。各组的记忆库独立，从空库开始，仅加入该组生成的轨迹。

## 评分与限制

这次源码对齐不改写 ec93a65 或更早的实验归档，也不把旧 DPO 权重认证为 SCORE 协议。
训练数据、流程与偏好证据尚未完整对齐，见 `docs/models/paper-protocol-alignment.md`。

本诊断显式替换原代码中 Math/QA 的外部 GPT 等价性裁判：HumanEval 使用完整官方测试，
Math 使用末答案抽取、单位清理及 SymPy LaTeX 数值等价，QA 使用标准化 EM 并记录 F1。
这是三组共享的诊断适配，不声称与历史远程裁判完全等价。

Math 无法确认的等价性保留 unresolved，整个运行不会标记完成，不将它静默当作错误。
仅当精确匹配的Actor响应记录明确为`finish_reason=length`、无可抽取终答、
且标量gold可解析时，无效表达式记为`budget_exhausted_without_final`失败；
这是既定token预算内未提交答案，不是等价性未决。正常终止、显式但不可解析的终答、
缺失或矛盾的截断证据、gold解析失败仍保持unresolved；不增加预算或抽样重试。
响应证据匹配仅允许忽略首尾空白，以兼容runner的`strip()`；内部内容不归一化，
同文本存在不同finish_reason时仍视为证据矛盾，不据此推定截断。
Code 的生成代码在有 CPU/内存/墙钟限制的子进程中执行；这不是安全沙箱。
先以 HumanEval canonical solution 检查选择的测试契约。

流程使用正确性反馈决定停止及成功轨迹入库，且第二阶段检索同批先前已解题的经验。
它是含 oracle 反馈和跨测试题经验的实现诊断，不是无反馈部署性能或严格独立测试认证。
不据这30题调路由阈值、选 checkpoint 或追求显著性。

## 运行

在隔离源码快照中运行，使用绝对输出路径。`--model-root` 指向实际服务器仓库以读取权重，
不修改该仓库。模型服务仅接受本地端点。

```bash
python scripts/run_parammem_pilot.py \
  --domain code --method parammem-dpo \
  --dataset /path/to/frozen/code.jsonl \
  --output /path/to/results/code/parammem-dpo \
  --model-root /root/autodl-tmp/lrr/paramagent/code/ParamAgent \
  --embedding-path /root/autodl-tmp/lrr/paramagent/models/all-MiniLM-L6-v2
```

身份文件固定输入、runner、harness和权重哈希；身份不同拒绝复用输出。
Math正式回答默认上限为8192生成token，入口显式设置`--math-answer-tokens 8192`、
`--actor-context-tokens 16384`及`--context-reserve-tokens 256`；三组共用这些参数。
Actor通过同一vLLM服务的`/tokenize`按实际chat template精确计数；服务报告的上下文必须
与入口配置一致，否则停止。实际输出上限为请求预算与上下文剩余量的较小值；不截断输入，
上下文耗尽时明确报错。每次调用记录`actor_budget`（输入token、期望/实际输出上限与预留），
身份文件记录预算参数。非pilot入口可用`MATH_MAX_ANSWER_TOKENS`配置回答上限，
`VLLM_MAX_MODEL_LEN`应与各自服务一致；未提供精确计数接口的旧后端仍采用原估算方式。
预算调整后必须使用全新输出目录，不以2048预算的响应缓存初始化扩大预算的比较；
原实验仅保留为旧预算诊断，不混入新结果。8192仍是有限上限，并不保证模型不会截断。
复测新训练的DPO时，同时提供`--dpo-adapter-path /absolute/new/adapter`和
`--expected-dpo-sha256 <已验证的adapter_model.safetensors哈希>`，替代默认历史登记项。
入口核对文件哈希和精确SFT底座路径，身份文件记录实际adapter路径；其余两组不接受该覆盖。
每次调用追加并 fsync，完整重放可从日志恢复相同请求及采样序号的输出。
恢复时使用新输出目录，仅以旧 `calls.jsonl` 初始化缓存，不复用阶段结果；禁止原地续跑，
避免原 runner 跳过阶段后无法还原采样次数和记忆库。
`66c8e85` 及更早快照重复使用同题同温度记忆样本，只适合作为固定记忆消融；
其首次作答可保留，多轮结果需用按序号采样版本重新验证。
`completed.json` 仅在10题结果齐全且不存在未决评分后生成。
最终结果读取 `second_stage_log.jsonl`；没有首阶段失败时读取 `first_stage_log.jsonl`。
`results.jsonl` 是原 runner 的首阶段记录，不能用来统计检索阶段后的最终成绩。
默认每域严格校验10题。补充小样本诊断时，可在生成入口和汇总入口均显式提供
`--expected-tasks 1`（每域1道、三域共3道），数量会写入运行身份；仍要求题目ID唯一、
同域三组同题及完整两阶段结果。不重复填充题目凑数，不复用前次记忆库或响应缓存。
首阶段失败但没有完成第二阶段的记录会阻止汇总。早期 `e21e0e1` 快照的
`completed.json.successes` 只统计首阶段，分析该快照时需按上述文件重新汇总。
实验输入、权重、轨迹和结果不随源码提交。

九组全部完成并下载后，运行 `python scripts/summarize_parammem_pilot.py /path/to/experiment`，
生成 `summary.json`。它按题配对首次作答、首阶段和最终结果，统计截断与生成token，
检查缺失/重复题目、未重试失败题及 runner 捕获的异常；适用于早期快照。

离线测试：`python -m unittest discover -s tests -p 'test_parammem_pilot*.py' -v`。
