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
- 记忆输入沿用 SFT 训练模板：Code 函数签名、Math 题目、QA 问题拆解及其既有示例。
  不使用 paired 的 task + current response 通用批评提示。
- 沿用领域 runner 的采样温度；模型请求种子固定42，记忆输出上限3072。
  所有 actor 原始请求/响应、记忆输出、截断标志和评分记录保存在 `calls.jsonl`。
- 小型 MiniLM 在 CPU 上做512-token截断、attention-mask均值池化和L2归一化。
  三组使用同一模型。各组的记忆库独立，从空库开始，仅加入该组生成的轨迹。

## 评分与限制

本诊断显式替换原代码中 Math/QA 的外部 GPT 等价性裁判：HumanEval 使用完整官方测试，
Math 使用末答案抽取、单位清理及 SymPy LaTeX 数值等价，QA 使用标准化 EM 并记录 F1。
这是三组共享的诊断适配，不声称与历史远程裁判完全等价。

Math 无法确认的等价性保留 unresolved，整个运行不会标记完成，不将它静默当作错误。
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
每次调用追加并 fsync，重启时从日志恢复相同请求的输出。
`completed.json` 仅在10题结果齐全且不存在未决评分后生成。
最终结果读取 `second_stage_log.jsonl`；没有首阶段失败时读取 `first_stage_log.jsonl`。
`results.jsonl` 是原 runner 的首阶段记录，不能用来统计检索阶段后的最终成绩。
首阶段失败但没有完成第二阶段的记录会阻止汇总。早期 `e21e0e1` 快照的
`completed.json.successes` 只统计首阶段，分析该快照时需按上述文件重新汇总。
实验输入、权重、轨迹和结果不随源码提交。

九组全部完成并下载后，运行 `python scripts/summarize_parammem_pilot.py /path/to/experiment`，
生成 `summary.json`。它按题配对首次作答、首阶段和最终结果，统计截断与生成token，
检查缺失/重复题目、未重试失败题及 runner 捕获的异常；适用于早期快照。

离线测试：`python -m unittest discover -s tests -p 'test_parammem_pilot*.py' -v`。
