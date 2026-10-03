# 路由数据生成代码与轨迹记录

更新：2026-10-03。当前整理仅在本地，尚未同步服务器、运行新实验或加载偏好模型。

## 方法边界

当前讨论的方向是：初始执行后，根据推理时可用的信息，二分类决定“不调用/调用多样性模块”。标准答案只用于离线监督，不能参与在线触发与纠错反馈。收益、成本、后续策略预算和数据隔离方式仍需冻结。

代码中的 `legacy.py` **不是这个正式流程**。它保留此前“初始 oracle 失败后比较普通修复 G 与多样性修复 D”的诊断 pilot，包括静态数据指导、简化 verifier 和独立哈希划分。这些已知问题本次没有伪装成已修复，也不能把对应标签用作正式训练数据。

旧四分类域识别在 `dataset/router/`，与本次收益二分类分开；本次不移动其他历史训练、推理和论文原型文件。

## 模块职责

```text
gain_router/
  common.py       JSONL I/O、旧标识/采样和统计
  datasets.py     Code/Math/QA 源数据连接（legacy pilot）
  actor.py        模型请求、logprobs、原始 API 返回、重试
  verifiers.py    离线评分（已知局限，非在线 QA/Math 反馈）
  traces.py       逐事件落盘、运行配置指纹、单写者锁
  legacy.py       旧 G/D pilot 编排与命令行
  audit.py        只读质量审计
  config.py       严格 YAML 配置与路径校验
  schema.py       在线/离线数据结构与特征白名单
  inputs.py       严格 JSONL 输入与外部 manifest 校验
  policies.py     不读取 gold 的策略执行
  collect.py      调用/不调用轨迹采集（无评分）
  offline.py      独立离线评分与标签版本
  export.py       只导出初始状态特征和二分类标签
scripts/
  generate_gain_router_data.py  保留旧入口，实际调用需显式 opt-in
  audit_gain_router_data.py     不调用模型的审计入口
  collect_router_traces.py      新轨迹采集
  build_router_labels.py        离线标签
  export_router_features.py     白名单特征导出
```

## 原来有没有轨迹？

有。此前服务器的 300 条 pilot 位于：

`/root/autodl-tmp/lrr/ParamAgent/outputs/gain_router_v1/pilot_seed42/trajectories.jsonl`

每行保存 initial/G/D 的提示消息、输出、停止原因、置信度、token 用量、耗时、验证结果和派生标签。但整题成功走完后才写入；如果后续分支抛出异常，前面生成的步骤可能丢失。历史输出保持原样，不能补造当时未保存的事件或原始响应。

## 新的记录文件

| 文件 | 内容 |
| --- | --- |
| `run.json` | run ID、冻结参数、源文件与实际选中输入的 checksum、工具源码 checksum、配置指纹 |
| `events.jsonl` | sample/request 开始、请求返回/失败、每次重试、离线验证、整题完成/失败 |
| `trajectories.jsonl` | 完整题级轨迹；包含 run/attempt ID；旧 pilot 标记 `valid_for_training=false` |
| `errors.jsonl` | 未完成题目的异常、sample/attempt/run ID |
| `manifest.json` | 累计统计、当前选中数量、`running/partial/complete` 状态 |

每次请求返回后、验证前就保存原始响应，包括 `content`、可能存在的 `reasoning_content`、usage 和 logprobs。截断/空输出额外记录 `quality_flags`；这些标记并不修复模型截断，也不认证 verifier 正确。

事件用 `run_id + sample_id + attempt_id + stage` 关联。重试用 `request_attempt` 区分。逐事件 flush/fsync，元数据使用临时文件加原子替换；同目录仅允许一个写者。journal 磁盘写失败会终止，不额外重试模型。

完整题目可续跑跳过；**中途失败的题目会重新执行整题**，之前失败尝试的事件仍保留。当前没有分支级复用，不能宣称已有断点恢复到每个步骤。

配置、源输入、选中样本、代码 checksum 不一致就拒绝同目录续跑。扩大样本量也需要新目录。采样 seed 由 sample ID 派生，不再随域顺序/行号变化。旧无版本输出目录、损坏 JSONL 或缺失行尾换行均拒绝追加，不自动删除或修复。

模型文件 checksum/修订版本尚未与正在服务的端点自动核验，run 中明确记录声明但未验证。新路径已实现严格 manifest 校验与白名单导出，但仍仅为 pilot；本次未恢复模型缺失资产或认证正式训练数据。

## 新路径：配置 / 执行 / 离线监督分开

配置示例是 `configs/router_data_pilot.yaml`；依赖 `requirements-router.txt`。示例路径、revision、偏好服务端口必须核验/替换，8001/8002/8003 并未部署。配置引用模型服务端点；**这套代码不会替你加载 HF adapter 或启动三个服务**，一张 GPU 可在后续调度中轮流服务，不能根据示例端口推断要同时加载全部模型。

`TaskSample` 只接受 sample_id/group_id/domain/problem/context/split/split_seed。输入中出现 gold、tests、pitfalls、decomposition 会报错；这些必须另存 `OfflineSupervision`。prepared 数据文件与 gain_router_v1 manifest 尚需按原始数据严格映射生成，不能直接用旧含 gold 的数组或 router_v1 manifest 替代。

`DecisionState` 只含任务与 initial ModelCall；在线反馈限于是否空输出/截断，没有 QA/Math 正确性反馈。`StrategyTrace` 明确比较 retain_initial 与 preference→actor 的一次修订，最终结果是最后一次 Actor 输出，不按正确性筛候选。这个策略只是显式 pilot 实现，提示模板、预算和收益仍待正式冻结。

新采集不导入 verifier，每个选中初始状态都生成两个动作结果，包括初始成功状态。离线标签文件保留完整评分与排除原因；导出器只白名单放行问题、context、初始回答/推理、数值置信度与非泄漏在线状态。

仅解析示例配置（不读模型/数据、不写文件）：

```bash
python scripts/collect_router_traces.py --config configs/router_data_pilot.yaml --validate-config
```

准备好并核验依赖后，三个阶段分开运行：

```bash
# 默认/--dry-run 校验数据与路径，不调用模型，不写目录。
python scripts/collect_router_traces.py --config configs/router_data_pilot.yaml --dry-run
# 只有显式 --execute 才调用模型；本次未执行。
python scripts/collect_router_traces.py --config configs/router_data_pilot.yaml --execute

python scripts/build_router_labels.py --config configs/router_data_pilot.yaml \
  --run outputs/router_call_v1/pilot_seed42 \
  --supervision dataset/gain_router/prepared/offline_supervision.jsonl

# 使用标签器打印的实际 checksum 子目录，不能写字面量占位符。
python scripts/export_router_features.py --config configs/router_data_pilot.yaml \
  --run outputs/router_call_v1/pilot_seed42 --labels "实际标签目录" \
  --output-dir outputs/router_call_v1/features_pilot_001 --allow-pilot
```

相对配置中的数据、模型、输出路径默认按当前仓库目录解析，可以用采集命令的 `--root` 指定服务器项目根。离线命令的路径参数按当前工作目录解析。

标签定义明确为 `success(call)-success(no_call)-lambda_tokens*额外total_tokens-lambda_latency*额外耗时 > 0`。示例系数为零，只表示诊断选择，不是论文已冻结的目标。QA F1/Code 部分通过率保留为分析数据，不冒充这版二值 success reward。初始执行成本是共享沉没成本，额外成本包含 preference 与 repair 两次调用。

`labels` 配置不参与采集指纹；改成本系数只需重跑标签器。每个标签版本存到独立 checksum 子目录，禁止覆盖已有版本；export 校验原轨迹/run/标签 checksum 和 sample/group/attempt 身份。日志里途中的失败保留为 events，只有完整轨迹才能标签化；不完整采集可只对已完成题目做 pilot 分析，不能声称全量完成。

截断、空输出、缺失监督、verifier 错误都产生 null 标签并排除，不是负类。Math 简化校验器无法证明不等价时保守标 ambiguous；Code 的运行器/调用适配仍需修正和已知正确实现回归验证。Code 离线执行必须显式 `--allow-code-execution`，且应在真正隔离环境中执行。token 成本缺失/重试成本不确定时，非零 token 成本目标会排除该行。

新 journal 采集阶段没有 `verification.completed` 是预期行为，审计器不会把它报告为“漏验证”。所有新产物标记 purpose=pilot/formal_training_certified=false，导出需显式 --allow-pilot。

## 使用

Python 3.10+；本地验证使用 Python 3.12。均从仓库根目录执行。

只读检查旧数据（无需 vLLM）：

```bash
python scripts/audit_gain_router_data.py \
  --input outputs/gain_router_v1/pilot_seed42
```

报告包含分域标签/case 数量、截断率、空输出、verifier 失败类型、重复样本，以及新日志中未结束的请求/未验证步骤。缺少 events 时会明确提示，审计不会补写数据。

只检查源数据选择，不调用模型、不写输出文件：

```bash
python scripts/generate_gain_router_data.py --dry-run --samples-per-domain 3
```

仅在明确要诊断旧方案时才使用：

```bash
python scripts/generate_gain_router_data.py \
  --allow-legacy-pilot --samples-per-domain 3 \
  --output-dir outputs/gain_router_legacy/diagnostic_001
```

这个命令不是正式数据生成建议。本次没有执行它。

## 数据安全与后续工作

- 日志含原始问题、回答和离线答案，保留在忽略的 outputs 目录，不提交 Git；API Authorization header 不写日志。
- 原始轨迹、离线 reward/label、可部署路由特征已分开；feature exporter 只支持显式承认的 pilot，不认证正式训练数据。
- 下一阶段恢复并服务真实偏好模型、准备 sanitized 输入与共享 manifest、冻结策略预算，并修正和回归验证 verifier。
- 需要已知正确答案回归验证、模型完整输出检查、偏好训练/路由数据隔离审计后，才能生成正式训练集。
- Code 当前只有资源限制，不是安全沙箱；不要在有敏感数据/凭证的环境直接执行不可信生成代码。

## 验证

CPU-only 测试不依赖服务器、权重或外部 API：

```bash
python -m unittest discover -s tests -p test_gain_router_traces.py -v
python -m unittest discover -s tests -p test_split_protocol.py -v
python -m unittest discover -s tests -p test_router_boundaries.py -v
```

原有生成器函数测试保留，可使用 pytest 运行；旧入口保留公开 helper 导入兼容，但测试/扩展应优先导入 `gain_router` 的具体模块。
