# 多任务二分类收益路由：数据、训练与实验计划

> **2026-10-03 状态更正**：下文 G/D 比较与 oracle 初始失败筛选保留为历史设计，不再作为当前实现依据。当前方向是初始执行后，二分类决定“不调用/调用多样性模块”；所有初始状态均需考虑，gold 只能用于离线监督。现已实现独立的调用/不调用 pilot 采集、离线净收益标签及初始状态白名单导出，并通过模拟测试；尚未部署偏好模型服务或生成真实新数据。具体成本、执行预算、偏好模板与正式 verifier 仍需冻结，不能宣称正式训练数据已可用。旧 pilot 保持隔离。代码与轨迹规范见 `gain_router_code_guide.md`。

> 状态：方法设计基线（source of truth）
>
> 最后更新：2026-10-01
>
> 适用范围：Code、Math、Multi-hop QA 共享路由器
> 约定：后续实现或论文叙述若要偏离本文档，必须先显式更新本文档并记录原因。

## 1. 目标与非目标

目标不是识别输入属于 Code、Math、QA 或 abstain，而是预测：在一次初始执行失败后，是否值得调用后续的多样性/参数化提示词策略。

- **任务形式**：多任务共享的二分类。
- **输出**：`p_gain`，即多样性策略相对普通修复取得严格正收益的概率。
- **决策**：当 `p_gain >= tau` 时调用多样性策略，否则执行普通、低成本修复。
- **共享方式**：Code、Math、QA 使用同一个共享编码器和二分类目标；`task_id` 是条件信息，不训练三个完全独立的路由器。
- **不是四分类**：旧的 `abstain/code/math/qa` 路由及其标签只能作为历史基线，不能作为最终方法。
- **不是让大模型直接判断标签**：大模型负责生成轨迹；标签来自冻结的、任务特定的 verifier。

## 2. 系统中的模型角色

### 2.1 Actor

当前数据生成 Actor 为临时服务器上通过 vLLM 部署的 `Qwen3.5-4B`。它负责：

1. 生成初始回答 `y0`；
2. 在相同失败状态下生成普通修复结果；
3. 在相同失败状态下生成多样性/参数化提示词条件下的修复结果。

Actor 在一版数据生成过程中必须冻结模型权重、采样参数、提示模板和 vLLM 版本。

### 2.2 多样性提示词模块

多样性模块负责生成或提供后续提示词/反思指导。最终路由数据必须使用将来部署时真正使用的模块版本；不能用一种临时提示策略生成标签，却用另一种策略做最终实验。

如果正式偏好模型尚未训练，可先用版本化的静态策略库生成 **pilot 数据**，但 pilot 标签不能冒充最终路由训练数据。

### 2.3 路由器

推荐结构为文本编码器与数值置信特征的融合模型：

- 文本编码器：DeBERTa-v3-base，编码任务、问题、初始失败回答及 verifier 反馈；
- 数值分支：小型 MLP，编码初始生成置信度、长度和 verifier 特征；
- 融合头：输出一个二分类 logit，并经 sigmoid 得到 `p_gain`。

采用 DeBERTa 而不是仅用普通 MLP，是因为路由需要理解失败轨迹与问题之间的语义关系；MLP 只适合处理置信度等数值特征。

## 3. 路由时可用的输入

路由器只能使用真实推理时在决策前已经可获得的信息：

```text
task_id
problem x
initial answer / reasoning trace y0
initial verifier feedback f0
initial generation confidence features c0
```

禁止把普通修复或多样性修复的真实结果、正确答案、未来 verifier 结果输入路由器。这些字段只用于离线构造监督标签。

建议记录的置信特征包括：

- 生成 token 的平均 log-probability；
- 最小或低分位 token log-probability；
- token entropy/top-1 margin（若 vLLM 接口提供 top-logprobs）；
- 输出长度、截断标记、停止原因；
- verifier 的非泄漏状态特征，例如测试通过比例或 QA 匹配分数；
- 缺失掩码，避免不同任务没有同一种数值特征时被错误填零。

置信度特征和提取方法必须在 train/validation/test 完全一致。

## 4. 原始轨迹与二分类标签

### 4.1 比较对象

初始回答 `y0` 只用于确定是否进入路由阶段以及构成路由输入。收益比较发生在同一个初始失败状态上：

- `G`：普通/低成本自反思修复策略；
- `D`：多样性或参数化提示词修复策略。

不要把 `D` 直接与初始回答 `y0` 比较。正确的反事实问题是：在已经失败的同一状态下，选择 `D` 是否比选择 `G` 更有用。

### 4.2 必须保留的全部情形

| 初始结果 | G 结果 | D 结果 | `case_type` | 二分类标签 | 含义 |
|---|---:|---:|---|---:|---|
| 成功 | 未运行 | 未运行 | `initial_success` | `null` | 无路由事件，不进入 gate 训练 |
| 失败 | 0 | 1 | `strict_positive_gain` | 1 | 只有多样性策略修复成功 |
| 失败 | 1 | 1 | `redundant_intervention` | 0 | 两者都成功，多样性调用冗余且更贵 |
| 失败 | 1 | 0 | `negative_transfer` | 0 | 普通修复成功，多样性策略反而失败 |
| 失败 | 0 | 0 | `shared_failure` | 0 | 两种策略都无效 |

核心监督定义为：

\[
y^*=\mathbf{1}\left[r_D=1 \land r_G=0\right]
=\mathbf{1}\left[r_D-r_G=1\right].
\]

因此二分类只有一个正类，但负类内部必须保留三种不同失败机制，不能只保存 `label=0`。

### 4.3 异常、格式错误和部分得分

- API 断开、vLLM 服务错误、请求超时等基础设施错误：标记为 `infra_error`，重试；超过重试上限则整行排除，不能当作负类。
- 模型正常返回但答案格式错误、代码编译错误、代码运行超时或无最终答案：属于真实策略失败，`reward=0`，并记录具体 `failure_type`。
- QA F1、代码测试通过率等连续指标必须原样保存；成功/失败由预先冻结的任务阈值派生。
- 阈值边界或 verifier 不确定的样本标记为 `ambiguous`。它不是第三个训练类别，可在训练时剔除或降权，但必须报告数量和处理规则。

### 4.4 多候选与随机性

每个候选必须单独落盘，记录种子、提示词、输出、reward、token 和延迟。之后才按冻结的策略预算聚合为 `r_G` 与 `r_D`。

生成前必须冻结下列两种定义中的一种：

1. **单次匹配比较**：G 与 D 各一次，和当前论文中的单次边际收益公式一致；
2. **策略级比较**：比较完整低预算策略与完整多样性策略，例如 G 的 `K=1` 与 D 的顺序 first-hit `K=3`，更贴近实际部署，但论文公式和成本项也必须同步修改。

在该选择冻结前只能做小规模 pilot，不能启动全量生成。

如果每个分支有多次重复采样，还要保留：

- `success_count`、`success_rate`；
- first-hit 的候选位置；
- `delta_success_rate = p_D - p_G`；
- 平均 token/延迟成本。

硬二分类标签仍由预先声明的聚合规则产生；不得看到数据分布后临时修改规则以平衡类别。

## 5. 任务 verifier

三类任务共享标签语义，但 verifier 不同：

- **Code**：隔离执行，固定可见测试用于训练轨迹；最终隐藏测试只用于最终评估。编译错误、运行异常和超时均是模型失败而非基础设施错误。
- **Math**：使用冻结的答案提取与等价性判断；优先可复现的规范化/符号等价，不以另一个大模型的主观判断直接充当唯一标签。
- **Multi-hop QA**：保存 exact match 和 token-level F1；在数据生成前冻结二值成功阈值，同时保留连续 F1 供分析。

verifier 版本、阈值和代码 commit 必须写入每个数据集 manifest。

## 6. 数据划分与防泄漏

### 6.1 先划分原始问题，再生成任何轨迹

一个原始问题是一个 `group_id`。同一问题产生的所有初始回答、多个采样种子、普通修复、多样性候选、偏好对和路由标签必须继承同一 split。

禁止根据生成后的轨迹文本重新随机划分。

### 6.2 与偏好学习的关系

多样性偏好学习和收益路由必须使用同一个外层 group manifest，以保证任何原始问题不会跨 train/validation/test。

还要避免用专家在其自身训练样本上的效果生成过于乐观的路由标签。优先方案为：

- 在 train split 内做 out-of-fold/cross-fitting；
- 为某个 fold 生成 D 分支时，使用未见过该 fold 的偏好模型；
- 或明确划分互斥的 expert-training pool 与 router-label pool。

最终采用哪一种要根据算力确定，但必须在论文和 manifest 中公开。

### 6.3 多划分种子

- 主划分：seed `42`；
- 鲁棒性划分：`123`、`2027`、`3407`、`8888`；
- 数据划分 seed 与模型训练 seed 分开记录；
- 所有方法、基线与消融必须使用完全相同的 split manifest。

现有 `split_manifests/router_v1` 来源于旧四分类数据。它只能作为历史协议或在 group ID 能严格映射时复用。新的收益数据不含人为的 `abstain` 类，建议建立独立版本 `gain_router_v1`，并从 Code/Math/QA 的原始训练问题生成。

最终 benchmark test 不参与提示词生成、DPO、路由训练、阈值选择或策略选择。若要报告 oracle 路由 AUROC，只能在全部策略冻结后一次性生成评估标签，且不能反馈到训练流程。

## 7. 建议的数据文件结构

### 7.1 一条决策事件

```json
{
  "schema_version": "gain_router_v1",
  "sample_id": "...",
  "group_id": "...",
  "domain": "code|math|qa",
  "source_dataset": "...",
  "split_seed": 42,
  "split": "train|validation|test",
  "problem": "...",
  "initial": {
    "answer": "...",
    "reward": 0,
    "verifier_feedback": "...",
    "confidence": {},
    "generation": {}
  },
  "generic_policy": {
    "policy_id": "...",
    "attempts": [],
    "reward": 0,
    "cost": {}
  },
  "diverse_policy": {
    "policy_id": "...",
    "attempts": [],
    "reward": 1,
    "cost": {}
  },
  "case_type": "strict_positive_gain",
  "label": 1,
  "valid_for_training": true,
  "provenance": {}
}
```

### 7.2 Provenance 最低要求

每条记录或其 manifest 至少包含：

- Git commit；
- Actor 模型路径/模型 ID 和权重 hash；
- vLLM 与 transformers 版本；
- 三套提示模板的 hash；
- verifier 版本和阈值；
- temperature、top-p、seed、max tokens；
- G/D 预算和停止规则；
- 数据划分 manifest 及 checksum；
- 生成时间、服务器和失败重试次数。

原始轨迹应采用 append-only JSONL，并支持按 `sample_id + policy_id + seed` 断点续跑。标签派生脚本与原始生成分离，避免修改标签定义时重新调用大模型。

## 8. 训练目标与评估

### 8.1 主损失

共享二分类模型使用 BCE：

\[
\mathcal{L}_{gate}=-\mathbb{E}[y^*\log p_{gain}+(1-y^*)\log(1-p_{gain})].
\]

类别不平衡优先使用显式 `pos_weight`、分层采样或 focal loss 消融，但验证集和测试集必须保持自然分布。多任务 batch 应避免样本量大的 QA 完全主导训练。

### 8.2 校准与阈值

在 validation 上做温度缩放或 isotonic calibration，并只在 validation 上选择 `tau`。阈值不只追求 F1，而应最大化冻结的系统效用，例如：

\[
U(\tau)=\text{task success gain}-\lambda_{tok}\text{token cost}-\lambda_{lat}\text{latency}.
\]

### 8.3 必报指标

- AUROC、AUPRC、Macro-F1、positive-class precision/recall/F1；
- Brier score、ECE 和可靠性曲线；
- 三种负类机制分别的误调用率；
- Negative Transfer Rate；
- expert/diversity activation rate；
- 端到端任务成功率、tokens、延迟和 utility；
- Code、Math、QA 分域结果与总体结果；
- 五个固定数据划分的均值与标准差。

## 9. 执行顺序

### Phase A：冻结数据协议

- [ ] 确定原始 Code/Math/QA 训练池及 group ID；
- [ ] 建立 `gain_router_v1` 的五套固定 manifest；
- [ ] 冻结 verifier、成功阈值和异常处理；
- [ ] 冻结 G/D 两条策略以及单次比较或策略级比较；
- [ ] 确定 preference expert 与 router label 的 cross-fitting 方案。

### Phase B：实现可恢复的数据生成器

- [ ] 调用临时服务器的 Qwen3.5-4B/vLLM；
- [ ] 获取初始输出及 logprobs/confidence；
- [ ] 仅对初始失败样本运行配对的 G/D 分支；
- [ ] 保存候选级原始轨迹、成本和 verifier 结果；
- [ ] 支持断点续跑、幂等去重、基础设施错误重试和 manifest 校验。

### Phase C：小规模 pilot

- [ ] 每个域先抽取约 100 个原始问题；
- [ ] 检查初始失败率、四种 `case_type` 分布和正类比例；
- [ ] 人工审计每域至少 20 条，重点核对 verifier 和异常分类；
- [ ] 估算全量生成所需 GPU 时间、token 和磁盘；
- [ ] pilot 通过后锁定配置，不用测试集调参。

### Phase D：全量生成与质检

- [ ] 生成 train 和 validation 原始轨迹；
- [ ] 派生标签并输出数据质量报告；
- [ ] 验证所有派生行的 `group_id` 与 split；
- [ ] 报告重复、缺失、infra error、ambiguous 和各类失败数量；
- [ ] 冻结数据版本和 checksum。

### Phase E：训练多任务二分类路由器

- [ ] DeBERTa + confidence MLP 主模型；
- [ ] 简单 MLP、prompt-only DeBERTa、confidence-only 等基线；
- [ ] 校准 `p_gain` 并在 validation 选择 `tau`；
- [ ] 保存模型 seed、split seed、配置和 checkpoint hash。

### Phase F：端到端实验与论文证据

- [ ] 冻结策略后执行一次最终测试；
- [ ] 与 always-generic、always-diverse、random gate、旧四分类 router 对比；
- [ ] 做无 confidence、无 trace、无 task conditioning、无 calibration 消融；
- [ ] 报告分域、总体、多划分 seed 与成本—收益曲线；
- [ ] 再决定是否以 JEV 替换/增强共享路由器，并保持相同数据和评估协议。

## 10. 当前服务器状态与阻塞项

- 临时服务器：`ssh -p 18761 root@connect.westx.seetacloud.com`；
- 项目目标目录：`/root/autodl-tmp/lrr/ParamAgent`；
- Actor 权重：`/root/autodl-tmp/lrr/models/Qwen3.5-4B`；
- vLLM API：`http://127.0.0.1:8000/v1`，模型名 `Qwen3.5-4B`；
- 当前聊天无法无交互复用 SSH 身份，交互式终端停在密码提示；密码只能由用户在终端中输入，不应发送到聊天。

在以下三项冻结前不启动全量数据生成：

1. G 与 D 的最终提示模板/模型版本；
2. 单次匹配比较还是完整策略级比较；
3. 偏好模型与路由标签之间采用 cross-fitting 还是互斥数据池。
