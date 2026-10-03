# 双分支多轮路由数据

目标是在 4B 首次回答后，选择纯 4B 多轮修订，还是 2B 指导的 4B 多轮修订。
新协议 `router_paired_multiround_v1` 与旧的“一次指导/一次修订”记录分开，不能混用标签。

## 生成规则

每道题独立生成一次首次回答，两条路线共享这个回答。A 每轮调用一次 4B 自我检查并修订；B 每轮先调用一次领域 2B DPO，再调用一次相同 4B 修订。
两条路线的 4B 修订轮数相同，对应轮次的 4B 随机种子相同。每轮读取本分支上一轮回答。
标准答案和隐藏测试不参与生成，不根据它们提前停止。固定最后一轮为最终结果，不选离线评分最高的历史回答。
第一版不加记忆库检索或候选答案搜索，是有独立版本的双分支协议，不能当作原始 DoT 两阶段程序的完整复现。

`rounds=5` 表示共享首次回答后，每个分支再修订5轮，相当于每分支最多6个4B回答。
每题总计11次4B调用、5次2B调用，共16次。先每域1题、2轮检查流程，再每域100题、5轮估计正负比例、截断率与成本。
正式数量根据这批数据的收益分布决定，不必先跑完整个训练池。

## 输入与划分

`prepare` 从现有 Code/Math/QA 训练源重建可评分题池，保留与 SFT/DPO 的重合。
Code 需要匹配测试与函数接口，QA 需要匹配上下文；因此可用题数量可能低于源文件行数。它们的缺失不能通过填答案或测试上下文补成在线信息。
只按规范化文本排除现有独立下游测试，不保证排除近重复。按原始问题分组、seed42 随机约9:1分配 train/val，比例不是精确数量配额。
没有使用旧 `router_v1` 白名单。输出 `audit.json` 保存源文件哈希、可用数量、测试排除量及划分数量。

`tasks.jsonl` 仅包含在线问题和上下文；`supervision.jsonl` 独立保存标准答案/测试。
采集命令只读取 tasks，评分命令才读取 supervision。同题的所有轮次属于同一集合。

## 服务器执行

在服务器仓库根目录，使用已经安装 torch/transformers/peft/bitsandbytes 的 `code2-merge-env` Python。
先确认服务器代码已同步此版本。下面路径是服务器路径，不是本机路径。

```bash
/root/autodl-tmp/lrr/code2-merge-env/bin/python -m gain_router.paired prepare \
  --root /root/autodl-tmp/lrr/ParamAgent \
  --output /root/autodl-tmp/lrr/router-paired-pool-20261003 --seed 42
```

复制 `configs/paired_router.example.json` 为运行配置；填写真实模型 revision 和绝对模型路径。
Actor 使用现有 `http://127.0.0.1:8000/v1` 的 `Qwen3.5-4B`。领域2B使用已恢复的SFT + 对应DPO：

| 领域 | SFT基础目录 | DPO adapter目录 |
|---|---|---|
| Code | models/Qwen3.5-2B-code-merged2 | lora-qwen3.5-2b-dpo/lora-qwen3.5-2b-code2-dpo-20261003 |
| Math | models/Qwen3.5-2B-math-merged2 | lora-qwen3.5-2b-dpo/lora-qwen3.5-2b-math-dpo3 |
| QA | models/Qwen3.5-2B-qa-merged | lora-qwen3.5-2b-dpo/lora-qwen3.5-2b-qa-dpo3 |

本地2B模式用 `--local-preference`，无需另起三个HTTP服务；GPU按领域切换4bit模型。
adapter revision 必须填 `sha256:<adapter_model.safetensors的SHA256>`，加载时核对路径与哈希。
Actor revision 为声明身份，还需在服务器核对运行服务的实际权重；不能仅靠模型名称证明身份。
之前单轮试跑中4B与量化2B可同时运行；增大长度后的多轮显存和截断率尚需真实试跑验证。
2026-10-03 已将服务器Actor的max-model-len从8192调整到16384，服务健康检查HTTP200。
实际10020 token输入的请求成功返回OK，API报告max_model_len=16384，验证退出码0。
验证后GPU已用19873 MiB、空闲4220 MiB。验证记录及8192回滚脚本位于服务器 /root/autodl-tmp/lrr/actor-context-16384-20261003。
16384是输入与输出的总上限；大问题、回答和指导的总长度仍需检查。2B同时加载下的新多轮长请求尚未验证。

```bash
# 不带 --execute：只验证配置、输入与调用预算
/root/autodl-tmp/lrr/code2-merge-env/bin/python -m gain_router.paired collect \
  --tasks /root/autodl-tmp/lrr/router-paired-pool-20261003/tasks.jsonl \
  --settings /root/autodl-tmp/lrr/router-paired-settings.json \
  --output /root/autodl-tmp/lrr/router-paired-smoke --local-preference

# 执行；中断后以相同参数重跑，已保存的首次回答和各轮调用不会重复生成
/root/autodl-tmp/lrr/code2-merge-env/bin/python -m gain_router.paired collect \
  --tasks /root/autodl-tmp/lrr/router-paired-pool-20261003/tasks.jsonl \
  --settings /root/autodl-tmp/lrr/router-paired-settings.json \
  --output /root/autodl-tmp/lrr/router-paired-smoke --local-preference --execute

# 离线评分；Code评分执行生成代码，应在隔离环境中运行
/root/autodl-tmp/lrr/code2-merge-env/bin/python -m gain_router.paired label \
  --run /root/autodl-tmp/lrr/router-paired-smoke \
  --supervision /root/autodl-tmp/lrr/router-paired-pool-20261003/supervision.jsonl \
  --output /root/autodl-tmp/lrr/router-paired-smoke-labels --allow-code-execution
```

## 标签与输出

`label=1` 当 B最终正确、A最终错误；其余可确定的比较标0。
`delta_success` 保留 -1/0/1，分别对应退步/无正确性增益/改善。
`labels.jsonl` 保存首次与每轮分数、分支成本、B减A的额外成本，以及排除原因。
`router_dataset.jsonl` 仅输出身份、split、初始状态特征和标签；没有后续回答、指导或标准答案。
不采集置信度时 feature明确记录不可用，不伪造概率。

任一调用截断/空输出、最终评分异常/不确定，不直接标0，留为未标注记录。
当前沿用 pilot 评分器：Math不匹配可能是符号等价问题，保守标不确定；Code APPS函数接口与测试匹配仍需检查。
QA保留EM/F1、Code保留通过率；第一版标签只使用最终是否正确，不混合不同领域的连续分数。
这些评分限制在正式批量训练前需要用实际样例验证，不能把调用跑通等同于标签质量合格。

每个输出目录绑定配置、输入和代码哈希；修改轮数/预算/模型必须换新目录。日志保留逐调用请求和响应。
已完成题可在批次结束前独立离线评分，之后用新输出目录对新增轨迹重新评分，不覆盖旧标签。

## 2026-10-03 服务器小批量验证

服务器目录：`/root/autodl-tmp/lrr/router-paired-smoke-20261003`。
用 seed=42 从每个领域的 train 池各选 1 题，固定 2 轮，共 21 次真实模型调用。
4B 输出上限 4096、上下文上限 16384；2B DPO 提示输出上限 3072，按领域顺序加载 4bit 模型。

- 3 条完整轨迹，15 次 4B 调用、6 次 2B 调用；19 次正常结束、2 次长度截断（均为 MATH 提示）。没有 OOM，结束后 4B health=200。
- CODE 两分支最终均通过 8/8 测试，标签 0。APPS 单元素输出包装通过显式 `tests.output_format=apps_singleton_wrapper` 解包；真实列表返回值仍保留。
- QA 两分支按当前 exact-match 评分均错误，标签 0。
- MATH 标签为空，排除原因包含提示截断和数学等价性无法确定。题目要求正交单位向量，模型给出标准答案的相反向量，该向量也可能满足题意；当前通用评分器不能识别所有合法解，不能将它判为错误训练样本。

最终结果使用 `labels-v2/report.json`、`labels-v2/labels.jsonl`、`labels-v2/router_dataset.jsonl`；
原始轨迹位于 `run/trajectories.jsonl`。初版 `labels/` 保留用于审计，CODE 初版测试评分受包装格式影响，不应使用。
`pool/supervision-apps.jsonl` 仅为原监督数据追加 APPS 输出格式字段，问题、输入和期望输出值均未改动；新 `prepare` 命令自动声明该格式。

训练池按问题分组约 90:10：CODE 3153/365，MATH 5717/614，QA 17220/1946。
MATH 原始标准答案为空的 2 条记录跳过并计入 `missing_gold_removed`。
此次验证证明生成、保存轨迹和离线打标流程能跑通；3 道题、0 个正样本不能用来判断偏好模型有效性。
扩大 MATH 造数前，应处理提示截断和多合法答案的数学评分。
本地及服务器均通过 45 项测试，代码修复提交为 `f6fb960` 和 `340a548`。
