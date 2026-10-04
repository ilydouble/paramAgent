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

2026-10-04 用户选择直接全量采集。服务器已准备目录
`/root/autodl-tmp/lrr/router-paired-full-20261004`，其中 `pool/` 保存重新生成的全量题池，
`settings-train.json` 和 `settings-val.json` 都设置5轮，不限制领域题量。
CODE为3153/365题、MATH为5717/614题、QA为17220/1946题（train/val），合计29015题、464240次模型调用。
启动器 `scripts/run_full_paired_router.py` 默认仅预检查，添加 `--execute` 才生成。
服务器预检查退出码0，确认全量选择、adapter哈希、SFT目录、4B健康及16384上下文上限。

```bash
cd /root/autodl-tmp/lrr/ParamAgent
screen -S router-full
/root/autodl-tmp/lrr/code2-merge-env/bin/python -u scripts/run_full_paired_router.py --execute
```

使用Ctrl+A、D脱离screen，重新连接用 `screen -r router-full`。
启动器顺序采集全量train和val，再离线评分；生成不读取标准答案。
各阶段日志位于输出目录的 `collect-train.log`、`collect-val.log`、`label-train.log`、`label-val.log`，
当前阶段保存在 `launcher-status.json`，完整轨迹在 `run-train/` 和 `run-val/`。
中断后同一命令重跑会复用已保存调用；不要修改运行配置或核心采集代码。
多合法答案等无法确认的MATH样本保留轨迹并排除标签，不能强行标0。
当前数据盘余量约19GiB，不能保证容纳全量轨迹；启动器每30秒检查磁盘，低于2GiB余量停止。
若触发磁盘停止，先增加空间，保留所有已有文件，再续跑；中断恰逢JSONL写入时可能需要显式尾部恢复。

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

## 精简存储与旧断点迁移

双分支采集使用 `paired_compact_v1`。仍请求每个生成token的logprob来计算平均、最低、
低10%分位概率，但不请求额外候选token（`top_logprobs=0`），统计后不落盘完整原始响应。
调用缓存保留完整请求以及回答、推理、统计、结束原因、质量标记、用量、耗时和随机种子；
结果不重复保存messages。2B生成token ID仍保留供EOS检查。
事件日志只记请求哈希、响应摘要和结果摘要；完整回答保存在缓存与轨迹中。
旧单轮/legacy客户端默认行为不变。该修改不改变双分支提示、采样、轮数或评分规则。

2026-10-04 当前批次缩小到每领域2000题，train/val为1800/200，共6000题；
目录为 `/root/autodl-tmp/lrr/router-paired-2000-per-domain-20261004`。
采集在82题完整轨迹、1315次缓存调用处主动暂停；第83题已保存3次调用。
服务器迁移已完成：记录文件由2,552,080,243字节降为41,333,827字节，
原始gzip备份为341,037,105字节，位于该目录的 `backups/run-train-before-compact.tar.gz`。
备份逐文件解压后SHA256一致；另逐条核对全部82条轨迹和1315个调用缓存，
除删除raw_response/messages外，回答、请求、置信度及其他结果字段完全一致。
新配置通过TraceRecorder断点兼容检查；服务器46项相关测试通过，采集仍暂停。

迁移前必须停止采集与启动器。迁移脚本获取两个写锁，要求题池、设置及模型身份完全一致，
只接受脚本内明确声明的旧actor/paired源码哈希，其他核心模块必须保持一致。
先在运行目录外生成gzip tar备份，逐文件解压并核对SHA256；验证后才替换精简文件。
run ID保留，代码指纹显式迁移；`storage-migration.json`记录原/新指纹、备份哈希、文件哈希与数量。
若替换中断，pending标记阻止续跑，需按记录的备份和staging目录恢复，不能直接删标记。

```bash
RUN=/root/autodl-tmp/lrr/router-paired-2000-per-domain-20261004
/root/autodl-tmp/lrr/code2-merge-env/bin/python scripts/compact_paired_run.py \
  --run "$RUN/run-train" --tasks "$RUN/pool/tasks.jsonl" \
  --settings "$RUN/settings-train.json" --local-preference \
  --backup "$RUN/backups/run-train-before-compact.tar.gz"
```

该命令不调用模型、不读取标准答案、不恢复采集。备份路径必须不存在，且位于run-train目录外。
迁移后用原启动命令续跑；已有完整题跳过，未完成题按精确请求复用缓存。

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
