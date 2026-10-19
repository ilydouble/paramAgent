# 项目文档索引

这里维护代码方法、数据协议、服务器模型和运行说明。模型文件留在服务器；来源记录、关键脚本和哈希随代码进入 Git。

## 方法与工作区

| 文档 | 用途 |
| --- | --- |
| [方法命名与入口](method_architecture.md) | 区分 ParamMem、ParamMem-DPO、ParamAgent，说明当前实现范围 |
| [本地工作区](local_workspace.md) | 本地代码、论文、实验和归档的分工 |
| [文档命名与维护规范](documentation-guide.md) | 新文档命名、层级、状态和证据规则 |

## 模型

| 文档 | 用途 |
| --- | --- |
| [服务器模型索引](models/README.md) | 现有模型、领域配对、版本关系和缺失依赖 |
| [2B 模型重建说明](models/rebuild-qwen35-2b.md) | 下载原版底座、恢复 SFT、重新训练 Code2 DPO 的过程与边界 |
| [模型登记表](models/model-registry.json) | 可读取的路径、大小、adapter 当前哈希与历史合并来源 |
| [2026-10-19 模型核查记录](models/records/2026-10-19/README.md) | 本次证据快照、验证范围与历史训练源码 |
| [模型重建记录模板](templates/model-rebuild-record.md) | 后续重建或重训时应填写的记录 |

## 数据、采集与历史研究

| 文档 | 用途 |
| --- | --- |
| [数据划分协议](data_split_protocol.md) | 已冻结的数据身份和划分协议；适用范围以正文为准 |
| [路由代码说明](gain_router_code_guide.md) | 采集、标签、特征与审计实现及历史局限 |
| [配对路由采集](paired_router_data.md) | 多轮 actor-only/guided 配对流程和运行方式 |
| [路由设计计划](multitask_binary_gain_router_plan.md) | 历史方案与分阶段设计，不代表每项均已实现 |
| [源码同步记录](server_sync_20261019.json) | 服务器代码合并来源及六个源码的核对哈希 |

已有文件名保留兼容。新增文件遵循命名规范；旧文档中的 SCORE、旧服务器路径和历史验证状态需结合各自日期阅读，不自动视为当前运行状态。
