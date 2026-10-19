# 2026 年 10 月 19 日服务器模型核查记录

本目录保存从临时服务器读取的模型元数据、重建脚本和来源记录，以及从本地 Git 找回的历史训练源码。不包含模型权重或训练数据。核查时间为 2026-10-19 22:43:05（Asia/Shanghai），对应 inventory 中的 UTC 时间。

## 采集范围

服务器代码实际目录为 `/root/autodl-tmp/lrr/paramagent/code/ParamAgent`，当时 Git HEAD 为 `938ef8e02c9e6a6e801366b2605944702ff9aa00`。本地整理后的新入口尚未部署到该版本。

| 路径 | 内容 |
| --- | --- |
| [inventory.json](inventory.json) | 17 个 2B 相关条目、当前 adapter 哈希、4B 权重文件存在/大小、34 个采集文件的来源路径和哈希 |
| `models/` | 原版下载来源、三份 SFT 合并来源和小型模型配置 |
| `lora-qwen3.5-2b/` | 五个保留的 SFT adapter 配置 |
| `lora-qwen3.5-2b-dpo/` | 八个保留的 DPO adapter 配置 |
| `actor/` | 4B 模型配置 |
| `operations/code2-recovery-20261003/` | 原版底座下载和 Code2 SFT 合并脚本 |
| `operations/qa-math-recovery-20261003/` | QA / Math SFT 合并脚本 |
| `operations/code2-dpo-run-20261003/` | Code2 DPO 参数、输入身份、验证、退出码及找回的历史 Python 源码 |

原始传输包保存在本地工作区的 `archive/model_evidence/2026-10-19/paramagent-model-evidence-20261019.tar.gz`，SHA-256 为 `eb3d48f6cdf4637149c745a57ef6a82222979883d66eb52aab12d64ab2278058`。传输包及其中 34 个采集文件均已核对哈希。

## 历史源码补充

历史 `training_source.json` 对 Python 源码的哈希为 `13bad8facad64037819590e19bef38546b95db1e96f2b1e65e57f6d52569bc3f`。Git blob `526b58cbc27bd571bf4bd08715961828364fd411` 的内容精确匹配；已另存为 `operations/code2-dpo-run-20261003/code-dpo-training-source.py`，配套 JSON 说明其来源和当前文件的不同身份。这两份是本地补充材料，不属于最初 34 个采集文件。

## 验证范围

本次重新计算了 13 个 adapter 的权重 SHA-256，并核对所选 SFT adapter 与合并来源、Code2 DPO adapter 与历史验证的身份一致。大权重仅检查存在和大小，没有重新计算原版/SFT/4B 完整权重哈希；没有运行合并、训练、模型加载或生成。

SFT 的 `status=verified`、Code2 DPO 的有限张量/logits 变化/短生成等内容均来自 2026-10-03 的原始记录，不应改写为 2026-10-19 重新执行成功。未做下游质量评测，也未检查 4B 实时服务。

这里的 Python / shell 是历史证据副本，不是新增通用生产入口，保留原固定路径和输出约束。使用方式及局限见 [重建说明](../../rebuild-qwen35-2b.md)。后续核查写入新的日期目录，不能覆写本快照。
