# 服务器模型索引

最近核查：2026-10-19。三域 SFT 完整模型已经重新合并；Code2 DPO 是在重建底座上重新训练的新 adapter。Math/QA DPO3 是保留下来的 adapter。重建操作发生于 2026-10-03，本次只读取文件和记录，没有重新训练、加载模型或调用生成。

## 路径约定

- 服务器代码实际目录：`/root/autodl-tmp/lrr/paramagent/code/ParamAgent`。
- 兼容入口：`/root/autodl-tmp/lrr/ParamAgent`。下表的相对路径从此仓库根目录解析；实际目标见登记表 `resolved`。
- 4B 主模型位于 `/root/autodl-tmp/lrr/paramagent/models/Qwen3.5-4B`，与领域 2B 记忆模型分开。此次确认权重文件存在，未核验正在运行的服务。

## 已登记的领域配对

| 领域 | SFT 来源 adapter | 重建的 SFT 完整底座 | DPO adapter |
| --- | --- | --- | --- |
| Code | `lora-qwen3.5-2b/lora-qwen3.5-2b-code2` | `models/Qwen3.5-2B-code-merged2` | `lora-qwen3.5-2b-dpo/lora-qwen3.5-2b-code2-dpo-20261003` |
| Math | `lora-qwen3.5-2b/lora-qwen3.5-2b-math2` | `models/Qwen3.5-2B-math-merged2` | `lora-qwen3.5-2b-dpo/lora-qwen3.5-2b-math-dpo3` |
| QA | `lora-qwen3.5-2b/lora-qwen3.5-2b-qa` | `models/Qwen3.5-2B-qa-merged` | `lora-qwen3.5-2b-dpo/lora-qwen3.5-2b-qa-dpo3` |

这是现有配对配置和依赖记录对应的组合，不是按下游性能重新选择出的最优版本。原版 `models/Qwen3.5-2B` 是下载底座，不包含领域 SFT。

- **ParamMem**：使用表中的 SFT 完整模型；或者原版 2B 加相应 SFT adapter，二选一。
- **ParamMem-DPO**：使用表中的 SFT 完整底座加对应 DPO adapter。不要再叠加一次 SFT adapter。
- **ParamAgent**：在 DPO 记忆模型之外还需收益路由器；完整在线门控入口尚未接通，模型可加载不等于完整系统已评测。

原 SFT adapter 当年训练所用的精确公开底座 revision 未被确认；来源记录明确标为 `historical_base_revision_verified=false`。因此这里描述的是可运行的重新合并模型，不能宣称与丢失的历史完整模型逐字节相同。

## 保留但不应混用的版本

- Code SFT 的 `code` 与 `code2` 是不同权重，均保留；本次重建选择 `code2`。
- `lora-qwen3.5-2b-code-dpo30` 声明依赖 `models/Qwen3.5-2B-code-merged3`，该底座不在当前模型盘点中。不能将它直接挂到 `code-merged2` 上称为原模型恢复。
- Math DPO/DPO2/DPO3、QA DPO/DPO2/DPO3 均保留，当前配置使用各自 DPO3。目录编号不是质量排序，不能据此判定性能。
- 历史 `infer_reflections_qwen35_2b*.sh` 仍含不同输入、采样参数及旧路径。尤其 Math DPO 历史脚本把 SFT adapter 目录放在 `BASE_MODEL`，不能照此作为完整底座。加载示例应使用上表配对。

## 查阅与核验

- [重建步骤、加载方式和验证边界](rebuild-qwen35-2b.md)
- [机器可读登记表](model-registry.json)：17 个 2B 相关条目、4B 文件盘点、当前 adapter 哈希及历史合并来源。
- [本次原始证据](records/2026-10-19/README.md)：34 个服务器元数据/脚本的副本和哈希，以及找回的精确历史训练源码。

登记表中 adapter 的 SHA-256 是此次实际计算的；完整 SFT 权重 SHA-256 来自 2026-10-03 的合并来源记录，本次只检查这些大权重的存在和大小，没有重新计算大权重哈希。历史短生成检查只说明当时能运行，不证明任务质量。
