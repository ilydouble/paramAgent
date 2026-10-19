# Qwen3.5 2B 记忆模型重建说明

重建日期：2026-10-03。证据核查日期：2026-10-19。

本次恢复分为三件事：重新下载公开 2B 底座；用保留下来的领域 SFT adapter 重新合并三份完整 SFT 模型；在新 Code2 SFT 底座上重新训练 Code2 DPO。Math/QA DPO3 沿用保留的 adapter，未在这次恢复中重新训练。4B 解题主模型独立存放，不是上述合并底座。

## 来源与版本关系

```text
Qwen/Qwen3.5-2B（固定下载 revision）
  + Code2 SFT adapter -> code-merged2 -> 新 Code2 DPO adapter（2026-10-03）
  + Math2 SFT adapter -> math-merged2 + 保留的 Math-DPO3 adapter
  + QA SFT adapter   -> qa-merged    + 保留的 QA-DPO3 adapter
```

下载记录中的 revision 为 `15852e8c16360a2fea060d615a32b45270f8a8fc`，endpoint 为 `https://hf-mirror.com`。合并脚本对下载权重核对的 SHA-256 为 `aa33250c4fc64891ddfaba3a314fd9542ea371843c387178b425fbcc5ed680b1`。

这证明使用了所记录的下载底座，不证明它就是旧 SFT adapter 当年训练时的精确 revision。三份合并来源记录均保留 `historical_base_revision_verified=false`。应称为“用保留 adapter 重新合并”，不能称为“逐字节恢复丢失的原模型”。

## SFT 如何重新合并

历史操作目录均可通过 `/root/autodl-tmp/lrr/` 下的兼容链接访问，实际位于 `paramagent/operations/`：

| 操作 | 历史执行脚本 | 产物 |
| --- | --- | --- |
| 下载公开底座 | `code2-recovery-20261003/download.py` | `models/Qwen3.5-2B` 与下载来源记录 |
| 合并 Code2 SFT | `code2-recovery-20261003/merge.py` | `models/Qwen3.5-2B-code-merged2` |
| 合并 QA SFT | `qa-math-recovery-20261003/merge_qa.py` | `models/Qwen3.5-2B-qa-merged` |
| 合并 Math2 SFT | `qa-math-recovery-20261003/merge_math.py` | `models/Qwen3.5-2B-math-merged2` |

合并使用独立的 `code2-merge-env`：torch `2.8.0+cu128`、transformers `5.18.0`、PEFT `0.21.2`。这些是来源记录中的当时版本，不是本次重新检测的环境版本。

每个领域的处理步骤为：

1. 在 CPU 上以 bfloat16 加载原版底座，检查原始下载权重 SHA-256。
2. 读取对应 SFT adapter，检查 LoRA 张量与底座层的形状匹配，共识别 114 个 LoRA 层。
3. 通过 PEFT `merge_and_unload(safe_merge=True)` 合并。对选定层比较实际合并结果与 PEFT 的 delta 计算，使用严格 `torch.equal`。
4. 保存到独立 `.partial-20261003` 目录，复制分词器、处理器配置及聊天模板。
5. 从磁盘重新加载保存后的完整模型，检查有限 logits，并执行 12-token 的非空短生成。
6. 写入 `merge_provenance.json`，记录底座、adapter 哈希、环境、检查结果与输出权重哈希，全部完成后才把临时目录改为正式输出名。

原脚本会拒绝覆盖既有正式输出和临时输出。首次附加的 delta 校验公式曾因浮点类型转换顺序不一致而失败；最终保存脚本按 PEFT 实际行为，先把 delta 转为底座 dtype 再相加，并保留严格相等检查。没有为通过检查改成宽松容差。

三份 SFT 输出的 `model.safetensors` 均为 4,426,558,864 字节；各自完整 SHA-256 见 [登记表](model-registry.json) 中 `reconstruction.weights`。当前 adapter 哈希与各自历史合并记录匹配。此次未重新计算这三份大权重的完整哈希，也未重新执行合并或短生成。

原始脚本与来源文件见 [核查记录](records/2026-10-19/README.md)。这些脚本保留固定的历史绝对路径；复现时应复制到独立运行目录、指定新的输出路径，并记录修改后的脚本哈希，不能覆盖已登记模型。

## Code2 DPO 如何重新训练

旧 `code-dpo30` 的配置依赖 `code-merged3`，当前缺少这一底座。因此没有把它挂到 Code2 上冒充恢复，而是在重建的 `code-merged2` 上训练了新 adapter：

```text
lora-qwen3.5-2b-dpo/lora-qwen3.5-2b-code2-dpo-20261003
```

数据文件为 `dataset/code/train/dpo.jsonl`，训练时记录的 SHA-256 为 `3f58909c82a4efb9ecf5205e4d814d6d106f566b8423c4e5cdb2ac2f370ffb36`。该哈希来自当时的来源记录；本次未下载或重新核对训练数据内容。

关键配置来自保存的 `train_code2.sh`：

| 参数 | 值 |
| --- | --- |
| epoch / seed | 1 / 42 |
| 设备 batch / 梯度累积 | 1 / 16 |
| 学习率 / beta | `1e-6` / `0.3` |
| LoRA r / alpha / dropout | 16 / 32 / 0.05 |
| max_length / max_prompt_length | 3072 / 1024 |
| max_chars / val_ratio | 3000 / 0.10 |
| warmup_ratio / save_steps | 0.03 / 10 |

底座为领域 SFT 完整模型，训练入口采用 4-bit QLoRA。保存的历史训练源码使用随机训练/验证划分，并以 eval loss 选择最佳 checkpoint；它不是收益路由器训练。DPO 的参照策略是冻结的 SFT 底座：历史源码显式使用 `ref_model=None`，由 TRL 在该 PEFT 模型上关闭新 DPO adapter 取得参照策略。

### 确切训练源码

`training_source.json` 记录的 Python 训练源码 SHA-256 与当前 `code/LoRA_Qwen35_Code_DPO_4090.py` 不同。已经从本地 Git blob `526b58cbc27bd571bf4bd08715961828364fd411` 找回完全匹配的历史源码：

- [历史训练源码](records/2026-10-19/operations/code2-dpo-run-20261003/code-dpo-training-source.py)
- [源码身份核对](records/2026-10-19/operations/code2-dpo-run-20261003/code-dpo-training-source.json)
- [当时的执行参数](records/2026-10-19/operations/code2-dpo-run-20261003/train_code2.sh)
- [训练输入来源](records/2026-10-19/operations/code2-dpo-run-20261003/training_source.json)

查阅或复现这次历史运行时，不能默认为当前文件就是当时版本。保存的 shell 原来调用仓库中的 Python 文件；若要重现，应将对应的历史 Python 源码、所需领域 SFT 辅助代码、数据及环境一起冻结到新的运行目录，而不是直接在当前仓库重跑原 shell。

### 已有验证及其含义

`train.exit` 与 `verify.exit` 均为 0。验证记录显示：

- 完整底座与 DPO adapter 成功加载，模型类为 `Qwen3_5ForCausalLM`。
- 228 个 adapter 张量均为有限值，验证脚本还检查 LoRA B 非零。
- 开启/关闭 DPO adapter 后，选定输入的 logits 最大绝对差为 `0.1875`。
- 128-token 贪心生成得到非空 pitfalls；DPO adapter 大小为 48,395,792 字节。
- 此次重新计算的 Code2 adapter 哈希与历史 `verification.json` 一致。

这些是历史运行检查，本次没有重跑模型推理，也不是下游效果证明。原始验证记录保留 `formal_training_certified=false` 和当时对冻结 router 组重合的说明；后来项目改回 SFT/DPO 独立随机划分，这并不自动完成新的隔离审计。不能把 router 组重合直接当作已证明下游测试泄漏，也不能据此宣称完整正式训练协议已认证。

## 正确加载和使用

从服务器仓库根目录使用以下配对原则：

| 目的 | 加载组合 |
| --- | --- |
| SFT 记忆 | 原版 2B + 对应 SFT adapter，或对应 SFT 合并模型 |
| DPO 记忆 | 对应 SFT 合并模型 + 对应 DPO adapter |
| 4B 解题 | 独立加载 4B 主模型，不拿它替代 2B 记忆底座 |

以 Math DPO 记忆为例，完成本地代码版本部署后可使用新入口；下列输入/输出路径需替换为实际实验文件：

```bash
python -m parammem.generate math --method parammem-dpo \
  --base_model models/Qwen3.5-2B-math-merged2 \
  --lora_path lora-qwen3.5-2b-dpo/lora-qwen3.5-2b-math-dpo3 \
  --input_jsonl <本次输入.jsonl> --output_jsonl <本次记忆输出.jsonl> \
  --prompt_key problem --output_key pitfalls_high_temp \
  --temperature 0.2 --max_new_tokens 1024 --num_versions 1 --batch_size 1
```

本次文档没有部署新入口到服务器。也不要把现有 SFT adapter 目录当成完整 SFT 底座，或对 SFT 合并模型再次叠加同一个 SFT adapter。模型选择、数据和采样预算需另行冻结后才能进行公平方法比较。

后续任何重建、重训或 adapter 替换都应按 [文档规范](../documentation-guide.md) 新增日期记录，更新 [模型索引](README.md) 和登记表，保留旧记录。
