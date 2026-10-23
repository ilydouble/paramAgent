# 三域作者训练模板 SFT 重训

用户于 2026-10-23 明确选择严格保留作者的训练/推理模板区别，不采用二者完全相同的修正版。
默认仍采用严格作者损失；2026-10-24 新增可选 EOS 监督修正版，须显式启用并独立记录。
`scripts/retrain_author_sft.py` 从有下载来源记录的原版 `Qwen/Qwen3.5-2B` 分别训练新 Code、Math、QA adapter，
不从旧 SFT 继续训练，不覆盖旧 SFT/DPO，不自动启动 DPO，也不直接将旧 DPO 挂到新 SFT 上。

## 固定合同

- 模板来源为 `tianyao-aka/ParamAgent@8b36cddda863ce38cc762976d9ffc55c3540af07`，由共享模块校验原文件哈希。
- Code/Math 训练无推理示例；独立推理和后续验证保留作者推理示例。QA 训练/推理均保留原始两个示例。
- 用户此前确认的作者实际全序列损失保持不变，prompt 与回答参与损失，EOS/padding 掩码为 -100。
- Code 优先完整 raw_texts/raw_text 监督（含错误实现），Math 用 pitfalls，QA 用 decomposition。
- 采用现有服务器三域训练文件，冻结副本、SHA256、行指纹和实际训练/验证行索引。这些数据尚未认证为作者原始 Llama 训练数据；模型替换也属于适配，不能声称逐位复现论文权重。

| 参数 | Code | Math | QA |
| --- | --- | --- | --- |
| epochs | 3 | 2 | 3 |
| lr | 3e-5 | 1.5e-5 | 3e-5 |
| 单卡 microbatch / 累积 | 1 / 96 | 1 / 128 | 1 / 96 |
| LoRA r / alpha / dropout | 64 / 32 / 0.05 | 64 / 32 / 0.05 | 64 / 32 / 0.05 |
| 最大序列长度 | 2048 | 2048 | 2048 |
| 验证比例 / 数据划分 seed | 0.15 / 1 | 0.15 / 1 | 0.15 / 1 |

cosine、warmup 0.03、AdamW、按 epoch 验证/保存，并按验证 loss 选最佳 checkpoint。
模型初始化显式 seed42，Trainer 使用作者未覆盖的默认 seed42；数据 shuffle/split 的 seed1 单独记录。
单卡 QLoRA/NF4/BF16、Qwen 原生 chat、关闭 thinking、可发现的文本投影模块均为模型/硬件适配，
不保证与作者八卡 Llama 训练数值一致。实际 LoRA 模块保存于 adapter_config 和重载验证报告。

## 运行与验证

```bash
python scripts/retrain_author_sft.py \
  --repo-root /root/autodl-tmp/lrr/paramagent/code/ParamAgent \
  --run-dir /path/to/new/run --train
```

不加 `--train` 仅预检。`run-dir` 必须不存在；训练前全量检查 token 长度和损失掩码，
记录所有截断行；任何样本丢失全部目标 token 时拒绝训练，不擅自删行或增加作者长度。
训练数据、底座及源码变化会被拒绝。训练参数关键字段不兼容时拒绝静默忽略。
只有 GPU 空闲才启动下一域，不停止其他任务。

Code → Math → QA 单卡串行，每域结束后使用独立进程验证：原版底座身份、r/alpha、有限非零权重、
完整 epochs、顶层 adapter 与最佳 checkpoint 张量一致、重载 logits 有限且相对原版改变，以及作者推理模板下的非空生成。
任一失败即停后续域，保留日志，不以训练进程退出0代替验证成功。输出为 adapter，尚不是供旧 pilot 直接读取的合并模型。

产物包含 `status.json`、`manifest.json`、`inputs/`、每域 `length_audit.json`、`rows.json`、`split.json`、
`prompt_sample.json`、`training_source.json`、训练/验证日志与退出码、`verification.json` 和 `adapters/{domain}/`。
来源清单保存原版模型权重、tokenizer/processor/chat template、源码和环境版本，不写入远程 Git 仓库。

测试：`python -m unittest discover -s tests -p 'test_retrain_author_sft.py' -v`。

## 可选：保留真实 EOS 监督

Qwen 训练设置 `pad_token=eos_token`。默认作者行为按 token ID 屏蔽，连真实回答末尾 EOS
也置为 `-100`。显式加入 `--preserve_eos` 后，先为未填充序列保留全部标签，只把后续补齐位置
置为 `-100`；真实 EOS（包括 prompt 内的轮次结束符）参与损失。固定/动态 padding 均支持。
这仍是 prompt + 回答的全序列监督，不改为 completion-only，不修改模板、示例、超参数或推理停止符。

```bash
python scripts/retrain_author_sft.py \
  --repo-root /path/to/ParamAgent \
  --run-dir /path/to/new/eos-supervised-run --preserve_eos
```

上述命令仅预检；明确需要训练时再加 `--train`。三个 `LoRA_Qwen35_*_4090.py` 独立入口也支持
同名开关，须显式指定新的 `--output_dir`，不覆盖旧版产物，也不要从旧损失协议 checkpoint 续训。
启动器要求全新目录，记录 `preserve_eos` 及 `loss=full_sequence_padding_only_masked`，并将开关
传给真实训练进程；严格默认记录 `loss=full_sequence_pad_eos_masked`。预检样本同时保存 attention mask，
逐行及汇总记录回答中保留/受监督的 EOS token 数。若 EOS 被长度截断，不伪造或强行补回 EOS。

这是区别于作者实际 collator 行为的 **Qwen 适配修正版**，不冒称严格作者复现。
代码修正不会修复已有权重，也未证明 EOS 掩码是重复生成的唯一原因；须用后续受控训练和推理比较验证。
