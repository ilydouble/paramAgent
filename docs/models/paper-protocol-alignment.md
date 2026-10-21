# ParamMem SFT 与 SCORE DPO 协议对齐

## 基准与状态

- 原 ParamMem 论文：<https://arxiv.org/abs/2602.23320>，正文与附录 B.2、B.6。
- 作者代码：<https://github.com/tianyao-aka/ParamAgent/tree/8b36cddda863ce38cc762976d9ffc55c3540af07>。
- SCORE：用户指定的 `icassp2027 (19).pdf`（SHA256 `0477bb9d6b7c5c29902c3b7b2fc6e52c8d4e16ccaae8fc266a930be862f9ae3e`），第 2.1 节和第 3.1 节；本文档不认定或修改本地四份 TeX 稿的主稿。
- 用户 2026-10-22 确认 SFT 严格复现作者实际的全序列损失行为，而非保留 completion-only 修正版。
- 当前只完成部分本地源码对齐和 CPU 检查。没有新权重、没有完整复现认证，没有部署、造数、训练或复跑。

## SFT 已修正的源码

1. `shared/parammem_prompts.py` 直接从仓库现存且与作者逐字节一致的六份 Llama 原脚本中读取字面量，校验固定 SHA256；不导入它们的训练依赖。
2. 分开保留作者的训练模板与批量推理模板。Code/Math 训练模板没有示例，批量推理有示例；QA 保留作者原始两个示例、弯引号和空白。不能为了“训练推理一致”删除作者推理示例或给训练擅自增加示例。
3. 三域 Qwen SFT、独立推理及 pilot 记忆提示共用上述来源；Qwen 采用原生 chat template，显式 `enable_thinking=False`。这属于模型替换的适配，不能声称与 Llama 的 `[INST]` token 序列一致。
4. Code 监督优先级恢复为 `raw_texts`、`raw_text`、最后才是 `pitfalls`，不再优先选择缩短版易错点；完整生成结果也不再被独立推理脚本裁掉错误实现部分。函数签名仅 strip，不再在 SFT 入口清理围栏或重拼 docstring。支持显式第二份 Code 训练数据。
5. 三域 prompt 与目标均参与 SFT 损失；与作者的 `pad_token=eos_token` 一样，EOS/padding token 标为 -100。CPU 最小例证已确认作者所用 LM collator 会将 `[-100, 6, -100]` 重建为 `[5, 6, -100]`，因此作者 Code/Math 预构造的 prompt 掩码在实际 collator 后不生效。
6. QA 加载器兼容作者 JSON list 与现有 JSONL，不替换数据本身。
7. Actor 提示恢复作者原始 Code 错误实现措辞和 Math 模板；未改变评分器、未恢复已撤销的 bac8b45，也未取消已授权的 Math 8192 输出/16384 上下文设置。

### SFT 配置基准

作者 shell 命令优先于 Python 默认值；单卡累积步数按作者 shell 中 8 卡的名义全局 batch 换算。

| 设置 | Code | Math | QA |
| --- | --- | --- | --- |
| epochs | 3 | 2 | 3 |
| lr | 3e-5 | 1.5e-5 | 3e-5 |
| LoRA r / alpha | 64 / 32 | 64 / 32 | 64 / 32 |
| 验证集比例 / seed | 0.15 / 1 | 0.15 / 1 | 0.15 / 1 |
| 原作者全局 batch | 8 x 1 x 12 = 96 | 8 x 1 x 16 = 128 | 8 x 1 x 12 = 96 |
| Qwen 单卡 microbatch / 累积 | 1 / 96 | 1 / 128 | 1 / 96 |
| 最大序列长度 | 2048 | 2048 | 2048 |

保留 cosine、0.03 warmup（按完整训练步数计算）、按 epoch 评估/保存、验证损失选 checkpoint。优化器改为 AdamW，而非原 Qwen 脚本的 paged 8-bit AdamW。新默认输出名含 `author-sft`，不覆盖既有权重目录。

Qwen 适配还包括单卡 NF4/BF16、原生 tokenizer、混合注意力架构上实际可发现的文本 LoRA 投影层；target_modules 不保证与 Llama 的四个投影层逐名相同，训练前需冻结实际列表。相同名义 batch 不保证与 8 卡训练数值逐位一致。

## SFT 尚未通过的复现条件

### 数据不是同一份已冻结训练集

服务器现有数据只读核对：

| 数据 | 数量 | SHA256 |
| --- | --- | --- |
| Code train/code.json | 8163 | 29d6a8ac10c5ffb3ac48571465d78649ebacd81fdb496a44a04725afc4407228 |
| Math train/math.json | 6333 | f39a600af522d2b734467cf5c5a89bcda234fb214f355714f2eae793017bf835 |
| QA train/sft.jsonl | 24309 | e80a8ed8fc0e5698ab8289b74db864b8e945d03926ed1ec5bf7b300835b1ff33 |

Code 其中 8152 行同时含 `pitfalls` 和完整 raw 字段；旧加载器选择了前者，不能用“输入文件一样”证明训练监督相同。

作者公开快照中 APP 文件有 4626 行，synthetic 文件有 4171 行，shell 指向的 augmented 文件缺失；Math shell 指向的文件有 9598 行；QA shell 指向的训练文件缺失。论文描述的采样规模与公开文件总量亦不是一回事。不能将任意同数量子集、同名文件或当前服务器数据直接认证为原论文训练集，更不能补造缺失数据。

需要明确数据来源、采样种子/划分、去重去污染、最终行 ID 和哈希，再冻结训练清单。当前 shell 仍指向现有服务器数据路径，仅用于显式选择数据；它本身不构成数据复现证明。

### 推理协议仍有历史差异

当前三域带 bank runner 的初始尝试数、部分循环起点/退出阈值、恢复与第二阶段落盘逻辑均不完全等于作者快照；尚未在本次改动中重写。作者在线 Math/QA 引用模块本身缺失，复现路径应明确为 README 的批量提示生成流程，不擅自拼出缺失类。

pilot 仍为诊断适配：CPU MiniLM 检索、SymPy/EM 评分、Code 官方测试参与反馈。它不等于稿件中 Code 仅用可见测试、QA F1 >= 0.80 早停且最终报无阈值 F1 的正式协议。保留当前保护，不改判历史成绩。

## DPO 必须满足的合同

以下来自 SCORE 第 2.1 节；SFT 初始化则是用户另行明确的要求，不冒称稿件已完整写出 SFT 预热细节。

1. 从经过上述 SFT 核验的 checkpoint 初始化 policy；reference 固定为同一 SFT。将 SFT 合并进底座再新增 DPO LoRA 可以保留 SFT，不要求机械地叠加两个未合并 adapter。
2. 在训练划分上保存初始失败 Actor 回答与验证结果。DPO 记忆输入必须可追溯到失败轨迹，而非只有题目字符串；不能仅因 schema 有 gold/solution 就当作失败轨迹。
3. 每个训练失败以温度 0.7 采样 4 条指导，保存采样模型/权重、提示、解码配置、seed 与候选 ID。
4. 冻结 Llama-3.1-8B-Instruct，逐条执行候选指导后的修复并记录回答、验证器版本、可见反馈与真实判定；不能用 Qwen4B 的旧诊断轨迹冒充稿件指定的偏好采集 Actor。QA 按稿件的 EM 或 F1 >= 0.80 判定修复，Code 隔离正式测试集；无法判定的评分需隔离，不能为凑 pair 擅自改判。
5. winner 来自成功修复，loser 来自失败或可核验的循环；全成功/全失败不得虚构成功-失败对。循环定义、pair 选择策略、每状态多个 pair 的处理及失败输入的精确模板，稿件尚未完全指定，需显式冻结为工程补充。
6. 目标为 Code 2000、Math 2000、QA 3000 对；按失败状态/原题分组划分，不能让同一状态的候选跨训练和验证。
7. 使用标准 DPO 对 chosen/rejected 指导的条件概率比优化；固定 reference、仅 memory adapter 接收梯度。新正式入口需显式冻结 sigmoid DPO loss、beta=0.1、lr=2e-6、3 epochs、r=32、alpha=64，并验证 TRL 版本和实际生效参数，不能静默忽略关键字段。
8. 冻结 prompt/长度预算后统计截断；不得把截掉指导或失败轨迹的样本当作完整训练记录。

## 现有 DPO 的结论

| 条件 | 20261021 重训状态 |
| --- | --- |
| 严格继承当时 SFT 权重、冻结 SFT reference | 有哈希和重载证据 |
| r32 / alpha64 / beta0.1 / lr2e-6 / 3epochs | 实际命令已满足；顶层 adapter 为验证损失选出的 checkpoint |
| SFT 自身已复现原论文 | 未通过；不能因继承正确而假定底座协议正确 |
| 失败轨迹作为记忆输入 | 不满足，三个训练器均仅以题目构造输入 |
| 4 候选与冻结 Llama Actor 执行证据 | 现有偏好文件没有相应字段或关联 ledger，来源无法证实 |
| 数据量 | Code 原始 530 对、长度过滤后 515；Math 2000；QA 3000 |

缺少 provenance 不能证明历史上绝未执行验证，但足以说明现有数据不能认证为稿件协议。旧三域 DPO 只可标为 legacy question-only 偏好训练，不是正式 SCORE DPO。

`scripts/retrain_sft_dpo.py --train` 现在默认拒绝旧协议；仅显式 `--allow-legacy-question-only` 才可复现历史训练。这个开关不是新 DPO 实现，也不补齐任何缺失证据。三个旧领域 DPO 脚本仍保留历史用途，不能绕开 launcher 直接运行后声称论文对齐。

必须先核验/重建 SFT，再重建执行偏好集、实现失败条件的 DPO 训练和同合同推理，最后才可重训。不能给旧 adapter 换路径、改 metadata 或重评旧轨迹来补齐这些步骤。

## 基线与消融的边界

原 ParamMem 是题目条件的 SFT 基线；SCORE 描述失败轨迹条件的 DPO。两者不能为“提示词相同”而消掉方法本身的输入差异。若报告 w/o DPO 消融，还需固定同一个失败条件推理合同，仅改变是否做 DPO；不能将该消融与原 ParamMem 基线混称。稿件未定义的 SFT 预热/输入格式需明确补充，不由代码审计擅自改论文。

## 验证与后续

新增 CPU 回归覆盖作者源文件哈希、训练/推理各自示例、全序列损失及 EOS/padding 掩码、完整 Code 监督优先级、QA 文件格式、SFT 默认参数、Actor 原始提示字符串和旧 DPO 拒绝入口。

下一阶段仍需：冻结完整数据清单；对齐尝试数与流程；实现可追溯偏好采集、失败条件训练/推理及独立验证；在实际训练环境核验 TRL/Transformers 参数、生效 token 模板和显存需求。没有这些证据前不声称两个方法已完整复现，也不启动新 GPU 作业。
