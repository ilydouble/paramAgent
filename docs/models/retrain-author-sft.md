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

仅正式训练 QA 时，使用全新目录并传 `--domains qa --qwen_adapted --train`。
`--domains` 同时限定预检、冻结输入、训练、重载验证和状态记录；不会在 QA 之后启动其他域。
省略该选项仍保留三域顺序；空域、重复域和未知域拒绝。正式 QA 仍为3 epochs、累积96、验证比例0.15，
不继承短测的1 epoch/累积4/16样本参数，也不从短测adapter或旧SFT继续训练。

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

修正版还在编码前统一回答结尾为单个 EOS：原回答已经带 EOS 时不再重复追加，连续多个结尾 EOS
（包括中间夹尾随空白）也合并为一个；正文内部的 EOS 不删除。训练和预检共用同一处理函数，
审计通过 `answer_eos_normalized_row_indices` 记录受影响行。原始数据文件及行指纹不变，
严格作者默认仍逐字保留原回答并无条件追加 EOS。已确认的 QA 零基行 18635 因此不会在修正版产生双 EOS。

这是区别于作者实际 collator 行为的 **Qwen 适配修正版**，不冒称严格作者复现。
代码修正不会修复已有权重，也未证明 EOS 掩码是重复生成的唯一原因；须用后续受控训练和推理比较验证。

## 可选：Qwen3.5 架构适配版

启动器和三个独立训练入口支持 `--qwen_adapted`，默认仅注意力，策略为 `qwen35_attention_projections_v1`。
只有额外显式加入 `--qwen_include_mlp` 才启用注意力＋MLP对照，策略为 `qwen35_language_projections_v1`。
MLP 开关必须与 `--qwen_adapted` 一起使用；两种模式的开关及策略均记录于运行清单和覆盖报告。
该模式自动开启 `--preserve_eos`（含单个结尾 EOS 规范化），但不修改提示词、示例、全序列损失范围、
学习率、rank/alpha、epochs、batch、长度或验证集。仅回答计损失仍未启用，以免同时改变损失和覆盖范围。
不开新开关时保持旧模块选择行为，不改原始作者文件、旧 adapter 或 DPO 入口。

覆盖按实际语言分支的完整路径和 config 中每层类型逐一校验：

| 部分 | 目标投影 |
| --- | --- |
| full_attention / self_attn | q_proj、k_proj、v_proj、o_proj |
| linear_attention / linear_attn | in_proj_qkv、in_proj_z、in_proj_a、in_proj_b、out_proj |
| 每层 MLP（仅显式对照开启） | gate_proj、up_proj、down_proj |

仅支持当前拆分投影的 dense Qwen3.5 架构；其他架构、未知/融合投影布局、缺少任何预期投影、
实际层数不符或语言分支不唯一均直接拒绝，不能把“发现任意一个模块”当作适配成功。
目标用锚定完整路径的正则传给 PEFT，不以尾部名称跨分支匹配；视觉分支、embedding、输出头均排除。
注入后要求实际 LoRA 模块集合与计划完全相同，各目标 A/B 可训练且无额外可训练底座参数。

```bash
python scripts/retrain_author_sft.py \
  --repo-root /path/to/ParamAgent \
  --run-dir /path/to/new/qwen-adapted-run --qwen_adapted
```

该主配置不向 MLP 注入 LoRA，MLP 原始权重仍冻结；线性注意力自身的门控投影不是 MLP，仍按上表覆盖。
需要注意力＋MLP 对照时，使用另一个全新 `--run-dir`，并额外加入 `--qwen_include_mlp`。
独立入口的两种默认输出名也分开，不能用同一个目录混用覆盖范围。
同日再次用真实配置和 PEFT 做 meta 检查：仅注意力为 114 个投影、29,528,064 个可训练参数，
MLP adapter 数为 0；显式对照仍为 186 个投影、67,276,800 个参数，其中 72 个 MLP adapter。
可训练参数减少不代表总训练耗时同比减少，训练速度和效果仍需实际验证。

上述命令仍仅预检，明确需要训练时再加 `--train`。正式训练在首次更新前校验并保存
`adapters/{domain}/qwen_lora_coverage.json`，含每层完整路径、目标正则、实际注入数和可训练参数量。
manifest、training_source、status 及最终 verification 区分适配模式；重载验证重新计算架构计划并核对
保存报告、adapter 配置和实际注入。不支持此新模式续训旧 checkpoint，独立入口要求全新输出目录，
默认输出名也与 author-sft 分离。当前版本要求新训练，不支持 checkpoint resume。

2026-10-24 初版含 MLP 的服务器 meta 检查得到 24 层（18 linear / 6 full）、186 个投影、
67,276,800 个可训练参数；该结果对应显式注意力＋MLP对照，不是当前默认主配置。
仅使用原版 2B 配置构造 meta 空模型并执行真实 PEFT 注入，所有参数仍为 meta。
同时确认 Linear4bit 是受支持的 Linear 子类。此验证未加载实际权重、未做量化模型前后向或 GPU 训练，
不证明任务收益或显存预算；该方案属于可审计的 Qwen 扩展覆盖，不宣称唯一正确或最优。

测试：`python -m unittest discover -s tests -p 'test_qwen_sft.py' -v`。

## 有界QA训练链路短测

`scripts/smoke_qwen_sft.py --repo-root /path/to/ParamAgent --run-dir /path/to/new/smoke --source-commit <commit>`
仅用于训练/保存/重载验证，不是任务效果实验，也不会接着启动三域训练。要求新目录和空闲GPU，
从原版2B起步，冻结16条QA（含已有结尾EOS的零基18635行），实际QA入口按seed1划分12训练/4验证。
短测专属覆盖为1 epoch、累积4、验证比例0.25，共3次更新；其余保持仅注意力、r64/alpha32、lr3e-5、
长度2048、全序列loss、EOS监督。LoRA初始化与Trainer seed42；不以短测配置替代正式训练参数。
训练和重载是独立进程，检查有限loss/梯度、非零B权重、选定checkpoint与保存张量一致、模块范围、
重载loss/logits有限且adapter改变logits、128-token预算下非空生成；生成未到EOS只记录，不冒称重复退化已解决。
记录完整来源哈希/样本索引、日志、退出码、status和verification；失败就停止，不干预其他GPU进程。
