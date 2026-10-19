# 方法命名、代码入口与本地维护

规范术语见 [CONTEXT.md](../CONTEXT.md)。模型底座、adapter 配对与重建来源见 [服务器模型索引](models/README.md)。仓库仍叫 ParamAgent，目录名不代表某次运行使用的实验方法。

| 实验名称 | 主模型 | 记忆模型 | 收益路由 |
| --- | --- | --- | --- |
| Qwen3.5-4B | Qwen3.5-4B | 无 | 无 |
| ParamMem（本文复现） | 同一个 Qwen3.5-4B | 领域 SFT 2B | 无 |
| ParamMem-DPO | 同一个 Qwen3.5-4B | 在对应 SFT 2B 上继续 DPO | 无 |
| ParamAgent | 同一个 Qwen3.5-4B | 对应 DPO 2B | 有 |

SFT/DPO 仍是训练阶段名。`score/` 和部分历史报告中的 SCORE 是完整方法的旧工作名。现有框架支持其他主模型，历史脚本可能仍默认 Llama；方法命名本身不会把主模型自动换成 Qwen。

## 代码组织

```text
parammem/runner.py     ParamMem / ParamMem-DPO 任务评测，共用领域流程
parammem/generate.py   两种记忆的生成入口，共用领域推理代码
paramagent/
  data/               配对轨迹、标签、特征与审计源码
  experts/            记忆专家接口
  policies/           策略执行接口
  controller/         收益控制器边界
  runner.py           离线 prepare/collect/label 命令
score/                旧工作名的兼容入口
gain_router/          旧模块身份及命令兼容层
shared/               保持历史脚本同目录导入的运行工具
training/sft/         SFT 训练入口
training/dpo/         DPO 训练入口
dataset/router/       路由器离线训练代码，以及旧四分类模型代码
configs/parammem/     无收益路由的两种记忆方法说明
configs/paramagent/   完整方法配置说明
```

## ParamMem 与 ParamMem-DPO

从仓库根目录运行，记忆文件必须来自相应训练阶段。`--method` 用于明确方法及默认结果目录，不会检测记忆内容是否确实来自 SFT/DPO；需保存生成命令与 checkpoint 来源。

```bash
python -m parammem.runner code --method parammem \
  --memory-file generated_reflections/sft/code/memory.jsonl \
  --dataset_path dataset/code/test/humaneval.jsonl \
  --model vllm:Qwen3.5-4B --language py --max_iters 6

python -m parammem.runner code --method parammem-dpo \
  --memory-file generated_reflections/dpo/code/memory.jsonl \
  --dataset_path dataset/code/test/humaneval.jsonl \
  --model vllm:Qwen3.5-4B --language py --max_iters 6
```

`code` 可换成 `math` 或 `qa`；默认策略为 `dot`，其余领域参数原样传递。新入口统一接收 `--memory-file`，内部映射到各领域历史参数，默认结果分别写入 `results/<domain>/parammem` 和 `results/<domain>/parammem-dpo`。自定义 `--run_name` 与 `--root_dir` 时请继续保留方法区分。

记忆生成使用同一份领域 Python 实现，所有生成参数原样转发：

```bash
python -m parammem.generate math --method parammem-dpo \
  --base_model models/Qwen3.5-2B-math-merged2 \
  --lora_path lora-qwen3.5-2b-dpo/lora-qwen3.5-2b-math-dpo3 \
  --input_jsonl dataset/math/test/questions.jsonl \
  --output_jsonl generated_reflections/parammem-dpo/math/memory.jsonl \
  --prompt_key problem --output_key pitfalls_high_temp \
  --temperature 0.2 --max_new_tokens 1024 --num_versions 1 --batch_size 1
```

这是命令格式示例；本地不包含权重和数据，不代表模型部署已验证。DPO 必须使用训练时对应的 SFT 合并底座与 DPO adapter。SFT 可以使用原始 2B + SFT adapter，或 SFT 合并权重。

## ParamAgent 与兼容入口

```bash
python -m paramagent.runner --help
python -m paramagent.runner collect --tasks ... --settings ... --output ...
python -m training.sft code --dataset1_path ...
python -m training.dpo code --dataset_path ...
```

`paramagent.runner` 当前是离线数据入口，采集默认预览，实际调用模型需 `--execute`。路由器已有离线训练实现，但尚未接通完整在线门控任务 runner；不能仅凭这个名称宣称完整方法已评测。

- `python -m score.runner` 和旧 `gain_router.*` 命令继续有效。
- 旧 `python -m paramagent.runner code|math|qa ...` 保持基线行为并打印迁移提示；它不会运行带路由器的完整方法。
- `code/`、`math/`、`qa/` 原始任务、训练与生成脚本保留，历史数据和结果不移动或重新标记。
- `gain_router.*` 仍是数据模块的统一 Python 身份，源码位于 `paramagent/data/`，防止 mock 和历史导入失效；避免直接导入同名 `paramagent.data.*` 或 `score.data.*` 子模块。
- 原 `infer_reflections_qwen35_2b*.sh` 是历史配置，Code 的题目字段/token 上限、Math 的温度/条数，以及三域输入路径存在差异。Math DPO 历史脚本还将 SFT adapter 目录作为底座。这轮只整理命名，未修正或统一这些实验参数。公平对照需另行冻结同一套协议。

## 本地为代码维护入口

本地 Git 仓库是 `icassp2027/ParamAgent`。服务器实际项目为 `/root/autodl-tmp/lrr/paramagent/code/ParamAgent`，旧 `/root/autodl-tmp/lrr/ParamAgent` 为兼容链接。

2026-10-19 已从服务器合并路由训练提交 `e593475`、`938ef8e`；6 个未提交源码与本地逐字节一致，保留本地迁移结构。来源与哈希见 [server_sync_20261019.json](server_sync_20261019.json)。此次命名变更在本地提交，尚未部署回服务器。

后续流程：本地改代码 → 本地验证 → 本地 Git 提交 → 服务器部署指定提交 → 记录运行代码版本。服务器临时修复需回传 diff/新增源码并本地提交；不可只留在服务器。模型、数据、缓存、密钥与实验输出不随源码提交。
