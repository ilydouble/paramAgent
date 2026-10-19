# ParamMem / ParamMem-DPO 配置

两种方法共用 `python -m parammem.runner` 和 `python -m parammem.generate`，通过必填 `--method parammem` / `--method parammem-dpo` 区分。比较时冻结主模型、数据、提示模板、采样参数、生成条数和迭代预算，仅替换记忆权重。

历史 `run_param_*` 默认所用记忆：Code 为 DPO，Math / QA 为 SFT；不是三域全为 DPO。权重与数据路径仍需在部署机器核实。历史生成脚本尚未统一协议，不可直接当作公平对照配置。
