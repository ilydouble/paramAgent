# ParamAgent 实验术语

区分主模型基线、记忆优化和带收益路由的完整方法。

## Language

**Qwen3.5-4B 基线**：仅由 Qwen3.5-4B 主模型解题的对照，不使用 ParamMem 记忆模块。

**ParamMem**：本文复现的参数化记忆对照，使用领域 SFT 的 2B 记忆模型为同一个主模型提供提示。
_Avoid_：ParamAgent 基线、ParamAgent+、ParamAgent-Plus

**ParamMem-DPO**：保留 ParamMem 推理流程，将其领域 SFT 2B 记忆模型替换为继续经过 DPO 优化的版本。
_Avoid_：DPO（单独用作系统方法名）、ParamAgent（指代只有 DPO 的版本）

**ParamAgent**：我们的完整方法，由 DPO 记忆模型与预期收益路由器共同组成，按预期收益决定是否调用记忆。
_Avoid_：SCORE（历史工作名）、ParamMem（指代完整方法）

**预期收益路由器**：根据推理时可见信息预测调用记忆的收益，并决定是否调用的控制器。
_Avoid_：领域分类路由器（旧四分类模型，含义不同）
