# ParamAgent 配置

完整方法：DPO 记忆模型 + 预期收益路由器。现有离线采集配置在 `configs/paired_router.example.json` 和 `configs/router_data_pilot.yaml`；这些文件是示例/数据采集配置，不是已部署的在线系统配置。

历史无路由基线配置说明见 `configs/parammem/README.md`。在线路由推理 runner 尚未接通；新增配置必须记录主模型、专家底座及 adapter、路由器、阈值、预算、停止规则和数据版本。
