# 架构整理

- [complete] 检查依赖和工作区
- [complete] 兼容分离模块和入口
- [complete] 测试与文档
- [complete] 提交准备与暂存审查

验证：Python 3.12，59 项测试通过。默认 Python 缺少 hashlib.file_digest，首次完整测试出现两个环境兼容错误；未修改原调度器，使用已有 Python 3.12 验证通过。

## 2026-10-19 方法命名与服务器源码归档

- [complete] 核对并同步服务器独有提交与未提交源码到本地，保留本地架构整理
- [complete] 统一 ParamMem / ParamMem-DPO / ParamAgent 名称及兼容入口
- [complete] 测试、同步来源记录与 Git 提交

约束：不下载权重和数据，不覆盖服务器已有修改，不将未实现的在线路由器标为完成。

## 2026-10-19 工作区整理与向量脚本纳管
- [complete] 将实验脚本纳入 scripts，并显式配置 source/work-dir/env-file
- [complete] 新增 3 项离线测试，验证缓存迁移、凭据选择与完整导出跳过
- [complete] 更新本地目录职责和使用说明，检查差异并提交
