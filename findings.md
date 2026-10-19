# Findings

旧采集依赖 gain_router 模块身份；保留该命名空间以兼容 patch 和历史命令。训练和基线依赖同目录导入，使用保持目录上下文的入口。仅修改本地，不操作正在运行的服务器。

2026-10-19：本地 HEAD ba84b40 且工作区干净；服务器 HEAD 938ef8e，含本地尚无的路由训练提交 e593475 / 938ef8e。服务器另有 actor.py、paired.py 两项修改，以及 paired_storage.py、compact_paired_run.py、run_full_paired_router.py、test_paired_storage.py 四项未跟踪源码。历史 paramagent.runner 实际是基线入口，与新确定的完整方法名冲突；score.runner 仍是离线数据工具。
