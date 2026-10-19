# 本地研究工作区

本代码仓库位于 `icassp2027/ParamAgent/`，同级目录按职责分开：`paper_write/` 保存论文，`experiments/` 保存实验输入和缓存，`project_notes/` 保存项目记录，`archive/` 保存历史备份。旧源码快照不作为开发入口。

## 两套独立版本记录

代码在本仓库中提交。论文源码和插图在 `../paper_write/` 的独立本地 Git 仓库中提交。整个工作区不再建立总仓库，避免重复跟踪源码或把缓存和传输包混入历史。目录指南位于上一级 `README.md`。

2026-10-19 的移动记录及原始文件哈希位于 `../project_notes/organization_20261019/moves.json`。本仓库先前 `docs/server_sync_20261019.json` 中的同步代码哈希继续有效，实际传输包现在放在 `../archive/server_sync/server_sync_20261019/`。

## 向量生成脚本与实验产物

`build_api_vectors.py` 已纳入 `scripts/`；不再依赖个人 Downloads 或其他项目的 `.env` 路径。输入 ZIP、SQLite 缓存、向量和导出 ZIP 保存在代码仓库外。

从本仓库根目录运行：

```bash
python scripts/build_api_vectors.py \
  --source ../experiments/router_embedding3/input/router-dataset-20261017.zip \
  --work-dir ../experiments/router_embedding3
```

重新生成时从环境变量 `ZHIPUAI_API_KEY` 读取密钥，或显式传入 `--env-file /你的本地凭据文件`；凭据不进入输出。已有完整 manifest 时直接返回，不读取凭据、不访问 API。缓存键和分块/聚合协议与原脚本一致，因此搬迁后可继续复用已有缓存。

输入 ZIP 是原 Downloads 文件的核对副本，保留原文件。此构建器继续使用原来的 3372 条 Code/QA 冻结协议；目录整理未修改样本、标签、向量、模型或实验结论。
