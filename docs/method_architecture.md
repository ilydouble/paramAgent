# ParamAgent 与 SCORE 的代码边界

## 方法与训练阶段

ParamAgent 保留历史反思、搜索和记忆库基线。SCORE 将偏好专家、策略执行、路由数据和收益控制器分开。SFT/DPO 是专家训练阶段，不是系统方法名称。

```text
shared/              保持旧脚本上下文的公共入口工具
paramagent/runner.py 历史基线入口
score/
  experts/           LocalPreference 专家接口
  policies/          配对分支策略接口
  controller/        收益控制器预留边界（尚无在线模型实现）
  data/              迁移后的轨迹、标签、特征及审计实现
  runner.py          配对数据 prepare/collect/label 入口
training/sft/        SFT 入口
training/dpo/        DPO 入口
configs/paramagent/  后续基线配置
configs/score/       后续 SCORE 配置
```

## 使用方式

从仓库根目录执行，剩余参数原样传给历史训练或基线脚本：

```bash
python -m paramagent.runner code --dataset_path ...
python -m training.sft code --dataset1_path ...
python -m training.dpo code --dataset_path ...
python -m score.runner --help
python -m score.runner collect --tasks ... --settings ... --output ...
```

训练参数以原脚本为准。配对 collect 默认预览，实际采集仍需显式 `--execute`。原 scripts/run_full_paired_router.py 保持完整池调度职责。

## 兼容与运行安全

- 原 code/、math/、qa/ 的训练和基线脚本保持原位，避免破坏同目录导入及服务器命令。
- gain_router 包通过 __path__ 定位 score/data 的源码；旧 gain_router.* 导入、mock 路径和 `python -m gain_router.paired` 保留。
- 数据实现暂统一使用 gain_router.* 模块身份，避免同时导入 score.data.* 产生两个模块实例；新调用方使用 score.runner、score.experts 和 score.policies。
- Actor、verifier 和数据 schema 当前仍位于 score/data，由现有模块复用；不复制实现。进一步提取公共组件应独立验证。
- 旧四分类 router 继续留在 dataset/router/，不作为 SCORE 收益二分类模型。
- 原 configs 的文件不移动，模型权重、数据、输出和提示模板不修改。本次只在本地整理，不同步服务器。
- 源文件内容保持不变，但物理路径与迁移提交变化，应为后续采集建立新代码版本记录；正在运行的服务器继续使用原版本。

## 当前实现范围

本轮完成模块归属和独立入口，未改变训练算法或采集协议。score.runner 是离线数据工具，不是已接入训练控制器的在线 SCORE 推理器。在线控制器训练、校准和系统 runner 仍需单独实现与验证。
