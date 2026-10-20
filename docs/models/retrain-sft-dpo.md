# 从已验证SFT结果重训三域DPO

`scripts/retrain_sft_dpo.py`固定使用Code2、Math2、QA的完整SFT合并底座，在其上新建DPO adapter。初始化和reference均保留SFT结果；原版2B不是本次DPO的训练起点。

用户于2026-10-21确认三域采用论文DPO参数：3 epochs、LoRA r32/alpha64、beta0.1、学习率2e-6。其余沿用现有领域数据处理和提示：4bit QLoRA、batch1、累积16、max_length3072、seed42；Code保留max_chars3000过滤。按epoch验证和保存，以验证loss选择最佳checkpoint。

入口重新计算SFT完整权重与SFT adapter的SHA256，对照merge_provenance中的合并来源和数值验证记录，拒绝不匹配、缺少来源或把adapter目录当完整底座。每次运行必须使用不存在的新目录，不覆盖旧DPO。

```bash
python scripts/retrain_sft_dpo.py \
  --repo-root /root/autodl-tmp/lrr/ParamAgent \
  --run-dir /root/autodl-tmp/lrr/paramagent/operations/sft-dpo-retrain-20261021/preflight
```

实际训练使用另一个新的`--run-dir`并加`--train`。在单GPU上串行执行Code→Math→QA，每域训练后用独立进程检查adapter底座身份、r/alpha、有限非零权重、相对SFT的logits变化及非空短生成，失败即停止后续领域。

运行目录保存`status.json`、每域`training_source.json`、训练/验证日志和退出码；新权重在`adapters/{code,math,qa}/`。来源记录包含实际SFT和数据哈希、源码哈希及完整argv。代码应先测试提交，然后冻结六份领域SFT/DPO源码和此入口到独立操作目录执行。

本次重训沿用现有偏好数据，不重新构造执行结果偏好，不改变输入协议，不涉及路由。Code原始530对、现有过滤后515对，与论文2000对仍不同；Math/QA分别2000/3000对。采用论文超参数不代表完整复现论文。SFT历史公开底座revision未验证的限制也仍保留。

本地验证：`python -m unittest discover -s tests -p 'test_retrain_sft_dpo.py' -v`。
