# 小型收益路由器

入口 train_small_gain_router.py，与旧的四分类 train_router.py 独立。
预测首次回答之后、进入固定五轮修订之前，指导分支能否优于纯 Actor 分支。
当前只支持 Code/QA；Math 原标签存在评分选择偏差，训练命令会拒绝 Math。

## 训练与推理

服务器基线环境 /root/miniconda3/bin/python 已包含 numpy/scipy/scikit-learn/joblib。
在服务器仓库 /root/autodl-tmp/lrr/paramagent/code/ParamAgent 下执行：

    OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 /root/miniconda3/bin/python dataset/router/train_small_gain_router.py train --run-dir /root/autodl-tmp/lrr/paramagent/experiments/router-paired-2000-per-domain-20261004 --output-dir /root/autodl-tmp/lrr/paramagent/experiments/router-training-new/tfidf

输出目录必须不存在，避免覆盖已有模型。固定 seed=42，3 折分组交叉验证。
在训练折内拟合词表，比较 C=0.1/1/10 与普通/平衡类别权重，共 6 个候选。
用训练集 OOF average precision 选参数；用同一 OOF 分数选择最大净收益阈值，
相同收益优先少调用。默认 --call-penalty 0.02，单位是每次调用折合的任务成功数；
它是实验成本假设，并非实际货币或精确 token 成本。正式实验应事先确定成本。
超参数和阈值不使用验证集；OOF 选择指标有选择乐观偏差，不能当无偏测试指标。

冻结编码器：使用独立 router-small-env，准备本地 all-MiniLM-L6-v2 模型后，
在上述训练命令添加 --backend embedding --encoder-path /root/autodl-tmp/lrr/paramagent/models/all-MiniLM-L6-v2 --device cuda。
编码器参数不更新；问题/上下文与首次回答/推理分别编码、attention mask 均值池化并 L2 归一化，
拼接两份向量与初始状态特征，只训练逻辑回归头。每份文本最多 256 token。
模型保留外部编码器绝对路径，部署时需同时保留它；TF-IDF 模型可独立使用。

推理输入为 JSONL，每条包含 sample_id（可选）、domain（code/qa）及 features 对象。
features 包含 problem、context、initial_answer、initial_reasoning，以及可选 confidence/online_feedback。
推理不需要 label、gold 或分支结果。已有包含 Math 的完整验证数据必须先筛选支持领域。

    /root/miniconda3/bin/python dataset/router/train_small_gain_router.py predict --model /absolute/output/router.joblib --input /absolute/code-qa-input.jsonl --output /absolute/predictions.jsonl

输出 score 和 call_guidance。score 是模型判别分数对应的加权训练概率，未承诺校准；
应使用随模型保存的阈值。非支持领域显式报错，不自动扩展为 Math 策略。

## 数据与评估边界

输入只允许问题、在线可见上下文、首次回答、首次推理、领域、长度、初始 logprob/token 数及非空/截断标志。
未知字段不会进入模型。gold、后续修订、指导内容、最终分数和分支成本均不能作为输入。
详细 labels.jsonl 仅用于身份核对和离线计算改进/退化/任务成功率/额外 token 成本。
训练/验证的 sample_id、group_id、规范化问题与上下文交集必须为空，且原划分标签一致。
这不证明与专家 DPO 数据独立：本批题参与过专家训练，泛化结论还需专家与路由都未见的独立测试题。
排除题复核池当前不合并，原标签与数据文件保持不变。

产物：router.joblib、report.json、train_oof_predictions.jsonl、val_predictions.jsonl。
report 保存输入 SHA256、依赖版本、完整命令、候选得分、选定阈值、always/never 基线和调用预算曲线。
预算曲线只用于探索，不用它在验证集上重选阈值，也不把它当独立测试结果。
验证正例只有 8 条，单次划分结果波动很大。训练结束再次校验源文件哈希。

## 检查

    /root/miniconda3/bin/python dataset/router/test_small_gain_router.py

覆盖离线信息隔离、划分交集、验证词表隔离、空调用阈值、退化收益以及训练/保存/预测闭环。
