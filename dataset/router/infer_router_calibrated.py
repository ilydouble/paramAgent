"""
infer_router_calibrated.py
--------------------------
带先验校正的 router 推理。原理:
    adjusted_logit_i = raw_logit_i - log(prior_used_i) + log(true_prior_i)
    prob_i = softmax(adjusted_logit)_i

为什么: 训练集类别分布(10/30/30/30)与真实部署分布(2.3/12/27/59)差异大,
        若直接用 softmax(raw_logit), 模型会因训练先验高估 code/math/qa、
        低估 abstain(该弃权时选了专家)。先验校正把这一偏移修正。

⚠️ 关键前提(务必阅读):
  1. 若训练时已对 code 用了 class_weight=3(参见 train_router.py), 则模型学到的
     "有效先验" 不是原始 train_prior。此时校正项里的 prior_used 必须用
     有效先验 = train_prior * weight / Z, 否则会对 code 双重校正。
     本脚本通过 --consider_weight 开关处理:
        开=True  -> effective_prior (推荐: 用了权重的训练)
        开=False -> raw train_prior (训练未加权)
  2. --true_prior 应填真实部署环境的类别频率; 若是"期望弃权率"也适用, 但语义是决策校准而非统计先验。
"""
import argparse
import json
import os
import sys
import math

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

LABELS = ["abstain", "code", "math", "qa"]  # 与 train_router.py 一致

# 训练集类别分布(来自 5988 条, 与 prepare_router_data 的结果一致)
TRAIN_PRIOR = {"abstain": 0.301, "code": 0.098, "math": 0.301, "qa": 0.301}
# 训练时对 code 使用的 class weight(与 train_router.py 的 CLASS_WEIGHTS 一致)
CLASS_WEIGHT = {"abstain": 1.0, "code": 3.0, "math": 1.0, "qa": 1.0}
# 真实部署类别分布(按观测/目标填写)
TRUE_PRIOR = {"abstain": 0.59, "code": 0.023, "math": 0.12, "qa": 0.27}


def effective_prior(consider_weight):
    """返回推理校正应减去的先验。考虑权重时用含权先验, 否则用原始 train_prior。"""
    if not consider_weight:
        return dict(TRAIN_PRIOR)
    raw = TRAIN_PRIOR
    Z = sum(raw[c] * CLASS_WEIGHT[c] for c in LABELS)
    return {c: raw[c] * CLASS_WEIGHT[c] / Z for c in LABELS}


def calibrate(logits, prior_used, true_prior):
    """adjusted_logit = raw_logit - log(prior_used) + log(true_prior), 返回校正后 softmax 概率(列表, 按LABELS顺序)。"""
    adj = []
    for i, lbl in enumerate(LABELS):
        pu = prior_used[lbl]
        tp = true_prior[lbl]
        adj.append(logits[i] - math.log(max(pu, 1e-12)) + math.log(max(tp, 1e-12)))
    mx = max(adj)
    exps = [math.e ** (a - mx) for a in adj]
    s = sum(exps)
    return [e / s for e in exps]


def parse_args():
    p = argparse.ArgumentParser(description="带先验校正的 router 推理")
    p.add_argument("--model_path", required=True, help="训练好的模型目录")
    p.add_argument("--text", default=None, help="单条输入文本")
    p.add_argument("--stdin", action="store_true", help="从标准输入逐行读取")
    p.add_argument("--file", default=None, help="jsonl, 用每行 --field 字段")
    p.add_argument("--field", default="text")
    p.add_argument("--max_length", type=int, default=512)
    p.add_argument("--topk", type=int, default=4)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    # 先验校正参数
    p.add_argument("--true_prior", default=", ".join(f"{k}:{v}" for k, v in TRUE_PRIOR.items()),
                   help="真实部署分布, 如 'abstain:0.59,code:0.023,math:0.12,qa:0.27'")
    p.add_argument("--no_weight", action="store_true",
                   help="训练未用 class_weight 时设此开关, 用原始 train_prior 校正")
    return p.parse_args()


def parse_true_prior(s):
    out = {}
    for tok in s.replace(";", ",").split(","):
        tok = tok.strip()
        if not tok:
            continue
        k, _, v = tok.partition(":")
        if k.strip() in LABELS:
            out[k.strip()] = float(v)
    # 补全缺失为均匀
    for lbl in LABELS:
        out.setdefault(lbl, 1.0 / len(LABELS))
    return out


def main():
    args = parse_args()
    consider_weight = not args.no_weight
    true_prior = parse_true_prior(args.true_prior)
    prior_used = effective_prior(consider_weight)

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_path).to(args.device)
    model.eval()

    print(f"校正先验:  used={({k: round(prior_used[k],3) for k in LABELS})}, "
          f"true={({k: round(true_prior[k],3) for k in LABELS})}", file=sys.stderr)

    texts = []
    if args.text is not None:
        texts = [args.text]
    elif args.stdin:
        texts = [l.rstrip("\n") for l in sys.stdin if l.strip()]
    elif args.file:
        for l in open(args.file, encoding="utf-8"):
            l = l.strip()
            if l:
                texts.append(json.loads(l).get(args.field, ""))
    else:
        p.error("请通过 --text / --stdin / --file 提供输入")

    with torch.no_grad():
        for text in texts:
            enc = tokenizer(text, truncation=True, max_length=args.max_length, return_tensors="pt").to(args.device)
            logits = model(**enc).logits[0].cpu().tolist()
            probs = calibrate(logits, prior_used, true_prior)
            ranked = sorted(zip(LABELS, probs), key=lambda x: x[1], reverse=True)
            top = ranked[: args.topk]
            best = top[0]
            detail = " | ".join(f"{name}={p:.4f}" for name, p in top)
            print(f"pred={best[0]}  {detail}")


if __name__ == "__main__":
    main()