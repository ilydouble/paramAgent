"""
infer_router.py
---------------
用训练好的 router 模型对输入做出路由预测（4 类: abstain/code/math/qa）。

用法(服务器):
    1) 对单条文本:
        python3 infer_router.py --model_path router_model_output --text "def fib(n): ..."
    2) 从 stdin 批量(每行一条文本):
        echo -e "文本1\\n文本2" | python3 infer_router.py --model_path router_model_output --stdin
    3) 或对 router_test.jsonl 里某类文本给出概率:
        python3 infer_router.py --model_path router_model_output --file router_test.jsonl --topk 3

标签顺序与 train_router.py 一致: LABELS = ["abstain","code","math","qa"]
"""
import argparse
import json
import os
import sys

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

LABELS = ["abstain", "code", "math", "qa"]  # 必须与训练脚本一致


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", required=True, help="训练好的模型目录/output_dir")
    p.add_argument("--text", default=None, help="单条输入文本")
    p.add_argument("--stdin", action="store_true", help="从标准输入逐行读取文本")
    p.add_argument("--file", default=None, help="读 jsonl, 用每行的 text 字段(或按 --field)")
    p.add_argument("--field", default="text", help="配合 --file 使用, 取哪一字段作输入")
    p.add_argument("--max_length", type=int, default=512)
    p.add_argument("--topk", type=int, default=4)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def predict(tokenizer, model, text, max_length, device):
    enc = tokenizer(text, truncation=True, max_length=max_length, return_tensors="pt").to(device)
    with torch.no_grad():
        logits = model(**enc).logits
        probs = torch.softmax(logits, dim=-1)[0]
    scores = probs.cpu().tolist()
    ranked = sorted(zip(LABELS, scores), key=lambda x: x[1], reverse=True)
    return ranked


def main():
    args = parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_path).to(args.device)
    model.eval()

    texts = []
    if args.text is not None:
        texts = [args.text]
    elif args.stdin:
        texts = [line.rstrip("\n") for line in sys.stdin if line.strip()]
    elif args.file:
        for line in open(args.file, encoding="utf-8"):
            line = line.strip()
            if line:
                texts.append(json.loads(line).get(args.field, ""))
    else:
        parser.error("请通过 --text / --stdin / --file 提供输入")

    for text in texts:
        ranked = predict(tokenizer, model, text, args.max_length, args.device)
        top = ranked[: args.topk]
        best = top[0]
        detail = " | ".join(f"{name}={score:.4f}" for name, score in top)
        print(f"pred={best[0]}  {detail}")


if __name__ == "__main__":
    main()