"""
prepare_router_data.py
----------------------
从 dataset/router/ 的 4 个源 jsonl 构建 router 分类训练数据（text + label），并做分层 train/val/test 切分。

本地运行（Mac），无需 torch/transformers 依赖。

路由字段与标签映射：
    code.jsonl      -> text=func_sign  , label=code
    math.jsonl      -> text=problem    , label=math
    qa.jsonl        -> text=question   , label=qa
    abs.jsonl       -> 0-599 用 func_sign, 600-1199 用 problem, 1200-1799 用 question -> label=abstain

切分策略（用户确认 ~80/10/10，code 保证 val/test 各 60）：
    code    : train 468 / val 60 / test 60
    math/qa/abstain: 每类 train 1440 / val 180 / test 180
    合计    : train 4788 / val 600 / test 600 = 5988
各类别内部用固定种子 Random(42) 分层抽样，保证精确、可复现、不被随机切分漏掉。
"""

import json
import os
import random

HERE = os.path.dirname(os.path.abspath(__file__))

LABEL_INDEX = {"abstain": 0, "code": 1, "math": 2, "qa": 3}  # 与训练脚本保持一致

# 切分规格: label -> (train, val, test)
SPLIT_SPEC = {
    "code":    (468, 60, 60),
    "math":    (1440, 180, 180),
    "qa":      (1440, 180, 180),
    "abstain": (1440, 180, 180),
}

SEED = 42


def read_all() -> dict:
    """返回 {label: [ (text, label), ... ]}，text 为路由字段原文。"""
    blocks = {"code": [], "math": [], "qa": [], "abstain": []}

    # code
    for line in open(os.path.join(HERE, "code.jsonl"), encoding="utf-8"):
        line = line.strip()
        if line:
            d = json.loads(line)
            blocks["code"].append((d["func_sign"], "code"))

    # math
    for line in open(os.path.join(HERE, "math.jsonl"), encoding="utf-8"):
        line = line.strip()
        if line:
            d = json.loads(line)
            blocks["math"].append((d["problem"], "math"))

    # qa
    for line in open(os.path.join(HERE, "qa.jsonl"), encoding="utf-8"):
        line = line.strip()
        if line:
            d = json.loads(line)
            blocks["qa"].append((d["question"], "qa"))

    # abs.jsonl: 按行段取不同字段
    abs_rows = [json.loads(l) for l in open(os.path.join(HERE, "abs.jsonl"), encoding="utf-8") if l.strip()]
    for i, d in enumerate(abs_rows):
        if i < 600:
            text = d["func_sign"]
        elif i < 1200:
            text = d["problem"]
        else:
            text = d["question"]
        blocks["abstain"].append((text, "abstain"))

    return blocks


def stratified_split(rows, rng):
    """把某一类别 rows 按 SPLIT_SPEC 切分为 (train, val, test)。"""
    label = rows[0][1]
    n_train, n_val, n_test = SPLIT_SPEC[label]
    if len(rows) != n_train + n_val + n_test:
        raise ValueError(
            f"label={label}: 期望 {n_train}+{n_val}+{n_test}={n_train+n_val+n_test}, "
            f"实际 {len(rows)}。请检查源文件行数。"
        )
    idx = list(range(len(rows)))
    rng.shuffle(idx)
    val_idx = idx[:n_val]
    test_idx = idx[n_val:n_val + n_test]
    val = [rows[i] for i in val_idx]
    test = [rows[i] for i in test_idx]
    train = [rows[i] for i in idx[n_val + n_test:]]
    return train, val, test


def main():
    blocks = read_all()

    # 校验每类数量是否符合规格
    for label, (t, v, te) in SPLIT_SPEC.items():
        n = len(blocks[label])
        need = t + v + te
        assert n == need, f"{label}: 有 {n} 条, 期望 {need}"

    rng = random.Random(SEED)
    train_all, val_all, test_all = [], [], []

    # 固定标签顺序，保证输出可读稳定
    for label in ["code", "math", "qa", "abstain"]:
        t, v, te = stratified_split(blocks[label], rng)
        train_all += t
        val_all += v
        test_all += te

    # 打散（分层后池内混排，避免同类连续堆积）
    rng.shuffle(train_all)
    rng.shuffle(val_all)
    rng.shuffle(test_all)

    for split_name, rows in [("train", train_all), ("val", val_all), ("test", test_all)]:
        out_path = os.path.join(HERE, f"router_{split_name}.jsonl")
        with open(out_path, "w", encoding="utf-8") as f:
            for text, label in rows:
                f.write(json.dumps({"text": text, "label": label, "label_id": LABEL_INDEX[label]},
                                   ensure_ascii=False) + "\n")
        print(f"[写] {split_name}: {len(rows)} 行 -> {out_path}")

    # 打印统计
    from collections import Counter
    print("\n===== 切分统计 =====")
    for split_name, rows in [("train", train_all), ("val", val_all), ("test", test_all)]:
        c = Counter(lbl for _, lbl in rows)
        total = len(rows)
        detail = ", ".join(f"{k}={v}" for k, v in c.items())
        print(f"{split_name:6s} {total:4d} | {detail}")
    print("\n完成。")


if __name__ == "__main__":
    main()