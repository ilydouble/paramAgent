"""
eval_router.py

评估 4-way Router:
    abstain / code / math / qa

新增：
1. 原始 4-way argmax 评估
2. validation threshold sweep:
       P(abstain) >= tau -> abstain
       otherwise -> argmax(code, math, qa)
3. 输出：
       accuracy
       abstain precision / recall / F1
       coverage
       abstain rate
4. 输出各真实类别的 P(abstain) 分布
5. 支持 --threshold 在 test 上评估指定阈值
"""

import argparse
import json
import math
import os
import sys

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


LABELS = ["abstain", "code", "math", "qa"]
IDX = {lbl: i for i, lbl in enumerate(LABELS)}

TRAIN_PRIOR = {
    "abstain": 0.301,
    "code": 0.098,
    "math": 0.301,
    "qa": 0.301,
}

CLASS_WEIGHT = {
    "abstain": 1.0,
    "code": 3.0,
    "math": 1.0,
    "qa": 1.0,
}

DEPLOY_PRIOR = {
    "abstain": 0.59,
    "code": 0.023,
    "math": 0.12,
    "qa": 0.27,
}


def parse_args():
    p = argparse.ArgumentParser(
        description="Evaluate router with abstain-threshold sweep"
    )

    p.add_argument(
        "--model_path",
        default=os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "router_model_output",
        ),
    )

    p.add_argument(
        "--data",
        default=None,
        help="JSONL file. Default: router_{split}.jsonl",
    )

    p.add_argument(
        "--data_dir",
        default=os.path.dirname(os.path.abspath(__file__)),
    )

    p.add_argument(
        "--split",
        default="test",
        choices=["train", "val", "test"],
    )

    p.add_argument("--batch_size", type=int, default=32)

    p.add_argument("--max_length", type=int, default=512)

    p.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )

    # 如果指定，就只评估这个 threshold
    p.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Evaluate one abstain threshold, e.g. --threshold 0.35",
    )

    # sweep 范围
    p.add_argument("--threshold_start", type=float, default=0.05)
    p.add_argument("--threshold_end", type=float, default=0.95)
    p.add_argument("--threshold_step", type=float, default=0.05)

    p.add_argument(
        "--deploy_prior",
        default=",".join(
            f"{k}:{v}" for k, v in DEPLOY_PRIOR.items()
        ),
    )

    p.add_argument(
        "--calibrate",
        action="store_true",
        help="额外报告 prior correction",
    )

    p.add_argument(
        "--no_weight",
        action="store_true",
        help="prior correction 时不考虑训练 class weight",
    )

    p.add_argument(
        "--dump_errors",
        default=None,
    )

    return p.parse_args()


def parse_prior(s):
    out = {}

    for tok in s.replace(";", ",").split(","):
        tok = tok.strip()

        if not tok:
            continue

        k, _, v = tok.partition(":")

        if k.strip() in LABELS:
            out[k.strip()] = float(v)

    for lbl in LABELS:
        out.setdefault(lbl, 1.0 / len(LABELS))

    return out


def load_rows(path):
    rows = []

    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            d = json.loads(line)

            label_id = d.get(
                "label_id",
                IDX.get(d.get("label")),
            )

            if label_id is None:
                raise ValueError(
                    f"Missing label_id/label: {d}"
                )

            rows.append(
                {
                    "text": d["text"],
                    "label_id": int(label_id),
                }
            )

    return rows


def effective_prior(consider_weight=True):
    if not consider_weight:
        return dict(TRAIN_PRIOR)

    z = sum(
        TRAIN_PRIOR[c] * CLASS_WEIGHT[c]
        for c in LABELS
    )

    return {
        c: TRAIN_PRIOR[c] * CLASS_WEIGHT[c] / z
        for c in LABELS
    }


def softmax(logits):
    logits = logits.astype(np.float64)

    logits = (
        logits
        - logits.max(axis=1, keepdims=True)
    )

    exp_logits = np.exp(logits)

    return exp_logits / exp_logits.sum(
        axis=1,
        keepdims=True,
    )


def calibrate_logits(
    logits,
    prior_used,
    true_prior,
):
    delta = np.array(
        [
            -math.log(max(prior_used[c], 1e-12))
            + math.log(max(true_prior[c], 1e-12))
            for c in LABELS
        ],
        dtype=np.float64,
    )

    adj = logits.astype(np.float64) + delta[None, :]

    return softmax(adj)


def confusion(
    y_true,
    y_pred,
    n=len(LABELS),
    weights=None,
):
    C = np.zeros(
        (n, n),
        dtype=np.float64,
    )

    if weights is None:
        weights = np.ones(len(y_true))

    for t, p, w in zip(
        y_true,
        y_pred,
        weights,
    ):
        C[t, p] += w

    return C


def prf_from_confusion(C):
    tp = np.diag(C)

    col = C.sum(axis=0)
    row = C.sum(axis=1)

    with np.errstate(
        divide="ignore",
        invalid="ignore",
    ):
        precision = np.where(
            col > 0,
            tp / col,
            0.0,
        )

        recall = np.where(
            row > 0,
            tp / row,
            0.0,
        )

        f1 = np.where(
            (precision + recall) > 0,
            2
            * precision
            * recall
            / (precision + recall),
            0.0,
        )

    total = C.sum()

    accuracy = (
        tp.sum() / total
        if total > 0
        else 0.0
    )

    return precision, recall, f1, accuracy


def print_confusion(
    C,
    title,
    normalized=False,
):
    print(f"\n--- {title} ---")

    if normalized:
        row_sum = C.sum(axis=1, keepdims=True)

        C_show = np.divide(
            C,
            row_sum,
            out=np.zeros_like(C),
            where=row_sum > 0,
        )
    else:
        C_show = C

    header = (
        "真\\预  "
        + "".join(
            f"{l:>12s}"
            for l in LABELS
        )
        + f"{'  合计':>10s}"
    )

    print(header)

    for i, lbl in enumerate(LABELS):

        if normalized:
            cells = "".join(
                f"{C_show[i, j]:12.3f}"
                for j in range(len(LABELS))
            )
        else:
            cells = "".join(
                f"{int(C_show[i, j]):12d}"
                for j in range(len(LABELS))
            )

        print(
            f"{lbl:>6s} "
            + cells
            + f"{C[i].sum():10.1f}"
        )

    print(
        "合计   "
        + "".join(
            f"{C[:, j].sum():12.1f}"
            for j in range(len(LABELS))
        )
        + f"{C.sum():10.1f}"
    )


def print_metrics(C, title):
    precision, recall, f1, accuracy = (
        prf_from_confusion(C)
    )

    print(f"\n--- {title} ---")

    print(
        f"{'类别':>10s}"
        f"{'precision':>12s}"
        f"{'recall':>12s}"
        f"{'f1':>12s}"
        f"{'support':>12s}"
    )

    for i, lbl in enumerate(LABELS):

        support = C[i].sum()

        print(
            f"{lbl:>10s}"
            f"{precision[i]:12.4f}"
            f"{recall[i]:12.4f}"
            f"{f1[i]:12.4f}"
            f"{support:12.1f}"
        )

    print(
        f"{'accuracy':>10s}"
        f"{'':>12s}"
        f"{'':>12s}"
        f"{accuracy:12.4f}"
        f"{C.sum():12.1f}"
    )

    return precision, recall, f1, accuracy


# ============================================================
# 新增：Utility-aware abstain threshold
# ============================================================

def threshold_predict(
    probs,
    threshold,
):
    """
    P(abstain) >= threshold:
        abstain

    otherwise:
        only compare code/math/qa
    """

    p_abstain = probs[:, IDX["abstain"]]

    domain_indices = [
        IDX["code"],
        IDX["math"],
        IDX["qa"],
    ]

    domain_probs = probs[:, domain_indices]

    domain_local_pred = np.argmax(
        domain_probs,
        axis=1,
    )

    domain_pred = np.array(
        [
            domain_indices[i]
            for i in domain_local_pred
        ],
        dtype=int,
    )

    pred = domain_pred.copy()

    abstain_mask = (
        p_abstain >= threshold
    )

    pred[abstain_mask] = IDX["abstain"]

    return pred


def evaluate_threshold(
    probs,
    y_true,
    threshold,
):
    y_pred = threshold_predict(
        probs,
        threshold,
    )

    C = confusion(
        y_true,
        y_pred,
    )

    precision, recall, f1, accuracy = (
        prf_from_confusion(C)
    )

    abstain_idx = IDX["abstain"]

    coverage = float(
        np.mean(
            y_pred != abstain_idx
        )
    )

    abstain_rate = 1.0 - coverage

    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy),
        "abstain_precision": float(
            precision[abstain_idx]
        ),
        "abstain_recall": float(
            recall[abstain_idx]
        ),
        "abstain_f1": float(
            f1[abstain_idx]
        ),
        "coverage": coverage,
        "abstain_rate": abstain_rate,
        "y_pred": y_pred,
        "C": C,
    }


def threshold_sweep(
    probs,
    y_true,
    start,
    end,
    step,
):
    thresholds = np.arange(
        start,
        end + step * 0.5,
        step,
    )

    results = []

    for tau in thresholds:
        results.append(
            evaluate_threshold(
                probs,
                y_true,
                float(tau),
            )
        )

    return results


def print_threshold_sweep(results):
    print("\n")
    print("=" * 110)
    print("ABSTAIN THRESHOLD SWEEP")
    print("=" * 110)

    print(
        f"{'tau':>7s}"
        f"{'accuracy':>12s}"
        f"{'abs_P':>12s}"
        f"{'abs_R':>12s}"
        f"{'abs_F1':>12s}"
        f"{'coverage':>12s}"
        f"{'abs_rate':>12s}"
    )

    for r in results:

        print(
            f"{r['threshold']:7.2f}"
            f"{r['accuracy']:12.4f}"
            f"{r['abstain_precision']:12.4f}"
            f"{r['abstain_recall']:12.4f}"
            f"{r['abstain_f1']:12.4f}"
            f"{r['coverage']:12.4f}"
            f"{r['abstain_rate']:12.4f}"
        )

    print("=" * 110)

    # 按 abstain F1 排序
    best_f1 = max(
        results,
        key=lambda x: x["abstain_f1"],
    )

    # 按 accuracy 排序
    best_acc = max(
        results,
        key=lambda x: x["accuracy"],
    )

    print(
        "\nBest by abstain F1:"
        f" tau={best_f1['threshold']:.2f}"
        f", F1={best_f1['abstain_f1']:.4f}"
        f", recall={best_f1['abstain_recall']:.4f}"
        f", precision={best_f1['abstain_precision']:.4f}"
        f", coverage={best_f1['coverage']:.4f}"
        f", accuracy={best_f1['accuracy']:.4f}"
    )

    print(
        "Best by accuracy:"
        f" tau={best_acc['threshold']:.2f}"
        f", accuracy={best_acc['accuracy']:.4f}"
        f", abstain_F1={best_acc['abstain_f1']:.4f}"
        f", recall={best_acc['abstain_recall']:.4f}"
        f", coverage={best_acc['coverage']:.4f}"
    )


def print_abstain_distribution(
    probs,
    y_true,
):
    p_abstain = probs[
        :,
        IDX["abstain"],
    ]

    print("\n")
    print("=" * 80)
    print("P(abstain) DISTRIBUTION")
    print("=" * 80)

    print(
        f"{'class':>12s}"
        f"{'n':>8s}"
        f"{'mean':>12s}"
        f"{'median':>12s}"
        f"{'p10':>12s}"
        f"{'p25':>12s}"
        f"{'p75':>12s}"
        f"{'p90':>12s}"
    )

    for label in LABELS:

        mask = (
            y_true
            == IDX[label]
        )

        values = p_abstain[mask]

        if len(values) == 0:
            continue

        print(
            f"{label:>12s}"
            f"{len(values):8d}"
            f"{values.mean():12.4f}"
            f"{np.median(values):12.4f}"
            f"{np.percentile(values, 10):12.4f}"
            f"{np.percentile(values, 25):12.4f}"
            f"{np.percentile(values, 75):12.4f}"
            f"{np.percentile(values, 90):12.4f}"
        )

    print("=" * 80)


def print_threshold_confusion(
    result,
    title,
):
    print_confusion(
        result["C"],
        title,
    )

    print_confusion(
        result["C"],
        title + "（行归一化）",
        normalized=True,
    )

    precision, recall, f1, accuracy = (
        prf_from_confusion(
            result["C"]
        )
    )

    print(
        f"\nthreshold = "
        f"{result['threshold']:.4f}"
    )

    print(
        f"accuracy       = {accuracy:.4f}"
    )

    print(
        f"abstain P      = "
        f"{precision[IDX['abstain']]:.4f}"
    )

    print(
        f"abstain R      = "
        f"{recall[IDX['abstain']]:.4f}"
    )

    print(
        f"abstain F1     = "
        f"{f1[IDX['abstain']]:.4f}"
    )

    print(
        f"coverage       = "
        f"{result['coverage']:.4f}"
    )

    print(
        f"abstain rate   = "
        f"{result['abstain_rate']:.4f}"
    )


# ============================================================
# Main
# ============================================================

def main():

    args = parse_args()

    deploy_prior = parse_prior(
        args.deploy_prior
    )

    prior_used = effective_prior(
        consider_weight=not args.no_weight
    )

    data_path = (
        args.data
        or os.path.join(
            args.data_dir,
            f"router_{args.split}.jsonl",
        )
    )

    if not os.path.isfile(data_path):
        sys.exit(
            f"找不到数据文件: {data_path}"
        )

    if not os.path.isdir(args.model_path):
        sys.exit(
            f"找不到模型目录: "
            f"{args.model_path}"
        )

    rows = load_rows(data_path)

    y_true = np.array(
        [
            r["label_id"]
            for r in rows
        ],
        dtype=int,
    )

    print(
        f"数据: {data_path} "
        f"({len(rows)} 条)"
    )

    print(
        f"设备: {args.device}"
        f" | 模型: {args.model_path}"
    )

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path
    )

    model = (
        AutoModelForSequenceClassification
        .from_pretrained(args.model_path)
        .to(args.device)
        .eval()
    )

    # --------------------------------------------------------
    # inference
    # --------------------------------------------------------

    all_logits = []

    with torch.no_grad():

        for i in range(
            0,
            len(rows),
            args.batch_size,
        ):

            batch = [
                r["text"]
                for r in rows[
                    i : i + args.batch_size
                ]
            ]

            enc = tokenizer(
                batch,
                truncation=True,
                max_length=args.max_length,
                padding=True,
                return_tensors="pt",
            ).to(args.device)

            logits = (
                model(**enc)
                .logits
                .float()
                .cpu()
                .numpy()
            )

            all_logits.append(logits)

    logits = np.concatenate(
        all_logits,
        axis=0,
    )

    probs = softmax(logits)

    # ========================================================
    # 1. 原始 4-way argmax
    # ========================================================

    y_pred = np.argmax(
        logits,
        axis=1,
    )

    C_raw = confusion(
        y_true,
        y_pred,
    )

    print_confusion(
        C_raw,
        "原始 4-way 混淆矩阵（计数）",
    )

    print_confusion(
        C_raw,
        "原始 4-way 混淆矩阵（行归一化）",
        normalized=True,
    )

    print_metrics(
        C_raw,
        "原始 4-way 指标",
    )

    # ========================================================
    # 2. 分布诊断
    # ========================================================

    true_prior = {
        lbl: float(
            (y_true == i).sum()
        ) / len(y_true)
        for i, lbl in enumerate(LABELS)
    }

    pred_prior = {
        lbl: float(
            (y_pred == i).sum()
        ) / len(y_pred)
        for i, lbl in enumerate(LABELS)
    }

    print("\n--- 分布诊断 ---")

    print(
        f"{'类别':>10s}"
        f"{'真实':>14s}"
        f"{'模型预测':>14s}"
        f"{'部署目标':>14s}"
    )

    for i, lbl in enumerate(LABELS):

        print(
            f"{lbl:>10s}"
            f"{true_prior[lbl]:14.3f}"
            f"{pred_prior[lbl]:14.3f}"
            f"{deploy_prior[lbl]:14.3f}"
        )

    # ========================================================
    # 3. P(abstain) distribution
    # ========================================================

    print_abstain_distribution(
        probs,
        y_true,
    )

    # ========================================================
    # 4. Threshold sweep
    # ========================================================

    if args.threshold is not None:

        result = evaluate_threshold(
            probs,
            y_true,
            args.threshold,
        )

        print_threshold_confusion(
            result,
            "指定 Threshold 的 Router 结果",
        )

    elif args.split == "val":

        results = threshold_sweep(
            probs,
            y_true,
            args.threshold_start,
            args.threshold_end,
            args.threshold_step,
        )

        print_threshold_sweep(
            results
        )

    # ========================================================
    # 5. 部署先验加权
    # ========================================================

    test_prior = true_prior

    weights = np.array(
        [
            deploy_prior[
                LABELS[t]
            ]
            / max(
                test_prior[
                    LABELS[t]
                ],
                1e-12,
            )
            for t in y_true
        ]
    )

    C_w = confusion(
        y_true,
        y_pred,
        weights=weights,
    )

    print_metrics(
        C_w,
        "原始 argmax：按部署先验加权",
    )

    # ========================================================
    # 6. Prior correction
    # ========================================================

    if args.calibrate:

        probs_cal = calibrate_logits(
            logits,
            prior_used,
            deploy_prior,
        )

        y_pred_cal = np.argmax(
            probs_cal,
            axis=1,
        )

        C_cal = confusion(
            y_true,
            y_pred_cal,
            weights=weights,
        )

        print("\n")
        print("=" * 80)
        print("PRIOR CORRECTION")
        print("=" * 80)

        print(
            "prior_used = "
            + str(
                {
                    k: round(
                        prior_used[k],
                        4,
                    )
                    for k in LABELS
                }
            )
        )

        print(
            "true_prior = "
            + str(
                {
                    k: round(
                        deploy_prior[k],
                        4,
                    )
                    for k in LABELS
                }
            )
        )

        precision0, recall0, f10, acc0 = (
            prf_from_confusion(C_w)
        )

        precision1, recall1, f11, acc1 = (
            prf_from_confusion(C_cal)
        )

        print(
            f"\n{'类别':>10s}"
            f"{'P_before':>12s}"
            f"{'P_after':>12s}"
            f"{'R_before':>12s}"
            f"{'R_after':>12s}"
            f"{'F1_before':>12s}"
            f"{'F1_after':>12s}"
        )

        for i, lbl in enumerate(LABELS):

            print(
                f"{lbl:>10s}"
                f"{precision0[i]:12.4f}"
                f"{precision1[i]:12.4f}"
                f"{recall0[i]:12.4f}"
                f"{recall1[i]:12.4f}"
                f"{f10[i]:12.4f}"
                f"{f11[i]:12.4f}"
            )

        print(
            f"\naccuracy:"
            f" {acc0:.4f}"
            f" -> {acc1:.4f}"
            f" ({acc1 - acc0:+.4f})"
        )

    # ========================================================
    # 7. Error dump
    # ========================================================

    if args.dump_errors:

        with open(
            args.dump_errors,
            "w",
            encoding="utf-8",
        ) as f:

            n_err = 0

            for i, r in enumerate(rows):

                if y_pred[i] == y_true[i]:
                    continue

                n_err += 1

                p = probs[i]

                f.write(
                    json.dumps(
                        {
                            "true": LABELS[
                                y_true[i]
                            ],
                            "pred": LABELS[
                                y_pred[i]
                            ],
                            "pred_conf": round(
                                float(
                                    p[
                                        y_pred[i]
                                    ]
                                ),
                                4,
                            ),
                            "p_abstain": round(
                                float(
                                    p[
                                        IDX[
                                            "abstain"
                                        ]
                                    ]
                                ),
                                4,
                            ),
                            "p_code": round(
                                float(
                                    p[
                                        IDX["code"]
                                    ]
                                ),
                                4,
                            ),
                            "p_math": round(
                                float(
                                    p[
                                        IDX["math"]
                                    ]
                                ),
                                4,
                            ),
                            "p_qa": round(
                                float(
                                    p[
                                        IDX["qa"]
                                    ]
                                ),
                                4,
                            ),
                            "text": r["text"][
                                :400
                            ],
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )

        print(
            f"\n误分类样本已写出: "
            f"{args.dump_errors}"
            f" ({n_err} 条)"
        )

    print("\n完成。")


if __name__ == "__main__":
    main()