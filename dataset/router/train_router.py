"""
train_router.py
---------------
训练 DeBERTa-v3 router 分类器（4 类: abstain/code/math/qa）。

在服务器(4090)上运行。默认加载 /root/autodl-tmp/lrr/ParamAgent/models/deberta-v3-base。

数据文件（由 prepare_router_data.py 生成, 本地跑好后拷到服务器同目录）:
    router_train.jsonl / router_val.jsonl / router_test.jsonl  字段: text, label, label_id

关键设计:
    - 4 类固定标签顺序 LABELS = ["abstain","code","math","qa"]
    - loss 加权: code(索引1) weight=3, 其它=1
    - 分层切分已在数据准备阶段完成, 此处直接读取

示例:
    python3 train_router.py \
        --model_path /root/autodl-tmp/lrr/ParamAgent/models/deberta-v3-base \
        --output_dir router_model_output \
        --batch_size 16 --lr 2e-5 --epochs 3
"""

import json
import os
import argparse
from collections import Counter

import numpy as np
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
)
from datasets import Dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

HERE = os.path.dirname(os.path.abspath(__file__))

# 固定标签顺序(与 prepare_router_data.py 的 LABEL_INDEX 对应)
LABELS = ["abstain", "code", "math", "qa"]
IDX = {lbl: i for i, lbl in enumerate(LABELS)}

# code 类别加权约 3 倍
CLASS_WEIGHTS = torch.tensor([1.0, 3.0, 1.0, 1.0])  # abstain, code, math, qa


def load_data(path):
    rows = []
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if line:
            d = json.loads(line)
            rows.append({"text": d["text"], "label": d["label_id"]})
    return rows


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", default="/root/autodl-tmp/lrr/ParamAgent/models/deberta-v3-base")
    p.add_argument("--output_dir", default=os.path.join(HERE, "router_model_output"))
    p.add_argument("--data_dir", default=HERE)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--eval_batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--epochs", type=float, default=3.0)
    p.add_argument("--warmup_ratio", type=float, default=0.1)
    p.add_argument("--max_length", type=int, default=512)
    p.add_argument("--grad_accum", type=int, default=1)
    p.add_argument("--fp16", action="store_true", help="4090 上建议开启")
    p.add_argument(
        "--eval_only",
        action="store_true",
        help="仅评估已保存的模型；模型和 tokenizer 从 output_dir 加载",
    )
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, preds)
    prec, rec, f1, _ = precision_recall_fscore_support(
        labels, preds, labels=list(range(len(LABELS))), average=None, zero_division=0
    )
    metrics = {"accuracy": acc}
    for i, name in enumerate(LABELS):
        metrics[f"{name}_precision"] = float(prec[i])
        metrics[f"{name}_recall"] = float(rec[i])
        metrics[f"{name}_f1"] = float(f1[i])
    # code 类加权 F1 重点看
    return metrics


def main():
    args = parse_args()

    train_rows = load_data(os.path.join(args.data_dir, "router_train.jsonl"))
    val_rows = load_data(os.path.join(args.data_dir, "router_val.jsonl"))
    test_rows = load_data(os.path.join(args.data_dir, "router_test.jsonl"))

    print("数据集规模:", {"train": len(train_rows), "val": len(val_rows), "test": len(test_rows)})
    print("train label 分布:", {LABELS[i]: c for i, c in Counter(r["label"] for r in train_rows).items()})
    print("val   label 分布:", {LABELS[i]: c for i, c in Counter(r["label"] for r in val_rows).items()})
    print("类别权重:", {LABELS[i]: float(CLASS_WEIGHTS[i]) for i in range(4)})

    # 评估模式直接读取训练产物；训练模式读取基础模型。
    model_path = args.output_dir if args.eval_only else args.model_path
    if args.eval_only and not os.path.isdir(model_path):
        raise FileNotFoundError(f"找不到已保存模型目录: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_path, num_labels=len(LABELS)
    )

    # GradScaler（TrainingArguments(fp16=True)）要求模型参数以 FP32 保存，
    # 由 Trainer 在前向/反向时自动进行混合精度。某些 checkpoint 或环境变量
    # 可能会让 from_pretrained 直接得到 FP16 参数，此时 unscale_gradients 会
    # 抛出 “Attempting to unscale FP16 gradients.”。
    # 强制恢复 FP32，避免与 Trainer 的 GradScaler 冲突。
    if args.fp16 and any(p.dtype == torch.float16 for p in model.parameters()):
        print("检测到模型参数为 FP16；转换为 FP32，由 Trainer 负责混合精度。")
        model.float()

    def tok(batch):
        return tokenizer(batch["text"], truncation=True, max_length=args.max_length)

    train_ds = Dataset.from_list(train_rows).map(tok, batched=True)
    val_ds = Dataset.from_list(val_rows).map(tok, batched=True)
    test_ds = Dataset.from_list(test_rows).map(tok, batched=True)

    # 加权 CrossEntropy
    model.config.label2id = IDX
    model.config.id2label = {i: name for name, i in IDX.items()}

    class WeightedTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            # **kwargs 吸收新版 Trainer 传入的额外参数(如 num_items_in_batch),
            # 避免 "unexpected keyword argument" 报错。
            labels = inputs.pop("labels")
            outputs = model(**inputs)
            logits = outputs.logits
            # 注意: 权重必须移到 logits 所在设备(而非构造时的 model.device,
            # 那时模型还在 CPU, Trainer 之后才搬到 GPU)。
            weight = CLASS_WEIGHTS.to(logits.device)
            loss = torch.nn.functional.cross_entropy(
                logits.view(-1, model.config.num_labels), labels.view(-1), weight=weight
            )
            return (loss, outputs) if return_outputs else loss

    # 版本敏感参数(evaluation_strategy/eval_strategy, warmup_ratio)在不同 transformers
    # 版本里命名不同。用 __dataclass_fields__ 探测实际支持的字段名, 只传存在的, 避免 TypeError。
    from transformers import TrainingArguments as TA
    ta_fields = set(TA.__dataclass_fields__)

    common_options = dict(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        fp16=args.fp16,
        logging_steps=50,
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        seed=args.seed,
        report_to=["none"],
    )
    # eval 策略字段名: 新版 eval_strategy, 旧版 evaluation_strategy
    eval_key = "eval_strategy" if "eval_strategy" in ta_fields else "evaluation_strategy"
    common_options[eval_key] = "epoch"
    # warmup: 优先 warmup_ratio(旧版也支持), 若该字段不存在则忽略(默认无warmup)
    if "warmup_ratio" in ta_fields:
        common_options["warmup_ratio"] = args.warmup_ratio

    train_args = TrainingArguments(**common_options)

    # 新版 transformers 移除了 Trainer.__init__ 的 tokenizer 参数(改为在 save_model 后手动
    # tokenizer.save_pretrained)。探测签名, 只在支持时才传。
    from transformers import Trainer as _Trainer
    import inspect as _inspect
    _sig = _inspect.signature(_Trainer.__init__).parameters
    trainer_kwargs = dict(
        model=model,
        args=train_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=compute_metrics,
    )
    if "tokenizer" in _sig:
        trainer_kwargs["tokenizer"] = tokenizer

    trainer = WeightedTrainer(**trainer_kwargs)

    if not args.eval_only:
        trainer.train()

    print("\n===== 验证集指标 =====")
    print(trainer.evaluate())

    print("\n===== 测试集指标(最终评估) =====")
    test_metrics = trainer.predict(test_ds)
    # PredictionOutput 同时包含 predictions、label_ids 和 metrics；compute_metrics
    # 只接受 (logits, labels)，不能把 metrics 字典直接传入。
    test_scores = compute_metrics((test_metrics.predictions, test_metrics.label_ids))
    print(test_scores)

    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    with open(os.path.join(args.output_dir, "label_order.json"), "w") as f:
        json.dump({"labels": LABELS, "class_weights": CLASS_WEIGHTS.tolist()}, f, ensure_ascii=False, indent=2)
    print(f"\n模型已保存到: {args.output_dir}")


if __name__ == "__main__":
    main()
