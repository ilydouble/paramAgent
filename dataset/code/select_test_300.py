"""从 dataset/code/meta/test 中按 difficulty 各随机选 300 道,
要求 question 未出现在 dataset/code/train/apps_original.json 中。
"""
import json
import os
import random
import re
from collections import Counter

ROOT = os.path.dirname(os.path.abspath(__file__))
TEST_DIR = os.path.join(ROOT, "meta", "test")
TRAIN_FILE = os.path.join(ROOT, "train", "apps_original.json")
OUT_FILE = os.path.join(ROOT, "test_selected_300.json")
SEED = 42
N_PER_DIFF = 300


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def main():
    # 1. 训练集中已有的 question(归一化后)用于去重
    with open(TRAIN_FILE) as f:
        train_data = json.load(f)
    train_qs = {normalize(item.get("question", "")) for item in train_data}
    print(f"train questions: {len(train_data)} (unique normalized: {len(train_qs)})")

    # 2. 收集测试集候选
    candidates = {d: [] for d in ("competition", "interview", "introductory")}
    skipped_dup = 0
    skipped_other = Counter()
    for d in sorted(os.listdir(TEST_DIR)):
        meta_path = os.path.join(TEST_DIR, d, "metadata.json")
        q_path = os.path.join(TEST_DIR, d, "question.txt")
        io_path = os.path.join(TEST_DIR, d, "input_output.json")
        if not os.path.exists(meta_path):
            continue
        meta = json.load(open(meta_path))
        diff = meta.get("difficulty")
        if diff not in candidates:
            skipped_other["bad_difficulty"] += 1
            continue
        q = open(q_path).read()
        if normalize(q) in train_qs:
            skipped_dup += 1
            continue
        io = json.load(open(io_path)) if os.path.exists(io_path) else None
        candidates[diff].append({
            "question": q.strip(),
            "difficulty": diff,
            "url": meta.get("url"),
            "meta_dir": d,
            "input_output": io,
        })
    print(f"candidates after dedup: { {k: len(v) for k, v in candidates.items()} }")
    print(f"skipped duplicates (in train): {skipped_dup}")

    # 3. 每类随机抽 300
    rng = random.Random(SEED)
    selected = []
    for diff in ("competition", "interview", "introductory"):
        pool = candidates[diff]
        assert len(pool) >= N_PER_DIFF, f"not enough {diff}: {len(pool)}"
        picked = rng.sample(pool, N_PER_DIFF)
        selected.extend(picked)
        print(f"{diff}: picked {len(picked)} / {len(pool)}")

    rng.shuffle(selected)

    # 4. 保存为 JSON 数组
    with open(OUT_FILE, "w") as f:
        json.dump(selected, f, ensure_ascii=False, indent=2)
    print(f"saved {len(selected)} items -> {OUT_FILE}")


if __name__ == "__main__":
    main()
