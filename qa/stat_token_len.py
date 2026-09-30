#!/usr/bin/env python
"""统计QA训练数据的token长度分布，用于确定max_seq_len。"""

import json
import statistics
from pathlib import Path

from transformers import AutoProcessor

SYSTEM_PROMPT = (
    "You will be given a question. Use your knowledge to extract the key point, "
    "the underlying intent, and possible inference patterns needed to answer the question."
)

PRE_INSIGHT_FEWSHOT = """
<Example 1>
q: Anatoly Maltsev and Valentin Turchin were both from Russia, which of the two is known for his work as a mathematician?

Question Parsing and Intent Extraction
Intent:
--------------------------------------------------------------------------------
🔍 Key Components
1. Entity A:
- **Anatoly Maltsev** — mathematician and logician known for contributions in mathematical logic and abstract algebra
2. Entity B:
- **Valentin Turchin** — computer scientist and philosopher known for work in cybernetics and philosophy of science
3. Implied Relationship:
- Comparative inquiry: which individual is more closely associated with the domain of mathematics
4. Answer Type Expected:
- Person name (e.g., "Anatoly Maltsev")
5. Reasoning Type:
- Comparative factual reasoning
6. Required Background:
- Biographical knowledge or retrieved professional profiles
--------------------------------------------------------------------------------
🧠 Inference Trace
1. Retrieve factual data about Maltsev and Turchin's academic domains.
2. Classify Maltsev as a mathematician based on core contributions to mathematical logic.
3. Classify Turchin as mainly working in cybernetics and philosophy.
4. Eliminate Turchin as primary mathematician.
5. Conclude Maltsev is the individual known for mathematics.
--------------------------------------------------------------------------------
📝 Disambiguation Note
- Nationality (Russia) does not help differentiate them.

<Example 2>
The Last Girl on Earth was the third concert tour by Barbadian recording artist Rihanna, the tour visited Europe, Asia, North America and Australia to support her fourth studio album, which 2009 , and fourth studio album by Barbadian singer Rihanna, and released on November 20, 2009 by Def Jam Recordings and SRP Records?

### Question Parsing and Intent Extraction

**Intent:**
--------------------------------------------------------------------------------
🔍 **Key Components**
1. **Entity A**:
   - *The Last Girl on Earth* — Rihanna's third concert tour, associated with promoting a studio album

2. **Entity B**:
   - *Fourth studio album* by Rihanna — referenced multiple times, released in 2009

3. **Key Relationship / Constraint**:
   - Identify the name of Rihanna's **fourth studio album**, which was released on **November 20, 2009**, and **supported by** her third concert tour, *The Last Girl on Earth*

4. **Answer Type Expected**:
   - Album title (e.g., *Rated R*)

5. **Reasoning Type**:
   - Factual entity retrieval based on event-album association and release date

6. **Required Background**:
   - Rihanna's discography: album release dates and which albums were promoted during which concert tours

--------------------------------------------------------------------------------

🧠 **Inference Trace**
- Determine the name of Rihanna's **fourth studio album**
- Confirm that this album was released on **November 20, 2009**
- Verify that this album was the basis for the **"The Last Girl on Earth"** tour
- Conclude that the album is **Rated R**

--------------------------------------------------------------------------------

📝 **Disambiguation Note**
- Although the question includes fragmented/redundant phrasing, the focus is clear: determine the album that matches both the **release date** and **tour association**
"""


def main():
    data_path = "dataset/multihop/train/qa.jsonl"
    model_name = "Qwen/Qwen3.5-2B"

    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
    tokenizer = processor.tokenizer

    lengths = []
    with open(data_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            question = item.get("question", "")
            decomposition = item.get("decomposition", "")

            user_content = (
                f"Here are some examples:{PRE_INSIGHT_FEWSHOT}\n\n"
                f"[Question]: {question}"
            )
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ]
            prompt = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            answer = decomposition.strip() + tokenizer.eos_token

            prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
            answer_ids = tokenizer(answer, add_special_tokens=False)["input_ids"]
            total_len = len(prompt_ids) + len(answer_ids)
            lengths.append(total_len)

    lengths.sort()
    n = len(lengths)
    print(f"样本数: {n}")
    print(f"最小 token 长度: {lengths[0]}")
    print(f"最大 token 长度: {lengths[-1]}")
    print(f"平均 token 长度: {statistics.mean(lengths):.1f}")
    print(f"中位数 token 长度: {statistics.median(lengths):.1f}")
    print(f"标准差: {statistics.stdev(lengths):.1f}")
    for p in [50, 75, 90, 95, 99]:
        idx = int(n * p / 100)
        print(f"P{p}: {lengths[min(idx, n - 1)]}")

    truncated = sum(1 for l in lengths if l > 2048)
    print(f"\n超过 2048 的样本数: {truncated} ({truncated / n * 100:.1f}%)")
    for threshold in [1024, 2048, 3072, 4096, 5120, 6144, 8192]:
        truncated = sum(1 for l in lengths if l > threshold)
        print(f"  max_seq_len={threshold:>5d} -> 截断 {truncated:>5d} 条 ({truncated / n * 100:.1f}%)")


if __name__ == "__main__":
    main()
