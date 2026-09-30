#!/usr/bin/env python3
"""
Generate multi-hop QA question decompositions using OpenAI-compatible API.

Input:  dataset/multihop/{hotpot,2w,mus}/*.jsonl
Output: {output_dir}/{dataset_name}_decomposition.json

- hotpot / 2w : call LLM to generate NL decomposition from question
- mus          : convert existing structured decomposition → NL format (no API call)

All outputs share the same schema: {"question": ..., "decomposition": ...}
"""

import json
import os
import time
import argparse
from typing import List, Dict, Optional
from tqdm import tqdm
from openai import OpenAI, APIStatusError, APIConnectionError, RateLimitError, APITimeoutError
from tenacity import retry, stop_after_attempt, wait_random_exponential, retry_if_exception_type


# ──────────────────── API configuration ────────────────────
API_BASE = os.getenv("OPENAI_API_BASE", "https://yunwu.ai/v1")
API_KEY  = os.getenv("OPENAI_API_KEY", "<REDACTED_HARDCODED_KEY>")
MODEL    = os.getenv("OPENAI_MODEL", "gemini-3.1-pro-preview")

_client = None

def get_client() -> OpenAI:
    global _client
    if _client is None:
        api_key = API_KEY or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("Set OPENAI_API_KEY before running this script.")
        _client = OpenAI(base_url=API_BASE, api_key=api_key)
    return _client


# ──────────────────── Prompts (same as original) ────────────────────

PRE_INSIGHT_FEWSHOT = """
<Example 1>
q: Anatoly Maltsev and Valentin Turchin were both from Russia, which of the two is known for his work as a mathematician?

### Question Parsing and Intent Extraction

**Intent:**
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
q: The Last Girl on Earth was the third concert tour by Barbadian recording artist Rihanna, the tour visited Europe, Asia, North America and Australia to support her fourth studio album, which was released on November 20, 2009, by Def Jam Recordings and SRP Records. What is the name of that fourth studio album?

### Question Parsing and Intent Extraction

**Intent:**
--------------------------------------------------------------------------------
🔍 **Key Components**
1. **Entity A**:
   - *The Last Girl on Earth* — Rihanna's third concert tour, associated with promoting a studio album

2. **Event**:
   - Release date **November 20, 2009** for the album

3. **Key Relationship**:
   - Identify Rihanna's **fourth studio album** released on that date and promoted by the tour

4. **Answer Type Expected**:
   - Album title (e.g., "Rated R")

5. **Reasoning Type**:
   - Factual retrieval from discography and tour association

6. **Required Background**:
   - Rihanna's discography and tour-album mapping
--------------------------------------------------------------------------------

🧠 **Inference Trace**
1. Locate Rihanna's albums around 2009.
2. Find the one released November 20, 2009.
3. Confirm it was promoted by "The Last Girl on Earth" tour.
4. Conclude the album is "Rated R".
--------------------------------------------------------------------------------
📝 **Disambiguation Note**
- Ignore redundant phrasing; focus on date & tour association.
"""

SYSTEM_PROMPT = (
    "You are an AI assistant for question parsing and intent extraction. "
    "Given a new question, produce a structured decomposition following "
    "the format shown in the examples."
)


# ──────────────────── LLM call (same as original OpenAI version) ────────────────────

TRANSIENT_ERRORS = (
    RateLimitError,
    APIStatusError,
    APIConnectionError,
    APITimeoutError,
)

@retry(
    retry=retry_if_exception_type(TRANSIENT_ERRORS),
    wait=wait_random_exponential(min=1, max=60),
    stop=stop_after_attempt(6),
)
def decompose_question(question: str) -> str:
    """Call the LLM to decompose `question` into structured NL format.

    Automatically retries on:
    - finish_reason == 'length' (truncated output → double max_tokens)
    - Malformed output (doesn't start with expected header)
    """
    user_prompt = (
        f"{PRE_INSIGHT_FEWSHOT}\n"
        f"q: {question}\n\n"
        "### Question Parsing and Intent Extraction"
    )

    # Progressive max_tokens: start at 2000, double on truncation
    max_tok = 2000
    for attempt in range(3):
        resp = get_client().chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.1,
            max_tokens=max_tok,
        )
        content = resp.choices[0].message.content.strip()
        finish = resp.choices[0].finish_reason

        # Check for truncation
        if finish == "length":
            max_tok *= 2
            continue

        # Check for malformed output (should start with expected header)
        if not (content.startswith("### Question Parsing") or content.startswith("**Intent:**")):
            continue

        # Check for model collapse (repetitive patterns)
        import re
        if len(re.findall(r"Could it be", content)) > 3:
            continue
        if len(re.findall(r"Let.s check:", content)) > 3:
            continue

        return content

    # All attempts exhausted — raise so caller skips this entry
    raise RuntimeError("Failed to generate valid decomposition after 3 attempts")


# ──────────────────── MuSiQue decomposition → NL format ────────────────────

def format_musique_decomposition(question: str, decomp: List[Dict]) -> str:
    """
    Convert MuSiQue's structured sub-question decomposition into the same
    NL format used by the LLM-generated decompositions.
    """
    sub_qs = decomp
    n = len(sub_qs)

    # Build Key Components from sub-questions
    key_components_lines = []
    for i, sq in enumerate(sub_qs):
        num = i + 1
        sq_q = sq["question"]
        sq_a = sq["answer"]
        # Replace placeholder #N with actual answer from previous step
        for j in range(i, 0, -1):
            sq_q = sq_q.replace(f"#{j}", sub_qs[j - 1]["answer"])
        key_components_lines.append(
            f"{num}. Sub-question {num}:\n"
            f"   - Q: {sq_q}\n"
            f"   - A: {sq_a}"
        )

    # Build Inference Trace from the chain of sub-questions
    inference_lines = []
    for i, sq in enumerate(sub_qs):
        sq_q = sq["question"]
        for j in range(i, 0, -1):
            sq_q = sq_q.replace(f"#{j}", sub_qs[j - 1]["answer"])
        inference_lines.append(
            f"{i + 1}. {sq_q} → {sq['answer']}"
        )

    # Determine reasoning type from hop count
    if n == 2:
        reasoning_type = "Two-hop factual retrieval and reasoning"
    elif n == 3:
        reasoning_type = "Three-hop factual retrieval and reasoning"
    elif n >= 4:
        reasoning_type = f"{n}-hop factual retrieval and reasoning"
    else:
        reasoning_type = "Factual retrieval"

    # Determine answer type from final answer
    final_answer = sub_qs[-1]["answer"]
    answer_type = "Entity / Concept name"

    return f"""**Intent:**
--------------------------------------------------------------------------------
🔍 Key Components
{chr(10).join(key_components_lines)}

{len(sub_qs) + 1}. Final Answer Expected:
   - {final_answer}
{len(sub_qs) + 2}. Reasoning Type:
   - {reasoning_type}
{len(sub_qs) + 3}. Answer Type Expected:
   - {answer_type}
--------------------------------------------------------------------------------
🧠 Inference Trace
{chr(10).join(inference_lines)}

{len(sub_qs) + 1}. Conclude the answer is: {final_answer}
--------------------------------------------------------------------------------
📝 Note
- This decomposition is derived from MuSiQue ground-truth sub-questions.
- Hop count: {n}
"""


# ──────────────────── Data loading ────────────────────

def load_jsonl(path: str) -> List[Dict]:
    """Load a JSONL file into a list of dicts."""
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


# ──────────────────── Dataset definitions ────────────────────

DATASETS = {
    "hotpot_full":  "dataset/multihop/meta/hotpot/full_train.jsonl",
    "hotpot_dis":   "dataset/multihop/meta/hotpot/dis_train.jsonl",
    "2w":           "dataset/multihop/meta/2w/2w-train.jsonl",
    "mus":          "dataset/multihop/meta/mus/mus_train.jsonl",
}


# ──────────────────── Main ────────────────────

def process_dataset(
    name: str,
    path: str,
    output_dir: str,
    use_api: bool,
    save_interval: int = 500,
) -> List[Dict]:
    """
    Process one dataset. Writes each result immediately (JSONL) so no data
    is lost on interruption. Automatically resumes by skipping already-processed
    questions found in the existing output file.

    Args:
        name: dataset key (e.g. 'hotpot_full', 'mus')
        path: path to the JSONL file
        output_dir: where to write the output JSON
        use_api: call LLM for decomposition (True for hotpot/2w, False for mus)
        save_interval: also save a JSON checkpoint every N entries (for backup)

    Returns:
        list of {question, decomposition} dicts
    """
    output_jsonl = os.path.join(output_dir, f"{name}_decomposition.jsonl")
    output_json  = os.path.join(output_dir, f"{name}_decomposition.json")

    print(f"\n{'=' * 60}")
    print(f"Dataset: {name}")
    print(f"Source: {path}")
    print(f"Method: {'LLM API' if use_api else 'MuSiQue ground-truth → NL'}")
    if use_api:
        print(f"Model: {MODEL}")
    print(f"Output (jsonl): {output_jsonl}")
    print(f"{'=' * 60}")

    # Load all entries
    entries = load_jsonl(path)
    total = len(entries)
    print(f"Total entries: {total}")

    # ── Resume: read already-processed questions ──
    processed_questions = set()
    if os.path.exists(output_jsonl):
        with open(output_jsonl, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    prev = json.loads(line)
                    processed_questions.add(prev["question"])
                except json.JSONDecodeError:
                    continue
        print(f"Resume: {len(processed_questions)} already processed, will skip")

    results = []      # all results for final JSON dump
    errors = 0
    skipped = 0

    pbar = tqdm(enumerate(entries), total=total, desc=f"[{name}]")
    for i, entry in pbar:
        question = entry["question"]

        # Skip already-processed entries
        if question in processed_questions:
            skipped += 1
            continue

        try:
            if use_api:
                decomposition = decompose_question(question)
            else:
                decomposition = format_musique_decomposition(
                    question, entry["question_decomposition"]
                )

            result = {"question": question, "decomposition": decomposition}
            results.append(result)

            # ── write immediately to JSONL ──
            with open(output_jsonl, "a", encoding="utf-8") as f:
                f.write(json.dumps(result, ensure_ascii=False) + "\n")

            # Periodic JSON checkpoint (backup, overwrites)
            if (i + 1) % save_interval == 0:
                # reload all from jsonl for a consistent checkpoint
                all_done = load_jsonl(output_jsonl)
                with open(output_json, "w", encoding="utf-8") as f:
                    json.dump(all_done, f, ensure_ascii=False, indent=2)
                pbar.set_postfix({"done": len(all_done), "err": errors, "skip": skipped})

            if use_api:
                time.sleep(0.01)

        except Exception as e:
            print(f"\n[Error] {name} entry {i}: {e}")
            errors += 1
            if use_api:
                time.sleep(1)
            continue

    # Final JSON dump (reload from jsonl for consistency)
    all_processed = load_jsonl(output_jsonl)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(all_processed, f, ensure_ascii=False, indent=2)

    print(f"Done: {len(all_processed)} saved to {output_jsonl} ({errors} errors, {skipped} skipped)")

    return all_processed


def main():
    parser = argparse.ArgumentParser(
        description="Generate multi-hop QA decompositions from dataset/multihop/ JSONL files"
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=list(DATASETS.keys()),
        choices=list(DATASETS.keys()) + ["all"],
        help="Which datasets to process (default: all)",
    )
    parser.add_argument(
        "--output_dir",
        default="dataset/multihop/train",
        help="Output directory for decomposition JSON files",
    )
    parser.add_argument(
        "--save_interval",
        type=int,
        default=500,
        help="Checkpoint save interval (default: 500)",
    )
    parser.add_argument(
        "--mus_use_api",
        action="store_true",
        help="Also use LLM API for MuSiQue instead of formatting existing decomposition",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit entries per dataset (for testing)",
    )

    args = parser.parse_args()

    if "all" in args.datasets:
        args.datasets = list(DATASETS.keys())

    os.makedirs(args.output_dir, exist_ok=True)

    # Print config
    print(f"{'=' * 60}")
    print(f"Multi-hop QA Decomposition Generation")
    print(f"{'=' * 60}")
    print(f"API Base: {API_BASE}")
    print(f"Model: {MODEL}")
    print(f"Datasets: {args.datasets}")
    print(f"Output dir: {args.output_dir}")
    print(f"MuSiQue strategy: {'LLM API' if args.mus_use_api else 'ground-truth → NL format'}")
    print(f"{'=' * 60}")

    all_results = []

    for ds_name in args.datasets:
        path = DATASETS[ds_name]
        if not os.path.exists(path):
            print(f"\n⚠️  Skipping {ds_name}: file not found at {path}")
            continue

        use_api = not (ds_name == "mus" and not args.mus_use_api)
        results = process_dataset(
            name=ds_name,
            path=path,
            output_dir=args.output_dir,
            use_api=use_api,
            save_interval=args.save_interval,
        )
        all_results.extend(results)

    # Save a combined file
    combined_path = os.path.join(args.output_dir, "multihop_decomposition_all.jsonl")
    with open(combined_path, "w", encoding="utf-8") as f:
        for r in all_results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"\n{'=' * 60}")
    print(f"All done! Combined output: {combined_path}")
    print(f"Total entries: {len(all_results)}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
