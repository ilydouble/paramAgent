"""
Match APP_code_datasets.json items to original APPS problems (from HuggingFace)
and extract test cases (input_output).

The matching is done by computing a hash/similarity on the question text.
"""
import json
import hashlib
import re
import sys
from typing import Optional

from datasets import load_dataset


def normalize_question(text: str) -> str:
    """Normalize question text for matching: lowercase, collapse whitespace, strip."""
    text = text.lower()
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    return text


def question_fingerprint(text: str) -> str:
    """Create a hash-based fingerprint of the first N chars for quick matching."""
    # Use first 200 chars after normalization as the fingerprint
    normalized = normalize_question(text)
    prefix = normalized[:200]
    return hashlib.md5(prefix.encode("utf-8")).hexdigest()


def main():
    # --- Load local dataset ---
    local_path = "benchmarks/APP_code_datasets.json"
    print(f"Loading {local_path}...")
    with open(local_path, encoding="utf-8") as f:
        local_data = json.load(f)
    print(f"  {len(local_data)} items")

    # Build fingerprint index for local data
    local_by_fp = {}
    for i, item in enumerate(local_data):
        q = item.get("question", "")
        fp = question_fingerprint(q)
        if fp not in local_by_fp:
            local_by_fp[fp] = []
        local_by_fp[fp].append(i)

    print(f"  {len(local_by_fp)} unique fingerprints")

    # --- Load APPS from HuggingFace ---
    print("Loading APPS from HuggingFace (codeparrot/apps, 'all')...")
    apps = load_dataset("codeparrot/apps", "all", split="train", streaming=True)

    # Build fingerprint index for APPS
    apps_by_fp = {}
    apps_data = {}  # fingerprint -> item
    for item in apps:
        q = item.get("question", "")
        fp = question_fingerprint(q)
        if fp not in apps_by_fp:
            apps_by_fp[fp] = []
        apps_by_fp[fp].append(item["problem_id"])
        apps_data[fp] = item  # store last one (should be unique)

    print(f"  {len(apps_by_fp)} unique fingerprints in APPS")

    # --- Match ---
    matched = 0
    unmatched = 0
    multi_match = 0

    for fp, local_indices in local_by_fp.items():
        if fp in apps_by_fp:
            if len(apps_by_fp[fp]) == 1:
                matched += 1
            else:
                multi_match += 1
                matched += 1  # still count as matched
        else:
            unmatched += 1

    print(f"\nMatching results:")
    print(f"  Matched (exact fingerprint): {matched}")
    print(f"  Multi-match (same fingerprint, multiple APPS): {multi_match}")
    print(f"  Unmatched: {unmatched}")

    # --- For unmatched, try more lenient matching ---
    if unmatched > 0:
        print(f"\nTrying lenient matching for {unmatched} unmatched items...")
        # Use first 100 chars instead of 200
        lenient_matched = 0
        for fp, local_indices in local_by_fp.items():
            if fp not in apps_by_fp:
                # Try with shorter prefix
                for local_idx in local_indices:
                    q = local_data[local_idx].get("question", "")
                    norm = normalize_question(q)
                    short_prefix = norm[:100]
                    short_fp = hashlib.md5(short_prefix.encode("utf-8")).hexdigest()
                    if short_fp in apps_by_fp:
                        lenient_matched += 1
                        # Add to apps_by_fp for later use
                        apps_by_fp[fp] = apps_by_fp[short_fp]
                        break

        print(f"  Additional lenient matches: {lenient_matched}")
        matched += lenient_matched
        unmatched -= lenient_matched

    print(f"\nFinal matching:")
    print(f"  Matched: {matched}")
    print(f"  Unmatched: {unmatched}")
    print(f"  Match rate: {100 * matched / len(local_data):.1f}%")

    # --- Build enriched dataset with test cases ---
    # For matched items, add input_output field
    # We need to handle the case where multiple local items map to same APPS problem
    # Build reverse mapping: local_idx -> APPS item
    local_to_apps = {}
    for fp, local_indices in local_by_fp.items():
        apps_item = apps_data.get(fp)
        if apps_item is None:
            # Try short fingerprint
            for local_idx in local_indices:
                q = local_data[local_idx].get("question", "")
                norm = normalize_question(q)
                short_fp = hashlib.md5(norm[:100].encode("utf-8")).hexdigest()
                apps_item = apps_data.get(short_fp)
                if apps_item:
                    break
        if apps_item:
            for local_idx in local_indices:
                local_to_apps[local_idx] = apps_item

    print(f"\nBuilt mapping for {len(local_to_apps)} local items")

    # Save enriched dataset
    enriched = []
    for i, item in enumerate(local_data):
        new_item = dict(item)
        apps_match = local_to_apps.get(i)
        if apps_match:
            new_item["problem_id"] = apps_match["problem_id"]
            new_item["url"] = apps_match.get("url", "")
            # input_output is a JSON string in the HF dataset, parse it
            raw_io = apps_match.get("input_output", "{}")
            if isinstance(raw_io, str):
                try:
                    io_data = json.loads(raw_io)
                except json.JSONDecodeError:
                    io_data = {"inputs": [], "outputs": []}
            else:
                io_data = raw_io
            new_item["input_output"] = io_data
            new_item["starter_code"] = apps_match.get("starter_code", "")
            raw_solutions = apps_match.get("solutions", "[]")
            if isinstance(raw_solutions, str):
                try:
                    solutions_data = json.loads(raw_solutions)
                except json.JSONDecodeError:
                    solutions_data = []
            else:
                solutions_data = raw_solutions
            new_item["solutions"] = solutions_data
        else:
            new_item["problem_id"] = None
            new_item["url"] = ""
            new_item["input_output"] = {"inputs": [], "outputs": []}
            new_item["starter_code"] = ""
            new_item["solutions"] = []

        enriched.append(new_item)

    # Save
    output_path = "benchmarks/APP_code_datasets_with_tests.json"
    print(f"\nSaving enriched dataset to {output_path}...")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(enriched, f, ensure_ascii=False, indent=2)

    # Stats
    with_tests = sum(1 for item in enriched if isinstance(item.get("input_output"), dict) and item["input_output"].get("inputs"))
    print(f"Items with test cases: {with_tests}/{len(enriched)}")

    # Show a sample
    for item in enriched[:3]:
        io = item.get("input_output", {})
        print(f"\n--- Sample: {item.get('func_sign','')[:80]} ---")
        print(f"  problem_id: {item.get('problem_id')}")
        print(f"  url: {item.get('url','')}")
        print(f"  test inputs: {len(io.get('inputs',[]))}")
        print(f"  test outputs: {len(io.get('outputs',[]))}")
        if io.get("inputs"):
            print(f"  first input (200 chars): {io['inputs'][0][:200]}")


if __name__ == "__main__":
    main()
