#!/usr/bin/env python3
"""Scan decomposition JSONL files for malformed entries."""
import json, re, sys

def check_file(filepath):
    with open(filepath) as f:
        lines = f.readlines()

    bad = []
    for i, line in enumerate(lines):
        if not line.strip():
            continue
        d = json.loads(line)
        decomp = d.get("decomposition", "")

        issues = []
        # 1. Must have all sections
        for sec in ["Key Components", "Inference Trace", "Disambiguation Note"]:
            if sec not in decomp:
                issues.append(f"missing {sec}")

        # 2. Must start with proper header
        if not (decomp.startswith("### Question Parsing") or decomp.startswith("**Intent:**")):
            issues.append("bad header")

        # 3. Repetitive patterns (model collapse)
        could_be = len(re.findall(r"Could it be", decomp))
        lets_check = len(re.findall(r"Let.s check:", decomp))
        if could_be > 3:
            issues.append(f"Could_it_be x{could_be}")
        if lets_check > 3:
            issues.append(f"Lets_check x{lets_check}")

        # 4. Very short (likely truncated)
        if len(decomp) < 300:
            issues.append(f"too short ({len(decomp)} chars)")

        if issues:
            bad.append((i + 1, issues, d["question"][:120], decomp[:80], decomp[-150:]))

    print(f"{filepath}: {len(bad)} problem entries out of {len(lines) - lines.count(chr(10))}")
    for row, issues, q, head, tail in bad:
        print(f"\n  Line {row}: {' | '.join(issues)}")
        print(f"  Q: {q}...")
        print(f"  Head: {head}")
        print(f"  Tail: ...{tail}")

if __name__ == "__main__":
    for fp in sys.argv[1:]:
        check_file(fp)
