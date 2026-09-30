#!/usr/bin/env python
"""
Clean app.json: remove items whose test data is in stdin multi-line format
that is incompatible with function-call evaluation.

An item is incompatible when:
  - inputs[0] is a list of strings with MORE THAN ONE element
  - These represent stdin lines for a complete program, not function arguments

Usage:
  python code/clean_app_data.py
"""

import json
import sys
import shutil
from pathlib import Path

def is_compatible(item: dict) -> bool:
    """Return True if the item's test data is compatible with function-call eval."""
    inputs = item.get("inputs", [])
    if not inputs:
        return True  # no test data — handled by LLM fallback

    inp0 = inputs[0]

    # Plain string → stdin-style, compatible
    if isinstance(inp0, str):
        return True

    # List of mixed types (e.g. [4, [1, 2]]) → function-call, compatible
    if isinstance(inp0, list) and not all(isinstance(x, str) for x in inp0):
        return True

    # List of strings with exactly 1 element → works as func(*[arg]) for single-param funcs
    if isinstance(inp0, list) and len(inp0) == 1:
        return True

    # List of strings with >1 element → stdin-format, incompatible with function-call
    return False


def main():
    app_path = Path("dataset/code/meta/app.json")
    bak_path = Path("dataset/code/meta/app.json.bak")

    sys.set_int_max_str_digits(0)

    print(f"Loading {app_path}...")
    with open(app_path, encoding="utf-8") as f:
        data = json.load(f)

    total = len(data)
    compatible = [item for item in data if is_compatible(item)]
    removed = total - len(compatible)

    print(f"Total items: {total}")
    print(f"Compatible (kept): {len(compatible)}")
    print(f"Incompatible (removed): {removed}")

    # Show a few removed items for verification
    removed_items = [item for item in data if not is_compatible(item)]
    print(f"\nSample removed items:")
    for item in removed_items[:5]:
        fs = (item.get("func_sign", "") or "").split("(")[0].replace("def ", "").strip()[:50]
        inp0 = item.get("inputs", [[]])[0]
        print(f"  {fs}: inputs[0] = {len(inp0)} strings")

    # Backup and save
    shutil.copy2(app_path, bak_path)
    print(f"\nBackup saved to {bak_path}")

    with open(app_path, "w", encoding="utf-8") as f:
        json.dump(compatible, f, ensure_ascii=False)
    print(f"Cleaned data saved to {app_path}")


if __name__ == "__main__":
    main()
