#!/usr/bin/env python3
"""Apply an existing problem-level split manifest to derived JSON/JSONL rows."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from split_protocol import SPLIT_NAMES, load_manifest, manifest_lookup


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Split derived traces/preferences with the original problem manifest. "
            "Every input row must retain group_id or sample_id."
        )
    )
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--input", required=True, help="Input .json or .jsonl")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--prefix", default="derived")
    parser.add_argument("--unmatched", choices=("error", "drop"), default="error")
    return parser.parse_args()


def read_rows(path: Path) -> list[dict]:
    if path.suffix == ".jsonl":
        with path.open(encoding="utf-8") as handle:
            return [json.loads(line) for line in handle if line.strip()]
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, list):
        raise ValueError("JSON input must contain a top-level list")
    return value


def apply_manifest(rows: list[dict], manifest: dict, unmatched: str) -> dict[str, list[dict]]:
    by_group, by_sample = manifest_lookup(manifest)
    output = {name: [] for name in SPLIT_NAMES}
    dropped = 0
    for index, original in enumerate(rows):
        gid = original.get("group_id")
        sid = original.get("sample_id") or original.get("source_sample_id")
        split_by_group = by_group.get(str(gid)) if gid is not None else None
        split_by_sample = by_sample.get(str(sid)) if sid is not None else None
        if split_by_group and split_by_sample and split_by_group != split_by_sample:
            raise ValueError(f"row {index}: group_id and sample_id map to different splits")
        split_name = split_by_group or split_by_sample
        if split_name is None:
            if unmatched == "drop":
                dropped += 1
                continue
            raise ValueError(
                f"row {index}: no manifest match; derived rows must carry the original group_id or sample_id"
            )
        row = dict(original)
        row["split"] = split_name
        row["split_seed"] = manifest["seed"]
        output[split_name].append(row)
    output["_dropped"] = dropped
    return output


def main() -> None:
    args = parse_args()
    manifest = load_manifest(args.manifest)
    result = apply_manifest(read_rows(Path(args.input)), manifest, args.unmatched)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    dropped = result.pop("_dropped")
    for split_name in SPLIT_NAMES:
        path = output_dir / f"{args.prefix}_{split_name}.jsonl"
        with path.open("w", encoding="utf-8") as handle:
            for row in result[split_name]:
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        print(f"{split_name}: {len(result[split_name])} -> {path}")
    if dropped:
        print(f"dropped unmatched rows: {dropped}")


if __name__ == "__main__":
    main()
