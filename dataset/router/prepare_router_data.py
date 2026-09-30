#!/usr/bin/env python3
"""Build fixed, leakage-resistant router splits for several data seeds.

The script runs on the AutoDL host where the ignored source JSONL files live.
It emits materialized training files under ``dataset/router/splits`` and
ID-only manifests under ``split_manifests`` for version control.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from split_protocol import (
    SPLIT_NAMES,
    assign_exact_groups,
    content_group_id,
    file_sha256,
    materialize_rows,
    sample_id,
    validate_materialized_splits,
)


HERE = Path(__file__).resolve().parent
LABEL_INDEX = {"abstain": 0, "code": 1, "math": 2, "qa": 3}
DEFAULT_SEEDS = (42, 123, 2027, 3407, 8888)

# Exact group counts retained from the existing router experiment.
SPLIT_SPEC = {
    "code": {"train": 468, "val": 60, "test": 60},
    "math": {"train": 1440, "val": 180, "test": 180},
    "qa": {"train": 1440, "val": 180, "test": 180},
    "abstain": {"train": 1440, "val": 180, "test": 180},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path, default=HERE)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=HERE / "splits" / "router_v1",
        help="Ignored directory for materialized JSONL data",
    )
    parser.add_argument(
        "--manifest-root",
        type=Path,
        default=REPO_ROOT / "split_manifests" / "router_v1",
        help="ID-only manifests safe to commit",
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def make_record(
    *,
    source_name: str,
    source_index: int,
    text: str,
    label: str,
    source_domain: str,
    explicit_id: Any | None = None,
) -> dict[str, Any]:
    return {
        "text": text,
        "label": label,
        "label_id": LABEL_INDEX[label],
        "sample_id": sample_id(source_name, text, explicit_id),
        "group_id": content_group_id(text),
        "stratum": label,
        "source_name": source_name,
        "source_index": source_index,
        "source_domain": source_domain,
    }


def load_records(source_dir: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    paths = {name: source_dir / f"{name}.jsonl" for name in ("code", "math", "qa", "abs")}
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing router sources: {missing}")

    records: list[dict[str, Any]] = []
    for index, row in enumerate(read_jsonl(paths["code"])):
        records.append(
            make_record(
                source_name="router/code",
                source_index=index,
                text=row["func_sign"],
                label="code",
                source_domain="code",
                explicit_id=row.get("task_id") or row.get("problem_id"),
            )
        )
    for index, row in enumerate(read_jsonl(paths["math"])):
        records.append(
            make_record(
                source_name="router/math",
                source_index=index,
                text=row["problem"],
                label="math",
                source_domain="math",
                explicit_id=row.get("unique_id") or row.get("id"),
            )
        )
    for index, row in enumerate(read_jsonl(paths["qa"])):
        records.append(
            make_record(
                source_name="router/qa",
                source_index=index,
                text=row["question"],
                label="qa",
                source_domain="qa",
                explicit_id=row.get("id") or row.get("question_id"),
            )
        )

    abstain_rows = read_jsonl(paths["abs"])
    if len(abstain_rows) != 1800:
        raise ValueError(f"abs.jsonl must contain 1800 rows, found {len(abstain_rows)}")
    for index, row in enumerate(abstain_rows):
        if index < 600:
            field, domain = "func_sign", "code"
        elif index < 1200:
            field, domain = "problem", "math"
        else:
            field, domain = "question", "qa"
        records.append(
            make_record(
                source_name=f"router/abstain/{domain}",
                source_index=index,
                text=row[field],
                label="abstain",
                source_domain=domain,
                explicit_id=row.get("unique_id") or row.get("task_id") or row.get("id"),
            )
        )

    sources = [
        {"name": name, "path": path.name, "sha256": file_sha256(path), "rows": len(read_jsonl(path))}
        for name, path in sorted(paths.items())
    ]
    return records, sources


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def write_seed(
    records: list[dict[str, Any]],
    sources: list[dict[str, Any]],
    *,
    seed: int,
    output_root: Path,
    manifest_root: Path,
    overwrite: bool,
) -> dict[str, Any]:
    seed_dir = output_root / f"seed_{seed}"
    tracked_manifest = manifest_root / f"seed_{seed}.json"
    existing = [path for path in (seed_dir, tracked_manifest) if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing split artifacts: {existing}")

    assignment, manifest = assign_exact_groups(
        records,
        seed=seed,
        targets=SPLIT_SPEC,
        metadata={
            "protocol": "router_v1",
            "grouping": "NFKC + lowercase + whitespace-normalized original problem text SHA-256",
            "sample_identity": "explicit source id when available, otherwise original text SHA-256",
            "sources": sources,
        },
    )
    rows_by_split = materialize_rows(records, assignment, seed=seed)
    seed_dir.mkdir(parents=True, exist_ok=True)
    for split_name in SPLIT_NAMES:
        output_path = seed_dir / f"router_{split_name}.jsonl"
        with output_path.open("w", encoding="utf-8") as handle:
            for row in rows_by_split[split_name]:
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    write_json(seed_dir / "manifest.json", manifest)
    write_json(tracked_manifest, manifest)
    validation = validate_materialized_splits(seed_dir)
    print(f"seed={seed}: {validation['rows_by_split']} manifest={tracked_manifest}")
    return {
        "seed": seed,
        "assignment_sha256": manifest["assignment_sha256"],
        "rows_by_split": validation["rows_by_split"],
    }


def main() -> None:
    args = parse_args()
    if len(args.seeds) != len(set(args.seeds)):
        raise ValueError(f"Duplicate seeds are not allowed: {args.seeds}")
    records, sources = load_records(args.source_dir)
    counts = Counter(row["stratum"] for row in records)
    expected = {label: sum(spec.values()) for label, spec in SPLIT_SPEC.items()}
    if dict(counts) != expected:
        raise ValueError(f"Source counts differ from protocol: actual={dict(counts)}, expected={expected}")

    results = [
        write_seed(
            records,
            sources,
            seed=seed,
            output_root=args.output_root,
            manifest_root=args.manifest_root,
            overwrite=args.overwrite,
        )
        for seed in args.seeds
    ]
    write_json(
        args.manifest_root / "index.json",
        {
            "protocol": "router_v1",
            "primary_seed": 42,
            "robustness_seeds": [seed for seed in args.seeds if seed != 42],
            "split_spec": SPLIT_SPEC,
            "sources": sources,
            "variants": results,
        },
    )
    print(f"Wrote {len(results)} fixed split variants. Primary paper split: seed 42.")


if __name__ == "__main__":
    main()
