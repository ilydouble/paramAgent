"""Legacy identifiers, local JSON I/O and deterministic sampling helpers."""
from __future__ import annotations

import hashlib
import json
import os
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

SCHEMA_VERSION = "gain_router_v1"
PROTOCOL_VERSION = "legacy_gd_pilot_logged_v2"
DEFAULT_MODEL = "Qwen3.5-4B"
DEFAULT_BASE_URL = "http://127.0.0.1:8000/v1"

# Legacy benchmark compatibility; not a claim that the child verifier handles
# huge integer fixtures correctly. This remains a pilot limitation.
if hasattr(sys, "set_int_max_str_digits"):
    sys.set_int_max_str_digits(0)

def stable_id(domain: str, text: str) -> str:
    normalized = normalize_space(text).lower()
    return f"{domain}:{hashlib.sha256(normalized.encode('utf-8')).hexdigest()}"


def normalize_space(text: Any) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def split_for_group(group_id: str, seed: int) -> str:
    digest = hashlib.sha256(
        f"{SCHEMA_VERSION}:{seed}:{group_id}".encode("utf-8")
    ).hexdigest()
    bucket = int(digest[:12], 16) % 10_000
    if bucket < 8_000:
        return "train"
    if bucket < 9_000:
        return "validation"
    return "test"


def ranked_sample(rows: list[dict[str, Any]], seed: int, limit: int) -> list[dict[str, Any]]:
    def key(row: dict[str, Any]) -> str:
        return hashlib.sha256(f"{seed}:{row['sample_id']}".encode()).hexdigest()

    return sorted(rows, key=key)[:limit]


def read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL at {path}:{line_number}: {exc}") from exc
            if isinstance(value, dict):
                yield value


def load_json_array(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, list):
        raise ValueError(f"expected a JSON array: {path}")
    return [row for row in value if isinstance(row, dict)]
def append_jsonl(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def completed_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    ids = set()
    for row in read_jsonl(path):
        if row.get("sample_id"):
            ids.add(str(row["sample_id"]))
    return ids


def summarize_completed(path: Path) -> dict[str, Any]:
    cases: Counter[str] = Counter()
    labels: Counter[str] = Counter()
    domains: dict[str, Counter[str]] = {}
    total = 0
    if path.exists():
        for row in read_jsonl(path):
            total += 1
            domain = str(row.get("domain") or "unknown")
            outcome = row.get("outcome") or {}
            case_type = str(outcome.get("case_type") or "unknown")
            label = outcome.get("label")
            label_key = "null" if label is None else str(label)
            cases[case_type] += 1
            labels[label_key] += 1
            domains.setdefault(domain, Counter())[case_type] += 1
    return {
        "completed_total": total,
        "case_counts": dict(cases),
        "label_counts": dict(labels),
        "domain_case_counts": {domain: dict(counts) for domain, counts in domains.items()},
    }


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
