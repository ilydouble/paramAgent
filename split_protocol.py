"""Deterministic, group-aware dataset splitting utilities.

The split unit is an original problem (``group_id``), never a generated trace.
All artifacts derived from the same problem must inherit the same group id.
"""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


SCHEMA_VERSION = "paramagent.split.v1"
SPLIT_NAMES = ("train", "val", "test")


def normalize_text(value: Any) -> str:
    """Return a stable representation used only for identity fingerprints."""

    text = unicodedata.normalize("NFKC", str(value)).strip().lower()
    return re.sub(r"\s+", " ", text)


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def content_group_id(text: Any) -> str:
    """Create a content-only group id so exact duplicates cannot cross splits."""

    normalized = normalize_text(text)
    if not normalized:
        raise ValueError("Cannot derive group_id from empty text")
    return f"content:{sha256_text(normalized)}"


def sample_id(namespace: str, text: Any, explicit_id: Any | None = None) -> str:
    """Create a stable, non-reversible identifier for a source sample."""

    identity = normalize_text(explicit_id) if explicit_id not in (None, "") else normalize_text(text)
    if not identity:
        raise ValueError(f"Cannot derive sample_id for namespace={namespace!r}")
    return f"{namespace}:{sha256_text(identity)}"


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _rank(seed: int, *parts: str) -> str:
    payload = "\0".join((str(seed), *parts))
    return sha256_text(payload)


def _canonical_digest(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return sha256_text(payload)


def assign_exact_groups(
    records: Sequence[Mapping[str, Any]],
    *,
    seed: int,
    targets: Mapping[str, Mapping[str, int]],
    metadata: Mapping[str, Any] | None = None,
) -> tuple[dict[str, str], dict[str, Any]]:
    """Assign groups to exact per-stratum group counts.

    ``targets`` counts groups, not rows. This is intentional: a problem with
    several generated traces still occupies one split and one target slot.
    """

    if not records:
        raise ValueError("No records supplied")

    samples_seen: set[str] = set()
    group_records: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    group_stratum: dict[str, str] = {}
    for record in records:
        sid = str(record["sample_id"])
        gid = str(record["group_id"])
        stratum = str(record["stratum"])
        if sid in samples_seen:
            raise ValueError(f"Duplicate sample_id: {sid}")
        samples_seen.add(sid)
        previous = group_stratum.setdefault(gid, stratum)
        if previous != stratum:
            raise ValueError(
                f"group_id {gid} spans strata {previous!r} and {stratum!r}; "
                "fix source grouping before splitting"
            )
        group_records[gid].append(record)

    groups_by_stratum: dict[str, list[str]] = defaultdict(list)
    for gid, stratum in group_stratum.items():
        groups_by_stratum[stratum].append(gid)

    unexpected = sorted(set(groups_by_stratum) - set(targets))
    missing = sorted(set(targets) - set(groups_by_stratum))
    if unexpected or missing:
        raise ValueError(f"Stratum mismatch: unexpected={unexpected}, missing={missing}")

    assignment: dict[str, str] = {}
    for stratum in sorted(targets):
        requested = {name: int(targets[stratum].get(name, 0)) for name in SPLIT_NAMES}
        if any(value < 0 for value in requested.values()):
            raise ValueError(f"Negative target for stratum={stratum}: {requested}")
        groups = sorted(groups_by_stratum[stratum], key=lambda gid: (_rank(seed, stratum, gid), gid))
        if sum(requested.values()) != len(groups):
            raise ValueError(
                f"stratum={stratum}: target groups={sum(requested.values())}, "
                f"actual groups={len(groups)}"
            )
        cursor = 0
        for split_name in SPLIT_NAMES:
            next_cursor = cursor + requested[split_name]
            for gid in groups[cursor:next_cursor]:
                assignment[gid] = split_name
            cursor = next_cursor

    assignments = []
    for gid in sorted(group_records):
        rows = group_records[gid]
        assignments.append(
            {
                "group_id": gid,
                "split": assignment[gid],
                "stratum": group_stratum[gid],
                "sample_ids": sorted(str(row["sample_id"]) for row in rows),
                "record_count": len(rows),
            }
        )

    row_counts = Counter(assignment[str(row["group_id"])] for row in records)
    group_counts = Counter(assignment.values())
    manifest: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "seed": seed,
        "split_names": list(SPLIT_NAMES),
        "targets_are": "group_counts",
        "targets": {key: dict(value) for key, value in sorted(targets.items())},
        "summary": {
            "sample_count": len(records),
            "group_count": len(group_records),
            "rows_by_split": dict(sorted(row_counts.items())),
            "groups_by_split": dict(sorted(group_counts.items())),
        },
        "metadata": dict(metadata or {}),
        "assignments": assignments,
    }
    manifest["assignment_sha256"] = _canonical_digest(assignments)
    return assignment, manifest


def materialize_rows(
    records: Iterable[Mapping[str, Any]], assignment: Mapping[str, str], *, seed: int
) -> dict[str, list[dict[str, Any]]]:
    """Add split metadata and deterministically order rows within each split."""

    result: dict[str, list[dict[str, Any]]] = {name: [] for name in SPLIT_NAMES}
    for original in records:
        row = dict(original)
        gid = str(row["group_id"])
        split_name = assignment[gid]
        row["split"] = split_name
        row["split_seed"] = seed
        result[split_name].append(row)
    for split_name, rows in result.items():
        rows.sort(key=lambda row: (_rank(seed, split_name, str(row["sample_id"])), str(row["sample_id"])))
    return result


def load_manifest(path: str | Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as handle:
        manifest = json.load(handle)
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"Unsupported split manifest schema: {manifest.get('schema_version')!r}")
    assignments = manifest.get("assignments")
    if not isinstance(assignments, list):
        raise ValueError("Manifest assignments must be a list")
    expected = manifest.get("assignment_sha256")
    actual = _canonical_digest(assignments)
    if expected != actual:
        raise ValueError(f"Manifest assignment checksum mismatch: expected={expected}, actual={actual}")
    return manifest


def manifest_lookup(manifest: Mapping[str, Any]) -> tuple[dict[str, str], dict[str, str]]:
    """Return group->split and sample->split mappings after overlap checks."""

    by_group: dict[str, str] = {}
    by_sample: dict[str, str] = {}
    for item in manifest["assignments"]:
        split_name = str(item["split"])
        if split_name not in SPLIT_NAMES:
            raise ValueError(f"Unknown split name: {split_name}")
        gid = str(item["group_id"])
        if gid in by_group:
            raise ValueError(f"Duplicate group assignment: {gid}")
        by_group[gid] = split_name
        for sid_value in item.get("sample_ids", []):
            sid = str(sid_value)
            if sid in by_sample:
                raise ValueError(f"Duplicate sample assignment: {sid}")
            by_sample[sid] = split_name
    return by_group, by_sample


def validate_materialized_splits(data_dir: str | Path) -> dict[str, Any]:
    """Validate router JSONL files against ``manifest.json`` in ``data_dir``."""

    directory = Path(data_dir)
    manifest = load_manifest(directory / "manifest.json")
    by_group, by_sample = manifest_lookup(manifest)
    seen_groups: dict[str, str] = {}
    seen_samples: set[str] = set()
    counts: Counter[str] = Counter()
    for split_name in SPLIT_NAMES:
        path = directory / f"router_{split_name}.jsonl"
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                row = json.loads(line)
                sid = str(row["sample_id"])
                gid = str(row["group_id"])
                if row.get("split") != split_name or row.get("split_seed") != manifest["seed"]:
                    raise ValueError(f"{path}:{line_number}: embedded split metadata mismatch")
                if by_group.get(gid) != split_name or by_sample.get(sid) != split_name:
                    raise ValueError(f"{path}:{line_number}: row does not match manifest")
                previous = seen_groups.setdefault(gid, split_name)
                if previous != split_name:
                    raise ValueError(f"group leakage: {gid} appears in {previous} and {split_name}")
                if sid in seen_samples:
                    raise ValueError(f"duplicate sample row: {sid}")
                seen_samples.add(sid)
                counts[split_name] += 1
    if seen_samples != set(by_sample):
        raise ValueError("Materialized rows and manifest sample ids differ")
    return {"seed": manifest["seed"], "rows_by_split": dict(sorted(counts.items()))}
