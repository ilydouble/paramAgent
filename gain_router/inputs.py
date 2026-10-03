"""Strict online inputs and immutable external split-manifest application."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterator

from split_protocol import load_manifest, manifest_lookup
from .schema import TaskSample


def read_objects(path: Path) -> Iterator[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{number}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"Expected object at {path}:{number}")
            yield value


def load_tasks(path: Path, manifest_path: Path) -> list[TaskSample]:
    manifest = load_manifest(manifest_path)
    by_group, by_sample = manifest_lookup(manifest)
    membership = {item["group_id"]: set(item["sample_ids"]) for item in manifest["assignments"]}
    seen = set()
    tasks = []
    for value in read_objects(path):
        task = TaskSample.from_dict(value)
        if task.sample_id in seen:
            raise ValueError(f"Duplicate sample_id: {task.sample_id}")
        seen.add(task.sample_id)
        if (by_group.get(task.group_id) != task.split or by_sample.get(task.sample_id) != task.split
                or task.sample_id not in membership.get(task.group_id, set())
                or task.split_seed != manifest["seed"]):
            raise ValueError(f"Task does not match frozen manifest: {task.sample_id}")
        tasks.append(task)
    if not tasks:
        raise ValueError("No tasks found")
    return tasks


def select_tasks(tasks: list[TaskSample], *, split: str, seed: int,
                 max_per_domain: int) -> list[TaskSample]:
    selected = []
    for domain in ("code", "math", "qa"):
        pool = [task for task in tasks if task.domain == domain and task.split == split]
        pool.sort(key=lambda task: (hashlib.sha256(f"{seed}:{task.sample_id}".encode()).hexdigest(), task.sample_id))
        selected.extend(pool[:max_per_domain])
    if not selected:
        raise ValueError(f"No tasks in configured split {split}")
    return selected


def call_seed(sample_id: str, seed: int, stage: str) -> int:
    digest = hashlib.sha256(f"{seed}:{sample_id}:{stage}".encode()).hexdigest()
    return int(digest[:8], 16) % (2**31 - 1)
