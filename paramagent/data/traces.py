"""Durable per-step trace journal, isolated from labels and router features.

One writer per run directory. Failed/incomplete attempts stay in events.jsonl;
only fully completed samples belong in trajectories.jsonl.
"""
from __future__ import annotations

import fcntl
import hashlib
import json
import os
import time
import uuid
from pathlib import Path
from typing import Any

from .common import append_jsonl, read_jsonl


TRACE_SCHEMA = "router_trace_v1"


def config_fingerprint(config: dict[str, Any]) -> str:
    packed = json.dumps(config, ensure_ascii=False, sort_keys=True, allow_nan=False)
    return hashlib.sha256(packed.encode("utf-8")).hexdigest()


def write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    temporary = path.parent / f".{path.name}-{uuid.uuid4().hex}.tmp"
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            handle.write(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


class TraceRecorder:
    """Owns an append-only event journal and refuses incompatible resume."""

    def __init__(self, output_dir: Path, config: dict[str, Any]):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._lock = (self.output_dir / ".writer.lock").open("a")
        try:
            fcntl.flock(self._lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            self.fingerprint = config_fingerprint(config)
            path = self.output_dir / "run.json"
            if path.exists():
                previous = json.loads(path.read_text(encoding="utf-8"))
                if previous.get("config_fingerprint") != self.fingerprint:
                    raise ValueError("Run configuration changed; choose a new output directory.")
                self.run_id = previous["run_id"]
                # Refuse to append after a corrupt/truncated JSONL tail.
                # Preserve original bytes; recovery must be explicit.
                for name in ("events.jsonl", "trajectories.jsonl", "errors.jsonl"):
                    artifact = self.output_dir / name
                    if artifact.exists():
                        for _ in read_jsonl(artifact):
                            pass
                        if artifact.stat().st_size:
                            with artifact.open("rb") as handle:
                                handle.seek(-1, os.SEEK_END)
                                if handle.read(1) != b"\n":
                                    raise ValueError(f"Missing JSONL newline at {artifact}; explicit recovery required.")
            else:
                for name in ("trajectories.jsonl", "events.jsonl", "errors.jsonl", "manifest.json"):
                    artifact = self.output_dir / name
                    if artifact.exists():
                        raise ValueError("Unversioned artifacts exist; preserve them and choose a new output directory.")
                self.run_id = uuid.uuid4().hex
                metadata = {
                    "trace_schema": TRACE_SCHEMA,
                    "run_id": self.run_id,
                    "config_fingerprint": self.fingerprint,
                    "config": config,
                    "created_at_unix": time.time(),
                }
                write_json_atomic(path, metadata)
        except BaseException:
            self._lock.close()
            raise

    def event(self, event: str, *, sample_id: str, attempt_id: str,
              stage: str, data: dict[str, Any] | None = None) -> None:
        append_jsonl(self.output_dir / "events.jsonl", {
            "trace_schema": TRACE_SCHEMA,
            "event_id": uuid.uuid4().hex,
            "run_id": self.run_id,
            "config_fingerprint": self.fingerprint,
            "sample_id": sample_id,
            "attempt_id": attempt_id,
            "stage": stage,
            "event": event,
            "created_at_unix": time.time(),
            "data": data or {},
        })

    def close(self) -> None:
        self._lock.close()

    def __enter__(self) -> "TraceRecorder":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()
