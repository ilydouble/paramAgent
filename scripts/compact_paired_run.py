#!/usr/bin/env python3
"""Back up and compact a stopped legacy paired run, preserving its run ID."""
import argparse
from contextlib import ExitStack
import fcntl
import hashlib
import json
import os
from pathlib import Path
import sys
import tarfile
import time
import uuid

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gain_router.common import file_sha256
from gain_router.paired import collection_config, validate_record
from gain_router.paired_storage import compact_event_data, compact_result
from gain_router.schema import ModelCall
from gain_router.traces import config_fingerprint, write_json_atomic

# Only this reviewed storage-only transition may change an existing run's code
# identity. Other changes still require a new output directory.
LEGACY_CODE = {
    "actor.py": "a2cd23c3618cfebf1114b9a09fbd1dd7519b9d6ce5ae6783635b02d5b9c09bb3",
    "paired.py": "a437dba85318e4a75745042e1d907f9ed048520c7a05bca142b3e295a10d732f",
}


def digest_stream(handle):
    digest = hashlib.sha256()
    for chunk in iter(lambda: handle.read(1024 * 1024), b""):
        digest.update(chunk)
    return digest.hexdigest()


def compact_trajectory(value):
    validate_record(value)
    calls = [value["state"]["initial"], *value["actor_only"]["calls"], *value["guided"]["calls"]]
    for call in calls:
        call["result"] = compact_result(call["result"])
    return validate_record(value)


def migrate(run, tasks, settings, backup, *, local_preference=False):
    run, backup = run.resolve(), backup.resolve()
    if backup == run or run in backup.parents:
        raise ValueError("Backup must be outside the run directory")
    if backup.exists():
        raise ValueError("Choose a new backup path; existing backups are never overwritten")
    pending = run / ".storage-migration.pending.json"
    if pending.exists():
        raise ValueError("Previous migration incomplete; inspect its backup and staging directory")
    with ExitStack() as locks:
        for path in (run.parent / ".launcher.lock", run / ".writer.lock"):
            handle = locks.enter_context(path.open("a"))
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        metadata = json.loads((run / "run.json").read_text())
        old = metadata["config"]
        if metadata["config_fingerprint"] != config_fingerprint(old):
            raise ValueError("Existing configuration fingerprint is corrupt")
        _, _, _, selected, new = collection_config(tasks, settings, local_preference=local_preference)
        old_behavior = {k: v for k, v in old.items() if k != "code_sha256"}
        new_behavior = {k: v for k, v in new.items() if k not in {"code_sha256", "storage_format"}}
        if old_behavior != new_behavior:
            raise ValueError("Tasks, settings, protocol or models differ; storage migration refused")
        previous_code, current_code = old["code_sha256"], new["code_sha256"]
        if set(current_code) != set(previous_code) | {"paired_storage.py"}:
            raise ValueError("Unexpected added or removed core modules")
        for name, digest in previous_code.items():
            if name in LEGACY_CODE:
                if digest != LEGACY_CODE[name]:
                    raise ValueError(f"Unsupported legacy source: {name}")
            elif current_code[name] != digest:
                raise ValueError(f"Non-storage code changed: {name}")

        sources = sorted(p for p in run.rglob("*") if p.is_file()
                         and p.name not in {".writer.lock", ".launcher.lock"})
        if any(p.is_symlink() for p in sources):
            raise ValueError("Symlinked records are unsupported")
        source_hashes = {str(p.relative_to(run)): file_sha256(p) for p in sources}
        before_bytes = sum(p.stat().st_size for p in sources)
        backup.parent.mkdir(parents=True, exist_ok=True)
        print("Creating gzip backup", backup, flush=True)
        with backup.open("xb") as handle:
            with tarfile.open(fileobj=handle, mode="w:gz", compresslevel=1) as archive:
                for p in sources:
                    archive.add(p, arcname=str(p.relative_to(run)), recursive=False)
            handle.flush()
            os.fsync(handle.fileno())
        restored = {}
        with tarfile.open(backup, "r:gz") as archive:
            for member in archive:
                if not member.isfile():
                    raise ValueError("Unexpected backup member")
                with archive.extractfile(member) as handle:
                    restored[member.name] = digest_stream(handle)
        if restored != source_hashes:
            raise ValueError("Backup decompression/hash verification failed")
        print("Backup verified byte-for-byte", backup.stat().st_size, "bytes", flush=True)

        stage = run.parent / (run.name + ".compact-staging-" + uuid.uuid4().hex)
        stage.mkdir()
        counts = {"trajectories": 0, "calls": 0, "events": 0}
        completed, requests = set(), {}
        selected_tasks = {t.sample_id: t.__dict__ for t in selected}
        new_fingerprint = config_fingerprint(new)
        for source in sources:
            name = str(source.relative_to(run))
            if name not in {"events.jsonl", "trajectories.jsonl"} and not name.startswith("calls/"):
                continue
            target = stage / name
            target.parent.mkdir(parents=True, exist_ok=True)
            with source.open() as reader, target.open("x") as writer:
                for line in reader:
                    if not line.endswith(chr(10)) or not line.strip():
                        raise ValueError(f"Invalid JSONL or checkpoint tail: {source}")
                    # Checkpoints are indented JSON; parse them once below.
                    if name.startswith("calls/"):
                        value = json.loads(line + reader.read())
                        request = value["request"]
                        sid, call_stage = request["sample_id"], request["stage"]
                        key = hashlib.sha256(f"{sid}:{call_stage}".encode()).hexdigest() + ".json"
                        if source.name != key or sid not in selected_tasks:
                            raise ValueError("Checkpoint identity differs from selected tasks")
                        ModelCall.from_dict(value["call"])
                        value["call"]["result"] = compact_result(value["call"]["result"])
                        requests[sid] = requests.get(sid, 0) + 1
                        counts["calls"] += 1
                    else:
                        value = json.loads(line)
                        if value["run_id"] != metadata["run_id"]:
                            raise ValueError("Record belongs to another run")
                        if name == "trajectories.jsonl":
                            value = compact_trajectory(value)
                            task = value["state"]["task"]
                            sid = task["sample_id"]
                            if sid in completed or selected_tasks.get(sid) != task:
                                raise ValueError("Duplicate or mismatched completed task")
                            completed.add(sid)
                            counts["trajectories"] += 1
                        else:
                            value["data"] = compact_event_data(value.get("data", {}))
                            value["config_fingerprint"] = new_fingerprint
                            counts["events"] += 1
                    writer.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + chr(10))
                writer.flush()
                os.fsync(writer.fileno())
        manifest = json.loads((run / "manifest.json").read_text())
        if manifest["completed_total"] != counts["trajectories"]:
            raise ValueError("Completed count differs from saved manifest")
        for name, digest in source_hashes.items():
            if file_sha256(run / name) != digest:
                raise ValueError("Source changed during migration")
        report = {"state": "prepared", "run_id": metadata["run_id"],
            "created_at_unix": time.time(), "backup": str(backup),
            "backup_sha256": file_sha256(backup), "source_sha256": source_hashes,
            "old_config_fingerprint": metadata["config_fingerprint"],
            "new_config_fingerprint": new_fingerprint, "counts": counts,
            "partial_task_call_counts": [n for sid, n in requests.items() if sid not in completed],
            "before_bytes": before_bytes, "staging": str(stage)}
        metadata.update(config=new, config_fingerprint=new_fingerprint,
                        storage_migration={k: report[k] for k in ("backup", "backup_sha256", "old_config_fingerprint")})
        write_json_atomic(stage / "run.json", metadata)
        report["replacement_sha256"] = {str(p.relative_to(stage)): file_sha256(p)
                                         for p in stage.rglob("*") if p.is_file()}
        write_json_atomic(pending, report)
        # Metadata is switched last. The pending marker blocks collection even
        # if the process stops halfway; verified original bytes remain in backup.
        replacements = sorted(p for p in stage.rglob("*") if p.is_file() and p.name != "run.json")
        replacements.append(stage / "run.json")
        for source in replacements:
            os.replace(source, run / source.relative_to(stage))
        for name, digest in report["replacement_sha256"].items():
            if file_sha256(run / name) != digest:
                raise ValueError("Replacement verification failed; recovery marker retained")
        report.update(state="complete", after_bytes=sum(p.stat().st_size for p in sources))
        write_json_atomic(run / "storage-migration.json", report)
        pending.unlink()
        for directory in sorted((p for p in stage.rglob("*") if p.is_dir()), reverse=True):
            directory.rmdir()
        stage.rmdir()
        print(json.dumps({k: report[k] for k in ("state", "counts", "partial_task_call_counts",
            "before_bytes", "after_bytes", "backup")}, ensure_ascii=False, indent=2), flush=True)
        return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("run", "tasks", "settings", "backup"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--local-preference", action="store_true")
    args = parser.parse_args()
    migrate(args.run, args.tasks, args.settings, args.backup, local_preference=args.local_preference)


if __name__ == "__main__":
    main()
