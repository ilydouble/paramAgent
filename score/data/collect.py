"""Collect paired call/no-call outcomes without reading any offline answers."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import time
import uuid

from .actor import call_actor
from .common import append_jsonl, file_sha256
from .config import ExperimentConfig, load_config, resolve_path, validate_model_paths
from .inputs import call_seed, load_tasks, read_objects, select_tasks
from .policies import execute_diversity, execute_initial, retain_initial
from .schema import CollectionRecord, ModelCall, SCHEMA_VERSION, TaskSample, serialize
from .traces import TraceRecorder, config_fingerprint, write_json_atomic


def prepare(root: Path, config: ExperimentConfig) -> tuple[list[TaskSample], dict]:
    tasks_path = resolve_path(root, config.tasks_path)
    manifest_path = resolve_path(root, config.split_manifest)
    tasks = select_tasks(load_tasks(tasks_path, manifest_path), split=config.split,
                         seed=config.seed, max_per_domain=config.max_per_domain)
    missing = {task.domain for task in tasks} - set(config.preference_models)
    if missing:
        raise ValueError(f"Missing preference model configuration for {sorted(missing)}")
    source = Path(__file__).parent
    identity = {
        "protocol": SCHEMA_VERSION,
        "experiment": config.collection_config(),
        "source_sha256": {"tasks": file_sha256(tasks_path), "split_manifest": file_sha256(manifest_path)},
        "selected_sha256": config_fingerprint({"tasks": [asdict(task) for task in tasks]}),
        "selected_tasks": [asdict(task) for task in tasks],
        "code_sha256": {p.name: file_sha256(p) for p in sorted(source.glob("*.py"))},
        "model_identity_status": "declared, not verified against running endpoint; pilot only",
    }
    return tasks, identity


def collect_one(task: TaskSample, config: ExperimentConfig,
                recorder: TraceRecorder, attempt_id: str) -> CollectionRecord:
    def emit(event: str, stage: str, data: dict) -> None:
        recorder.event(event, sample_id=task.sample_id, attempt_id=attempt_id, stage=stage, data=data)

    def invoke(stage, role, model, system, user):
        result = call_actor(
            base_url=model.endpoint, model=model.model_id, system=system, user=user,
            temperature=model.temperature, max_tokens=model.max_tokens,
            seed=call_seed(task.sample_id, config.seed, stage),
            timeout=config.request_timeout, retries=config.retries,
            top_p=model.top_p, enable_thinking=model.enable_thinking,
            trace=lambda event, data: emit(event, stage, data),
        )
        return ModelCall(role, model.model_id, model.revision, result)

    emit("sample.started", "sample", {"task": serialize(task)})
    state = execute_initial(task, config.actor, invoke)
    no_call = retain_initial(state)
    emit("policy.retained", "no_call", {"policy_version": no_call.policy_version})
    # Every selected initial state gets the call branch, including correct ones.
    # Collection has no gold correctness or verifier decision available.
    call = execute_diversity(state, config.actor, config.preference_models[task.domain], invoke)
    record = CollectionRecord(SCHEMA_VERSION, "pilot", recorder.run_id, attempt_id, state, no_call, call)
    # Validate structural invariants before appending a complete row.
    return CollectionRecord.from_dict(json.loads(json.dumps(serialize(record))))


def run_collection(root: Path, config: ExperimentConfig) -> int:
    tasks, identity = prepare(root, config)
    validate_model_paths(root, config)
    if any(model.revision.startswith("REPLACE_") for model in [config.actor, *config.preference_models.values()]):
        raise ValueError("Replace example model revisions before executing requests")
    output = resolve_path(root, config.output_dir)
    selected_ids = {task.sample_id for task in tasks}
    with TraceRecorder(output, identity) as recorder:
        trajectory_path = output / "trajectories.jsonl"
        done = set()
        if trajectory_path.exists():
            for value in read_objects(trajectory_path):
                previous = CollectionRecord.from_dict(value)
                sid = previous.state.task.sample_id
                if previous.run_id != recorder.run_id or sid in done or sid not in selected_ids:
                    raise ValueError("Completed rows do not belong to this immutable run")
                done.add(sid)
        manifest = {"schema_version": SCHEMA_VERSION, "purpose": "pilot", "status": "running",
                    "run_id": recorder.run_id, "config_fingerprint": recorder.fingerprint,
                    "selected_total": len(tasks), "completed_total": len(done),
                    "updated_at_unix": time.time()}
        write_json_atomic(output / "manifest.json", manifest)
        errors = 0
        for task in tasks:
            if task.sample_id in done:
                continue
            attempt_id = uuid.uuid4().hex
            try:
                record = collect_one(task, config, recorder, attempt_id)
                append_jsonl(trajectory_path, serialize(record))
                done.add(task.sample_id)
                recorder.event("sample.completed", sample_id=task.sample_id,
                               attempt_id=attempt_id, stage="sample", data={"labels_generated": False})
                print(f"[{len(done)}/{len(tasks)}] {task.domain} {task.sample_id}", flush=True)
            except OSError:
                raise  # Journal/storage failures must stop, not silently skip a sample.
            except Exception as exc:
                error = {"run_id": recorder.run_id, "sample_id": task.sample_id, "attempt_id": attempt_id,
                         "error_type": type(exc).__name__, "error": str(exc), "created_at_unix": time.time()}
                append_jsonl(output / "errors.jsonl", error)
                recorder.event("sample.failed", sample_id=task.sample_id,
                               attempt_id=attempt_id, stage="sample", data=error)
                errors += 1
            manifest.update(completed_total=len(done), new_error_count=errors, updated_at_unix=time.time())
            write_json_atomic(output / "manifest.json", manifest)
        manifest.update(status="complete" if done == selected_ids else "partial",
                        completed_total=len(done), new_error_count=errors)
        write_json_atomic(output / "manifest.json", manifest)
        return 0 if manifest["status"] == "complete" else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--root", type=Path, default=Path.cwd(), help="Relative data/model/output paths resolve here")
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--execute", action="store_true", help="Explicitly permit model requests")
    modes.add_argument("--dry-run", action="store_true", help="Validate inputs/paths without calls or writes (default)")
    parser.add_argument("--validate-config", action="store_true", help="Only parse config; do not read dataset/model paths")
    args = parser.parse_args(argv)
    if args.validate_config and args.execute:
        parser.error("--validate-config cannot be combined with --execute")
    config = load_config(args.config)
    root = args.root.resolve()
    if args.validate_config:
        print(json.dumps({"config_valid": True, "purpose": config.purpose}, indent=2))
        return 0
    if not args.execute:
        tasks, identity = prepare(root, config)
        validate_model_paths(root, config)
        print(json.dumps({"dry_run": True, "selected_total": len(tasks), "identity": identity}, ensure_ascii=False, indent=2))
        return 0
    return run_collection(root, config)
