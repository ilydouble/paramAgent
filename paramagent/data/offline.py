"""Offline-only scoring and labeling. No actor/client calls in this module."""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
import json
import math
from pathlib import Path
import platform
from typing import Any

from .common import append_jsonl, file_sha256
from .config import ExperimentConfig, LabelConfig, load_config
from .inputs import read_objects
from .schema import CollectionRecord, OfflineSupervision, StrategyTrace
from .traces import TraceRecorder, config_fingerprint, write_json_atomic
from . import verifiers


def _score(supervision: OfflineSupervision, output: str) -> tuple[dict[str, Any], str | None]:
    try:
        metrics = verifiers.verify({"domain": supervision.domain, "gold": supervision.gold,
                                    "tests": supervision.tests}, output)
    except Exception as exc:
        return {"error_type": type(exc).__name__, "error": str(exc)}, "verifier_error"
    if metrics.get("failure_type") == "runner_error":
        return metrics, "verifier_error"
    # Pilot Math checker cannot distinguish symbolic mismatch from incorrectness.
    if supervision.domain == "math" and not metrics["success"]:
        return metrics, "ambiguous_math_equivalence"
    return metrics, None


def _cost(trace: StrategyTrace, config: LabelConfig) -> tuple[dict[str, Any], list[str]]:
    tokens, latency, exclusions = 0, 0.0, []
    tokens_available = latency_available = True
    for call in trace.calls:
        usage = call.result.get("usage") or {}
        total_tokens = usage.get("total_tokens")
        duration = call.result.get("latency_seconds")
        if type(total_tokens) is not int or total_tokens < 0:
            tokens_available = False
        else:
            tokens += total_tokens
        if isinstance(duration, bool) or not isinstance(duration, (int, float)) or not math.isfinite(duration) or duration < 0:
            latency_available = False
        else:
            latency += duration
    if not tokens_available and config.lambda_tokens > 0:
        exclusions.append("missing_token_cost")
    if not latency_available and config.lambda_latency > 0:
        exclusions.append("missing_latency_cost")
    # Request retries/fallback may have consumed unreported tokens. Conservative
    # exclusion when token-aware labels are requested, rather than undercounting.
    if config.lambda_tokens > 0:
        for call in trace.calls:
            if call.result.get("attempts", 1) > 1 or call.result.get("logprobs_fallback", False):
                exclusions.append("uncertain_retry_token_cost")
    return {"total_tokens": tokens if tokens_available else None,
            "latency_seconds": latency if latency_available else None}, exclusions


def derive_label(record: CollectionRecord, supervision: OfflineSupervision | None,
                 config: LabelConfig) -> dict[str, Any]:
    task = record.state.task
    exclusions = []
    calls = (record.state.initial, *record.call.calls)
    for index, call in enumerate(calls):
        exclusions.extend(f"call_{index}:{flag}" for flag in call.quality_flags)
    if supervision is None:
        exclusions.append("missing_supervision")
    elif (supervision.sample_id, supervision.group_id, supervision.domain) != (task.sample_id, task.group_id, task.domain):
        raise ValueError(f"Supervision identity mismatch: {task.sample_id}")
    costs, cost_exclusions = _cost(record.call, config)
    exclusions.extend(cost_exclusions)
    baseline_metrics = call_metrics = None
    delta = None
    if not exclusions:
        baseline_metrics, baseline_error = _score(supervision, record.no_call.final_output)
        call_metrics, call_error = _score(supervision, record.call.final_output)
        exclusions.extend(error for error in (baseline_error, call_error) if error)
        if not exclusions:
            # This configured objective is binary task success, NOT QA F1 or
            # code partial pass rate. Detailed continuous metrics stay below.
            delta = (float(call_metrics["success"]) - float(baseline_metrics["success"])
                     - config.lambda_tokens * (costs["total_tokens"] or 0)
                     - config.lambda_latency * (costs["latency_seconds"] or 0))
    return {
        "label_schema": "router_label_v1",
        "purpose": "pilot", "run_id": record.run_id, "attempt_id": record.attempt_id,
        "sample_id": task.sample_id, "group_id": task.group_id, "domain": task.domain,
        "split": task.split, "split_seed": task.split_seed,
        "label": int(delta > 0) if delta is not None else None,
        "valid_for_pilot_training": delta is not None,
        "formal_training_certified": False,
        "exclusion_reasons": sorted(set(exclusions)),
        "delta_utility": delta,
        "offline_rewards": {"no_call": baseline_metrics, "call": call_metrics},
        "incremental_cost": costs,
    }


def load_collection(run_dir: Path, config: ExperimentConfig) -> tuple[dict, list[CollectionRecord]]:
    metadata = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    if metadata["config"].get("protocol") != "router_call_v1":
        raise ValueError("Legacy G/D traces cannot be relabeled as call/no-call")
    if metadata["config"]["experiment"] != config.collection_config():
        raise ValueError("Collection configuration mismatch; only labels may change")
    if metadata["config_fingerprint"] != config_fingerprint(metadata["config"]):
        raise ValueError("Run metadata fingerprint mismatch")
    records = []
    seen = set()
    selected = {task["sample_id"]: task for task in metadata["config"]["selected_tasks"]}
    for value in read_objects(run_dir / "trajectories.jsonl"):
        record = CollectionRecord.from_dict(value)
        if record.run_id != metadata["run_id"] or record.state.task.sample_id in seen:
            raise ValueError("Mismatched run or duplicate sample in collection")
        if asdict(record.state.task) != selected.get(record.state.task.sample_id):
            raise ValueError("Recorded task differs from immutable selected input")
        models = (record.state.initial, record.call.calls[1], record.call.calls[0])
        expected = (config.actor, config.actor, config.preference_models[record.state.task.domain])
        if any((call.model_id, call.revision) != (model.model_id, model.revision) for call, model in zip(models, expected)):
            raise ValueError("Recorded model differs from collection configuration")
        seen.add(record.state.task.sample_id)
        records.append(record)
    if not records:
        raise ValueError("No completed collection records")
    return metadata, records


def build_labels(run_dir: Path, config: ExperimentConfig, supervision_path: Path,
                 *, allow_code_execution: bool = False) -> Path:
    metadata, records = load_collection(run_dir, config)
    supervision = {}
    for value in read_objects(supervision_path):
        item = OfflineSupervision.from_dict(value)
        if item.sample_id in supervision:
            raise ValueError(f"Duplicate supervision: {item.sample_id}")
        supervision[item.sample_id] = item
    if any(record.state.task.domain == "code" for record in records) and not allow_code_execution:
        raise ValueError("Code verifier is NOT a security sandbox; pass --allow-code-execution only in an isolated environment")
    identity = {
        "operation": "offline_labels_v1", "run_id": metadata["run_id"], "labels": asdict(config.labels),
        "trajectory_sha256": file_sha256(run_dir / "trajectories.jsonl"),
        "supervision_sha256": file_sha256(supervision_path),
        "run_sha256": file_sha256(run_dir / "run.json"),
        "verifier_sha256": file_sha256(Path(verifiers.__file__)),
        "labeler_sha256": file_sha256(Path(__file__)),
        "common_sha256": file_sha256(Path(__file__).parent / "common.py"),
        "python_version": platform.python_version(),
    }
    directory = run_dir / "labels" / config_fingerprint(identity)
    with TraceRecorder(directory, identity):
        output = directory / "labels.jsonl"
        if output.exists():
            raise FileExistsError(f"Immutable label version already exists: {directory}")
        # Score before publishing any rows so exceptions do not leave a partial
        # label file that can be mistaken for a complete version.
        labels = [derive_label(record, supervision.get(record.state.task.sample_id), config.labels) for record in records]
        for label in labels:
            append_jsonl(output, label)
        counts = Counter("excluded" if row["label"] is None else str(row["label"]) for row in labels)
        report = {"purpose": "pilot", "formal_training_certified": False, "status": "complete",
                  "identity": identity, "label_sha256": file_sha256(output),
                  "record_count": len(labels), "label_counts": dict(counts),
                  "selected_total": len(metadata["config"]["selected_tasks"]),
                  "incomplete_sample_count": len(metadata["config"]["selected_tasks"]) - len(records)}
        write_json_atomic(directory / "manifest.json", report)
    return directory


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--supervision", required=True, type=Path)
    parser.add_argument("--allow-code-execution", action="store_true")
    args = parser.parse_args(argv)
    directory = build_labels(args.run.resolve(), load_config(args.config), args.supervision.resolve(),
                             allow_code_execution=args.allow_code_execution)
    print(directory)
    return 0
