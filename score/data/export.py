"""Whitelist router features; offline metrics and future calls never enter X."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

from .common import append_jsonl, file_sha256
from .config import ExperimentConfig, load_config
from .inputs import read_objects
from .offline import load_collection
from .traces import TraceRecorder, write_json_atomic


def export_features(run_dir: Path, labels_dir: Path, output_dir: Path,
                    config: ExperimentConfig, *, allow_pilot: bool = False) -> int:
    if not allow_pilot:
        raise ValueError("Current verifier/policy is pilot-only; acknowledge with --allow-pilot")
    metadata, records = load_collection(run_dir, config)
    manifest = json.loads((labels_dir / "manifest.json").read_text(encoding="utf-8"))
    identity = manifest["identity"]
    if (manifest["status"] != "complete" or identity["run_id"] != metadata["run_id"]
            or identity["trajectory_sha256"] != file_sha256(run_dir / "trajectories.jsonl")
            or identity["run_sha256"] != file_sha256(run_dir / "run.json")
            or identity["labels"] != asdict(config.labels)
            or manifest["label_sha256"] != file_sha256(labels_dir / "labels.jsonl")):
        raise ValueError("Label provenance/checksum mismatch")
    labels = {}
    for value in read_objects(labels_dir / "labels.jsonl"):
        sid = value["sample_id"]
        if sid in labels:
            raise ValueError("Duplicate label sample")
        labels[sid] = value
    if set(labels) != {r.state.task.sample_id for r in records}:
        raise ValueError("Label and trajectory sample sets differ")
    features = []
    for record in records:
        task = record.state.task
        label = labels[task.sample_id]
        if (label["group_id"], label["domain"], label["split"], label["split_seed"], label["run_id"], label["attempt_id"]) != (
                task.group_id, task.domain, task.split, task.split_seed, record.run_id, record.attempt_id):
            raise ValueError("Label identity does not match trajectory")
        if not label["valid_for_pilot_training"]:
            continue
        if type(label["label"]) is not int or label["label"] not in {0, 1}:
            raise ValueError("Expected binary label")
        features.append({
            "schema_version": "router_features_v1", "purpose": "pilot",
            "sample_id": task.sample_id, "group_id": task.group_id,
            "split": task.split, "split_seed": task.split_seed,
            "inputs": record.state.router_features(), "label": label["label"],
        })
    export_identity = {"operation": "feature_export_v1", "run_id": metadata["run_id"],
                       "labels_sha256": manifest["label_sha256"],
                       "exporter_sha256": file_sha256(Path(__file__)),
                       "schema_sha256": file_sha256(Path(__file__).parent / "schema.py")}
    with TraceRecorder(output_dir, export_identity):
        target = output_dir / "router_features.jsonl"
        if target.exists():
            raise FileExistsError("Feature export exists; use a new directory")
        # Create an empty file explicitly when all rows were excluded.
        with target.open("x", encoding="utf-8"):
            pass
        for row in features:
            append_jsonl(target, row)
        write_json_atomic(output_dir / "manifest.json", {
            "status": "complete", "purpose": "pilot", "formal_training_certified": False,
            "identity": export_identity, "record_count": len(features),
            "excluded_count": len(records) - len(features), "features_sha256": file_sha256(target),
        })
    return len(features)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--labels", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--allow-pilot", action="store_true")
    args = parser.parse_args(argv)
    count = export_features(args.run.resolve(), args.labels.resolve(), args.output_dir.resolve(),
                            load_config(args.config), allow_pilot=args.allow_pilot)
    print(json.dumps({"exported_pilot_rows": count}))
    return 0
