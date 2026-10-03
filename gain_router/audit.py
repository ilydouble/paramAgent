"""Read-only quality report for old complete rows and new per-step journals."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from .common import read_jsonl
from .schema import CollectionRecord


def audit_trajectories(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    report = {}
    case_counts, label_counts, domain_cases = Counter(), Counter(), {}
    completed = unlabeled = 0
    calls = Counter()
    failures = Counter()
    truncations = Counter()
    seen = set()
    duplicates = []
    total_calls = empty_calls = training_eligible = unknown_eligibility = 0
    for row in read_jsonl(path):
        completed += 1
        if row.get("schema_version") == "router_call_v1":
            record = CollectionRecord.from_dict(row)
            sample_id = record.state.task.sample_id
            domain = record.state.task.domain
            stages = [("initial", record.state.initial.result),
                      ("preference", record.call.calls[0].result), ("repair", record.call.calls[1].result)]
            case_type, label_key = "unlabeled_call_no_call", "not_generated"
            unlabeled += 1
        else:
            sample_id = row.get("sample_id")
            domain = str(row.get("domain", "unknown"))
            stages = [(stage, row.get(stage)) for stage in ("initial", "generic", "diverse")]
            outcome = row.get("outcome") or {}
            case_type = str(outcome.get("case_type") or "unknown")
            label = outcome.get("label")
            label_key = "null" if label is None else str(label)
        case_counts[case_type] += 1
        label_counts[label_key] += 1
        domain_cases.setdefault(domain, Counter())[case_type] += 1
        if sample_id in seen:
            duplicates.append(sample_id)
        seen.add(sample_id)
        training_eligible += row.get("valid_for_training") is True
        unknown_eligibility += "valid_for_training" not in row
        for stage, call in stages:
            if not call:
                continue
            total_calls += 1
            key = f"{domain}/{stage}"
            calls[key] += 1
            if call.get("finish_reason") == "length":
                truncations[key] += 1
            empty_calls += not str(call.get("output") or "").strip()
            failure = (call.get("verifier") or {}).get("failure_type")
            if failure:
                failures[f"{domain}/{failure}"] += 1
    truncated = sum(truncations.values())
    report.update({
        "completed_total": completed, "unlabeled_call_no_call_rows": unlabeled,
        "case_counts": dict(case_counts), "label_counts": dict(label_counts),
        "domain_case_counts": {domain: dict(counts) for domain, counts in domain_cases.items()},
        "total_calls": total_calls,
        "calls_by_domain_stage": dict(calls),
        "truncated_calls": truncated,
        "truncation_rate": truncated / total_calls if total_calls else 0.0,
        "truncated_by_domain_stage": dict(truncations),
        "empty_output_calls": empty_calls,
        "failure_types": dict(failures),
        "duplicate_sample_ids": duplicates,
        "explicit_training_eligible": training_eligible,
        "unknown_training_eligibility": unknown_eligibility,
        "warning": "Quality statistics do not certify correct labels or absence of leakage.",
    })
    return report


def audit_events(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"available": False, "warning": "No per-step journal; failed intermediate calls may be missing."}
    counts = Counter()
    started, ended, samples_started, samples_ended = set(), set(), set(), set()
    returned, verified = set(), set()
    for event in read_jsonl(path):
        name = event["event"]
        counts[name] += 1
        sample = (event["sample_id"], event["attempt_id"])
        stage = (*sample, event["stage"])
        request = (*stage, event.get("data", {}).get("request_attempt"))
        if name == "request.started":
            started.add(request)
        if name in ("request.completed", "request.failed"):
            ended.add(request)
        if name == "request.completed":
            returned.add(stage)
        if name == "verification.completed":
            verified.add(stage)
        if name == "sample.started":
            samples_started.add(sample)
        if name in ("sample.completed", "sample.failed"):
            samples_ended.add(sample)
    report = {
        "available": True, "event_counts": dict(counts),
        "requests_without_terminal_event": len(started - ended),
        "returned_stages_without_verification": len(returned - verified),
        "sample_attempts_without_terminal_event": len(samples_started - samples_ended),
    }
    metadata = path.parent / "run.json"
    if metadata.exists():
        run = json.loads(metadata.read_text(encoding="utf-8"))
        if run.get("config", {}).get("protocol") == "router_call_v1":
            report.pop("returned_stages_without_verification")
            report["online_verification_expected"] = False
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path,
                        help="Run directory or trajectories.jsonl; no model calls or writes")
    args = parser.parse_args(argv)
    path = args.input / "trajectories.jsonl" if args.input.is_dir() else args.input
    if args.input.is_dir() and not path.exists():
        report = {"completed_total": 0, "warning": "No completed trajectories; inspect per-step events."}
    else:
        report = audit_trajectories(path)
    report["events"] = audit_events(path.parent / "events.jsonl")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0
