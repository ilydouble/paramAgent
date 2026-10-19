"""Paired multi-round routing: shared initial answer, actor-only vs guided actor.

Generation never reads supervision. Scoring is a separate command. This pilot
uses fixed rounds and the last answer, without oracle feedback or best-of-rounds
selection. It is a new protocol, not a replay of the legacy two-stage runner.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import uuid

from .actor import call_actor
from .common import append_jsonl, file_sha256
from .config import LabelConfig, ModelConfig
from .inputs import call_seed, read_objects, select_tasks
from .policies import execute_initial, task_prompt
from .schema import DecisionState, ModelCall, OfflineSupervision, StrategyTrace, TaskSample, strict_keys
from .traces import TraceRecorder, write_json_atomic
from .paired_storage import STORAGE_FORMAT, compact_event_data, compact_result

PROTOCOL = "router_paired_multiround_v1"


def execute_branch(state, actor, preference, invoke, *, rounds, guided):
    if type(rounds) is not int or rounds < 1:
        raise ValueError("rounds must be a positive integer")
    system, task = task_prompt(state.task)
    current = state.initial.output
    calls = []
    branch = "guided" if guided else "actor_only"
    for index in range(1, rounds + 1):
        guidance = ""
        if guided:
            call = invoke(
                f"{branch}.{index}.preference", "preference", preference,
                "Generate actionable, diverse guidance to improve the supplied response. "
                "Use only the task and response; no reference answer or hidden tests are available.",
                f"Task:\n{task}\n\nCurrent response:\n{current}",
            )
            calls.append(call)
            guidance = f"\n\nDiversity guidance:\n{call.output}"
        # One actor request per round in BOTH branches, with the same instruction.
        revised = invoke(
            f"{branch}.{index}.repair", "actor", actor, system,
            f"Task:\n{task}\n\nCurrent response:\n{current}{guidance}\n\n"
            "Check the response for mistakes and alternatives. Produce one revised final response.",
        )
        calls.append(revised)
        current = revised.output
    return StrategyTrace(branch + "_multiround_v1", tuple(calls), current)


def validate_record(value):
    keys = {"protocol", "run_id", "attempt_id", "rounds", "state", "actor_only", "guided"}
    strict_keys(value, keys, keys, "PairedRecord")
    if value["protocol"] != PROTOCOL or type(value["rounds"]) is not int or value["rounds"] < 1:
        raise ValueError("Unsupported paired protocol/budget")
    state = DecisionState.from_dict(value["state"])
    if state.task.split not in {"train", "val"}:
        raise ValueError("Collection excludes test tasks")
    for branch, expected in (("actor_only", ["actor"]), ("guided", ["preference", "actor"])):
        trace = StrategyTrace.from_dict(value[branch])
        if (trace.policy_version != branch + "_multiround_v1"
                or [c.role for c in trace.calls] != expected * value["rounds"]
                or trace.final_output != trace.calls[-1].output):
            raise ValueError("Invalid branch roles, budget or final answer")
    return value


def collect_pair(task, actor, preference, invoke, rounds, run_id, attempt_id):
    state = execute_initial(task, actor, invoke)
    a = execute_branch(state, actor, preference, invoke, rounds=rounds, guided=False)
    b = execute_branch(state, actor, preference, invoke, rounds=rounds, guided=True)
    value = json.loads(json.dumps({"protocol": PROTOCOL, "run_id": run_id,
        "attempt_id": attempt_id, "rounds": rounds, "state": asdict(state),
        "actor_only": asdict(a), "guided": asdict(b)}))
    return validate_record(value)


def load_settings(path):
    value = json.loads(path.read_text())
    keys = {"actor", "preference_models", "rounds", "seed", "max_per_domain", "split"}
    strict_keys(value, keys, keys, "PairedSettings")
    for key in ("rounds", "max_per_domain"):
        if type(value[key]) is not int or value[key] < 1:
            raise ValueError(f"Invalid {key}")
    if type(value["seed"]) is not int or value["seed"] < 0 or value["split"] not in {"train", "val"}:
        raise ValueError("Invalid seed/split")
    actor = ModelConfig.from_dict(value["actor"])
    preferences = {d: ModelConfig.from_dict(m) for d, m in value["preference_models"].items()}
    if any(m.revision.startswith("REPLACE_") for m in [actor, *preferences.values()]):
        raise ValueError("Replace model revisions before collection")
    return value, actor, preferences


def prepare_pool(root, output, seed=42):
    """Recover full source pools, not a whitelist of old router experiment IDs."""
    from .datasets import load_code_rows, load_math_rows, load_qa_rows, find_qa_context_files
    from split_protocol import content_group_id
    import unicodedata

    def norm(s):
        return " ".join(unicodedata.normalize("NFKC", str(s)).lower().split())

    def read(path):
        return json.loads(path.read_text()) if path.suffix == ".json" else list(read_objects(path))

    sources = {"code": "dataset/code/train/code.json", "math": "dataset/math/train/math.json",
               "qa": "dataset/multihop/train/sft.jsonl"}
    tests = {"code": ("dataset/code/test/humaneval.jsonl", "prompt"),
             "math": ("samples_math_150/sample_math_150.jsonl", "problem"),
             "qa": ("dataset/multihop/test/hotpot_qa.jsonl", "question")}
    task_rows, supervision_rows, audit = [], [], {}
    for domain, source in sources.items():
        original = read(root / source)
        heldout_path, field = tests[domain]
        heldout = {norm(r[field]) for r in read(root / heldout_path)}
        rows = []
        # The legacy loader partitions its input. Combine ALL partitions before
        # applying our new grouping so old router pools never restrict the source.
        for old_split in ("train", "validation", "test"):
            if domain == "code":
                part, _ = load_code_rows(root / source, seed, old_split)
            elif domain == "math":
                part, _ = load_math_rows(root / source, seed, old_split)
            else:
                part, _ = load_qa_rows(root / source, find_qa_context_files(root, None), seed, old_split)
            rows.extend(part)
        retained, overlap, missing_gold = {}, 0, 0
        for row in rows:
            if domain != "code" and (not isinstance(row.get("gold"), str) or not row["gold"].strip()):
                missing_gold += 1
                continue
            raw = original[row["source_index"]]
            text = raw["problem"] if domain == "math" else raw["question"]
            if norm(text) in heldout or (domain == "code" and norm(raw.get("func_sign", "")) in heldout):
                overlap += 1
                continue
            gid = content_group_id(text)
            sid = f"{domain}:{gid}"
            split = "train" if int(hashlib.sha256(f"{seed}:{gid}".encode()).hexdigest()[:8], 16) % 10000 < 9000 else "val"
            task = TaskSample.from_dict({"sample_id": sid, "group_id": gid, "domain": domain,
                "problem": row["problem"], "context": row.get("context", ""), "split": split, "split_seed": seed})
            supervision = OfflineSupervision.from_dict({"sample_id": sid, "group_id": gid, "domain": domain,
                "gold": row.get("gold"), "tests": ({**row["tests"],
                    "output_format": "apps_singleton_wrapper"} if domain == "code" else None)})
            retained.setdefault(sid, (asdict(task), asdict(supervision)))
        task_rows.extend(t for t, _ in retained.values())
        supervision_rows.extend(s for _, s in retained.values())
        audit[domain] = {"source_rows": len(original), "loader_rows": len(rows),
            "eligible_groups": len(retained), "exact_test_overlap_removed": overlap,
            "missing_gold_removed": missing_gold,
            "source_sha256": file_sha256(root / source), "test_sha256": file_sha256(root / heldout_path),
            "train": sum(t["split"] == "train" for t, _ in retained.values()),
            "val": sum(t["split"] == "val" for t, _ in retained.values())}
    output.mkdir(parents=True, exist_ok=False)
    for row in task_rows:
        append_jsonl(output / "tasks.jsonl", row)
    for row in supervision_rows:
        append_jsonl(output / "supervision.jsonl", row)
    write_json_atomic(output / "audit.json", {"seed": seed, "ratio": "random 90:10 by question group",
        "domains": audit, "note": "SFT/DPO overlap permitted; normalized exact test exclusion only"})
    return audit


def collection_config(tasks_path, settings_path, *, local_preference=False):
    settings, actor, preferences = load_settings(settings_path)
    all_tasks = [TaskSample.from_dict(v) for v in read_objects(tasks_path)]
    seen, groups = set(), {}
    for task in all_tasks:
        if task.sample_id in seen or groups.get(task.group_id, task.split) != task.split:
            raise ValueError("Duplicate task or group across splits")
        if task.split_seed != settings["seed"]:
            raise ValueError("Task split seed differs from settings")
        seen.add(task.sample_id)
        groups[task.group_id] = task.split
    tasks = select_tasks(all_tasks, split=settings["split"], seed=settings["seed"],
                         max_per_domain=settings["max_per_domain"])
    if any(t.domain not in preferences for t in tasks):
        raise ValueError("Missing domain preference model")
    identity = {"protocol": PROTOCOL, "settings": settings, "local_preference": local_preference,
        "tasks_sha256": file_sha256(tasks_path), "selected_tasks": [asdict(t) for t in tasks],
        "code_sha256": {p.name: file_sha256(p) for p in Path(__file__).parent.glob("*.py")},
        "storage_format": STORAGE_FORMAT,
        "endpoint_identity": "declared; verify deployed weights separately"}
    return settings, actor, preferences, tasks, identity


def collect(tasks_path, settings_path, output, *, execute=False, local_preference=False):
    settings, actor, preferences, tasks, identity = collection_config(
        tasks_path, settings_path, local_preference=local_preference)
    if not execute:
        return {"dry_run": True, "selected_total": len(tasks), "rounds": settings["rounds"],
                "calls_per_task": 1 + 3 * settings["rounds"]}
    if (output / ".storage-migration.pending.json").exists():
        raise ValueError("Storage migration incomplete; finish explicit recovery before resuming")
    preference_client = None
    if local_preference:
        from .local_preference import LocalPreference
        preference_client = LocalPreference()
    with TraceRecorder(output, identity) as recorder:
        checkpoints = output / "calls"
        checkpoints.mkdir(exist_ok=True)
        completed = {}
        path = output / "trajectories.jsonl"
        selected = {t.sample_id: t for t in tasks}
        if path.exists():
            for value in read_objects(path):
                validate_record(value)
                sid = value["state"]["task"]["sample_id"]
                if (sid in completed or sid not in selected or value["run_id"] != recorder.run_id
                        or value["state"]["task"] != asdict(selected[sid])):
                    raise ValueError("Completed row does not belong to this run")
                completed[sid] = value
        for task in tasks:
            if task.sample_id in completed:
                continue
            attempt = uuid.uuid4().hex

            def invoke(stage, role, model, system, user):
                key = hashlib.sha256(f"{task.sample_id}:{stage}".encode()).hexdigest()
                checkpoint = checkpoints / (key + ".json")
                request = {"sample_id": task.sample_id, "stage": stage, "role": role,
                           "model": asdict(model), "system": system, "user": user}
                if checkpoint.exists():
                    cached = json.loads(checkpoint.read_text())
                    if cached["request"] != request:
                        raise ValueError("Checkpoint request mismatch")
                    return ModelCall.from_dict(cached["call"])
                # Matched sampling seed for A/B actor repairs at the same round.
                seed_stage = stage.replace("actor_only.", "paired.").replace("guided.", "paired.")
                seed = call_seed(task.sample_id, settings["seed"], seed_stage)
                if role == "preference" and preference_client is not None:
                    result = preference_client(model, system, user, seed)
                    recorder.event("local_preference.completed", sample_id=task.sample_id,
                        attempt_id=attempt, stage=stage,
                        data=compact_event_data({"request": request, "result": result}))
                else:
                    result = call_actor(base_url=model.endpoint, model=model.model_id,
                        system=system, user=user, temperature=model.temperature, max_tokens=model.max_tokens,
                        seed=seed, timeout=240, retries=1,
                        top_p=model.top_p, enable_thinking=model.enable_thinking,
                        compact=True,
                        trace=lambda event, data: recorder.event(event, sample_id=task.sample_id,
                            attempt_id=attempt, stage=stage, data=data))
                call = ModelCall(role, model.model_id, model.revision, compact_result(result))
                write_json_atomic(checkpoint, {"request": request, "call": asdict(call)})
                return call

            record = collect_pair(task, actor, preferences[task.domain], invoke,
                                  settings["rounds"], recorder.run_id, attempt)
            append_jsonl(path, record)
            completed[task.sample_id] = record
            write_json_atomic(output / "manifest.json", {"protocol": PROTOCOL,
                "status": "complete" if len(completed) == len(tasks) else "running",
                "selected_total": len(tasks), "completed_total": len(completed)})
            print(f"[{len(completed)}/{len(tasks)}] {task.domain} {task.sample_id}", flush=True)
    return {"status": "complete", "completed_total": len(completed)}


def score_pair(value, supervision):
    from .offline import _cost, _score

    validate_record(value)
    state = DecisionState.from_dict(value["state"])
    task = state.task
    if (supervision.sample_id, supervision.group_id, supervision.domain) != (task.sample_id, task.group_id, task.domain):
        raise ValueError("Supervision identity mismatch")
    traces = {k: StrategyTrace.from_dict(value[k]) for k in ("actor_only", "guided")}
    exclusions = [f"initial:{f}" for f in state.initial.quality_flags]
    scores, costs = {}, {}
    initial_score, initial_error = _score(supervision, state.initial.output)
    cost_config = LabelConfig("binary_success_net_gain_v1", "legacy_deterministic_v1", 0, 0)
    for name, trace in traces.items():
        exclusions.extend(f"{name}.{i}:{f}" for i, c in enumerate(trace.calls) for f in c.quality_flags)
        costs[name], _ = _cost(trace, cost_config)
        per_round = []
        for c in trace.calls:
            if c.role == "actor":
                metrics, error = _score(supervision, c.output)
                per_round.append({"metrics": metrics, "scoring_issue": error})
        scores[name] = per_round
        if per_round[-1]["scoring_issue"]:
            exclusions.append(name + ":" + per_round[-1]["scoring_issue"])
    delta = None if exclusions else (int(scores["guided"][-1]["metrics"]["success"])
                                    - int(scores["actor_only"][-1]["metrics"]["success"]))
    cost_delta = {k: costs["guided"][k] - costs["actor_only"][k]
                  if costs["guided"][k] is not None and costs["actor_only"][k] is not None else None
                  for k in ("total_tokens", "latency_seconds")}
    return {"protocol": PROTOCOL, "label_version": "paired_final_success_gain_v1",
        "sample_id": task.sample_id, "group_id": task.group_id, "domain": task.domain,
        "split": task.split, "label": int(delta > 0) if delta is not None else None,
        "delta_success": delta, "exclusion_reasons": sorted(set(exclusions)),
        "initial_score": initial_score, "initial_scoring_issue": initial_error,
        "round_scores": scores, "branch_costs": costs, "extra_guided_cost": cost_delta,
        "features": state.router_features(), "purpose": "pilot"}


def label(run, supervision_path, output, *, allow_code_execution=False):
    values = list(read_objects(run / "trajectories.jsonl"))
    if not values:
        raise ValueError("No completed paired trajectories")
    supervision = {}
    for v in read_objects(supervision_path):
        item = OfflineSupervision.from_dict(v)
        if item.sample_id in supervision:
            raise ValueError("Duplicate supervision")
        supervision[item.sample_id] = item
    if any(v["state"]["task"]["domain"] == "code" for v in values) and not allow_code_execution:
        raise ValueError("Code scoring requires --allow-code-execution in an isolated environment")
    metadata = json.loads((run / "run.json").read_text())
    selected = {v["sample_id"]: v for v in metadata["config"]["selected_tasks"]}
    seen = set()
    rows = []
    for value in values:
        validate_record(value)
        task = value["state"]["task"]
        sid = task["sample_id"]
        if sid in seen or selected.get(sid) != task or value["run_id"] != metadata["run_id"]:
            raise ValueError("Trajectory run/task identity mismatch")
        if value["rounds"] != metadata["config"]["settings"]["rounds"]:
            raise ValueError("Trajectory budget differs from configured budget")
        seen.add(sid)
        rows.append(score_pair(value, supervision[sid]))
    output.mkdir(parents=True, exist_ok=False)
    for row in rows:
        # Features and labels are exported without gold, branch outputs or rewards.
        append_jsonl(output / "labels.jsonl", {k: v for k, v in row.items() if k != "features"})
        if row["label"] is not None:
            append_jsonl(output / "router_dataset.jsonl", {k: row[k] for k in
                ("sample_id", "group_id", "domain", "split", "features", "label")})
    report = {"protocol": PROTOCOL, "records": len(rows), "selected_total": len(selected),
        "valid": sum(r["label"] is not None for r in rows),
        "positive": sum(r["label"] == 1 for r in rows), "negative": sum(r["label"] == 0 for r in rows),
        "trajectory_sha256": file_sha256(run / "trajectories.jsonl"),
        "supervision_sha256": file_sha256(supervision_path),
        "scorer_sha256": file_sha256(Path(__file__)),
        "verifier_sha256": file_sha256(Path(__file__).with_name("verifiers.py"))}
    write_json_atomic(output / "report.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    pool = sub.add_parser("prepare")
    pool.add_argument("--root", type=Path, required=True)
    pool.add_argument("--output", type=Path, required=True)
    pool.add_argument("--seed", type=int, default=42)
    gen = sub.add_parser("collect")
    gen.add_argument("--tasks", type=Path, required=True)
    gen.add_argument("--settings", type=Path, required=True)
    gen.add_argument("--output", type=Path, required=True)
    gen.add_argument("--execute", action="store_true")
    gen.add_argument("--local-preference", action="store_true", help="Load 2B adapters sequentially as 4bit GPU models")
    score = sub.add_parser("label")
    score.add_argument("--run", type=Path, required=True)
    score.add_argument("--supervision", type=Path, required=True)
    score.add_argument("--output", type=Path, required=True)
    score.add_argument("--allow-code-execution", action="store_true")
    args = parser.parse_args()
    if args.command == "prepare":
        result = prepare_pool(args.root, args.output, args.seed)
    elif args.command == "collect":
        result = collect(args.tasks, args.settings, args.output, execute=args.execute,
                         local_preference=args.local_preference)
    else:
        result = label(args.run, args.supervision, args.output,
                       allow_code_execution=args.allow_code_execution)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
