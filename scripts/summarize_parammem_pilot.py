#!/usr/bin/env python3
"""Summarize all nine frozen pilot runs, including legacy Stage 2 results."""
import argparse
import collections
import json
from pathlib import Path

from run_parammem_pilot import read_rows, stage_results, qa_score


def summarize(root):
    groups, tasks = [], []
    for domain in ("code", "math", "qa"):
        for method in ("qwen", "parammem", "parammem-dpo"):
            directory = root / "results" / domain / method
            if not (directory / "completed.json").exists():
                raise RuntimeError(f"Run has not completed: {domain}/{method}")
            first, final = stage_results(directory)
            if len(first) != 10:
                raise RuntimeError(f"Expected ten tasks: {directory}")
            log = root / f"{domain}-{method}.log"
            if log.exists():
                errors = [line for line in log.read_text().splitlines() if any(s in line for s in
                    ("Error in item ", "Exception in ", "Traceback (most recent call last)", "ERROR: Failed to match"))]
                if errors:
                    raise RuntimeError(f"Runner errors require inspection: {directory}: {errors}")
            calls = read_rows(directory / "calls.jsonl")
            consumed = {c["key"] for c in calls if c["type"] == "sample_use"}
            if consumed:
                calls = [c for c in calls if c["type"] not in {"actor", "memory"} or c["key"] in consumed]
            evaluations = collections.defaultdict(list)
            actor_tokens = memory_tokens = 0
            finish = collections.Counter()
            for call in calls:
                if call["type"] == "evaluation":
                    if call["result"]["success"] is None:
                        raise RuntimeError("Unresolved evaluation")
                    evaluations[call.get("task_id")].append(call)
                elif call["type"] == "actor":
                    actor_tokens += call["response"].get("usage", {}).get("completion_tokens", 0)
                    finish.update("actor_" + str(c.get("finish_reason")) for c in call["response"]["choices"])
                elif call["type"] == "memory":
                    memory_tokens += call["completion_tokens"]
                    finish["memory_" + str(call["finish_reason"])] += 1
            first_by_id = {r["task_id"]: r for r in first}
            for row in final:
                task_id = row["task_id"]
                if not evaluations[task_id]:
                    raise RuntimeError(f"Task lacks scored attempts: {directory}/{task_id}")
                initial = evaluations[task_id][0]["result"]
                task = {"domain": domain, "method": method, "task_id": task_id,
                        "first_attempt": initial["success"],
                        "stage1": first_by_id[task_id]["is_solved"], "final": row["is_solved"],
                        "stage2": bool(row.get("stage2")), "evaluated_attempts": len(evaluations[task_id])}
                if domain == "qa":
                    task["initial_f1"] = initial["f1"]
                    task["final_f1"] = qa_score(row["solution"], row["answer"])["f1"]
                tasks.append(task)
            subset = [t for t in tasks if t["domain"] == domain and t["method"] == method]
            groups.append({"domain": domain, "method": method, "n": len(subset),
                           **{k: sum(bool(t[k]) for t in subset) for k in ("first_attempt", "stage1", "final")},
                           "actor_completion_tokens": actor_tokens, "memory_completion_tokens": memory_tokens,
                           "finish_reasons": dict(finish),
                           **({"seeded_cache": json.loads((directory / "cache-origin.json").read_text()),
                               "token_accounting": ("Consumed cached/new responses; not the cost of new inference alone." if consumed else
                                                    "Logged responses include all seeded cache entries; some may not be used by the repaired replay.")}
                              if (directory / "cache-origin.json").exists() else {}),
                           **({"final_mean_f1": sum(t["final_f1"] for t in subset)/len(subset)} if domain == "qa" else {})})
    pairs = []
    for domain in ("code", "math", "qa"):
        by_method = {m: {t["task_id"]: t for t in tasks if t["domain"] == domain and t["method"] == m}
                     for m in ("qwen", "parammem", "parammem-dpo")}
        if any(set(v) != set(by_method["qwen"]) for v in by_method.values()):
            raise RuntimeError("Methods used different task IDs")
        for control in ("qwen", "parammem"):
            for stage in ("first_attempt", "stage1", "final"):
                improved = [i for i, t in by_method["parammem-dpo"].items() if t[stage] and not by_method[control][i][stage]]
                regressed = [i for i, t in by_method["parammem-dpo"].items() if not t[stage] and by_method[control][i][stage]]
                pairs.append({"domain": domain, "control": control, "stage": stage,
                              "dpo_improved": improved, "dpo_regressed": regressed})
    return {"protocol": "oracle-feedback implementation diagnostic; 10 frozen tasks per domain",
            "groups": groups, "tasks": tasks, "dpo_pairs": pairs}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    result = summarize(args.root)
    output = args.root / "summary.json"
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    for row in result["groups"]:
        print(row["domain"], row["method"], "initial/stage1/final", *(row[k] for k in ("first_attempt", "stage1", "final")))
    print(output)


if __name__ == "__main__":
    main()
