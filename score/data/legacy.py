"""Historical G/D-oracle pilot, preserved only for diagnosis.

Not the intended binary call/no-call routing protocol. Never use its labels as
formal training data. See docs/gain_router_code_guide.md.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import uuid
from collections import Counter
from pathlib import Path
from typing import Any

from .actor import call_actor
from .common import (DEFAULT_BASE_URL, DEFAULT_MODEL, PROTOCOL_VERSION, SCHEMA_VERSION,
                     append_jsonl, completed_ids, file_sha256, ranked_sample, summarize_completed)
from .datasets import find_qa_context_files, load_code_rows, load_math_rows, load_qa_rows
from .verifiers import verify
from .traces import TraceRecorder, config_fingerprint, write_json_atomic


def sample_seed(sample_id: str, generation_seed: int) -> int:
    """Stable across domain order, subset size and resume."""
    digest = hashlib.sha256(f"{generation_seed}:{sample_id}".encode()).hexdigest()
    return int(digest[:8], 16) % (2**31 - 2)

def base_task_prompt(row: dict[str, Any]) -> tuple[str, str]:
    if row["domain"] == "math":
        return (
            "You are a careful mathematical problem solver. Show concise reasoning and end with exactly one line: FINAL_ANSWER: <answer>.",
            row["problem"],
        )
    if row["domain"] == "qa":
        return (
            "Answer the multi-hop question using only the supplied context. Reason concisely and end with exactly one line: FINAL_ANSWER: <short answer>.",
            f"Context:\n{row['context']}\n\nQuestion:\n{row['problem']}",
        )
    return (
        "You are an expert Python programmer. Return one complete Python function implementation. Do not include tests or prose outside a Python code block.",
        row["problem"],
    )


def repair_prompt(
    row: dict[str, Any],
    initial_output: str,
    feedback: str,
    diverse: bool,
) -> tuple[str, str]:
    system, task = base_task_prompt(row)
    if diverse:
        instruction = (
            "Use the following preference-aligned failure guidance to explore a distinct correction, "
            "check multiple plausible failure modes, and then produce one final repaired answer.\n\n"
            f"Failure guidance:\n{row.get('guidance') or '[no stored guidance]'}"
        )
    else:
        instruction = (
            "Perform a standard low-cost self-correction. Inspect the verifier feedback, identify the "
            "most likely error, and produce one repaired answer."
        )
    user = (
        f"Original task:\n{task}\n\nInitial response:\n{initial_output}\n\n"
        f"Verifier feedback:\n{feedback}\n\n{instruction}"
    )
    return system, user
def derive_case(initial_success: bool, generic_success: bool | None, diverse_success: bool | None) -> tuple[str, int | None]:
    if initial_success:
        return "initial_success", None
    if generic_success is None or diverse_success is None:
        raise ValueError("failed initial attempts require both counterfactual branches")
    if diverse_success and not generic_success:
        return "strict_positive_gain", 1
    if generic_success and diverse_success:
        return "redundant_intervention", 0
    if generic_success and not diverse_success:
        return "negative_transfer", 0
    return "shared_failure", 0


def trajectory_payload(call: dict[str, Any], metrics: dict[str, Any]) -> dict[str, Any]:
    return {**call, "verifier": metrics, "reward": metrics["reward"], "success": metrics["success"]}


def generate_one(
    row: dict[str, Any],
    args: argparse.Namespace,
    ordinal: int = 0,
    *,
    recorder: TraceRecorder | None = None,
    attempt_id: str | None = None,
) -> dict[str, Any]:
    """Execute the historical pilot and persist every call before verification.

    ordinal is accepted for compatibility but no longer controls sample seeds.
    This function does not implement the intended call/no-call protocol.
    """
    attempt_id = attempt_id or uuid.uuid4().hex
    seed_base = sample_seed(row["sample_id"], args.generation_seed)

    def emit(event: str, stage: str, data: dict[str, Any]) -> None:
        if recorder:
            recorder.event(event, sample_id=row["sample_id"],
                           attempt_id=attempt_id, stage=stage, data=data)

    def execute(stage: str, prompts: tuple[str, str], temperature: float, seed: int) -> dict[str, Any]:
        call = call_actor(
            base_url=args.base_url, model=args.model,
            system=prompts[0], user=prompts[1],
            temperature=temperature, max_tokens=args.max_tokens,
            seed=seed, timeout=args.request_timeout, retries=args.retries,
            trace=lambda event, data: emit(event, stage, data),
        )
        # Raw response is already durable even if the verifier crashes here.
        metrics = verify(row, call["output"])
        emit("verification.completed", stage, {"offline_verifier": metrics})
        return trajectory_payload(call, metrics)

    emit("sample.started", "sample", {"domain": row["domain"],
                                     "group_id": row["group_id"], "split": row["split"]})
    initial = execute("initial", base_task_prompt(row), args.initial_temperature, seed_base)
    generic = diverse = None
    if not initial["success"]:
        generic = execute(
            "generic",
            repair_prompt(row, initial["output"], initial["verifier"]["feedback"], diverse=False),
            args.repair_temperature, seed_base + 1,
        )
        diverse = execute(
            "diverse",
            repair_prompt(row, initial["output"], initial["verifier"]["feedback"], diverse=True),
            args.repair_temperature, seed_base + 1,
        )
    case_type, label = derive_case(
        initial["success"],
        generic["success"] if generic else None,
        diverse["success"] if diverse else None,
    )
    record = {key: value for key, value in row.items()
              if key not in {"gold", "tests", "guidance"}}
    record.update({
        "protocol_version": PROTOCOL_VERSION,
        "run_id": recorder.run_id if recorder else None,
        "attempt_id": attempt_id,
        "valid_for_training": False,
        "exclusion_reasons": ["legacy_gd_oracle_pilot"],
        "actor_model": args.model,
        "initial": initial,
        "generic": generic,
        "diverse": diverse,
        "outcome": {
            "case_type": case_type, "label": label,
            "generic_reward": generic["reward"] if generic else None,
            "diverse_reward": diverse["reward"] if diverse else None,
            "delta_reward": diverse["reward"] - generic["reward"] if generic and diverse else None,
        },
        "offline_supervision": {
            "gold": row.get("gold"), "tests": row.get("tests"),
            "diverse_guidance": row.get("guidance"),
        },
        "created_at_unix": time.time(),
    })
    return record


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".")
    parser.add_argument("--domains", nargs="+", choices=("code", "math", "qa"), default=("code", "math", "qa"))
    parser.add_argument("--samples-per-domain", type=int, default=100)
    parser.add_argument("--split", choices=("train", "validation", "test"), default="train")
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--generation-seed", type=int, default=42000)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--output-dir", default="outputs/gain_router_legacy/pilot_seed42")
    parser.add_argument("--allow-legacy-pilot", action="store_true",
                        help="Explicitly acknowledge G/D oracle pilot; NOT formal call/no-call training data")
    parser.add_argument("--code-file", default="dataset/code/train/apps_original.json")
    parser.add_argument("--math-file", default="dataset/math/train/math.json")
    parser.add_argument("--qa-file", default="dataset/multihop/train/sft.jsonl")
    parser.add_argument(
        "--qa-context-file",
        action="append",
        help="repeat for each exact-match multihop context JSONL; defaults to raw 2Wiki/HotpotQA/MuSiQue train sources",
    )
    parser.add_argument("--initial-temperature", type=float, default=0.2)
    parser.add_argument("--repair-temperature", type=float, default=0.4)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--request-timeout", type=float, default=180.0)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.dry_run and not args.allow_legacy_pilot:
        parser.error("Generation disabled: legacy G/D pilot is not the final binary protocol. "
                     "For diagnosis only, pass --allow-legacy-pilot and use a new output directory.")
    root = Path(args.root).resolve()
    output_dir = (root / args.output_dir).resolve()
    qa_context_files: list[Path] = []
    all_rows: list[dict[str, Any]] = []
    source_stats: dict[str, Any] = {}
    source_paths: list[Path] = []

    if "code" in args.domains:
        path = root / args.code_file
        rows, stats = load_code_rows(path, args.split_seed, args.split)
        chosen = ranked_sample(rows, args.generation_seed, args.samples_per_domain)
        all_rows.extend(chosen)
        source_stats["code"] = {**stats, "selected": len(chosen)}
        source_paths.append(path)
    if "math" in args.domains:
        path = root / args.math_file
        rows, stats = load_math_rows(path, args.split_seed, args.split)
        chosen = ranked_sample(rows, args.generation_seed, args.samples_per_domain)
        all_rows.extend(chosen)
        source_stats["math"] = {**stats, "selected": len(chosen)}
        source_paths.append(path)
    if "qa" in args.domains:
        path = root / args.qa_file
        qa_context_files = find_qa_context_files(root, args.qa_context_file)
        rows, stats = load_qa_rows(path, qa_context_files, args.split_seed, args.split)
        chosen = ranked_sample(rows, args.generation_seed, args.samples_per_domain)
        all_rows.extend(chosen)
        source_stats["qa"] = {
            **stats,
            "selected": len(chosen),
            "context_files": [str(value) for value in qa_context_files],
        }
        source_paths.append(path)
        source_paths.extend(qa_context_files)

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "protocol_version": PROTOCOL_VERSION,
        "status": "dry_run" if args.dry_run else "running",
        "pilot_only": True,
        "non_oof_diverse_guidance": True,
        "label_definition": "1 iff initial fails, D succeeds, and G fails; initial successes have null label",
        "comparison": "single matched G attempt versus single matched D attempt",
        "actor_model": args.model,
        "base_url": args.base_url,
        "split_seed": args.split_seed,
        "split": args.split,
        "generation_seed": args.generation_seed,
        "sampling": {
            "initial_temperature": args.initial_temperature,
            "repair_temperature": args.repair_temperature,
            "max_tokens": args.max_tokens,
        },
        "verifiers": {
            "code": "resource-limited subprocess (not a sandbox); visible-test pass rate",
            "math": "normalized exact or simple numeric equivalence; no LLM judge",
            "qa": "multihop normalized exact match; token F1 retained",
        },
        "source_stats": source_stats,
        "source_sha256": {str(path): file_sha256(path) for path in source_paths},
        "selected_total": len(all_rows),
        "created_at_unix": time.time(),
    }
    print(json.dumps({"manifest": manifest, "output_dir": str(output_dir)}, ensure_ascii=False, indent=2))
    if args.dry_run:
        # A dry run must never overwrite metadata from a real experiment.
        return 0
    return run_samples(args, all_rows, manifest, output_dir)


def run_samples(args: argparse.Namespace, rows: list[dict[str, Any]],
                manifest: dict[str, Any], output_dir: Path) -> int:
    package_dir = Path(__file__).resolve().parent
    code_hashes = {p.name: file_sha256(p) for p in sorted(package_dir.glob("*.py"))}
    config = {
        "protocol_version": PROTOCOL_VERSION,
        "arguments": {key: value for key, value in vars(args).items()
                      if key not in {"output_dir", "dry_run", "allow_legacy_pilot"}},
        "source_sha256": manifest["source_sha256"],
        # Includes actual selected tests/guidance, not only source-array paths.
        "selected_input_sha256": config_fingerprint({"rows": rows}),
        "source_code_sha256": code_hashes,
        "model_weight_identity": "unverified; pilot only",
    }
    output_file = output_dir / "trajectories.jsonl"
    error_file = output_dir / "errors.jsonl"
    with TraceRecorder(output_dir, config) as recorder:
        manifest["run_id"] = recorder.run_id
        manifest["config_fingerprint"] = recorder.fingerprint
        write_json_atomic(output_dir / "manifest.json", manifest)
        done = completed_ids(output_file)
        cases = Counter()
        selected_ids = {row["sample_id"] for row in rows}
        errors = 0
        for row in rows:
            if row["sample_id"] in done:
                continue
            attempt_id = uuid.uuid4().hex
            try:
                record = generate_one(row, args, recorder=recorder, attempt_id=attempt_id)
                append_jsonl(output_file, record)
                done.add(row["sample_id"])
                cases[record["outcome"]["case_type"]] += 1
                recorder.event("sample.completed", sample_id=row["sample_id"],
                               attempt_id=attempt_id, stage="sample",
                               data={"outcome": record["outcome"], "valid_for_training": False})
                print(f"[{len(done & selected_ids)}/{len(rows)}] {row['domain']} "
                      f"{row['sample_id'][:20]} => {record['outcome']['case_type']}", flush=True)
            except Exception as exc:
                error = {
                    "run_id": recorder.run_id, "attempt_id": attempt_id,
                    "sample_id": row["sample_id"], "domain": row["domain"],
                    "error_type": type(exc).__name__, "error": str(exc),
                    "created_at_unix": time.time(),
                }
                # Disk/journal failures must stop immediately.
                if isinstance(exc, OSError):
                    raise
                append_jsonl(error_file, error)
                recorder.event("sample.failed", sample_id=row["sample_id"],
                               attempt_id=attempt_id, stage="sample", data=error)
                errors += 1
                print(f"ERROR {row['sample_id']}: {exc}", file=sys.stderr, flush=True)
        manifest.update(summarize_completed(output_file))
        manifest["status"] = "complete" if selected_ids <= done else "partial"
        manifest["new_case_counts"] = dict(cases)
        manifest["new_error_count"] = errors
        manifest["finished_at_unix"] = time.time()
        write_json_atomic(output_dir / "manifest.json", manifest)
        return 0 if manifest["status"] == "complete" else 1
