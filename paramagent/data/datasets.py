"""Legacy pilot source joins. Shared manifest application is not implemented here."""
from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from .common import SCHEMA_VERSION, load_json_array, normalize_space, read_jsonl, split_for_group, stable_id

def format_context(context: Any, max_chars: int = 14_000) -> str:
    chunks: list[str] = []
    if isinstance(context, dict):
        titles = context.get("title") or context.get("titles") or []
        sentences = context.get("sentences") or context.get("sentence") or []
        if isinstance(titles, list) and isinstance(sentences, list):
            for title, sentence_group in zip(titles, sentences):
                if isinstance(sentence_group, list):
                    body = " ".join(str(item) for item in sentence_group)
                else:
                    body = str(sentence_group)
                chunks.append(f"[{title}] {body}")
        else:
            chunks.append(json.dumps(context, ensure_ascii=False))
    elif isinstance(context, list):
        for item in context:
            if isinstance(item, (list, tuple)) and len(item) >= 2:
                body = item[1]
                if isinstance(body, list):
                    body = " ".join(str(part) for part in body)
                chunks.append(f"[{item[0]}] {body}")
            elif isinstance(item, dict):
                title = item.get("title") or item.get("idx") or item.get("id") or "paragraph"
                body = (
                    item.get("paragraph_text")
                    or item.get("text")
                    or item.get("sentences")
                    or item.get("paragraph")
                    or ""
                )
                if isinstance(body, list):
                    body = " ".join(str(part) for part in body)
                if body:
                    chunks.append(f"[{title}] {body}")
            else:
                chunks.append(str(item))
    elif context:
        chunks.append(str(context))
    return "\n".join(chunks)[:max_chars]


def find_qa_context_files(root: Path, explicit: list[str] | None) -> list[Path]:
    if explicit:
        paths = [Path(value).resolve() for value in explicit]
    else:
        # The SFT pool is a union of 2Wiki, HotpotQA and MuSiQue. Join it to the
        # raw training contexts, never to generated-reflection/DPO rows (which
        # may come from a different split and would create an invalid mapping).
        paths = [
            root / "dataset/multihop/meta/2w/2w-train.jsonl",
            root / "dataset/multihop/meta/hotpot/full_train.jsonl",
            root / "dataset/multihop/meta/mus/mus_train.jsonl",
        ]
    if not paths:
        raise FileNotFoundError(
            "no raw multihop context JSONL found; pass one or more --qa-context-file values"
        )
    missing = [path for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"QA context files do not exist: {missing}")
    return paths


def load_math_rows(path: Path, split_seed: int, split: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows = []
    for source_index, item in enumerate(load_json_array(path)):
        problem = item.get("problem") or item.get("question")
        answer = item.get("answer")
        if not problem or answer is None:
            continue
        group_id = stable_id("math", item.get("unique_id") or problem)
        assigned = split_for_group(group_id, split_seed)
        if assigned != split:
            continue
        rows.append(
            {
                "schema_version": SCHEMA_VERSION,
                "sample_id": group_id,
                "group_id": group_id,
                "domain": "math",
                "source_dataset": path.name,
                "source_index": source_index,
                "split_seed": split_seed,
                "split": assigned,
                "problem": str(problem),
                "gold": str(answer),
                "context": "",
                "guidance": str(item.get("pitfalls") or ""),
                "source_meta": {
                    "unique_id": item.get("unique_id"),
                    "subject": item.get("subject") or item.get("type"),
                    "level": item.get("level"),
                },
            }
        )
    return rows, {"source_total": len(load_json_array(path)), "eligible": len(rows)}


def load_code_rows(path: Path, split_seed: int, split: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    source = load_json_array(path)
    meta_root = path.parents[1] / "meta" / "train"
    meta_by_question: dict[str, dict[str, Any]] = {}
    duplicate_meta_questions = 0
    if meta_root.is_dir():
        for problem_dir in sorted(meta_root.iterdir()):
            question_path = problem_dir / "question.txt"
            io_path = problem_dir / "input_output.json"
            if not problem_dir.is_dir() or not question_path.is_file() or not io_path.is_file():
                continue
            try:
                question = question_path.read_text(encoding="utf-8")
                io_data = json.loads(io_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError, UnicodeDecodeError):
                continue
            question_key = normalize_space(question).lower()
            if not question_key or question_key in meta_by_question:
                duplicate_meta_questions += 1
                continue
            starter_path = problem_dir / "starter_code.py"
            try:
                starter_code = starter_path.read_text(encoding="utf-8") if starter_path.is_file() else ""
            except (OSError, UnicodeDecodeError):
                starter_code = ""
            meta_by_question[question_key] = {
                "problem_id": problem_dir.name,
                "io_data": io_data,
                "starter_code": starter_code,
                "meta_dir": str(problem_dir),
            }
    rows = []
    excluded = Counter()
    matched_meta = 0
    for source_index, item in enumerate(source):
        problem = item.get("question")
        meta = meta_by_question.get(normalize_space(problem).lower()) if problem else None
        signature = item.get("func_sign") or item.get("starter_code") or (meta or {}).get("starter_code")
        # The backed-up APPS derivative uses the literal key ``input/output``;
        # upstream APPS exports commonly use ``input_output``. Support both.
        io_data = item.get("input_output") or item.get("input/output")
        if not io_data and meta:
            io_data = meta.get("io_data")
            matched_meta += 1
        if isinstance(io_data, str):
            try:
                io_data = json.loads(io_data)
            except json.JSONDecodeError:
                io_data = None
        if not problem or not signature:
            excluded["missing_problem_or_signature"] += 1
            continue
        if not isinstance(io_data, dict) or not io_data.get("inputs") or not io_data.get("outputs"):
            excluded["missing_tests"] += 1
            continue
        fn_name = io_data.get("fn_name")
        if not fn_name:
            match = re.search(r"\bdef\s+([A-Za-z_]\w*)\s*\(", str(signature))
            fn_name = match.group(1) if match else None
        if not fn_name:
            excluded["missing_function_name"] += 1
            continue
        problem_id = item.get("problem_id") or (meta or {}).get("problem_id")
        group_id = stable_id("code", problem_id or problem)
        assigned = split_for_group(group_id, split_seed)
        if assigned != split:
            continue
        rows.append(
            {
                "schema_version": SCHEMA_VERSION,
                "sample_id": group_id,
                "group_id": group_id,
                "domain": "code",
                "source_dataset": path.name,
                "source_index": source_index,
                "split_seed": split_seed,
                "split": assigned,
                "problem": f"{problem}\n\nRequired function:\n{signature}",
                "gold": "",
                "context": "",
                "guidance": str(item.get("pitfalls") or ""),
                "tests": {
                    "fn_name": fn_name,
                    "inputs": io_data.get("inputs", []),
                    "outputs": io_data.get("outputs", []),
                },
                "source_meta": {
                    "problem_id": problem_id,
                    "difficulty": item.get("difficulty"),
                    "url": item.get("url"),
                    "raw_meta_dir": (meta or {}).get("meta_dir"),
                },
            }
        )
    return rows, {
        "source_total": len(source),
        "eligible": len(rows),
        "excluded": dict(excluded),
        "meta_root": str(meta_root),
        "meta_questions_indexed": len(meta_by_question),
        "duplicate_meta_questions": duplicate_meta_questions,
        "matched_meta_rows": matched_meta,
    }


def load_qa_rows(
    train_path: Path,
    context_paths: list[Path],
    split_seed: int,
    split: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    train_rows = list(read_jsonl(train_path))
    train_by_question: dict[str, list[tuple[int, dict[str, Any]]]] = {}
    for index, item in enumerate(train_rows):
        question = normalize_space(item.get("question")).lower()
        if question:
            train_by_question.setdefault(question, []).append((index, item))

    context_by_question: dict[str, dict[str, Any]] = {}
    duplicate_contexts = 0
    for context_path in context_paths:
        for item in read_jsonl(context_path):
            question = normalize_space(item.get("question")).lower()
            if not question or question not in train_by_question:
                continue
            raw_context = item.get("context") or item.get("paragraphs")
            context = format_context(raw_context)
            if not context:
                continue
            if question in context_by_question:
                duplicate_contexts += 1
                continue
            context_by_question[question] = {
                **item,
                "_raw_context": raw_context,
                "_context_source": str(context_path),
            }

    rows = []
    for question_key, matches in train_by_question.items():
        if len(matches) != 1:
            continue
        context_item = context_by_question.get(question_key)
        if not context_item:
            continue
        source_index, train_item = matches[0]
        problem = str(train_item["question"])
        answer = train_item.get("answer")
        if answer is None:
            continue
        group_id = stable_id("qa", context_item.get("id") or problem)
        assigned = split_for_group(group_id, split_seed)
        if assigned != split:
            continue
        guidance = train_item.get("decomposition") or ""
        if not guidance:
            for key in (
                "insights",
                "pitfalls",
                "pitfalls_high_temp",
                "high_temp_pitfall",
                "decomposition",
            ):
                if context_item.get(key):
                    guidance = context_item[key]
                    break
        rows.append(
            {
                "schema_version": SCHEMA_VERSION,
                "sample_id": group_id,
                "group_id": group_id,
                "domain": "qa",
                "source_dataset": train_path.name,
                "source_index": source_index,
                "split_seed": split_seed,
                "split": assigned,
                "problem": problem,
                "gold": str(answer),
                "context": format_context(context_item.get("_raw_context")),
                "guidance": str(guidance),
                "source_meta": {
                    "context_source": context_item.get("_context_source"),
                    "context_id": context_item.get("id"),
                    "type": context_item.get("type"),
                    "level": context_item.get("level"),
                },
            }
        )
    stats = {
        "source_total": len(train_rows),
        "unique_train_questions": len(train_by_question),
        "matched_context_questions": len(context_by_question),
        "duplicate_context_rows": duplicate_contexts,
        "eligible": len(rows),
    }
    return rows, stats
