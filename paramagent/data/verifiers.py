"""Offline pilot verifiers; these are not inference-visible QA/Math feedback.

Known limitations (symbolic Math and APPS adapters) are documented. Code runs
in a resource-limited subprocess, NOT a security sandbox.
"""
from __future__ import annotations

import json
import math
import os
import re
import subprocess
import sys
import tempfile
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any

from .common import normalize_space

def extract_boxed(text: str) -> str | None:
    marker = r"\boxed{"
    result = None
    start_at = 0
    while True:
        index = text.find(marker, start_at)
        if index < 0:
            return result
        start = index + len(marker)
        depth = 1
        for pos in range(start, len(text)):
            if text[pos] == "{":
                depth += 1
            elif text[pos] == "}":
                depth -= 1
                if depth == 0:
                    result = text[start:pos].strip()
                    start_at = pos + 1
                    break
        else:
            return result


def extract_final_answer(text: str) -> str:
    patterns = (
        r"FINAL_ANSWER\s*:\s*(.+?)(?:\n|$)",
        r"Final answer\s*:\s*(.+?)(?:\n|$)",
        r"Answer\s*:\s*(.+?)(?:\n|$)",
    )
    for pattern in patterns:
        matches = list(re.finditer(pattern, text or "", re.IGNORECASE))
        if matches:
            return matches[-1].group(1).strip().strip("`* ")
    boxed = extract_boxed(text or "")
    return boxed if boxed is not None else normalize_space(text)


def normalize_qa(text: str) -> str:
    text = unicodedata.normalize("NFKC", extract_final_answer(text)).lower()
    text = re.sub(r"[^\w\s]", " ", text)
    tokens = [token for token in text.split() if token not in {"a", "an", "the"}]
    return " ".join(tokens)


def qa_metrics(prediction: str, gold: str) -> dict[str, Any]:
    pred_tokens = normalize_qa(prediction).split()
    gold_tokens = normalize_qa(gold).split()
    exact = pred_tokens == gold_tokens and bool(gold_tokens)
    common = Counter(pred_tokens) & Counter(gold_tokens)
    overlap = sum(common.values())
    if not pred_tokens or not gold_tokens:
        f1 = float(pred_tokens == gold_tokens)
    elif overlap == 0:
        f1 = 0.0
    else:
        precision = overlap / len(pred_tokens)
        recall = overlap / len(gold_tokens)
        f1 = 2 * precision * recall / (precision + recall)
    return {
        "success": exact,
        "reward": f1,
        "exact_match": exact,
        "f1": f1,
        "extracted_answer": extract_final_answer(prediction),
        "failure_type": None if exact else "wrong_answer",
        "feedback": "The answer is correct." if exact else "The answer is incorrect. Re-check the evidence and reasoning without assuming the gold answer.",
    }


def normalize_math(text: str) -> str:
    value = extract_final_answer(text)
    value = unicodedata.normalize("NFKC", value).lower().strip()
    value = value.strip("$`* .,:;!?")
    value = re.sub(r"\\(?:text|mathrm)\{([^{}]*)\}", r"\1", value)
    value = value.replace(r"\left", "").replace(r"\right", "")
    value = value.replace(r"\dfrac", r"\frac").replace(r"\tfrac", r"\frac")
    value = re.sub(r"\s+", "", value)
    return value


def simple_numeric_math(text: str) -> float | None:
    value = normalize_math(text)
    fraction = re.fullmatch(r"\\frac\{(-?\d+(?:\.\d+)?)\}\{(-?\d+(?:\.\d+)?)\}", value)
    if fraction and float(fraction.group(2)) != 0:
        return float(fraction.group(1)) / float(fraction.group(2))
    plain_fraction = re.fullmatch(r"(-?\d+(?:\.\d+)?)/(-?\d+(?:\.\d+)?)", value)
    if plain_fraction and float(plain_fraction.group(2)) != 0:
        return float(plain_fraction.group(1)) / float(plain_fraction.group(2))
    try:
        return float(value.replace(",", ""))
    except ValueError:
        return None


def math_metrics(prediction: str, gold: str) -> dict[str, Any]:
    pred_norm = normalize_math(prediction)
    gold_norm = normalize_math(gold)
    success = bool(gold_norm) and pred_norm == gold_norm
    equivalence = "normalized_exact" if success else "none"
    if not success:
        pred_num = simple_numeric_math(prediction)
        gold_num = simple_numeric_math(gold)
        if pred_num is not None and gold_num is not None and math.isclose(pred_num, gold_num, rel_tol=1e-9, abs_tol=1e-12):
            success = True
            equivalence = "numeric"
    return {
        "success": success,
        "reward": float(success),
        "equivalence": equivalence,
        "extracted_answer": extract_final_answer(prediction),
        "failure_type": None if success else "wrong_or_unverified_answer",
        "feedback": "The answer is correct." if success else "The answer is incorrect or not in a verifiable final-answer format. Re-derive it carefully.",
    }


def extract_code(text: str) -> str:
    blocks = re.findall(r"```(?:python)?\s*(.*?)```", text or "", re.IGNORECASE | re.DOTALL)
    code = blocks[-1] if blocks else text
    if "<Solution>" in code:
        code = code.split("<Solution>", 1)[1]
    return code.strip()


RUNNER_SOURCE = r'''
import json, math, sys, traceback

if hasattr(sys, "set_int_max_str_digits"):
    sys.set_int_max_str_digits(0)
payload = json.load(open(sys.argv[1], encoding="utf-8"))
namespace = {"__name__": "__router_candidate__"}
result = {"passed": 0, "total": 0, "failures": []}

def canonical(value):
    if isinstance(value, tuple):
        return [canonical(x) for x in value]
    if isinstance(value, list):
        return [canonical(x) for x in value]
    if isinstance(value, dict):
        return {str(k): canonical(v) for k, v in value.items()}
    if isinstance(value, float) and math.isnan(value):
        return "NaN"
    return value

try:
    exec(payload["code"], namespace, namespace)
    func = namespace[payload["fn_name"]]
    cases = list(zip(payload["inputs"], payload["outputs"]))
    result["total"] = len(cases)
    for index, (case, expected) in enumerate(cases):
        try:
            args = case if isinstance(case, list) else [case]
            actual = func(*args)
            if payload.get("output_format") == "apps_singleton_wrapper":
                if not isinstance(expected, list) or len(expected) != 1:
                    raise ValueError("APPS expected output must have one wrapper element")
                expected = expected[0]
            if canonical(actual) == canonical(expected):
                result["passed"] += 1
            elif len(result["failures"]) < 3:
                result["failures"].append({"index": index, "kind": "wrong_answer", "actual": repr(actual)[:300]})
        except BaseException as exc:
            if len(result["failures"]) < 3:
                result["failures"].append({"index": index, "kind": type(exc).__name__, "detail": str(exc)[:300]})
except BaseException as exc:
    result["fatal"] = {"kind": type(exc).__name__, "detail": str(exc)[:500]}

print(json.dumps(result, ensure_ascii=False))
'''


def _limit_child_resources() -> None:
    try:
        import resource

        resource.setrlimit(resource.RLIMIT_CPU, (8, 8))
        resource.setrlimit(resource.RLIMIT_AS, (1_500_000_000, 1_500_000_000))
        resource.setrlimit(resource.RLIMIT_FSIZE, (4_000_000, 4_000_000))
        resource.setrlimit(resource.RLIMIT_NOFILE, (64, 64))
        resource.setrlimit(resource.RLIMIT_NPROC, (32, 32))
    except Exception:
        pass


def code_metrics(prediction: str, tests: dict[str, Any], timeout: int = 12) -> dict[str, Any]:
    code = extract_code(prediction)
    payload = {
        "code": code,
        "fn_name": tests["fn_name"],
        "inputs": tests["inputs"],
        "outputs": tests["outputs"],
        "output_format": tests.get("output_format", "direct"),
    }
    with tempfile.TemporaryDirectory(prefix="gain_router_code_") as temp_dir:
        temp = Path(temp_dir)
        runner = temp / "runner.py"
        data = temp / "payload.json"
        runner.write_text(RUNNER_SOURCE, encoding="utf-8")
        data.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        try:
            completed = subprocess.run(
                [sys.executable, "-I", str(runner), str(data)],
                cwd=temp_dir,
                env={"PATH": os.environ.get("PATH", "")},
                capture_output=True,
                text=True,
                timeout=timeout,
                preexec_fn=_limit_child_resources if os.name == "posix" else None,
            )
        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "reward": 0.0,
                "passed": 0,
                "total": len(tests.get("inputs", [])),
                "failure_type": "execution_timeout",
                "feedback": "The implementation timed out on the visible tests.",
                "code": code,
            }
    try:
        result = json.loads(completed.stdout.strip().splitlines()[-1])
    except (json.JSONDecodeError, IndexError):
        result = {
            "passed": 0,
            "total": len(tests.get("inputs", [])),
            "fatal": {"kind": "runner_error", "detail": completed.stderr[-500:]},
        }
    total = int(result.get("total") or len(tests.get("inputs", [])))
    passed = int(result.get("passed") or 0)
    success = total > 0 and passed == total and not result.get("fatal")
    failures = result.get("failures") or []
    fatal = result.get("fatal")
    feedback_parts = [f"Visible tests passed: {passed}/{total}."]
    if fatal:
        feedback_parts.append(f"Fatal error: {fatal.get('kind')}: {fatal.get('detail')}")
    for failure in failures[:3]:
        feedback_parts.append(
            f"Test {failure.get('index')} failed ({failure.get('kind')}): "
            f"{failure.get('detail') or failure.get('actual') or ''}"
        )
    return {
        "success": success,
        "reward": passed / total if total else 0.0,
        "passed": passed,
        "total": total,
        "failure_type": None if success else (fatal or {}).get("kind", "test_failure"),
        "failures": failures,
        "feedback": "\n".join(feedback_parts),
        "code": code,
    }


def verify(row: dict[str, Any], output: str) -> dict[str, Any]:
    if row["domain"] == "qa":
        return qa_metrics(output, row["gold"])
    if row["domain"] == "math":
        return math_metrics(output, row["gold"])
    if row["domain"] == "code":
        return code_metrics(output, row["tests"])
    raise ValueError(f"unsupported domain: {row['domain']}")
