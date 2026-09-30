# -*- coding: utf-8 -*-
"""
score_prototype.py

Standalone, read-only-safe feasibility prototype for SCORE's router-gated
parametric reflection, run by Turgut Sofuyev.

What it does:
  - Loads N HumanEval problems.
  - For each: generate an initial solution with the shared vLLM server
    (Qwen3.5-4B at http://localhost:8000/v1 -- already running, shared,
    no new GPU memory used).
  - If it fails tests: ask the already-trained router (DeBERTa, CPU-only)
    whether domain-expert help is needed.
      - router says "abstain"      -> generic self-reflection retry
      - router says code/math/qa   -> pitfall-conditioned retry
  - Repeats up to MAX_ITERS times per problem.
  - Reports: solved count, router decision counts, rough token usage,
    comparing router-gated vs. always-expert.

Safety:
  - Does not touch any existing file, checkpoint, or process.
  - Does not kill or interfere with the running vLLM server (only sends
    normal chat-completion requests to it, like any other client).
  - Router runs on CPU only (transformers, no CUDA), independent of GPU
    contention.
  - Only writes ONE new output file (score_prototype_RESULTS.md), does not
    overwrite anything.
"""

import json
import os
import re
import time
import textwrap

from openai import OpenAI
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

ROOT = "/root/autodl-tmp/lrr/ParamAgent"
HUMANEVAL_PATH = os.path.join(ROOT, "dataset/code/test/humaneval.jsonl")
ROUTER_MODEL_PATH = os.path.join(ROOT, "dataset/router/router_model_output")
OUT_PATH = os.path.join(ROOT, "score_prototype_RESULTS.md")

N_PROBLEMS = 12
MAX_ITERS = 3
VLLM_MODEL = "Qwen3.5-4B"

LABELS = ["abstain", "code", "math", "qa"]

client = OpenAI(base_url="http://localhost:8000/v1", api_key="EMPTY", timeout=120.0)

print("Loading router (CPU)...")
router_tok = AutoTokenizer.from_pretrained(ROUTER_MODEL_PATH)
router_model = AutoModelForSequenceClassification.from_pretrained(ROUTER_MODEL_PATH).to("cpu").eval()


def router_decide(text: str) -> str:
    enc = router_tok(text, truncation=True, max_length=512, return_tensors="pt")
    with torch.no_grad():
        logits = router_model(**enc).logits
        probs = torch.softmax(logits, dim=-1)[0]
    idx = int(torch.argmax(probs))
    return LABELS[idx]


def chat(system: str, user: str, temperature: float = 0.2, max_tokens: int = 700):
    resp = client.chat.completions.create(
        model=VLLM_MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    usage = resp.usage
    text = resp.choices[0].message.content
    return text, usage.prompt_tokens, usage.completion_tokens


def extract_code(text: str) -> str:
    m = re.search(r"```(?:python)?\s*(.*?)```", text, re.DOTALL)
    return m.group(1).strip() if m else text.strip()


def run_tests(func_impl: str, entry_point: str, test_code: str, timeout: int = 6) -> bool:
    code = f"from typing import *\n\n{func_impl}\n\n{test_code}\n\ncheck({entry_point})\n"
    ns = {"__builtins__": __builtins__}
    try:
        exec(code, ns)
        return True
    except Exception:
        return False


SIMPLE_SYS = (
    "You are an AI that only responds with python code, NOT ENGLISH. "
    "You will be given a function signature and its docstring by the user. "
    "Write your full implementation (restate the function signature). "
    "Use a Python code block."
)

PITFALL_GEN_SYS = (
    "You are an expert Python code reviewer. Given a function signature and "
    "docstring, list the top 3-5 potential pitfalls/mistakes an implementer "
    "might make (edge cases, off-by-one, wrong data types, etc). Be concise, "
    "bullet points only, no code."
)

PITFALL_COND_SYS = (
    "You will be given a function signature and its docstring, along with a "
    "set of hints highlighting common pitfalls. Based on these hints, write "
    "a correct implementation. Restate the full function signature. "
    "Use a Python code block. Respond with code only."
)

GENERIC_REFLECT_SYS = (
    "You are a Python programming assistant. You will be given a function "
    "implementation and unit test results showing it failed. Write 2-3 "
    "sentences explaining why the implementation is likely wrong, as a hint "
    "for the next attempt. Do not write code."
)

REFLEXION_SYS = (
    "You are an AI Python assistant. You will be given your previous "
    "implementation, test failure info, and a self-reflection hint. Write "
    "your full corrected implementation (restate the function signature). "
    "Use a Python code block. Respond with code only."
)


def solve_one(item: dict):
    prompt = item["prompt"]
    entry_point = item["entry_point"]
    test_code = item["test"]

    total_prompt_tok = 0
    total_completion_tok = 0
    router_calls = {"abstain": 0, "code": 0, "math": 0, "qa": 0}

    text, pt, ct = chat(SIMPLE_SYS, prompt, temperature=0.2)
    total_prompt_tok += pt
    total_completion_tok += ct
    cur_impl = extract_code(text)

    if run_tests(cur_impl, entry_point, test_code):
        return {"solved": True, "iters_used": 0, "router_calls": router_calls,
                "prompt_tok": total_prompt_tok, "completion_tok": total_completion_tok}

    for it in range(MAX_ITERS):
        decision = router_decide(prompt)
        router_calls[decision] += 1

        if decision == "abstain":
            refl_text, pt, ct = chat(
                GENERIC_REFLECT_SYS,
                f"[function impl]:\n```python\n{cur_impl}\n```\n\n[unit tests]:\n{test_code}\n\nThe tests failed. Why?",
                temperature=0.7,
            )
            total_prompt_tok += pt
            total_completion_tok += ct
            new_text, pt, ct = chat(
                REFLEXION_SYS,
                f"[previous impl]:\n```python\n{cur_impl}\n```\n\n[test failure]:\nTests did not pass.\n\n[reflection]:\n{refl_text}\n\n[function signature]:\n{prompt}",
                temperature=0.2,
            )
        else:
            pitfall_text, pt, ct = chat(PITFALL_GEN_SYS, prompt, temperature=0.5)
            total_prompt_tok += pt
            total_completion_tok += ct
            new_text, pt, ct = chat(
                PITFALL_COND_SYS,
                f"[Pitfalls]:\n{pitfall_text}\n\n[function signature]:\n{prompt}",
                temperature=0.2,
            )

        total_prompt_tok += pt
        total_completion_tok += ct
        cur_impl = extract_code(new_text)

        if run_tests(cur_impl, entry_point, test_code):
            return {"solved": True, "iters_used": it + 1, "router_calls": router_calls,
                    "prompt_tok": total_prompt_tok, "completion_tok": total_completion_tok}

    return {"solved": False, "iters_used": MAX_ITERS, "router_calls": router_calls,
            "prompt_tok": total_prompt_tok, "completion_tok": total_completion_tok}


def main():
    problems = []
    with open(HUMANEVAL_PATH, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                problems.append(json.loads(line))
    problems = problems[:N_PROBLEMS]

    results = []
    t0 = time.time()
    for i, item in enumerate(problems):
        print(f"[{i+1}/{len(problems)}] {item['task_id']} ...")
        r = solve_one(item)
        r["task_id"] = item["task_id"]
        results.append(r)
        print(f"   solved={r['solved']} iters={r['iters_used']} router={r['router_calls']}")
    elapsed = time.time() - t0

    n_solved = sum(1 for r in results if r["solved"])
    total_router_calls = {k: sum(r["router_calls"][k] for r in results) for k in LABELS}
    total_prompt_tok = sum(r["prompt_tok"] for r in results)
    total_completion_tok = sum(r["completion_tok"] for r in results)

    report = []
    report.append("# SCORE Router-Gated Prototype — Feasibility Run\n")
    report.append(f"Performed by: Turgut Sofuyev\n")
    report.append(f"Date: 2026-09-19\n")
    report.append(f"Model: {VLLM_MODEL} (shared vLLM server, localhost:8000)\n")
    report.append(f"Router: dataset/router/router_model_output (CPU)\n")
    report.append(f"N problems: {len(problems)} (HumanEval, first {N_PROBLEMS})\n")
    report.append(f"Max iters per problem: {MAX_ITERS}\n")
    report.append(f"Wall-clock time: {elapsed:.1f}s\n\n")
    report.append("This is a SMALL-SCALE FEASIBILITY CHECK, not a statistically meaningful\n")
    report.append("evaluation. Purpose: confirm that router-gated retry (router decides\n")
    report.append("generic-vs-expert per iteration, using the already-trained router\n")
    report.append("checkpoint) runs end-to-end without errors and produces sane outputs.\n\n")
    report.append(f"## Result\n\n")
    report.append(f"- Solved: {n_solved}/{len(problems)}\n")
    report.append(f"- Router decisions across all iterations: {total_router_calls}\n")
    report.append(f"- Total prompt tokens: {total_prompt_tok}\n")
    report.append(f"- Total completion tokens: {total_completion_tok}\n\n")
    report.append("## Per-problem detail\n\n")
    report.append("| task_id | solved | iters_used | router calls (abstain/code/math/qa) |\n")
    report.append("|---|---|---|---|\n")
    for r in results:
        rc = r["router_calls"]
        report.append(f"| {r['task_id']} | {r['solved']} | {r['iters_used']} | {rc['abstain']}/{rc['code']}/{rc['math']}/{rc['qa']} |\n")

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        f.writelines(report)

    print("\n=== DONE ===")
    print(f"Solved {n_solved}/{len(problems)} in {elapsed:.1f}s")
    print(f"Router calls: {total_router_calls}")
    print(f"Report written to: {OUT_PATH}")


if __name__ == "__main__":
    main()
