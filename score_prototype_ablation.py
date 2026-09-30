# -*- coding: utf-8 -*-
"""
score_prototype_ablation.py

Follow-up to score_prototype.py, run by Turgut Sofuyev.

Runs the SAME HumanEval problems twice on the shared vLLM server
(Qwen3.5-4B, localhost:8000, no new GPU memory used):

  Mode A "router_gated": on each failed attempt, ask the trained router
    (CPU) whether to use generic self-reflection (abstain) or an
    expert/pitfall-conditioned retry.
  Mode B "always_expert": on each failed attempt, ALWAYS use the
    expert/pitfall-conditioned retry (router is not consulted) -- this
    mirrors the paper's own "w/o Gating" ablation (Table 2).

Reports solved count and total tokens for both modes side by side, on the
identical problem set, so the effect of gating vs. always-on can be seen
directly.

Safety: same as score_prototype.py -- no existing files/models touched,
only sends chat requests to the already-running shared vLLM server, router
runs on CPU. Writes ONE new file (score_prototype_ABLATION_RESULTS.md).
"""

import json
import os
import re
import time

from openai import OpenAI
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

ROOT = "/root/autodl-tmp/lrr/ParamAgent"
HUMANEVAL_PATH = os.path.join(ROOT, "dataset/code/test/humaneval.jsonl")
ROUTER_MODEL_PATH = os.path.join(ROOT, "dataset/router/router_model_output")
OUT_PATH = os.path.join(ROOT, "score_prototype_ABLATION_RESULTS.md")

N_PROBLEMS = 50
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
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    usage = resp.usage
    return resp.choices[0].message.content, usage.prompt_tokens, usage.completion_tokens


def extract_code(text: str) -> str:
    m = re.search(r"```(?:python)?\s*(.*?)```", text, re.DOTALL)
    return m.group(1).strip() if m else text.strip()


def run_tests(func_impl: str, entry_point: str, test_code: str) -> bool:
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


def solve_one(item: dict, mode: str):
    """mode: 'router_gated' or 'always_expert'"""
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
        if mode == "router_gated":
            decision = router_decide(prompt)
        else:  # always_expert
            decision = "code"
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


def run_mode(problems, mode):
    print(f"\n=== MODE: {mode} ===")
    results = []
    t0 = time.time()
    for i, item in enumerate(problems):
        r = solve_one(item, mode)
        r["task_id"] = item["task_id"]
        results.append(r)
        print(f"[{i+1}/{len(problems)}] {item['task_id']}: solved={r['solved']} iters={r['iters_used']}")
    elapsed = time.time() - t0
    return results, elapsed


def summarize(results):
    n_solved = sum(1 for r in results if r["solved"])
    total_pt = sum(r["prompt_tok"] for r in results)
    total_ct = sum(r["completion_tok"] for r in results)
    total_router = {k: sum(r["router_calls"][k] for r in results) for k in LABELS}
    return n_solved, total_pt, total_ct, total_router


def main():
    problems = []
    with open(HUMANEVAL_PATH, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                problems.append(json.loads(line))
    problems = problems[:N_PROBLEMS]

    gated_results, gated_time = run_mode(problems, "router_gated")
    always_results, always_time = run_mode(problems, "always_expert")

    g_solved, g_pt, g_ct, g_router = summarize(gated_results)
    a_solved, a_pt, a_ct, a_router = summarize(always_results)

    report = []
    report.append("# SCORE Router-Gated vs Always-Expert Ablation — Feasibility Run\n\n")
    report.append("Performed by: Turgut Sofuyev\n")
    report.append("Date: 2026-09-19\n")
    report.append(f"Model: {VLLM_MODEL} (shared vLLM server, localhost:8000)\n")
    report.append(f"Router: dataset/router/router_model_output (CPU)\n")
    report.append(f"N problems: {len(problems)} (HumanEval, first {N_PROBLEMS}, identical set for both modes)\n")
    report.append(f"Max iters per problem: {MAX_ITERS}\n\n")
    report.append("SMALL-SCALE FEASIBILITY CHECK, not a statistically meaningful evaluation.\n")
    report.append("Purpose: reproduce, at small scale, the paper's own Table 2 comparison\n")
    report.append("(SCORE full / gated vs. w/o Gating / always-on expert).\n\n")
    report.append("## Result\n\n")
    report.append("| Mode | Solved | Prompt tokens | Completion tokens | Router calls (abstain/code/math/qa) | Wall time |\n")
    report.append("|---|---|---|---|---|---|\n")
    report.append(f"| Router-gated | {g_solved}/{len(problems)} | {g_pt} | {g_ct} | {g_router['abstain']}/{g_router['code']}/{g_router['math']}/{g_router['qa']} | {gated_time:.1f}s |\n")
    report.append(f"| Always-expert (no gating) | {a_solved}/{len(problems)} | {a_pt} | {a_ct} | {a_router['abstain']}/{a_router['code']}/{a_router['math']}/{a_router['qa']} | {always_time:.1f}s |\n\n")

    report.append("## Per-problem detail\n\n")
    report.append("| task_id | router_gated solved | always_expert solved |\n")
    report.append("|---|---|---|\n")
    for gr, ar in zip(gated_results, always_results):
        report.append(f"| {gr['task_id']} | {gr['solved']} | {ar['solved']} |\n")

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        f.writelines(report)

    print("\n=== DONE ===")
    print(f"Router-gated:  {g_solved}/{len(problems)} solved, {g_pt+g_ct} total tokens")
    print(f"Always-expert: {a_solved}/{len(problems)} solved, {a_pt+a_ct} total tokens")
    print(f"Report written to: {OUT_PATH}")


if __name__ == "__main__":
    main()
