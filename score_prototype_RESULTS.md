# SCORE Router-Gated Prototype — Feasibility Run
Performed by: Turgut Sofuyev
Date: 2026-09-19
Model: Qwen3.5-4B (shared vLLM server, localhost:8000)
Router: dataset/router/router_model_output (CPU)
N problems: 12 (HumanEval, first 12)
Max iters per problem: 3
Wall-clock time: 463.0s

This is a SMALL-SCALE FEASIBILITY CHECK, not a statistically meaningful
evaluation. Purpose: confirm that router-gated retry (router decides
generic-vs-expert per iteration, using the already-trained router
checkpoint) runs end-to-end without errors and produces sane outputs.

## Result

- Solved: 3/12
- Router decisions across all iterations: {'abstain': 0, 'code': 27, 'math': 0, 'qa': 0}
- Total prompt tokens: 31121
- Total completion tokens: 46200

## Per-problem detail

| task_id | solved | iters_used | router calls (abstain/code/math/qa) |
|---|---|---|---|
| HumanEval/0 | False | 3 | 0/3/0/0 |
| HumanEval/1 | False | 3 | 0/3/0/0 |
| HumanEval/2 | True | 0 | 0/0/0/0 |
| HumanEval/3 | False | 3 | 0/3/0/0 |
| HumanEval/4 | False | 3 | 0/3/0/0 |
| HumanEval/5 | False | 3 | 0/3/0/0 |
| HumanEval/6 | False | 3 | 0/3/0/0 |
| HumanEval/7 | True | 0 | 0/0/0/0 |
| HumanEval/8 | False | 3 | 0/3/0/0 |
| HumanEval/9 | False | 3 | 0/3/0/0 |
| HumanEval/10 | False | 3 | 0/3/0/0 |
| HumanEval/11 | True | 0 | 0/0/0/0 |
