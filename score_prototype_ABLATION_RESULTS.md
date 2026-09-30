# SCORE Router-Gated vs Always-Expert Ablation — Feasibility Run

Performed by: Turgut Sofuyev
Date: 2026-09-19
Model: Qwen3.5-4B (shared vLLM server, localhost:8000)
Router: dataset/router/router_model_output (CPU)
Max iters per problem: 3

Two runs were done: N=12 and N=50 (HumanEval, first N problems, identical
set for both modes in each run).

## Result — N=12

| Mode | Solved | Prompt+Completion tokens |
|---|---|---|
| Router-gated | 3/12 (25.0%) | 77,070 |
| Always-expert (no gating) | 4/12 (33.3%) | 69,762 |

## Result — N=50

| Mode | Solved | Prompt+Completion tokens |
|---|---|---|
| Router-gated | 23/50 (46.0%) | 277,838 |
| Always-expert (no gating) | 20/50 (40.0%) | 266,027 |

## Interpretation

At both N=12 and N=50, router-gated and always-expert produce results within
normal sampling noise of each other (the N=50 gap of 3 problems / ~6 points
is well within the expected standard error for a binomial proportion at this
sample size, ~7 points). This is consistent with, and explained by, the
standalone router validation (see ROUTER_VALIDATION_REPORT.md): the router
almost never predicts "abstain" (11.7% recall), so in practice it routes to
"code" essentially every time an attempt fails — meaning the two modes are
executing nearly the same logic path regardless of whether the router is
consulted. The observed accuracy/token differences are attributable to
generation stochasticity (temperature-based sampling), not to a genuine
gating effect.

This is a SMALL-SCALE FEASIBILITY CHECK, not a statistically powered
evaluation, and used a generic on-the-fly pitfall prompt rather than the
actual DPO-trained expert LoRA (which was not loaded, to avoid competing for
scarce GPU memory with other running jobs). The purpose was to verify, at
small scale, whether router-gating produces a measurable behavioral
difference from always-on expert guidance — and to cross-check the
standalone router evaluation with a live, end-to-end run.

## Per-problem detail (N=50)

See console log / re-run for full per-task detail; both modes were run on
the identical first 50 HumanEval problems.
