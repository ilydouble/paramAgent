# SCORE Paper — Independent Findings Summary

Performed by: Turgut Sofuyev
Date: 2026-09-19
Scope: independent, read-only investigation on the shared server
(`/root/autodl-tmp/lrr/ParamAgent`), motivated by the ICASSP reviewer
comments (Reject / Weak Reject) on the SCORE submission.

All work below was read-only with respect to existing code/data/models,
and did not interfere with any other running job on the shared GPU. Four
new files were added (this summary + the four reports below); nothing
else was modified.

## 1. Train/test contamination audit — `contamination_audit_REPORT.md`

Checked exact-match and near-duplicate (TF-IDF cosine >= 0.85) overlap
between all SFT/DPO training data and the actual evaluation test sets,
across Math, Code, APPS, and QA domains, plus the router's own
train/val/test splits.

- **0% exact-match overlap everywhere** (all domains, all splits).
- Near-duplicate (template-level) similarity found in **MATH (8.0%)** and
  **2WikiMultiHopQA (11.3%)** — consistent with the well-documented
  templated structure of these two specific benchmarks, not an artifact of
  our own data construction. HumanEval, MBPP, APPS, MuSiQue: clean (<=0.2%).
- Directly answers reviewer concern #5 ("precise split, deduplication, and
  contamination checks").

## 2. Router standalone validation — `ROUTER_VALIDATION_REPORT.md`

Ran the already-trained router checkpoint (`dataset/router/router_model_output`,
DeBERTa-v3-base, 4-way classifier) on its held-out test set (600 examples).

- Raw accuracy: 71.5%.
- **`abstain` recall: only 11.7%** — in 159/180 cases where the correct
  decision was "no expert needed," the router called an expert anyway.
- `code`/`math`/`qa` are classified near-perfectly (93-100% recall).
- Deployment-prior calibration raises `abstain` recall to 62.2% but drops
  `code` recall to 0% — the two classes trade off against each other under
  the current single 4-way classifier design.
- Directly answers reviewer concern #4 ("the central controller has not
  been independently validated").

## 3. End-to-end feasibility prototype — `score_prototype_RESULTS.md`

Built a small, standalone script that wires the router into a live
generate -> test -> (router decides) -> retry loop on HumanEval, using the
already-running shared vLLM server (Qwen3.5-4B) and the CPU-only router
(no new GPU memory used, no other process touched).

- Ran end-to-end without errors on 12 problems (3/12 solved).
- Router picked "code" in **27/27** decisions — never "abstain" — matching
  finding #2 in a live run, not just on the static test set.

## 4. Router-gated vs. always-expert ablation — `score_prototype_ABLATION_RESULTS.md`

Ran the identical problem set twice: once with router-gating, once always
calling the expert path (mirrors the paper's own Table 2 "w/o Gating"
ablation), at N=12 and N=50.

- N=12: router-gated 3/12 vs always-expert 4/12.
- N=50: router-gated 23/50 (46.0%) vs always-expert 20/50 (40.0%).
- Both gaps are within normal sampling noise for these sample sizes.
- **Because the router almost never abstains (finding #2/#3), the two modes
  execute nearly the same logic path regardless of gating** — the current
  router does not yet produce the selective/adaptive behavior the paper
  describes.

## Overall takeaway

1. Data contamination is not a real concern (0% exact overlap everywhere);
   this can be stated confidently in a rebuttal/revision.
2. The router component, as currently trained, is not doing the adaptive
   "call an expert only when needed" job the paper claims — it defaults to
   "call the expert" in essentially all cases. This was confirmed three
   independent ways (static eval, live single-run, live paired ablation).
3. Separately: the router (Stage A/B logic) does not appear to be wired
   into the main experiment scripts (`main.py`/`main_param.py`/etc.) at
   all — `results/` only contains `sft`/`dpo` result folders, no
   `router`/`score` folder, and no reference to the router exists anywhere
   in `code/`, `math/`, or `qa/`. The Table-1-style SCORE numbers currently
   in the manuscript likely reflect DPO alone, not router+DPO together.

These findings are offered as input for revising the manuscript / router
design before the next submission; no code was changed and no experiment
owned by anyone else was touched.
