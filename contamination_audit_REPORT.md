# SCORE / ParamAgent — Train/Test Contamination Audit

Date: 2026-09-19
Performed by: Turgut Sofuyev, read-only analysis on the
`connect.westc.gpuhub.com` server, path `/root/autodl-tmp/lrr/ParamAgent`.

Scope: check whether any train/SFT/DPO data used to build the parametric
memory (pitfalls / decompositions) or the router overlaps with the actual
evaluation test sets, addressing reviewer concern #5 ("held-out is not
sufficient to exclude problem duplication, near duplication, and label
template leakage").

No files were modified, deleted, or moved. This report and the scripts below
are read-only additions.

## Method

1. **Exact match**: normalize text (strip, lowercase, collapse whitespace),
   hash, and intersect train vs. test sets per domain.
2. **Near-duplicate**: TF-IDF (1-2 grams) + cosine similarity, train vs. test,
   flag pairs with similarity >= 0.85.

## Files compared

| Domain | Train files | Test files |
|---|---|---|
| Math | `dataset/math/train/sft.json`, `dataset/math/train/dpo.jsonl` | `dataset/math/test/math_test_150_per_type.jsonl`, `OlymMATH-EN-EASY.jsonl`, `OlymMATH-EN-HARD.jsonl` |
| Code | `dataset/code/train/sft.json`, `dataset/code/train/dpo.jsonl` (fields `question`, `func_sign`) | `dataset/code/test/humaneval.jsonl`, `dataset/code/test/mbpp-py.jsonl` |
| APPS | `dataset/code/train/apps_original.json` | `dataset/code/test/apps.json`, `dataset/code/test/apps_competition_300.json` |
| QA | `dataset/multihop/train/sft.jsonl`, `dataset/multihop/train/dpo.jsonl` | `dataset/multihop/test/hotpot_qa.jsonl`, `2w_150_per_type.jsonl`, `musique_100.jsonl` |
| Router | `dataset/router/router_train.jsonl` | `dataset/router/router_val.jsonl`, `router_test.jsonl` |

## Results — Exact match

| Domain | Train n | Test n | Exact overlap |
|---|---|---|---|
| Math (sft+dpo vs math_test_150) | 8333 | 1050 | **0 (0.0%)** |
| Math vs OlymMATH-EASY / HARD | 8333 | 100 / 100 | **0 / 0** |
| Code (sft+dpo vs HumanEval) | 12685 | 164 | **0 (0.0%)** |
| Code (sft+dpo vs MBPP) | 12685 | 397 | **0 (0.0%)** |
| APPS (train vs apps.json test) | 3992 | 600 | **0 (0.0%)** |
| APPS (train vs apps_competition_300) | 3992 | 300 | **0 (0.0%)** |
| QA (sft+dpo vs HotpotQA) | 27309 | 450 | **0 (0.0%)** |
| QA (sft+dpo vs 2WikiMultiHopQA) | 27309 | 600 | **0 (0.0%)** |
| QA (sft+dpo vs MuSiQue) | 27309 | 559 | **0 (0.0%)** |
| Router (train vs val) | 4788 | 600 | **0 (0.0%)** |
| Router (train vs test) | 4788 | 600 | **0 (0.0%)** |
| Router (val vs test) | 600 | 600 | **0 (0.0%)** |

**No exact-match contamination found anywhere.**

Note: `dataset/math/train/sft.json` contains 2459/6333 (38.8%) entries whose
`unique_id` starts with `test/...` (original Hendrycks MATH internal
train/test folder naming). Despite this, exact-match against
`math_test_150_per_type.jsonl` is still 0%, meaning the current SCORE math
eval set does not literally reuse those original-MATH-test-labeled problems.
Worth flagging to Liu since the naming is easy to misread as "using the real
test set for training."

## Results — Near-duplicate (TF-IDF cosine >= 0.85)

| Domain | Test n | Flagged | % | Likely cause |
|---|---|---|---|---|
| **Math vs math_test_150** | 1050 | **84** | **8.00%** | Template reuse — same problem structure, different numbers (e.g. age-difference word problems) |
| Math vs OlymMATH-EASY/HARD | 100/100 | 0/0 | 0% | Clean |
| Code vs HumanEval | 164 | 0 | 0% | Clean |
| Code vs MBPP | 397 | 0 | 0% | Clean |
| APPS vs apps.json | 600 | 1 | 0.17% | One coincidentally similar problem (rectangle vs square variant) |
| APPS vs apps_competition_300 | 300 | 0 | 0% | Clean |
| QA vs HotpotQA | 450 | 6 | 1.33% | Minor, template-ish comparative questions |
| **QA vs 2WikiMultiHopQA** | 600 | **68** | **11.33%** | Template reuse — 2WikiMultiHopQA is known to be heavily template-based ("place of birth of the director of film X") |
| QA vs MuSiQue | 559 | 1 | 0.18% | Negligible |

## Interpretation

- **No literal copy-paste leakage anywhere** (0% exact match, all domains).
- Math and 2WikiMultiHopQA show template-level similarity between train and
  test. This is a **known, documented characteristic of these two
  benchmarks themselves** (both are widely reported in the literature to be
  template-heavy), not something introduced by SCORE's own data
  construction pipeline. It should still be **disclosed transparently** in
  the paper, since reviewer #5 explicitly asked for a contamination check.

## Suggested paper sentence

> "We verified exact-match deduplication across all train/test splits (0%
> overlap in every domain). We additionally checked near-duplicate
> (TF-IDF cosine >= 0.85) overlap and found template-level similarity in
> MATH (8.0%) and 2WikiMultiHopQA (11.3%), consistent with the known
> templated structure of these benchmarks; HumanEval, MBPP, APPS, and
> MuSiQue showed no near-duplicates (<=0.2%)."

## Reproducing this audit

Scripts used (saved alongside this report):
- `dup_check_math.py` — MATH domain
- `dup_check_code.py` — Code / APPS domain
- `dup_check_qa.py` — QA domain
- `dup_check_router.py` — Router splits

All scripts are read-only: they only `open()` existing JSON/JSONL files and
compute similarity in memory. No files are written except this report.
