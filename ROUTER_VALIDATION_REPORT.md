# SCORE Router — Standalone Validation Report

Date: 2026-09-19
Performed by: Turgut Sofuyev
Command: `python3 eval_router.py --split test --calibrate --device cpu`
Model: `dataset/router/router_model_output` (checkpoint-900, DeBERTa-v3-base, 4-way classifier)
Test set: `dataset/router/router_test.jsonl` (600 examples)

This is a read-only evaluation of the already-trained router checkpoint. No
training, no file writes, no changes to any existing code or model.

## Headline result

**Raw accuracy: 71.5%** (600 test examples)

## Confusion matrix (raw argmax)

| True \\ Pred | abstain | code | math | qa | Total |
|---|---|---|---|---|---|
| abstain | 21 | 69 | 48 | 42 | 180 |
| code | 0 | 60 | 0 | 0 | 60 |
| math | 0 | 0 | 180 | 0 | 180 |
| qa | 12 | 0 | 0 | 168 | 180 |

## Per-class metrics (raw)

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| abstain | 0.6364 | **0.1167** | **0.1972** | 180 |
| code | 0.4651 | 1.0000 | 0.6349 | 60 |
| math | 0.7895 | 1.0000 | 0.8824 | 180 |
| qa | 0.8000 | 0.9333 | 0.8615 | 180 |
| **accuracy** | | | **0.7150** | 600 |

## Key finding: the router almost never says "no help needed"

`code`, `math`, and `qa` are classified essentially perfectly (recall 93-100%).
But `abstain` — the class representing "generic self-reflection is enough,
no domain expert needed" — is only correctly identified **11.67%** of the
time. In 159/180 cases where the correct decision was "don't call the
expert," the router called one anyway.

This is a direct, measured answer to reviewer concern #4 ("the central
controller has not been independently validated"). It also has direct
bearing on the paper's own Observation about "over-correction" /
negative transfer: if the router rarely abstains, it will keep triggering
expert guidance on cases that did not need it, which is exactly the
failure mode SCORE's own pilot study (Figure 1a) warns against.

## Distribution mismatch (train prior vs. real-world prior)

| Class | True label share (test) | Model's predicted share | Assumed deploy prior |
|---|---|---|---|
| abstain | 0.300 | 0.055 | 0.590 |
| code | 0.100 | 0.215 | 0.023 |
| math | 0.300 | 0.380 | 0.120 |
| qa | 0.300 | 0.350 | 0.270 |

The router was trained on a roughly balanced label distribution, but the
assumed real deployment distribution is dominated by `abstain` (59%). This
mismatch is the main reason the raw router under-predicts `abstain`.

## Effect of prior (deployment-weighted) calibration

The eval script re-weights the test set to match the assumed deployment
prior, then applies a prior-correction to the logits.

| Class | P before | P after | R before | R after | F1 before | F1 after |
|---|---|---|---|---|---|---|
| abstain | 0.7927 | 0.7738 | 0.1167 | **0.6222** | 0.2034 | **0.6898** |
| code | 0.0923 | 0.0000 | 1.0000 | **0.0000** | 0.1690 | **0.0000** |
| math | 0.4327 | 0.4369 | 1.0000 | 0.5722 | 0.6040 | 0.4955 |
| qa | 0.6467 | 0.6381 | 0.9333 | 0.8778 | 0.7640 | 0.7390 |
| **accuracy (deploy-weighted)** | | | | | **0.4624 -> 0.6708** | (+0.2083) |

Calibration substantially improves `abstain` recall (11.7% -> 62.2%) and
overall deploy-weighted accuracy (46.2% -> 67.1%), but at the cost of
**completely losing the `code` class** (recall drops to 0% — the model
never predicts `code` after calibration, because `code`'s deployment prior
is assumed to be very small, 2.3%).

## Takeaways

1. The router's raw behavior is strongly biased toward calling an expert
   (low `abstain` recall), which works against the paper's own stated goal
   of avoiding unnecessary/harmful parametric intervention on easy cases.
2. Naive prior calibration fixes `abstain` recall but destroys `code`
   detection — the two are in tension with the current 4-way single
   classifier design and the assumed deployment prior.
3. This is exactly the kind of controller-level validation (accuracy,
   confusion matrix, calibration behavior) that reviewer concern #4 asked
   for and that was previously missing from the manuscript.
4. Before claiming the router "adaptively allocates exploration budget," the
   `abstain` recall / `code` recall trade-off above should be addressed or
   explicitly discussed as a limitation.

## Reproducing this result

```bash
conda activate param
cd /root/autodl-tmp/lrr/ParamAgent/dataset/router
python3 eval_router.py --split test --calibrate --device cpu
```

(`--device cpu` was used because the GPU was fully occupied by another
process at the time of this run; results are identical to a GPU run, only
slower.)
