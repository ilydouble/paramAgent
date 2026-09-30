# Fixed multi-seed data split protocol

This project assigns the **original benchmark problem** to a split before any
model output is generated. Router traces, confidence features, reward labels,
reflections, and preference pairs inherit that assignment through `group_id`.

## Frozen variants

- Primary development/reporting split: `42`
- Robustness splits: `123`, `2027`, `3407`, `8888`

The data-split seed controls membership only. Each training run must record a
separate model seed. Compare methods on the same data-split seed and aggregate
robustness results across all five manifests.

## Leakage boundary

One original problem is one group. Every attempt, failed trace, generic answer,
diverse-prompt answer, reward observation, and chosen/rejected pair derived from
that problem must retain its original `sample_id` and `group_id`.

The test split is final evaluation only. Do not use it to:

- construct router/reward labels;
- generate SFT, DPO, or other preference-training examples;
- tune thresholds, prompts, budgets, temperatures, or checkpoints;
- select the best data-split seed or training seed.

Use train for fitting and validation for all selection. Evaluate the frozen test
split after the complete policy is fixed.

## Generate manifests on AutoDL

The source JSONL files stay on the server and are ignored by Git. From the
repository root run:

```bash
python3 dataset/router/prepare_router_data.py
```

This creates:

```text
dataset/router/splits/router_v1/seed_<seed>/
  router_train.jsonl
  router_val.jsonl
  router_test.jsonl
  manifest.json

split_manifests/router_v1/
  index.json
  seed_42.json
  seed_123.json
  seed_2027.json
  seed_3407.json
  seed_8888.json
```

The materialized JSONL files contain prompt text and remain ignored. The
version-controlled manifests contain hashed identifiers, assignments, source
file hashes, and an assignment checksum, but no prompt text.

The generator refuses to overwrite existing variants unless `--overwrite` is
passed. Treat overwriting a published manifest as a protocol change and record
a new protocol version instead.

## Reuse a manifest for reward or preference data

Derived rows must contain `group_id`, `sample_id`, or `source_sample_id` copied
from the original problem. Then run:

```bash
python3 scripts/apply_split_manifest.py \
  --manifest split_manifests/router_v1/seed_42.json \
  --input outputs/reward_examples.jsonl \
  --output-dir outputs/reward_split_seed_42 \
  --prefix reward
```

The command fails on unmatched rows by default. It never guesses membership
from generated trace text because that would recreate the leakage risk.

## Router training

`train_router.py` validates all materialized rows against the manifest before
loading the model:

```bash
python3 dataset/router/train_router.py --split_seed 42 --seed 1 --fp16
```

Here `--split_seed` is the frozen data partition and `--seed` is the independent
optimization seed. Store both in every experiment result.

## Reporting recommendation

Report the primary split result and the mean ± standard deviation over the five
fixed data-split variants. Use the identical variant for every baseline and
ablation. If compute allows, use multiple optimization seeds inside each split;
otherwise keep one declared optimization seed and state that the robustness
analysis varies data membership only.
