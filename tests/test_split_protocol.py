import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from split_protocol import (
    assign_exact_groups,
    content_group_id,
    load_manifest,
    materialize_rows,
    sample_id,
    validate_materialized_splits,
)


def make_records() -> list[dict]:
    records = []
    for stratum in ("code", "math"):
        for index in range(10):
            text = f"{stratum} problem {index}"
            records.append(
                {
                    "text": text,
                    "label": stratum,
                    "label_id": 0,
                    "sample_id": sample_id(stratum, text),
                    "group_id": content_group_id(text),
                    "stratum": stratum,
                }
            )
    return records


TARGETS = {
    "code": {"train": 6, "val": 2, "test": 2},
    "math": {"train": 6, "val": 2, "test": 2},
}


class SplitProtocolTest(unittest.TestCase):
    def test_assignment_is_deterministic_and_order_independent(self):
        records = make_records()
        first, first_manifest = assign_exact_groups(records, seed=42, targets=TARGETS)
        second, second_manifest = assign_exact_groups(list(reversed(records)), seed=42, targets=TARGETS)
        self.assertEqual(first, second)
        self.assertEqual(first_manifest["assignment_sha256"], second_manifest["assignment_sha256"])

    def test_different_split_seeds_produce_different_assignments(self):
        records = make_records()
        first, _ = assign_exact_groups(records, seed=42, targets=TARGETS)
        second, _ = assign_exact_groups(records, seed=2027, targets=TARGETS)
        self.assertNotEqual(first, second)

    def test_all_rows_from_one_group_stay_together(self):
        records = make_records()
        duplicate = dict(records[0])
        duplicate["sample_id"] = sample_id("trace", "derived trace")
        records.append(duplicate)
        grouped_targets = {
            "code": {"train": 6, "val": 2, "test": 2},
            "math": TARGETS["math"],
        }
        assignment, _ = assign_exact_groups(records, seed=42, targets=grouped_targets)
        rows = materialize_rows(records, assignment, seed=42)
        locations = {
            split_name
            for split_name, split_rows in rows.items()
            if any(row["group_id"] == records[0]["group_id"] for row in split_rows)
        }
        self.assertEqual(len(locations), 1)

    def test_group_crossing_strata_is_rejected(self):
        records = make_records()
        records[10]["group_id"] = records[0]["group_id"]
        with self.assertRaisesRegex(ValueError, "spans strata"):
            assign_exact_groups(records, seed=42, targets=TARGETS)

    def test_manifest_validation_and_derived_application(self):
        records = make_records()
        assignment, manifest = assign_exact_groups(records, seed=42, targets=TARGETS)
        rows_by_split = materialize_rows(records, assignment, seed=42)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with (root / "manifest.json").open("w", encoding="utf-8") as handle:
                json.dump(manifest, handle)
            for split_name, rows in rows_by_split.items():
                with (root / f"router_{split_name}.jsonl").open("w", encoding="utf-8") as handle:
                    for row in rows:
                        handle.write(json.dumps(row) + "\n")
            self.assertEqual(validate_materialized_splits(root)["seed"], 42)
            self.assertEqual(load_manifest(root / "manifest.json")["assignment_sha256"], manifest["assignment_sha256"])

            derived = root / "derived.jsonl"
            with derived.open("w", encoding="utf-8") as handle:
                handle.write(json.dumps({"group_id": records[0]["group_id"], "reward": 1.0}) + "\n")
            output_dir = root / "derived_splits"
            subprocess.run(
                [
                    sys.executable,
                    str(REPO_ROOT / "scripts" / "apply_split_manifest.py"),
                    "--manifest",
                    str(root / "manifest.json"),
                    "--input",
                    str(derived),
                    "--output-dir",
                    str(output_dir),
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            expected_split = assignment[records[0]["group_id"]]
            output_path = output_dir / f"derived_{expected_split}.jsonl"
            row = json.loads(output_path.read_text(encoding="utf-8").strip())
            self.assertEqual(row["split"], expected_split)
            self.assertEqual(row["split_seed"], 42)


if __name__ == "__main__":
    unittest.main()
