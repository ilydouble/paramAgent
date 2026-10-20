import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/retrain_sft_dpo.py"
SPEC = importlib.util.spec_from_file_location("retrain_sft_dpo", SCRIPT)
retrain = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(retrain)


class SFTLineageTests(unittest.TestCase):
    def fixture(self, root):
        base = root / "models/Qwen3.5-2B-code-merged2"
        base.mkdir(parents=True)
        adapter = root / "lora-qwen3.5-2b/lora-qwen3.5-2b-code2"
        adapter.mkdir(parents=True)
        (base / "model.safetensors").write_bytes(b"merged-sft-weights")
        (adapter / "adapter_model.safetensors").write_bytes(b"sft-adapter")
        record = {"kind": "SFT_only", "status": "verified", "adapter": str(adapter),
                  "adapter_sha256": retrain.sha256(adapter / "adapter_model.safetensors"),
                  "verification": {key: True for key in ("adapter_shapes_match", "selected_delta_match", "reload_success", "finite_logits")},
                  "weights": {"model.safetensors": {"bytes": 18, "sha256": retrain.sha256(base / "model.safetensors")}}}
        (base / "merge_provenance.json").write_text(json.dumps(record))
        return base, adapter, record

    def test_verified_sft_is_used_by_training_command(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            base, _, _ = self.fixture(root)
            record = retrain.validate_sft(root, "code")
            command = retrain.build_command(root, root / "source", root / "new-output", "code")
            self.assertEqual(command[command.index("--base_model") + 1], record["base_model"])
            self.assertEqual(record["base_model"], str(base.resolve()))
            for name, expected in (("--num_epochs", "3"), ("--lora_r", "32"), ("--lora_alpha", "64"),
                                   ("--beta", "0.1"), ("--learning_rate", "2e-6")):
                self.assertEqual(command[command.index(name) + 1], expected)

    def test_replacing_sft_with_original_weights_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            base, _, _ = self.fixture(root)
            (base / "model.safetensors").write_bytes(b"original-2b-weights")
            with self.assertRaisesRegex(ValueError, "weight hash mismatch"):
                retrain.validate_sft(root, "code")

    def test_changed_sft_adapter_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _, adapter, _ = self.fixture(root)
            (adapter / "adapter_model.safetensors").write_bytes(b"wrong-adapter")
            with self.assertRaisesRegex(ValueError, "adapter hash mismatch"):
                retrain.validate_sft(root, "code")

    def test_unverified_merge_and_adapter_directory_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            base, _, record = self.fixture(root)
            record["verification"]["selected_delta_match"] = False
            (base / "merge_provenance.json").write_text(json.dumps(record))
            with self.assertRaisesRegex(ValueError, "verification is incomplete"):
                retrain.validate_sft(root, "code")
            record["verification"]["selected_delta_match"] = True
            (base / "merge_provenance.json").write_text(json.dumps(record))
            (base / "adapter_config.json").write_text("{}")
            with self.assertRaisesRegex(ValueError, "complete SFT model"):
                retrain.validate_sft(root, "code")


if __name__ == "__main__":
    unittest.main()
