import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest


SOURCE = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("retrain_author_sft", SOURCE / "scripts/retrain_author_sft.py")
sft = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sft)


class Tokenizer:
    eos_token = "<EOS>"
    pad_token_id = 0

    def __init__(self, prompt_tokens=2):
        self.prompt_tokens = prompt_tokens

    def __call__(self, text, **kwargs):
        return {"input_ids": [11] * self.prompt_tokens if text == "prompt" else [21, 0]}


class RetrainAuthorSFTTests(unittest.TestCase):
    def base_fixture(self, root):
        base = root / "models/Qwen3.5-2B"
        base.mkdir(parents=True)
        (base / "download_provenance.json").write_text(json.dumps(
            {"repo_id": "Qwen/Qwen3.5-2B", "revision": "pinned"}))
        (base / "model.safetensors.index.json").write_text(json.dumps(
            {"weight_map": {"weight": "model.safetensors"}}))
        (base / "model.safetensors").write_bytes(b"original model")
        return base

    def test_requires_original_model_and_pins_all_weight_files(self):
        with tempfile.TemporaryDirectory() as directory:
            repo = Path(directory)
            base = self.base_fixture(repo)
            record = sft.validate_base(repo)
            self.assertEqual(record["weights"], ["model.safetensors"])
            self.assertEqual(record["files_sha256"]["model.safetensors"], sft.sha256(base / "model.safetensors"))
            for forbidden in ("adapter_config.json", "merge_provenance.json"):
                path = base / forbidden
                path.write_text("{}")
                with self.assertRaisesRegex(ValueError, "original 2B"):
                    sft.validate_base(repo)
                path.unlink()

    def test_missing_or_wrong_base_provenance_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            repo = Path(directory)
            base = self.base_fixture(repo)
            (base / "download_provenance.json").write_text(json.dumps({"repo_id": "other-model"}))
            with self.assertRaisesRegex(ValueError, "provenance"):
                sft.validate_base(repo)

    def test_commands_preserve_author_domain_hyperparameters_and_new_outputs(self):
        for domain, epochs, rate, accumulation in (("code", "3", "3e-05", "96"),
                                                   ("math", "2", "1.5e-05", "128"),
                                                   ("qa", "3", "3e-05", "96")):
            command = sft.training_command(Path("/new/run"), domain, Path("/original/2B"))
            values = dict(zip(command[3::2], command[4::2]))
            self.assertEqual(values["--base_model"], "/original/2B")
            self.assertEqual(values["--output_dir"], f"/new/run/adapters/{domain}")
            self.assertEqual(values["--dataset_path"], f"/new/run/inputs/{domain}.jsonl")
            self.assertEqual((values["--num_epochs"], values["--lr"], values["--grad_accum_steps"]),
                             (epochs, rate, accumulation))
            self.assertEqual((values["--lora_r"], values["--lora_alpha"], values["--max_seq_len"]),
                             ("64", "32", "2048"))
            self.assertEqual((values["--val_ratio"], values["--seed"]), ("0.15", "1"))
            self.assertNotIn("--resume_from_checkpoint", values)

    def audit(self, prompt_tokens):
        processor = SimpleNamespace(tokenizer=Tokenizer(prompt_tokens), apply_chat_template=lambda *args, **kwargs: "prompt")
        return sft.audit_examples("math", [{"problem": "question", "pitfalls": "guidance"}], processor)

    def test_audit_checks_full_sequence_loss_and_records_author_train_prompt(self):
        audit, rows, sample = self.audit(2)
        self.assertEqual(audit["no_target_row_indices"], [])
        self.assertEqual(audit["truncated_row_indices"], [])
        self.assertEqual(rows[0]["prompt_tokens"], 2)
        self.assertEqual(sample["labels"][:4], [11, 11, 21, -100])
        self.assertNotIn("[Example]", sample["messages"][0]["content"])
        self.assertEqual(sample["messages"][1]["content"], "PROBLEM:\nquestion")

    def test_audit_distinguishes_target_truncation_from_zero_supervised_target(self):
        audit, rows, _ = self.audit(2047)
        self.assertEqual(audit["truncated_row_indices"], [0])
        self.assertEqual(audit["no_target_row_indices"], [])
        self.assertEqual(rows[0]["retained_target_tokens"], 1)
        audit, _, _ = self.audit(2048)
        self.assertEqual(audit["no_target_row_indices"], [0])

    def test_frozen_inputs_detect_weight_or_data_changes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base = self.base_fixture(root)
            manifest = {"base": sft.validate_base(root), "source_sha256": {}}
            (root / "manifest.json").write_text(json.dumps(manifest))
            for domain in sft.DOMAINS:
                (root / domain).mkdir()
                data = root / domain / "data.json"
                data.write_text("[]")
                (root / domain / "training_source.json").write_text(json.dumps(
                    {"dataset_path": str(data), "dataset_sha256": sft.sha256(data)}))
            sft.assert_frozen(root)
            (root / "code/data.json").write_text("changed")
            with self.assertRaisesRegex(ValueError, "Frozen dataset"):
                sft.assert_frozen(root)
            (root / "code/data.json").write_text("[]")
            (base / "model.safetensors").write_bytes(b"changed model")
            with self.assertRaisesRegex(ValueError, "Original model"):
                sft.assert_frozen(root)


if __name__ == "__main__":
    unittest.main()
