import importlib.util
import json
from pathlib import Path
import tempfile
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch


SOURCE = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("retrain_author_sft", SOURCE / "scripts/retrain_author_sft.py")
sft = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sft)


class Tokenizer:
    eos_token = "<EOS>"
    eos_token_id = 0
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
            self.assertNotIn("--preserve_eos", command)

    def test_revised_commands_pass_explicit_flag_to_every_domain(self):
        for domain in sft.DOMAINS:
            original = sft.training_command(Path("/new/run"), domain, Path("/original/2B"))
            revised = sft.training_command(Path("/new/run"), domain, Path("/original/2B"), preserve_eos=True)
            self.assertEqual(revised, original + ["--preserve_eos"])

    def audit(self, prompt_tokens, preserve_eos=False):
        processor = SimpleNamespace(tokenizer=Tokenizer(prompt_tokens), apply_chat_template=lambda *args, **kwargs: "prompt")
        return sft.audit_examples("math", [{"problem": "question", "pitfalls": "guidance"}], processor,
                                  preserve_eos=preserve_eos)

    def test_revised_audit_records_actual_eos_supervision_and_padding(self):
        for preserve_eos in (False, True):
            audit, rows, sample = self.audit(2, preserve_eos=preserve_eos)
            self.assertEqual(audit["preserve_eos"], preserve_eos)
            self.assertEqual(audit["retained_answer_eos_tokens"], 1)
            self.assertEqual(audit["supervised_answer_eos_tokens"], int(preserve_eos))
            self.assertEqual(rows[0]["supervised_answer_eos_tokens"], int(preserve_eos))
            self.assertEqual(sample["labels"][:5], [11, 11, 21, 0 if preserve_eos else -100, -100])
            self.assertEqual(sample["attention_mask"][:5], [1, 1, 1, 1, 0])

    def test_revised_audit_reports_eos_clipping_without_inventing_supervision(self):
        for prompt_tokens, retained_eos in ((2046, 1), (2047, 0), (2048, 0)):
            audit, rows, _ = self.audit(prompt_tokens, preserve_eos=True)
            self.assertEqual(audit["retained_answer_eos_tokens"], retained_eos)
            self.assertEqual(audit["supervised_answer_eos_tokens"], retained_eos)
            self.assertEqual(rows[0]["retained_answer_eos_tokens"], retained_eos)
            self.assertEqual(audit["no_target_row_indices"], [0] if prompt_tokens == 2048 else [])

    def test_launcher_forwards_mode_to_preflight_and_status_without_training(self):
        for preserve_eos in (False, True):
            with tempfile.TemporaryDirectory() as directory:
                run_dir = Path(directory) / "new-run"
                command = ["retrain", "--repo-root", directory, "--run-dir", str(run_dir)]
                if preserve_eos:
                    command.append("--preserve_eos")
                with patch.object(sys, "argv", command), patch.object(sft, "preflight") as preflight:
                    sft.main()
                preflight.assert_called_once_with(Path(directory).resolve(), run_dir.resolve(),
                                                   preserve_eos=preserve_eos)
                status = json.loads((run_dir / "status.json").read_text())
                self.assertEqual(status["state"], "preflight_complete")
                self.assertEqual(status["preserve_eos"], preserve_eos)

    def test_preflight_records_same_loss_mode_in_manifest_commands_and_samples(self):
        for preserve_eos in (False, True):
            with tempfile.TemporaryDirectory() as directory:
                repo = Path(directory)
                self.base_fixture(repo)
                run_dir = repo / "run"
                run_dir.mkdir()
                modules = {}
                for domain, config in sft.DOMAINS.items():
                    data = repo / "dataset" / config["dataset"]
                    data.parent.mkdir(parents=True)
                    examples = [{config["question"]: "question", config["answer"]: "guidance"}] * 2
                    data.write_text(json.dumps(examples))
                    modules[domain] = SimpleNamespace(**{config["loader"]: lambda path: json.loads(Path(path).read_text())})
                dataset = SimpleNamespace()
                dataset.shuffle = lambda **kwargs: dataset
                dataset.train_test_split = lambda **kwargs: {"train": {"row_index": [0]}, "test": {"row_index": [1]}}
                processor = SimpleNamespace(tokenizer=Tokenizer(), apply_chat_template=lambda *args, **kwargs: "prompt")
                dependencies = {"datasets": SimpleNamespace(Dataset=SimpleNamespace(from_dict=lambda rows: dataset)),
                                "transformers": SimpleNamespace(TrainingArguments=type("Arguments", (), {}),
                                    AutoProcessor=SimpleNamespace(from_pretrained=lambda path: processor))}
                required = {"eval_strategy", "save_strategy", "load_best_model_at_end", "warmup_steps", "optim",
                            "gradient_accumulation_steps", "lr_scheduler_type", "bf16", "gradient_checkpointing"}
                with patch.dict(sys.modules, dependencies), patch.object(sft, "load_training_module", modules.get), \
                        patch.object(sft.inspect, "signature", return_value=SimpleNamespace(parameters=required)), \
                        patch.object(sft.importlib.metadata, "version", return_value="test"):
                    sft.preflight(repo, run_dir, preserve_eos=preserve_eos)
                manifest = json.loads((run_dir / "manifest.json").read_text())
                self.assertEqual(manifest["loss"], "full_sequence_padding_only_masked" if preserve_eos
                                 else "full_sequence_pad_eos_masked")
                self.assertEqual(manifest["preserve_eos"], preserve_eos)
                for domain in sft.DOMAINS:
                    record = json.loads((run_dir / domain / "training_source.json").read_text())
                    sample = json.loads((run_dir / domain / "prompt_sample.json").read_text())
                    self.assertEqual(record["loss"], manifest["loss"])
                    self.assertEqual(record["preserve_eos"], preserve_eos)
                    self.assertEqual("--preserve_eos" in record["command"], preserve_eos)
                    self.assertEqual(sample["labels"][3], 0 if preserve_eos else -100)

    def test_worker_uses_recorded_training_flag_instead_of_losing_it_in_subprocess(self):
        for preserve_eos in (False, True):
            with tempfile.TemporaryDirectory() as directory:
                run_dir = Path(directory)
                (run_dir / "qa").mkdir()
                command = sft.training_command(run_dir, "qa", Path("/base"), preserve_eos=preserve_eos)
                (run_dir / "qa/training_source.json").write_text(json.dumps({"command": command}))
                captured = []
                module = SimpleNamespace(main=lambda: captured.append(sys.argv[:]))
                with patch.dict(sys.modules, {"transformers": SimpleNamespace(set_seed=lambda seed: None)}), \
                        patch.object(sft, "assert_frozen"), patch.object(sft, "load_training_module", return_value=module), \
                        patch.object(sys, "argv", []):
                    sft.train_worker(run_dir, "qa")
                self.assertEqual(captured, [command[2:]])

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
