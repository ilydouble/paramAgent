import importlib.util
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("smoke_qwen_sft", ROOT / "scripts/smoke_qwen_sft.py")
smoke = importlib.util.module_from_spec(spec)
spec.loader.exec_module(smoke)


class SmokeQwenSFTTests(unittest.TestCase):
    def test_sampling_is_bounded_unique_reproducible_and_covers_known_eos_row(self):
        rows = [{"question": str(index), "decomposition": "guidance"} for index in range(20000)]
        indices, selected = smoke.sample_rows(rows)
        self.assertEqual(len(indices), 16)
        self.assertEqual(len(set(indices)), 16)
        self.assertEqual(indices[0], 18635)
        self.assertEqual(selected[0], rows[18635])
        self.assertEqual(smoke.sample_rows(rows), (indices, selected))
        self.assertEqual(len(smoke.sample_rows(rows[:16])[0]), 16)
        with self.assertRaises(ValueError):
            smoke.sample_rows(rows[:15])

    def test_command_uses_actual_qa_entrypoint_attention_only_and_three_updates(self):
        command = smoke.training_command(Path("/new/smoke"), Path("/original/2B"))
        self.assertTrue(command[2].endswith("qa/LoRA_Qwen35_QA_4090.py"))
        self.assertIn("--qwen_adapted", command)
        self.assertIn("--preserve_eos", command)
        self.assertNotIn("--qwen_include_mlp", command)
        self.assertNotIn("--resume_from_checkpoint", command)
        for flag, value in (("--num_epochs", "1"), ("--grad_accum_steps", "4"), ("--val_ratio", "0.25"),
                            ("--lora_r", "64"), ("--lora_alpha", "32"), ("--lr", "3e-5"),
                            ("--max_seq_len", "2048"), ("--output_dir", "/new/smoke/adapter")):
            self.assertEqual(command[command.index(flag) + 1], value)
        self.assertEqual(16 * (1 - 0.25) / 4, 3)


if __name__ == "__main__":
    unittest.main()
