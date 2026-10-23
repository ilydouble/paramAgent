import argparse
import ast
from contextlib import redirect_stderr
import io
import json
from pathlib import Path
import re
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from shared.qwen_sft import MLP, PROJECTIONS, qwen_lora_plan, save_qwen_lora_report, verify_qwen_lora


ROOT = Path(__file__).resolve().parents[1]


class Linear:
    pass


class Parameter:
    requires_grad = True

    def numel(self):
        return 4


class Adapter:
    def __init__(self):
        self.weight = Parameter()

    def parameters(self):
        return [self.weight]


class WrappedLinear:
    def __init__(self):
        self.lora_A = {"default": Adapter()}
        self.lora_B = {"default": Adapter()}

    def get_base_layer(self):
        return Linear()


class QwenSFTTests(unittest.TestCase):
    def setUp(self):
        self.torch = patch.dict(sys.modules, {"torch": SimpleNamespace(nn=SimpleNamespace(Linear=Linear))})
        self.torch.start()
        self.addCleanup(self.torch.stop)

    def model(self, text_only=False):
        layer_types = ["linear_attention", "full_attention", "linear_attention"]
        text = SimpleNamespace(layer_types=layer_types, num_hidden_layers=len(layer_types))
        config = text if text_only else SimpleNamespace(text_config=text)
        config.model_type = "qwen3_5_text" if text_only else "qwen3_5"
        root = "model" if text_only else "model.language_model"
        modules = {root: object(), "visual.layers.0.self_attn.q_proj": Linear(), "lm_head": Linear()}
        for index, layer_type in enumerate(layer_types):
            modules[f"{root}.layers.{index}"] = object()
            block, projections = PROJECTIONS[layer_type]
            for family, leaves in ((block, projections), ("mlp", MLP)):
                for leaf in leaves:
                    modules[f"{root}.layers.{index}.{family}.{leaf}"] = Linear()
        model = SimpleNamespace(config=config, named_modules=lambda: modules.items(), modules=modules)
        return model

    def injected(self):
        base = self.model()
        plan = qwen_lora_plan(base)
        for name in plan["targets"]:
            base.modules[name] = WrappedLinear()
        parameters = []
        for name in plan["targets"]:
            module = base.modules[name]
            parameters.extend((f"{name}.{family}.default.weight", getattr(module, family)["default"].weight)
                              for family in ("lora_A", "lora_B"))
        model = SimpleNamespace(get_base_model=lambda: base, named_parameters=lambda: parameters)
        return model, base, plan, parameters

    def test_exact_paths_cover_both_attention_types_and_mlp_but_not_vision_or_head(self):
        for text_only in (False, True):
            model = self.model(text_only)
            plan = qwen_lora_plan(model)
            self.assertEqual(len(plan["targets"]), 23)
            self.assertEqual([len(layer["projections"]) for layer in plan["coverage"]], [8, 7, 8])
            matched = {name for name in model.modules if re.fullmatch(plan["target_regex"], name)}
            self.assertEqual(matched, set(plan["targets"]))
            self.assertFalse(re.fullmatch(plan["target_regex"], "visual." + plan["targets"][0]))

    def test_missing_projection_wrong_type_and_layer_inventory_fail_closed(self):
        for leaf in ("in_proj_qkv", "in_proj_z", "in_proj_a", "in_proj_b", "out_proj"):
            model = self.model()
            del model.modules[f"model.language_model.layers.0.linear_attn.{leaf}"]
            with self.assertRaisesRegex(ValueError, "projection"):
                qwen_lora_plan(model)
        model = self.model()
        model.modules["model.language_model.layers.0.mlp.up_proj"] = object()
        with self.assertRaisesRegex(ValueError, "projection"):
            qwen_lora_plan(model)
        model = self.model()
        del model.modules["model.language_model.layers.1"]
        with self.assertRaisesRegex(ValueError, "decoder layers"):
            qwen_lora_plan(model)

    def test_unknown_architecture_and_ambiguous_language_root_are_rejected(self):
        model = self.model()
        model.config.model_type = "qwen3"
        with self.assertRaisesRegex(ValueError, "Qwen3.5"):
            qwen_lora_plan(model)
        model = self.model()
        model.modules["language_model.layers.0"] = object()
        with self.assertRaisesRegex(ValueError, "Ambiguous"):
            qwen_lora_plan(model)
        model = self.model()
        model.config.text_config.layer_types[0] = "unknown"
        with self.assertRaisesRegex(ValueError, "layer type"):
            qwen_lora_plan(model)

    def test_injection_verification_and_saved_report(self):
        model, base, plan, _ = self.injected()
        self.assertEqual(qwen_lora_plan(base), plan)
        with tempfile.TemporaryDirectory() as directory:
            report = save_qwen_lora_report(model, plan, directory)
            self.assertEqual(report["injected_modules"], 23)
            self.assertEqual(report["trainable_parameters"], 184)
            self.assertEqual(json.loads((Path(directory) / "qwen_lora_coverage.json").read_text()), report)

    def test_missing_and_unexpected_adapter_injection_are_rejected(self):
        for extra in (False, True):
            model, base, plan, _ = self.injected()
            if extra:
                base.modules["visual.layers.0.self_attn.q_proj"] = WrappedLinear()
            else:
                base.modules[plan["targets"][0]] = Linear()
            with self.assertRaisesRegex(ValueError, "injection mismatch"):
                verify_qwen_lora(model, plan)

    def test_frozen_adapter_or_unexpected_base_training_is_rejected(self):
        model, _, plan, parameters = self.injected()
        parameters[0][1].requires_grad = False
        with self.assertRaisesRegex(ValueError, "not trainable"):
            verify_qwen_lora(model, plan)
        self.assertEqual(verify_qwen_lora(model, plan, require_trainable=False)["injected_modules"], 23)
        model, _, plan, parameters = self.injected()
        parameters.append(("visual.weight", Parameter()))
        with self.assertRaisesRegex(ValueError, "non-LoRA"):
            verify_qwen_lora(model, plan)

    def test_all_training_entrypoints_enable_eos_and_require_fresh_output(self):
        for domain, label in (("code", "Code"), ("math", "Math"), ("qa", "QA")):
            path = ROOT / domain / f"LoRA_Qwen35_{label}_4090.py"
            tree = ast.parse(path.read_text())
            function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "parse_args")
            namespace = {"argparse": argparse, "Path": Path}
            exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), "exec"), namespace)
            with tempfile.TemporaryDirectory() as directory:
                command = [str(path), "--qwen_adapted", "--output_dir", str(Path(directory) / "fresh")]
                with patch.object(sys, "argv", command):
                    args = namespace["parse_args"]()
                self.assertTrue(args.qwen_adapted)
                self.assertTrue(args.preserve_eos)
                self.assertEqual((args.lora_r, args.lora_alpha), (64, 32))
                for flags in (("--resume_from_checkpoint", "old"), ("--output_dir", directory)):
                    with patch.object(sys, "argv", command + list(flags)), redirect_stderr(io.StringIO()):
                        with self.assertRaises(SystemExit):
                            namespace["parse_args"]()
            source = path.read_text()
            self.assertIn('qwen_plan["target_regex"] if qwen_plan else select_lora_modules(model)', source)
            self.assertIn('save_qwen_lora_report(model, qwen_plan, args.output_dir)', source)


if __name__ == "__main__":
    unittest.main()
