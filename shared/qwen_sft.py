"""Explicit language-branch LoRA coverage for Qwen3.5 SFT."""
import json
from pathlib import Path
import re


POLICY = "qwen35_language_projections_v1"
PROJECTIONS = {
    "full_attention": ("self_attn", {"q_proj", "k_proj", "v_proj", "o_proj"}),
    "linear_attention": ("linear_attn", {"in_proj_qkv", "in_proj_z", "in_proj_a", "in_proj_b", "out_proj"}),
}
MLP = {"gate_proj", "up_proj", "down_proj"}


def qwen_lora_plan(model):
    import torch
    config = model.config
    text_config = getattr(config, "text_config", config)
    if getattr(config, "model_type", None) not in {"qwen3_5", "qwen3_5_text"}:
        raise ValueError("Qwen adaptation requires a Qwen3.5 dense model")
    layer_types = list(text_config.layer_types)
    if not layer_types or len(layer_types) != text_config.num_hidden_layers:
        raise ValueError("Qwen layer inventory does not match num_hidden_layers")
    modules = dict(model.named_modules())
    roots = [root for root in ("model.language_model", "language_model", "model")
             if f"{root}.layers.0" in modules and (root != "model" or text_config is config)]
    if len(roots) != 1:
        raise ValueError(f"Ambiguous or missing Qwen language branch: {roots}")
    root = roots[0]
    layer_prefix = root + ".layers."
    actual_layers = {int(name[len(layer_prefix):]) for name in modules
                     if name.startswith(layer_prefix) and name[len(layer_prefix):].isdigit()}
    if actual_layers != set(range(len(layer_types))):
        raise ValueError("Qwen actual decoder layers differ from config")
    targets = []
    coverage = []
    for index, layer_type in enumerate(layer_types):
        if layer_type not in PROJECTIONS:
            raise ValueError(f"Unsupported Qwen layer type: {layer_type}")
        block, projections = PROJECTIONS[layer_type]
        expected = [f"{root}.layers.{index}.{block}.{leaf}" for leaf in sorted(projections)]
        expected += [f"{root}.layers.{index}.mlp.{leaf}" for leaf in sorted(MLP)]
        for name in expected:
            module = modules.get(name)
            linear = module.get_base_layer() if hasattr(module, "get_base_layer") else module
            if not isinstance(linear, torch.nn.Linear):
                raise ValueError(f"Missing or unsupported Qwen linear projection: {name}")
        targets.extend(expected)
        coverage.append({"layer": index, "type": layer_type, "projections": expected})
    return {"policy": POLICY, "language_root": root, "targets": sorted(targets), "coverage": coverage,
            "target_regex": "^(?:" + "|".join(re.escape(name) for name in sorted(targets)) + ")$"}


def verify_qwen_lora(model, plan, require_trainable=True):
    base = model.get_base_model()
    actual = {name for name, module in base.named_modules() if hasattr(module, "lora_A")}
    if actual != set(plan["targets"]):
        raise ValueError(f"Qwen LoRA injection mismatch: missing={set(plan['targets']) - actual}, "
                         f"unexpected={actual - set(plan['targets'])}")
    trainable = [(name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad]
    if any(".lora_A." not in name and ".lora_B." not in name for name, _ in trainable):
        raise ValueError("Qwen adaptation unexpectedly trains non-LoRA parameters")
    if require_trainable:
        for name, module in base.named_modules():
            if name in actual:
                for family in (module.lora_A, module.lora_B):
                    if "default" not in family or not all(parameter.requires_grad
                                                         for parameter in family["default"].parameters()):
                        raise ValueError(f"Qwen LoRA projection is not trainable: {name}")
        if not trainable:
            raise ValueError("Qwen adaptation has no trainable parameters")
    return {**plan, "injected_modules": len(actual),
            "trainable_parameters": sum(parameter.numel() for _, parameter in trainable)}


def save_qwen_lora_report(model, plan, output_dir):
    report = verify_qwen_lora(model, plan)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "qwen_lora_coverage.json").write_text(json.dumps(report, indent=2) + "\n")
    return report
