"""Bounded QA training and reload check, not a task-quality experiment."""
import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import random
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.retrain_author_sft import SOURCE, load_training_module, save_json, sha256, validate_base


def sample_rows(examples):
    if len(examples) < 16:
        raise ValueError("QA smoke requires at least 16 examples")
    anchor = 18635 if len(examples) > 18635 else 0
    pool = [index for index in range(len(examples)) if index != anchor]
    indices = [anchor, *random.Random(42).sample(pool, 15)]
    return indices, [examples[index] for index in indices]


def training_command(run_dir, base):
    return [sys.executable, "-u", str(SOURCE / "qa/LoRA_Qwen35_QA_4090.py"),
            "--dataset_path", str(run_dir / "inputs.jsonl"), "--base_model", str(base),
            "--output_dir", str(run_dir / "adapter"), "--qwen_adapted", "--preserve_eos",
            "--num_epochs", "1", "--per_device_batch_size", "1", "--grad_accum_steps", "4",
            "--val_ratio", "0.25", "--seed", "1", "--lr", "3e-5", "--lora_r", "64",
            "--lora_alpha", "32", "--max_seq_len", "2048"]


def assert_frozen(run_dir):
    manifest = json.loads((run_dir / "manifest.json").read_text())
    if sha256(run_dir / "inputs.jsonl") != manifest["inputs_sha256"]:
        raise ValueError("Smoke inputs changed")
    if sha256(Path(manifest["original_dataset"])) != manifest["original_dataset_sha256"]:
        raise ValueError("Original QA dataset changed")
    for relative, expected in manifest["source_sha256"].items():
        if sha256(SOURCE / relative) != expected:
            raise ValueError(f"Smoke source changed: {relative}")
    for name, expected in manifest["base"]["files_sha256"].items():
        if sha256(Path(manifest["base"]["path"]) / name) != expected:
            raise ValueError(f"Original model changed: {name}")
    return manifest


def verify(run_dir):
    import torch
    from peft import PeftModel
    from safetensors.torch import load_file
    from transformers import AutoModelForMultimodalLM, AutoProcessor, BitsAndBytesConfig
    from shared.parammem_prompts import memory_messages, tokenize_sft
    from shared.qwen_sft import qwen_lora_plan, verify_qwen_lora
    manifest = assert_frozen(run_dir)
    adapter = run_dir / "adapter"
    weights = load_file(str(adapter / "adapter_model.safetensors"))
    if not weights or not all(torch.isfinite(value).all().item() for value in weights.values()):
        raise ValueError("Nonfinite or empty adapter")
    if not any(torch.count_nonzero(value).item() for name, value in weights.items() if "lora_B" in name):
        raise ValueError("No adapter B weight was updated")
    states = [json.loads(path.read_text()) for path in adapter.glob("checkpoint-*/trainer_state.json")]
    if not states:
        raise ValueError("Missing saved checkpoint")
    state = max(states, key=lambda value: value["global_step"])
    if state["global_step"] != 3 or abs(state["epoch"] - 1) > 1e-6:
        raise ValueError("Bounded training did not finish exactly 3 updates / 1 epoch")
    best = load_file(str(Path(state["best_model_checkpoint"]) / "adapter_model.safetensors"))
    if set(best) != set(weights) or any(not torch.equal(weights[name], best[name]) for name in best):
        raise ValueError("Saved adapter differs from selected checkpoint")
    logs = json.loads((adapter / "loss_history.json").read_text())
    losses = [item["loss"] for item in logs if "loss" in item]
    gradients = [item["grad_norm"] for item in logs if "grad_norm" in item]
    eval_losses = [item["eval_loss"] for item in logs if "eval_loss" in item]
    if not losses or not gradients or not eval_losses or not all(math.isfinite(value) for value in
                                                               losses + gradients + eval_losses):
        raise ValueError("Missing or nonfinite training/gradient/evaluation metrics")
    if max(gradients) <= 0:
        raise ValueError("No nonzero training gradient")
    base = manifest["base"]["path"]
    quantization = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                     bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
    model = AutoModelForMultimodalLM.from_pretrained(base, quantization_config=quantization,
                                                    dtype=torch.bfloat16, device_map={"": 0})
    model = PeftModel.from_pretrained(model, adapter).eval()
    plan = qwen_lora_plan(model.get_base_model())
    coverage = verify_qwen_lora(model, plan, require_trainable=False)
    saved_coverage = json.loads((adapter / "qwen_lora_coverage.json").read_text())
    config = json.loads((adapter / "adapter_config.json").read_text())
    if (config["target_modules"] != plan["target_regex"] or saved_coverage["targets"] != plan["targets"]
            or saved_coverage["include_mlp"] or config["r"] != 64 or config["lora_alpha"] != 32
            or Path(config["base_model_name_or_path"]).resolve() != Path(base).resolve()):
        raise ValueError("Reloaded adapter identity or attention-only scope mismatch")
    processor = AutoProcessor.from_pretrained(base)
    tokenizer = processor.tokenizer
    tokenizer.pad_token = tokenizer.eos_token
    example = manifest["reload_example"]
    record = tokenize_sft(processor, tokenizer, "qa", example["question"], example["decomposition"],
                          2048, preserve_eos=True)
    batch = {key: torch.tensor([value], device="cuda") for key, value in record.items()}
    prompt = processor.apply_chat_template(memory_messages("qa", example["question"]), tokenize=False,
                                           add_generation_prompt=True, enable_thinking=False)
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    with torch.inference_mode():
        reloaded_loss = model(**batch).loss.item()
        enabled = model(**inputs).logits[:, -1].float()
        with model.disable_adapter():
            original = model(**inputs).logits[:, -1].float()
        delta = (enabled - original).abs().max().item()
        if not math.isfinite(reloaded_loss) or not torch.isfinite(enabled).all() or not math.isfinite(delta) or delta <= 0:
            raise ValueError("Reloaded loss/logits invalid or adapter has no effect")
        stop = model.generation_config.eos_token_id
        stop = stop if isinstance(stop, list) else [stop]
        stop = sorted({value for value in [*stop, tokenizer.eos_token_id] if value is not None})
        generated = model.generate(**inputs, do_sample=False, max_new_tokens=128,
                                   eos_token_id=stop, pad_token_id=tokenizer.pad_token_id)
    new_ids = generated[0, inputs["input_ids"].shape[1]:]
    response = tokenizer.decode(new_ids, skip_special_tokens=True)
    if not response.strip():
        raise ValueError("Reloaded generation is empty")
    save_json(run_dir / "verification.json", {
        "status": "verified", "global_step": state["global_step"], "epoch": state["epoch"],
        "adapter_sha256": sha256(adapter / "adapter_model.safetensors"),
        "best_checkpoint_tensors_match": True, "training_loss": losses, "grad_norm": gradients,
        "eval_loss": eval_losses, "reloaded_loss": reloaded_loss, "max_logit_delta": delta,
        "injected_modules": coverage["injected_modules"], "include_mlp": coverage["include_mlp"],
        "training_trainable_parameters": saved_coverage["trainable_parameters"],
        "generated_tokens": len(new_ids), "ended_with_eos": int(new_ids[-1]) in stop,
        "response": response, "task_quality_validated": False})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--train-worker", action="store_true")
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    if args.train_worker:
        from transformers import set_seed
        manifest = assert_frozen(run_dir)
        set_seed(42)
        sys.argv = manifest["command"][2:]
        load_training_module("qa").main()
        return
    if args.verify:
        verify(run_dir)
        return
    from datasets import Dataset
    from shared.parammem_prompts import tokenize_sft
    from transformers import AutoProcessor
    run_dir.mkdir(parents=True, exist_ok=False)
    status = {"state": "preflight", "pid": os.getpid(), "started_utc": datetime.now(timezone.utc).isoformat()}
    save_json(run_dir / "status.json", status)
    try:
        memory = int(subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"], text=True).strip())
        if memory > 256:
            raise RuntimeError("GPU occupied; no existing process will be stopped")
        base = validate_base(args.repo_root.resolve())
        data = args.repo_root.resolve() / "dataset/multihop/train/sft.jsonl"
        examples = load_training_module("qa").load_qa_examples(str(data))
        indices, selected = sample_rows(examples)
        (run_dir / "inputs.jsonl").write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in selected) + "\n")
        split = Dataset.from_dict({"row_index": list(range(16))}).shuffle(seed=1).train_test_split(test_size=0.25, seed=1)
        processor = AutoProcessor.from_pretrained(base["path"])
        processor.tokenizer.pad_token = processor.tokenizer.eos_token
        eos_counts = []
        for example in selected:
            record = tokenize_sft(processor, processor.tokenizer, "qa", example["question"],
                                  example["decomposition"], 2048, preserve_eos=True)
            active = sum(record["attention_mask"])
            if record["labels"][active - 1] != processor.tokenizer.eos_token_id:
                raise ValueError("Smoke target EOS missing or truncated")
            eos_counts.append(record["labels"].count(processor.tokenizer.eos_token_id))
        manifest = {"kind": "qa_three_update_pipeline_smoke", "source_commit": args.source_commit,
                    "source_sha256": {str(path.relative_to(SOURCE)): sha256(path) for path in SOURCE.rglob("*.py")},
                    "base": base, "original_dataset": str(data), "original_dataset_sha256": sha256(data),
                    "original_row_indices": indices, "inputs_sha256": sha256(run_dir / "inputs.jsonl"),
                    "split": {name: split[name]["row_index"][:] for name in split},
                    "reload_example": selected[split["test"]["row_index"][0]],
                    "supervised_eos_per_sample": eos_counts, "command": training_command(run_dir, base["path"]),
                    "smoke_only_overrides": {"epochs": 1, "accumulation": 4, "validation_ratio": 0.25,
                                             "examples": 16, "expected_optimizer_updates": 3},
                    "full_sequence_loss": True, "include_mlp": False, "task_quality_validated": False}
        manifest["initialization_and_trainer_seed"] = 42
        save_json(run_dir / "manifest.json", manifest)
        environment = dict(os.environ, CUDA_VISIBLE_DEVICES="0", TOKENIZERS_PARALLELISM="false",
                           HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", WANDB_DISABLED="true")
        for phase, flag in (("training", "--train-worker"), ("verifying", "--verify")):
            command = [sys.executable, "-u", str(Path(__file__).resolve()), "--repo-root", str(args.repo_root),
                       "--run-dir", str(run_dir), "--source-commit", args.source_commit, flag]
            assert_frozen(run_dir)
            status["state"] = phase
            with (run_dir / f"{phase}.log").open("x") as log:
                child = subprocess.Popen(command, cwd=SOURCE, env=environment, stdin=subprocess.DEVNULL,
                                         stdout=log, stderr=subprocess.STDOUT)
                status["child_pid"] = child.pid
                save_json(run_dir / "status.json", status)
                code = child.wait()
            (run_dir / f"{phase}.exit").write_text(str(code) + "\n")
            if code:
                raise RuntimeError(f"{phase} failed: exit {code}")
        assert_frozen(run_dir)
        status.update(state="complete", completed_utc=datetime.now(timezone.utc).isoformat())
    except Exception as error:
        status.update(state="failed", error=str(error))
        raise
    finally:
        save_json(run_dir / "status.json", status)


if __name__ == "__main__":
    main()
