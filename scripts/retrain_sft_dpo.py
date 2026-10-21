import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timezone


DOMAINS = {
    "code": ("Code", "Qwen3.5-2B-code-merged2", "code", "code2", 0.10),
    "math": ("Math", "Qwen3.5-2B-math-merged2", "math", "math2", 0.10),
    "qa": ("QA", "Qwen3.5-2B-qa-merged", "multihop", "qa", 0.05),
}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def validate_sft(repo, domain):
    _, model_name, _, adapter_name, _ = DOMAINS[domain]
    base = (repo / "models" / model_name).resolve(strict=True)
    provenance_path = base / "merge_provenance.json"
    provenance = json.loads(provenance_path.read_text())
    if provenance.get("kind") != "SFT_only" or provenance.get("status") != "verified":
        raise ValueError(f"{domain}: a verified SFT merge is required")
    adapter = (repo / "lora-qwen3.5-2b" / f"lora-qwen3.5-2b-{adapter_name}").resolve(strict=True)
    if Path(provenance["adapter"]).name != adapter.name:
        raise ValueError(f"{domain}: SFT adapter identity mismatch")
    if sha256(adapter / "adapter_model.safetensors") != provenance["adapter_sha256"]:
        raise ValueError(f"{domain}: SFT adapter hash mismatch")
    checks = provenance.get("verification", {})
    if not all(checks.get(key) is True for key in ("adapter_shapes_match", "selected_delta_match", "reload_success", "finite_logits")):
        raise ValueError(f"{domain}: SFT merge verification is incomplete")
    weights = provenance.get("weights", {})
    if not weights or (base / "adapter_config.json").exists():
        raise ValueError(f"{domain}: expected a complete SFT model, not an adapter")
    actual_files = {path.name for path in base.glob("*.safetensors")}
    if actual_files != set(weights):
        raise ValueError(f"{domain}: SFT weight file inventory mismatch")
    hashes = {}
    for name, expected in weights.items():
        if Path(name).name != name:
            raise ValueError("Invalid weight filename")
        path = base / name
        actual = sha256(path)
        if path.stat().st_size != expected["bytes"] or actual != expected["sha256"]:
            raise ValueError(f"{domain}: SFT weight hash mismatch: {name}")
        hashes[name] = actual
    return {"base_model": str(base), "weights_sha256": hashes,
            "sft_adapter": str(adapter), "sft_adapter_sha256": provenance["adapter_sha256"],
            "merge_provenance_sha256": sha256(provenance_path),
            "historical_base_revision_verified": provenance.get("historical_base_revision_verified", False)}


def build_command(repo, source, output, domain):
    label, model_name, dataset_name, _, val_ratio = DOMAINS[domain]
    command = [sys.executable, "-u", str(source / domain / f"LoRA_Qwen35_{label}_DPO_4090.py"),
               "--base_model", str((repo / "models" / model_name).resolve()),
               "--dataset_path", str(repo / "dataset" / dataset_name / "train/dpo.jsonl"),
               "--output_dir", str(output), "--num_epochs", "3", "--lora_r", "32",
               "--lora_alpha", "64", "--beta", "0.1", "--learning_rate", "2e-6",
               "--per_device_batch_size", "1", "--grad_accum_steps", "16",
               "--lora_dropout", "0.05", "--max_length", "3072",
               "--max_prompt_length", "1024" if domain == "code" else "2048",
               "--val_ratio", str(val_ratio), "--save_strategy", "epoch", "--seed", "42"]
    if domain == "code":
        command += ["--max_chars", "3000"]
    return command


def verify_adapter(run_dir, domain):
    import torch
    from peft import PeftModel
    from safetensors.torch import load_file
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    record = json.loads((run_dir / domain / "training_source.json").read_text())
    output = run_dir / "adapters" / domain
    config = json.loads((output / "adapter_config.json").read_text())
    if Path(config["base_model_name_or_path"]).resolve() != Path(record["base_model"]):
        raise ValueError("Saved DPO adapter does not reference the verified SFT base")
    if config["r"] != 32 or config["lora_alpha"] != 64:
        raise ValueError("Saved DPO adapter does not use the requested rank/alpha")
    tensors = load_file(str(output / "adapter_model.safetensors"))
    if not tensors or not all(torch.isfinite(tensor).all().item() for tensor in tensors.values()):
        raise ValueError("DPO adapter contains invalid weights")
    if not any(torch.count_nonzero(tensor).item() for name, tensor in tensors.items() if "lora_B" in name):
        raise ValueError("DPO adapter has no trained LoRA B weights")
    quantization = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                     bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
    base = AutoModelForCausalLM.from_pretrained(record["base_model"], quantization_config=quantization,
                                               dtype=torch.bfloat16, device_map={"": 0})
    model = PeftModel.from_pretrained(base, output).eval()
    tokenizer = AutoTokenizer.from_pretrained(record["base_model"])
    probes = {"code": "FUNC_SIGNATURE:\ndef add(a: int, b: int) -> int:\n    return a + b",
              "math": "PROBLEM:\nA triangle has perimeter 12. What is the side length if equilateral?",
              "qa": "[Question]: Which country contains Paris?"}
    prompt = tokenizer.apply_chat_template([{"role": "user", "content": probes[domain]}],
                                           tokenize=False, add_generation_prompt=True, enable_thinking=False)
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    with torch.inference_mode():
        enabled = model(**inputs).logits[:, -1].float()
        with model.disable_adapter():
            reference = model(**inputs).logits[:, -1].float()
        difference = (enabled - reference).abs().max().item()
        if not torch.isfinite(enabled).all() or not torch.isfinite(reference).all() or difference <= 0:
            raise ValueError("DPO does not produce finite, changed logits relative to SFT")
        generated = model.generate(**inputs, do_sample=False, max_new_tokens=64,
                                   pad_token_id=tokenizer.eos_token_id)
    text = tokenizer.decode(generated[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    if not text.strip():
        raise ValueError("DPO smoke generation is empty")
    result = {"status": "verified", "adapter_sha256": sha256(output / "adapter_model.safetensors"),
              "tensor_count": len(tensors), "max_logit_delta_from_sft": difference,
              "smoke_response": text, "base_model": record["base_model"]}
    save_json(run_dir / domain / "verification.json", result)
    print(json.dumps(result, ensure_ascii=False), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--allow-legacy-question-only", action="store_true",
                        help="Explicitly reproduce legacy text-pair training, not the ICASSP execution-grounded protocol")
    parser.add_argument("--verify-domain", choices=tuple(DOMAINS))
    args = parser.parse_args()
    if args.train and not args.allow_legacy_question_only:
        parser.error("This launcher consumes legacy question-only preference pairs. "
                     "ICASSP requires failed traces, four guidance candidates and frozen-Actor execution evidence. "
                     "It cannot be used for paper-aligned DPO; --allow-legacy-question-only is only for historical reproduction.")
    repo, run_dir = args.repo_root.resolve(), args.run_dir.resolve()
    source = Path(__file__).resolve().parents[1]
    if args.verify_domain:
        verify_adapter(run_dir, args.verify_domain)
        return
    run_dir.mkdir(parents=True, exist_ok=False)
    status = {"state": "preflight", "pid": os.getpid(), "started_utc": datetime.now(timezone.utc).isoformat(),
              "completed": [], "data_scope": "existing preference data; no new outcome-grounded pair collection"}
    save_json(run_dir / "status.json", status)
    try:
        for domain, (label, _, dataset_name, _, _) in DOMAINS.items():
            record = validate_sft(repo, domain)
            domain_dir = run_dir / domain
            domain_dir.mkdir()
            output = run_dir / "adapters" / domain
            command = build_command(repo, source, output, domain)
            record.update({"command": command, "dataset_sha256": sha256(repo / "dataset" / dataset_name / "train/dpo.jsonl"),
                           "source_sha256": {str(path.relative_to(source)): sha256(path) for path in
                               (source / domain / f"LoRA_Qwen35_{label}_DPO_4090.py",
                                source / domain / f"LoRA_Qwen35_{label}_4090.py",
                                source / "shared/parammem_prompts.py",
                                source / domain / f"LoRA_Llama3_{label}_multigpu.py", Path(__file__))},
                           "reference": "frozen SFT base with new DPO adapter disabled"})
            save_json(domain_dir / "training_source.json", record)
            print(f"SFT_VERIFIED {domain}: {record['base_model']}", flush=True)
        if not args.train:
            status["state"] = "preflight_complete"
            save_json(run_dir / "status.json", status)
            return
        environment = dict(os.environ, CUDA_VISIBLE_DEVICES="0", TOKENIZERS_PARALLELISM="false",
                           HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", WANDB_DISABLED="true")
        for domain in DOMAINS:
            status.update(state="training", domain=domain)
            save_json(run_dir / "status.json", status)
            command = json.loads((run_dir / domain / "training_source.json").read_text())["command"]
            with (run_dir / domain / "train.log").open("x") as log:
                result = subprocess.run(command, cwd=source, env=environment, stdout=log, stderr=subprocess.STDOUT)
            (run_dir / domain / "train.exit").write_text(str(result.returncode) + "\n")
            if result.returncode:
                raise RuntimeError(f"{domain} training failed: exit {result.returncode}")
            status["state"] = "verifying"
            save_json(run_dir / "status.json", status)
            with (run_dir / domain / "verify.log").open("x") as log:
                result = subprocess.run([sys.executable, str(Path(__file__).resolve()), "--repo-root", str(repo),
                                         "--run-dir", str(run_dir), "--verify-domain", domain],
                                        cwd=source, env=environment, stdout=log, stderr=subprocess.STDOUT)
            (run_dir / domain / "verify.exit").write_text(str(result.returncode) + "\n")
            if result.returncode:
                raise RuntimeError(f"{domain} verification failed: exit {result.returncode}")
            status["completed"].append(domain)
        status["state"] = "complete"
        save_json(run_dir / "status.json", status)
    except Exception as error:
        status.update(state="failed", error=str(error))
        save_json(run_dir / "status.json", status)
        raise


if __name__ == "__main__":
    main()
