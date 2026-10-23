"""Train separate Qwen SFT adapters using the pinned author's training prompts."""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import importlib.util
import inspect
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys


SOURCE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SOURCE))
DOMAINS = {
    "code": {"label": "Code", "dataset": "code/train/code.json", "loader": "load_code_examples",
             "question": "func_sign", "answer": "answer", "epochs": 3, "lr": 3e-5, "accumulation": 96},
    "math": {"label": "Math", "dataset": "math/train/math.json", "loader": "load_examples",
             "question": "problem", "answer": "pitfalls", "epochs": 2, "lr": 1.5e-5, "accumulation": 128},
    "qa": {"label": "QA", "dataset": "multihop/train/sft.jsonl", "loader": "load_qa_examples",
           "question": "question", "answer": "decomposition", "epochs": 3, "lr": 3e-5, "accumulation": 96},
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


def validate_base(repo):
    base = (repo / "models/Qwen3.5-2B").resolve(strict=True)
    if (base / "adapter_config.json").exists() or (base / "merge_provenance.json").exists():
        raise ValueError("SFT must start from the original 2B model, not an adapter or merged SFT")
    provenance = json.loads((base / "download_provenance.json").read_text())
    if provenance.get("repo_id") != "Qwen/Qwen3.5-2B" or not provenance.get("revision"):
        raise ValueError("Missing original Qwen2B download provenance")
    index = json.loads((base / "model.safetensors.index.json").read_text())
    weights = sorted(set(index["weight_map"].values()))
    if not weights or any(Path(name).name != name for name in weights):
        raise ValueError("Invalid original-model weight inventory")
    files = {path.name for path in base.iterdir() if path.is_file()
             and path.suffix in {".json", ".jinja", ".txt"}}
    files.update(weights)
    return {"path": str(base), "download_provenance": provenance,
            "weights": weights, "files_sha256": {name: sha256(base / name) for name in sorted(files)}}


def training_command(run_dir, domain, base, preserve_eos=False):
    config = DOMAINS[domain]
    command = [sys.executable, "-u", str(SOURCE / domain / f"LoRA_Qwen35_{config['label']}_4090.py"),
            "--base_model", str(base), "--dataset_path", str(run_dir / "inputs" / f"{domain}.jsonl"),
            "--output_dir", str(run_dir / "adapters" / domain),
            "--num_epochs", str(config["epochs"]), "--per_device_batch_size", "1",
            "--grad_accum_steps", str(config["accumulation"]), "--lr", str(config["lr"]),
            "--lora_r", "64", "--lora_alpha", "32", "--lora_dropout", "0.05",
            "--max_seq_len", "2048", "--val_ratio", "0.15", "--seed", "1", "--warmup_ratio", "0.03"]
    if preserve_eos:
        command.append("--preserve_eos")
    return command


def load_training_module(domain):
    path = SOURCE / domain / f"LoRA_Qwen35_{DOMAINS[domain]['label']}_4090.py"
    spec = importlib.util.spec_from_file_location(f"author_sft_{domain}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def audit_examples(domain, examples, processor, preserve_eos=False):
    from shared.parammem_prompts import memory_messages, tokenize_sft
    config = DOMAINS[domain]
    tokenizer = processor.tokenizer
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    lengths = []
    truncated = []
    no_target = []
    rows = []
    for index, example in enumerate(examples):
        question, answer = example[config["question"]], example[config["answer"]]
        if not question.strip() or not answer.strip():
            raise ValueError(f"{domain}: empty input/target at row {index}")
        messages = memory_messages(domain, question, "train")
        prompt = processor.apply_chat_template(messages, tokenize=False,
                                               add_generation_prompt=True, enable_thinking=False)
        prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        answer_ids = tokenizer(answer.strip() + tokenizer.eos_token, add_special_tokens=False)["input_ids"]
        sequence_length = len(prompt_ids) + len(answer_ids)
        lengths.append(sequence_length)
        if sequence_length > 2048:
            truncated.append(index)
        target = answer_ids[:max(0, 2048 - len(prompt_ids))]
        if not any(token != tokenizer.pad_token_id for token in target):
            no_target.append(index)
        retained_eos = target.count(tokenizer.eos_token_id)
        supervised_eos = retained_eos if preserve_eos or tokenizer.eos_token_id != tokenizer.pad_token_id else 0
        if index == 0:
            record = tokenize_sft(processor, tokenizer, domain, question, answer, 2048,
                                  preserve_eos=preserve_eos)
            expected_labels = [token if (active if preserve_eos else token != tokenizer.pad_token_id) else -100
                               for token, active in zip(record["input_ids"], record["attention_mask"])]
            if record["labels"] != expected_labels:
                raise ValueError("SFT full-sequence loss invariant failed")
            sample = {"messages": messages, "rendered_prompt": prompt, "answer": answer,
                      "input_ids": record["input_ids"], "labels": record["labels"],
                      "attention_mask": record["attention_mask"], "preserve_eos": preserve_eos}
        rows.append({"row_index": index, "prompt_tokens": len(prompt_ids), "answer_tokens": len(answer_ids),
                     "retained_answer_eos_tokens": retained_eos, "supervised_answer_eos_tokens": supervised_eos,
                     "retained_target_tokens": len(target), "sha256": hashlib.sha256(
                         json.dumps(example, sort_keys=True, ensure_ascii=False).encode()).hexdigest()})
    if not lengths:
        raise ValueError(f"{domain}: empty dataset")
    ordered = sorted(lengths)
    return {"rows": len(rows), "max_sequence_tokens": ordered[-1],
            "preserve_eos": preserve_eos,
            "retained_answer_eos_tokens": sum(row["retained_answer_eos_tokens"] for row in rows),
            "supervised_answer_eos_tokens": sum(row["supervised_answer_eos_tokens"] for row in rows),
            "median_sequence_tokens": ordered[len(ordered) // 2],
            "truncated_row_indices": truncated, "no_target_row_indices": no_target}, rows, sample


def assert_frozen(run_dir):
    manifest = json.loads((run_dir / "manifest.json").read_text())
    for relative, expected in manifest["source_sha256"].items():
        if sha256(SOURCE / relative) != expected:
            raise ValueError(f"Frozen training source changed: {relative}")
    base = Path(manifest["base"]["path"])
    for name, expected in manifest["base"]["files_sha256"].items():
        if sha256(base / name) != expected:
            raise ValueError(f"Original model changed: {name}")
    for domain in DOMAINS:
        record = json.loads((run_dir / domain / "training_source.json").read_text())
        if sha256(Path(record["dataset_path"])) != record["dataset_sha256"]:
            raise ValueError(f"Frozen dataset changed: {domain}")


def preflight(repo, run_dir, preserve_eos=False):
    from datasets import Dataset
    from transformers import AutoProcessor, TrainingArguments
    from shared.parammem_prompts import AUTHOR_COMMIT, author_constants
    required = {"eval_strategy", "save_strategy", "load_best_model_at_end", "warmup_steps",
                "optim", "gradient_accumulation_steps", "lr_scheduler_type", "bf16", "gradient_checkpointing"}
    if required - set(inspect.signature(TrainingArguments.__init__).parameters):
        raise ValueError("TrainingArguments would silently discard required author settings")
    base = validate_base(repo)
    manifest = {"kind": "author_training_prompts_qwen_sft", "author_commit": AUTHOR_COMMIT,
                "base": base, "prompt_stage": "train", "preserve_eos": preserve_eos,
                "loss": "full_sequence_padding_only_masked" if preserve_eos else "full_sequence_pad_eos_masked",
                "split_seed": 1, "trainer_and_initialization_seed": 42, "max_seq_len": 2048,
                "qwen_adaptation": "native chat, thinking disabled, single-GPU NF4/BF16, discovered text LoRA modules",
                "data_limit": "existing local datasets, not certified identical to the author's Llama training data",
                "source_sha256": {str(path.relative_to(SOURCE)): sha256(path) for path in SOURCE.rglob("*.py")},
                "versions": {name: importlib.metadata.version(name) for name in
                             ("torch", "transformers", "peft", "datasets", "accelerate", "bitsandbytes")}}
    processor = AutoProcessor.from_pretrained(base["path"])
    (run_dir / "inputs").mkdir()
    for domain, config in DOMAINS.items():
        (run_dir / domain).mkdir()
        author_constants(domain, "train")
        author_constants(domain, "inference")
        original = repo / "dataset" / config["dataset"]
        frozen = run_dir / "inputs" / f"{domain}.jsonl"
        shutil.copyfile(original, frozen)
        if sha256(original) != sha256(frozen):
            raise ValueError("Dataset changed during snapshot")
        module = load_training_module(domain)
        examples = getattr(module, config["loader"])(str(frozen))
        audit, row_manifest, sample = audit_examples(domain, examples, processor, preserve_eos=preserve_eos)
        save_json(run_dir / domain / "length_audit.json", audit)
        save_json(run_dir / domain / "rows.json", row_manifest)
        save_json(run_dir / domain / "prompt_sample.json", sample)
        if audit["no_target_row_indices"]:
            raise ValueError(f"{domain}: some examples lose all target tokens at author length 2048")
        split = Dataset.from_dict({"row_index": list(range(len(examples)))}).shuffle(seed=1).train_test_split(
            test_size=0.15, seed=1)
        save_json(run_dir / domain / "split.json", {name: split[name]["row_index"][:] for name in split})
        record = {"domain": domain, "dataset_path": str(frozen), "original_dataset": str(original),
                  "dataset_sha256": sha256(frozen), "rows": len(examples), "train_rows": len(split["train"]),
                  "eval_rows": len(split["test"]),
                  "command": training_command(run_dir, domain, base["path"], preserve_eos=preserve_eos),
                  "preserve_eos": preserve_eos, "loss": manifest["loss"],
                  "prompt_stage": "train", "train_inference_difference": "preserved exactly as author",
                  "initialization_seed": 42, "trainer_seed": 42, "split_seed": 1}
        save_json(run_dir / domain / "training_source.json", record)
        print("SFT_PREFLIGHT", domain, json.dumps(audit), flush=True)
    save_json(run_dir / "manifest.json", manifest)


def train_worker(run_dir, domain):
    from transformers import set_seed
    assert_frozen(run_dir)
    record = json.loads((run_dir / domain / "training_source.json").read_text())
    if (run_dir / "adapters" / domain).exists():
        raise ValueError("Refusing to overwrite SFT output")
    set_seed(42)
    module = load_training_module(domain)
    sys.argv = record["command"][2:]
    module.main()


def verify_worker(run_dir, domain):
    import torch
    from peft import PeftModel
    from safetensors.torch import load_file
    from transformers import AutoModelForMultimodalLM, AutoProcessor, BitsAndBytesConfig
    from shared.parammem_prompts import memory_messages
    manifest = json.loads((run_dir / "manifest.json").read_text())
    base = manifest["base"]["path"]
    adapter = run_dir / "adapters" / domain
    config = json.loads((adapter / "adapter_config.json").read_text())
    if Path(config["base_model_name_or_path"]).resolve() != Path(base).resolve():
        raise ValueError("SFT adapter base is not the original Qwen2B")
    if (config["r"], config["lora_alpha"]) != (64, 32):
        raise ValueError("SFT adapter rank/alpha mismatch")
    weights = load_file(str(adapter / "adapter_model.safetensors"))
    if not weights or not all(torch.isfinite(value).all().item() for value in weights.values()):
        raise ValueError("Invalid SFT adapter tensors")
    if not any(torch.count_nonzero(value).item() for name, value in weights.items() if "lora_B" in name):
        raise ValueError("SFT adapter has no trained B weights")
    state_paths = list(adapter.glob("checkpoint-*/trainer_state.json"))
    if not state_paths:
        raise ValueError("Missing SFT trainer state")
    state = max((json.loads(path.read_text()) for path in state_paths), key=lambda value: value["global_step"])
    if state["epoch"] < DOMAINS[domain]["epochs"] - 1e-6 or not state.get("best_model_checkpoint"):
        raise ValueError("SFT epochs or best checkpoint selection incomplete")
    best = load_file(str(Path(state["best_model_checkpoint"]) / "adapter_model.safetensors"))
    if set(best) != set(weights) or not all(torch.equal(weights[name].float(), best[name].float()) for name in best):
        raise ValueError("Top-level SFT is not the selected best checkpoint")
    quantization = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                     bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
    model = AutoModelForMultimodalLM.from_pretrained(base, quantization_config=quantization,
                                                    dtype=torch.bfloat16, device_map={"": 0})
    model = PeftModel.from_pretrained(model, adapter).eval()
    processor = AutoProcessor.from_pretrained(base)
    tokenizer = processor.tokenizer
    question = {"code": "def add(a: int, b: int) -> int:\n    \"\"\"Return their sum.\"\"\"",
                "math": "An equilateral triangle has perimeter 12. What is its side length?",
                "qa": "Which country contains Paris?"}[domain]
    prompt = processor.apply_chat_template(memory_messages(domain, question, "inference"), tokenize=False,
                                           add_generation_prompt=True, enable_thinking=False)
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    with torch.inference_mode():
        enabled = model(**inputs).logits[:, -1].float()
        with model.disable_adapter():
            original = model(**inputs).logits[:, -1].float()
        delta = (enabled - original).abs().max().item()
        if not torch.isfinite(enabled).all() or not torch.isfinite(original).all() or delta <= 0:
            raise ValueError("SFT logits must be finite and differ from original model")
        generated = model.generate(**inputs, do_sample=False, max_new_tokens=64,
                                   pad_token_id=tokenizer.eos_token_id)
    response = tokenizer.decode(generated[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    if not response.strip():
        raise ValueError("SFT smoke response is empty")
    save_json(run_dir / domain / "verification.json", {
        "status": "verified", "base_model": base, "adapter_sha256": sha256(adapter / "adapter_model.safetensors"),
        "preserve_eos": manifest.get("preserve_eos", False), "loss": manifest["loss"],
        "epochs": state["epoch"], "global_step": state["global_step"],
        "best_checkpoint": state["best_model_checkpoint"], "best_checkpoint_tensors_match": True,
        "max_logit_delta_from_original": delta, "smoke_response": response,
        "target_modules": config["target_modules"], "prompt_stage": "inference"})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--preserve_eos", action="store_true",
                        help="Qwen revision: supervise real EOS tokens; mask only appended padding.")
    parser.add_argument("--worker", choices=DOMAINS)
    parser.add_argument("--verify", choices=DOMAINS)
    args = parser.parse_args()
    repo, run_dir = args.repo_root.resolve(), args.run_dir.resolve()
    if args.worker or args.verify:
        (train_worker if args.worker else verify_worker)(run_dir, args.worker or args.verify)
        return
    run_dir.mkdir(parents=True, exist_ok=False)
    status = {"state": "preflight", "pid": os.getpid(), "started_utc": datetime.now(timezone.utc).isoformat(),
              "completed": [], "training_order": list(DOMAINS), "author_training_templates": True,
              "preserve_eos": args.preserve_eos}
    save_json(run_dir / "status.json", status)
    try:
        preflight(repo, run_dir, preserve_eos=args.preserve_eos)
        if not args.train:
            status["state"] = "preflight_complete"
            return
        environment = dict(os.environ, CUDA_VISIBLE_DEVICES="0", TOKENIZERS_PARALLELISM="false",
                           HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", WANDB_DISABLED="true")
        for domain in DOMAINS:
            memory = int(subprocess.check_output(
                ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"], text=True).strip())
            if memory > 256:
                raise RuntimeError("GPU is occupied; no existing process will be stopped")
            for phase, flag in (("training", "--worker"), ("verifying", "--verify")):
                status.update(state=phase, domain=domain)
                save_json(run_dir / "status.json", status)
                command = [sys.executable, "-u", str(Path(__file__).resolve()), "--repo-root", str(repo),
                           "--run-dir", str(run_dir), flag, domain]
                with (run_dir / domain / f"{phase}.log").open("x") as log:
                    process = subprocess.Popen(command, cwd=SOURCE, env=environment,
                                               stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
                    status["child_pid"] = process.pid
                    save_json(run_dir / "status.json", status)
                    returncode = process.wait()
                (run_dir / domain / f"{phase}.exit").write_text(str(returncode) + "\n")
                if returncode:
                    raise RuntimeError(f"{domain} {phase} failed: exit {returncode}")
            assert_frozen(run_dir)
            status["completed"].append(domain)
        status.update(state="complete", completed_utc=datetime.now(timezone.utc).isoformat())
    except Exception as error:
        status.update(state="failed", error=str(error))
        raise
    finally:
        save_json(run_dir / "status.json", status)


if __name__ == "__main__":
    main()
