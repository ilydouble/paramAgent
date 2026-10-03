import os
import gc
import argparse
import json
from pathlib import Path
from split_protocol import load_manifest, file_sha256
from training_splits import validate_training_artifact
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


def cleanup_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def merge_lora_weights(base_model_path, lora_path, output_path, *, split_manifest, domain, device="cpu"):
    manifest = load_manifest(split_manifest)
    report = validate_training_artifact(lora_path, manifest, stage="sft", domain=domain)
    if Path(output_path).exists():
        raise FileExistsError(f"Merged output already exists: {output_path}")

    print(f"\nMerging LoRA weights")
    print(f"  Base model: {base_model_path}")
    print(f"  LoRA adapter: {lora_path}")
    print(f"  Output: {output_path}")

    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.bfloat16,
        device_map={"": device},
        low_cpu_mem_usage=True,
        attn_implementation="sdpa",
    )

    model = PeftModel.from_pretrained(
        base_model,
        lora_path,
        torch_dtype=torch.bfloat16,
    )

    print("  Merging weights...")
    model = model.merge_and_unload()

    os.makedirs(output_path, exist_ok=True)
    model.save_pretrained(output_path, safe_serialization=True)

    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    tokenizer.save_pretrained(output_path)

    report["source_adapter_artifacts"] = report["artifacts"]
    report["artifacts"] = {p.name: file_sha256(p) for p in Path(output_path).glob("*.safetensors")}
    report["artifacts"]["config.json"] = file_sha256(Path(output_path) / "config.json")
    (Path(output_path) / "training_split.json").write_text(json.dumps(report, indent=2))
    print(f"  Merged model saved to {output_path}")

    del model, base_model
    cleanup_memory()

    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge a manifest-verified SFT adapter")
    parser.add_argument("--base_model", required=True)
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--split_manifest", required=True)
    parser.add_argument("--domain", choices=("code", "math", "qa"), required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    merge_lora_weights(args.base_model, args.adapter, args.output,
                       split_manifest=args.split_manifest, domain=args.domain, device=args.device)
