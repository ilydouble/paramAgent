import os
import gc
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


def cleanup_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def merge_lora_weights(base_model_path, lora_path, output_path):
    if os.path.exists(os.path.join(output_path, "config.json")):
        print(f"Merged model already exists at {output_path}, skipping merge...")
        return False

    print(f"\nMerging LoRA weights")
    print(f"  Base model: {base_model_path}")
    print(f"  LoRA adapter: {lora_path}")
    print(f"  Output: {output_path}")

    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
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

    print(f"  Merged model saved to {output_path}")

    del model, base_model
    cleanup_memory()

    return True


if __name__ == "__main__":
    base_model_path = "./models/Qwen3.5-2B"
    lora_path = "./lora-qwen3.5-2b/lora-qwen3.5-2b-code3"
    output_path = "./models/Qwen3.5-2B-code-merged3"

    merge_lora_weights(base_model_path, lora_path, output_path)
