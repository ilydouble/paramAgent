"""Generate ParamMem / ParamMem-DPO memories with the shared domain inference code.

All generation options are forwarded unchanged. DPO requires the matching
SFT merged base and DPO adapter; method selection does not locate weights.
"""
import argparse

from shared.entrypoints import run_script

SCRIPTS = {"code": "code/LoRA_Qwen35_Code_4090_inference.py",
           "math": "math/LoRA_Qwen35_Math_4090_inference.py",
           "qa": "qa/LoRA_Qwen35_QA_4090_inference.py"}


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("domain", choices=SCRIPTS)
    parser.add_argument("--method", choices=("parammem", "parammem-dpo"), required=True)
    parser.add_argument("--base_model", required=True)
    parser.add_argument("--lora_path")
    args, remaining = parser.parse_known_args()
    if args.method == "parammem-dpo" and not args.lora_path:
        parser.error("ParamMem-DPO requires --lora_path for the DPO adapter")
    forwarded = ["--base_model", args.base_model, *remaining]
    if args.lora_path:
        forwarded += ["--lora_path", args.lora_path]
    print(f"Method: {args.method}; base: {args.base_model}; adapter: {args.lora_path}")
    return run_script(SCRIPTS[args.domain], forwarded)


if __name__ == "__main__":
    main()
