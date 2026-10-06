"""SFT and DPO training entry points for the domain experts."""
import argparse
from shared.entrypoints import run_script


def main(stage=None):
    parser = argparse.ArgumentParser(description=__doc__)
    if stage is None:
        parser.add_argument("stage", choices=("sft", "dpo"))
    parser.add_argument("domain", choices=("code", "math", "qa"))
    args, remaining = parser.parse_known_args()
    selected = stage or args.stage
    suffix = "_DPO" if selected == "dpo" else ""
    name = {"code": "Code", "math": "Math", "qa": "QA"}[args.domain]
    return run_script(f"{args.domain}/LoRA_Qwen35_{name}{suffix}_4090.py", remaining)


if __name__ == "__main__":
    main()
