"""Independent entry point for the historical ParamAgent baseline."""
import argparse
from shared.entrypoints import run_script

SCRIPTS = {"code": "code/main_param.py", "math": "math/mainMath_param.py",
           "qa": "qa/mainQA_parametric.py"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("domain", choices=SCRIPTS)
    args, remaining = parser.parse_known_args()
    return run_script(SCRIPTS[args.domain], remaining)


if __name__ == "__main__":
    main()
