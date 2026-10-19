"""ParamAgent offline paired-data tools; online gated inference is not implemented."""
import sys
from shared.entrypoints import run_script

# Preserve old baseline commands without labelling them as the full method.
LEGACY_SCRIPTS = {"code": "code/main_param.py", "math": "math/mainMath_param.py",
                  "qa": "qa/mainQA_parametric.py"}


def main():
    if len(sys.argv) > 1 and sys.argv[1] in LEGACY_SCRIPTS:
        print("Legacy baseline command: use python -m parammem.runner with an explicit "
              "--method and --memory-file. This command does not run routed ParamAgent.",
              file=sys.stderr)
        return run_script(LEGACY_SCRIPTS[sys.argv[1]], sys.argv[2:])
    from gain_router.paired import main as data_main
    return data_main()


if __name__ == "__main__":
    main()
