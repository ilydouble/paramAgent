"""Run historical scripts with their original sibling-import context."""
from pathlib import Path
import runpy
import sys

ROOT = Path(__file__).resolve().parents[1]


def run_script(relative_path, arguments):
    path = ROOT / relative_path
    old_argv, old_path = sys.argv, sys.path[:]
    try:
        sys.argv = [str(path), *arguments]
        sys.path.insert(0, str(path.parent))
        runpy.run_path(str(path), run_name="__main__")
    finally:
        sys.argv = old_argv
        sys.path[:] = old_path
