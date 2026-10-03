#!/usr/bin/env python3
"""Score recorded outcomes offline; never calls a model."""
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gain_router.offline import main

if __name__ == "__main__":
    raise SystemExit(main())
