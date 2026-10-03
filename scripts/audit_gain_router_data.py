#!/usr/bin/env python3
"""Read-only trajectory quality audit. Does not load or call a model."""
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gain_router.audit import main

if __name__ == "__main__":
    raise SystemExit(main())
