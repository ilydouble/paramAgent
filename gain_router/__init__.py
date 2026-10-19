"""Compatibility namespace for ParamAgent data tooling.

Preserve historical gain_router module identities and CLI commands.
"""
from pathlib import Path

__path__ = [str(Path(__file__).resolve().parents[1] / "paramagent" / "data")]
