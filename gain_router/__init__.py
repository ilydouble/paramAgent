"""Compatibility namespace for SCORE data tooling.

Keep historical module identities and CLI commands stable during migration.
"""
from pathlib import Path

__path__ = [str(Path(__file__).resolve().parents[1] / "score" / "data")]
