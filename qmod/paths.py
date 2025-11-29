# qmod/paths.py
"""
Package-level path resolver for qmod.

This mirrors the root-level paths.py but is safe to import
as `from qmod.paths import artifacts_dir` from anywhere in the package.
"""

from __future__ import annotations
import os
from pathlib import Path

# qmod/.. is the repo root
REPO_ROOT = Path(__file__).resolve().parent.parent

# Base folder for all artifacts (can override with env var)
ARTIFACTS_BASE = Path(
    os.getenv("APP_ARTIFACTS_DIR", REPO_ROOT / "artifacts")
).resolve()


def ensure_dir(p: Path) -> Path:
    """Make sure directory exists and return it."""
    p.mkdir(parents=True, exist_ok=True)
    return p


def artifacts_dir(*parts: str) -> Path:
    """
    For writeable dirs like runs/, logs/, models/, vis_macd, vis_rsi, composite, etc.
    Example:
        artifacts_dir("vis_macd", "optuna", "AAPL")
    """
    return ensure_dir(ARTIFACTS_BASE.joinpath(*parts))


def data_file(*parts: str) -> Path:
    """For read-only bundled data/config files inside repo."""
    return REPO_ROOT.joinpath(*parts).resolve()


def resolve_out(*parts: str) -> Path:
    """General purpose: always gives absolute file path."""
    return ensure_dir(ARTIFACTS_BASE).joinpath(*parts).resolve()
