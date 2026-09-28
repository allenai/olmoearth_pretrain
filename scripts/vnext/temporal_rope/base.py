"""Shared v1.1 baseline builders for temporal RoPE experiment scripts.

This keeps the RoPE experiments out of ``scripts/official`` while preserving
the exact v1.1 baseline config they were run against.
"""

import importlib
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

_base = importlib.import_module("scripts.official.v1_1.base")

__all__ = [name for name in dir(_base) if not name.startswith("_")]
globals().update({name: getattr(_base, name) for name in __all__})
