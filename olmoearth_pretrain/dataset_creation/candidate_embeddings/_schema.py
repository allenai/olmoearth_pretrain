"""Shared constants for candidate-embedding scoring and selection."""

from __future__ import annotations

STRATEGIES: list[str] = [
    "novelty",
    "xglobal_bridge",
    "sparse_infill",
    "xlocal_bridge",
    "prototypes",
]

METADATA_COLUMNS: list[str] = [
    "sample_idx",
    "window_name",
    "lat",
    "lon",
    "parent_label",
]

STRATEGY_FILE_STEMS: dict[str, str] = {
    strategy: f"{strategy}_scores" for strategy in STRATEGIES
}
