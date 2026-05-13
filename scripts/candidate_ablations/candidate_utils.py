"""Utilities for selecting candidate samples from a scored parquet file."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

SCORE_COLUMNS = [
    "in_top_combined",
    "in_top_solo_novelty",
    "in_top_solo_xglobal_bridge",
    "in_top_solo_sparse_infill",
    "in_top_solo_xlocal_bridge",
    "in_top_solo_prototypes",
    "in_top_drop_novelty",
    "in_top_drop_xglobal_bridge",
    "in_top_drop_sparse_infill",
    "in_top_drop_xlocal_bridge",
    "in_top_drop_prototypes",
]


def load_candidate_sample_ids(
    parquet_path: str | Path,
    columns: str | list[str],
) -> list[str]:
    """Load sample_ids from the scored parquet for the given score columns.

    A candidate is selected if *any* of the specified columns has a value of 1.

    Args:
        parquet_path: Path to the scored candidates parquet file.
        columns: One or more score column names to filter on (e.g.
            ``"in_top_combined"`` or
            ``["in_top_solo_novelty", "in_top_drop_novelty"]``).

    Returns:
        Deduplicated list of sample_id strings (= window_name from the parquet).
    """
    if isinstance(columns, str):
        columns = [columns]

    df = pd.read_parquet(parquet_path)

    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(
            f"Column(s) {missing} not found in parquet. "
            f"Available score columns: {[c for c in df.columns if c.startswith('in_top_')]}"
        )

    mask = df[columns].any(axis=1)
    window_names = df.loc[mask, "window_name"].unique()
    selected = [str(name).rsplit("_", 1)[-1] for name in window_names]

    logger.info(
        f"Selected {len(selected)} unique candidate sample_ids from columns {columns}"
    )
    return selected
