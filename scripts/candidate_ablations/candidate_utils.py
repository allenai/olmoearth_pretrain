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

    logger.info(f"Loading parquet file: {parquet_path}")
    df = pd.read_parquet(parquet_path, columns=["window_name"] + columns)
    logger.info(f"Loaded parquet with {len(df)} rows.")

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


def save_candidate_sample_ids_file(
    parquet_path: str | Path,
    columns: str | list[str],
    output_path: str | Path,
) -> Path:
    """Filter the parquet and write matching sample IDs to a text file.

    If the output file already exists it is reused without re-reading the parquet.

    Returns:
        Path to the written file.
    """
    output_path = Path(output_path)
    if output_path.exists():
        n_lines = sum(1 for _ in output_path.open())
        logger.info(f"Reusing existing sample IDs file {output_path} ({n_lines} IDs)")
        return output_path

    sample_ids = load_candidate_sample_ids(parquet_path, columns)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(sample_ids) + "\n")
    logger.info(f"Wrote {len(sample_ids)} sample IDs to {output_path}")
    return output_path
