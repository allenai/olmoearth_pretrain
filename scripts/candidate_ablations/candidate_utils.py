"""Utilities for selecting candidate samples from a scored parquet file.

Selects the top-N candidates per strategy from raw normalized scores,
validated against h5 sample availability.
"""

from __future__ import annotations

import itertools
import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

STRATEGY_NAMES: list[str] = [
    "novelty",
    "xglobal_bridge",
    "sparse_infill",
    "xlocal_bridge",
    "prototypes",
]


def strategy_to_score_column(strategy: str) -> str:
    """Map a short strategy name to its normalized score column in the parquet."""
    return f"{strategy}_normalized_score"


def _load_h5_sample_ids(h5py_dir: str | Path) -> set[str]:
    """Load available sample IDs from the h5py directory's metadata CSV."""
    meta_path = Path(h5py_dir) / "sample_metadata.csv"
    if not meta_path.exists():
        raise FileNotFoundError(
            f"sample_metadata.csv not found in {h5py_dir}. "
            "Cannot validate candidate availability."
        )
    meta = pd.read_csv(meta_path, usecols=["sample_id"])
    ids = set(meta["sample_id"].astype(str))
    logger.info(f"Loaded {len(ids)} available sample IDs from {meta_path}")
    return ids


def load_candidate_sample_ids(
    parquet_path: str | Path,
    strategies: list[str],
    select_top: int,
    h5py_dir: str | Path,
) -> list[str]:
    """Select top-N candidates per strategy, validated against h5 availability.

    For each strategy, the parquet is sorted by its normalized score in
    descending order and the top ``select_top`` samples are taken.  Results
    are unioned across strategies and deduplicated.

    Only samples that have corresponding h5 files (verified via
    ``sample_metadata.csv``) are considered.

    Args:
        parquet_path: Path to ``combined_acquisition_scores.parquet``.
        strategies: Short strategy names (e.g. ``["novelty", "xglobal_bridge"]``).
        select_top: Number of top-scoring samples to select per strategy.
        h5py_dir: Path to the candidate h5py directory (must contain
            ``sample_metadata.csv``).

    Returns:
        Deduplicated list of sample_id strings.
    """
    h5_ids = _load_h5_sample_ids(h5py_dir)

    score_cols = [strategy_to_score_column(s) for s in strategies]
    logger.info(f"Loading parquet: {parquet_path}")
    df = pd.read_parquet(parquet_path, columns=["window_name"] + score_cols)
    logger.info(f"Loaded parquet with {len(df)} rows.")

    missing = [c for c in score_cols if c not in df.columns]
    if missing:
        raise ValueError(
            f"Score column(s) {missing} not found in parquet. "
            f"Available columns: {list(df.columns)}"
        )

    df["sample_id"] = df["window_name"].apply(lambda x: str(x).rsplit("_", 1)[-1])
    pre_filter_len = len(df)
    df = df[df["sample_id"].isin(h5_ids)]
    logger.info(
        f"Pre-filtered to {len(df)} rows with h5 data "
        f"(dropped {pre_filter_len - len(df)})"
    )

    if len(df) < select_top:
        raise ValueError(
            f"Only {len(df)} h5-available samples but select_top={select_top}. "
            "Reduce select_top or add more h5 data."
        )

    # Also load the full (unfiltered) parquet for gap analysis.
    # We only need window_name + score cols, which are already in memory as
    # df_full before the h5 filter was applied.  To avoid a second read we
    # do the naive ranking on the full parquet loaded earlier, so we reload
    # with the same columns but without the h5 filter.
    df_all = pd.read_parquet(parquet_path, columns=["window_name"] + score_cols)
    df_all["sample_id"] = df_all["window_name"].apply(
        lambda x: str(x).rsplit("_", 1)[-1]
    )

    per_strategy_ids: dict[str, set[str]] = {}
    for strategy, col in zip(strategies, score_cols):
        # Naive top-N (ignoring h5 availability)
        naive_top = set(df_all.nlargest(select_top, col)["sample_id"])
        missing_h5 = naive_top - h5_ids
        logger.info(
            f"  {strategy}: naive top-{select_top} has {len(missing_h5)} samples "
            f"missing from h5 ({len(missing_h5) / select_top:.1%})"
        )

        # Actual selection from h5-filtered set
        top = df.nlargest(select_top, col)
        ids = set(top["sample_id"])
        backfilled = ids - naive_top
        logger.info(
            f"  {strategy}: selected {len(ids)} h5-available samples "
            f"({len(backfilled)} backfilled from lower ranks)"
        )
        per_strategy_ids[strategy] = ids

    all_ids: set[str] = set()
    for ids in per_strategy_ids.values():
        all_ids |= ids

    _log_overlap_stats(strategies, per_strategy_ids, len(all_ids))

    return sorted(all_ids)


def _log_overlap_stats(
    strategies: list[str],
    per_strategy_ids: dict[str, set[str]],
    total_unique: int,
) -> None:
    """Log pairwise overlap and total unique count."""
    logger.info(f"Total unique candidates after union: {total_unique}")
    for a, b in itertools.combinations(strategies, 2):
        overlap = len(per_strategy_ids[a] & per_strategy_ids[b])
        logger.info(f"  overlap({a}, {b}): {overlap}")


def save_candidate_sample_ids_file(
    parquet_path: str | Path,
    strategies: list[str],
    select_top: int,
    h5py_dir: str | Path,
    output_path: str | Path,
) -> Path:
    """Select top-N candidates and write their sample IDs to a text file.

    If the output file already exists it is reused without re-reading the
    parquet.

    Returns:
        Path to the written file.
    """
    output_path = Path(output_path)
    if output_path.exists():
        n_lines = sum(1 for _ in output_path.open())
        logger.info(f"Reusing existing sample IDs file {output_path} ({n_lines} IDs)")
        return output_path

    sample_ids = load_candidate_sample_ids(
        parquet_path, strategies, select_top, h5py_dir
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(sample_ids) + "\n")
    logger.info(f"Wrote {len(sample_ids)} sample IDs to {output_path}")
    return output_path
