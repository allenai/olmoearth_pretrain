"""Combine multiple acquisition strategy outputs into one weighted ranking.

Supports two modes:

1. Normal mode
   - Weighted average of per-strategy scores using the user-supplied weights.
   - Emits a single ``combined_score`` column plus the raw and normalized
     per-strategy component scores.

2. Ablation mode (``--ablation``)
   - Same ``combined_score`` column as normal mode.
   - Also emits, in the same parquet file, one ``combined_score_drop_<s>``
     column per active strategy: the weighted average with that strategy's
     weight set to 0 and the remaining weights left unchanged.
   - A ranked ``sample_idx`` array is written for each drop column so every
     leave-one-out variant can directly feed a top-X pretraining run.

Solo (top-X from one strategy alone) and random baseline runs are supported
directly by the per-strategy ``<strategy>_ranked_sample_idx.npy`` files and
by sampling random sample indices yourself; this script does not duplicate
them.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd
from reference_model import save_summary_json

STRATEGY_FILE_STEMS = {
    "novelty": "novelty_scores",
    "xglobal_bridge": "xglobal_bridge_scores",
    "sparse_infill": "sparse_infill_scores",
    "xlocal_bridge": "xlocal_bridge_scores",
    "prototypes": "prototypes_scores",
}

SCORE_COLUMNS = {
    "novelty": "novelty_score",
    "xglobal_bridge": "score",
    "sparse_infill": "score",
    "xlocal_bridge": "score",
    "prototypes": "score",
}

METADATA_COLUMNS = ["sample_idx", "window_name", "lat", "lon", "parent_label"]


def _resolve_strategy_path(scores_dir: str, strategy: str) -> str:
    """Return the existing score file path for a strategy.

    Prefers parquet; falls back to csv so legacy score dirs still work.
    """
    stem = STRATEGY_FILE_STEMS[strategy]
    parquet_path = os.path.join(scores_dir, f"{stem}.parquet")
    if os.path.exists(parquet_path):
        return parquet_path
    csv_path = os.path.join(scores_dir, f"{stem}.csv")
    if os.path.exists(csv_path):
        return csv_path
    raise FileNotFoundError(
        f"Missing {strategy} scores file (looked for {parquet_path} and {csv_path})"
    )


def load_strategy_frame(scores_dir: str, strategy: str) -> pd.DataFrame:
    """Load one strategy's score table as a DataFrame."""
    path = _resolve_strategy_path(scores_dir, strategy)
    if path.endswith(".parquet"):
        return pd.read_parquet(path)
    return pd.read_csv(path)


def build_base_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Extract the shared metadata columns used in the combined output."""
    missing = [col for col in METADATA_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Strategy frame is missing metadata columns: {missing}")
    base = df[METADATA_COLUMNS].copy()
    base["sample_idx"] = base["sample_idx"].astype(np.int64)
    base["parent_label"] = base["parent_label"].astype(np.int32)
    return base.reset_index(drop=True)


def assert_aligned_frames(
    base: pd.DataFrame, other: pd.DataFrame, strategy: str
) -> None:
    """Require all strategy tables to align by sample_idx and parent assignment."""
    if len(base) != len(other):
        raise ValueError(
            f"{strategy} score row count mismatch: expected {len(base)}, got {len(other)}"
        )
    for key in ("sample_idx", "parent_label"):
        base_vals = base[key].to_numpy()
        other_vals = other[key].astype(base_vals.dtype).to_numpy()
        if not np.array_equal(base_vals, other_vals):
            mismatch = int(np.argmax(base_vals != other_vals))
            raise ValueError(
                f"{strategy} score alignment mismatch at row {mismatch}: "
                f"{key} differs ({base_vals[mismatch]!r} vs {other_vals[mismatch]!r})"
            )


def extract_scores(df: pd.DataFrame, strategy: str) -> np.ndarray:
    """Read the numeric score column for one strategy."""
    score_key = SCORE_COLUMNS[strategy]
    if score_key not in df.columns:
        raise ValueError(f"{strategy} frame missing score column '{score_key}'")
    return df[score_key].to_numpy(dtype=np.float32)


def normalize_scores(scores: np.ndarray, method: str) -> np.ndarray:
    """Normalize scores across strategies onto a comparable scale."""
    if scores.size == 0:
        return scores.astype(np.float32)

    if method == "rank":
        if scores.size == 1:
            return np.ones(1, dtype=np.float32)
        order = np.argsort(scores, kind="stable")
        ranks = np.empty(scores.size, dtype=np.float32)
        ranks[order] = np.arange(scores.size, dtype=np.float32)
        return (ranks / float(scores.size - 1)).astype(np.float32)

    if method == "minmax":
        lo = float(np.min(scores))
        hi = float(np.max(scores))
        if hi <= lo:
            return np.ones_like(scores, dtype=np.float32)
        return ((scores - lo) / (hi - lo)).astype(np.float32)

    if method == "zscore":
        mean = float(np.mean(scores))
        std = float(np.std(scores))
        if std <= 0:
            return np.ones_like(scores, dtype=np.float32)
        z = (scores - mean) / std
        return (1.0 / (1.0 + np.exp(-z))).astype(np.float32)

    raise ValueError(f"Unknown normalization method: {method}")


def active_weights(args: argparse.Namespace) -> dict[str, float]:
    """Return positive strategy weights supplied on the CLI."""
    weights = {
        "novelty": float(args.weight_novelty),
        "xglobal_bridge": float(args.weight_xglobal_bridge),
        "sparse_infill": float(args.weight_sparse_infill),
        "xlocal_bridge": float(args.weight_xlocal_bridge),
        "prototypes": float(args.weight_prototypes),
    }
    return {name: weight for name, weight in weights.items() if weight > 0}


def weighted_combine(
    normalized_scores: dict[str, np.ndarray],
    weights: dict[str, float],
    n_rows: int,
) -> np.ndarray:
    """Return a weighted average of already-normalized per-strategy scores.

    Skips strategies with zero weight and renormalizes by the sum of active
    weights. Returns zeros when the total weight is zero.
    """
    combined = np.zeros(n_rows, dtype=np.float32)
    total_weight = 0.0
    for strategy, weight in weights.items():
        if weight <= 0:
            continue
        combined += (weight * normalized_scores[strategy]).astype(np.float32)
        total_weight += weight
    if total_weight <= 0:
        return combined
    return (combined / total_weight).astype(np.float32)


def build_combined_frame(
    base: pd.DataFrame,
    combined: np.ndarray,
    strategy_scores: dict[str, np.ndarray],
    normalized_scores: dict[str, np.ndarray],
    weights: dict[str, float],
    ablation_columns: dict[str, np.ndarray] | None = None,
) -> pd.DataFrame:
    """Assemble the final DataFrame with scores and per-strategy components."""
    active_strategies = list(weights.keys())
    df = base.copy()
    df["combined_score"] = combined.astype(np.float32)

    if ablation_columns:
        for name, arr in ablation_columns.items():
            df[name] = arr.astype(np.float32)

    for strategy in active_strategies:
        df[f"{strategy}_weight"] = np.float32(weights[strategy])
    for strategy in active_strategies:
        df[f"{strategy}_raw_score"] = strategy_scores[strategy].astype(np.float32)
    for strategy in active_strategies:
        df[f"{strategy}_normalized_score"] = normalized_scores[strategy].astype(
            np.float32
        )

    return df


def compute_leave_one_out_scores(
    normalized_scores: dict[str, np.ndarray],
    weights: dict[str, float],
    n_rows: int,
) -> dict[str, np.ndarray]:
    """Compute one leave-one-out combined score per active strategy.

    For each strategy ``s`` in ``weights``, the dropped variant uses the same
    user-supplied weights except ``weights[s]`` is forced to zero. The
    remaining weights are renormalized by their sum so the column is
    comparable in scale to ``combined_score``.
    """
    columns: dict[str, np.ndarray] = {}
    for strategy in weights:
        dropped = {k: (0.0 if k == strategy else w) for k, w in weights.items()}
        columns[f"combined_score_drop_{strategy}"] = weighted_combine(
            normalized_scores, dropped, n_rows
        )
    return columns


def save_ranked_indices(
    output_dir: str,
    name: str,
    sample_indices: np.ndarray,
    scores: np.ndarray,
) -> str:
    """Save argsort-by-score sample_idx array and return its path."""
    path = os.path.join(output_dir, f"{name}.npy")
    ranked = np.argsort(-scores, kind="stable")
    np.save(path, sample_indices[ranked])
    print(f"[save] Ranked sample indices -> {path}")
    return path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for weighted acquisition-score combination."""
    parser = argparse.ArgumentParser(
        description="Combine novelty and acquisition strategy outputs into one ranking.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--scores-dir",
        required=True,
        help="Directory containing per-strategy *_scores.parquet (or legacy .csv) files.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Where to write combined outputs. Defaults to --scores-dir.",
    )
    parser.add_argument(
        "--normalization",
        default="rank",
        choices=["rank", "minmax", "zscore"],
        help="How to normalize each strategy score before weighting.",
    )
    parser.add_argument("--weight-novelty", type=float, default=0.4)
    parser.add_argument(
        "--weight-xglobal-bridge", dest="weight_xglobal_bridge", type=float, default=0.2
    )
    parser.add_argument(
        "--weight-sparse-infill", dest="weight_sparse_infill", type=float, default=0.2
    )
    parser.add_argument(
        "--weight-xlocal-bridge", dest="weight_xlocal_bridge", type=float, default=0.1
    )
    parser.add_argument("--weight-prototypes", type=float, default=0.1)
    parser.add_argument(
        "--ablation",
        action="store_true",
        help=(
            "Also emit one leave-one-out combined score per active strategy "
            "(the strategy's weight set to 0, remaining weights renormalized) "
            "as additional parquet columns and ranked-sample-idx files."
        ),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Load per-strategy scores, combine them, and persist ranked outputs."""
    args = parse_args(argv)
    weights = active_weights(args)
    if not weights:
        raise ValueError("At least one positive strategy weight is required")

    output_dir = args.output_dir or args.scores_dir
    os.makedirs(output_dir, exist_ok=True)

    base_strategy = next(iter(weights.keys()))
    base = build_base_frame(load_strategy_frame(args.scores_dir, base_strategy))

    strategy_scores: dict[str, np.ndarray] = {}
    for strategy in weights:
        df = load_strategy_frame(args.scores_dir, strategy)
        assert_aligned_frames(base, df, strategy)
        strategy_scores[strategy] = extract_scores(df, strategy)

    normalized_scores = {
        strategy: normalize_scores(scores, args.normalization)
        for strategy, scores in strategy_scores.items()
    }

    n_rows = len(base)
    combined = weighted_combine(normalized_scores, weights, n_rows)

    ablation_columns: dict[str, np.ndarray] = {}
    if args.ablation:
        ablation_columns = compute_leave_one_out_scores(
            normalized_scores,
            weights,
            n_rows,
        )

    combined_df = build_combined_frame(
        base,
        combined,
        strategy_scores,
        normalized_scores,
        weights,
        ablation_columns=ablation_columns or None,
    )

    parquet_path = os.path.join(output_dir, "combined_acquisition_scores.parquet")
    combined_df.to_parquet(parquet_path, index=False)
    print(f"[save] Combined parquet -> {parquet_path}")

    sample_indices = base["sample_idx"].to_numpy(dtype=np.int64)
    save_ranked_indices(
        output_dir, "combined_ranked_sample_idx", sample_indices, combined
    )

    summary: dict[str, object] = {
        "scores_dir": args.scores_dir,
        "output_dir": output_dir,
        "normalization": args.normalization,
        "weights": weights,
        "ablation": bool(args.ablation),
        "candidate_size": int(n_rows),
        "combined_score_min": float(np.min(combined)) if n_rows else 0.0,
        "combined_score_max": float(np.max(combined)) if n_rows else 0.0,
        "combined_score_mean": float(np.mean(combined)) if n_rows else 0.0,
        "top10_sample_idx": sample_indices[np.argsort(-combined, kind="stable")[:10]]
        .astype(int)
        .tolist(),
    }

    if ablation_columns:
        ablation_tops: dict[str, list[int]] = {}
        for column_name, arr in ablation_columns.items():
            variant_name = column_name.replace(
                "combined_score", "combined_ranked_sample_idx"
            )
            save_ranked_indices(output_dir, variant_name, sample_indices, arr)
            ablation_tops[column_name] = (
                sample_indices[np.argsort(-arr, kind="stable")[:10]]
                .astype(int)
                .tolist()
            )
        summary["ablation_top10_sample_idx"] = ablation_tops

    summary_path = os.path.join(output_dir, "combined_acquisition_summary.json")
    save_summary_json(summary_path, summary)


if __name__ == "__main__":
    main(sys.argv[1:])
