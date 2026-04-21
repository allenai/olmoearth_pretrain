"""Build a per-sample selection table from combined_acquisition_scores.parquet.

Given a total budget ``X``, this step picks:

1. the top ``3 * X / 5`` candidates for each of the five standalone
   strategies, ranked by that strategy's normalized score in the combined
   file (``<strategy>_normalized_score``)
2. the top ``4 * X / 5`` candidates for each of the five leave-one-out
   ablation scores (``combined_score_drop_<strategy>``)
3. the top ``X`` candidates for the full weighted score
   (``combined_score``)

The output is a deduplicated parquet table with the sample metadata
(``sample_idx``, ``window_name``, ``lat``, ``lon``, ``parent_label``) and one
0/1 indicator column per selection criterion. A sample that made it into
several selections has ``1`` in several columns.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

METADATA_COLUMNS = ["sample_idx", "window_name", "lat", "lon", "parent_label"]

STRATEGIES = [
    "novelty",
    "xglobal_bridge",
    "sparse_infill",
    "xlocal_bridge",
    "prototypes",
]


def top_row_indices(scores: np.ndarray, k: int) -> np.ndarray:
    """Return the row positions of the top-``k`` scores (ties broken stably)."""
    if k <= 0 or scores.size == 0:
        return np.empty(0, dtype=np.int64)
    k = min(k, scores.size)
    order = np.argsort(-scores, kind="stable")
    return order[:k].astype(np.int64)


def collect_selections(
    df: pd.DataFrame,
    k_solo: int,
    k_drop: int,
    k_full: int,
) -> dict[str, np.ndarray]:
    """Return row-position selections for every criterion, in output order."""
    selections: dict[str, np.ndarray] = {}

    if "combined_score" not in df.columns:
        raise ValueError("Input parquet is missing the 'combined_score' column")
    selections["in_top_combined"] = top_row_indices(
        df["combined_score"].to_numpy(dtype=np.float32), k_full
    )

    for strategy in STRATEGIES:
        column = f"{strategy}_normalized_score"
        if column not in df.columns:
            raise ValueError(f"Input parquet is missing standalone column '{column}'")
        selections[f"in_top_solo_{strategy}"] = top_row_indices(
            df[column].to_numpy(dtype=np.float32), k_solo
        )

    for strategy in STRATEGIES:
        column = f"combined_score_drop_{strategy}"
        if column not in df.columns:
            raise ValueError(
                f"Input parquet is missing ablation column '{column}'. "
                "Re-run combine_acquisition.py with --ablation."
            )
        selections[f"in_top_drop_{strategy}"] = top_row_indices(
            df[column].to_numpy(dtype=np.float32), k_drop
        )

    return selections


def build_selection_frame(
    df: pd.DataFrame,
    selections: dict[str, np.ndarray],
) -> pd.DataFrame:
    """Assemble the deduplicated metadata table with 0/1 indicator columns."""
    missing = [col for col in METADATA_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Input parquet is missing metadata columns: {missing}")

    union_rows = np.unique(
        np.concatenate(
            [rows for rows in selections.values() if rows.size > 0]
            or [np.empty(0, dtype=np.int64)]
        )
    )

    result = df.iloc[union_rows][METADATA_COLUMNS].reset_index(drop=True).copy()

    position = {int(row_idx): i for i, row_idx in enumerate(union_rows.tolist())}
    for column_name, rows in selections.items():
        mask = np.zeros(len(result), dtype=np.int8)
        for row_idx in rows.tolist():
            mask[position[int(row_idx)]] = 1
        result[column_name] = mask

    return result


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for top-sample table generation."""
    parser = argparse.ArgumentParser(
        description=(
            "Select top-X samples across the full weighted score, every "
            "standalone strategy and every leave-one-out ablation."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--combined-parquet",
        required=True,
        help="Path to combined_acquisition_scores.parquet (produced with --ablation).",
    )
    parser.add_argument(
        "--num-samples",
        "-x",
        type=int,
        required=True,
        help=(
            "Total sample budget X. Each standalone strategy picks 3X/5, each "
            "leave-one-out ablation picks 4X/5, the full combined score picks X."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Where to write selection_top<X>.parquet. Defaults to the combined parquet's directory.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> str:
    """Build and write the deduplicated top-sample selection parquet."""
    args = parse_args(argv)

    X = int(args.num_samples)
    if X <= 0:
        raise ValueError(f"--num-samples must be a positive integer, got {X}")

    k_solo = (3 * X) // 5
    k_drop = (4 * X) // 5
    k_full = X

    df = pd.read_parquet(args.combined_parquet)
    selections = collect_selections(df, k_solo=k_solo, k_drop=k_drop, k_full=k_full)
    result = build_selection_frame(df, selections)

    output_dir = args.output_dir or os.path.dirname(
        os.path.abspath(args.combined_parquet)
    )
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"selection_top{X}.parquet")
    result.to_parquet(output_path, index=False)

    ids_path = os.path.join(output_dir, f"selected_sample_ids_top{X}.json")
    with open(ids_path, "w") as f:
        json.dump(result["window_name"].tolist(), f)

    print(
        f"[save] Selection parquet -> {output_path} "
        f"({len(result):,} unique samples; k_solo={k_solo}, k_drop={k_drop}, k_full={k_full})"
    )
    print(f"[save] Selected sample IDs -> {ids_path}")
    return output_path


if __name__ == "__main__":
    main(sys.argv[1:])
