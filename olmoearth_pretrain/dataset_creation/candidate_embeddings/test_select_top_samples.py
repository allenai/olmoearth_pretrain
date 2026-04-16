"""Tests for the top-sample selection parquet builder."""

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
import select_top_samples as sts

STRATEGIES = [
    "novelty",
    "xglobal_bridge",
    "sparse_infill",
    "xlocal_bridge",
    "prototypes",
]


def _make_combined_parquet(path: Path, n_rows: int, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    base = pd.DataFrame(
        {
            "sample_idx": np.arange(n_rows, dtype=np.int64),
            "window_name": [f"window_{i}" for i in range(n_rows)],
            "lat": rng.uniform(-60.0, 60.0, size=n_rows).astype(np.float64),
            "lon": rng.uniform(-180.0, 180.0, size=n_rows).astype(np.float64),
            "parent_label": rng.integers(0, 5, size=n_rows).astype(np.int32),
            "combined_score": rng.random(n_rows, dtype=np.float32),
        }
    )
    for strategy in STRATEGIES:
        base[f"{strategy}_normalized_score"] = rng.random(n_rows, dtype=np.float32)
        base[f"combined_score_drop_{strategy}"] = rng.random(n_rows, dtype=np.float32)
    base.to_parquet(path, index=False)
    return base


def test_select_top_samples_writes_expected_columns_and_sizes(tmp_path: Path) -> None:
    """The selection parquet should contain the expected metadata and flags."""
    combined_path = tmp_path / "combined_acquisition_scores.parquet"
    _make_combined_parquet(combined_path, n_rows=200)

    X = 50
    output_path = sts.main(
        [
            "--combined-parquet",
            str(combined_path),
            "--num-samples",
            str(X),
            "--output-dir",
            str(tmp_path),
        ]
    )

    assert output_path == str(tmp_path / f"selection_top{X}.parquet")
    selection = pd.read_parquet(output_path)

    expected_cols = [
        "sample_idx",
        "window_name",
        "lat",
        "lon",
        "parent_label",
        "in_top_combined",
        *[f"in_top_solo_{s}" for s in STRATEGIES],
        *[f"in_top_drop_{s}" for s in STRATEGIES],
    ]
    assert list(selection.columns) == expected_cols

    assert selection["in_top_combined"].sum() == X
    for strategy in STRATEGIES:
        assert selection[f"in_top_solo_{strategy}"].sum() == (3 * X) // 5
        assert selection[f"in_top_drop_{strategy}"].sum() == (4 * X) // 5

    for col in expected_cols[5:]:
        assert set(selection[col].unique()).issubset({0, 1})

    assert selection["sample_idx"].is_unique


def test_select_top_samples_matches_manual_top_k(tmp_path: Path) -> None:
    """Computed top-k flags should match direct manual ranking."""
    combined_path = tmp_path / "combined_acquisition_scores.parquet"
    df = _make_combined_parquet(combined_path, n_rows=120, seed=7)

    X = 25
    sts.main(
        [
            "--combined-parquet",
            str(combined_path),
            "--num-samples",
            str(X),
            "--output-dir",
            str(tmp_path),
        ]
    )
    selection = pd.read_parquet(tmp_path / f"selection_top{X}.parquet")

    manual_full_top = set(
        df.nlargest(X, "combined_score", keep="first")["sample_idx"].tolist()
    )
    got_full_top = set(
        selection.loc[selection["in_top_combined"] == 1, "sample_idx"].tolist()
    )
    assert manual_full_top == got_full_top

    manual_solo = set(
        df.nlargest((3 * X) // 5, "novelty_normalized_score", keep="first")[
            "sample_idx"
        ].tolist()
    )
    got_solo = set(
        selection.loc[selection["in_top_solo_novelty"] == 1, "sample_idx"].tolist()
    )
    assert manual_solo == got_solo

    manual_drop = set(
        df.nlargest(4 * X // 5, "combined_score_drop_prototypes", keep="first")[
            "sample_idx"
        ].tolist()
    )
    got_drop = set(
        selection.loc[selection["in_top_drop_prototypes"] == 1, "sample_idx"].tolist()
    )
    assert manual_drop == got_drop


def test_select_top_samples_fails_without_ablation_columns(tmp_path: Path) -> None:
    """Missing ablation columns should produce a clear validation error."""
    combined_path = tmp_path / "combined_acquisition_scores.parquet"
    df = _make_combined_parquet(combined_path, n_rows=50)
    df = df.drop(columns=[f"combined_score_drop_{s}" for s in STRATEGIES])
    df.to_parquet(combined_path, index=False)

    try:
        sts.main(
            [
                "--combined-parquet",
                str(combined_path),
                "--num-samples",
                "10",
                "--output-dir",
                str(tmp_path),
            ]
        )
    except ValueError as exc:
        assert "combined_score_drop_" in str(exc)
    else:
        raise AssertionError("Expected ValueError when ablation columns are missing")
