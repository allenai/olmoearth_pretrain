"""Tests for weighted acquisition-score combination."""

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
import combine_acquisition as ca


def _write_parquet(path: Path, df: pd.DataFrame) -> None:
    df.to_parquet(path, index=False)


def _make_scores_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    base = pd.DataFrame(
        {
            "sample_idx": np.array([0, 1, 2], dtype=np.int64),
            "window_name": ["window_0", "window_1", "window_2"],
            "lat": np.zeros(3, dtype=np.float64),
            "lon": np.zeros(3, dtype=np.float64),
            "parent_label": np.array([0, 0, 1], dtype=np.int32),
        }
    )

    novelty = base.copy()
    novelty.insert(0, "source", "candidate")
    novelty["novelty_score"] = np.array([0.10, 0.95, 0.40], dtype=np.float32)
    novelty["threshold"] = np.zeros(3, dtype=np.float32)
    novelty["accepted"] = np.array([False, True, False], dtype=bool)
    _write_parquet(path / "novelty_scores.parquet", novelty)

    def _with_score(values: list[float]) -> pd.DataFrame:
        df = base.copy()
        df["score"] = np.array(values, dtype=np.float32)
        return df

    _write_parquet(
        path / "xglobal_bridge_scores.parquet", _with_score([0.20, 0.30, 0.95])
    )
    _write_parquet(
        path / "sparse_infill_scores.parquet", _with_score([0.70, 0.40, 0.20])
    )
    _write_parquet(
        path / "xlocal_bridge_scores.parquet", _with_score([0.10, 0.80, 0.20])
    )
    _write_parquet(path / "prototypes_scores.parquet", _with_score([0.95, 0.50, 0.10]))


def test_combine_acquisition_prefers_weighted_mix(tmp_path: Path) -> None:
    """The combined ranking should reflect the configured weighted mix."""
    scores_dir = tmp_path / "scores"
    _make_scores_dir(scores_dir)

    ca.main(
        [
            "--scores-dir",
            str(scores_dir),
            "--normalization",
            "rank",
            "--weight-novelty",
            "0.4",
            "--weight-xglobal-bridge",
            "0.2",
            "--weight-sparse-infill",
            "0.2",
            "--weight-xlocal-bridge",
            "0.1",
            "--weight-prototypes",
            "0.1",
        ]
    )

    ranked = np.load(scores_dir / "combined_ranked_sample_idx.npy")
    assert ranked.tolist() == [1, 2, 0]

    df = pd.read_parquet(scores_dir / "combined_acquisition_scores.parquet")
    assert df["sample_idx"].tolist() == [0, 1, 2]
    assert "novelty_normalized_score" in df.columns
    assert "xglobal_bridge_raw_score" in df.columns
    assert "combined_score" in df.columns
    assert not any(col.startswith("combined_score_drop_") for col in df.columns)


def test_combine_acquisition_ignores_unweighted_missing_files(tmp_path: Path) -> None:
    """Zero-weight strategies should not require their parquet inputs."""
    scores_dir = tmp_path / "scores"
    scores_dir.mkdir(parents=True, exist_ok=True)

    novelty = pd.DataFrame(
        {
            "source": ["candidate", "candidate"],
            "sample_idx": np.array([0, 1], dtype=np.int64),
            "window_name": ["window_0", "window_1"],
            "lat": np.zeros(2, dtype=np.float64),
            "lon": np.zeros(2, dtype=np.float64),
            "parent_label": np.array([0, 0], dtype=np.int32),
            "novelty_score": np.array([0.2, 0.9], dtype=np.float32),
            "threshold": np.zeros(2, dtype=np.float32),
            "accepted": np.array([False, True], dtype=bool),
        }
    )
    _write_parquet(scores_dir / "novelty_scores.parquet", novelty)

    ca.main(
        [
            "--scores-dir",
            str(scores_dir),
            "--weight-novelty",
            "1.0",
            "--weight-xglobal-bridge",
            "0.0",
            "--weight-sparse-infill",
            "0.0",
            "--weight-xlocal-bridge",
            "0.0",
            "--weight-prototypes",
            "0.0",
        ]
    )

    ranked = np.load(scores_dir / "combined_ranked_sample_idx.npy")
    assert ranked.tolist() == [1, 0]


def test_combine_acquisition_ablation_emits_leave_one_out_columns(
    tmp_path: Path,
) -> None:
    """Ablation mode should emit leave-one-out score columns and rankings."""
    scores_dir = tmp_path / "scores"
    _make_scores_dir(scores_dir)

    ca.main(
        [
            "--scores-dir",
            str(scores_dir),
            "--normalization",
            "rank",
            "--weight-novelty",
            "0.4",
            "--weight-xglobal-bridge",
            "0.2",
            "--weight-sparse-infill",
            "0.2",
            "--weight-xlocal-bridge",
            "0.1",
            "--weight-prototypes",
            "0.1",
            "--ablation",
        ]
    )

    df = pd.read_parquet(scores_dir / "combined_acquisition_scores.parquet")

    expected_strategies = [
        "novelty",
        "xglobal_bridge",
        "sparse_infill",
        "xlocal_bridge",
        "prototypes",
    ]
    for strategy in expected_strategies:
        assert f"combined_score_drop_{strategy}" in df.columns
        assert (scores_dir / f"combined_ranked_sample_idx_drop_{strategy}.npy").exists()

    assert not any(col.startswith("combined_score_solo_") for col in df.columns)
    assert "combined_score_random" not in df.columns

    # leave-one-out with one strategy dropped must equal a weighted mean of
    # the other four normalized scores using the remaining weights.
    weights = {
        "novelty": 0.4,
        "xglobal_bridge": 0.2,
        "sparse_infill": 0.2,
        "xlocal_bridge": 0.1,
        "prototypes": 0.1,
    }
    dropped = "novelty"
    remaining = {k: v for k, v in weights.items() if k != dropped}
    expected = np.zeros(len(df), dtype=np.float32)
    total = 0.0
    for strategy, weight in remaining.items():
        expected += weight * df[f"{strategy}_normalized_score"].to_numpy(
            dtype=np.float32
        )
        total += weight
    expected /= total
    assert np.allclose(
        df[f"combined_score_drop_{dropped}"].to_numpy(), expected, atol=1e-6
    )
