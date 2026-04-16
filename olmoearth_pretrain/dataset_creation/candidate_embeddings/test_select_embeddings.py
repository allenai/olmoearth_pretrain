"""Tests for frozen-reference novelty selection."""

import csv
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
import select_embeddings as se


def _write_embedding_dir(path: Path, embeddings: np.ndarray) -> None:
    path.mkdir(parents=True, exist_ok=True)
    np.savez(path / "shard_0000.npz", embeddings=embeddings.astype(np.float32))

    with open(path / "index.csv", "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["sample_idx", "shard", "row", "lat", "lon", "window_name"],
        )
        writer.writeheader()
        for i in range(embeddings.shape[0]):
            writer.writerow(
                {
                    "sample_idx": i,
                    "shard": 0,
                    "row": i,
                    "lat": 0.0,
                    "lon": 0.0,
                    "window_name": f"window_{i}",
                }
            )


def test_fit_and_score_pipeline_selects_novel_candidate(tmp_path: Path) -> None:
    """The end-to-end novelty pipeline should keep the clearly novel point."""
    ref_dir = tmp_path / "reference"
    candidate_dir = tmp_path / "candidate"

    reference = np.array(
        [
            [0.00, 0.00, 0.00, 0.00],
            [0.10, 0.00, 0.00, 0.00],
            [-0.10, 0.00, 0.00, 0.00],
            [0.00, 0.08, 0.00, 0.00],
            [0.00, -0.08, 0.00, 0.00],
            [0.04, 0.02, 0.00, 0.00],
        ],
        dtype=np.float32,
    )
    candidates = np.array(
        [
            [0.03, 0.01, 0.00, 0.00],
            [1.20, 0.90, 0.00, 0.00],
        ],
        dtype=np.float32,
    )

    _write_embedding_dir(ref_dir, reference)
    _write_embedding_dir(candidate_dir, candidates)

    se.main(
        [
            "fit-reference",
            "--input-dir",
            str(ref_dir),
            "--output-dir",
            str(ref_dir / "_scores"),
            "--pca-dim",
            "2",
            "--k",
            "1",
            "--knn-k",
            "2",
            "--percentile",
            "90",
            "--seed",
            "123",
        ]
    )
    se.main(
        [
            "novelty-score",
            "--input-dir",
            str(candidate_dir),
            "--reference-dir",
            str(ref_dir / "_scores"),
            "--output-dir",
            str(candidate_dir / "_scores"),
        ]
    )

    accepted = np.load(candidate_dir / "_scores" / "novelty_accepted_mask.npy")
    assert accepted.tolist() == [False, True]

    accepted_sample_idx = np.load(
        candidate_dir / "_scores" / "novelty_accepted_sample_idx.npy"
    )
    assert accepted_sample_idx.tolist() == [1]


def test_parent_assignment_and_small_cluster_knn_cap(tmp_path: Path) -> None:
    """Candidate scoring should cap kNN usage by the parent reference size."""
    ref_dir = tmp_path / "reference"
    candidate_dir = tmp_path / "candidate"

    reference = np.array(
        [
            [2.0, 0.0, 0.0, 0.0],
            [2.1, 0.1, 0.0, 0.0],
            [1.9, -0.1, 0.0, 0.0],
            [-2.0, 0.0, 0.0, 0.0],
            [-2.1, -0.1, 0.0, 0.0],
            [-1.9, 0.1, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    candidates = np.array(
        [
            [2.05, 0.02, 0.0, 0.0],
            [-2.05, -0.02, 0.0, 0.0],
        ],
        dtype=np.float32,
    )

    _write_embedding_dir(ref_dir, reference)
    _write_embedding_dir(candidate_dir, candidates)

    se.main(
        [
            "fit-reference",
            "--input-dir",
            str(ref_dir),
            "--output-dir",
            str(ref_dir / "_scores"),
            "--pca-dim",
            "2",
            "--k",
            "2",
            "--knn-k",
            "5",
            "--percentile",
            "95",
            "--seed",
            "7",
        ]
    )
    se.main(
        [
            "novelty-score",
            "--input-dir",
            str(candidate_dir),
            "--reference-dir",
            str(ref_dir / "_scores"),
            "--output-dir",
            str(candidate_dir / "_scores"),
        ]
    )

    parent_assignments = np.load(
        candidate_dir / "_scores" / "novelty_parent_assignments.npy"
    )
    assert parent_assignments.shape == (2,)
    assert parent_assignments[0] != parent_assignments[1]

    scores_df = pd.read_parquet(candidate_dir / "_scores" / "novelty_scores.parquet")
    assert scores_df["k_used"].tolist() == [3, 3]
