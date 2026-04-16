"""Tests for acquisition scoring strategies and CLI outputs."""

import csv
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
import acquisition_strategies as strat
import score_acquisition as sa
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


def _residuals_by_parent_from_points(
    parent_centroids: np.ndarray,
    points_by_parent: dict[int, np.ndarray],
) -> dict[int, np.ndarray]:
    return {
        parent_idx: (points - parent_centroids[parent_idx]).astype(np.float32)
        for parent_idx, points in points_by_parent.items()
    }


def test_xglobal_bridge_prefers_ambiguous_points() -> None:
    """Bridge scores should favor candidates between two parent clusters."""
    parent_centroids = np.array([[1.0, 0.0], [-1.0, 0.0]], dtype=np.float32)
    reference_points = {
        0: np.array([[0.70, -0.02], [0.85, 0.01], [0.95, -0.01]], dtype=np.float32),
        1: np.array([[-0.70, 0.02], [-0.85, -0.01], [-0.95, 0.01]], dtype=np.float32),
    }
    pca_data = np.array(
        [
            [0.0, 0.0],
            [0.9, 0.0],
            [-0.9, 0.0],
            [0.0, 1.5],
        ],
        dtype=np.float32,
    )

    payload = strat.compute_xglobal_bridge_scores(
        pca_data,
        parent_centroids,
        _residuals_by_parent_from_points(parent_centroids, reference_points),
    )

    assert payload["score"][0] > payload["score"][1]
    assert payload["score"][0] > payload["score"][2]
    assert payload["score"][0] > payload["score"][3]
    assert payload["off_axis_distance"][0] < payload["off_axis_distance"][3]


def test_xglobal_bridge_score_is_invariant_to_other_candidates() -> None:
    """A candidate's xglobal score should not depend on unrelated peers."""
    parent_centroids = np.array([[1.0, 0.0], [-1.0, 0.0]], dtype=np.float32)
    reference_points = {
        0: np.array([[0.70, -0.02], [0.85, 0.01], [0.95, -0.01]], dtype=np.float32),
        1: np.array([[-0.70, 0.02], [-0.85, -0.01], [-0.95, 0.01]], dtype=np.float32),
    }
    residuals_by_parent = _residuals_by_parent_from_points(
        parent_centroids, reference_points
    )

    probe = np.array([[0.0, 0.5]], dtype=np.float32)
    alone = strat.compute_xglobal_bridge_scores(
        probe, parent_centroids, residuals_by_parent
    )
    mixed = strat.compute_xglobal_bridge_scores(
        np.array([[0.0, 0.5], [0.0, 100.0]], dtype=np.float32),
        parent_centroids,
        residuals_by_parent,
    )

    assert np.isclose(alone["score"][0], mixed["score"][0])
    assert np.isclose(alone["between_term"][0], mixed["between_term"][0])
    assert np.isclose(alone["off_axis_scale"][0], mixed["off_axis_scale"][0])


def test_bridge_pair_scale_ignores_points_outside_the_pair_transition_band() -> None:
    """Pair scales should be driven by the transition band between the pair."""
    centers = np.array([[-1.0, 0.0], [1.0, 0.0], [-1.0, 1.0]], dtype=np.float32)
    reference_points = np.array(
        [
            [-0.95, 0.75],
            [-0.90, 0.80],
            [0.00, 0.00],
            [0.00, 0.05],
            [0.95, 0.00],
            [0.90, -0.02],
            [-1.00, 1.00],
            [-0.95, 0.95],
        ],
        dtype=np.float32,
    )
    owner_labels = np.array([0, 0, 0, 1, 1, 1, 2, 2], dtype=np.int32)

    pair_scales = strat._estimate_bridge_pair_scales(
        reference_points, owner_labels, centers
    )

    assert pair_scales[(0, 1)] < 0.2


def test_xglobal_bridge_handles_single_parent_cluster() -> None:
    """Single-parent references should yield zero bridge scores."""
    payload = strat.compute_xglobal_bridge_scores(
        pca_data=np.array([[0.0, 0.0], [1.0, 0.0]], dtype=np.float32),
        parent_centroids=np.array([[0.0, 0.0]], dtype=np.float32),
        residuals_by_parent={0: np.array([[0.0, 0.0]], dtype=np.float32)},
    )

    assert np.allclose(payload["score"], 0.0)
    assert np.all(payload["secondary_parent"] == -1)


def test_sparse_infill_prefers_sparse_but_supported_points() -> None:
    """Sparse-infill should reward points in sparse but supported regions."""
    reference = np.array(
        [
            [0.00, 0.00],
            [0.02, 0.00],
            [0.04, 0.00],
            [0.06, 0.00],
            [0.08, 0.00],
            [0.30, 0.00],
            [0.36, 0.00],
            [0.42, 0.00],
            [0.48, 0.00],
        ],
        dtype=np.float32,
    )
    queries = np.array(
        [
            [0.03, 0.00],
            [0.22, 0.00],
            [1.20, 0.00],
        ],
        dtype=np.float32,
    )
    parent_labels = np.zeros(queries.shape[0], dtype=np.int32)

    payload = strat.compute_sparse_infill_scores(
        residuals=queries,
        parent_labels=parent_labels,
        residuals_by_parent={0: reference},
        k_sparse=4,
        k_support=2,
        sparse_percentile=70.0,
        support_percentile=60.0,
        metric="euclidean",
    )

    assert payload["score"][1] > payload["score"][0]
    assert payload["score"][1] > payload["score"][2]


def test_sparse_infill_support_requires_more_than_one_nearby_point() -> None:
    """Support should drop when a candidate lacks multiple nearby references."""
    reference = np.array(
        [
            [0.00, 0.00],
            [0.02, 0.00],
            [0.04, 0.00],
            [0.06, 0.00],
            [0.08, 0.00],
            [0.30, 0.00],
            [0.34, 0.00],
            [0.38, 0.00],
            [1.00, 0.00],
        ],
        dtype=np.float32,
    )
    queries = np.array(
        [
            [0.24, 0.00],
            [1.02, 0.00],
        ],
        dtype=np.float32,
    )
    parent_labels = np.zeros(queries.shape[0], dtype=np.int32)

    payload = strat.compute_sparse_infill_scores(
        residuals=queries,
        parent_labels=parent_labels,
        residuals_by_parent={0: reference},
        k_sparse=4,
        k_support=2,
        sparse_percentile=70.0,
        support_percentile=60.0,
        metric="euclidean",
    )

    assert (
        payload["support_mean_knn_distance"][0]
        < payload["support_mean_knn_distance"][1]
    )
    assert payload["support_term"][0] > payload["support_term"][1]


def test_xlocal_bridge_prefers_connector_between_two_modes() -> None:
    """Local-bridge scoring should favor connectors between residual modes."""
    reference = np.array(
        [
            [-0.45, -0.02],
            [-0.40, 0.02],
            [-0.35, -0.01],
            [0.35, 0.01],
            [0.40, -0.02],
            [0.45, 0.02],
        ],
        dtype=np.float32,
    )
    queries = np.array(
        [
            [-0.40, 0.00],
            [0.00, 0.00],
            [1.20, 0.00],
        ],
        dtype=np.float32,
    )
    parent_labels = np.zeros(queries.shape[0], dtype=np.int32)

    payload = strat.compute_xlocal_bridge_scores(
        residuals=queries,
        parent_labels=parent_labels,
        residuals_by_parent={0: reference},
        n_local_modes=2,
        seed=42,
    )

    assert payload["score"][1] > payload["score"][0]
    assert payload["score"][1] > payload["score"][2]


def test_xlocal_bridge_penalizes_off_axis_and_beyond_segment_points() -> None:
    """Local-bridge scoring should penalize off-axis and overshot candidates."""
    reference = np.array(
        [
            [-0.45, -0.02],
            [-0.40, 0.01],
            [-0.35, -0.01],
            [0.35, 0.01],
            [0.40, -0.01],
            [0.45, 0.02],
        ],
        dtype=np.float32,
    )
    queries = np.array(
        [
            [0.00, 0.00],
            [0.00, 0.30],
            [0.80, 0.00],
        ],
        dtype=np.float32,
    )
    parent_labels = np.zeros(queries.shape[0], dtype=np.int32)

    payload = strat.compute_xlocal_bridge_scores(
        residuals=queries,
        parent_labels=parent_labels,
        residuals_by_parent={0: reference},
        n_local_modes=2,
        seed=42,
    )

    assert payload["score"][0] > payload["score"][1]
    assert payload["score"][0] > payload["score"][2]
    assert payload["off_axis_distance"][0] < payload["off_axis_distance"][1]
    assert payload["segment_position_term"][0] > payload["segment_position_term"][2]


def test_prototype_score_prefers_local_representative_points() -> None:
    """Prototype scoring should rank local representatives above distant points."""
    reference = np.array(
        [
            [-0.04, 0.00],
            [0.00, 0.00],
            [0.04, 0.00],
            [0.96, 0.00],
            [1.00, 0.00],
            [1.04, 0.00],
        ],
        dtype=np.float32,
    )
    queries = np.array(
        [
            [0.00, 0.00],
            [0.25, 0.00],
            [1.80, 0.00],
        ],
        dtype=np.float32,
    )
    parent_labels = np.zeros(queries.shape[0], dtype=np.int32)

    payload = strat.compute_prototype_scores(
        residuals=queries,
        parent_labels=parent_labels,
        residuals_by_parent={0: reference},
        n_local_prototypes=2,
        radius_percentile=80.0,
        coverage_k=0,
        seed=42,
    )

    assert payload["score"][0] > payload["score"][1]
    assert payload["score"][1] > payload["score"][2]
    assert payload["normalized_distance"][0] < payload["normalized_distance"][1]
    assert payload["normalized_distance"][1] < payload["normalized_distance"][2]


def test_prototype_coverage_penalizes_dense_regions() -> None:
    """Coverage penalties should reduce scores in already-dense regions."""
    reference = np.array(
        [
            [0.00, 0.00],
            [0.01, 0.00],
            [0.02, 0.00],
            [0.03, 0.00],
            [0.04, 0.00],
            [0.05, 0.00],
            [1.00, 0.00],
            [1.10, 0.00],
            [1.20, 0.00],
            [1.30, 0.00],
        ],
        dtype=np.float32,
    )
    queries = np.array(
        [
            [0.02, 0.00],
            [1.15, 0.00],
        ],
        dtype=np.float32,
    )
    parent_labels = np.zeros(queries.shape[0], dtype=np.int32)

    payload_no_cov = strat.compute_prototype_scores(
        residuals=queries,
        parent_labels=parent_labels,
        residuals_by_parent={0: reference},
        n_local_prototypes=2,
        radius_percentile=80.0,
        coverage_k=0,
        seed=42,
    )
    payload_with_cov = strat.compute_prototype_scores(
        residuals=queries,
        parent_labels=parent_labels,
        residuals_by_parent={0: reference},
        n_local_prototypes=2,
        radius_percentile=80.0,
        coverage_k=3,
        seed=42,
    )

    assert np.allclose(payload_no_cov["coverage_term"], 1.0)
    assert payload_with_cov["coverage_term"][0] <= 1.0
    assert payload_with_cov["coverage_term"][1] <= 1.0


def test_xglobal_bridge_cli_outputs_align_with_index(tmp_path: Path) -> None:
    """CLI parquet outputs should stay aligned with the candidate index order."""
    ref_dir = tmp_path / "reference"
    candidate_dir = tmp_path / "candidate"

    reference = np.array(
        [
            [2.0, 0.0],
            [2.1, 0.1],
            [1.9, -0.1],
            [-2.0, 0.0],
            [-2.1, -0.1],
            [-1.9, 0.1],
        ],
        dtype=np.float32,
    )
    candidates = np.array(
        [
            [2.0, 0.0],
            [0.0, 0.0],
            [-2.0, 0.0],
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
            "2",
            "--percentile",
            "95",
            "--seed",
            "11",
        ]
    )
    sa.main(
        [
            "xglobal_bridge",
            "--input-dir",
            str(candidate_dir),
            "--reference-dir",
            str(ref_dir / "_scores"),
            "--output-dir",
            str(candidate_dir / "_scores"),
            "--seed",
            "11",
        ]
    )

    scores_df = pd.read_parquet(
        candidate_dir / "_scores" / "xglobal_bridge_scores.parquet"
    )
    assert scores_df["sample_idx"].tolist() == [0, 1, 2]

    ranked = np.load(candidate_dir / "_scores" / "xglobal_bridge_ranked_sample_idx.npy")
    assert ranked[0] == 1
