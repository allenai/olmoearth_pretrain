"""Tests for overlay projection into frozen embedding layouts."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(__file__))
import visualize_embeddings as ve


def _make_reference_model(centroids: np.ndarray) -> dict[str, np.ndarray]:
    normalized_centers = centroids / np.linalg.norm(centroids, axis=1, keepdims=True)
    return {
        "pca_mean": np.zeros(centroids.shape[1], dtype=np.float32),
        "pca_components": np.eye(centroids.shape[1], dtype=np.float32),
        "normalized_centers": normalized_centers.astype(np.float32),
        "parent_centroids": centroids.astype(np.float32),
    }


def test_overlay_projection_matches_reference_layout_for_identical_points() -> None:
    """Reference points should project back onto their stored layout coordinates."""
    centroids = np.array(
        [
            [2.0, 0.0],
            [-2.0, 0.0],
        ],
        dtype=np.float32,
    )
    reference_points = np.array(
        [
            [2.0, -0.2],
            [2.0, 0.0],
            [2.0, 0.2],
            [-2.0, -0.2],
            [-2.0, 0.0],
            [-2.0, 0.2],
        ],
        dtype=np.float32,
    )
    labels = np.array([0, 0, 0, 1, 1, 1], dtype=np.int32)

    layout_model = ve.build_cluster_layout_model(
        pca_data=reference_points,
        centroids=centroids,
        labels_int=labels,
        metric="euclidean",
        separation=0.6,
        seed=123,
    )
    reference_model = _make_reference_model(centroids)

    overlay_coords, overlay_labels = ve.project_overlay_embeddings(
        embeddings=reference_points,
        reference_model=reference_model,
        layout_model=layout_model,
    )

    assert np.array_equal(overlay_labels, labels)
    assert np.allclose(
        overlay_coords,
        np.asarray(layout_model["coords"], dtype=np.float32),
        atol=1e-5,
    )


def test_validate_overlay_reference_compatibility_rejects_mismatched_centroids() -> (
    None
):
    """Mismatched centroid sets should be rejected before overlay projection."""
    bundle = {
        "centroids": np.array([[1.0, 0.0], [-1.0, 0.0]], dtype=np.float32),
    }
    reference_model = {
        "parent_centroids": np.array([[1.5, 0.0], [-1.5, 0.0]], dtype=np.float32),
    }

    with pytest.raises(ValueError, match="do not match"):
        ve.validate_overlay_reference_compatibility(bundle, reference_model)
