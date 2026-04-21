"""Score candidate embeddings for novelty against a frozen reference set.

Each candidate is:

1. projected with the frozen PCA
2. assigned to a frozen parent spherical k-means cluster
3. converted into a residual vector relative to that parent centroid
4. scored by mean distance to the k nearest reference residuals inside
   that parent
5. compared to the stored parent-specific percentile threshold
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
from reference_model import (
    assign_spherical_kmeans,
    compute_residuals,
    ensure_index_rows,
    load_embeddings,
    load_reference_artifacts,
    save_summary_json,
    scores_to_accept_mask,
    transform_with_pca,
    write_scores_parquet,
)


def compute_candidate_scores(
    candidate_residuals: np.ndarray,
    parent_labels: np.ndarray,
    residuals_by_parent: dict[int, np.ndarray],
    knn_k: int,
    metric: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Score candidates against reference residuals in their assigned parent."""
    from sklearn.neighbors import NearestNeighbors

    scores = np.zeros(candidate_residuals.shape[0], dtype=np.float32)
    parent_k_used = np.zeros(candidate_residuals.shape[0], dtype=np.int32)

    for parent_idx, reference_residuals in residuals_by_parent.items():
        mask = parent_labels == parent_idx
        if not np.any(mask):
            continue

        parent_queries = candidate_residuals[mask]
        if reference_residuals.shape[0] == 0:
            scores[mask] = 0.0
            parent_k_used[mask] = 0
            continue

        effective_k = min(knn_k, reference_residuals.shape[0])
        nn = NearestNeighbors(n_neighbors=effective_k, metric=metric)
        nn.fit(reference_residuals)
        distances, _ = nn.kneighbors(parent_queries)
        scores[mask] = distances.mean(axis=1).astype(np.float32)
        parent_k_used[mask] = effective_k

    return scores, parent_k_used


def score_candidates(args: argparse.Namespace) -> None:
    """Score a candidate directory against a frozen reference model."""
    output_dir = args.output_dir or os.path.join(args.input_dir, "_scores")
    os.makedirs(output_dir, exist_ok=True)

    model, residuals_by_parent = load_reference_artifacts(args.reference_dir)
    embeddings, kept_indices = load_embeddings(args.input_dir, 0, args.seed)
    if kept_indices is not None:
        raise AssertionError("Candidate scoring should not subsample embeddings")
    index_rows = ensure_index_rows(args.input_dir, embeddings.shape[0], None)

    pca_data = transform_with_pca(
        embeddings,
        model["pca_mean"],
        model["pca_components"],
    )
    del embeddings

    parent_labels = assign_spherical_kmeans(pca_data, model["normalized_centers"])
    residuals = compute_residuals(pca_data, parent_labels, model["parent_centroids"])
    knn_k = int(model["knn_k"].item())
    distance_metric = str(model["distance_metric"].item())
    scores, k_used = compute_candidate_scores(
        residuals,
        parent_labels,
        residuals_by_parent,
        knn_k=knn_k,
        metric=distance_metric,
    )

    thresholds_by_parent = model["thresholds"].astype(np.float32)
    thresholds = thresholds_by_parent[parent_labels]
    accepted = scores_to_accept_mask(scores, thresholds)

    np.save(
        os.path.join(output_dir, "novelty_accepted_mask.npy"), accepted.astype(bool)
    )
    np.save(
        os.path.join(output_dir, "novelty_accepted_sample_idx.npy"),
        np.array([int(row["sample_idx"]) for row in index_rows], dtype=np.int64)[
            accepted
        ],
    )
    np.save(
        os.path.join(output_dir, "novelty_parent_assignments.npy"),
        parent_labels.astype(np.int32),
    )
    print(
        f"[save] Accepted mask -> {os.path.join(output_dir, 'novelty_accepted_mask.npy')}"
    )
    print(
        f"[save] Accepted sample indices -> "
        f"{os.path.join(output_dir, 'novelty_accepted_sample_idx.npy')}"
    )
    print(
        f"[save] Parent assignments -> "
        f"{os.path.join(output_dir, 'novelty_parent_assignments.npy')}"
    )

    scores_path = os.path.join(output_dir, "novelty_scores.parquet")
    write_scores_parquet(
        scores_path,
        index_rows=index_rows,
        parent_labels=parent_labels,
        scores=scores,
        thresholds=thresholds,
        accepted=accepted,
        source_name="candidate",
        k_used=k_used,
    )

    summary = {
        "input_dir": args.input_dir,
        "reference_dir": args.reference_dir,
        "output_dir": output_dir,
        "candidate_size": int(pca_data.shape[0]),
        "accepted_count": int(accepted.sum()),
        "accepted_fraction": float(accepted.mean()) if accepted.size else 0.0,
        "knn_k": knn_k,
        "distance_metric": distance_metric,
    }
    save_summary_json(os.path.join(output_dir, "novelty_summary.json"), summary)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for novelty scoring."""
    parser = argparse.ArgumentParser(
        description="Score candidate embeddings for novelty against frozen reference artifacts.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input-dir", required=True, help="Candidate directory with shard_*.npz"
    )
    parser.add_argument(
        "--reference-dir",
        required=True,
        help="Directory containing reference_model.npz and reference_residuals.npz",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Where to write candidate scores. Defaults to {input_dir}/_scores",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Unused except for CLI parity"
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Run novelty scoring."""
    args = parse_args(argv)
    score_candidates(args)


if __name__ == "__main__":
    main(sys.argv[1:])
