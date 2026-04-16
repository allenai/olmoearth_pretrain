"""Select novel embeddings against a frozen reference set.

This script implements a two-step workflow:

1. fit-reference
   - Load a reference embedding directory containing ``shard_*.npz``.
   - Fit a global PCA model.
   - Fit parent spherical k-means clusters in the PCA space.
   - Compute residual vectors within each parent cluster.
   - Calibrate a novelty threshold per parent cluster using reference
     self-scores based on k-nearest-neighbor residual distance.

2. novelty-score
   - Load a candidate embedding directory.
   - Transform with the frozen PCA model.
   - Assign each candidate to a parent cluster using frozen spherical
     k-means centers.
   - Score local novelty as mean distance to the k nearest reference
     residuals in the assigned parent cluster.
   - Accept candidates whose score is above the parent-specific threshold.

Artifacts are written under ``_scores/`` by default so the raw shards remain
unchanged.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import sys
import time

import numpy as np
import pandas as pd

EPS = 1e-10


def load_embeddings(
    input_dir: str,
    subsample: int = 0,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Load shard_*.npz embeddings and optionally return kept row positions."""
    shard_paths = sorted(glob.glob(os.path.join(input_dir, "shard_*.npz")))
    if not shard_paths:
        raise FileNotFoundError(f"No shard_*.npz files found in {input_dir}")

    print(f"[load] Found {len(shard_paths)} shards in {input_dir}")
    arrays = []
    for path in shard_paths:
        data = np.load(path)
        arrays.append(data["embeddings"])

    embeddings = np.concatenate(arrays, axis=0).astype(np.float32)
    print(
        f"[load] Loaded {embeddings.shape[0]:,} embeddings of dim {embeddings.shape[1]}  "
        f"({embeddings.nbytes / 1e9:.2f} GB)"
    )

    kept_indices: np.ndarray | None = None
    if subsample > 0 and subsample < embeddings.shape[0]:
        rng = np.random.RandomState(seed)
        kept_indices = rng.choice(embeddings.shape[0], size=subsample, replace=False)
        kept_indices.sort()
        embeddings = embeddings[kept_indices]
        print(f"[load] Subsampled to {embeddings.shape[0]:,} embeddings")

    return embeddings, kept_indices


def load_index_rows(
    input_dir: str,
    kept_indices: np.ndarray | None = None,
) -> list[dict[str, str]]:
    """Load index.csv rows and optionally subset them to kept_indices."""
    index_csv_path = os.path.join(input_dir, "index.csv")
    if not os.path.exists(index_csv_path):
        print(f"[warn] No index.csv found at {index_csv_path}; using synthetic rows")
        rows = [
            {
                "sample_idx": str(i),
                "shard": "",
                "row": str(i),
                "lat": "",
                "lon": "",
                "window_name": "",
            }
            for i in range(0)
        ]
        return rows

    with open(index_csv_path, newline="") as f:
        rows = list(csv.DictReader(f))

    if kept_indices is not None:
        rows = [rows[int(i)] for i in kept_indices]

    return rows


def build_fallback_index_rows(n_rows: int) -> list[dict[str, str]]:
    """Create synthetic index rows when index.csv is unavailable."""
    return [
        {
            "sample_idx": str(i),
            "shard": "",
            "row": str(i),
            "lat": "",
            "lon": "",
            "window_name": "",
        }
        for i in range(n_rows)
    ]


def ensure_index_rows(
    input_dir: str,
    n_rows: int,
    kept_indices: np.ndarray | None = None,
) -> list[dict[str, str]]:
    """Return index rows aligned with the loaded embeddings."""
    rows = load_index_rows(input_dir, kept_indices)
    if not rows:
        rows = build_fallback_index_rows(n_rows)

    if len(rows) != n_rows:
        raise ValueError(
            f"index alignment mismatch for {input_dir}: {len(rows)} rows in index "
            f"but {n_rows} embeddings"
        )
    return rows


def fit_global_pca(
    embeddings: np.ndarray,
    n_components: int,
    seed: int,
) -> tuple[np.ndarray, dict[str, np.ndarray | float | int]]:
    """Fit PCA and return transformed embeddings plus serializable state."""
    from sklearn.decomposition import PCA

    print(f"[pca] Reducing {embeddings.shape} -> (N, {n_components}) ...")
    t0 = time.time()
    pca = PCA(n_components=n_components, random_state=seed)
    reduced = pca.fit_transform(embeddings).astype(np.float32)
    elapsed = time.time() - t0
    print(
        f"[pca] Done in {elapsed:.1f}s  "
        f"(explained variance: {pca.explained_variance_ratio_.sum():.2%})"
    )

    state: dict[str, np.ndarray | float | int] = {
        "pca_mean": pca.mean_.astype(np.float32),
        "pca_components": pca.components_.astype(np.float32),
        "pca_explained_variance": pca.explained_variance_.astype(np.float32),
        "pca_explained_variance_ratio": pca.explained_variance_ratio_.astype(
            np.float32
        ),
        "pca_n_components": np.int32(pca.n_components_),
    }
    return reduced, state


def transform_with_pca(
    embeddings: np.ndarray,
    pca_mean: np.ndarray,
    pca_components: np.ndarray,
) -> np.ndarray:
    """Project embeddings with a frozen PCA model."""
    return ((embeddings - pca_mean) @ pca_components.T).astype(np.float32)


def normalize_rows(data: np.ndarray) -> np.ndarray:
    """L2-normalize rows with the same epsilon policy as cluster_embeddings.py."""
    norms = np.linalg.norm(data, axis=1, keepdims=True)
    norms = np.maximum(norms, EPS)
    return data / norms


def fit_spherical_kmeans(
    data: np.ndarray,
    n_clusters: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fit spherical MiniBatchKMeans on PCA vectors."""
    from sklearn.cluster import MiniBatchKMeans

    normalized = normalize_rows(data)
    print(
        f"[kmeans] Clustering {data.shape[0]:,} points in {data.shape[1]}D  "
        f"(k={n_clusters}, spherical k-means) ..."
    )
    t0 = time.time()
    km = MiniBatchKMeans(
        n_clusters=n_clusters,
        random_state=seed,
        batch_size=max(1024, n_clusters * 10),
    )
    labels = km.fit_predict(normalized)
    elapsed = time.time() - t0
    print(f"[kmeans] Done in {elapsed:.1f}s  (inertia={km.inertia_:.2e})")

    parent_centroids = np.zeros((n_clusters, data.shape[1]), dtype=np.float32)
    for k in range(n_clusters):
        parent_centroids[k] = data[labels == k].mean(axis=0)

    return (
        labels.astype(np.int32),
        km.cluster_centers_.astype(np.float32),
        parent_centroids,
    )


def assign_spherical_kmeans(
    data: np.ndarray,
    normalized_centers: np.ndarray,
) -> np.ndarray:
    """Assign rows to the nearest frozen spherical k-means center."""
    from sklearn.metrics import pairwise_distances_argmin

    normalized = normalize_rows(data)
    labels = pairwise_distances_argmin(
        normalized, normalized_centers, metric="euclidean"
    )
    return labels.astype(np.int32)


def compute_residuals(
    data: np.ndarray,
    labels: np.ndarray,
    parent_centroids: np.ndarray,
) -> np.ndarray:
    """Residual vectors in the original PCA space."""
    return (data - parent_centroids[labels]).astype(np.float32)


def compute_parent_knn_self_scores(
    residuals: np.ndarray,
    labels: np.ndarray,
    n_clusters: int,
    knn_k: int,
    metric: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute per-reference-point novelty scores and per-parent thresholds inputs."""
    from sklearn.neighbors import NearestNeighbors

    if knn_k <= 0:
        raise ValueError(f"knn_k must be > 0, got {knn_k}")

    scores = np.zeros(residuals.shape[0], dtype=np.float32)
    parent_counts = np.zeros(n_clusters, dtype=np.int32)
    parent_k_used = np.zeros(n_clusters, dtype=np.int32)

    for parent_idx in range(n_clusters):
        mask = labels == parent_idx
        parent_residuals = residuals[mask]
        n_points = parent_residuals.shape[0]
        parent_counts[parent_idx] = n_points
        if n_points == 0:
            continue
        if n_points == 1:
            scores[mask] = 0.0
            parent_k_used[parent_idx] = 0
            continue

        effective_k = min(knn_k, n_points - 1)
        parent_k_used[parent_idx] = effective_k
        nn = NearestNeighbors(n_neighbors=effective_k + 1, metric=metric)
        nn.fit(parent_residuals)
        distances, _ = nn.kneighbors(parent_residuals)
        scores[mask] = distances[:, 1:].mean(axis=1).astype(np.float32)

    return scores, parent_counts, parent_k_used


def compute_thresholds(
    scores: np.ndarray,
    labels: np.ndarray,
    n_clusters: int,
    percentile: float,
) -> np.ndarray:
    """Compute a novelty threshold per parent cluster."""
    thresholds = np.zeros(n_clusters, dtype=np.float32)
    for parent_idx in range(n_clusters):
        parent_scores = scores[labels == parent_idx]
        if parent_scores.size == 0:
            thresholds[parent_idx] = 0.0
            continue
        thresholds[parent_idx] = float(np.percentile(parent_scores, percentile))
    return thresholds


def save_reference_artifacts(
    output_dir: str,
    model_state: dict[str, np.ndarray | float | int],
    residuals: np.ndarray,
    labels: np.ndarray,
    self_scores: np.ndarray,
    index_rows: list[dict[str, str]],
) -> None:
    """Persist frozen model state and reference residual shards."""
    os.makedirs(output_dir, exist_ok=True)

    model_path = os.path.join(output_dir, "reference_model.npz")
    np.savez(model_path, **model_state)
    print(f"[save] Reference model -> {model_path}")

    labels_path = os.path.join(output_dir, "reference_parent_labels.npy")
    np.save(labels_path, labels)
    print(f"[save] Reference parent labels -> {labels_path}")

    residual_payload: dict[str, np.ndarray] = {}
    n_clusters = int(np.asarray(model_state["n_clusters"]).item())
    sample_indices = np.array(
        [int(row["sample_idx"]) for row in index_rows], dtype=np.int64
    )
    for parent_idx in range(n_clusters):
        mask = labels == parent_idx
        residual_payload[f"parent_{parent_idx:03d}_residuals"] = residuals[mask]
        residual_payload[f"parent_{parent_idx:03d}_sample_idx"] = sample_indices[mask]
    residual_path = os.path.join(output_dir, "reference_residuals.npz")
    np.savez(residual_path, **residual_payload)
    print(f"[save] Reference residuals -> {residual_path}")

    reference_scores_path = os.path.join(output_dir, "reference_scores.parquet")
    write_scores_parquet(
        reference_scores_path,
        index_rows=index_rows,
        parent_labels=labels,
        scores=self_scores,
        thresholds=np.asarray(model_state["thresholds"], dtype=np.float32)[labels],
        accepted=scores_to_accept_mask(
            self_scores,
            np.asarray(model_state["thresholds"], dtype=np.float32)[labels],
        ),
        source_name="reference",
    )


def load_reference_artifacts(
    reference_dir: str,
) -> tuple[dict[str, np.ndarray], dict[int, np.ndarray]]:
    """Load frozen model state and residual arrays keyed by parent index."""
    model_path = os.path.join(reference_dir, "reference_model.npz")
    residual_path = os.path.join(reference_dir, "reference_residuals.npz")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Missing reference model: {model_path}")
    if not os.path.exists(residual_path):
        raise FileNotFoundError(f"Missing reference residuals: {residual_path}")

    with np.load(model_path, allow_pickle=False) as data:
        model = {key: data[key] for key in data.files}

    residuals_by_parent: dict[int, np.ndarray] = {}
    with np.load(residual_path, allow_pickle=False) as data:
        n_clusters = int(model["n_clusters"].item())
        for parent_idx in range(n_clusters):
            residuals_by_parent[parent_idx] = data[f"parent_{parent_idx:03d}_residuals"]

    return model, residuals_by_parent


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


def scores_to_accept_mask(scores: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    """Accept scores strictly above the calibrated threshold."""
    return scores > thresholds


def _row_float(value: object) -> float:
    """Best-effort float cast that tolerates blanks from index.csv."""
    if value is None or value == "":
        return float("nan")
    if isinstance(
        value,
        int | float | str | bytes | bytearray | np.integer | np.floating,
    ):
        try:
            return float(value)
        except ValueError:
            return float("nan")
    return float("nan")


def write_scores_parquet(
    path: str,
    index_rows: list[dict[str, str]],
    parent_labels: np.ndarray,
    scores: np.ndarray,
    thresholds: np.ndarray,
    accepted: np.ndarray,
    source_name: str,
    k_used: np.ndarray | None = None,
) -> None:
    """Write aligned scores keyed by sample_idx and window_name as Parquet."""
    data: dict[str, np.ndarray | list] = {
        "source": [source_name] * len(index_rows),
        "sample_idx": np.array(
            [int(row.get("sample_idx", i)) for i, row in enumerate(index_rows)],
            dtype=np.int64,
        ),
        "window_name": [row.get("window_name", "") for row in index_rows],
        "lat": np.array(
            [_row_float(row.get("lat", "")) for row in index_rows], dtype=np.float64
        ),
        "lon": np.array(
            [_row_float(row.get("lon", "")) for row in index_rows], dtype=np.float64
        ),
        "parent_label": parent_labels.astype(np.int32),
        "novelty_score": np.asarray(scores, dtype=np.float32),
        "threshold": np.asarray(thresholds, dtype=np.float32),
        "accepted": np.asarray(accepted, dtype=bool),
    }
    if k_used is not None:
        data["k_used"] = np.asarray(k_used, dtype=np.int32)
    pd.DataFrame(data).to_parquet(path, index=False)
    print(f"[save] Scores parquet -> {path}")


def save_summary_json(path: str, payload: dict[str, object]) -> None:
    """Write a small JSON summary for inspection."""
    with open(path, "w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
    print(f"[save] Summary JSON -> {path}")


def fit_reference(args: argparse.Namespace) -> None:
    """Fit frozen reference artifacts used for candidate scoring."""
    output_dir = args.output_dir or os.path.join(args.input_dir, "_scores")
    os.makedirs(output_dir, exist_ok=True)

    embeddings, kept_indices = load_embeddings(
        args.input_dir, args.subsample, args.seed
    )
    index_rows = ensure_index_rows(args.input_dir, embeddings.shape[0], kept_indices)

    pca_data, pca_state = fit_global_pca(embeddings, args.pca_dim, args.seed)
    del embeddings

    parent_labels, normalized_centers, parent_centroids = fit_spherical_kmeans(
        pca_data,
        n_clusters=args.k,
        seed=args.seed,
    )
    residuals = compute_residuals(pca_data, parent_labels, parent_centroids)
    self_scores, parent_counts, parent_k_used = compute_parent_knn_self_scores(
        residuals,
        parent_labels,
        n_clusters=args.k,
        knn_k=args.knn_k,
        metric=args.distance_metric,
    )
    thresholds = compute_thresholds(
        self_scores,
        parent_labels,
        n_clusters=args.k,
        percentile=args.percentile,
    )

    model_state: dict[str, np.ndarray | float | int] = {
        **pca_state,
        "normalized_centers": normalized_centers.astype(np.float32),
        "parent_centroids": parent_centroids.astype(np.float32),
        "parent_counts": parent_counts.astype(np.int32),
        "parent_k_used": parent_k_used.astype(np.int32),
        "thresholds": thresholds.astype(np.float32),
        "n_clusters": np.int32(args.k),
        "knn_k": np.int32(args.knn_k),
        "percentile": np.float32(args.percentile),
        "seed": np.int32(args.seed),
        "subsample": np.int32(args.subsample),
        "distance_metric": np.array(args.distance_metric),
        "reference_size": np.int32(pca_data.shape[0]),
    }
    save_reference_artifacts(
        output_dir, model_state, residuals, parent_labels, self_scores, index_rows
    )

    summary = {
        "input_dir": args.input_dir,
        "output_dir": output_dir,
        "reference_size": int(pca_data.shape[0]),
        "pca_dim": int(args.pca_dim),
        "k": int(args.k),
        "knn_k": int(args.knn_k),
        "percentile": float(args.percentile),
        "distance_metric": args.distance_metric,
        "parent_counts": parent_counts.astype(int).tolist(),
        "parent_k_used": parent_k_used.astype(int).tolist(),
        "thresholds": thresholds.astype(float).tolist(),
    }
    save_summary_json(os.path.join(output_dir, "reference_summary.json"), summary)


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
    """Build the CLI parser."""
    parser = argparse.ArgumentParser(
        description="Fit frozen reference artifacts and score candidate embeddings.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    fit = sub.add_parser(
        "fit-reference",
        help="Fit frozen PCA/k-means artifacts on a reference embedding directory.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    fit.add_argument(
        "--input-dir", required=True, help="Directory containing shard_*.npz files"
    )
    fit.add_argument(
        "--output-dir",
        default=None,
        help="Where to write frozen artifacts. Defaults to {input_dir}/_scores",
    )
    fit.add_argument(
        "--subsample", type=int, default=0, help="Optional random subsample size"
    )
    fit.add_argument(
        "--pca-dim", type=int, default=128, help="Global PCA dimensionality"
    )
    fit.add_argument("--k", type=int, default=15, help="Number of parent clusters")
    fit.add_argument(
        "--knn-k", type=int, default=5, help="k for local residual kNN novelty"
    )
    fit.add_argument(
        "--percentile",
        type=float,
        default=99.0,
        help="Parent-specific acceptance percentile computed from reference self-scores",
    )
    fit.add_argument(
        "--distance-metric",
        default="euclidean",
        choices=["euclidean", "cosine"],
        help="Metric used for residual-space kNN scoring",
    )
    fit.add_argument("--seed", type=int, default=42, help="Random seed")
    fit.set_defaults(func=fit_reference)

    score = sub.add_parser(
        "novelty-score",
        help="Score novelty for a candidate embedding directory against frozen reference artifacts.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    score.add_argument(
        "--input-dir", required=True, help="Candidate directory with shard_*.npz"
    )
    score.add_argument(
        "--reference-dir",
        required=True,
        help="Directory containing reference_model.npz and reference_residuals.npz",
    )
    score.add_argument(
        "--output-dir",
        default=None,
        help="Where to write candidate scores. Defaults to {input_dir}/_scores",
    )
    score.add_argument(
        "--seed", type=int, default=42, help="Unused except for CLI parity"
    )
    score.set_defaults(func=score_candidates)

    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Run the reference fitting or novelty-scoring subcommand."""
    args = parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main(sys.argv[1:])
