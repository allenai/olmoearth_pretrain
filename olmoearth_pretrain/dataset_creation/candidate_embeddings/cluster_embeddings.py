"""Cluster high-dimensional embeddings via PCA + HDBSCAN or k-means.

Pipeline:
  1. Load sharded .npz embeddings from a directory
  2. PCA reduction to a clustering-friendly dimensionality (CPU, sklearn)
  3. Cluster via HDBSCAN or k-means
  4. Save labels as .npy (compatible with visualize_embeddings.py --labels)
  5. Save cluster bundle .npz (for visualize_embeddings.py --layout cluster)

The cluster bundle contains PCA-reduced data, centroids, and labels —
everything needed by the cluster-distance layout in visualize_embeddings.py.

Intermediate PCA results are cached to disk. Cluster labels are saved
alongside them so you can feed them into the visualization script.
"""

from __future__ import annotations

import argparse
import glob
import os
import time

import numpy as np

# ---------------------------------------------------------------------------
# Stage 1 -- Load  (same logic as visualize_embeddings.py)
# ---------------------------------------------------------------------------


def load_embeddings(input_dir: str, subsample: int = 0, seed: int = 42) -> np.ndarray:
    """Load all shard_*.npz files and concatenate into (N, D) float32."""
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

    if subsample > 0 and subsample < embeddings.shape[0]:
        rng = np.random.RandomState(seed)
        indices = rng.choice(embeddings.shape[0], size=subsample, replace=False)
        indices.sort()
        embeddings = embeddings[indices]
        print(f"[load] Subsampled to {embeddings.shape[0]:,} embeddings")

    return embeddings


# ---------------------------------------------------------------------------
# Stage 2 -- PCA
# ---------------------------------------------------------------------------


def run_pca(
    embeddings: np.ndarray,
    n_components: int,
    output_path: str,
    seed: int = 42,
) -> np.ndarray:
    """PCA reduction on CPU via sklearn. Caches result to output_path."""
    if os.path.exists(output_path):
        cached = np.load(output_path)
        if cached.shape[0] == embeddings.shape[0]:
            print(f"[pca] Loading cached PCA result from {output_path}")
            return cached
        print(
            f"[pca] Cached file has {cached.shape[0]:,} rows but input has "
            f"{embeddings.shape[0]:,} -- recomputing"
        )
        del cached

    from sklearn.decomposition import PCA

    print(f"[pca] Reducing {embeddings.shape} -> (N, {n_components}) ...")
    t0 = time.time()
    pca = PCA(n_components=n_components, random_state=seed)
    reduced = pca.fit_transform(embeddings)
    elapsed = time.time() - t0
    print(
        f"[pca] Done in {elapsed:.1f}s  "
        f"(explained variance: {pca.explained_variance_ratio_.sum():.2%})"
    )

    np.save(output_path, reduced)
    print(f"[pca] Saved to {output_path}")
    return reduced


# ---------------------------------------------------------------------------
# Stage 3 -- Clustering
# ---------------------------------------------------------------------------


def _get_hdbscan_class() -> tuple[type, bool]:
    """Return (HDBSCAN_class, is_gpu). Prefer cuML GPU, fall back to CPU."""
    try:
        from cuml.cluster import HDBSCAN as cuHDBSCAN

        return cuHDBSCAN, True
    except ImportError:
        pass
    import hdbscan

    return hdbscan.HDBSCAN, False


def run_hdbscan(
    data: np.ndarray,
    min_cluster_size: int = 50,
    min_samples: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """HDBSCAN clustering.

    Returns:
        labels: (N,) int labels (-1 = noise).
        centroids: (K, D) float32 cluster centroids (mean of each cluster).
    """
    HDBSCAN, use_gpu = _get_hdbscan_class()

    if min_samples is None:
        min_samples = min_cluster_size

    backend = "GPU (cuML)" if use_gpu else "CPU"
    print(
        f"[hdbscan] Clustering {data.shape[0]:,} points in {data.shape[1]}D  "
        f"(min_cluster_size={min_cluster_size}, min_samples={min_samples}, "
        f"backend={backend}) ..."
    )
    t0 = time.time()

    if use_gpu:
        clusterer = HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
        )
    else:
        clusterer = HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            core_dist_n_jobs=-1,
        )

    labels = clusterer.fit_predict(data)
    if use_gpu:
        labels = labels.get() if hasattr(labels, "get") else np.asarray(labels)
    elapsed = time.time() - t0

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = (labels == -1).sum()
    print(
        f"[hdbscan] Done in {elapsed:.1f}s  "
        f"({n_clusters} clusters, {n_noise:,} noise points "
        f"[{n_noise / len(labels):.1%}])"
    )

    centroids = np.zeros((n_clusters, data.shape[1]), dtype=np.float32)
    for k in range(n_clusters):
        centroids[k] = data[labels == k].mean(axis=0)

    return labels, centroids


def run_kmeans(
    data: np.ndarray,
    n_clusters: int = 20,
    seed: int = 42,
    spherical: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """k-means clustering via sklearn.

    When spherical=True, L2-normalizes the data before clustering so that
    Euclidean k-means is equivalent to minimizing cosine distance (spherical
    k-means). This is more appropriate for neural network embeddings that
    live on or near a hypersphere.

    Returns:
        labels: (N,) int labels.
        centroids: (K, D) float32 cluster centroids.
    """
    from sklearn.cluster import MiniBatchKMeans

    variant = "spherical k-means" if spherical else "k-means"
    print(
        f"[kmeans] Clustering {data.shape[0]:,} points in {data.shape[1]}D  "
        f"(k={n_clusters}, {variant}) ..."
    )

    orig_data = data
    if spherical:
        norms = np.linalg.norm(data, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-10)
        data = data / norms

    t0 = time.time()
    km = MiniBatchKMeans(
        n_clusters=n_clusters,
        random_state=seed,
        batch_size=max(1024, n_clusters * 10),
    )
    labels = km.fit_predict(data)
    elapsed = time.time() - t0
    print(f"[kmeans] Done in {elapsed:.1f}s  (inertia={km.inertia_:.2e})")

    # Return centroids in the original PCA coordinate system so downstream
    # residual computations and cluster-layout visualization stay aligned.
    centroids = np.zeros((n_clusters, orig_data.shape[1]), dtype=np.float32)
    for k in range(n_clusters):
        centroids[k] = orig_data[labels == k].mean(axis=0)

    return labels, centroids


def run_kmeans_residual_hdbscan(
    data: np.ndarray,
    n_clusters: int = 20,
    seed: int = 42,
    spherical: bool = False,
    residual_pca_dim: int = 30,
    min_cluster_size: int = 50,
    min_samples: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Run k-means, then HDBSCAN on residuals within each k-means cluster.

    The parent k-means centroids are computed in the original PCA space.
    Residual HDBSCAN labels are flattened into a single global label space
    so they can be used directly as datamapplot labels.
    """
    from sklearn.decomposition import PCA

    parent_labels, parent_centroids = run_kmeans(
        data,
        n_clusters=n_clusters,
        seed=seed,
        spherical=spherical,
    )

    print(
        "[kmeans-hdbscan] Running nested HDBSCAN on residuals within each "
        f"k-means cluster (residual_pca_dim={residual_pca_dim}) ..."
    )
    t0 = time.time()

    nested_labels_int = np.full(data.shape[0], -1, dtype=np.int32)
    nested_labels_str = np.full(data.shape[0], "Unlabelled", dtype=object)
    next_nested_label = 0
    total_noise = 0
    total_subclusters = 0
    for parent_idx in range(parent_centroids.shape[0]):
        mask = parent_labels == parent_idx
        n_points = int(mask.sum())
        if n_points == 0:
            continue

        residuals = data[mask] - parent_centroids[parent_idx]
        local_dim = min(residual_pca_dim, residuals.shape[1], max(n_points - 1, 1))

        if local_dim < residuals.shape[1]:
            pca = PCA(n_components=local_dim, random_state=seed)
            local_data = pca.fit_transform(residuals).astype(np.float32)
        else:
            local_data = residuals.astype(np.float32, copy=False)

        print(
            f"[kmeans-hdbscan] Parent cluster {parent_idx}: {n_points:,} "
            f"points -> HDBSCAN in {local_data.shape[1]}D"
        )
        local_labels, _ = run_hdbscan(
            local_data,
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
        )

        unique_local_labels = sorted(
            int(lbl) for lbl in np.unique(local_labels) if lbl != -1
        )
        total_subclusters += len(unique_local_labels)
        total_noise += int((local_labels == -1).sum())

        cluster_indices = np.flatnonzero(mask)
        for local_label in unique_local_labels:
            global_mask = local_labels == local_label
            nested_labels_int[cluster_indices[global_mask]] = next_nested_label
            nested_labels_str[cluster_indices[global_mask]] = (
                f"K{parent_idx}_H{local_label}"
            )
            next_nested_label += 1

    elapsed = time.time() - t0
    print(
        f"[kmeans-hdbscan] Done in {elapsed:.1f}s  "
        f"({total_subclusters} subclusters, {total_noise:,} noise points "
        f"[{total_noise / len(nested_labels_int):.1%}])"
    )
    return parent_labels, parent_centroids, nested_labels_int, nested_labels_str


def labels_to_strings(labels: np.ndarray, method: str) -> np.ndarray:
    """Convert integer labels to string labels for datamapplot compatibility.

    Noise points (label -1 from HDBSCAN) are mapped to 'Unlabelled' which
    datamapplot treats as its default noise category.
    """
    str_labels = np.array([f"{method}_{label_id}" for label_id in labels], dtype=object)
    if -1 in labels:
        str_labels[labels == -1] = "Unlabelled"
    return str_labels


def save_labels(
    output_dir: str,
    stem: str,
    int_labels: np.ndarray,
    str_labels: np.ndarray,
) -> tuple[str, str]:
    """Save integer and string labels for downstream visualization."""
    int_path = os.path.join(output_dir, f"labels_{stem}_int.npy")
    str_path = os.path.join(output_dir, f"labels_{stem}.npy")

    np.save(int_path, int_labels)
    np.save(str_path, str_labels)
    print(f"[save] Integer labels -> {int_path}")
    print(f"[save] String labels  -> {str_path}")
    return int_path, str_path


def save_cluster_bundle(
    output_dir: str,
    stem: str,
    pca_data: np.ndarray,
    centroids: np.ndarray,
    int_labels: np.ndarray,
    str_labels: np.ndarray,
    sample_indices_kwargs: dict,
) -> str:
    """Save a cluster bundle for cluster-layout visualization."""
    bundle_path = os.path.join(output_dir, f"cluster_bundle_{stem}.npz")
    np.savez(
        bundle_path,
        pca_data=pca_data,
        centroids=centroids,
        labels_int=int_labels,
        labels_str=str_labels,
        **sample_indices_kwargs,
    )
    print(f"[save] Cluster bundle -> {bundle_path}")
    return bundle_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments for the clustering workflow."""
    p = argparse.ArgumentParser(
        description="Cluster embeddings: load -> PCA -> HDBSCAN, k-means, "
        "or nested k-means+HDBSCAN -> labels.npy",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--input-dir", required=True, help="Directory containing shard_*.npz files"
    )
    p.add_argument(
        "--output-dir",
        default=None,
        help="Where to write cached arrays and labels. "
        "Defaults to {input_dir}/_cluster",
    )
    p.add_argument(
        "--subsample",
        type=int,
        default=0,
        help="Randomly subsample to this many points (0 = all)",
    )
    p.add_argument(
        "--pca-dim",
        type=int,
        default=30,
        help="PCA dimensionality for clustering (20-50 recommended)",
    )
    p.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility"
    )

    sub = p.add_subparsers(dest="method", required=True, help="Clustering method")

    hdb = sub.add_parser(
        "hdbscan",
        help="HDBSCAN density-based clustering",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    hdb.add_argument(
        "--min-cluster-size", type=int, default=50, help="Minimum cluster size"
    )
    hdb.add_argument(
        "--min-samples",
        type=int,
        default=None,
        help="Min samples for core points (defaults to min-cluster-size)",
    )

    km = sub.add_parser(
        "kmeans",
        help="k-means clustering",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    km.add_argument("--k", type=int, default=20, help="Number of clusters")
    km.add_argument(
        "--spherical",
        action="store_true",
        help="L2-normalize before clustering (spherical k-means, "
        "i.e. cosine distance). Recommended for embeddings.",
    )

    km_hdb = sub.add_parser(
        "kmeans-hdbscan",
        help="k-means followed by per-cluster HDBSCAN on residuals",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    km_hdb.add_argument(
        "--k", type=int, default=20, help="Number of parent k-means clusters"
    )
    km_hdb.add_argument(
        "--spherical",
        action="store_true",
        help="L2-normalize before parent k-means clustering "
        "(spherical k-means, i.e. cosine distance). "
        "Recommended for embeddings.",
    )
    km_hdb.add_argument(
        "--residual-pca-dim",
        type=int,
        default=30,
        help="Per-cluster PCA dimensionality applied to residuals before HDBSCAN",
    )
    km_hdb.add_argument(
        "--min-cluster-size",
        type=int,
        default=50,
        help="Minimum HDBSCAN subcluster size",
    )
    km_hdb.add_argument(
        "--min-samples",
        type=int,
        default=None,
        help="Min samples for HDBSCAN core points (defaults to min-cluster-size)",
    )

    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Run clustering and write label arrays plus the visualization bundle."""
    args = parse_args(argv)

    output_dir = args.output_dir or os.path.join(args.input_dir, "_cluster")
    os.makedirs(output_dir, exist_ok=True)

    pca_path = os.path.join(output_dir, f"pca_{args.pca_dim}.npy")

    # -- Load & reduce ------------------------------------------------------
    embeddings = load_embeddings(args.input_dir, args.subsample, args.seed)
    pca_data = run_pca(embeddings, args.pca_dim, pca_path, args.seed)
    del embeddings

    # -- Cluster ------------------------------------------------------------
    nested_labels_int = None
    nested_labels_str = None
    nested_label_stem = None

    if args.method == "hdbscan":
        int_labels, centroids = run_hdbscan(
            pca_data,
            min_cluster_size=args.min_cluster_size,
            min_samples=args.min_samples,
        )
    elif args.method == "kmeans":
        int_labels, centroids = run_kmeans(
            pca_data,
            n_clusters=args.k,
            seed=args.seed,
            spherical=args.spherical,
        )
    elif args.method == "kmeans-hdbscan":
        int_labels, centroids, nested_labels_int, nested_labels_str = (
            run_kmeans_residual_hdbscan(
                pca_data,
                n_clusters=args.k,
                seed=args.seed,
                spherical=args.spherical,
                residual_pca_dim=args.residual_pca_dim,
                min_cluster_size=args.min_cluster_size,
                min_samples=args.min_samples,
            )
        )
        nested_label_stem = "kmeans_residual_hdbscan"
    else:
        raise ValueError(f"Unknown method: {args.method}")

    # -- Save labels --------------------------------------------------------
    label_stem = "kmeans" if args.method == "kmeans-hdbscan" else args.method
    str_labels = labels_to_strings(int_labels, label_stem)

    # -- Load sample_indices from index.csv so downstream scripts can map
    #    embedding positions back to actual H5 file numbers. ---------------
    index_csv_path = os.path.join(args.input_dir, "index.csv")
    sample_indices_kwargs: dict = {}
    if os.path.exists(index_csv_path):
        import csv

        with open(index_csv_path, newline="") as f:
            reader = csv.DictReader(f)
            sample_indices_list = [int(row["sample_idx"]) for row in reader]
        sample_indices_arr = np.array(sample_indices_list, dtype=np.int64)

        if sample_indices_arr.shape[0] == pca_data.shape[0]:
            sample_indices_kwargs["sample_indices"] = sample_indices_arr
            print(f"[save] Including sample_indices from {index_csv_path}")
        else:
            print(
                f"[warn] index.csv has {sample_indices_arr.shape[0]} rows "
                f"but embeddings have {pca_data.shape[0]} -- skipping "
                f"sample_indices in bundle"
            )
    else:
        print(
            f"[warn] No index.csv found at {index_csv_path} -- bundle "
            f"will not include sample_indices"
        )

    int_path, str_path = save_labels(output_dir, label_stem, int_labels, str_labels)
    bundle_path = save_cluster_bundle(
        output_dir,
        label_stem,
        pca_data,
        centroids,
        int_labels,
        str_labels,
        sample_indices_kwargs,
    )

    nested_str_path = None
    if nested_labels_int is not None and nested_labels_str is not None:
        assert nested_label_stem is not None
        _, nested_str_path = save_labels(
            output_dir,
            nested_label_stem,
            nested_labels_int,
            nested_labels_str,
        )

    del pca_data

    print("\nDone. Use with visualization:")
    print(
        f"  python visualize_embeddings.py --input-dir {args.input_dir} "
        f"--skip-reduction --labels {str_path}"
    )
    print(
        f"  python visualize_embeddings.py --layout cluster "
        f"--cluster-bundle {bundle_path}"
    )
    if nested_str_path is not None:
        print(
            "  python visualize_embeddings.py --layout cluster "
            f"--cluster-bundle {bundle_path} --labels {nested_str_path}"
        )


if __name__ == "__main__":
    main()
