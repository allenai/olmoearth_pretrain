"""Visualize high-dimensional embeddings via interactive datamapplot.

Two layout modes are available (--layout):

  tsne (default):
    1. Load sharded .npz embeddings
    2. PCA reduction (CPU, sklearn)
    3. t-SNE reduction (GPU, cuML)
    4. Interactive HTML visualization (datamapplot / DeckGL)

  cluster (requires output from cluster_embeddings.py):
    1. Load cluster bundle (.npz) produced by cluster_embeddings.py
    2. MDS on inter-centroid distances -> centroid 2D positions
    3. Per-cluster PCA -> point offsets around centroids
    4. Interactive HTML visualization (datamapplot / DeckGL)

    The cluster layout places centroids so that their 2D distances
    reflect high-dimensional distances (cosine or euclidean), and
    arranges points around their centroid preserving within-cluster
    structure.  Unlike t-SNE, global distances are meaningful.

Intermediate results are cached to disk so you can re-run visualization
without recomputing the expensive reduction steps.
"""

from __future__ import annotations

import argparse
import colorsys
import glob
import importlib
import os
import re
import sys
import time
from typing import TYPE_CHECKING, TypedDict

import numpy as np

if TYPE_CHECKING:
    import pandas as pd


NESTED_HDBSCAN_LABEL_RE = re.compile(r"^K(?P<parent>\d+)_H(?P<child>\d+)$")


class ClusterLayoutModel(TypedDict):
    """Reusable geometry needed to project points into a frozen cluster layout."""

    coords: np.ndarray
    centroid_2d: np.ndarray
    layout_labels: np.ndarray
    scale: float
    local_pca_means: list[np.ndarray]
    local_pca_components: list[np.ndarray]


def _rgb_to_hex(rgb: tuple[float, float, float]) -> str:
    return "#{:02x}{:02x}{:02x}".format(
        *(max(0, min(255, round(channel * 255))) for channel in rgb)
    )


def _build_nested_label_color_map(
    labels: np.ndarray,
    darkmode: bool,
) -> dict[str, str] | None:
    """Return a high-contrast palette for nested labels like K0_H2."""
    parsed_labels: list[tuple[str, int, int]] = []
    for label in sorted(set(labels) - {"Unlabelled"}):
        match = NESTED_HDBSCAN_LABEL_RE.fullmatch(str(label))
        if match is None:
            return None
        parsed_labels.append(
            (str(label), int(match.group("parent")), int(match.group("child")))
        )

    by_parent: dict[int, list[tuple[str, int]]] = {}
    for label, parent_idx, child_idx in parsed_labels:
        by_parent.setdefault(parent_idx, []).append((label, child_idx))

    label_color_map: dict[str, str] = {}
    saturation = 0.95
    value = 1.0 if darkmode else 0.9

    for parent_idx, entries in by_parent.items():
        entries.sort(key=lambda item: item[1])
        n_entries = len(entries)
        parent_offset = (parent_idx * 0.61803398875) % 1.0

        for color_idx, (label, _) in enumerate(entries):
            hue = (parent_offset + (color_idx / max(n_entries, 1))) % 1.0
            label_color_map[label] = _rgb_to_hex(
                colorsys.hsv_to_rgb(hue, saturation, value)
            )

    label_color_map["Unlabelled"] = "#66666666" if darkmode else "#99999966"
    return label_color_map


def _build_label_color_map(
    labels: np.ndarray,
    darkmode: bool,
    monochrome: bool = False,
) -> dict[str, str] | None:
    if monochrome:
        default_gray = "#9a9a9a88" if darkmode else "#6f6f6f88"
        color_map = {str(label): default_gray for label in sorted(set(labels))}
        color_map["Unlabelled"] = "#66666655" if darkmode else "#99999955"
        return color_map

    nested_color_map = _build_nested_label_color_map(labels, darkmode)
    if nested_color_map is not None:
        return nested_color_map

    try:
        glasbey = importlib.import_module("glasbey")

        n_unique = len(set(labels)) - (1 if "Unlabelled" in set(labels) else 0)
        palette = glasbey.create_palette(
            palette_size=max(n_unique, 2),
            lightness_bounds=(40, 80),
            chroma_bounds=(50, 100),
        )
        label_color_map = dict(zip(sorted(set(labels) - {"Unlabelled"}), palette))
        label_color_map["Unlabelled"] = "#55555544" if darkmode else "#bbbbbb44"
        return label_color_map
    except ImportError:
        return None


# ---------------------------------------------------------------------------
# Stage 1 -- Load
# ---------------------------------------------------------------------------


def load_embeddings(
    input_dir: str,
    subsample: int = 0,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Load all shard_*.npz files and concatenate into (N, D) float32.

    Returns (embeddings, subsample_indices) where subsample_indices is None
    when no subsampling is applied, or an int64 array of kept indices.
    """
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

    kept_indices = None
    if subsample > 0 and subsample < embeddings.shape[0]:
        rng = np.random.RandomState(seed)
        kept_indices = rng.choice(embeddings.shape[0], size=subsample, replace=False)
        kept_indices.sort()
        embeddings = embeddings[kept_indices]
        print(f"[load] Subsampled to {embeddings.shape[0]:,} embeddings")

    return embeddings, kept_indices


def _load_sample_indices_from_csv(
    input_dir: str,
    n_points: int,
    subsample_positions: np.ndarray | None = None,
) -> np.ndarray | None:
    """Read ``sample_idx`` from *input_dir*/index.csv.

    The extraction step writes an index.csv whose ``sample_idx`` column
    maps each embedding position to the actual H5 file number.  When the
    dataset filtered out H5 files, the sample_idx sequence has gaps, so
    positional indices (0 .. N-1) do NOT equal H5 file numbers.

    If *subsample_positions* is given (an array of kept positions from
    ``load_embeddings``), only those rows are returned.
    """
    import csv

    csv_path = os.path.join(input_dir, "index.csv")
    if not os.path.exists(csv_path):
        return None

    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        all_indices = np.array(
            [int(row["sample_idx"]) for row in reader],
            dtype=np.int64,
        )

    if subsample_positions is not None:
        if int(subsample_positions.max()) >= all_indices.shape[0]:
            print(
                "[warn] subsample positions exceed index.csv rows "
                "-- skipping index.csv sample_indices",
                file=sys.stderr,
            )
            return None
        all_indices = all_indices[subsample_positions]

    if all_indices.shape[0] != n_points:
        print(
            f"[warn] index.csv sample_idx count ({all_indices.shape[0]}) "
            f"!= n_points ({n_points}) -- skipping",
            file=sys.stderr,
        )
        return None

    is_sequential = np.array_equal(all_indices, np.arange(n_points))
    if is_sequential:
        print("[load] index.csv sample_idx is sequential -- no remapping needed")
        return None

    n_gaps = int(all_indices[-1]) - int(all_indices[0]) + 1 - all_indices.shape[0]
    print(
        f"[load] Loaded {all_indices.shape[0]:,} sample_indices from "
        f"{csv_path} ({n_gaps} gaps detected)"
    )
    return all_indices


# ---------------------------------------------------------------------------
# Stage 1b -- Load metadata parquet
# ---------------------------------------------------------------------------

_SAMPLE_IDX_ALIASES = ("sample_idx", "sample_index")


def load_metadata(
    metadata_path: str,
    n_points: int,
    sample_indices: np.ndarray | None = None,
    columns: list[str] | None = None,
) -> pd.DataFrame | None:
    """Load a parquet file and align rows with the visualization points.

    The parquet should have a ``sample_index`` column (or ``sample_idx``)
    whose integer values match the ``sample_idx`` column in the
    extraction index.csv (0-based, one per embedding).  Float values are
    cast to int automatically.

    When subsampling was applied during embedding loading, pass the
    *kept* indices so that the correct metadata rows are selected.

    Returns a DataFrame with *n_points* rows (same order as coords),
    ready to be passed as ``extra_point_data`` to datamapplot.
    """
    import pandas as pd

    print(f"[meta] Loading metadata from {metadata_path} ...")

    # Read only the columns we need (+ index column) for efficiency.
    read_cols = None
    if columns is not None:
        import pyarrow.parquet as pq

        all_parquet_cols = pq.read_schema(metadata_path).names
        idx_col_for_read = None
        for alias in _SAMPLE_IDX_ALIASES:
            if alias in all_parquet_cols:
                idx_col_for_read = alias
                break
        read_cols = list(
            dict.fromkeys(([idx_col_for_read] if idx_col_for_read else []) + columns)
        )
        unknown = set(columns) - set(all_parquet_cols)
        if unknown:
            print(
                f"[meta] Warning: requested columns not in parquet: {sorted(unknown)}",
                file=sys.stderr,
            )
        read_cols = [c for c in read_cols if c in all_parquet_cols]

    meta = pd.read_parquet(metadata_path, columns=read_cols)
    print(f"[meta] Parquet has {len(meta):,} rows, {len(meta.columns)} columns")

    idx_col = None
    for alias in _SAMPLE_IDX_ALIASES:
        if alias in meta.columns:
            idx_col = alias
            break

    if idx_col is None:
        print(
            "[meta] Warning: no sample index column found "
            f"(tried {_SAMPLE_IDX_ALIASES}) -- assuming rows "
            "are already in point order.",
            file=sys.stderr,
        )
        if len(meta) != n_points:
            print(
                f"[meta] Row count mismatch ({len(meta):,} vs "
                f"{n_points:,} points). Skipping metadata.",
                file=sys.stderr,
            )
            return None
        return meta.reset_index(drop=True)

    print(f"[meta] Using '{idx_col}' as join key")
    meta[idx_col] = meta[idx_col].astype(int)
    meta = meta.set_index(idx_col).sort_index()

    if sample_indices is None:
        sample_indices = np.arange(n_points)

    missing = set(sample_indices.tolist()) - set(meta.index.tolist())
    if missing:
        n_miss = len(missing)
        print(
            f"[meta] Warning: {n_miss:,} sample index values not found "
            f"in metadata -- those rows will have NaN.",
            file=sys.stderr,
        )

    aligned = meta.reindex(sample_indices).reset_index(drop=True)
    print(
        f"[meta] Aligned {len(aligned):,} metadata rows with points "
        f"({len(aligned.columns)} columns: {list(aligned.columns[:8])}"
        f"{'...' if len(aligned.columns) > 8 else ''})"
    )
    return aligned


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
# Stage 3 -- t-SNE (GPU)
# ---------------------------------------------------------------------------


def run_tsne(
    data: np.ndarray,
    n_components: int,
    output_path: str,
    perplexity: float = 30.0,
    seed: int = 42,
) -> np.ndarray:
    """t-SNE reduction on GPU via cuML. Caches result to output_path."""
    if os.path.exists(output_path):
        cached = np.load(output_path)
        if cached.shape[0] == data.shape[0]:
            print(f"[tsne] Loading cached t-SNE result from {output_path}")
            return cached
        print(
            f"[tsne] Cached file has {cached.shape[0]:,} rows but input has "
            f"{data.shape[0]:,} -- recomputing"
        )
        del cached

    from cuml.manifold import TSNE

    print(
        f"[tsne] Reducing {data.shape} -> (N, {n_components}) "
        f"(perplexity={perplexity}) ..."
    )
    t0 = time.time()
    n = data.shape[0]
    tsne = TSNE(
        n_components=n_components,
        perplexity=perplexity,
        n_neighbors=max(90, int(3 * perplexity)),
        learning_rate=max(n / 12.0, 200.0),
        learning_rate_method="none",
        random_state=seed,
        method="fft",
    )
    coords = tsne.fit_transform(data)
    if hasattr(coords, "values"):
        coords = coords.values
    coords = np.asarray(coords, dtype=np.float32)
    elapsed = time.time() - t0
    print(f"[tsne] Done in {elapsed:.1f}s")

    np.save(output_path, coords)
    print(f"[tsne] Saved to {output_path}")
    return coords


# ---------------------------------------------------------------------------
# Stage 3b -- Cluster-distance layout (alternative to t-SNE)
# ---------------------------------------------------------------------------


def compute_cluster_layout(
    pca_data: np.ndarray,
    centroids: np.ndarray,
    labels_int: np.ndarray,
    metric: str = "cosine",
    separation: float = 0.6,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """2D layout from pre-computed clustering artifacts.

    Centroids are positioned via MDS so that inter-centroid 2D distances
    reflect high-D distances.  Individual points are scattered around
    their centroid using PCA of within-cluster residuals, then globally
    scaled so clusters do not overlap.

    Noise points (label -1, e.g. from HDBSCAN) are assigned to their
    nearest centroid for layout purposes.

    Args:
        pca_data: (N, D) PCA-reduced embeddings (from cluster_embeddings.py).
        centroids: (K, D) cluster centroids in the same PCA space.
        labels_int: (N,) integer cluster labels (may contain -1 for noise).
        metric: Distance metric for inter-centroid MDS.
        separation: Fraction of the inter-centroid gap used for the
            intra-cluster radius (0-1).  Higher = clusters fill more space.
        seed: Random seed.

    Returns:
        Tuple of (coords, centroid_2d) where coords is (N, 2) float32
        layout coordinates and centroid_2d is (K, 2) float32 centroid
        positions in the same 2D space.
    """
    layout_model = build_cluster_layout_model(
        pca_data=pca_data,
        centroids=centroids,
        labels_int=labels_int,
        metric=metric,
        separation=separation,
        seed=seed,
    )
    return layout_model["coords"], layout_model["centroid_2d"]


def build_cluster_layout_model(
    pca_data: np.ndarray,
    centroids: np.ndarray,
    labels_int: np.ndarray,
    metric: str = "cosine",
    separation: float = 0.6,
    seed: int = 42,
) -> ClusterLayoutModel:
    """Fit the reusable geometry for the cluster layout.

    In addition to the reference coordinates, this stores the per-parent local
    PCA projectors and global scaling factor so external embeddings can be
    projected into the frozen layout later.
    """
    from scipy.spatial.distance import cdist, pdist, squareform
    from sklearn.decomposition import PCA
    from sklearn.manifold import MDS

    n_clusters = centroids.shape[0]
    print(
        f"[cluster-layout] Computing layout for {pca_data.shape[0]:,} points "
        f"({n_clusters} clusters, metric={metric}) ..."
    )
    t0 = time.time()

    layout_labels = labels_int.copy()
    noise_mask = layout_labels == -1
    if noise_mask.any():
        dists_to_centroids = cdist(pca_data[noise_mask], centroids)
        layout_labels[noise_mask] = dists_to_centroids.argmin(axis=1)
        print(
            f"[cluster-layout] Assigned {noise_mask.sum():,} noise points "
            f"to nearest centroids for layout"
        )

    if n_clusters > 1:
        dist_matrix = squareform(pdist(centroids, metric=metric))
        mds = MDS(
            n_components=2,
            dissimilarity="precomputed",
            random_state=seed,
            normalized_stress="auto",
        )
        centroid_2d = mds.fit_transform(dist_matrix).astype(np.float32)
    else:
        centroid_2d = np.zeros((1, 2), dtype=np.float32)

    local_offsets: dict[int, np.ndarray] = {}
    local_pca_means: list[np.ndarray] = []
    local_pca_components: list[np.ndarray] = []
    max_local_radius = 0.0

    for k in range(n_clusters):
        mask = layout_labels == k
        n_k = int(mask.sum())
        residual_dim = pca_data.shape[1]
        if n_k == 0:
            local_pca_means.append(np.zeros(residual_dim, dtype=np.float32))
            local_pca_components.append(np.zeros((0, residual_dim), dtype=np.float32))
            continue

        residuals = pca_data[mask] - centroids[k]

        if n_k <= 2:
            local_2d = np.zeros((n_k, 2), dtype=np.float32)
            local_mean = np.zeros(residuals.shape[1], dtype=np.float32)
            components = np.zeros((0, residuals.shape[1]), dtype=np.float32)
        else:
            n_comp = min(2, residuals.shape[1], n_k - 1)
            pca = PCA(n_components=n_comp, random_state=seed)
            local_2d = pca.fit_transform(residuals).astype(np.float32)
            local_mean = pca.mean_.astype(np.float32)
            components = pca.components_.astype(np.float32)
            if local_2d.shape[1] < 2:
                local_2d = np.column_stack([local_2d, np.zeros(n_k, dtype=np.float32)])

        local_offsets[k] = local_2d
        local_pca_means.append(local_mean)
        local_pca_components.append(components)
        r_max = float(np.linalg.norm(local_2d, axis=1).max())
        max_local_radius = max(max_local_radius, r_max)

    if n_clusters > 1:
        c2d_dists = squareform(pdist(centroid_2d))
        np.fill_diagonal(c2d_dists, np.inf)
        min_gap = float(c2d_dists.min())
    else:
        min_gap = 1.0

    target_radius = separation * min_gap / 2.0
    scale = target_radius / max(max_local_radius, 1e-10)

    coords = np.zeros((pca_data.shape[0], 2), dtype=np.float32)
    for k, offsets in local_offsets.items():
        mask = layout_labels == k
        coords[mask] = centroid_2d[k] + offsets * scale

    elapsed = time.time() - t0
    print(
        f"[cluster-layout] Done in {elapsed:.1f}s  "
        f"(scale={scale:.4f}, min_gap_2d={min_gap:.4f})"
    )
    return {
        "coords": coords,
        "centroid_2d": centroid_2d,
        "layout_labels": layout_labels,
        "scale": float(scale),
        "local_pca_means": local_pca_means,
        "local_pca_components": local_pca_components,
    }


def _project_residuals_with_local_pca(
    residuals: np.ndarray,
    mean: np.ndarray,
    components: np.ndarray,
) -> np.ndarray:
    if components.shape[0] == 0 or residuals.shape[0] == 0:
        return np.zeros((residuals.shape[0], 2), dtype=np.float32)

    local_2d = ((residuals - mean) @ components.T).astype(np.float32)
    if local_2d.shape[1] < 2:
        local_2d = np.column_stack(
            [local_2d, np.zeros(local_2d.shape[0], dtype=np.float32)]
        )
    return local_2d


def load_reference_model(reference_model_path: str) -> dict[str, np.ndarray]:
    """Load frozen PCA/k-means reference artifacts from select_embeddings.py."""
    with np.load(reference_model_path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def _normalize_rows(data: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(data, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-10)
    return data / norms


def transform_with_reference_pca(
    embeddings: np.ndarray,
    reference_model: dict[str, np.ndarray],
) -> np.ndarray:
    """Project embeddings into the frozen reference PCA space."""
    return (
        (embeddings - reference_model["pca_mean"]) @ reference_model["pca_components"].T
    ).astype(np.float32)


def assign_reference_parents(
    pca_data: np.ndarray,
    reference_model: dict[str, np.ndarray],
) -> np.ndarray:
    """Assign embeddings to the frozen spherical k-means parent clusters."""
    from sklearn.metrics import pairwise_distances_argmin

    normalized = _normalize_rows(pca_data)
    return pairwise_distances_argmin(
        normalized,
        reference_model["normalized_centers"],
        metric="euclidean",
    ).astype(np.int32)


def project_overlay_embeddings(
    embeddings: np.ndarray,
    reference_model: dict[str, np.ndarray],
    layout_model: ClusterLayoutModel,
) -> tuple[np.ndarray, np.ndarray]:
    """Project external embeddings into a frozen reference cluster layout."""
    pca_data = transform_with_reference_pca(embeddings, reference_model)
    parent_labels = assign_reference_parents(pca_data, reference_model)
    residuals = pca_data - reference_model["parent_centroids"][parent_labels]

    centroid_2d = np.asarray(layout_model["centroid_2d"], dtype=np.float32)
    scale = float(layout_model["scale"])
    local_pca_means = layout_model["local_pca_means"]
    local_pca_components = layout_model["local_pca_components"]

    coords = np.zeros((embeddings.shape[0], 2), dtype=np.float32)
    for parent_idx in range(centroid_2d.shape[0]):
        mask = parent_labels == parent_idx
        if not np.any(mask):
            continue
        local_2d = _project_residuals_with_local_pca(
            residuals[mask],
            mean=np.asarray(local_pca_means[parent_idx], dtype=np.float32),
            components=np.asarray(local_pca_components[parent_idx], dtype=np.float32),
        )
        coords[mask] = centroid_2d[parent_idx] + local_2d * scale

    return coords, parent_labels


def validate_overlay_reference_compatibility(
    bundle: np.lib.npyio.NpzFile,
    reference_model: dict[str, np.ndarray],
) -> None:
    """Ensure the cluster bundle matches the frozen overlay reference model."""
    bundle_centroids = np.asarray(bundle["centroids"], dtype=np.float32)
    parent_centroids = np.asarray(reference_model["parent_centroids"], dtype=np.float32)

    if bundle_centroids.shape != parent_centroids.shape:
        raise ValueError(
            "Cluster bundle centroids do not match reference_model parent centroids "
            f"shape: {bundle_centroids.shape} vs {parent_centroids.shape}"
        )

    if not np.allclose(bundle_centroids, parent_centroids, atol=1e-4, rtol=1e-4):
        max_abs_diff = float(np.max(np.abs(bundle_centroids - parent_centroids)))
        raise ValueError(
            "Cluster bundle centroids do not match the supplied reference_model. "
            "Use artifacts built from the same reference embeddings and PCA/k-means "
            f"fit. max_abs_diff={max_abs_diff:.3e}"
        )


def _compute_cluster_stats(
    pca_data: np.ndarray,
    centroids: np.ndarray,
    labels_int: np.ndarray,
    metric: str = "cosine",
) -> list[dict]:
    """Per-cluster statistics: point count, mean/std distance to centroid."""
    from scipy.spatial.distance import cdist

    stats: list[dict] = []
    for k in range(centroids.shape[0]):
        mask = labels_int == k
        n_k = int(mask.sum())
        if n_k == 0:
            stats.append({"count": 0, "mean_dist": 0.0, "std_dist": 0.0})
            continue
        dists = cdist(
            pca_data[mask],
            centroids[k : k + 1],
            metric=metric,
        ).ravel()
        stats.append(
            {
                "count": n_k,
                "mean_dist": float(np.mean(dists)),
                "std_dist": float(np.std(dists)),
            }
        )
    return stats


# ---------------------------------------------------------------------------
# Stage 3c -- Inject centroid markers into datamapplot HTML
# ---------------------------------------------------------------------------


def _inject_centroid_layer(
    html_path: str,
    centroid_2d: np.ndarray,
    coords: np.ndarray,
    darkmode: bool = False,
    cluster_stats: list[dict] | None = None,
    metric: str = "cosine",
) -> None:
    """Post-process datamapplot HTML to overlay centroid markers.

    Adds a DeckGL ScatterplotLayer rendered above the point layer with a
    high radiusMinPixels so centroids remain visible at every zoom level.

    ``coords`` is the full (N, 2) point array *before* datamapplot rescales
    it.  We replicate datamapplot's center-and-scale transform so the
    centroids land on top of their clusters.
    """
    import json

    # Replicate the rescaling that datamapplot applies in create_plots.py
    # (see create_interactive_plot → compute_percentile_bounds + affine).
    mean = np.mean(coords, axis=0)
    n_select = int(coords.shape[0] * 0.999)
    vecs = coords - mean
    dists = np.linalg.norm(vecs**2, axis=1)  # matches datamapplot
    sel = coords[np.argsort(dists)[:n_select]]
    raw_w = float(sel[:, 0].max() - sel[:, 0].min())
    raw_h = float(sel[:, 1].max() - sel[:, 1].min())
    raw_scale = max(raw_w, raw_h)

    centroid_scaled = (30.0 / raw_scale) * (centroid_2d - mean)

    with open(html_path) as f:
        html = f.read()

    html = html.replace(
        "'dataPointLayer',",
        "'dataPointLayer','centroidLayer',",
        1,
    )

    centroid_list = [
        [round(float(centroid_scaled[i, 0]), 8), round(float(centroid_scaled[i, 1]), 8)]
        for i in range(centroid_scaled.shape[0])
    ]
    centroid_json = json.dumps(centroid_list)

    fill_rgba = "[255,255,255,230]" if darkmode else "[20,20,20,230]"
    line_rgba = "[0,0,0,200]" if darkmode else "[255,255,255,200]"

    stats_js = "null"
    pickable_js = "false"
    tip_css = ""
    metric_name = ""
    if cluster_stats:
        pickable_js = "true"
        stats_arr = [
            [s["count"], round(s["mean_dist"], 6), round(s["std_dist"], 6)]
            for s in cluster_stats
        ]
        stats_js = json.dumps(stats_arr)
        metric_name = metric
        tip_bg = "rgba(10,10,10,0.92)" if darkmode else "rgba(255,255,255,0.95)"
        tip_fg = "#ddd" if darkmode else "#222"
        tip_bd = "#555" if darkmode else "#ccc"
        tip_css = (
            f"background:{tip_bg};color:{tip_fg};"
            f"border:1px solid {tip_bd};"
            f"box-shadow:0 2px 12px rgba(0,0,0,0.3);"
        )

    script = f"""\
<script>
(function(){{
  var C={centroid_json};
  var S={stats_js};
  var M='{metric_name}';
  var tip=document.createElement('div');
  tip.style.cssText='position:fixed;padding:8px 12px;border-radius:8px;font-family:monospace;font-size:12px;line-height:1.6;pointer-events:none;z-index:10000;display:none;{tip_css}';
  document.body.appendChild(tip);
  function go(){{
    if(typeof datamap==='undefined'||!datamap.pointLayer){{
      setTimeout(go,200);return;}}
    var n=C.length,pos=new Float32Array(n*2),col=new Uint8Array(n*4);
    var fc={fill_rgba};
    for(var i=0;i<n;i++){{
      pos[i*2]=C[i][0];pos[i*2+1]=C[i][1];
      col[i*4]=fc[0];col[i*4+1]=fc[1];col[i*4+2]=fc[2];col[i*4+3]=fc[3];
    }}
    var lyr=new deck.ScatterplotLayer({{
      id:'centroidLayer',
      data:{{length:n,attributes:{{
        getPosition:{{value:pos,size:2}},
        getFillColor:{{value:col,size:4}}
      }}}},
      getRadius:datamap.pointSize*4,
      getLineColor:{line_rgba},
      getLineWidth:datamap.pointLineWidth*3,
      lineWidthMaxPixels:3,lineWidthMinPixels:1.5,
      radiusMaxPixels:32,radiusMinPixels:5,
      radiusUnits:"common",lineWidthUnits:"common",
      stroked:true,filled:true,pickable:{pickable_js},
      onHover:function(info){{
        if(info.picked&&S){{
          var s=S[info.index];
          tip.innerHTML='<b>Cluster '+info.index+'</b><br>'
            +'Count: '+s[0]+'<br>'
            +'Mean '+M+' dist: '+s[1].toFixed(4)+'<br>'
            +'Std '+M+' dist: '+s[2].toFixed(4);
          tip.style.left=(info.x+14)+'px';
          tip.style.top=(info.y-10)+'px';
          tip.style.display='block';
        }}else{{tip.style.display='none';}}
      }},
      parameters:{{depthTest:false}}
    }});
    datamap.layers.push(lyr);
    datamap.layers.sort(function(a,b){{return getLayerIndex(a)-getLayerIndex(b);}});
    datamap.deckgl.setProps({{layers:[].concat(datamap.layers)}});
  }}
  go();
}})();
</script>"""

    html = html.replace("</html>", script + "\n</html>")

    with open(html_path, "w") as f:
        f.write(html)

    print(
        f"[viz] Injected centroid markers ({centroid_2d.shape[0]} centroids) "
        f"into {html_path}"
    )


# ---------------------------------------------------------------------------
# Stage 3d -- Inject click-to-view image panel into datamapplot HTML
# ---------------------------------------------------------------------------


def _inject_click_image_panel(
    html_path: str,
    url_prefix: str,
    ext: str = ".jpg",
    darkmode: bool = False,
) -> None:
    """Post-process datamapplot HTML to add a click-to-view image panel.

    When the user clicks a point, the panel shows a thumbnail loaded from
    ``{url_prefix}/{filename}{ext}`` alongside the metadata table.  Images
    are fetched lazily (one at a time) so there is zero impact on initial
    page load.

    Requires serving the HTML via HTTP (e.g. ``python -m http.server``)
    so the browser can fetch images.
    """
    bg = "rgba(10,10,10,0.92)" if darkmode else "rgba(255,255,255,0.95)"
    text_color = "#ddd" if darkmode else "#222"
    border_color = "#444" if darkmode else "#ccc"

    if url_prefix and not url_prefix.endswith("/"):
        url_prefix += "/"

    style_block = (
        "<style>\n"
        "#thumbnail-panel{"
        "position:fixed;top:80px;right:16px;width:260px;max-height:80vh;"
        "overflow-y:auto;background:PANEL_BG;color:PANEL_TEXT;"
        "border:1px solid PANEL_BORDER;border-radius:12px;padding:12px;"
        "z-index:1000;font-family:monospace;font-size:11px;"
        "box-shadow:0 4px 24px rgba(0,0,0,0.4);backdrop-filter:blur(8px);"
        "display:none;pointer-events:auto;}\n"
        "#thumbnail-panel .thumb-close{"
        "position:absolute;top:6px;right:10px;cursor:pointer;"
        "font-size:20px;line-height:1;color:PANEL_TEXT;opacity:0.5;}\n"
        "#thumbnail-panel .thumb-close:hover{opacity:1;}\n"
        "#thumbnail-img{"
        "width:100%;border-radius:6px;margin-bottom:8px;}\n"
        "#thumbnail-info table{width:100%;border-collapse:collapse;}\n"
        "#thumbnail-info td{"
        "padding:2px 4px;border-bottom:1px solid PANEL_BORDER33;"
        "word-break:break-all;}\n"
        "#thumbnail-error{"
        "text-align:center;padding:16px 0;color:PANEL_TEXT;opacity:0.5;}\n"
        "</style>\n"
    )
    style_block = (
        style_block.replace("PANEL_BG", bg)
        .replace("PANEL_TEXT", text_color)
        .replace("PANEL_BORDER33", border_color + "33")
        .replace("PANEL_BORDER", border_color)
    )

    # JS is built as a plain string to avoid f-string brace escaping issues.
    js_block = r"""<script>
(function(){
  var PREFIX='__PREFIX__';
  var EXT='__EXT__';
  var p=document.createElement('div');
  p.id='thumbnail-panel';
  p.innerHTML='<span class="thumb-close" id="thumb-close">\u00d7</span>'
    +'<img id="thumbnail-img" style="display:none">'
    +'<div id="thumbnail-error" style="display:none">No image available</div>'
    +'<div id="thumbnail-info"></div>';
  document.body.appendChild(p);
  document.getElementById('thumb-close').addEventListener('click',function(){
    p.style.display='none';
  });
  document.addEventListener('keydown',function(e){
    if(e.key==='Escape') p.style.display='none';
  });
  function show(index){
    var m=datamap.metaData;
    if(!m||!m.filename) return;
    var img=document.getElementById('thumbnail-img');
    var err=document.getElementById('thumbnail-error');
    var info=document.getElementById('thumbnail-info');
    img.style.display='none'; err.style.display='none';
    img.src=PREFIX+m.filename[index]+EXT;
    img.onload=function(){ img.style.display='block'; };
    img.onerror=function(){ err.style.display='block'; };
    var rows='';
    for(var k in m){
      if(k==='hover_text') continue;
      var v=m[k][index]; if(v==null) v='';
      rows+='<tr><td style="font-weight:bold;padding-right:8px;">'+k+'</td><td>'+v+'</td></tr>';
    }
    info.innerHTML='<table>'+rows+'</table>';
    p.style.display='block';
  }
  function go(){
    if(typeof datamap==='undefined'||!datamap.metaData){
      setTimeout(go,300);return;}
    datamap.deckgl.setProps({onClick:function(info,event){
      if(info.picked){ show(info.index); }
      else { p.style.display='none'; }
    }});
  }
  go();
})();
</script>"""
    js_block = js_block.replace("__PREFIX__", url_prefix).replace("__EXT__", ext)

    with open(html_path) as f:
        html = f.read()

    html = html.replace("</html>", style_block + js_block + "\n</html>")

    with open(html_path, "w") as f:
        f.write(html)

    print(
        f"[viz] Injected click-to-image panel (url_prefix={url_prefix!r}, "
        f"ext={ext!r}) into {html_path}"
    )


def _rescale_like_datamapplot(
    reference_coords: np.ndarray,
    coords_to_scale: np.ndarray,
) -> np.ndarray:
    """Apply the same affine rescale datamapplot uses internally."""
    mean = np.mean(reference_coords, axis=0)
    n_select = int(reference_coords.shape[0] * 0.999)
    vecs = reference_coords - mean
    dists = np.linalg.norm(vecs**2, axis=1)
    sel = reference_coords[np.argsort(dists)[:n_select]]
    raw_w = float(sel[:, 0].max() - sel[:, 0].min())
    raw_h = float(sel[:, 1].max() - sel[:, 1].min())
    raw_scale = max(raw_w, raw_h, 1e-10)
    return ((30.0 / raw_scale) * (coords_to_scale - mean)).astype(np.float32)


def _inject_overlay_layer(
    html_path: str,
    reference_coords: np.ndarray,
    overlay_coords: np.ndarray,
) -> None:
    """Overlay projected candidate points on top of the reference map."""
    import json

    overlay_scaled = _rescale_like_datamapplot(reference_coords, overlay_coords)
    overlay_list = [
        [round(float(overlay_scaled[i, 0]), 8), round(float(overlay_scaled[i, 1]), 8)]
        for i in range(overlay_scaled.shape[0])
    ]
    overlay_json = json.dumps(overlay_list)

    script = f"""\
<script>
(function(){{
  var O={overlay_json};
  function go(){{
    if(typeof datamap==='undefined'||!datamap.pointLayer){{
      setTimeout(go,200);return;}}
    var n=O.length,pos=new Float32Array(n*2);
    for(var i=0;i<n;i++){{
      pos[i*2]=O[i][0];pos[i*2+1]=O[i][1];
    }}
    var lyr=new deck.ScatterplotLayer({{
      id:'candidateOverlayLayer',
      data:{{length:n,attributes:{{
        getPosition:{{value:pos,size:2}}
      }}}},
      getRadius:Math.max(datamap.pointSize*0.9,0.02),
      getFillColor:[0,0,0,235],
      getLineColor:[168,255,112,255],
      getLineWidth:Math.max(datamap.pointLineWidth*3.0,0.6),
      lineWidthMaxPixels:3,lineWidthMinPixels:1.2,
      radiusMaxPixels:8,radiusMinPixels:1.8,
      radiusUnits:"common",lineWidthUnits:"common",
      stroked:true,filled:true,pickable:false,
      parameters:{{depthTest:false}}
    }});
    datamap.layers.push(lyr);
    datamap.deckgl.setProps({{layers:[].concat(datamap.layers)}});
  }}
  go();
}})();
</script>"""

    with open(html_path) as f:
        html = f.read()

    html = html.replace("</html>", script + "\n</html>")

    with open(html_path, "w") as f:
        f.write(html)

    print(
        f"[viz] Injected overlay layer ({overlay_coords.shape[0]:,} points) "
        f"into {html_path}"
    )


# ---------------------------------------------------------------------------
# Stage 4 -- Interactive Visualization
# ---------------------------------------------------------------------------


def visualize(
    coords: np.ndarray,
    output_path: str,
    title: str = "Embedding Map",
    darkmode: bool = False,
    labels: np.ndarray | None = None,
    extra_point_data: pd.DataFrame | None = None,
    monochrome_labels: bool = False,
) -> None:
    """Create an interactive datamapplot HTML file."""
    import datamapplot

    print(f"[viz] Building interactive plot for {coords.shape[0]:,} points ...")
    t0 = time.time()

    label_args = (labels,) if labels is not None else ()

    cmap_kwargs = {}
    if labels is not None:
        label_color_map = _build_label_color_map(
            labels,
            darkmode,
            monochrome=monochrome_labels,
        )
        if label_color_map is not None:
            cmap_kwargs["label_color_map"] = label_color_map

    hover_template = None
    hover_text = None
    if extra_point_data is not None:
        print(
            f"[viz] Including {len(extra_point_data.columns)} metadata "
            f"columns in hover tooltip"
        )
        # datamapplot only enables the deck.gl tooltip hook when hover_text
        # exists, even if a custom HTML tooltip function is provided.
        hover_text = np.full(coords.shape[0], "", dtype=object)
        rows = "".join(
            f'<tr><td style="font-weight:bold;padding-right:8px;">{col}</td>'
            f"<td>{{{col}}}</td></tr>"
            for col in extra_point_data.columns
        )
        hover_template = (
            '<div style="font-size:12px;font-family:monospace;">'
            f"<table>{rows}</table></div>"
        )

    plot = datamapplot.create_interactive_plot(
        coords,
        *label_args,
        hover_text=hover_text,
        title=title,
        sub_title=f"{coords.shape[0]:,} embeddings",
        darkmode=darkmode,
        inline_data=True,
        enable_search=labels is not None,
        cluster_boundary_polygons=labels is not None,
        extra_point_data=extra_point_data,
        hover_text_html_template=hover_template,
        **cmap_kwargs,
    )

    plot.save(output_path)
    elapsed = time.time() - t0
    print(f"[viz] Saved interactive map to {output_path}  ({elapsed:.1f}s)")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for t-SNE or frozen-cluster-layout visualization."""
    p = argparse.ArgumentParser(
        description="Visualize embeddings: load -> PCA -> (t-SNE | cluster layout) "
        "-> interactive datamapplot map",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--input-dir",
        default=None,
        help="Directory containing shard_*.npz files. Required for --layout tsne.",
    )
    p.add_argument(
        "--output-dir",
        default=None,
        help="Where to write cached arrays and HTML. "
        "Defaults to {input_dir}/_viz (tsne) or next "
        "to the cluster bundle (cluster).",
    )
    p.add_argument(
        "--subsample",
        type=int,
        default=0,
        help="Randomly subsample to this many points (0 = all)",
    )
    p.add_argument(
        "--pca-dim", type=int, default=128, help="PCA intermediate dimensionality"
    )
    p.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility"
    )
    p.add_argument(
        "--skip-reduction",
        action="store_true",
        help="Skip reduction, load cached coords and go straight to viz",
    )
    p.add_argument(
        "--labels",
        default=None,
        help="Optional .npy file with per-point string labels "
        "(e.g. from clustering). With --layout cluster, these "
        "override the displayed labels/colors but do not "
        "change the layout geometry.",
    )
    p.add_argument(
        "--title", default="Embedding Map", help="Title shown on the interactive map"
    )
    p.add_argument(
        "--darkmode", action="store_true", help="Use dark theme for the interactive map"
    )
    p.add_argument(
        "--output-name",
        default="embedding_map.html",
        help="Filename for the output HTML (written inside --output-dir)",
    )
    p.add_argument(
        "--metadata",
        default=None,
        help="Parquet file with per-sample attributes. Must have "
        "a 'sample_index' (or 'sample_idx') column matching "
        "the extraction index.csv. All other columns are "
        "shown in the hover tooltip.",
    )
    p.add_argument(
        "--metadata-columns",
        nargs="+",
        default=None,
        help="Subset of metadata columns to include in the hover "
        "tooltip. If omitted, all columns are included.",
    )
    p.add_argument(
        "--thumbnail-url-prefix",
        default=None,
        help="URL prefix for thumbnail images (e.g. './thumbnails/'). "
        "When set, clicking a point loads "
        "{prefix}/{filename}{ext} in a side panel. "
        "Requires serving via HTTP. Omit to disable.",
    )
    p.add_argument(
        "--thumbnail-ext",
        default=".jpg",
        help="File extension appended to the metadata filename "
        "when constructing thumbnail URLs.",
    )

    # -- Layout selection ---------------------------------------------------
    p.add_argument(
        "--layout",
        default="tsne",
        choices=["tsne", "cluster"],
        help="2D layout method. 'tsne': PCA + GPU t-SNE (original). "
        "'cluster': MDS on k-means centroids with per-cluster "
        "PCA placement (distance-faithful).",
    )

    # -- t-SNE-specific args -----------------------------------------------
    tsne_group = p.add_argument_group("t-SNE options (--layout tsne)")
    tsne_group.add_argument(
        "--tsne-dim",
        type=int,
        default=2,
        choices=[2, 3],
        help="t-SNE output dimensionality",
    )
    tsne_group.add_argument(
        "--perplexity", type=float, default=30.0, help="t-SNE perplexity"
    )

    # -- Cluster-layout-specific args --------------------------------------
    cl_group = p.add_argument_group("Cluster layout options (--layout cluster)")
    cl_group.add_argument(
        "--cluster-bundle",
        default=None,
        help="Path to cluster_bundle_*.npz produced by "
        "cluster_embeddings.py. Required for "
        "--layout cluster.",
    )
    cl_group.add_argument(
        "--metric",
        default="cosine",
        choices=["cosine", "euclidean"],
        help="Distance metric for inter-centroid MDS",
    )
    cl_group.add_argument(
        "--separation",
        type=float,
        default=0.6,
        help="Fraction of inter-centroid gap used for "
        "intra-cluster radius (0-1). Higher = denser.",
    )
    cl_group.add_argument(
        "--overlay-input-dir",
        default=None,
        help="Optional candidate embedding directory to project into the frozen "
        "cluster layout. Requires --overlay-reference-model.",
    )
    cl_group.add_argument(
        "--overlay-reference-model",
        default=None,
        help="Path to reference_model.npz from select_embeddings.py fit-reference. "
        "Used to project --overlay-input-dir into the reference PCA and "
        "parent clusters.",
    )

    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Build the requested visualization and optional overlay artifacts."""
    args = parse_args(argv)

    if args.layout == "cluster":
        if args.overlay_input_dir and not args.overlay_reference_model:
            print(
                "[error] --overlay-input-dir requires --overlay-reference-model",
                file=sys.stderr,
            )
            sys.exit(1)
        if args.output_dir:
            output_dir = args.output_dir
        elif args.cluster_bundle:
            output_dir = os.path.join(os.path.dirname(args.cluster_bundle), "_viz")
        else:
            output_dir = "."
    else:
        if not args.input_dir:
            print("[error] --input-dir is required for --layout tsne", file=sys.stderr)
            sys.exit(1)
        output_dir = args.output_dir or os.path.join(args.input_dir, "_viz")

    os.makedirs(output_dir, exist_ok=True)
    html_path = os.path.join(output_dir, args.output_name)

    # Track which sample_idx values correspond to the N visualized points.
    # None means sequential (0..N-1); otherwise an int array of kept indices.
    sample_indices: np.ndarray | None = None
    overlay_coords: np.ndarray | None = None

    if args.layout == "cluster":
        # -- Cluster-distance layout ---------------------------------------
        if not args.cluster_bundle:
            print(
                "[error] --layout cluster requires --cluster-bundle "
                "(run cluster_embeddings.py first)",
                file=sys.stderr,
            )
            sys.exit(1)

        bundle = np.load(args.cluster_bundle, allow_pickle=True)
        labels = bundle["labels_str"]
        layout_cache = os.path.join(output_dir, "cluster_layout.npy")
        centroid_2d_cache = os.path.join(output_dir, "centroid_2d.npy")
        layout_model: ClusterLayoutModel | None = None

        if args.skip_reduction and os.path.exists(layout_cache):
            coords = np.load(layout_cache)
            print(
                f"[skip] Loaded cached cluster layout {coords.shape} "
                f"from {layout_cache}"
            )
            if os.path.exists(centroid_2d_cache):
                centroid_2d = np.load(centroid_2d_cache)
            else:
                centroid_2d = None
        else:
            layout_model = build_cluster_layout_model(
                pca_data=bundle["pca_data"],
                centroids=bundle["centroids"],
                labels_int=bundle["labels_int"],
                metric=args.metric,
                separation=args.separation,
                seed=args.seed,
            )
            coords = np.asarray(layout_model["coords"], dtype=np.float32)
            centroid_2d = np.asarray(layout_model["centroid_2d"], dtype=np.float32)
            np.save(layout_cache, coords)
            np.save(centroid_2d_cache, centroid_2d)
            print(f"[cluster-layout] Cached coords to {layout_cache}")

        if "sample_indices" in bundle:
            sample_indices = bundle["sample_indices"]
        elif args.input_dir:
            sample_indices = _load_sample_indices_from_csv(
                args.input_dir,
                n_points=coords.shape[0],
            )

        if args.labels:
            override_labels = np.load(args.labels, allow_pickle=True)
            if override_labels.shape[0] != coords.shape[0]:
                print(
                    f"[warn] Label count ({override_labels.shape[0]}) != coord "
                    f"count ({coords.shape[0]}). Using bundle labels instead.",
                    file=sys.stderr,
                )
            else:
                labels = override_labels
                print(f"[cluster-layout] Using display labels from {args.labels}")

        if args.overlay_input_dir:
            if layout_model is None:
                layout_model = build_cluster_layout_model(
                    pca_data=bundle["pca_data"],
                    centroids=bundle["centroids"],
                    labels_int=bundle["labels_int"],
                    metric=args.metric,
                    separation=args.separation,
                    seed=args.seed,
                )
            reference_model = load_reference_model(args.overlay_reference_model)
            validate_overlay_reference_compatibility(bundle, reference_model)
            overlay_embeddings, _ = load_embeddings(
                args.overlay_input_dir,
                subsample=0,
                seed=args.seed,
            )
            overlay_coords, overlay_parent_labels = project_overlay_embeddings(
                overlay_embeddings,
                reference_model,
                layout_model,
            )
            parent_counts = np.bincount(
                overlay_parent_labels,
                minlength=bundle["centroids"].shape[0],
            )
            print(
                f"[overlay] Projected {overlay_coords.shape[0]:,} candidate points "
                f"into frozen layout"
            )
            print(f"[overlay] Parent cluster counts: {parent_counts.tolist()}")

    else:
        # -- t-SNE layout (original) ---------------------------------------
        centroid_2d = None
        pca_path = os.path.join(output_dir, f"pca_{args.pca_dim}.npy")
        tsne_path = os.path.join(output_dir, f"tsne_{args.tsne_dim}d.npy")

        if args.skip_reduction:
            if not os.path.exists(tsne_path):
                print(
                    f"[error] --skip-reduction but {tsne_path} not found",
                    file=sys.stderr,
                )
                sys.exit(1)
            coords = np.load(tsne_path)
            print(f"[skip] Loaded cached t-SNE coords {coords.shape} from {tsne_path}")
            sample_indices = _load_sample_indices_from_csv(
                args.input_dir,
                n_points=coords.shape[0],
            )
        else:
            embeddings, subsample_positions = load_embeddings(
                args.input_dir, args.subsample, args.seed
            )
            pca_result = run_pca(embeddings, args.pca_dim, pca_path, args.seed)
            del embeddings
            coords = run_tsne(
                pca_result,
                args.tsne_dim,
                tsne_path,
                args.perplexity,
                args.seed,
            )
            del pca_result

            sample_indices = _load_sample_indices_from_csv(
                args.input_dir,
                n_points=coords.shape[0],
                subsample_positions=subsample_positions,
            )

        if args.tsne_dim != 2:
            print(
                "[warn] datamapplot requires 2D coords; skipping "
                "interactive viz for 3D. Use --tsne-dim 2 or a "
                "different 3D viewer."
            )
            return

        labels = None
        if args.labels:
            labels = np.load(args.labels, allow_pickle=True)
            if labels.shape[0] != coords.shape[0]:
                print(
                    f"[warn] Label count ({labels.shape[0]}) != coord "
                    f"count ({coords.shape[0]}). Ignoring labels.",
                    file=sys.stderr,
                )
                labels = None

    # -- Load optional metadata ---------------------------------------------
    extra_point_data = None
    if args.metadata:
        extra_point_data = load_metadata(
            args.metadata,
            n_points=coords.shape[0],
            sample_indices=sample_indices,
            columns=args.metadata_columns,
        )

    # -- Visualize ----------------------------------------------------------
    visualize(
        coords,
        html_path,
        title=args.title,
        darkmode=args.darkmode,
        labels=labels,
        extra_point_data=extra_point_data,
        monochrome_labels=overlay_coords is not None,
    )

    if args.layout == "cluster" and centroid_2d is not None:
        cluster_stats = _compute_cluster_stats(
            bundle["pca_data"],
            bundle["centroids"],
            bundle["labels_int"],
            metric=args.metric,
        )
        _inject_centroid_layer(
            html_path,
            centroid_2d,
            coords,
            darkmode=args.darkmode,
            cluster_stats=cluster_stats,
            metric=args.metric,
        )

    if args.thumbnail_url_prefix:
        _inject_click_image_panel(
            html_path,
            args.thumbnail_url_prefix,
            ext=args.thumbnail_ext,
            darkmode=args.darkmode,
        )

    if overlay_coords is not None:
        _inject_overlay_layer(html_path, coords, overlay_coords)

    print(f"\nDone. Open {html_path} in a browser.")
    if args.thumbnail_url_prefix:
        print(
            "     (Serve via HTTP for image thumbnails, e.g. "
            "python -m http.server -d "
            f"{os.path.dirname(html_path) or '.'})"
        )


if __name__ == "__main__":
    main()
