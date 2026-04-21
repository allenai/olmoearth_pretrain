"""Shared acquisition scoring strategies for embedding selection.

These strategies operate on candidate embeddings that have already been:
1. projected with the frozen reference PCA
2. assigned to frozen spherical k-means parent clusters
3. converted into parent-relative residual vectors

The goal is to expose several complementary acquisition scores:
- xglobal bridge: ambiguity between top parent assignments
- sparse cluster infill: sparse but still supported regions within a parent
- xlocal bridge: points supported by two nearby local modes
- prototypes: locally central, representative points
"""

from __future__ import annotations

import numpy as np
from reference_model import EPS


def _safe_scale(values: np.ndarray, fallback: float = 1.0) -> float:
    """Return a positive scale even when values are empty or degenerate."""
    if values.size == 0:
        return fallback
    scale = float(np.percentile(values, 90))
    if scale <= 0:
        scale = float(np.mean(values)) if values.size else fallback
    if scale <= 0:
        scale = fallback
    return scale


def _percentile_value(values: np.ndarray, percentile: float) -> float:
    """Return a percentile value with sensible fallbacks for empty inputs."""
    if not 0.0 <= percentile <= 100.0:
        raise ValueError(f"percentile must be in [0, 100], got {percentile}")
    if values.size == 0:
        return 0.0
    return float(np.percentile(values, percentile))


def _soft_gate_width(values: np.ndarray, percentile: float) -> float:
    """Estimate a local transition width around a percentile gate."""
    if values.size == 0:
        return 1.0

    lower = max(0.0, percentile - 5.0)
    upper = min(100.0, percentile + 5.0)
    width = 0.5 * (_percentile_value(values, upper) - _percentile_value(values, lower))
    if width <= 0:
        width = float(np.std(values))
    if width <= 0:
        width = _safe_scale(values)
    if width <= 0:
        width = 1.0
    return width


def _sigmoid(values: np.ndarray) -> np.ndarray:
    """Stable logistic mapping used for soft gates."""
    clipped = np.clip(values, -60.0, 60.0)
    return (1.0 / (1.0 + np.exp(-clipped))).astype(np.float32)


def _compute_segment_geometry(
    points: np.ndarray,
    starts: np.ndarray,
    ends: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return segment position and perpendicular distance to the segment axis."""
    segment = ends - starts
    segment_norm_sq = np.sum(segment * segment, axis=-1)
    relative = points - starts
    t_raw = np.sum(relative * segment, axis=-1) / (segment_norm_sq + EPS)
    line_proj = starts + t_raw[..., None] * segment
    d_off = np.linalg.norm(points - line_proj, axis=-1).astype(np.float32)
    return t_raw.astype(np.float32), d_off


def _stack_reference_points(
    parent_centroids: np.ndarray,
    residuals_by_parent: dict[int, np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    """Reconstruct reference PCA points from stored residuals."""
    points: list[np.ndarray] = []
    parent_ids: list[np.ndarray] = []

    for parent_idx in range(parent_centroids.shape[0]):
        residuals = residuals_by_parent.get(parent_idx)
        if residuals is None or residuals.shape[0] == 0:
            continue
        points.append(
            (residuals + parent_centroids[parent_idx]).astype(np.float32, copy=False)
        )
        parent_ids.append(np.full(residuals.shape[0], parent_idx, dtype=np.int32))

    if not points:
        return (
            np.zeros((0, parent_centroids.shape[1]), dtype=np.float32),
            np.zeros(0, dtype=np.int32),
        )
    return (
        np.concatenate(points, axis=0).astype(np.float32, copy=False),
        np.concatenate(parent_ids, axis=0),
    )


def _estimate_bridge_pair_scales(
    reference_points: np.ndarray,
    owner_labels: np.ndarray,
    centers: np.ndarray,
) -> dict[tuple[int, int], float]:
    """Estimate pair-specific off-axis scales from frozen reference geometry."""
    if reference_points.shape[0] == 0 or centers.shape[0] < 2:
        return {}

    center_dists = np.linalg.norm(
        reference_points - centers[owner_labels],
        axis=1,
    ).astype(np.float32)
    fallback = _safe_scale(center_dists)

    distances = np.linalg.norm(
        reference_points[:, None, :] - centers[None, :, :],
        axis=2,
    ).astype(np.float32)
    nearest_two = np.argsort(distances, axis=1)[:, :2]
    primary = nearest_two[:, 0].astype(np.int32)
    secondary = nearest_two[:, 1].astype(np.int32)
    pair_lo = np.minimum(primary, secondary)
    pair_hi = np.maximum(primary, secondary)
    d1 = distances[np.arange(distances.shape[0]), primary]
    d2 = distances[np.arange(distances.shape[0]), secondary]
    balance = (1.0 - (np.abs(d1 - d2) / (d1 + d2 + EPS))).astype(np.float32)

    pair_scales: dict[tuple[int, int], float] = {}
    n_centers = centers.shape[0]
    for i in range(n_centers):
        for j in range(i + 1, n_centers):
            pair_mask = np.logical_and(pair_lo == i, pair_hi == j)
            if np.any(pair_mask):
                pair_points = reference_points[pair_mask]
                pair_balance = balance[pair_mask]
                t_raw, off_axis = _compute_segment_geometry(
                    pair_points,
                    centers[i],
                    centers[j],
                )
                strict_mask = np.logical_and(
                    np.logical_and(t_raw >= -0.1, t_raw <= 1.1),
                    pair_balance >= 0.7,
                )
                relaxed_mask = np.logical_and(t_raw >= -0.1, t_raw <= 1.1)
                if np.any(strict_mask):
                    selected = off_axis[strict_mask]
                elif np.any(relaxed_mask):
                    selected = off_axis[relaxed_mask]
                else:
                    selected = off_axis
                pair_scales[(i, j)] = _safe_scale(selected, fallback=fallback)
                continue

            union_mask = np.logical_or(owner_labels == i, owner_labels == j)
            union_points = reference_points[union_mask]
            if union_points.shape[0] == 0:
                pair_scales[(i, j)] = fallback
                continue
            _, off_axis = _compute_segment_geometry(
                union_points,
                centers[i],
                centers[j],
            )
            pair_scales[(i, j)] = _safe_scale(off_axis, fallback=fallback)

    return pair_scales


def _estimate_xglobal_bridge_pair_scales(
    parent_centroids: np.ndarray,
    residuals_by_parent: dict[int, np.ndarray],
) -> dict[tuple[int, int], float]:
    """Estimate pair-specific off-axis scales from frozen reference parent geometry."""
    reference_points, reference_parent_labels = _stack_reference_points(
        parent_centroids,
        residuals_by_parent,
    )
    return _estimate_bridge_pair_scales(
        reference_points,
        reference_parent_labels,
        parent_centroids,
    )


def _compute_query_knn_stats(
    reference: np.ndarray,
    queries: np.ndarray,
    knn_k: int,
    metric: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return mean-kNN distance, nearest distance, and effective k per query."""
    from sklearn.neighbors import NearestNeighbors

    mean_dist = np.zeros(queries.shape[0], dtype=np.float32)
    nearest_dist = np.zeros(queries.shape[0], dtype=np.float32)
    k_used = np.zeros(queries.shape[0], dtype=np.int32)

    if queries.shape[0] == 0 or reference.shape[0] == 0:
        return mean_dist, nearest_dist, k_used
    if knn_k <= 0:
        raise ValueError(f"knn_k must be > 0, got {knn_k}")

    effective_k = min(knn_k, reference.shape[0])
    nn = NearestNeighbors(n_neighbors=effective_k, metric=metric)
    nn.fit(reference)
    distances, _ = nn.kneighbors(queries)
    mean_dist[:] = distances.mean(axis=1).astype(np.float32)
    nearest_dist[:] = distances[:, 0].astype(np.float32)
    k_used[:] = effective_k
    return mean_dist, nearest_dist, k_used


def _compute_reference_self_knn_stats(
    reference: np.ndarray,
    knn_k: int,
    metric: str,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Return self mean-kNN and self nearest-neighbor distances."""
    from sklearn.neighbors import NearestNeighbors

    if reference.shape[0] == 0:
        return (
            np.zeros(0, dtype=np.float32),
            np.zeros(0, dtype=np.float32),
            0,
        )
    if reference.shape[0] == 1:
        return (
            np.zeros(1, dtype=np.float32),
            np.zeros(1, dtype=np.float32),
            0,
        )
    if knn_k <= 0:
        raise ValueError(f"knn_k must be > 0, got {knn_k}")

    effective_k = min(knn_k, reference.shape[0] - 1)
    nn = NearestNeighbors(n_neighbors=effective_k + 1, metric=metric)
    nn.fit(reference)
    distances, _ = nn.kneighbors(reference)
    return (
        distances[:, 1:].mean(axis=1).astype(np.float32),
        distances[:, 1].astype(np.float32),
        effective_k,
    )


def compute_xglobal_bridge_scores(
    pca_data: np.ndarray,
    parent_centroids: np.ndarray,
    residuals_by_parent: dict[int, np.ndarray],
) -> dict[str, np.ndarray]:
    """Score candidates that bridge two frozen parent clusters."""
    n_centers = parent_centroids.shape[0]
    secondary = np.full(pca_data.shape[0], -1, dtype=np.int32)
    zeros = np.zeros(pca_data.shape[0], dtype=np.float32)
    if n_centers < 2:
        return {
            "score": zeros.copy(),
            "secondary_parent": secondary,
            "primary_distance": zeros.copy(),
            "secondary_distance": zeros.copy(),
            "balance_term": zeros.copy(),
            "between_term": zeros.copy(),
            "off_axis_distance": zeros.copy(),
            "segment_position": zeros.copy(),
            "off_axis_scale": zeros.copy(),
        }

    distances = np.linalg.norm(
        pca_data[:, None, :] - parent_centroids[None, :, :],
        axis=2,
    ).astype(np.float32)
    nearest_two = np.argsort(distances, axis=1)[:, :2]
    primary = nearest_two[:, 0].astype(np.int32)
    secondary = nearest_two[:, 1].astype(np.int32)
    d1 = distances[np.arange(distances.shape[0]), primary]
    d2 = distances[np.arange(distances.shape[0]), secondary]

    pair_scales = _estimate_xglobal_bridge_pair_scales(
        parent_centroids, residuals_by_parent
    )
    c1 = parent_centroids[primary]
    c2 = parent_centroids[secondary]
    t_raw, d_off = _compute_segment_geometry(pca_data, c1, c2)

    pair_keys = [
        (int(min(a, b)), int(max(a, b)))
        for a, b in zip(primary.tolist(), secondary.tolist(), strict=False)
    ]
    global_fallback = _safe_scale(d_off)
    scale_off = np.array(
        [pair_scales.get(key, global_fallback) for key in pair_keys],
        dtype=np.float32,
    )
    overshoot = np.maximum(0.0, np.maximum(-t_raw, t_raw - 1.0)).astype(np.float32)

    balance = (1.0 - (np.abs(d1 - d2) / (d1 + d2 + EPS))).astype(np.float32)
    off_gate = np.exp(-d_off / (scale_off + EPS)).astype(np.float32)
    position_gate = np.exp(-(overshoot / 0.25)).astype(np.float32)
    between = (off_gate * position_gate).astype(np.float32)
    score = (balance * between).astype(np.float32)

    return {
        "score": score,
        "secondary_parent": secondary,
        "primary_distance": d1.astype(np.float32),
        "secondary_distance": d2.astype(np.float32),
        "balance_term": balance,
        "between_term": between,
        "off_axis_distance": d_off,
        "segment_position": t_raw.astype(np.float32),
        "off_axis_scale": scale_off,
    }


def compute_sparse_infill_scores(
    residuals: np.ndarray,
    parent_labels: np.ndarray,
    residuals_by_parent: dict[int, np.ndarray],
    k_sparse: int,
    k_support: int,
    sparse_percentile: float,
    support_percentile: float,
    metric: str,
) -> dict[str, np.ndarray]:
    """Prefer sparse-but-supported regions within each parent residual space."""
    score = np.zeros(residuals.shape[0], dtype=np.float32)
    sparse_knn = np.zeros(residuals.shape[0], dtype=np.float32)
    support_knn = np.zeros(residuals.shape[0], dtype=np.float32)
    sparse_term = np.zeros(residuals.shape[0], dtype=np.float32)
    support_term = np.zeros(residuals.shape[0], dtype=np.float32)
    sparse_gate_distance = np.zeros(residuals.shape[0], dtype=np.float32)
    support_gate_distance = np.zeros(residuals.shape[0], dtype=np.float32)
    sparse_gate_width = np.zeros(residuals.shape[0], dtype=np.float32)
    support_gate_width = np.zeros(residuals.shape[0], dtype=np.float32)
    k_sparse_used = np.zeros(residuals.shape[0], dtype=np.int32)
    k_support_used = np.zeros(residuals.shape[0], dtype=np.int32)

    for parent_idx, reference in residuals_by_parent.items():
        mask = parent_labels == parent_idx
        if not np.any(mask):
            continue

        queries = residuals[mask]
        q_sparse, _, q_sparse_used = _compute_query_knn_stats(
            reference,
            queries,
            knn_k=k_sparse,
            metric=metric,
        )
        ref_sparse, _, _ = _compute_reference_self_knn_stats(
            reference,
            knn_k=k_sparse,
            metric=metric,
        )
        q_support, _, q_support_used = _compute_query_knn_stats(
            reference,
            queries,
            knn_k=k_support,
            metric=metric,
        )
        ref_support, _, _ = _compute_reference_self_knn_stats(
            reference,
            knn_k=k_support,
            metric=metric,
        )

        sparse_gate = _percentile_value(ref_sparse, sparse_percentile)
        support_gate = _percentile_value(ref_support, support_percentile)
        sparse_width = _soft_gate_width(ref_sparse, sparse_percentile)
        support_width = _soft_gate_width(ref_support, support_percentile)

        q_sparse_term = _sigmoid((q_sparse - sparse_gate) / (sparse_width + EPS))
        q_support_term = _sigmoid((support_gate - q_support) / (support_width + EPS))
        q_score = (q_sparse_term * q_support_term).astype(np.float32)

        score[mask] = q_score
        sparse_knn[mask] = q_sparse
        support_knn[mask] = q_support
        sparse_term[mask] = q_sparse_term
        support_term[mask] = q_support_term
        sparse_gate_distance[mask] = sparse_gate
        support_gate_distance[mask] = support_gate
        sparse_gate_width[mask] = sparse_width
        support_gate_width[mask] = support_width
        k_sparse_used[mask] = q_sparse_used
        k_support_used[mask] = q_support_used

    return {
        "score": score,
        "sparse_mean_knn_distance": sparse_knn,
        "support_mean_knn_distance": support_knn,
        "sparse_term": sparse_term,
        "support_term": support_term,
        "sparse_gate_distance": sparse_gate_distance,
        "support_gate_distance": support_gate_distance,
        "sparse_gate_width": sparse_gate_width,
        "support_gate_width": support_gate_width,
        "k_sparse_used": k_sparse_used,
        "k_support_used": k_support_used,
    }


def _fit_local_mode_centers(
    reference: np.ndarray,
    n_modes: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fit simple local mode centers for xlocal bridge and prototype scoring."""
    from sklearn.cluster import KMeans

    if n_modes <= 0:
        raise ValueError(f"n_modes must be > 0, got {n_modes}")

    if reference.shape[0] == 0:
        return (
            np.zeros((0, 0), dtype=np.float32),
            np.zeros(0, dtype=np.int32),
            np.zeros(0, dtype=np.float32),
        )
    if reference.shape[0] == 1:
        return (
            reference.astype(np.float32, copy=False),
            np.zeros(1, dtype=np.int32),
            np.zeros(1, dtype=np.float32),
        )

    effective_modes = max(2, min(n_modes, reference.shape[0]))
    km = KMeans(n_clusters=effective_modes, random_state=seed, n_init=10)
    labels = km.fit_predict(reference)
    centers = km.cluster_centers_.astype(np.float32)
    center_dists = np.linalg.norm(reference - centers[labels], axis=1).astype(
        np.float32
    )
    return centers, labels.astype(np.int32), center_dists


def _pairwise_off_axis_scales(
    reference: np.ndarray,
    labels: np.ndarray,
    centers: np.ndarray,
) -> dict[tuple[int, int], float]:
    """Estimate an off-axis scale for each pair of local centers."""
    return _estimate_bridge_pair_scales(reference, labels, centers)


def compute_xlocal_bridge_scores(
    residuals: np.ndarray,
    parent_labels: np.ndarray,
    residuals_by_parent: dict[int, np.ndarray],
    n_local_modes: int,
    seed: int,
) -> dict[str, np.ndarray]:
    """Prefer candidates that are comparably close to two local modes."""
    score = np.zeros(residuals.shape[0], dtype=np.float32)
    primary_mode_distance = np.zeros(residuals.shape[0], dtype=np.float32)
    secondary_mode_distance = np.zeros(residuals.shape[0], dtype=np.float32)
    balance_term = np.zeros(residuals.shape[0], dtype=np.float32)
    between_term = np.zeros(residuals.shape[0], dtype=np.float32)
    off_axis_term = np.zeros(residuals.shape[0], dtype=np.float32)
    segment_position_term = np.zeros(residuals.shape[0], dtype=np.float32)
    off_axis_distance = np.zeros(residuals.shape[0], dtype=np.float32)
    segment_position = np.zeros(residuals.shape[0], dtype=np.float32)
    off_axis_scale = np.zeros(residuals.shape[0], dtype=np.float32)
    secondary_mode = np.full(residuals.shape[0], -1, dtype=np.int32)

    for parent_idx, reference in residuals_by_parent.items():
        mask = parent_labels == parent_idx
        if not np.any(mask):
            continue
        if reference.shape[0] < 2:
            continue

        centers, local_labels, center_dists = _fit_local_mode_centers(
            reference,
            n_modes=n_local_modes,
            seed=seed,
        )
        if centers.shape[0] < 2:
            continue
        pair_scales = _pairwise_off_axis_scales(reference, local_labels, centers)

        queries = residuals[mask]
        distances = np.linalg.norm(
            queries[:, None, :] - centers[None, :, :],
            axis=2,
        ).astype(np.float32)
        nearest_two = np.argsort(distances, axis=1)[:, :2]
        mode1 = nearest_two[:, 0]
        mode2 = nearest_two[:, 1]
        d1 = distances[np.arange(distances.shape[0]), mode1]
        d2 = distances[np.arange(distances.shape[0]), mode2]
        c1 = centers[mode1]
        c2 = centers[mode2]
        t_raw, d_off = _compute_segment_geometry(queries, c1, c2)

        pair_keys = [
            (int(min(a, b)), int(max(a, b)))
            for a, b in zip(mode1.tolist(), mode2.tolist(), strict=False)
        ]
        scale_off = np.array([pair_scales[key] for key in pair_keys], dtype=np.float32)
        overshoot = np.maximum(0.0, np.maximum(-t_raw, t_raw - 1.0)).astype(np.float32)

        balance = (1.0 - (np.abs(d1 - d2) / (d1 + d2 + EPS))).astype(np.float32)
        off_gate = np.exp(-d_off / (scale_off + EPS)).astype(np.float32)
        position_gate = np.exp(-(overshoot / 0.25)).astype(np.float32)
        q_between = (off_gate * position_gate).astype(np.float32)
        q_score = (balance * q_between).astype(np.float32)

        score[mask] = q_score
        primary_mode_distance[mask] = d1
        secondary_mode_distance[mask] = d2
        balance_term[mask] = balance
        between_term[mask] = q_between
        off_axis_term[mask] = off_gate
        segment_position_term[mask] = position_gate
        off_axis_distance[mask] = d_off
        segment_position[mask] = t_raw.astype(np.float32)
        off_axis_scale[mask] = scale_off
        secondary_mode[mask] = mode2.astype(np.int32)

    return {
        "score": score,
        "primary_mode_distance": primary_mode_distance,
        "secondary_mode_distance": secondary_mode_distance,
        "balance_term": balance_term,
        "between_term": between_term,
        "off_axis_term": off_axis_term,
        "segment_position_term": segment_position_term,
        "off_axis_distance": off_axis_distance,
        "segment_position": segment_position,
        "off_axis_scale": off_axis_scale,
        "secondary_mode": secondary_mode,
    }


def compute_prototype_scores(
    residuals: np.ndarray,
    parent_labels: np.ndarray,
    residuals_by_parent: dict[int, np.ndarray],
    n_local_prototypes: int,
    radius_percentile: float,
    coverage_k: int,
    seed: int,
) -> dict[str, np.ndarray]:
    """Prefer candidates that lie close to local prototype centroids.

    When ``coverage_k > 0``, a sparsity-ratio coverage term is computed per
    prototype subcluster so that candidates in already-dense prototype regions
    are softly penalized.
    """
    score = np.zeros(residuals.shape[0], dtype=np.float32)
    prototype_distance = np.zeros(residuals.shape[0], dtype=np.float32)
    prototype_radius = np.zeros(residuals.shape[0], dtype=np.float32)
    normalized_distance = np.zeros(residuals.shape[0], dtype=np.float32)
    proximity_term = np.zeros(residuals.shape[0], dtype=np.float32)
    coverage_term = np.ones(residuals.shape[0], dtype=np.float32)
    ref_sparsity_out = np.zeros(residuals.shape[0], dtype=np.float32)
    cand_sparsity_out = np.zeros(residuals.shape[0], dtype=np.float32)
    nearest_prototype = np.full(residuals.shape[0], -1, dtype=np.int32)
    prototype_count = np.zeros(residuals.shape[0], dtype=np.int32)

    for parent_idx, reference in residuals_by_parent.items():
        mask = parent_labels == parent_idx
        if not np.any(mask):
            continue
        if reference.shape[0] == 0:
            continue

        centers, local_labels, center_dists = _fit_local_mode_centers(
            reference,
            n_modes=n_local_prototypes,
            seed=seed,
        )
        n_centers = centers.shape[0]
        if n_centers == 0:
            continue

        fallback_radius = _safe_scale(center_dists)
        radii = np.zeros(n_centers, dtype=np.float32)
        counts = np.bincount(local_labels, minlength=n_centers).astype(np.int32)

        ref_sparsity_per_proto = np.zeros(n_centers, dtype=np.float32)
        for idx in range(n_centers):
            assigned = center_dists[local_labels == idx]
            radii[idx] = np.float32(
                _percentile_value(assigned, radius_percentile)
                if assigned.size
                else fallback_radius
            )
            if coverage_k > 0:
                sub_ref = reference[local_labels == idx]
                if sub_ref.shape[0] > 1:
                    sub_mean, _, _ = _compute_reference_self_knn_stats(
                        sub_ref,
                        knn_k=coverage_k,
                        metric="euclidean",
                    )
                    ref_sparsity_per_proto[idx] = float(np.mean(sub_mean))
                else:
                    ref_sparsity_per_proto[idx] = fallback_radius

        queries = residuals[mask]
        distances = np.linalg.norm(
            queries[:, None, :] - centers[None, :, :],
            axis=2,
        ).astype(np.float32)
        nearest_idx = np.argmin(distances, axis=1)
        q_distance = distances[np.arange(distances.shape[0]), nearest_idx]
        q_radius = radii[nearest_idx]
        q_normalized = (q_distance / (q_radius + EPS)).astype(np.float32)
        q_proximity = np.exp(-q_normalized).astype(np.float32)

        q_counts = counts[nearest_idx]

        if coverage_k > 0:
            q_ref_sparsity = ref_sparsity_per_proto[nearest_idx]
            cand_sparsity = np.zeros(queries.shape[0], dtype=np.float32)
            for idx in range(n_centers):
                q_mask = nearest_idx == idx
                if not np.any(q_mask):
                    continue
                sub_ref = reference[local_labels == idx]
                if sub_ref.shape[0] == 0:
                    continue
                sub_queries = queries[q_mask]
                sub_mean, _, _ = _compute_query_knn_stats(
                    sub_ref,
                    sub_queries,
                    knn_k=coverage_k,
                    metric="euclidean",
                )
                cand_sparsity[q_mask] = sub_mean

            coverage_ratio = cand_sparsity / (q_ref_sparsity + EPS)
            q_coverage = np.minimum(1.0, coverage_ratio).astype(np.float32)
        else:
            q_coverage = np.ones(queries.shape[0], dtype=np.float32)
            q_ref_sparsity = np.zeros(queries.shape[0], dtype=np.float32)
            cand_sparsity = np.zeros(queries.shape[0], dtype=np.float32)

        q_score = (q_proximity * q_coverage).astype(np.float32)

        score[mask] = q_score
        prototype_distance[mask] = q_distance
        prototype_radius[mask] = q_radius
        normalized_distance[mask] = q_normalized
        proximity_term[mask] = q_proximity
        coverage_term[mask] = q_coverage
        ref_sparsity_out[mask] = q_ref_sparsity
        cand_sparsity_out[mask] = cand_sparsity
        nearest_prototype[mask] = nearest_idx.astype(np.int32)
        prototype_count[mask] = q_counts

    return {
        "score": score,
        "prototype_distance": prototype_distance,
        "prototype_radius": prototype_radius,
        "normalized_distance": normalized_distance,
        "proximity_term": proximity_term,
        "coverage_term": coverage_term,
        "ref_sparsity": ref_sparsity_out,
        "cand_sparsity": cand_sparsity_out,
        "nearest_prototype": nearest_prototype,
        "prototype_count": prototype_count,
    }
