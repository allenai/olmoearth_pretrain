"""Embedding quality diagnostics for detecting representation collapse.

Computes geometry metrics on embedding matrices to diagnose failure modes
in self-supervised pretraining (dimensional collapse, crowding, etc.).

Supports two embedding shapes:
- [N, D]: image-level (classification). One embedding per sample.
- [N, P, D] or [N, H, W, D]: patch-level (segmentation). Multiple patches per sample.
  Computes global, inter-sample, and intra-sample diagnostics.

Can be used standalone on any embedding tensor, or integrated
into the eval pipeline via the evaluator callback.
"""

from __future__ import annotations

import logging

import torch
from torch import Tensor

from olmoearth_pretrain.evals.embedding_transforms import QUANTIZE_CLIP_THRESHOLD

logger = logging.getLogger(__name__)

MAX_PAIRWISE_SAMPLES = 2048
MAX_SVD_SAMPLES = 4096
MAX_INTRA_SAMPLE_IMAGES = 256
# Rows kept for the pipeline (raw -> normalized -> int8 round-trip) diagnostics.
# Bounded so a dense task's millions of patch embeddings cost a fixed amount.
MAX_PIPELINE_ROWS = 4096


def effective_rank(embeddings: Tensor) -> float:
    """Effective rank via Shannon entropy of singular values.

    Returns a value between 1 (full collapse) and min(N, D) (maximally spread).
    Roy & Bhattacharyya (2007).
    """
    n = embeddings.shape[0]
    if n > MAX_SVD_SAMPLES:
        idx = torch.randperm(n, device=embeddings.device)[:MAX_SVD_SAMPLES]
        embeddings = embeddings[idx]
    S = torch.linalg.svdvals(embeddings.float())
    S = S[S > 0]
    if S.numel() == 0:
        return 0.0
    p = S / S.sum()
    entropy = -(p * p.log()).sum()
    return entropy.exp().item()


def uniformity(embeddings: Tensor, t: float = 2.0) -> float:
    """Uniformity metric (Wang & Isola 2020). More negative = more uniform."""
    z = torch.nn.functional.normalize(embeddings.float(), dim=-1)
    n = z.shape[0]
    if n > MAX_PAIRWISE_SAMPLES:
        idx = torch.randperm(n, device=z.device)[:MAX_PAIRWISE_SAMPLES]
        z = z[idx]
        n = MAX_PAIRWISE_SAMPLES
    sq_dists = torch.cdist(z, z, p=2).pow(2)
    mask = torch.triu(torch.ones(n, n, device=z.device, dtype=torch.bool), diagonal=1)
    sq_dists_upper = sq_dists[mask]
    return torch.log(torch.exp(-t * sq_dists_upper).mean()).item()


def pairwise_cosine_stats(embeddings: Tensor) -> dict[str, float]:
    """Pairwise cosine similarity stats. High mean + low std = crowding."""
    z = torch.nn.functional.normalize(embeddings.float(), dim=-1)
    n = z.shape[0]
    if n > MAX_PAIRWISE_SAMPLES:
        idx = torch.randperm(n, device=z.device)[:MAX_PAIRWISE_SAMPLES]
        z = z[idx]
        n = MAX_PAIRWISE_SAMPLES
    sim = z @ z.T
    mask = torch.triu(torch.ones(n, n, device=z.device, dtype=torch.bool), diagonal=1)
    sims = sim[mask]
    return {
        "cosine_sim_mean": sims.mean().item(),
        "cosine_sim_std": sims.std().item(),
        "cosine_sim_min": sims.min().item(),
        "cosine_sim_max": sims.max().item(),
    }


def embedding_norm_stats(embeddings: Tensor) -> dict[str, float]:
    """L2 norm statistics across samples."""
    norms = embeddings.float().norm(dim=-1)
    return {
        "norm_mean": norms.mean().item(),
        "norm_std": norms.std().item(),
        "norm_min": norms.min().item(),
        "norm_max": norms.max().item(),
    }


def compute_embedding_diagnostics(embeddings: Tensor) -> dict[str, float]:
    """Compute all embedding quality diagnostics on [N, D] embeddings."""
    if embeddings.ndim != 2:
        raise ValueError(f"Expected 2D embeddings [N, D], got shape {embeddings.shape}")
    n, d = embeddings.shape
    if n < 2:
        logger.warning("Need at least 2 samples for embedding diagnostics")
        return {}

    metrics: dict[str, float] = {}
    metrics["effective_rank"] = effective_rank(embeddings)
    metrics["embedding_dim"] = float(d)
    metrics["num_samples"] = float(n)
    metrics.update(embedding_norm_stats(embeddings))

    if n >= 4:
        metrics["uniformity"] = uniformity(embeddings)
        metrics.update(pairwise_cosine_stats(embeddings))

    return metrics


def flatten_rows(embeddings: Tensor) -> Tensor:
    """Collapse ``[N, ..., D]`` to a ``[rows, D]`` float32 matrix."""
    return embeddings.reshape(-1, embeddings.shape[-1]).float()


def sample_row_indices(
    num_rows: int, max_rows: int = MAX_PIPELINE_ROWS, seed: int = 0
) -> Tensor | None:
    """Deterministic row subsample, or None when everything fits.

    Shared across the stages of one pipeline so raw / normalized / round-tripped
    views stay row-aligned and can be compared elementwise.
    """
    if num_rows <= max_rows:
        return None
    generator = torch.Generator().manual_seed(seed)
    return torch.randperm(num_rows, generator=generator)[:max_rows]


def per_dim_stats(rows: Tensor) -> dict[str, float]:
    """Per-dimension scale and shared-offset stats on ``[rows, D]``.

    ``dim_std_ratio`` is the anisotropy a KNN distance or a probe without
    BatchNorm sees directly; ``mean_component_ratio`` is the share of a typical
    embedding's magnitude that is a constant offset every embedding carries
    (0 = centered, ->1 = every vector points the same way, which compresses
    cosine similarities and drives KNN hubness).
    """
    dim_std = rows.std(dim=0)
    mean_vec = rows.mean(dim=0)
    mean_norm = rows.norm(dim=-1).mean()
    return {
        "dim_std_mean": dim_std.mean().item(),
        "dim_std_min": dim_std.min().item(),
        "dim_std_max": dim_std.max().item(),
        "dim_std_ratio": (dim_std.max() / dim_std.min().clamp_min(1e-12)).item(),
        "mean_component_ratio": ((mean_vec.norm() / mean_norm.clamp_min(1e-12)).item()),
    }


def quantization_clip_stats(rows: Tensor) -> dict[str, float]:
    """How much of ``rows`` the legacy int8 power scheme would saturate.

    ``clip_fraction`` counts coordinates past ``QUANTIZE_CLIP_THRESHOLD`` (which
    all map to the same int8 code, so their magnitude is erased);
    ``clip_fraction_rows`` counts embeddings losing at least one coordinate.
    """
    magnitude = rows.abs()
    clipped = magnitude > QUANTIZE_CLIP_THRESHOLD
    return {
        "clip_fraction": clipped.float().mean().item(),
        "clip_fraction_rows": clipped.any(dim=-1).float().mean().item(),
        "abs_max": magnitude.max().item(),
        "abs_p999": torch.quantile(magnitude.flatten().float(), 0.999).item(),
    }


def round_trip_stats(before: Tensor, after: Tensor) -> dict[str, float]:
    """Damage done by a quantize -> dequantize round trip, row-aligned.

    ``roundtrip_cosine_mean`` is what a cosine KNN loses; ``roundtrip_rel_mse``
    is the squared error relative to the embedding's own energy.
    """
    before = before.float()
    after = after.float()
    cosine = torch.nn.functional.cosine_similarity(before, after, dim=-1)
    rel_mse = (after - before).pow(2).sum(dim=-1) / before.pow(2).sum(dim=-1).clamp_min(
        1e-12
    )
    return {
        "roundtrip_cosine_mean": cosine.mean().item(),
        "roundtrip_cosine_min": cosine.min().item(),
        "roundtrip_rel_mse": rel_mse.mean().item(),
    }


def compute_geometry_diagnostics(rows: Tensor) -> dict[str, float]:
    """Geometry of a ``[rows, D]`` embedding matrix as the probes consume it.

    ``effective_rank`` is uncentered (a large shared offset alone can pin it
    near 1); ``effective_rank_centered`` reports the spread that survives
    removing that offset, so the two together separate "collapsed" from
    "off-center".
    """
    if rows.shape[0] < 2:
        logger.warning("Need at least 2 rows for geometry diagnostics")
        return {}
    metrics: dict[str, float] = {
        "effective_rank": effective_rank(rows),
        "effective_rank_centered": effective_rank(rows - rows.mean(dim=0)),
        "embedding_dim": float(rows.shape[-1]),
    }
    metrics.update(embedding_norm_stats(rows))
    metrics.update(per_dim_stats(rows))
    metrics.update(quantization_clip_stats(rows))
    if rows.shape[0] >= 4:
        metrics.update(pairwise_cosine_stats(rows))
    return metrics


def compute_pipeline_diagnostics(
    raw: Tensor,
    normalized: Tensor | None = None,
    round_tripped: Tensor | None = None,
    seed: int = 0,
) -> dict[str, float]:
    """Diagnostics along one split's extract -> normalize -> int8 pipeline.

    Emits ``raw_*`` for the embeddings as the model produced them, ``norm_*``
    for the normalized ones (omitted when no normalization ran, since the
    numbers would be identical), and ``roundtrip_*`` comparing what the probe
    actually receives against its pre-quantization input.

    All three views are subsampled with the same row indices, so the round-trip
    comparison is elementwise. Each may be passed at full size; only the
    subsample is materialized.
    """
    raw_rows = flatten_rows(raw)
    idx = sample_row_indices(raw_rows.shape[0], seed=seed)
    if idx is not None:
        raw_rows = raw_rows[idx]

    metrics = {f"raw_{k}": v for k, v in compute_geometry_diagnostics(raw_rows).items()}

    probe_input = raw_rows
    if normalized is not None:
        norm_rows = flatten_rows(normalized)
        if idx is not None:
            norm_rows = norm_rows[idx]
        metrics.update(
            {f"norm_{k}": v for k, v in compute_geometry_diagnostics(norm_rows).items()}
        )
        probe_input = norm_rows

    if round_tripped is not None:
        rt_rows = flatten_rows(round_tripped)
        if idx is not None:
            rt_rows = rt_rows[idx]
        metrics.update(round_trip_stats(probe_input, rt_rows))
    return metrics


def _compute_intra_sample_diagnostics(embeddings: Tensor) -> dict[str, float]:
    """Compute per-image patch diagnostics, averaged across images.

    Args:
        embeddings: [N, P, D] tensor where P is patches per image.

    Measures whether patches within an image are diverse (good for segmentation)
    or collapsed (all patches identical = segmentation impossible).
    """
    n, p, d = embeddings.shape
    if p < 2:
        logger.warning("Need at least 2 patches per image for intra-sample diagnostics")
        return {}

    num_images = min(n, MAX_INTRA_SAMPLE_IMAGES)
    if num_images < n:
        idx = torch.randperm(n, device=embeddings.device)[:num_images]
        embeddings = embeddings[idx]

    # Batch cosine sim: normalize then bmm → [num_images, P, P]
    z = torch.nn.functional.normalize(embeddings.float(), dim=-1)
    sim_matrices = torch.bmm(z, z.transpose(1, 2))
    tri_mask = torch.triu(
        torch.ones(p, p, device=z.device, dtype=torch.bool), diagonal=1
    )

    cosine_means = []
    cosine_stds = []
    for i in range(num_images):
        sims = sim_matrices[i][tri_mask]
        cosine_means.append(sims.mean().item())
        cosine_stds.append(sims.std().item())

    # Batch norm std
    norms = embeddings.float().norm(dim=-1)  # [num_images, P]
    norm_stds = norms.std(dim=1)  # [num_images]

    metrics: dict[str, float] = {
        "norm_std": norm_stds.mean().item(),
        "num_patches": float(p),
        "num_images_sampled": float(num_images),
    }
    if cosine_means:
        metrics["cosine_sim_mean"] = sum(cosine_means) / len(cosine_means)
        metrics["cosine_sim_std"] = sum(cosine_stds) / len(cosine_stds)
    return metrics


def compute_spatial_embedding_diagnostics(embeddings: Tensor) -> dict[str, float]:
    """Compute diagnostics for spatial (patch-level) embeddings.

    Accepts [N, *, D] where * is one or more spatial dims (e.g. [N, H, W, D]
    or [N, P, D]). Returns metrics with flat prefixes (global_, inter_, intra_)
    to avoid deep nesting in wandb.
    """
    if embeddings.ndim < 3:
        raise ValueError(
            f"Expected 3+ dim embeddings [N, *, D], got shape {embeddings.shape}"
        )

    n = embeddings.shape[0]
    d = embeddings.shape[-1]
    patches = embeddings.reshape(n, -1, d)
    p = patches.shape[1]

    if n < 2:
        logger.warning("Need at least 2 samples for spatial embedding diagnostics")
        return {}

    metrics: dict[str, float] = {}

    # Global: flatten all patches, subsample if huge
    flat = patches.reshape(-1, d)
    if flat.shape[0] > MAX_SVD_SAMPLES:
        idx = torch.randperm(flat.shape[0], device=flat.device)[:MAX_SVD_SAMPLES]
        flat = flat[idx]
    for k, v in compute_embedding_diagnostics(flat).items():
        metrics[f"global_{k}"] = v

    # Inter-sample: mean pool patches per image -> [N, D]
    pooled = patches.float().mean(dim=1)
    for k, v in compute_embedding_diagnostics(pooled).items():
        metrics[f"inter_{k}"] = v

    # Intra-sample: per-image patch diversity
    if p >= 2:
        for k, v in _compute_intra_sample_diagnostics(patches).items():
            metrics[f"intra_{k}"] = v

    return metrics
