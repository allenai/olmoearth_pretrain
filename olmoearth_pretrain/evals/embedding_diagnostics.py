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

logger = logging.getLogger(__name__)

MAX_PAIRWISE_SAMPLES = 2048
MAX_SVD_SAMPLES = 4096
MAX_INTRA_SAMPLE_IMAGES = 256


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
