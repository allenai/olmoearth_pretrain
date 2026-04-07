"""Embedding quality diagnostics for detecting representation collapse.

Computes geometry metrics on embedding matrices to diagnose failure modes
in self-supervised pretraining (dimensional collapse, crowding, etc.).

Can be used standalone on any [N, D] embedding tensor, or integrated
into the eval pipeline via the evaluator callback.
"""

from __future__ import annotations

import logging

import torch
from torch import Tensor

logger = logging.getLogger(__name__)

DEAD_DIM_THRESHOLD = 1e-6
MAX_UNIFORMITY_SAMPLES = 2048


def effective_rank(embeddings: Tensor) -> float:
    """Effective rank via Shannon entropy of singular values.

    Returns a value between 1 (full collapse) and min(N, D) (maximally spread).
    Roy & Bhattacharyya (2007).
    """
    S = torch.linalg.svdvals(embeddings.float())
    S = S[S > 0]
    if S.numel() == 0:
        return 0.0
    p = S / S.sum()
    entropy = -(p * p.log()).sum()
    return entropy.exp().item()


def per_dim_variance_stats(embeddings: Tensor) -> dict[str, float]:
    """Per-dimension variance statistics. Dead dims indicate dimensional collapse."""
    var = embeddings.float().var(dim=0)
    return {
        "dim_var_mean": var.mean().item(),
        "dim_var_min": var.min().item(),
        "dim_var_max": var.max().item(),
        "dim_var_std": var.std().item(),
        "num_dead_dims": (var < DEAD_DIM_THRESHOLD).sum().item(),
        "frac_dead_dims": (var < DEAD_DIM_THRESHOLD).float().mean().item(),
    }


def uniformity(embeddings: Tensor, t: float = 2.0) -> float:
    """Uniformity metric (Wang & Isola 2020). More negative = more uniform."""
    z = torch.nn.functional.normalize(embeddings.float(), dim=-1)
    n = z.shape[0]
    if n > MAX_UNIFORMITY_SAMPLES:
        idx = torch.randperm(n, device=z.device)[:MAX_UNIFORMITY_SAMPLES]
        z = z[idx]
        n = MAX_UNIFORMITY_SAMPLES
    sq_dists = torch.cdist(z, z, p=2).pow(2)
    mask = torch.triu(torch.ones(n, n, device=z.device, dtype=torch.bool), diagonal=1)
    sq_dists_upper = sq_dists[mask]
    return torch.log(torch.exp(-t * sq_dists_upper).mean()).item()


def pairwise_cosine_stats(embeddings: Tensor) -> dict[str, float]:
    """Pairwise cosine similarity stats. High mean + low std = crowding."""
    z = torch.nn.functional.normalize(embeddings.float(), dim=-1)
    n = z.shape[0]
    if n > MAX_UNIFORMITY_SAMPLES:
        idx = torch.randperm(n, device=z.device)[:MAX_UNIFORMITY_SAMPLES]
        z = z[idx]
        n = MAX_UNIFORMITY_SAMPLES
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
    metrics.update(per_dim_variance_stats(embeddings))
    metrics.update(embedding_norm_stats(embeddings))

    if n >= 4:
        metrics["uniformity"] = uniformity(embeddings)
        metrics.update(pairwise_cosine_stats(embeddings))

    return metrics
