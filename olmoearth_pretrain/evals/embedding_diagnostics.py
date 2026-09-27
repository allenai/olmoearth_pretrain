"""Embedding quality diagnostics for detecting representation collapse and tiling artifacts.

Computes geometry metrics on embedding matrices to diagnose failure modes
in self-supervised pretraining (dimensional collapse, crowding, etc.).

Also detects spatial tiling/striping artifacts (see GitHub issue #499) by measuring
row/column variance anisotropy and periodic energy in the Fourier domain.

Supports two embedding shapes:
- [N, D]: image-level (classification). One embedding per sample.
- [N, P, D] or [N, H, W, D]: patch-level (segmentation). Multiple patches per sample.
  Computes global, inter-sample, and intra-sample diagnostics.

Can be used standalone on any embedding tensor, or integrated
into the eval pipeline via the evaluator callback.
"""

from __future__ import annotations

import logging

import numpy as np
import torch
from sklearn.decomposition import PCA
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


# ---------------------------------------------------------------------------
# Tiling / striping artifact detection (GitHub issue #499)
# ---------------------------------------------------------------------------

MAX_TILING_SAMPLES = 64


def _row_col_variance_ratio(embeddings: Tensor) -> float:
    """Detect striping via variance of row-means vs column-means.

    Horizontal stripes → high row-mean variance relative to column-mean variance.
    Vertical stripes → high column-mean variance relative to row-mean variance.

    Args:
        embeddings: [N, H, W, D] spatial embeddings.

    Returns:
        Ratio of row-variance to col-variance (1.0 = isotropic).
    """
    emb = embeddings.float()
    row_means = emb.mean(dim=2)  # [N, H, D]
    col_means = emb.mean(dim=1)  # [N, W, D]

    row_var = row_means.var(dim=1).mean().item()
    col_var = col_means.var(dim=1).mean().item()

    return row_var / (col_var + 1e-12)


def _fourier_grid_energy(embeddings: Tensor, patch_size: int) -> dict[str, float]:
    """Detect periodic tiling artifacts via 2D FFT on the first PCA component.

    Computes the fraction of spectral energy concentrated on the
    horizontal and vertical axes of the frequency domain (excluding DC).
    Also identifies the dominant frequency and its period in pixels.

    Args:
        embeddings: [N, H, W, D] spatial embeddings (H, W are in patch space).
        patch_size: pixel size of each patch, used to convert period to pixels.

    Returns:
        fft_axis_energy_frac: fraction of energy on grid axes (~0.12 healthy, >0.25 artifacts).
        fft_dominant_period_px: period of the strongest axis frequency in pixels.
    """
    emb = embeddings.float()
    n, h, w, d = emb.shape
    flat = emb.reshape(-1, d)
    pca = PCA(n_components=1)
    pc1 = pca.fit_transform(flat.cpu().numpy())  # [N*H*W, 1]
    pc1_map = torch.from_numpy(pc1.reshape(n, h, w))  # [N, H, W]

    fft_2d = torch.fft.fft2(pc1_map, norm="ortho")
    mag = fft_2d.abs().mean(dim=0)  # [H, W]

    mag[0, 0] = 0.0

    total_energy = mag.sum().item() + 1e-12
    h_axis_energy = mag[:, 0].sum().item()
    w_axis_energy = mag[0, :].sum().item()
    axis_energy = h_axis_energy + w_axis_energy

    # Find dominant axis-aligned frequency, skipping k=1 (just the overall
    # spatial gradient) to find actual periodic artifacts.
    min_k = 2
    axis_mags = []
    for k in range(min_k, h):
        axis_mags.append((mag[k, 0].item(), h / k))
    for k in range(min_k, w):
        axis_mags.append((mag[0, k].item(), w / k))

    dominant_period_patches = 0.0
    if axis_mags:
        _, dominant_period_patches = max(axis_mags, key=lambda x: x[0])

    return {
        "fft_axis_energy_frac": axis_energy / total_energy,
        "fft_dominant_period_px": dominant_period_patches * patch_size,
    }


def compute_tiling_artifact_metrics(
    embeddings: Tensor, patch_size: int = 4
) -> dict[str, float]:
    """Compute metrics that detect spatial tiling/striping artifacts.

    Returns 3 metrics:
      - row_col_var_ratio: 1.0 = isotropic, far from 1.0 = directional stripes
      - fft_axis_energy_frac: ~0.12 = healthy, >0.25 = periodic grid artifacts
      - fft_dominant_period_px: period of strongest artifact in pixels

    Args:
        embeddings: [N, H, W, D] spatial embeddings (H, W in patch space).
        patch_size: pixel size of each patch for converting periods.

    Returns empty dict if input doesn't have spatial dimensions (H, W >= 2).
    """
    if embeddings.ndim != 4:
        logger.warning(
            "Tiling artifact metrics require [N, H, W, D] embeddings, "
            f"got shape {embeddings.shape}"
        )
        return {}

    n, h, w, _d = embeddings.shape
    if h < 2 or w < 2:
        logger.warning(f"Spatial dims too small for tiling metrics: H={h}, W={w}")
        return {}

    if n > MAX_TILING_SAMPLES:
        idx = torch.randperm(n, device=embeddings.device)[:MAX_TILING_SAMPLES]
        embeddings = embeddings[idx]

    metrics: dict[str, float] = {}

    metrics["row_col_var_ratio"] = _row_col_variance_ratio(embeddings)

    if h >= 4 and w >= 4:
        fft_stats = _fourier_grid_energy(embeddings, patch_size)
        metrics["fft_axis_energy_frac"] = fft_stats["fft_axis_energy_frac"]
        metrics["fft_dominant_period_px"] = fft_stats["fft_dominant_period_px"]

    return metrics


def pca_rgb_image(embeddings: Tensor) -> np.ndarray:
    """Render the first 3 PCA components of spatial embeddings as an RGB image.

    Takes a single image's spatial embeddings [H, W, D] and returns
    an [H, W, 3] uint8 array suitable for wandb.Image / matplotlib.

    If called with [N, H, W, D], uses the first sample.
    """
    if embeddings.ndim == 4:
        embeddings = embeddings[0]
    if embeddings.ndim != 3:
        raise ValueError(f"Expected [H, W, D] or [N, H, W, D], got {embeddings.shape}")

    h, w, d = embeddings.shape
    flat = embeddings.reshape(-1, d).float().cpu().numpy()

    n_components = min(3, d)
    pca = PCA(n_components=n_components)
    components = pca.fit_transform(flat)  # [H*W, 3]

    # Normalize each component to [0, 1]
    for i in range(n_components):
        c = components[:, i]
        cmin, cmax = c.min(), c.max()
        if cmax - cmin > 1e-8:
            components[:, i] = (c - cmin) / (cmax - cmin)
        else:
            components[:, i] = 0.5

    # Pad to 3 channels if fewer
    if n_components < 3:
        pad = np.zeros((components.shape[0], 3 - n_components), dtype=np.float32)
        components = np.concatenate([components, pad], axis=1)

    rgb = (components.reshape(h, w, 3) * 255).astype(np.uint8)
    return rgb
