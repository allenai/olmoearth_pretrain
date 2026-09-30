"""Geometry diagnostics for batches of embeddings (representation-collapse monitors).

Three quantities, all computed without gradients:

* **Spread ratio** ``r`` (logged as ``*_std_r``): scale each embedding to unit length, take the std of every
  dimension across the samples, average over dimensions, and multiply by ``sqrt(d)``.
  The per-dimension std of unit vectors averages at most ``1/sqrt(d)`` (reached only
  when the samples are centred and every dimension carries the same variance), so
  ``r`` lies in ``[0, 1]``. A high ``r`` rules out collapse to a few nearby points;
  it does not measure rank (a randomly rotated rank-10 cloud still scores ~0.98).
* **Mean pairwise cosine similarity** between the unit-length embeddings, excluding
  each sample's similarity with itself: 0 for random directions, 1 when every sample
  maps to the same vector. This is the direct collapse signal.
* **Effective rank** (RankMe, Garrido et al., 2023): ``exp`` of the entropy of the
  normalized singular values of the centred embedding matrix. It ranges from 1 to
  ``min(N - 1, d)`` and does not depend on the basis, so it detects dimensional
  collapse that the spread ratio misses. Needs ``N`` well above ``d`` to be
  meaningful, so it is computed on eval embeddings, not training batches.
  ``top10pc_var_share`` is the variance share of the 10 largest principal
  directions (singular vectors), not of 10 raw dimensions.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor


@torch.no_grad()
def spread_and_mean_cosine(embeddings: Tensor) -> tuple[Tensor, Tensor]:
    """Spread ratio and mean pairwise cosine similarity of a batch of embeddings.

    Args:
        embeddings: ``[N, d]`` embeddings with ``N >= 2``; any dtype and scale.

    Returns:
        ``(spread, mean_cos)`` as 0-d float32 tensors: the spread ratio in ``[0, 1]``
        and the mean cosine similarity over the ``N * (N - 1)`` ordered pairs of
        distinct samples.
    """
    if embeddings.ndim != 2 or embeddings.shape[0] < 2:
        raise ValueError("expected [N, d] embeddings with N >= 2")
    n, d = embeddings.shape
    with torch.autocast(device_type=embeddings.device.type, enabled=False):
        unit = F.normalize(embeddings.float(), dim=-1)
        spread = unit.std(dim=0, correction=0).mean() * math.sqrt(d)
        total = unit.sum(dim=0)
        mean_cos = (total.pow(2).sum() - unit.pow(2).sum()) / (n * (n - 1))
    return spread, mean_cos


@torch.no_grad()
def effective_rank(embeddings: Tensor, top_k: int = 10) -> tuple[float, float]:
    """Effective rank and top-k variance share of a set of embeddings.

    Args:
        embeddings: ``[N, d]`` embeddings; each dimension is centred before the SVD.
        top_k: Number of leading directions whose variance share is reported.

    Returns:
        ``(erank, top_k_share)``: the effective rank in ``[1, min(N - 1, d)]`` and the
        fraction of total variance carried by the ``top_k`` largest directions.
    """
    if embeddings.ndim != 2 or embeddings.shape[0] < 2:
        raise ValueError("expected [N, d] embeddings with N >= 2")
    with torch.autocast(device_type=embeddings.device.type, enabled=False):
        x = embeddings.float()
        x = x - x.mean(dim=0, keepdim=True)
        sigma = torch.linalg.svdvals(x)
        if float(sigma.sum()) <= 0.0:
            return 1.0, 1.0
        p = sigma / sigma.sum()
        p = p[p > 0]
        erank = torch.exp(-(p * p.log()).sum())
        var = sigma.pow(2)
        share = var[:top_k].sum() / var.sum()
    return float(erank), float(share)
