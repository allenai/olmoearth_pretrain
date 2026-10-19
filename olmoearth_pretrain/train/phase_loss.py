"""Sub-token phase loss for location tokens finer than the patch tokens.

With location queries of ``s`` pixels under tokens of ``p`` pixels, each location token
sits at one of ``r = p / s`` phases per axis inside its token (at patch size 2 and
location query patch size 1: even or odd pixel). All location queries start from the
same learned query, so the model can learn a constant offset per phase, which shows up as
one-pixel stripes in the embeddings. This loss is the share of the location tokens'
variance explained by phase, over the whole batch::

    sum_c n_c * ||m_c - m||^2 / sum_i ||z_i - m||^2

(``m_c`` the mean location token of phase class ``c``, ``m`` the overall mean). Content
averages out of ``m_c`` across the batch, so only the content-independent offset is
penalised.
"""

import torch
from torch import Tensor


def phase_loss(location_tokens: Tensor, token_grid: tuple[int, int], r: int) -> Tensor:
    """Share of variance of ``location_tokens`` explained by sub-token phase.

    Args:
        location_tokens: ``[B, H * W, D]`` row-major grid with ``H = n_h * r``.
        token_grid: ``(n_h, n_w)`` patch-token grid.
        r: Location tokens per patch token along each axis (> 1).
    """
    n_h, n_w = token_grid
    rows = torch.arange(n_h * r, device=location_tokens.device) % r
    cols = torch.arange(n_w * r, device=location_tokens.device) % r
    phase = (rows[:, None] * r + cols[None, :]).reshape(-1)
    batch, num, dim = location_tokens.shape
    z = location_tokens.float().reshape(batch * num, dim)
    labels = phase.repeat(batch)
    centred = z - z.mean(0)
    sums = torch.zeros(r * r, dim, device=z.device).index_add_(0, labels, centred)
    counts = torch.bincount(labels, minlength=r * r).to(z.dtype)
    between = (sums.pow(2).sum(-1) / counts).sum()
    return between / centred.pow(2).sum().clamp(min=torch.finfo(z.dtype).tiny)
