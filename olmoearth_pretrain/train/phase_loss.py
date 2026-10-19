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
from einops import rearrange, reduce
from torch import Tensor


def phase_loss(location_tokens: Tensor, token_grid: tuple[int, int], r: int) -> Tensor:
    """Share of variance of ``location_tokens`` explained by sub-token phase.

    Args:
        location_tokens: ``[B, H * W, D]`` row-major grid with ``H = n_h * r``.
        token_grid: ``(n_h, n_w)`` patch-token grid.
        r: Location tokens per patch token along each axis (> 1).
    """
    n_h, n_w = token_grid
    # [B, n_h, r, n_w, r, D]: axes 2 and 4 are the row / column phase inside a token.
    z = rearrange(
        location_tokens.float(), "b (h i w j) d -> b h i w j d", h=n_h, i=r, w=n_w, j=r
    )
    overall_mean = reduce(z, "b h i w j d -> d", "mean")
    phase_means = reduce(z, "b h i w j d -> i j d", "mean")  # one mean per phase
    # Every phase has the same number of tokens, so the between-phase share of the
    # variance is the mean over phases over the mean over tokens.
    between = (phase_means - overall_mean).pow(2).sum(-1).mean()
    total = (z - overall_mean).pow(2).sum(-1).mean()
    return between / total.clamp(min=torch.finfo(z.dtype).tiny)
