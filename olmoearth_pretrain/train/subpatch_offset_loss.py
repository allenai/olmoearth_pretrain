"""Sub-patch offset loss for location tokens finer than the patch tokens.

With location queries of ``s`` pixels under tokens of ``p`` pixels, each location token
sits at one of ``r = p / s`` offsets per axis inside its patch token (at patch size 2 and
location query patch size 1: even or odd pixel). All location queries start from the
same learned query, so the model can learn a constant shift per offset, which shows up as
one-pixel stripes in the embeddings. This loss is the share of the location tokens'
variance explained by their offset, over the whole batch::

    sum_c n_c * ||m_c - m||^2 / sum_i ||z_i - m||^2

(``m_c`` the mean location token at offset ``c``, ``m`` the overall mean). Content
averages out of ``m_c`` across the batch, so only the content-independent shift is
penalised.
"""

import torch
from einops import rearrange, reduce
from torch import Tensor


def subpatch_offset_loss(
    location_tokens: Tensor, token_grid: tuple[int, int], r: int
) -> Tensor:
    """Share of variance of ``location_tokens`` explained by their sub-patch offset.

    Args:
        location_tokens: ``[B, H * W, D]`` row-major grid with ``H = n_h * r``.
        token_grid: ``(n_h, n_w)`` patch-token grid.
        r: Location tokens per patch token along each axis (> 1).
    """
    n_h, n_w = token_grid
    # [B, n_h, r, n_w, r, D]: i and j are the row / column offset inside a patch token.
    z = rearrange(
        location_tokens.float(), "b (h i w j) d -> b h i w j d", h=n_h, i=r, w=n_w, j=r
    )
    overall_mean = reduce(z, "b h i w j d -> d", "mean")
    offset_means = reduce(z, "b h i w j d -> i j d", "mean")  # one mean per offset
    # Squared distance from the overall mean (summed over features), averaged over
    # offsets and over tokens. Every offset has the same number of tokens, so the ratio
    # is the between-offset share of the variance.
    between = reduce((offset_means - overall_mean) ** 2, "i j d -> i j", "sum").mean()
    total = reduce((z - overall_mean) ** 2, "b h i w j d -> b h i w j", "sum").mean()
    return between / total.clamp(min=torch.finfo(z.dtype).tiny)
