"""Sub-token phase bias of a latent grid finer than the token grid.

With latents at ``s`` pixels each under tokens of ``p`` pixels, every latent sits at
one of ``r = p / s`` phases along each axis inside its token (at ps2 / latent ps1:
even or odd pixel). All latents start from one shared query and differ only by RoPE,
so the read can add a content-independent offset per phase: the one-pixel stripes of
the v1.3 release. The phase ratio measures that offset as the share of the grid's
feature variance explained by the batch mean of each phase class::

    ratio = sum_c n_c * ||m_c - m||^2 / sum_i ||z_i - m||^2

(``m_c`` the mean latent of phase class ``c``, ``m`` the grand mean). It is
scale-free, 0 when every phase has the same mean, and samples' content averages out
of ``m_c`` over a batch. A sampling-noise floor of about ``(classes - 1) / n`` remains.
"""

import torch
from torch import Tensor


def _phase_ratio(z: Tensor, classes: Tensor, num_classes: int) -> Tensor:
    """Between-class share of the variance of ``z`` ``[N, D]`` for labels ``classes``."""
    grand = z.mean(0)
    centred = z - grand
    sums = torch.zeros(num_classes, z.shape[1], device=z.device, dtype=z.dtype)
    sums.index_add_(0, classes, centred)
    counts = torch.bincount(classes, minlength=num_classes).to(z.dtype)
    between = (sums.pow(2).sum(-1) / counts.clamp(min=1)).sum()
    total = centred.pow(2).sum().clamp(min=torch.finfo(z.dtype).tiny)
    return between / total


def latent_phase_stats(
    registers: Tensor, token_grid: tuple[int, int]
) -> dict[str, Tensor] | None:
    """Phase ratios of a register grid; None when latents are not finer than tokens.

    Args:
        registers: ``[B, H * W, D]`` row-major register grid (``H = n_h * r``).
        token_grid: ``(n_h, n_w)`` token grid of the same forward pass.

    Returns:
        ``phase``: ratio over the joint (row, col) phase classes (differentiable,
        the penalty); ``col`` / ``row``: ratios over the column / row phase alone;
        ``control``: the same ratio over token-column parity, which no position bias
        should separate (the noise floor). The last three are detached.
    """
    n_h, n_w = token_grid
    batch, num, dim = registers.shape
    r_sq, rem = divmod(num, n_h * n_w)
    r = int(round(r_sq**0.5))
    if rem != 0 or r * r != r_sq:
        raise ValueError(
            f"{num} registers do not tile the {n_h}x{n_w} token grid by a square factor"
        )
    if r == 1:
        return None
    height, width = n_h * r, n_w * r
    rows = torch.arange(height, device=registers.device)
    cols = torch.arange(width, device=registers.device)
    row_phase = (rows % r)[:, None].expand(height, width)
    col_phase = (cols % r)[None, :].expand(height, width)
    # Token-column parity at a fixed phase spacing; a single token column has no
    # second class, so the control is skipped there.
    col_token_parity = ((cols // r) % 2)[None, :].expand(height, width)

    z = registers.float().reshape(batch * num, dim)

    def labels(grid: Tensor) -> Tensor:
        return grid.reshape(1, -1).expand(batch, -1).reshape(-1)

    stats = {
        "phase": _phase_ratio(z, labels(row_phase * r + col_phase), r * r),
    }
    with torch.no_grad():
        zd = z.detach()
        stats["col"] = _phase_ratio(zd, labels(col_phase), r)
        stats["row"] = _phase_ratio(zd, labels(row_phase), r)
        if n_w > 1:
            stats["control"] = _phase_ratio(zd, labels(col_token_parity), 2)
    return stats
