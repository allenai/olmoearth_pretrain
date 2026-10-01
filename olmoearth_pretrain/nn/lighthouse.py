"""Lighthouse inference for the joint latent-token transformer.

Tiled inference cuts a domain into training-size windows and runs each one on its
own, so a token's attention covers exactly its window, and two neighbouring pixels on
either side of a seam see contexts that differ by a whole stride. Lighthouse runs the
whole domain (or a chunk of it) in one forward pass and gives every query its OWN
training-size field of view (FOV), which slides with the query one cell at a time:
neighbouring queries share almost all their keys in every block, so there is nothing
for a seam to come from.

The FOV is a half-open box of ``W`` patch cells per axis around the query's cell,
``[r - W // 2, r - W // 2 + W)``, shifted inward at the domain edge rather than
shrunk. Every query therefore sees exactly the ``W x W`` cells a training window of
``W`` cells shows it -- the same key count, and query-key offsets inside the trained
range. Applied to the joint pattern (see :mod:`olmoearth_pretrain.nn.joint_latent`):

* token query: latents in its FOV + tokens of its own cell (unchanged);
* latent query: latents and tokens in its FOV (``latent_reads_all`` within the FOV).

Positions need no change: the encoder's RoPE depends only on query-key offsets. With a
domain exactly one window wide the FOV is the whole domain and Lighthouse reproduces
the stock forward (the parity gate in the tests).

What does change: context spreads by up to ``W // 2`` cells per block, so after the
joint blocks an output depends on inputs up to ``depth * W // 2`` cells away, which
training never showed it. Each attention is in distribution; the indirect reach is
new. The same reach makes spatial chunking exact: a chunk whose halo is at least that
reach reproduces the full-domain forward on its core (see :func:`lighthouse_reach_px`).

Implementation. Tokens and latents are laid out in spatial groups of ``G x G`` cells
(each group's tokens, then its latents, each padded to the 128-element FlexAttention
block), so a query block only touches the groups its FOVs overlap. The block mask is
built directly from that geometry with ``BlockMask.from_kv_blocks`` -- the stock
``create_block_mask`` would materialise an ``(L/128)^2`` table, ~3G entries at the
lengths a domain reaches. Every projection, norm and MLP runs in sequence chunks, so
peak memory is the residual stream plus Q/K/V and the attention output.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor

from olmoearth_pretrain.nn.encodings import (
    PositionEncoding,
    apply_3d_axial_rope,
    apply_3d_mixed_rope,
)
from olmoearth_pretrain.nn.joint_latent import build_pixel_latent_positions

if TYPE_CHECKING:
    from olmoearth_pretrain.nn.attention import Block
    from olmoearth_pretrain.nn.joint_latent import JointLatentTransformer

BLOCK = 128
# Row/col of padding elements: never inside any FOV.
_FAR = -(10**6)


@dataclass
class LighthouseSettings:
    """Inference-only switch on a :class:`JointLatentTransformer`.

    Args:
        fov_px: FOV side in pixels; must be a multiple of the patch size. The
            training/eval window is the natural value (16 for the ws16 evals).
        group_cells: Side of the spatial layout groups, in cells. None picks
            ``max(W // 2, 1)`` for ``W >= 8`` FOV cells, else ``W``.
        seq_chunk: Elements per chunk for projections, norms and MLPs.
        dense: Use a dense boolean mask + SDPA instead of FlexAttention (the
            reference path; CPU, tests, and small parity checks only).
        origin_px: ``(row, col)`` pixel offset of this chunk inside the domain. RoPE
            is relative, so this changes nothing mathematically, but mixed RoPE forms
            one fp32 angle from ``(t, row, col)`` with ``t`` ~7e3 days, whose rounding
            depends on the absolute coordinates (~1e-4 differences). Giving every
            chunk its domain coordinates makes chunked runs reproduce the
            full-domain forward to fp32 precision.
        mask_impl: ``"packed"`` (default) evaluates the FOV rule from two packed
            int64 codes per query-key pair; ``"gather"`` reads ten per-element
            tensors (the reference; much slower, the mask runs for every score).
        attn_dtype: Dtype Q/K/V are cast to for the FlexAttention kernel. Under bf16
            autocast the mixed-RoPE output is fp32, which would otherwise run the
            kernel in fp32. None keeps whatever the projections + RoPE return.
        profile: Synchronize and record per-phase seconds (attention vs the rest)
            in ``last_lighthouse_stats``.
    """

    fov_px: int
    group_cells: int | None = None
    seq_chunk: int = 1 << 18
    dense: bool = False
    origin_px: tuple[int, int] = (0, 0)
    mask_impl: str = "packed"
    attn_dtype: str | None = "bfloat16"
    profile: bool = False


def lighthouse_reach_px(
    fov_px: int, patch_size: int, joint_depth: int, max_patch_size: int = 0
) -> int:
    """Pixels an output can depend on in each direction (the exact chunk halo).

    The FOV extends ``W // 2`` cells back and ``W - W // 2 - 1`` forward, so the
    joint blocks grow the receptive field by at most ``W // 2`` cells each. Below
    ``max_patch_size`` the flexi patch embedding resamples the input and leaks a
    couple of pixels across patch boundaries (measured: 2 px at patch size 1 with
    ``max_patch_size`` 4), so ``max_patch_size`` is added as a margin for it.
    """
    fov_cells = fov_px // patch_size
    return joint_depth * (fov_cells // 2) * patch_size + max_patch_size


def _window_start(index: np.ndarray, fov: int, extent: int) -> np.ndarray:
    """First cell of the half-open FOV ``[s, s + fov)``, shifted inward at edges."""
    return np.clip(index - fov // 2, 0, extent - fov)


@dataclass
class _Layout:
    """Grouped element layout plus everything the mask needs, per element."""

    length: int
    token_dest: Tensor  # [N] layout index of each input token
    latent_dest: Tensor  # [M] layout index of each latent (grid order)
    is_latent: Tensor  # [L] bool
    valid: Tensor  # [L] bool (False on padding)
    cell: Tensor  # [L] long, cell id (-1 on padding)
    row: Tensor  # [L] long, cell row (_FAR on padding)
    col: Tensor  # [L] long
    row0: Tensor  # [L] long, FOV start row of this element as a query
    col0: Tensor  # [L] long
    kv_num_blocks: Tensor  # [1, 1, Q]
    kv_indices: Tensor  # [1, 1, Q, K]
    stats: dict[str, float]


def _build_layout(
    token_cells: np.ndarray,
    latent_cells: np.ndarray,
    n_h: int,
    n_w: int,
    fov: int,
    group: int,
    device: torch.device,
) -> _Layout:
    """Group tokens and latents spatially and derive the FOV block lists."""
    if fov > n_h or fov > n_w:
        raise ValueError(f"FOV of {fov} cells exceeds the {n_h}x{n_w} cell domain")
    if (token_cells < 0).any():
        raise ValueError("Lighthouse needs every token on the grid (no static tokens)")
    g_h, g_w = math.ceil(n_h / group), math.ceil(n_w / group)
    n_groups = g_h * g_w

    def group_of(cells: np.ndarray) -> np.ndarray:
        return (cells // n_w // group) * g_w + (cells % n_w) // group

    tok_group = group_of(token_cells)
    lat_group = group_of(latent_cells)
    # Stable sorts: tokens by (group, cell) keep their within-cell order; latents by
    # group keep grid order inside a group.
    tok_order = np.lexsort((token_cells, tok_group))
    lat_order = np.argsort(lat_group, kind="stable")
    tok_count = np.bincount(tok_group, minlength=n_groups)
    lat_count = np.bincount(lat_group, minlength=n_groups)
    tok_pad = -(-tok_count // BLOCK) * BLOCK
    lat_pad = -(-lat_count // BLOCK) * BLOCK
    seg = tok_pad + lat_pad
    start = np.concatenate([[0], np.cumsum(seg)[:-1]])
    length = int(seg.sum())

    def ranks(sorted_groups: np.ndarray, counts: np.ndarray) -> np.ndarray:
        firsts = np.concatenate([[0], np.cumsum(counts)[:-1]])
        return np.arange(sorted_groups.size) - firsts[sorted_groups]

    token_dest = np.empty(token_cells.size, dtype=np.int64)
    sg = tok_group[tok_order]
    token_dest[tok_order] = start[sg] + ranks(sg, tok_count)
    latent_dest = np.empty(latent_cells.size, dtype=np.int64)
    sg = lat_group[lat_order]
    latent_dest[lat_order] = start[sg] + tok_pad[sg] + ranks(sg, lat_count)

    is_latent = np.zeros(length, dtype=bool)
    is_latent[latent_dest] = True
    cell = np.full(length, -1, dtype=np.int64)
    cell[token_dest] = token_cells
    cell[latent_dest] = latent_cells
    valid = cell >= 0
    row = np.where(valid, cell // n_w, _FAR)
    col = np.where(valid, cell % n_w, _FAR)
    row0 = np.where(valid, _window_start(np.maximum(row, 0), fov, n_h), 0)
    col0 = np.where(valid, _window_start(np.maximum(col, 0), fov, n_w), 0)

    # --- KV block lists -------------------------------------------------------
    # Group -> block ranges.
    tok_blk0 = start // BLOCK
    tok_nblk = tok_pad // BLOCK
    lat_blk0 = (start + tok_pad) // BLOCK
    lat_nblk = lat_pad // BLOCK
    # Groups each group's FOVs can reach (FOV starts are monotone in the cell index).
    gr = np.arange(g_h)
    gc = np.arange(g_w)
    r_lo = _window_start(gr * group, fov, n_h)
    r_hi = _window_start(np.minimum((gr + 1) * group, n_h) - 1, fov, n_h) + fov - 1
    c_lo = _window_start(gc * group, fov, n_w)
    c_hi = _window_start(np.minimum((gc + 1) * group, n_w) - 1, fov, n_w) + fov - 1
    reach_r = [range(r_lo[i] // group, r_hi[i] // group + 1) for i in range(g_h)]
    reach_c = [range(c_lo[j] // group, c_hi[j] // group + 1) for j in range(g_w)]

    # Own-cell token span per token element, for token q-blocks.
    n_cells = n_h * n_w
    cell_first = np.full(n_cells, length, dtype=np.int64)
    cell_last = np.full(n_cells, -1, dtype=np.int64)
    np.minimum.at(cell_first, token_cells, token_dest)
    np.maximum.at(cell_last, token_cells, token_dest)

    n_q = length // BLOCK
    rows_kv: list[np.ndarray] = [None] * n_q  # type: ignore[list-item]
    for g in range(n_groups):
        i, j = divmod(g, g_w)
        reach = [a * g_w + b for a in reach_r[i] for b in reach_c[j]]
        lat_blocks = np.concatenate(
            [np.arange(lat_blk0[h], lat_blk0[h] + lat_nblk[h]) for h in reach]
        )
        tok_blocks = np.concatenate(
            [np.arange(tok_blk0[h], tok_blk0[h] + tok_nblk[h]) for h in reach]
        )
        latent_row = np.concatenate([tok_blocks, lat_blocks])
        for qb in range(lat_blk0[g], lat_blk0[g] + lat_nblk[g]):
            rows_kv[qb] = latent_row
        for qb in range(tok_blk0[g], tok_blk0[g] + tok_nblk[g]):
            cells_here = cell[qb * BLOCK : (qb + 1) * BLOCK]
            cells_here = cells_here[cells_here >= 0]
            lo = cell_first[cells_here].min() // BLOCK
            hi = cell_last[cells_here].max() // BLOCK
            own = np.arange(min(lo, qb), max(hi, qb) + 1)
            rows_kv[qb] = np.concatenate([own, lat_blocks])
    n_kv = np.array([r.size for r in rows_kv], dtype=np.int32)
    kv_indices = np.zeros((n_q, int(n_kv.max())), dtype=np.int32)
    for qb, r in enumerate(rows_kv):
        kv_indices[qb, : r.size] = r

    def t(a: np.ndarray, dtype: torch.dtype = torch.long) -> Tensor:
        return torch.from_numpy(a).to(device=device, dtype=dtype)

    stats = {
        "elements": float(length),
        "padding_frac": float(1 - valid.mean()),
        "q_blocks": float(n_q),
        "kv_blocks_mean": float(n_kv.mean()),
        "kv_blocks_max": float(n_kv.max()),
        "block_density": float(n_kv.sum() / (n_q * n_q)),
    }
    return _Layout(
        length=length,
        token_dest=t(token_dest),
        latent_dest=t(latent_dest),
        is_latent=t(is_latent, torch.bool),
        valid=t(valid, torch.bool),
        cell=t(cell),
        row=t(row),
        col=t(col),
        row0=t(row0),
        col0=t(col0),
        kv_num_blocks=t(n_kv, torch.int32)[None, None],
        kv_indices=t(kv_indices, torch.int32)[None, None],
        stats=stats,
    )


def _fov_rule(lay: _Layout, fov: int, q: Tensor, kv: Tensor) -> Tensor:
    """Whether query ``q`` may attend key ``kv`` (elementwise, broadcastable)."""
    in_fov = (
        (lay.row[kv] >= lay.row0[q])
        & (lay.row[kv] < lay.row0[q] + fov)
        & (lay.col[kv] >= lay.col0[q])
        & (lay.col[kv] < lay.col0[q] + fov)
    )
    token_rule = torch.where(lay.is_latent[kv], in_fov, lay.cell[q] == lay.cell[kv])
    rule = torch.where(lay.is_latent[q], in_fov, token_rule)
    # Padding rows attend only to themselves, so no row is empty.
    return (lay.valid[q] & lay.valid[kv] & rule) | (q == kv)


def lighthouse_dense_mask(lay: _Layout, fov: int) -> Tensor:
    """Dense ``[L, L]`` reference mask (small domains only)."""
    idx = torch.arange(lay.length, device=lay.valid.device)
    return _fov_rule(lay, fov, idx[:, None], idx[None, :])


_BITS = 14  # rows/cols < 16384 cells
_M = (1 << _BITS) - 1


def _packed_codes(lay: _Layout) -> tuple[Tensor, Tensor]:
    """Per-element int64 codes so the mask needs two reads per score, not ten.

    Key code: ``row | col << 14 | is_latent << 28 | valid << 29``. Query code:
    ``row0 | col0 << 14 | row << 28 | col << 42 | is_latent << 56 | valid << 57``.
    Padding rows/cols are stored as 0 with valid=0 (the rule is gated on validity).
    """
    row = torch.where(lay.valid, lay.row, 0)
    col = torch.where(lay.valid, lay.col, 0)
    lat = lay.is_latent.long()
    val = lay.valid.long()
    kv_code = row | (col << _BITS) | (lat << (2 * _BITS)) | (val << (2 * _BITS + 1))
    q_code = (
        lay.row0
        | (lay.col0 << _BITS)
        | (row << (2 * _BITS))
        | (col << (3 * _BITS))
        | (lat << (4 * _BITS))
        | (val << (4 * _BITS + 1))
    )
    return q_code, kv_code


def _packed_rule(fov: int, qc: Tensor, kc: Tensor, q: Tensor, kv: Tensor) -> Tensor:
    """:func:`_fov_rule` from the packed codes of one query and one key."""
    r0, c0 = qc & _M, (qc >> _BITS) & _M
    qr, qcol = (qc >> (2 * _BITS)) & _M, (qc >> (3 * _BITS)) & _M
    q_lat, q_val = (qc >> (4 * _BITS)) & 1, (qc >> (4 * _BITS + 1)) & 1
    kr, kcol = kc & _M, (kc >> _BITS) & _M
    k_lat, k_val = (kc >> (2 * _BITS)) & 1, (kc >> (2 * _BITS + 1)) & 1
    in_fov = (kr >= r0) & (kr < r0 + fov) & (kcol >= c0) & (kcol < c0 + fov)
    same_cell = (kr == qr) & (kcol == qcol)
    # latent query: FOV; token query: latent keys in FOV, token keys of its own cell.
    rule = torch.where((q_lat | k_lat) == 1, in_fov, same_cell)
    return ((q_val & k_val) == 1) & rule | (q == kv)


def _flex_block_mask(lay: _Layout, fov: int, mask_impl: str = "packed") -> Any:
    from torch.nn.attention.flex_attention import BlockMask

    if mask_impl == "packed":
        q_code, kv_code = _packed_codes(lay)

        def mask_mod(b: Tensor, h: Tensor, q: Tensor, kv: Tensor) -> Tensor:
            return _packed_rule(fov, q_code[q], kv_code[kv], q, kv)

    elif mask_impl == "gather":

        def mask_mod(b: Tensor, h: Tensor, q: Tensor, kv: Tensor) -> Tensor:
            return _fov_rule(lay, fov, q, kv)

    else:
        raise ValueError(f"unknown mask_impl {mask_impl!r}")

    return BlockMask.from_kv_blocks(
        lay.kv_num_blocks,
        lay.kv_indices,
        BLOCK_SIZE=BLOCK,
        mask_mod=mask_mod,
        seq_lengths=(lay.length, lay.length),
        # Inference only: the q-major tables are for the backward pass, and building
        # them transposes a dense (L/128)^2 table.
        compute_q_blocks=False,
    )


def _rope(attn: Any, x: Tensor, positions: Tensor) -> Tensor:
    """The rotation :meth:`Attention.forward` applies, for 3D RoPE modes."""
    if attn.position_encoding == PositionEncoding.MIXED_3D_ROPE:
        return apply_3d_mixed_rope(x, positions, attn.rope_mixed_freqs)
    if attn.position_encoding == PositionEncoding.AXIAL_3D_ROPE:
        return apply_3d_axial_rope(
            x,
            positions,
            base=attn.rope_base,
            temporal_dim_frac=attn.temporal_rope_dim_frac,
            temporal_base=attn.rope_temporal_base,
        )
    raise NotImplementedError(f"Lighthouse: {attn.position_encoding} not supported")


def _chunks(length: int, size: int) -> list[slice]:
    return [slice(s, min(s + size, length)) for s in range(0, length, size)]


@torch.no_grad()
def _lighthouse_block(
    blk: Block,
    x: Tensor,
    positions: Tensor,
    attend: Any,
    seq_chunk: int,
    attn_dtype: torch.dtype | None = None,
    timings: dict[str, float] | None = None,
) -> Tensor:
    """One joint block (attention + full MLP on every element), chunked, in place."""

    def tick(key: str, t0: float) -> float:
        if timings is None:
            return t0
        torch.cuda.synchronize()
        now = time.perf_counter()
        timings[key] = timings.get(key, 0.0) + now - t0
        return now

    t0 = tick("_", time.perf_counter()) if timings is not None else 0.0
    attn = blk.attn
    heads, head_dim = attn.num_heads, attn.head_dim
    length = x.shape[1]
    q_all = k_all = v_all = None
    for s in _chunks(length, seq_chunk):
        h = blk.norm1(x[:, s])
        q = rearrange(attn.q(h), "b n (h d) -> b h n d", h=heads)
        k = rearrange(attn.k(h), "b n (h d) -> b h n d", h=heads)
        v = rearrange(attn.v(h), "b n (h d) -> b h n d", h=heads)
        q, k = attn.q_norm(q), attn.k_norm(k)
        q, k = _rope(attn, q, positions[:, s]), _rope(attn, k, positions[:, s])
        if attn_dtype is not None:
            q, k, v = q.to(attn_dtype), k.to(attn_dtype), v.to(attn_dtype)
        if q_all is None:
            shape = (x.shape[0], heads, length, head_dim)
            q_all = torch.empty(shape, dtype=q.dtype, device=x.device)
            k_all = torch.empty(shape, dtype=k.dtype, device=x.device)
            v_all = torch.empty(shape, dtype=v.dtype, device=x.device)
        assert k_all is not None and v_all is not None
        q_all[:, :, s], k_all[:, :, s], v_all[:, :, s] = q, k, v
        del h, q, k, v
    t0 = tick("qkv_rope_s", t0)
    out = attend(q_all, k_all, v_all)
    del q_all, k_all, v_all
    t0 = tick("attention_s", t0)
    for s in _chunks(length, seq_chunk):
        o = rearrange(out[:, :, s], "b h n d -> b n (h d)")
        xs = x[:, s] + blk.ls1(attn.proj(o).to(x.dtype))
        x[:, s] = xs + blk.ls2(blk.mlp(blk.norm2(xs)))
    tick("proj_mlp_s", t0)
    return x


@torch.no_grad()
def lighthouse_forward(
    module: JointLatentTransformer,
    patch_tokens: Tensor,
    patch_positions: Tensor,
    visible_mask: Tensor | None,
    cell_ids: Tensor,
    spatial_grid: tuple[int, int],
    grid_extent_positions: Tensor | None = None,
    patch_size: int = 1,
    patch_spacing: float | None = None,
) -> tuple[Tensor, Tensor]:
    """Drop-in for :meth:`JointLatentTransformer.forward` under a sliding FOV.

    Same inputs and outputs; ``module.lighthouse`` holds the settings. Batch size 1
    (one domain per call). See the module docstring for the attention rule.
    """
    del grid_extent_positions  # the latent grid comes from the pixel layout
    settings: LighthouseSettings = module.lighthouse  # type: ignore[assignment]
    _check_supported(module)
    if patch_tokens.shape[0] != 1:
        raise ValueError("Lighthouse runs one domain per call (batch size 1)")
    if settings.fov_px % patch_size != 0:
        raise ValueError(f"fov_px {settings.fov_px} is not a multiple of {patch_size}")
    if patch_spacing is None:
        raise ValueError("pixel_latents requires patch_spacing")
    device = patch_tokens.device
    n_h, n_w = spatial_grid
    fov = settings.fov_px // patch_size
    group = settings.group_cells or (max(fov // 2, 1) if fov >= 8 else fov)
    stride = module.eval_latent_stride
    lat_h, lat_w = n_h * patch_size // stride, n_w * patch_size // stride
    n_tokens = patch_tokens.shape[1]

    valid_tokens = (
        visible_mask[0].bool()
        if visible_mask is not None
        else torch.ones(n_tokens, dtype=torch.bool, device=device)
    )
    if not bool(valid_tokens.all()):
        # Missing tokens would change each tile's latent time anchor (the mean
        # visible-token time); Lighthouse uses one anchor for the whole domain.
        raise NotImplementedError("Lighthouse expects every token visible (fast_pass)")

    rows = torch.arange(lat_h, device=device) * stride // patch_size
    cols = torch.arange(lat_w, device=device) * stride // patch_size
    latent_cells = (rows[:, None] * n_w + cols[None, :]).reshape(-1)
    t_layout = time.perf_counter()
    lay = _build_layout(
        cell_ids[0].long().cpu().numpy(),
        latent_cells.cpu().numpy(),
        n_h,
        n_w,
        fov,
        group,
        device,
    )
    lay.stats["layout_s"] = time.perf_counter() - t_layout
    module.last_lighthouse_stats = lay.stats  # type: ignore[attr-defined]

    latent_xy = build_pixel_latent_positions(
        1, (lat_h, lat_w), patch_size, patch_spacing, device, stride
    )
    if module.is_3d:
        # Every token is visible and every cell carries the same timesteps, so each
        # training window's anchor (the mean visible-token time) is one cell's mean.
        # Taken in float64 over one cell: a float32 mean over the whole domain (t is
        # days since 2000, ~7e3) rounds differently with the domain size, which made
        # chunked and full-domain outputs disagree at ~2e-4.
        one_cell = cell_ids[0] == cell_ids[0, 0]
        t_anchor = (
            patch_positions[0, one_cell, 0].double().mean().to(patch_positions.dtype)
        )
        latent_positions = torch.cat(
            [t_anchor.expand(latent_xy.shape[1], 1), latent_xy[0]], dim=-1
        )
    else:
        latent_positions = latent_xy[0]

    if settings.origin_px != (0, 0):
        if any(o % patch_size for o in settings.origin_px):
            raise ValueError(f"origin_px {settings.origin_px} not patch-aligned")
        shift = torch.tensor(
            settings.origin_px, dtype=patch_positions.dtype, device=device
        ) * (patch_spacing / patch_size)
        patch_positions = patch_positions.clone()
        patch_positions[..., -2:] += shift
        latent_positions = latent_positions.clone()
        latent_positions[..., -2:] += shift

    dim = patch_tokens.shape[-1]
    x = torch.zeros(1, lay.length, dim, dtype=patch_tokens.dtype, device=device)
    x[0, lay.token_dest] = patch_tokens[0]
    x[0, lay.latent_dest] = module.register[0].to(x.dtype)
    positions = torch.zeros(
        1,
        lay.length,
        patch_positions.shape[-1],
        dtype=patch_positions.dtype,
        device=device,
    )
    positions[0, lay.token_dest] = patch_positions[0]
    positions[0, lay.latent_dest] = latent_positions.to(positions.dtype)

    if settings.dense:
        mask = lighthouse_dense_mask(lay, fov)[None, None]

        def attend(q: Tensor, k: Tensor, v: Tensor) -> Tensor:
            return F.scaled_dot_product_attention(q, k, v, attn_mask=mask)

    else:
        from olmoearth_pretrain.nn.joint_latent import flex_attention_cuda

        t_mask = time.perf_counter()
        block_mask = _flex_block_mask(lay, fov, settings.mask_impl)
        lay.stats["block_mask_s"] = time.perf_counter() - t_mask

        def attend(q: Tensor, k: Tensor, v: Tensor) -> Tensor:
            return flex_attention_cuda(q, k, v, block_mask)

    # The dense path is the fp32 reference; the dtype cast is for the flex kernel.
    attn_dtype = (
        getattr(torch, settings.attn_dtype)
        if settings.attn_dtype and not settings.dense
        else None
    )
    timings: dict[str, float] | None = {} if settings.profile else None
    for blk in module.joint_blocks:
        x = _lighthouse_block(
            blk, x, positions, attend, settings.seq_chunk, attn_dtype, timings
        )
    if timings is not None:
        timings.pop("_", None)
        lay.stats.update(timings)

    latents = x[:, lay.latent_dest]
    del x
    out = module.norm(latents)
    out = rearrange(out, "b (h w) d -> b h w d", h=lat_h, w=lat_w)
    return out, latent_xy


def _check_supported(module: JointLatentTransformer) -> None:
    """Refuse configurations whose semantics Lighthouse does not reproduce."""
    problems = []
    if not module.pixel_latents:
        problems.append("pixel_latents=False (patch-grid latents)")
    if not module.latent_reads_all:
        problems.append("latent_reads_all=False")
    if len(module.latent_blocks) > 0:
        problems.append("latent_only_depth > 0 (global latent self-attention)")
    if not module.token_mlp or module.token_mlp_ratio is not None:
        problems.append("token_mlp=False or token_mlp_ratio set")
    if module.latent_time_range or module.latent_spatial_range:
        problems.append("interval / footprint latents")
    if module.local_radius is not None or module.latent_radius is not None:
        problems.append("local_radius / latent_radius")
    if problems:
        raise NotImplementedError("Lighthouse does not support: " + "; ".join(problems))
