"""Searchlight inference: every query attends within its own sliding neighborhood.

Tiled inference runs the encoder on training-size windows (16 px) and stitches them,
so a pixel's context jumps at every tile seam. Searchlight instead runs a whole domain
in one pass and gives every query the window it would see if a 16 px training window
were centred on it: ``W = neighborhood_attention_size_px / patch_size`` cells per side, sliding one cell at a
time and shifted inward (not shrunk) at the domain edge. The rule applies to all three
attentions of the ViT + Perceiver encoder (v1.3 RC, pix512):

* ViT self-attention: a token attends every token whose cell is in its box;
* Perceiver read: a latent attends every token in the box of the latent's cell;
* latent self-attention: a latent attends every latent whose cell is in that box.

Each is neighbourhood attention over a ``(rows, cols, K)`` grid with kernel
``(W, W, K)``, where ``K`` is the number of elements per cell; this needs the same
``K`` in every cell, i.e. missing data must be whole timesteps (which is how rslearn
exports it). On H100 NATTEN computes it exactly and fast; on A100 FlexAttention is
faster (see :func:`neighborhood_attention`). On CPU a dense masked reference is used
(tests and small checks only).

A domain one window wide reproduces the stock forward; larger domains are run in
chunks with a halo by :func:`embed_domain`.

How to run it, setup and measured speed / quality: ``docs/Searchlight-Inference.md``.

Installing NATTEN (not a declared dependency; what worked on our H100 nodes):

* There is no source build in our images (no CUDA compiler), so use a prebuilt wheel
  from https://whl.natten.org. Each wheel is built for ONE torch + CUDA pair, and new
  wheels only target the two most recent torch releases.
* With the locked torch (2.9.1+cu128): ``natten==0.21.5+torch290cu128``.
* Newer torch: upgrade torch AND torchvision together (torch alone breaks the env),
  then pick the matching wheel, e.g. ``natten==0.21.6+torch2110cu128`` for torch
  2.11 (the fastest we measured) or ``natten==0.21.7+torch2130cu126`` for 2.13.
* The H100 nodes run NVIDIA driver 570, which cannot load CUDA 13 builds: use the
  cu12x torch and NATTEN wheels even when cu13x ones exist.
* The fast kernels are Hopper's. On A100 NATTEN runs (Ampere kernels) but slowly,
  so ``backend="auto"`` uses FlexAttention there and NATTEN is not needed.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor

from olmoearth_pretrain.data.constants import Modality
from olmoearth_pretrain.datatypes import MaskedOlmoEarthSample, MaskValue
from olmoearth_pretrain.nn.encodings import (
    PositionEncoding,
    apply_2d_axial_rope,
    apply_2d_mixed_rope,
    apply_3d_axial_rope,
    apply_3d_mixed_rope,
)
from olmoearth_pretrain.nn.flexi_vit import (
    CompositeEncodings,
    Encoder,
    get_modalities_to_process,
    return_modalities_from_dict,
)

try:  # Searchlight on H100 only; not a declared dependency (wheels are per torch/CUDA)
    import natten
except ImportError:
    natten = None

if TYPE_CHECKING:
    from olmoearth_pretrain.nn.attention import Block


@dataclass
class SearchlightSettings:
    """Inference-only: pass as ``Encoder.forward(..., searchlight=...)``.

    Args:
        neighborhood_attention_size_px: Side of each query's attention neighborhood
            in PIXELS (the 16 px training window); a multiple of the patch size. In
            cells (tokens per side) it is this divided by the patch size, e.g. 8 at
            ps2.
        origin_px: ``(row, col)`` pixel offset of this domain in a larger one. RoPE
            is relative, so this only matters at fp32 rounding, but it keeps chunks
            of one domain consistent with each other.
        tokens_per_call: Tokens (or latents) per projection / MLP call. These ops act
            on each token independently, so this only bounds peak memory (e.g. the
            MLP hidden activation); results do not depend on it.
        compile: ``torch.compile`` the per-chunk projection and MLP math (~1.3x).
        backend: Attention kernel on GPU (see :func:`neighborhood_attention`).
    """

    neighborhood_attention_size_px: int = 16
    origin_px: tuple[int, int] = (0, 0)
    tokens_per_call: int = 1 << 18
    compile: bool = False
    backend: str = "auto"


def searchlight_reach_px(
    neighborhood_attention_size_px: int,
    patch_size: int,
    vit_depth: int,
    perceiver_depth: int,
) -> int:
    """Pixels an output can depend on in each direction (the exact chunk halo).

    Every attention spreads context by at most ``W // 2`` cells: each ViT block, the
    last read (earlier reads see the same tokens) and each latent self-attention.
    A much shorter halo is enough in practice (16 px: cosine p01 0.9999 vs exact).
    """
    return (
        (vit_depth + 1 + perceiver_depth)
        * (neighborhood_attention_size_px // patch_size // 2)
        * patch_size
    )


# ----------------------------------------------------------------- attention kernels


# torch.compile'd functions, compiled once per process (see _maybe_compiled).
_COMPILED: dict[str, Callable[..., Any]] = {}


def _box_start(i: Tensor, n: int, neighborhood_cells: int) -> Tensor:
    """First cell of the ``neighborhood_cells``-cell box around cell ``i`` of ``n``, shifted inward."""
    return (i - neighborhood_cells // 2).clamp(0, n - neighborhood_cells)


def _round_up(n: int, block: int) -> int:
    return -(-n // block) * block


def _reference_na(q: Tensor, k: Tensor, v: Tensor, neighborhood_cells: int) -> Tensor:
    """Dense masked attention implementing the Searchlight rule (CPU / tests).

    ``q`` is ``[h, w, Kq, H, D]``, ``k`` and ``v`` ``[h, w, Kk, H, D]``.
    """
    h, w, kq = q.shape[:3]
    kk = k.shape[2]

    def inside(n: int) -> Tensor:  # [query cell, key cell] along one axis
        cells = torch.arange(n, device=q.device)
        start = _box_start(cells, n, neighborhood_cells)[:, None]
        return (cells[None, :] >= start) & (cells[None, :] < start + neighborhood_cells)

    mask = (inside(h)[:, None, :, None] & inside(w)[None, :, None, :]).reshape(
        h * w, h * w
    )
    mask = mask.repeat_interleave(kq, 0).repeat_interleave(kk, 1)
    o = F.scaled_dot_product_attention(
        rearrange(q, "h w k n d -> 1 n (h w k) d"),
        rearrange(k, "h w k n d -> 1 n (h w k) d"),
        rearrange(v, "h w k n d -> 1 n (h w k) d"),
        attn_mask=mask,
    )
    return rearrange(o, "1 n (h w k) d -> h w k n d", h=h, w=w)


def _natten_na(q: Tensor, k: Tensor, v: Tensor, neighborhood_cells: int) -> Tensor:
    """:func:`_reference_na` with NATTEN.

    NATTEN needs queries and keys on the same grid, so each cell's ``Kq`` queries are
    zero-padded to groups of ``Kk`` (one group per batch entry); the kernel spans all
    ``Kk`` slots of a cell, so a query's slot does not change what it sees, and the
    padding queries' outputs are dropped.
    """
    if natten is None:
        raise ImportError(
            "Searchlight on GPU needs NATTEN: pip install the wheel matching your "
            "torch and CUDA from https://whl.natten.org (e.g. "
            "natten==0.21.7+torch2130cu126 -f https://whl.natten.org)"
        )
    h, w, kq, heads, dim = q.shape
    kk = k.shape[2]
    groups = -(-kq // kk)
    if groups * kk != kq:
        q = F.pad(q, (0, 0, 0, 0, 0, groups * kk - kq))
    q = rearrange(q, "h w (g k) n d -> g h w k n d", g=groups)
    k = k.expand(groups, *k.shape)
    v = v.expand(groups, *v.shape)
    if kk == 1:  # NATTEN rejects kernel sizes < 2; one element per cell is 2D anyway
        o = natten.na2d(
            q[:, :, :, 0],
            k[:, :, :, 0],
            v[:, :, :, 0],
            (neighborhood_cells, neighborhood_cells),
        )
        o = o[:, :, :, None]
    else:
        o = natten.na3d(q, k, v, (neighborhood_cells, neighborhood_cells, kk))
    return rearrange(o, "g h w k n d -> h w (g k) n d")[:, :, :kq]


# ------------------------------------------------- FlexAttention (GPUs before Hopper)
#
# Queries and keys are laid out row-major by cell, (row, col, k), with each cell row
# padded to a multiple of the attention block. A block of queries then lies in one
# cell row, and the keys its boxes need are one run of key blocks in each of the
# box's ``neighborhood_cells`` rows. The block tables list exactly those runs; the mask only tests
# columns.


def _flex_tables(
    h: int,
    w: int,
    kq: int,
    kk: int,
    neighborhood_cells: int,
    block: int,
    device: torch.device,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Key blocks of every query block for FlexAttention: partial and full lists.

    In the row-padded layout of :func:`_flex_na` a query block lies in one cell row
    and its queries' boxes span ``neighborhood_cells`` cell rows and one run of columns, which is
    the same run of key blocks in each of those rows. Blocks inside EVERY query's
    box are "full" and skip the mask. Returns ``(partial count, partial indices,
    full count, full indices)``.
    """
    lq, lk = _round_up(w * kq, block), _round_up(w * kk, block)
    qb = torch.arange(h * lq // block, device=device)
    row = qb // (lq // block)
    first_el = (qb % (lq // block)) * block
    last_el = (first_el + block - 1).clamp(max=w * kq - 1)
    r0 = _box_start(row, h, neighborhood_cells)
    c0_first = _box_start((first_el // kq).clamp(max=w - 1), w, neighborhood_cells)
    c0_last = _box_start(last_el // kq, w, neighborhood_cells)
    # Key-block offsets within a key row: [first, first + n) covers every query's
    # box; [full_lo, full_hi) lies inside all of them.
    first = c0_first * kk // block
    n = ((c0_last + neighborhood_cells) * kk - 1) // block - first + 1
    full_lo = -(-c0_last * kk // block)
    full_hi = (c0_first + neighborhood_cells) * kk // block
    i = torch.arange(neighborhood_cells, device=device)[None, :, None]
    offset = first[:, None, None] + torch.arange(int(n.max()), device=device)
    idx = (r0[:, None, None] + i) * (lk // block) + offset
    in_run = (offset < (first + n)[:, None, None]).expand_as(idx)
    full = (
        in_run & (offset >= full_lo[:, None, None]) & (offset < full_hi[:, None, None])
    )
    n_kb = h * lk // block

    def pack(keep: Tensor) -> tuple[Tensor, Tensor]:
        # Kept blocks first (sorted), the rest pushed to the end and zeroed.
        kept = idx.masked_fill(~keep, n_kb).flatten(1).sort(1).values
        return keep.flatten(1).sum(1), kept.masked_fill(kept == n_kb, 0)

    return (*pack(in_run & ~full), *pack(full))


def _flex_cells(
    h: int, w: int, k: int, block: int, device: torch.device
) -> tuple[Tensor, Tensor]:
    """``(row, col)`` cell of every slot of the row-padded layout (padding: col -1)."""
    j = torch.arange(_round_up(w * k, block), device=device)
    col = torch.where(j < w * k, j // k, -1)
    return torch.arange(h, device=device).repeat_interleave(j.numel()), col.repeat(h)


def _flex_mask_mod(
    q_cells: tuple[Tensor, Tensor],
    k_cells: tuple[Tensor, Tensor],
    w: int,
    neighborhood_cells: int,
) -> Callable[..., Tensor]:
    """The box rule's column test, from each slot's cell (padding keys: col -1).

    The row test is left to the block tables, which list only key blocks of the box's
    rows (each key block lies in one cell row). The mask runs for every score, so it
    is kept minimal: on A100 a ViT call took 1.5 s testing rows + cols in int64 and
    0.9 s testing int32 cols only.
    """
    q_c0 = _box_start(q_cells[1].clamp(min=0), w, neighborhood_cells).int()
    k_col = k_cells[1].int()

    def mask_mod(b: Tensor, hd: Tensor, qi: Tensor, ki: Tensor) -> Tensor:
        c0, kc = q_c0[qi], k_col[ki]
        return (kc >= c0) & (kc < c0 + neighborhood_cells)

    return mask_mod


# Block masks of the current domain, reused by every layer (see encoder_searchlight).
_BLOCK_MASKS: dict[tuple, list[tuple[slice, Any]]] = {}


def _flex_block_masks(
    h: int,
    w: int,
    kq: int,
    kk: int,
    neighborhood_cells: int,
    block: int,
    chunk: int,
    device: Any,
) -> list[tuple[slice, Any]]:
    """``(query slice, BlockMask)`` per chunk of ``chunk`` query blocks, cached."""
    from torch.nn.attention.flex_attention import BlockMask

    key = (h, w, kq, kk, neighborhood_cells, block, chunk, device)
    if key in _BLOCK_MASKS:
        return _BLOCK_MASKS[key]
    part_num, part_idx, full_num, full_idx = _flex_tables(
        h, w, kq, kk, neighborhood_cells, block, device
    )
    q_rows, q_cols = _flex_cells(h, w, kq, block, device)
    k_cells = _flex_cells(h, w, kk, block, device)
    n_qb, n_kb = part_num.numel(), k_cells[0].numel() // block
    masks = []
    for b0 in range(0, n_qb, chunk):
        b1 = min(b0 + chunk, n_qb)
        s = slice(b0 * block, b1 * block)

        def table(t: Tensor) -> Tensor:
            # Padded to the key-block count: narrower tables gave WRONG outputs
            # (torch 2.9).
            return F.pad(t[b0:b1], (0, n_kb - t.shape[1]))[None, None].int()

        block_mask = BlockMask.from_kv_blocks(
            part_num[None, None, b0:b1].int(),
            table(part_idx),
            full_num[None, None, b0:b1].int(),
            table(full_idx),
            BLOCK_SIZE=block,
            mask_mod=_flex_mask_mod(
                (q_rows[s], q_cols[s]), k_cells, w, neighborhood_cells
            ),
            seq_lengths=((b1 - b0) * block, n_kb * block),
            compute_q_blocks=False,
        )
        masks.append((s, block_mask))
    _BLOCK_MASKS[key] = masks
    return masks


def _flex_na(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    neighborhood_cells: int,
    block: int = 128,
    chunk: int = 1 << 13,
) -> Tensor:
    """:func:`_reference_na` with FlexAttention (GPUs without fast NATTEN, e.g. A100).

    See the section comment for the layout. Padding queries' outputs are dropped and
    padding keys are masked out. Queries run in chunks of ``chunk`` blocks to bound
    the size of the block tables, which are built once per domain.
    """
    from torch.nn.attention.flex_attention import flex_attention

    h, w, kq, heads, dim = q.shape
    kk = k.shape[2]
    lq, lk = _round_up(w * kq, block), _round_up(w * kk, block)

    def padded_rows(x: Tensor, length: int) -> Tensor:
        x = rearrange(x, "h w k n d -> n h (w k) d")
        return F.pad(x, (0, 0, 0, length - x.shape[2])).reshape(1, heads, -1, dim)

    qf, kf, vf = padded_rows(q, lq), padded_rows(k, lk), padded_rows(v, lk)
    attend = flex_attention
    if q.is_cuda:
        if "flex" not in _COMPILED:
            _COMPILED["flex"] = torch.compile(flex_attention, dynamic=True)
        attend = _COMPILED["flex"]
    out = torch.empty_like(qf)
    for s, block_mask in _flex_block_masks(
        h, w, kq, kk, neighborhood_cells, block, chunk, q.device
    ):
        out[:, :, s] = attend(qf[:, :, s], kf, vf, block_mask=block_mask)
    out = out.view(heads, h, lq, dim)[:, :, : w * kq]
    return rearrange(out, "n h (w k) d -> h w k n d", w=w)


def neighborhood_attention(
    q: Tensor, k: Tensor, v: Tensor, neighborhood_cells: int, backend: str = "auto"
) -> Tensor:
    """Searchlight attention of ``[h, w, Kq, H, D]`` queries over ``[h, w, Kk, H, D]``.

    ``backend``: ``"natten"``, ``"flex"``, or ``"auto"`` = NATTEN on Hopper and newer
    (fast kernels), FlexAttention on older GPUs (A100). CPU uses the reference.
    """
    if q.device.type != "cuda":
        return _reference_na(q, k, v, neighborhood_cells)
    if backend == "auto":
        hopper = torch.cuda.get_device_capability(q.device)[0] >= 9
        backend = "natten" if hopper else "flex"
    if backend == "natten":
        return _natten_na(q, k, v, neighborhood_cells)
    return _flex_na(q, k, v, neighborhood_cells)


# -------------------------------------------------------------------------- blocks


def _rope(attn: Any, x: Tensor, positions: Tensor) -> Tensor:
    """The rotation :meth:`Attention.forward` applies to ``[1, H, n, D]`` q or k."""
    mode = attn.position_encoding
    if mode == PositionEncoding.AXIAL_2D_ROPE:
        return apply_2d_axial_rope(x, positions, base=attn.rope_base)
    if mode == PositionEncoding.MIXED_2D_ROPE:
        return apply_2d_mixed_rope(x, positions, attn.rope_mixed_freqs)
    if mode == PositionEncoding.AXIAL_3D_ROPE:
        return apply_3d_axial_rope(
            x,
            positions,
            base=attn.rope_base,
            temporal_dim_frac=attn.temporal_rope_dim_frac,
            temporal_base=attn.rope_temporal_base,
        )
    if mode == PositionEncoding.MIXED_3D_ROPE:
        return apply_3d_mixed_rope(x, positions, attn.rope_mixed_freqs)
    raise NotImplementedError(f"Searchlight needs RoPE, got {mode}")


def _project(
    attn: Any, linear: Any, norm: Any, x: Tensor, positions: Tensor | None
) -> Tensor:
    """``linear(x)`` as ``[n, H, D]``, q/k-normed and rotated unless it is V."""
    y = rearrange(linear(x), "b n (h d) -> b h n d", h=attn.num_heads)
    if positions is not None:
        y = _rope(attn, norm(y), positions)
    return rearrange(y, "1 h n d -> n h d")


def _tail(blk: Any, x: Tensor, o: Tensor) -> Tensor:
    """Output projection + residual, then the MLP + residual (:meth:`Block.forward`)."""
    x = x + blk.ls1(blk.attn.proj(o).to(x.dtype))
    return x + blk.ls2(blk.mlp(blk.norm2(x))).to(x.dtype)


def _maybe_compiled(fn: Callable[..., Any], on: bool) -> Callable[..., Any]:
    if not on:
        return fn
    if fn.__name__ not in _COMPILED:
        _COMPILED[fn.__name__] = torch.compile(fn, dynamic=True)
    return _COMPILED[fn.__name__]


def _block(
    blk: Block,
    x: Tensor,
    x_pos: Tensor,
    grid: tuple[int, int, int],
    neighborhood_cells: int,
    settings: SearchlightSettings,
    keys: tuple[Callable[[slice], Tensor], Tensor, int] | None = None,
) -> None:
    """One attention block over grid-ordered elements, in place on ``x``.

    ``x`` is ``[1, h * w * K, D]`` in ``(row, col, k)`` order with RoPE positions
    ``x_pos``. ``keys`` = ``(source, positions, K)`` makes it cross-attention (the
    Perceiver read): ``source(slice)`` gives the key inputs of those key elements.
    """
    attn = blk.attn
    project = _maybe_compiled(_project, settings.compile)
    tail = _maybe_compiled(_tail, settings.compile)
    dtype = torch.bfloat16 if x.is_cuda else x.dtype

    def spans(n: int) -> list[slice]:
        return [
            slice(s, min(s + settings.tokens_per_call, n))
            for s in range(0, n, settings.tokens_per_call)
        ]

    def qkv(
        lin: Any, norm: Any, src: Callable[[slice], Tensor], pos: Any, n: int
    ) -> Tensor:
        out = torch.empty(
            n, attn.num_heads, attn.head_dim, dtype=dtype, device=x.device
        )
        for s in spans(n):
            out[s] = project(
                attn, lin, norm, src(s), None if pos is None else pos[:, s]
            )
        return out

    h, w, kq = grid
    n_q = x.shape[1]
    q = qkv(attn.q, attn.q_norm, lambda s: blk.norm1(x[:, s]), x_pos, n_q)
    if keys is None:
        key_src, k_pos, kk = (lambda s: blk.norm1(x[:, s])), x_pos, kq
    else:
        key_src, k_pos, kk = keys
    n_k = h * w * kk
    k = qkv(attn.k, attn.k_norm, key_src, k_pos, n_k)
    v = qkv(attn.v, None, key_src, None, n_k)
    o = neighborhood_attention(
        q.view(h, w, kq, *q.shape[1:]),
        k.view(h, w, kk, *k.shape[1:]),
        v.view(h, w, kk, *v.shape[1:]),
        neighborhood_cells,
        settings.backend,
    )
    del q, k, v
    o = rearrange(o, "h w k n d -> 1 (h w k) (n d)")
    for s in spans(n_q):
        x[:, s] = tail(blk, x[:, s], o[:, s])


# ------------------------------------------------------------------------- encoder


def _cell_ids(
    encoder: Encoder,
    tokens_only_dict: dict[str, Tensor],
    original_masks_dict: dict[str, Tensor],
    grid: tuple[int, int],
) -> Tensor:
    """Row-major patch cell of every token, ``[N]``, in the collapsed token order."""
    modalities = get_modalities_to_process(
        return_modalities_from_dict(tokens_only_dict), encoder.supported_modality_names
    )
    ids_dict: dict[str, Tensor] = {}
    for name in modalities:
        tokens = tokens_only_dict[name]
        if not Modality.get(name).is_spatial or tuple(tokens.shape[1:3]) != grid:
            raise NotImplementedError(
                f"Searchlight needs every modality on the {grid} patch grid ({name})"
            )
        cells = torch.arange(grid[0] * grid[1], device=tokens.device).view(*grid)
        shape = (*tokens.shape[:-1], 1)
        ids_dict[name] = cells.view(1, *grid, *[1] * (tokens.ndim - 3)).expand(shape)
    ids_dict.update(original_masks_dict)
    cell_ids, _ = encoder.collapse_and_combine_hwtc(ids_dict)
    return cell_ids[0, :, 0]


@torch.no_grad()
def encoder_searchlight(
    encoder: Encoder,
    settings: SearchlightSettings,
    tokens: Tensor,
    mask: Tensor,
    positions: Tensor,
    tokens_only_dict: dict[str, Tensor],
    original_masks_dict: dict[str, Tensor],
    modalities_to_dims_dict: dict[str, Any],
    patch_size: int,
    input_res: int,
    latent_patch_size: int | None,
) -> tuple[dict[str, Tensor], None, dict[str, Any] | None]:
    """The ViT, norm and Perceiver of :meth:`Encoder.apply_attn` in sliding neighborhoods.

    Takes ``apply_attn``'s collapsed ``tokens``, ``mask`` and RoPE ``positions`` of a
    single domain (batch size 1) and returns what ``apply_attn`` returns. Tokens that
    are not ``ONLINE_ENCODER`` are dropped (and come back as zeros).
    """
    perceiver = encoder.perceiver
    _BLOCK_MASKS.clear()  # a new domain: the previous one's block masks are stale
    if encoder.has_register_tokens:
        raise NotImplementedError("Searchlight: encoder register tokens are global")
    if settings.neighborhood_attention_size_px % patch_size:
        raise ValueError(
            f"neighborhood_attention_size_px {settings.neighborhood_attention_size_px} is not a multiple of {patch_size}"
        )
    n_h, n_w = encoder._patch_grid_hw(tokens_only_dict)
    neighborhood_cells = settings.neighborhood_attention_size_px // patch_size
    if neighborhood_cells > min(n_h, n_w):
        raise ValueError(
            f"the {n_h}x{n_w} cell domain is smaller than the neighborhood"
        )
    device = tokens.device

    # Visible tokens in (row, col, k) grid order; NATTEN needs the same k per cell.
    visible = (mask[0] == MaskValue.ONLINE_ENCODER.value).nonzero()[:, 0]
    cells = _cell_ids(encoder, tokens_only_dict, original_masks_dict, (n_h, n_w))
    counts = torch.bincount(cells[visible], minlength=n_h * n_w)
    k_tok = int(counts[0])
    if k_tok == 0 or bool((counts != k_tok).any()):
        raise NotImplementedError(
            "Searchlight needs the same number of tokens in every cell "
            "(missing data per timestep, not per pixel)"
        )
    order = visible[torch.argsort(cells[visible], stable=True)]
    # Distance between adjacent token centres in the RoPE frame (as in apply_attn).
    gsd_ratio = (
        CompositeEncodings.calculate_gsd_ratio(input_res, patch_size)
        * encoder.rope_coordinate_scale
    )
    shift = torch.tensor(settings.origin_px, device=device) * (gsd_ratio / patch_size)
    x = tokens[:, order]
    pos = positions[:, order]
    pos[..., -2:] += shift.to(pos.dtype)

    for blk in encoder.blocks:
        _block(blk, x, pos, (n_h, n_w, k_tok), neighborhood_cells, settings)
    x = encoder.norm(x)
    tokens_out = torch.zeros_like(tokens)
    tokens_out[:, order] = x.to(tokens.dtype)

    register_output = None
    if perceiver is not None:
        s = latent_patch_size or patch_size
        r = patch_size // s
        lat_positions = perceiver.build_pixel_latent_positions(
            1,
            (n_h * r, n_w * r),
            patch_size,
            gsd_ratio,
            device,
            s,
        )
        # Latents in (row, col, k) grid order: the r x r latents of each cell.
        lat_pos = rearrange(
            lat_positions[0], "(h a w b) c -> 1 (h w a b) c", h=n_h, a=r, b=r
        )
        lat_pos = lat_pos + shift.to(lat_pos.dtype)
        lat = perceiver.register.to(x.dtype).expand(1, n_h * n_w * r * r, -1).clone()
        key_pos = pos[..., -2:]  # the reads rotate over (row, col) only
        for i, (read_blk, lat_blk) in enumerate(
            zip(perceiver.read_blocks, perceiver.latent_blocks)
        ):
            if perceiver.per_depth_read_proj:
                norm, proj = perceiver.input_norms[i], perceiver.kv_projs[i]
            else:
                norm, proj = perceiver.input_norm, perceiver.kv_proj
            source = lambda sl, _n=norm, _p=proj: _p(_n(x[:, sl]))  # noqa: E731
            _block(
                read_blk,
                lat,
                lat_pos,
                (n_h, n_w, r * r),
                neighborhood_cells,
                settings,
                keys=(source, key_pos, k_tok),
            )
            _block(
                lat_blk, lat, lat_pos, (n_h, n_w, r * r), neighborhood_cells, settings
            )
        registers = rearrange(
            perceiver.norm(lat), "1 (h w a b) d -> 1 (h a) (w b) d", h=n_h, a=r, b=r
        )
        register_output = {
            "registers": registers,
            "register_positions": lat_positions,
        }
        if perceiver.student is not None:
            register_output["student_registers"] = perceiver.student(registers)

    _BLOCK_MASKS.clear()  # free the tables (~0.5 GB per chunk and attention kind)
    tokens_dict = encoder.split_and_expand_per_modality(
        tokens_out, modalities_to_dims_dict
    )
    tokens_dict.update(original_masks_dict)
    return tokens_dict, None, register_output


# -------------------------------------------------------------------------- domain


@torch.no_grad()
def embed_domain(
    encoder: Encoder,
    sample: MaskedOlmoEarthSample,
    patch_size: int,
    latent_patch_size: int | None,
    core_px: int,
    halo_px: int,
    output_key: str = "student_registers",
    settings: SearchlightSettings | None = None,
    input_res: int = 10,
) -> Tensor:
    """Searchlight embeddings of a whole domain, run as cores of ``core_px`` + halo.

    ``sample`` has batch size 1 and covers the domain. Each chunk is one Searchlight
    forward over its core plus ``halo_px`` on every side (clipped at the domain edge);
    only the core is kept. Returns ``[H / s, W / s, D]`` for latent patch size ``s``.
    """
    settings = settings or SearchlightSettings()
    s = latent_patch_size or patch_size
    if core_px % patch_size or halo_px % patch_size:
        raise ValueError("core_px and halo_px must be multiples of the patch size")
    assert sample.sentinel2_l2a is not None
    H, W = sample.sentinel2_l2a.shape[1:3]
    out: Tensor | None = None
    for r in range(0, H, core_px):
        for c in range(0, W, core_px):
            r0, c0 = max(r - halo_px, 0), max(c - halo_px, 0)
            r1, c1 = min(r + core_px + halo_px, H), min(c + core_px + halo_px, W)
            emb = encoder(
                sample.crop(slice(r0, r1), slice(c0, c1)),
                patch_size=patch_size,
                input_res=input_res,
                latent_patch_size=latent_patch_size,
                searchlight=replace(settings, origin_px=(r0, c0)),
            )[output_key][0]
            if out is None:
                out = emb.new_zeros(H // s, W // s, emb.shape[-1])
            rc, cc = min(r + core_px, H), min(c + core_px, W)
            out[r // s : rc // s, c // s : cc // s] = emb[
                (r - r0) // s : (rc - r0) // s, (c - c0) // s : (cc - c0) // s
            ]
    assert out is not None
    return out
