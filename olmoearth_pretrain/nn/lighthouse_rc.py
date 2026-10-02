"""Lighthouse inference for the register-bottleneck encoder (v1.3 RC, rc_pix512).

The v1.3 encoder is a ViT whose tokens attend every token of their window, followed by
a Perceiver: a grid of latents (one per token cell, or one every ``eval_latent_stride``
pixels with ``pixel_latents``) that ``[read the tokens -> self-attend]`` per depth.
Tiled inference runs that on training-size windows, so a pixel's context jumps by a
whole stride at every tile seam. Lighthouse runs a whole domain (or a large chunk of
it) in one pass and gives every query its OWN window of ``W = fov_px / patch_size``
cells, centred on its cell and sliding one cell at a time:

* ViT token query: every token whose cell lies in its FOV box;
* Perceiver read (latent query): every token in the FOV box of the latent's cell;
* latent self-attention: every latent whose cell lies in that box.

The box is half-open, ``[r - W // 2, r - W // 2 + W)``, shifted inward at the domain
edge rather than shrunk, so every query sees exactly the keys a training window shows
it. A domain one window wide reproduces the stock forward exactly (the parity test).
What training never showed: context now spreads by ``W // 2`` cells per block, so a
deep output depends on inputs far outside one window. Chunked runs therefore differ
from a full-domain run by an amount that decays with the chunk halo; see
:func:`lighthouse_rc_reach_px` for the exact halo and the AOI driver for the measured
sensitivity to a shorter one.

Implementation. Elements are laid out spatially -- tokens row-major by cell, each cell
row padded to the attention block, latents in small 2D tiles of cells -- so a block
of queries shares nearly one FOV and the keys of that FOV are a few contiguous block
runs per cell row. The FlexAttention block tables come straight from that geometry
(``BlockMask.from_kv_blocks``); blocks whose every key is inside every query's FOV
are *full* (no mask evaluated), only the FOV borders and padding go through
``mask_mod``. Keys and values are computed once per block for the whole sequence;
queries, the output projection and the MLP run in chunks, so peak memory is the
residual stream plus one layer's K/V.
"""

from __future__ import annotations

import dataclasses
import math
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor

from olmoearth_pretrain.nn.encodings import (
    PositionEncoding,
    apply_2d_axial_rope,
    apply_2d_mixed_rope,
    apply_3d_axial_rope,
    apply_3d_mixed_rope,
)
from olmoearth_pretrain.nn.joint_latent import (
    build_pixel_latent_positions,
    choose_latent_stride,
)

if TYPE_CHECKING:
    from olmoearth_pretrain.nn.attention import Block
    from olmoearth_pretrain.nn.flexi_vit import Encoder, Perceiver

_BITS = 14  # rows / cols < 16384 cells
_M = (1 << _BITS) - 1
_BIG = 1 << 40


@dataclass
class RCLighthouseSettings:
    """Inference-only switch on an :class:`Encoder` (``encoder.lighthouse``).

    Args:
        fov_px: FOV side in pixels; a multiple of the patch size. The eval window
            (16 px) is the natural value.
        block: FlexAttention block size (queries and keys).
        q_chunk: Query slots per attention call (rounded to ``block``); bounds the
            memory of Q, the attention output and the MLP intermediate.
        mlp_chunk: Slots per MLP / projection call inside a query chunk.
        latent_tile: ``(rows, cols)`` of token cells per latent layout tile. None picks
            the smallest square-ish tile with at least ``block`` latents.
        dense: Dense boolean masks + SDPA instead of FlexAttention (the reference;
            small parity checks only). Always on for CPU tensors: the compiled
            FlexAttention kernel is CUDA-only.
        origin_px: ``(row, col)`` pixel offset of this chunk in the domain. RoPE is
            relative, but mixed RoPE forms one fp32 angle from ``(t, row, col)``, so
            giving every chunk its domain coordinates keeps chunk outputs consistent
            with the full-domain forward to fp32 rounding.
        attn_dtype: Q/K/V dtype for the FlexAttention kernel (mixed RoPE returns fp32
            under autocast, which would run the kernel in fp32).
        fov_quantum: Queries share their FOV over aligned tiles of this many cells
            per side (see :func:`_fov_start`): 1 is exact Lighthouse; larger values
            make every attention block full (no mask evaluated) at the price of a
            FOV that moves in steps of ``fov_quantum`` cells and sits up to half a
            step off-centre. Must divide the FOV.
        code_dtype: Integer dtype of the mask codes (see :func:`_codes`).
        full_blocks: Declare blocks inside every query's FOV *full* (skip
            ``mask_mod``). False sends every listed block through the mask.
        pad_index_width: Pad the block-index tables to the dense table width (the
            number of KV blocks). Required with full blocks: narrower partial + full
            tables gave WRONG FlexAttention outputs on GPU (cos -0.48 vs dense,
            torch 2.9.1; scripts/tools/lighthouse_rc_flex_check.py).
        column_mask: Use the column-only mask where the layout allows it (the
            exact ViT plan; see :func:`_column_codes`). False = packed codes.
        profile: Synchronize and record per-phase seconds in ``last_lighthouse_stats``.
        return_tokens: Also return the encoded tokens. Off by default: the embedding
            is the Perceiver output, and the extra copy costs ~3 KB per token. When
            off, the returned tokens are the INPUT tokens, unchanged.
    """

    fov_px: int
    block: int = 128
    q_chunk: int = 1 << 20
    mlp_chunk: int = 1 << 18
    latent_tile: tuple[int, int] | None = None
    dense: bool = False
    origin_px: tuple[int, int] = (0, 0)
    attn_dtype: str | None = "bfloat16"
    profile: bool = False
    return_tokens: bool = False
    fov_quantum: int = 1
    column_mask: bool = True
    code_dtype: str = "int64"
    full_blocks: bool = True
    pad_index_width: bool = True


def lighthouse_rc_reach_px(
    fov_px: int,
    patch_size: int,
    vit_depth: int,
    perceiver_depth: int,
    max_patch_size: int = 0,
) -> int:
    """Pixels an output can depend on in each direction (the exact chunk halo).

    Every attention spreads context by at most ``W // 2`` cells: the ViT blocks, the
    last read (earlier reads see the same tokens) and each latent self-attention.
    ``max_patch_size`` is a margin for the flexi patch embedding, which resamples the
    input below it and leaks a couple of pixels across patch boundaries.
    """
    half = (fov_px // patch_size) // 2
    return (vit_depth + 1 + perceiver_depth) * half * patch_size + max_patch_size


# --------------------------------------------------------------------------- layout


def _fov_start(
    index: np.ndarray, fov: int, extent: int, quantum: int = 1
) -> np.ndarray:
    """First cell of the half-open FOV ``[s, s + fov)``, shifted inward at edges.

    ``quantum`` > 1: every query of a ``quantum``-aligned tile of cells shares one FOV,
    the box around the tile's centre rounded down to the quantum grid, so the FOV is
    a whole number of tiles (all attention blocks full) and sits at most
    ``quantum / 2`` cells off-centre. ``quantum`` 1 is the exact per-cell FOV.
    """
    if quantum == 1:
        return np.clip(index - fov // 2, 0, extent - fov)
    base = (index // quantum) * quantum
    start = ((base + quantum // 2 - fov // 2) // quantum) * quantum
    return np.clip(start, 0, extent - fov)


@dataclass
class _Slots:
    """Spatially grouped slot layout of one element set (tokens or latents)."""

    length: int
    dest: np.ndarray  # [n] slot of each element, input order
    row: np.ndarray  # [length] cell row, -1 on padding
    col: np.ndarray  # [length] cell col, -1 on padding
    tile_h: int
    tile_w: int
    device_dest: Tensor | None = None
    quantum: int = 1  # FOV quantum of these elements as queries (see _fov_start)

    @property
    def valid(self) -> np.ndarray:
        return self.row >= 0


def _slot_layout(
    rows: np.ndarray,
    cols: np.ndarray,
    n_w: int,
    tile: tuple[int, int],
    pad: int,
) -> _Slots:
    """Order elements by (tile, row, col, input order); pad each tile to ``pad``."""
    tile_h, tile_w = tile
    n = rows.size
    tiles_w = -(-n_w // tile_w)
    tile_id = (rows // tile_h) * tiles_w + cols // tile_w
    order = np.lexsort((np.arange(n), cols, rows, tile_id))
    n_tiles = int(tile_id.max()) + 1 if n else 0
    counts = np.bincount(tile_id, minlength=n_tiles)
    padded = -(-counts // pad) * pad
    start = np.concatenate([[0], np.cumsum(padded)[:-1]]).astype(np.int64)
    first_sorted = np.concatenate([[0], np.cumsum(counts)[:-1]])
    sorted_tiles = tile_id[order]
    dest = np.empty(n, dtype=np.int64)
    dest[order] = start[sorted_tiles] + np.arange(n) - first_sorted[sorted_tiles]
    length = int(padded.sum())
    row = np.full(length, -1, dtype=np.int64)
    col = np.full(length, -1, dtype=np.int64)
    row[dest], col[dest] = rows, cols
    return _Slots(length, dest, row, col, tile_h, tile_w)


@dataclass
class _Plan:
    """Attention plan for one (query layout, key layout) pair."""

    q_len: int
    k_len: int
    chunks: list[tuple[slice, Any]]  # (query slot slice, BlockMask or dense mask)
    stats: dict[str, float] = field(default_factory=dict)


def _block_tables(
    q: _Slots, k: _Slots, fov: int, n_h: int, n_w: int, block: int
) -> dict[str, np.ndarray]:
    """Partial and full KV block lists per query block, from the FOV geometry."""
    nq, nk = q.length // block, k.length // block

    # Query blocks: union and intersection of their valid queries' FOV boxes.
    qr = q.row.reshape(nq, block)
    qc = q.col.reshape(nq, block)
    qv = qr >= 0
    r0 = np.where(qv, _fov_start(np.maximum(qr, 0), fov, n_h, q.quantum), 0)
    c0 = np.where(qv, _fov_start(np.maximum(qc, 0), fov, n_w, q.quantum), 0)
    any_q = qv.any(1)
    u_r0 = np.where(any_q, np.where(qv, r0, _BIG).min(1), 0)
    u_r1 = np.where(any_q, np.where(qv, r0, -1).max(1) + fov, fov)
    u_c0 = np.where(any_q, np.where(qv, c0, _BIG).min(1), 0)
    u_c1 = np.where(any_q, np.where(qv, c0, -1).max(1) + fov, fov)
    i_r0 = np.where(qv, r0, -1).max(1)
    i_r1 = np.where(qv, r0, _BIG).min(1) + fov
    i_c0 = np.where(qv, c0, -1).max(1)
    i_c1 = np.where(qv, c0, _BIG).min(1) + fov
    i_r1 = np.where(any_q, i_r1, -1)  # empty intersection -> nothing is full

    # Key blocks: bounding boxes, padding, and monotone search codes per band.
    kr = k.row.reshape(nk, block)
    kc = k.col.reshape(nk, block)
    kvd = kr >= 0
    any_k = kvd.any(1)
    k_r0 = np.where(kvd, kr, _BIG).min(1)
    k_r1 = np.where(kvd, kr, -1).max(1)
    k_c0 = np.where(kvd, kc, _BIG).min(1)
    k_c1 = np.where(kvd, kc, -1).max(1)
    k_pad = ~kvd.all(1)
    band = np.where(any_k, k_r0 // k.tile_h, -1)
    if k.tile_h == 1:
        lo_col, hi_col = k_c0, k_c1
    else:  # tiles of several rows: the block's cols are not monotone, its tile's are
        lo_col = (k_c0 // k.tile_w) * k.tile_w
        hi_col = lo_col + k.tile_w - 1
    first_code = np.maximum.accumulate(np.where(any_k, band * n_w + lo_col, -1))
    last_code = np.maximum.accumulate(np.where(any_k, band * n_w + hi_col, -1))

    # Candidate block runs: one per (query block, key band the union box touches).
    b0 = u_r0 // k.tile_h
    b1 = (u_r1 - 1) // k.tile_h
    n_bands = int((b1 - b0).max()) + 1
    bands = b0[:, None] + np.arange(n_bands)[None, :]
    active = bands <= b1[:, None]
    lo = np.searchsorted(last_code, bands * n_w + u_c0[:, None], "left")
    hi = np.searchsorted(first_code, bands * n_w + u_c1[:, None], "left")
    counts = np.where(active, np.maximum(hi - lo, 0), 0).ravel()
    starts = lo.ravel()
    qb_of_run = np.repeat(np.arange(nq), n_bands)
    total = int(counts.sum())
    qb = np.repeat(qb_of_run, counts)
    run_first = np.repeat(np.cumsum(counts) - counts, counts)
    kb = np.repeat(starts, counts) + (np.arange(total) - run_first)

    # Exact filter on the union box, then full = inside the intersection, no padding.
    keep = (
        any_k[kb]
        & (k_r1[kb] >= u_r0[qb])
        & (k_r0[kb] < u_r1[qb])
        & (k_c1[kb] >= u_c0[qb])
        & (k_c0[kb] < u_c1[qb])
    )
    qb, kb = qb[keep], kb[keep]
    full = (
        # Padding queries may see a full block's keys: their outputs are discarded.
        ~k_pad[kb]
        & (k_r0[kb] >= i_r0[qb])
        & (k_r1[kb] < i_r1[qb])
        & (k_c0[kb] >= i_c0[qb])
        & (k_c1[kb] < i_c1[qb])
    )

    def pack(sel: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        q_sel, k_sel = qb[sel], kb[sel]
        num = np.bincount(q_sel, minlength=nq).astype(np.int32)
        width = max(int(num.max()) if num.size else 0, 1)
        first = np.concatenate([[0], np.cumsum(num)[:-1]])
        pos = np.arange(q_sel.size) - first[q_sel]
        idx = np.zeros((nq, width), dtype=np.int32)
        idx[q_sel, pos] = k_sel
        return num, idx

    part_num, part_idx = pack(~full)
    full_num, full_idx = pack(full)
    n_scored = part_num.sum() + full_num.sum()
    # The ideal: valid keys inside each valid query's FOV (integral image of key cells).
    counts = np.zeros((n_h + 1, n_w + 1), dtype=np.int64)
    np.add.at(counts, (k.row[k.valid] + 1, k.col[k.valid] + 1), 1)
    integral = counts.cumsum(0).cumsum(1)
    a, b = r0[qv], c0[qv]
    ideal = (
        integral[a + fov, b + fov]
        - integral[a, b + fov]
        - integral[a + fov, b]
        + integral[a, b]
    ).sum()
    return {
        "part_num": part_num,
        "part_idx": part_idx,
        "full_num": full_num,
        "full_idx": full_idx,
        "stats": np.array(
            [
                n_scored / max(nq, 1),
                full_num.sum() / max(n_scored, 1),
                # Scores computed vs scores the FOV rule needs.
                n_scored * block * block / max(ideal, 1),
            ]
        ),
    }


def _codes(
    q: _Slots,
    k: _Slots,
    fov: int,
    n_h: int,
    n_w: int,
    device: torch.device,
    code_dtype: str = "int64",
) -> tuple[Tensor, Tensor]:
    """Per-slot codes: query ``r0 | c0 << 14 | valid << 28``, key the same.

    int64 by default: int32 codes gave WRONG FlexAttention outputs on GPU (cos 0.15
    vs the dense reference) and illegal memory accesses, torch 2.9.
    """
    qv = q.valid
    r0 = np.where(qv, _fov_start(np.maximum(q.row, 0), fov, n_h, q.quantum), 0)
    c0 = np.where(qv, _fov_start(np.maximum(q.col, 0), fov, n_w, q.quantum), 0)
    q_code = r0 | (c0 << _BITS) | (qv.astype(np.int64) << (2 * _BITS))
    kv = k.valid
    k_code = (
        np.where(kv, k.row, 0)
        | (np.where(kv, k.col, 0) << _BITS)
        | (kv.astype(np.int64) << (2 * _BITS))
    )
    return (
        torch.from_numpy(q_code.astype(np.dtype(code_dtype))).to(device),
        torch.from_numpy(k_code.astype(np.dtype(code_dtype))).to(device),
    )


def _rule(fov: int, qc: Tensor, kc: Tensor) -> Tensor:
    """Key inside the query's FOV; padding queries attend anything (discarded)."""
    r0, c0, q_ok = qc & _M, (qc >> _BITS) & _M, (qc >> (2 * _BITS)) & 1
    kr, kcol, k_ok = kc & _M, (kc >> _BITS) & _M, (kc >> (2 * _BITS)) & 1
    inside = (kr >= r0) & (kr < r0 + fov) & (kcol >= c0) & (kcol < c0 + fov)
    return (q_ok == 0) | ((k_ok == 1) & inside)


def _mask_mod(fov: int, q_code: Tensor, k_code: Tensor) -> Callable[..., Tensor]:
    def mask_mod(b: Tensor, h: Tensor, qi: Tensor, ki: Tensor) -> Tensor:
        return _rule(fov, q_code[qi], k_code[ki])

    return mask_mod


_NO_COL = 1 << 24  # column code of a padding key: inside no FOV


def _column_codes(
    q: _Slots,
    k: _Slots,
    fov: int,
    n_w: int,
    block: int,
    device: torch.device,
    code_dtype: str = "int64",
) -> tuple[Tensor, Tensor] | None:
    """Column-only mask codes when every query block lies in ONE cell row.

    Then the FOV's row test is the same for the whole block and the block tables
    already enforce it (only blocks of rows inside the FOV are listed), so the mask
    only has to test the key's column against the query's FOV: one load and two
    compares per score instead of decoding packed codes. Padding queries take the
    FOV of the block's last valid query so no row is left without keys. Returns
    None when the layout does not qualify (several rows per block, or a quantum).
    """
    if q.quantum != 1 or q.tile_h != 1 or k.tile_h != 1:
        return None
    rows = q.row.reshape(-1, block)
    valid = rows >= 0
    lo = np.where(valid, rows, _BIG).min(1)
    hi = np.where(valid, rows, -1).max(1)
    if not (lo[valid.any(1)] == hi[valid.any(1)]).all():
        return None
    c0 = np.where(q.valid, _fov_start(np.maximum(q.col, 0), fov, n_w), -1)
    # Forward-fill padding queries (each row's padding follows its valid queries).
    idx = np.where(c0 >= 0, np.arange(c0.size), 0)
    np.maximum.accumulate(idx, out=idx)
    c0 = np.maximum(c0[idx], 0)
    k_col = np.where(k.valid, k.col, _NO_COL)
    return (
        torch.from_numpy(c0.astype(np.dtype(code_dtype))).to(device),
        torch.from_numpy(k_col.astype(np.dtype(code_dtype))).to(device),
    )


def _column_mask_mod(fov: int, q_c0: Tensor, k_col: Tensor) -> Callable[..., Tensor]:
    def mask_mod(b: Tensor, h: Tensor, qi: Tensor, ki: Tensor) -> Tensor:
        c0, kc = q_c0[qi], k_col[ki]
        return (kc >= c0) & (kc < c0 + fov)

    return mask_mod


def _merge_lists(
    a_num: np.ndarray, a_idx: np.ndarray, b_num: np.ndarray, b_idx: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Per-row sorted union of two (num, idx) block lists."""
    rows = [
        np.sort(np.concatenate([a_idx[i, : a_num[i]], b_idx[i, : b_num[i]]]))
        for i in range(a_num.size)
    ]
    num = np.array([r.size for r in rows], dtype=np.int32)
    idx = np.zeros((num.size, max(int(num.max()) if num.size else 0, 1)), np.int32)
    for i, r in enumerate(rows):
        idx[i, : r.size] = r
    return num, idx


def _pad_width(idx: np.ndarray, width: int) -> np.ndarray:
    """Pad a block-index table to ``width`` columns (the dense table's width)."""
    out = np.zeros((idx.shape[0], max(width, idx.shape[1])), dtype=np.int32)
    out[:, : idx.shape[1]] = idx
    return out


def _make_plan(
    q: _Slots,
    k: _Slots,
    fov: int,
    n_h: int,
    n_w: int,
    settings: RCLighthouseSettings,
    device: torch.device,
) -> _Plan:
    q_code, k_code = _codes(q, k, fov, n_h, n_w, device, settings.code_dtype)
    block = settings.block
    if settings.dense:
        mask = _rule(fov, q_code[:, None], k_code[None, :])
        return _Plan(q.length, k.length, [(slice(0, q.length), mask)])

    from torch.nn.attention.flex_attention import BlockMask

    t0 = time.perf_counter()
    tab = _block_tables(q, k, fov, n_h, n_w, block)
    columns = (
        _column_codes(q, k, fov, n_w, block, device, settings.code_dtype)
        if settings.column_mask
        else None
    )
    per_chunk = max(settings.q_chunk // block, 1)
    nq = q.length // block
    chunks = []
    for qb0 in range(0, nq, per_chunk):
        qb1 = min(qb0 + per_chunk, nq)
        qs = slice(qb0 * block, qb1 * block)
        if columns is not None:
            mask_mod = _column_mask_mod(fov, columns[0][qs], columns[1])
        else:
            mask_mod = _mask_mod(fov, q_code[qs], k_code)

        def t(a: np.ndarray) -> Tensor:
            return torch.from_numpy(np.ascontiguousarray(a)).to(device)[None, None]

        part_num, part_idx = tab["part_num"][qb0:qb1], tab["part_idx"][qb0:qb1]
        full_num, full_idx = tab["full_num"][qb0:qb1], tab["full_idx"][qb0:qb1]
        if not settings.full_blocks:
            # Every listed block through mask_mod (diagnostic / fallback).
            part_num, part_idx = _merge_lists(part_num, part_idx, full_num, full_idx)
            full_num, full_idx = np.zeros_like(part_num), np.zeros_like(part_idx)
        if settings.pad_index_width:
            nk = k.length // block
            part_idx = _pad_width(part_idx, nk)
            full_idx = _pad_width(full_idx, nk)
        bm = BlockMask.from_kv_blocks(
            t(part_num),
            t(part_idx),
            t(full_num),
            t(full_idx),
            BLOCK_SIZE=block,
            mask_mod=mask_mod,
            seq_lengths=((qb1 - qb0) * block, k.length),
            compute_q_blocks=False,
        )
        chunks.append((slice(qb0 * block, qb1 * block), bm))
    kv_per_q, full_frac, overscore = tab["stats"]
    return _Plan(
        q.length,
        k.length,
        chunks,
        {
            "kv_blocks_per_q": float(kv_per_q),
            "full_block_frac": float(full_frac),
            "scored_vs_ideal": float(overscore),
            "plan_s": time.perf_counter() - t0,
            "column_mask": float(columns is not None),
        },
    )


def _default_latent_tile(per_cell: int, block: int) -> tuple[int, int]:
    """A ``h x ~2h`` tile of cells holding at least ``block`` latents."""
    cells = -(-block // per_cell)
    h = max(math.ceil(math.sqrt(cells / 2)), 1)
    return h, -(-cells // h)


# ------------------------------------------------------------------------ compute


def _rope(attn: Any, x: Tensor, positions: Tensor) -> Tensor:
    """The rotation :meth:`Attention.forward` applies to q or k."""
    mode = attn.position_encoding
    if mode == PositionEncoding.MIXED_3D_ROPE:
        return apply_3d_mixed_rope(x, positions, attn.rope_mixed_freqs)
    if mode == PositionEncoding.AXIAL_3D_ROPE:
        return apply_3d_axial_rope(
            x,
            positions,
            base=attn.rope_base,
            temporal_dim_frac=attn.temporal_rope_dim_frac,
            temporal_base=attn.rope_temporal_base,
        )
    if mode == PositionEncoding.AXIAL_2D_ROPE:
        return apply_2d_axial_rope(x, positions, base=attn.rope_base)
    if mode == PositionEncoding.MIXED_2D_ROPE:
        return apply_2d_mixed_rope(x, positions, attn.rope_mixed_freqs)
    raise NotImplementedError(f"Lighthouse: {mode} not supported")


def _spans(length: int, size: int) -> list[slice]:
    return [slice(s, min(s + size, length)) for s in range(0, length, size)]


class _Runner:
    """Shared attention / profiling machinery for one forward."""

    def __init__(self, settings: RCLighthouseSettings) -> None:
        self.settings = settings
        self.dtype = (
            getattr(torch, settings.attn_dtype)
            if settings.attn_dtype and not settings.dense
            else None
        )
        self.timings: dict[str, float] | None = {} if settings.profile else None
        self._t = time.perf_counter()

    def tick(self, key: str) -> None:
        if self.timings is None:
            return
        torch.cuda.synchronize()
        now = time.perf_counter()
        self.timings[key] = self.timings.get(key, 0.0) + now - self._t
        self._t = now

    def attend(self, q: Tensor, k: Tensor, v: Tensor, mask: Any) -> Tensor:
        if self.settings.dense:
            return F.scaled_dot_product_attention(q, k, v, attn_mask=mask[None, None])
        from olmoearth_pretrain.nn.joint_latent import flex_attention_cuda

        return flex_attention_cuda(q, k, v, mask)

    def heads(self, attn: Any, t: Tensor) -> Tensor:
        return rearrange(t, "b n (h d) -> b h n d", h=attn.num_heads)

    def kv(
        self,
        blk: Block,
        source: Callable[[slice], Tensor],
        positions: Tensor,
        length: int,
    ) -> tuple[Tensor, Tensor]:
        """Rotated K and V of every key slot, in chunks."""
        attn = blk.attn
        k_all = v_all = None
        for s in _spans(length, self.settings.mlp_chunk):
            y = source(s)
            k = attn.k_norm(self.heads(attn, attn.k(y)))
            v = self.heads(attn, attn.v(y))
            k = _rope(attn, k, positions[:, s])
            if self.dtype is not None:
                k, v = k.to(self.dtype), v.to(self.dtype)
            if k_all is None:
                shape = (y.shape[0], attn.num_heads, length, attn.head_dim)
                k_all = torch.empty(shape, dtype=k.dtype, device=y.device)
                v_all = torch.empty(shape, dtype=v.dtype, device=y.device)
            assert v_all is not None
            k_all[:, :, s], v_all[:, :, s] = k, v
        assert k_all is not None and v_all is not None
        return k_all, v_all

    def block(
        self,
        blk: Block,
        x: Tensor,
        x_positions: Tensor,
        k_all: Tensor,
        v_all: Tensor,
        plan: _Plan,
    ) -> None:
        """Queries, attention, projection and MLP of ``blk``, per chunk, in place."""
        attn = blk.attn
        for qs, mask in plan.chunks:
            h = blk.norm1(x[:, qs])
            q = attn.q_norm(self.heads(attn, attn.q(h)))
            q = _rope(attn, q, x_positions[:, qs])
            if self.dtype is not None:
                q = q.to(self.dtype)
            del h
            self.tick("q_s")
            o = self.attend(q, k_all, v_all, mask)
            del q
            self.tick("attention_s")
            o = rearrange(o, "b h n d -> b n (h d)")
            for s in _spans(o.shape[1], self.settings.mlp_chunk):
                g = slice(qs.start + s.start, qs.start + s.stop)
                xs = x[:, g] + blk.ls1(attn.proj(o[:, s]).to(x.dtype))
                x[:, g] = xs + blk.ls2(blk.mlp(blk.norm2(xs))).to(x.dtype)
            del o
            self.tick("proj_mlp_s")


def _check_supported(encoder: Encoder) -> None:
    from olmoearth_pretrain.nn.flexi_vit import Perceiver

    problems = []
    if encoder.has_register_tokens:
        problems.append("encoder register tokens (global within a window)")
    p = encoder.perceiver
    if p is not None:
        if not isinstance(p, Perceiver):
            problems.append(f"perceiver type {type(p).__name__}")
        else:
            if not p.use_2d_rope:
                problems.append("Perceiver without 2D RoPE")
            if p.token_mix_layout is not None:
                problems.append("token_mix_layout")
            if p.time_rope_encoding is not None or p.read_time_range:
                problems.append("time-aware reads (window time anchor)")
            if p.share_read_kv:
                problems.append("share_read_kv")
    if problems:
        raise NotImplementedError(
            "Lighthouse RC does not support: " + "; ".join(problems)
        )


@torch.no_grad()
def encoder_lighthouse(
    encoder: Encoder,
    tokens: Tensor,
    mask: Tensor,
    positions: Tensor,
    cell_ids: Tensor,
    spatial_grid: tuple[int, int],
    patch_size: int,
    patch_spacing: float,
) -> tuple[Tensor, Tensor | None, Tensor | None, dict[str, float]]:
    """The encoder blocks, norm and Perceiver of one domain under a sliding FOV.

    Args:
        encoder: The encoder; ``encoder.lighthouse`` holds the settings.
        tokens: ``[1, N, D]`` collapsed tokens with composite encodings.
        mask: ``[1, N]`` mask values; only ``ONLINE_ENCODER`` tokens take part.
        positions: ``[1, N, 3]`` (or 2) RoPE positions, collapsed order.
        cell_ids: ``[1, N]`` row-major patch-cell index per token.
        spatial_grid: ``(n_h, n_w)`` patch grid.
        patch_size: Token patch size.
        patch_spacing: Distance between patch centres in the RoPE frame.

    Returns:
        ``(tokens_out [1, N, D] normed, zeros where not visible -- or the input
        tokens unless ``return_tokens``; registers
        [1, h, w, register_dim] or None, register_positions or None, stats)``.
    """
    from olmoearth_pretrain.train.masking import MaskValue

    settings: RCLighthouseSettings = encoder.lighthouse  # type: ignore[assignment]
    _check_supported(encoder)
    if tokens.device.type != "cuda" and not settings.dense:
        settings = dataclasses.replace(settings, dense=True)
    if tokens.shape[0] != 1:
        raise ValueError("Lighthouse runs one domain per call (batch size 1)")
    if settings.fov_px % patch_size:
        raise ValueError(f"fov_px {settings.fov_px} is not a multiple of {patch_size}")
    device = tokens.device
    n_h, n_w = spatial_grid
    fov = settings.fov_px // patch_size
    if fov > n_h or fov > n_w:
        raise ValueError(f"FOV of {fov} cells exceeds the {n_h}x{n_w} cell domain")
    block = settings.block
    quantum = settings.fov_quantum
    if fov % quantum:
        raise ValueError(f"fov_quantum {quantum} does not divide the {fov}-cell FOV")
    run = _Runner(settings)
    stats: dict[str, float] = {}

    visible = mask[0] == MaskValue.ONLINE_ENCODER.value
    vis_idx = visible.nonzero()[:, 0]
    cells = cell_ids[0, vis_idx].long()
    if bool((cells < 0).any()):
        raise NotImplementedError("Lighthouse needs every token on the grid")
    cells_np = cells.cpu().numpy()

    shift = None
    if settings.origin_px != (0, 0):
        if any(o % patch_size for o in settings.origin_px):
            raise ValueError(f"origin_px {settings.origin_px} not patch-aligned")
        shift = torch.tensor(
            settings.origin_px, dtype=positions.dtype, device=device
        ) * (patch_spacing / patch_size)

    # --- tokens -------------------------------------------------------------------
    t0 = time.perf_counter()
    # Exact FOV: cell rows (a block's queries span ~2 cells). Quantized: the quantum
    # tiles themselves, so a block never straddles two FOVs.
    tok_tile = (1, n_w) if quantum == 1 else (quantum, quantum)
    tok = _slot_layout(cells_np // n_w, cells_np % n_w, n_w, tok_tile, block)
    tok.quantum = quantum
    tok_dest = torch.from_numpy(tok.dest).to(device)
    stats["tokens"] = float(vis_idx.numel())
    stats["token_slots"] = float(tok.length)
    x = torch.zeros(1, tok.length, tokens.shape[-1], dtype=tokens.dtype, device=device)
    x[0, tok_dest] = tokens[0, vis_idx]
    pos = torch.zeros(
        1, tok.length, positions.shape[-1], dtype=positions.dtype, device=device
    )
    pos[0, tok_dest] = positions[0, vis_idx]
    if shift is not None:
        pos[..., -2:] += shift
    vit_plan = _make_plan(tok, tok, fov, n_h, n_w, settings, device)
    stats.update({f"vit_{k}": v for k, v in vit_plan.stats.items()})
    stats["layout_s"] = time.perf_counter() - t0
    run._t = time.perf_counter()

    for blk in encoder.blocks:
        h_src = lambda s, _b=blk: _b.norm1(x[:, s])  # noqa: E731
        k_all, v_all = run.kv(blk, h_src, pos, tok.length)
        run.tick("kv_s")
        run.block(blk, x, pos, k_all, v_all, vit_plan)
        del k_all, v_all
    for s in _spans(tok.length, settings.mlp_chunk):
        x[:, s] = encoder.norm(x[:, s])

    tokens_out = tokens
    if settings.return_tokens:
        tokens_out = torch.zeros_like(tokens)
        tokens_out[0, vis_idx] = x[0, tok_dest]

    p: Perceiver | None = encoder.perceiver
    registers = register_positions = None
    if p is not None:
        stride = patch_size
        if p.pixel_latents:
            stride = choose_latent_stride(
                training=False,
                spatial_grid=spatial_grid,
                patch_size=patch_size,
                random_latent_stride=p.random_latent_stride,
                max_latents=p.max_latents,
                eval_latent_stride=p.eval_latent_stride,
            )
        lat_h, lat_w = n_h * patch_size // stride, n_w * patch_size // stride
        per_cell = (patch_size // stride) ** 2
        lr = np.repeat(np.arange(lat_h), lat_w)
        lc = np.tile(np.arange(lat_w), lat_h)
        l_rows, l_cols = lr * stride // patch_size, lc * stride // patch_size
        tile = settings.latent_tile or _default_latent_tile(per_cell, block)
        tile = (-(-tile[0] // quantum) * quantum, -(-tile[1] // quantum) * quantum)
        lat = _slot_layout(l_rows, l_cols, n_w, tile, block)
        lat.quantum = quantum
        lat_dest = torch.from_numpy(lat.dest).to(device)
        read_plan = _make_plan(lat, tok, fov, n_h, n_w, settings, device)
        self_plan = _make_plan(lat, lat, fov, n_h, n_w, settings, device)
        stats.update({f"read_{k}": v for k, v in read_plan.stats.items()})
        stats.update({f"latself_{k}": v for k, v in self_plan.stats.items()})
        stats["latent_slots"] = float(lat.length)

        # Latent positions: pixel latent centres (at stride == patch they are the
        # patch coordinates, which is what the stock register grid lays down).
        register_positions = build_pixel_latent_positions(
            1, (lat_h, lat_w), patch_size, patch_spacing, device, stride
        )
        lpos = torch.zeros(1, lat.length, 2, dtype=positions.dtype, device=device)
        lpos[0, lat_dest] = register_positions[0].to(positions.dtype)
        if shift is not None:
            lpos += shift
        key_pos = pos[..., -2:]
        reg = torch.zeros(
            1, lat.length, p.register.shape[-1], dtype=x.dtype, device=device
        )
        reg[0, lat_dest] = p.register[0].to(x.dtype)
        run._t = time.perf_counter()
        for i, (read_blk, lat_blk) in enumerate(zip(p.read_blocks, p.latent_blocks)):
            if p.per_depth_read_proj:
                norm, proj = p.input_norms[i], p.kv_projs[i]
            else:
                norm, proj = p.input_norm, p.kv_proj
            src = lambda s, _n=norm, _p=proj: _p(_n(x[:, s]))  # noqa: E731
            k_all, v_all = run.kv(read_blk, src, key_pos, tok.length)
            run.tick("read_kv_s")
            run.block(read_blk, reg, lpos, k_all, v_all, read_plan)
            del k_all, v_all
            l_src = lambda s, _b=lat_blk: _b.norm1(reg[:, s])  # noqa: E731
            k_all, v_all = run.kv(lat_blk, l_src, lpos, lat.length)
            run.tick("latent_kv_s")
            run.block(lat_blk, reg, lpos, k_all, v_all, self_plan)
            del k_all, v_all
        out = p.norm(reg[0, lat_dest])
        registers = rearrange(out[None], "b (h w) d -> b h w d", h=lat_h, w=lat_w)
    if run.timings is not None:
        stats.update(run.timings)
    return tokens_out, registers, register_positions, stats
