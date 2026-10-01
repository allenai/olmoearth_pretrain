"""Mask-free attention for the joint latent pattern at inference.

The joint blocks (:class:`~olmoearth_pretrain.nn.joint_latent.JointLatentTransformer`)
attend over ``[tokens ; latents]`` under a structured mask. With ``latent_reads_all``
and no local radii the rule is

* a latent query sees every valid key (all tokens and all latents);
* a token query sees all latents and the valid tokens of its own patch cell.

Under FlexAttention that is one masked kernel over the whole ~9.5k-long sequence
(ws16 / ps1 / T12 / S1+S2+L8), with partial 128-blocks wherever a cell's run of tokens
(36 at that shape) straddles a block edge. Profiling put FlexAttention at ~40% of
the joint arms' GPU time against ~19% for every GEMM combined, so the arm's MAC
savings over the RC never reached wall-clock.

The same attention splits exactly into dense pieces:

* latent queries: one dense attention over every key (a key-padding mask only when
  the batch has missing tokens);
* token queries: two attentions over DISJOINT key sets -- the latents (dense) and
  the own cell's tokens (``flash_attn_varlen_func`` over cell segments) -- merged
  with their log-sum-exps. Softmax over a union of disjoint key sets is the
  LSE-weighted average of the per-set softmaxes, so this is exact up to
  floating-point reassociation.

Invalid (padding) tokens are never keys and only the latents leave the joint blocks,
so invalid query rows are left with the latent-only part; nothing reads them.

Inference only: flash-attn does not propagate gradients through the returned
log-sum-exps, so the merge is not differentiable. Callers fall back to the masked
path whenever grad is enabled.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor

try:
    import flash_attn
except ImportError:  # pragma: no cover - CPU / no-flash installs
    flash_attn = None


@dataclass
class DenseJointLayout:
    """Packing of the token queries into own-cell segments, built once per forward.

    Attributes:
        n_tokens: Tokens at the front of the joint sequence (latents follow).
        token_index: Flat ``b * n_tokens + i`` indices of the valid tokens, ordered
            by ``(batch, cell)`` so each cell's tokens are one contiguous segment.
        cu_seqlens: ``int32`` segment boundaries into ``token_index``.
        max_seqlen: Longest segment.
        key_valid: ``[B, L]`` bool, True for keys that may be attended.
        all_valid: Whether every token is valid (no key-padding mask needed).
    """

    n_tokens: int
    token_index: Tensor
    cu_seqlens: Tensor
    max_seqlen: int
    key_valid: Tensor
    all_valid: bool


def build_dense_joint_layout(
    cell_ids: Tensor, valid_tokens: Tensor, n_cells: int, n_latents: int
) -> DenseJointLayout:
    """Group the valid tokens of every ``(batch, cell)`` into contiguous segments.

    Args:
        cell_ids: Long ``[B, N]`` cell of each token (``-1`` for non-spatial tokens,
            which form one segment per sample: they see each other, as under the mask).
        valid_tokens: Bool ``[B, N]``.
        n_cells: Number of patch cells (``cell_ids < n_cells``).
        n_latents: Latents appended after the tokens.
    """
    batch, n_tokens = cell_ids.shape
    device = cell_ids.device
    flat_valid = valid_tokens.reshape(-1)
    # (batch, cell) key, monotone in both; +1 lifts the non-spatial cell -1 to 0.
    batch_ids = torch.arange(batch, device=device)[:, None].expand(-1, n_tokens)
    key = (batch_ids * (n_cells + 1) + cell_ids + 1).reshape(-1)
    valid_index = flat_valid.nonzero().squeeze(1)
    order = torch.argsort(key[valid_index], stable=True)
    token_index = valid_index[order]
    _, counts = torch.unique_consecutive(key[token_index], return_counts=True)
    cu_seqlens = F.pad(counts.cumsum(0), (1, 0)).to(torch.int32)
    key_valid = torch.cat(
        [
            valid_tokens,
            torch.ones(batch, n_latents, dtype=torch.bool, device=device),
        ],
        dim=1,
    )
    return DenseJointLayout(
        n_tokens=n_tokens,
        token_index=token_index,
        cu_seqlens=cu_seqlens,
        max_seqlen=int(counts.max().item()) if counts.numel() else 0,
        key_valid=key_valid,
        all_valid=bool(flat_valid.all().item()),
    )


def _attention_with_lse(
    q: Tensor, k: Tensor, v: Tensor, scale: float
) -> tuple[Tensor, Tensor]:
    """Reference ``[..., Nq, D]`` attention returning ``(out, lse)`` (fp32 lse)."""
    scores = (q.float() @ k.float().transpose(-2, -1)) * scale
    lse = torch.logsumexp(scores, dim=-1)
    out = torch.softmax(scores, dim=-1) @ v.float()
    return out.to(q.dtype), lse


def _token_parts_flash(
    q_tok: Tensor,
    k_tok: Tensor,
    v_tok: Tensor,
    k_lat: Tensor,
    v_lat: Tensor,
    layout: DenseJointLayout,
    scale: float,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Latent part ``(B, N, H, D) / (B, N, H)`` and own-cell part ``(T, H, D) / (T, H)``.

    Inputs are ``[B, n, H, D]`` (flash layout).
    """
    assert flash_attn is not None
    batch, n_tokens, heads, dim = q_tok.shape
    out_lat, lse_lat, _ = flash_attn.flash_attn_func(
        q_tok, k_lat, v_lat, softmax_scale=scale, return_attn_probs=True
    )
    idx = layout.token_index
    flat = lambda x: x.reshape(batch * n_tokens, heads, dim)[idx]  # noqa: E731
    out_cell, lse_cell, _ = flash_attn.flash_attn_varlen_func(
        flat(q_tok),
        flat(k_tok),
        flat(v_tok),
        layout.cu_seqlens,
        layout.cu_seqlens,
        layout.max_seqlen,
        layout.max_seqlen,
        softmax_scale=scale,
        return_attn_probs=True,
    )
    # flash-attn >= 2.4 returns the varlen lse as (H, total_q).
    if lse_cell.shape != (heads, idx.shape[0]):
        raise RuntimeError(
            f"unexpected varlen lse shape {tuple(lse_cell.shape)}; expected "
            f"{(heads, idx.shape[0])}"
        )
    return out_lat, lse_lat.transpose(1, 2), out_cell, lse_cell.transpose(0, 1)


def _token_parts_reference(
    q_tok: Tensor,
    k_tok: Tensor,
    v_tok: Tensor,
    k_lat: Tensor,
    v_lat: Tensor,
    layout: DenseJointLayout,
    scale: float,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Same contract as :func:`_token_parts_flash`, in plain torch (CPU / tests)."""
    batch, n_tokens, heads, dim = q_tok.shape
    hfirst = lambda x: x.transpose(1, 2)  # noqa: E731  [B, n, H, D] -> [B, H, n, D]
    out_lat, lse_lat = _attention_with_lse(
        hfirst(q_tok), hfirst(k_lat), hfirst(v_lat), scale
    )
    idx = layout.token_index
    flat = lambda x: x.reshape(batch * n_tokens, heads, dim)[idx]  # noqa: E731
    qf, kf, vf = flat(q_tok), flat(k_tok), flat(v_tok)
    out_cell = torch.empty_like(qf)
    lse_cell = torch.empty(qf.shape[:2], dtype=torch.float32, device=qf.device)
    bounds = layout.cu_seqlens.tolist()
    for start, end in zip(bounds[:-1], bounds[1:]):
        seg = lambda x: x[start:end].transpose(0, 1)  # noqa: E731  [H, n, D]
        o, lse = _attention_with_lse(seg(qf), seg(kf), seg(vf), scale)
        out_cell[start:end] = o.transpose(0, 1)
        lse_cell[start:end] = lse.transpose(0, 1)
    return hfirst(out_lat), hfirst(lse_lat), out_cell, lse_cell


def dense_joint_attention(
    q: Tensor, k: Tensor, v: Tensor, layout: DenseJointLayout, use_flash: bool
) -> Tensor:
    """The joint-pattern attention without a mask; ``[B, H, L, D]`` in and out.

    Args:
        q: Post-RoPE queries ``[B, H, L, D]`` with ``L = n_tokens + n_latents``.
        k: Post-RoPE keys, same shape.
        v: Values, same shape.
        layout: From :func:`build_dense_joint_layout` for this forward pass.
        use_flash: flash-attn kernels (CUDA, fp16/bf16) or the plain-torch reference.
    """
    out_dtype = v.dtype
    if use_flash:
        # qk-norm under autocast hands back fp32 q/k next to a bf16 v; flash needs
        # one half-precision dtype.
        half = v.dtype if v.dtype in (torch.float16, torch.bfloat16) else torch.bfloat16
        q, k, v = q.to(half), k.to(half), v.to(half)
    batch, heads, length, dim = q.shape
    n_tokens = layout.n_tokens
    scale = dim**-0.5

    # Latent queries: every valid key.
    q_lat = q[:, :, n_tokens:]
    if use_flash and layout.all_valid:
        out_latq = flash_attn.flash_attn_func(
            q_lat.transpose(1, 2),
            k.transpose(1, 2),
            v.transpose(1, 2),
            softmax_scale=scale,
        ).transpose(1, 2)
    else:
        mask = None if layout.all_valid else layout.key_valid[:, None, None, :]
        out_latq = F.scaled_dot_product_attention(
            q_lat, k, v, attn_mask=mask, scale=scale
        )

    # Token queries: latents (dense) + own cell (segments), merged by LSE.
    seq = lambda x: x.transpose(1, 2)  # noqa: E731  [B, H, n, D] -> [B, n, H, D]
    parts = _token_parts_flash if use_flash else _token_parts_reference
    out_lat, lse_lat, out_cell, lse_cell = parts(
        seq(q[:, :, :n_tokens]),
        seq(k[:, :, :n_tokens]),
        seq(v[:, :, :n_tokens]),
        seq(k[:, :, n_tokens:]),
        seq(v[:, :, n_tokens:]),
        layout,
        scale,
    )
    idx = layout.token_index
    out_tok = out_lat.reshape(batch * n_tokens, heads, dim)
    lse_a = lse_lat.reshape(batch * n_tokens, heads)[idx]
    peak = torch.maximum(lse_a, lse_cell)
    w_a = torch.exp(lse_a - peak)[..., None]
    w_b = torch.exp(lse_cell - peak)[..., None]
    merged = (out_tok[idx].float() * w_a + out_cell.float() * w_b) / (w_a + w_b)
    out_tok = out_tok.index_copy(0, idx, merged.to(out_tok.dtype))
    out_tok = out_tok.reshape(batch, n_tokens, heads, dim).transpose(1, 2)
    return torch.cat([out_tok, out_latq.to(out_tok.dtype)], dim=2).to(out_dtype)
