"""Per-cell slot merge: shrink each patch cell's tokens partway through the encoder.

After ``merge_after`` ViT blocks, the ``n`` visible tokens of each patch cell (all its
timesteps and modalities) are merged into ``t = max(ceil(n / ratio), min(n, min_slots))``
slots with slot attention (Locatello et al. 2020): the softmax is taken over SLOTS, so
every token's content is split among the cell's slots and the slots compete for tokens.
Cells with ``t == n`` (few tokens: single-timestep inputs, heavily masked cells) bypass
the merge and keep their tokens unchanged. The remaining blocks run over the bypassed
tokens and the slots together.

Slots start at chunk means: the cell's tokens in (modality, time) order are split into
``t`` consecutive chunks, so the initial slots are shared across modalities in
proportion to their visible counts, and then move freely over ``iters`` rounds
(GRU or residual update, then an MLP). A slot's RoPE position is the cell's row/col and the
attention-weighted mean time of the tokens it absorbed (a point position).

The merge also returns an *unmerge* for callers that want per-token outputs in the
pre-merge layout (evals without the Perceiver): each token of a merged cell gets the
MEAN of its cell's slot outputs, and bypassed tokens get their own output -- what
pooling the outputs over the cell would give anyway. Training never reads it (the
decoder and losses read the Perceiver registers).

Narrow stage (``narrow_depth > 0``, then ``merge_after = 0``): before any encoder
block, the tokens are projected down to ``narrow_dim`` and run through
``narrow_depth`` blocks that attend only WITHIN each patch cell (all its timesteps
and modalities), the merge runs at ``narrow_dim``, and the merged sequence is
projected back up to the encoder width. The pre-narrow encoder-width tokens are
pooled with the same slot weights and added (a skip around the narrow stage), so
the wide blocks start from the full-width input plus the narrow stage's output.
This is how every-capture inputs (hundreds of tokens per cell) stay at v1.3's cost:
the per-token work happens at low width, the encoder-width blocks see <=
``max_slots_per_cell`` tokens per cell.

Narrow front end (``narrow_front_end``): the patch embedding and composite encodings
produce ``narrow_dim``-wide tokens directly (no down-projection) and there is no
encoder-width skip, so nothing before the merge is encoder-width: the per-input-token
cost and memory are all at the narrow width, which lets far more input tokens through
than the encoder-width blocks would allow.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch import Tensor

from olmoearth_pretrain.config import Config
from olmoearth_pretrain.nn.attention import Block


@dataclass
class SlotMergeConfig(Config):
    """Configuration for :class:`SlotMerge`.

    Args:
        merge_after: Number of encoder blocks run on all tokens before the merge.
        ratio: Target tokens per slot (``t = ceil(n / ratio)``).
        min_slots: Cells are never merged below ``min(n, min_slots)`` slots; cells
            whose slot count would equal their token count bypass the merge.
        iters: Slot-attention rounds.
        max_slots: Size of the learned slot-index bank (max slots per cell).
        mlp_ratio: Hidden width of the slot MLP, as a multiple of the token width.
        max_slots_per_cell: Optional cap on a merged cell's slot count (e.g. 24 for
            every-capture cells of 100-300 tokens). None = no cap.
        narrow_depth: Blocks of the narrow within-cell stage before the merge
            (0 = no narrow stage; then the merge sits after ``merge_after`` blocks).
        narrow_dim: Width of the narrow stage (and of the merge); required with
            ``narrow_depth > 0``. Heads = ``narrow_dim // 64``.
        narrow_front_end: Embed straight to ``narrow_dim`` (patch embedding and
            composite encodings at the narrow width) and drop the encoder-width skip;
            requires the narrow stage.
        narrow_out_norm: LayerNorm the narrow stage's output before the merge. Without
            it the narrow residual stream reaches the merge un-normalized: slots start
            at its scale and are the GRU's hidden state (at d768 it grew to RMS ~33 by
            80k, and the merge GRU's gradients ~900x those at d256).
        month_length: One slot per calendar month instead of ``ceil(n / ratio)``:
            the length of a month in the RoPE time coordinate (30.4375 days times
            the encoder's ``rope_temporal_coordinate_scale``). A cell gets
            ``max(min(months, cap), min(n, min_slots))`` slots, ``months`` the
            calendar months its tokens fall in; with one slot per month, slot
            ``k`` starts from the tokens of the cell's ``k``-th month, otherwise
            (short spans) from ``t`` time-ordered chunks of its tokens.
        slot_update: How each round's attention output updates the slots: "gru"
            (Locatello et al.) or "residual" (``slots + proj(updates)``, ``proj``
            zero-initialized, so the slots start as their month/chunk means).
    """

    merge_after: int = 4
    ratio: int = 3
    min_slots: int = 4
    iters: int = 3
    max_slots: int = 36
    mlp_ratio: float = 2.0
    max_slots_per_cell: int | None = None
    narrow_depth: int = 0
    narrow_dim: int | None = None
    narrow_front_end: bool = False
    month_length: float | None = None
    narrow_out_norm: bool = False
    slot_update: str = "gru"

    def validate(self, depth: int) -> None:
        """Check the merge point fits the encoder depth."""
        if self.narrow_depth > 0:
            if self.merge_after != 0 or self.narrow_dim is None:
                raise ValueError(
                    "a narrow stage needs merge_after=0 and narrow_dim set, got "
                    f"merge_after={self.merge_after}, narrow_dim={self.narrow_dim}"
                )
            if self.narrow_dim % 64:
                raise ValueError(
                    f"narrow_dim must be a multiple of 64: {self.narrow_dim}"
                )
        elif self.narrow_front_end:
            raise ValueError("narrow_front_end needs a narrow stage (narrow_depth > 0)")
        elif not 0 < self.merge_after < depth:
            raise ValueError(
                f"merge_after must be in (0, depth={depth}), got {self.merge_after}"
            )
        if self.slot_update not in ("gru", "residual"):
            raise ValueError(f"slot_update must be gru or residual: {self.slot_update}")
        if self.ratio < 2 or self.min_slots < 1 or self.iters < 1:
            raise ValueError(f"invalid slot merge settings: {self}")
        cap = self.max_slots_per_cell
        if cap is not None and not self.min_slots <= cap <= self.max_slots:
            raise ValueError(
                f"max_slots_per_cell={cap} must be in [min_slots, max_slots]"
            )

    def build(self, dim: int) -> SlotMerge:
        """Build the merge module for ``dim``-wide tokens."""
        return SlotMerge(
            dim=dim,
            ratio=self.ratio,
            min_slots=self.min_slots,
            iters=self.iters,
            max_slots=self.max_slots,
            mlp_ratio=self.mlp_ratio,
            max_slots_per_cell=self.max_slots_per_cell,
            month_length=self.month_length,
            slot_update=self.slot_update,
        )


class NarrowStage(nn.Module):
    """Down-projection + blocks that attend within each patch cell (narrow width)."""

    def __init__(
        self,
        in_dim: int,
        dim: int,
        depth: int,
        out_norm: bool = False,
        **block_kwargs: object,
    ):
        """Initialize: ``in_dim -> dim`` projection and ``depth`` blocks at ``dim``.

        ``out_norm``: a LayerNorm on the stage's output (the merge's input), as a
        pre-norm stack's final norm.
        """
        super().__init__()
        # Tokens already at the narrow width (narrow front end) need no projection.
        self.down: nn.Module = (
            nn.Linear(in_dim, dim) if in_dim != dim else nn.Identity()
        )
        self.blocks = nn.ModuleList(
            Block(dim, dim // 64, **block_kwargs)  # type: ignore[arg-type]
            for _ in range(depth)
        )
        self.out_norm: nn.Module = nn.LayerNorm(dim) if out_norm else nn.Identity()

    def forward(
        self, x: Tensor, positions: Tensor, valid: Tensor, cell_ids: Tensor
    ) -> Tensor:
        """``x`` ``[B, L, in_dim]`` -> ``[B, L, dim]``, attention within each cell.

        Valid tokens are regrouped into ``[G, P, dim]`` (one group per sample and
        cell; a sample's non-spatial tokens form one group), padded with a key mask.
        """
        B, L, _ = x.shape
        device = x.device
        h = self.down(x)
        b_idx, l_idx = valid.nonzero(as_tuple=True)
        cell = cell_ids[b_idx, l_idx]
        num_cells = int(cell_ids.max().item()) + 1 if bool((cell >= 0).any()) else 0
        key = b_idx * (num_cells + 1) + torch.where(cell >= 0, cell, num_cells)
        groups, g = torch.unique(key, return_inverse=True)
        order = torch.argsort(g, stable=True)
        b_idx, l_idx, g = b_idx[order], l_idx[order], g[order]
        counts = torch.bincount(g, minlength=groups.numel())
        start = torch.cumsum(counts, 0) - counts
        rank = torch.arange(g.numel(), device=device) - start[g]
        G, P = groups.numel(), int(counts.max().item())
        grouped = h.new_zeros(G, P, h.shape[-1])
        grouped[g, rank] = h[b_idx, l_idx]
        grouped_pos = positions.new_zeros(G, P, positions.shape[-1])
        grouped_pos[g, rank] = positions[b_idx, l_idx]
        key_mask = torch.zeros(G, P, dtype=torch.bool, device=device)
        key_mask[g, rank] = True
        for blk in self.blocks:
            grouped = blk(x=grouped, attn_mask=key_mask, rope_positions=grouped_pos)
        grouped = self.out_norm(grouped)
        out = h.new_zeros(B, L, grouped.shape[-1])
        out[b_idx, l_idx] = grouped[g, rank].to(out.dtype)
        return out


class SlotMerge(nn.Module):
    """Merge each patch cell's tokens into slots (see the module docstring)."""

    def __init__(
        self,
        dim: int,
        ratio: int,
        min_slots: int,
        iters: int,
        max_slots: int,
        mlp_ratio: float,
        max_slots_per_cell: int | None = None,
        month_length: float | None = None,
        slot_update: str = "gru",
    ) -> None:
        """Initialize the merge."""
        super().__init__()
        self.ratio = ratio
        self.month_length = month_length
        self.min_slots = min_slots
        self.max_slots_per_cell = max_slots_per_cell
        self.iters = iters
        self.max_slots = max_slots
        self.scale = dim**-0.5
        self.slot_index = nn.Parameter(torch.randn(max_slots, dim) * 0.02)
        self.norm_tokens = nn.LayerNorm(dim)
        self.norm_slots = nn.LayerNorm(dim)
        self.norm_mlp = nn.LayerNorm(dim)
        self.q = nn.Linear(dim, dim, bias=False)
        self.k = nn.Linear(dim, dim, bias=False)
        self.v = nn.Linear(dim, dim, bias=False)
        if slot_update == "gru":
            self.gru = nn.GRUCell(dim, dim)
        else:
            self.update_proj = nn.Linear(dim, dim)
            nn.init.zeros_(self.update_proj.weight)
            nn.init.zeros_(self.update_proj.bias)
        self.slot_update = slot_update
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim)
        )

    def num_slots(self, n: Tensor, months: Tensor | None = None) -> Tensor:
        """Slots for a cell with ``n`` tokens (``n`` itself means bypass).

        ``months``: the cell's calendar-month count, with ``month_length``.
        """
        target = torch.ceil(n / self.ratio).long() if months is None else months
        if self.max_slots_per_cell is not None:
            target = torch.clamp(target, max=self.max_slots_per_cell)
        return torch.maximum(target, torch.clamp(n, max=self.min_slots))

    def _zero_touch(self) -> Tensor:
        """0 * every parameter: keeps all parameters in the graph when nothing merges."""
        return sum(p.sum() for p in self.parameters()) * 0.0

    def forward(
        self,
        x: Tensor,
        positions: Tensor,
        valid: Tensor,
        cell_ids: Tensor,
        modality_ids: Tensor,
        extra: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor, Callable[[Tensor], Tensor], Tensor | None]:
        """Merge ``x`` ``[B, L, D]`` (padded) into ``[B, L', D]``.

        Args:
            x: Encoder tokens, padded per sample.
            positions: RoPE positions ``[B, L, 3]`` as ``(t, row, col)``.
            valid: ``[B, L]`` bool, real (visible) tokens.
            cell_ids: ``[B, L]`` row-major patch cell, ``-1`` for non-spatial tokens
                (never merged).
            modality_ids: ``[B, L]`` modality index (orders the chunk init).
            extra: Optional ``[B, L, E]`` tensor merged with the SAME final slot
                weights (a slot gets the weighted mean of its tokens' ``extra``;
                kept tokens keep theirs) -- the narrow stage's encoder-width skip.

        Returns:
            ``(tokens, positions, valid, unmerge, merged_extra)``: the merged, padded
            sequence, a function mapping a ``[B, L', D]`` sequence in that layout
            back to ``[B, L, D]`` in the input layout (zeros at padding), and
            ``extra`` in the merged layout (None without ``extra``).
        """
        B, L, D = x.shape
        device = x.device
        b_idx, l_idx = valid.nonzero(as_tuple=True)
        cell = cell_ids[b_idx, l_idx]
        spatial = cell >= 0
        num_cells = int(cell_ids.max().item()) + 1 if spatial.any() else 1
        group = torch.where(spatial, b_idx * num_cells + cell, -1)

        # Tokens per (sample, cell) and the slots each cell gets.
        n_group = torch.bincount(group[spatial], minlength=B * num_cells)
        month = None
        months_group = None
        if self.month_length is not None and bool(spatial.any()):
            month = torch.floor(
                positions[b_idx, l_idx, 0].float() / self.month_length
            ).long()
            month = month - month.min()
            # Distinct (cell, month) pairs, counted per cell.
            pairs = torch.unique(
                group[spatial] * (int(month.max()) + 1) + month[spatial]
            )
            months_group = torch.bincount(
                pairs // (int(month.max()) + 1), minlength=B * num_cells
            )
        t_group = self.num_slots(n_group, months_group)
        merging_group = (t_group < n_group) & (n_group > 0)
        merges = spatial & merging_group[group.clamp(min=0)]
        keep = ~merges

        if not bool(merges.any()):
            return x + self._zero_touch(), positions, valid, lambda y: y, extra

        # ---- merging cells, padded to [G, P, D] (tokens) and [G, S, D] (slots)
        m_b, m_l, m_group = b_idx[merges], l_idx[merges], group[merges]
        groups, g_local = torch.unique(m_group, return_inverse=True)
        G = groups.numel()
        n_g = n_group[groups]
        t_g = t_group[groups]
        P, S = int(n_g.max().item()), int(t_g.max().item())
        if S > self.max_slots:
            raise ValueError(f"{S} slots exceed max_slots={self.max_slots}")
        # Order inside a cell by (modality, time) -> rank j; chunk-init slot floor(j t/n).
        # Lexicographic (cell, modality, time) by stable sorts, least significant first.
        # Per month: by time alone (chunks are time spans, months are contiguous).
        t_coord = positions[m_b, m_l, 0].float()
        order = torch.argsort(t_coord, stable=True)
        if month is None:
            mod = modality_ids[m_b, m_l]
            order = order[torch.argsort(mod[order], stable=True)]
        order = order[torch.argsort(g_local[order], stable=True)]
        m_b, m_l, g_local, t_coord = (
            m_b[order],
            m_l[order],
            g_local[order],
            t_coord[order],
        )
        start = torch.cumsum(n_g, 0) - n_g
        rank = torch.arange(m_b.numel(), device=device) - start[g_local]
        init_slot = (rank * t_g[g_local]) // n_g[g_local]
        if month is not None:
            # Cells with one slot per month start slot k from their k-th month.
            m_month = month[merges][order]
            new_month = torch.ones_like(m_month, dtype=torch.bool)
            new_month[1:] = (m_month[1:] != m_month[:-1]) | (
                g_local[1:] != g_local[:-1]
            )
            month_rank = torch.cumsum(new_month.long(), 0) - 1
            month_rank = month_rank - month_rank[start][g_local]
            assert months_group is not None
            by_month = (t_g == months_group[groups])[g_local]
            init_slot = torch.where(by_month, month_rank, init_slot)

        tokens = x.new_zeros(G, P, D)
        tokens[g_local, rank] = x[m_b, m_l]
        tok_valid = torch.zeros(G, P, dtype=torch.bool, device=device)
        tok_valid[g_local, rank] = True
        times = torch.zeros(G, P, device=device)
        times[g_local, rank] = t_coord
        slot_valid = torch.arange(S, device=device)[None] < t_g[:, None]

        init = x.new_zeros(G, S, D).index_put(
            (g_local, init_slot), x[m_b, m_l], accumulate=True
        )
        counts = torch.zeros(G, S, device=device).index_put(
            (g_local, init_slot), torch.ones_like(t_coord), accumulate=True
        )
        slots = init / counts.clamp(min=1).unsqueeze(-1).to(init.dtype)
        slots = slots + self.slot_index[:S].to(slots.dtype)

        normed = self.norm_tokens(tokens)
        k, v = self.k(normed), self.v(normed)
        neg = torch.finfo(torch.float32).min
        for _ in range(self.iters):
            q = self.q(self.norm_slots(slots))
            logits = torch.einsum("gpd,gsd->gps", k, q).float() * self.scale
            logits = logits.masked_fill(~slot_valid[:, None, :], neg)
            attn = logits.softmax(dim=-1) * tok_valid[..., None]  # over slots
            weights = attn / (attn.sum(dim=1, keepdim=True) + 1e-8)  # mean per slot
            updates = torch.einsum("gps,gpd->gsd", weights.to(v.dtype), v)
            if self.slot_update == "gru":
                slots = self.gru(
                    updates.reshape(-1, D), slots.reshape(-1, D).to(updates.dtype)
                ).view(G, S, D)
            else:
                slots = slots.to(updates.dtype) + self.update_proj(updates)
            slots = slots + self.mlp(self.norm_mlp(slots))
        slot_time = torch.einsum("gps,gp->gs", weights, times)

        # ---- reassemble per sample: kept tokens, then the valid slots
        g_b = groups // num_cells
        # Each cell's tokens are contiguous after the sort: its first one gives row/col.
        cell_rc = positions[m_b[start], m_l[start], 1:]
        s_g, s_j = slot_valid.nonzero(as_tuple=True)
        slot_pos = torch.cat(
            [slot_time[s_g, s_j, None].to(positions.dtype), cell_rc[s_g]], dim=-1
        )

        out_b = torch.cat([b_idx[keep], g_b[s_g]])
        out_x = torch.cat([x[b_idx[keep], l_idx[keep]], slots[s_g, s_j].to(x.dtype)])
        out_pos = torch.cat([positions[b_idx[keep], l_idx[keep]], slot_pos])
        out_extra = None
        if extra is not None:
            grouped_extra = extra.new_zeros(G, P, extra.shape[-1])
            grouped_extra[g_local, rank] = extra[m_b, m_l]
            slot_extra = torch.einsum(
                "gps,gpe->gse", weights.to(extra.dtype), grouped_extra
            )
            out_extra = torch.cat(
                [extra[b_idx[keep], l_idx[keep]], slot_extra[s_g, s_j]]
            )
        out_order = torch.argsort(out_b, stable=True)
        out_b, out_x, out_pos = out_b[out_order], out_x[out_order], out_pos[out_order]
        per_sample = torch.bincount(out_b, minlength=B)
        L_new = int(per_sample.max().item())
        slot_in_sample = (
            torch.arange(out_b.numel(), device=device)
            - (torch.cumsum(per_sample, 0) - per_sample)[out_b]
        )
        new_x = x.new_zeros(B, L_new, D)
        new_x[out_b, slot_in_sample] = out_x
        new_pos = positions.new_zeros(B, L_new, positions.shape[-1])
        new_pos[out_b, slot_in_sample] = out_pos
        new_valid = torch.zeros(B, L_new, dtype=torch.bool, device=device)
        new_valid[out_b, slot_in_sample] = True
        merged_extra = None
        if out_extra is not None:
            assert extra is not None
            merged_extra = extra.new_zeros(B, L_new, extra.shape[-1])
            merged_extra[out_b, slot_in_sample] = out_extra[out_order]

        # Where each kept token / slot landed, in concatenation order.
        dst_i = torch.empty_like(slot_in_sample)
        dst_i[out_order] = slot_in_sample
        n_keep = int(keep.sum().item())
        keep_b, keep_l = b_idx[keep], l_idx[keep]
        slot_b, slot_i = g_b[s_g], dst_i[n_keep:]
        n_slots = t_g.clamp(min=1)

        def unmerge(y: Tensor) -> Tensor:
            out = y.new_zeros(B, L, y.shape[-1])
            out[keep_b, keep_l] = y[keep_b, dst_i[:n_keep]]
            slot_out = y.new_zeros(G, S, y.shape[-1])
            slot_out[s_g, s_j] = y[slot_b, slot_i]
            # Each merged token: the mean of its cell's slot outputs.
            cell_mean = slot_out.sum(dim=1) / n_slots[:, None].to(y.dtype)
            out[m_b, m_l] = cell_mean[g_local].to(out.dtype)
            return out

        return new_x, new_pos, new_valid, unmerge, merged_extra
