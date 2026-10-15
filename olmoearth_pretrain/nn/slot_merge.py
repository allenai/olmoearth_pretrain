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
(GRU + MLP update). A slot's RoPE position is the cell's row/col and the
attention-weighted mean time of the tokens it absorbed (a point position).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
from torch import Tensor

from olmoearth_pretrain.config import Config


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
    """

    merge_after: int = 4
    ratio: int = 3
    min_slots: int = 4
    iters: int = 3
    max_slots: int = 36
    mlp_ratio: float = 2.0

    def validate(self, depth: int) -> None:
        """Check the merge point fits the encoder depth."""
        if not 0 < self.merge_after < depth:
            raise ValueError(
                f"merge_after must be in (0, depth={depth}), got {self.merge_after}"
            )
        if self.ratio < 2 or self.min_slots < 1 or self.iters < 1:
            raise ValueError(f"invalid slot merge settings: {self}")

    def build(self, dim: int) -> SlotMerge:
        """Build the merge module for ``dim``-wide tokens."""
        return SlotMerge(
            dim=dim,
            ratio=self.ratio,
            min_slots=self.min_slots,
            iters=self.iters,
            max_slots=self.max_slots,
            mlp_ratio=self.mlp_ratio,
        )


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
    ) -> None:
        """Initialize the merge."""
        super().__init__()
        self.ratio = ratio
        self.min_slots = min_slots
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
        self.gru = nn.GRUCell(dim, dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim)
        )

    def num_slots(self, n: Tensor) -> Tensor:
        """Slots for a cell with ``n`` tokens (``n`` itself means bypass)."""
        return torch.maximum(
            torch.ceil(n / self.ratio).long(), torch.clamp(n, max=self.min_slots)
        )

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
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Merge ``x`` ``[B, L, D]`` (padded) into ``[B, L', D]``.

        Args:
            x: Encoder tokens, padded per sample.
            positions: RoPE positions ``[B, L, 3]`` as ``(t, row, col)``.
            valid: ``[B, L]`` bool, real (visible) tokens.
            cell_ids: ``[B, L]`` row-major patch cell, ``-1`` for non-spatial tokens
                (never merged).
            modality_ids: ``[B, L]`` modality index (orders the chunk init).

        Returns:
            ``(tokens, positions, valid)`` of the merged, padded sequence.
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
        t_group = self.num_slots(n_group)
        merging_group = (t_group < n_group) & (n_group > 0)
        merges = spatial & merging_group[group.clamp(min=0)]
        keep = ~merges

        if not bool(merges.any()):
            return x + self._zero_touch(), positions, valid

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
        t_coord = positions[m_b, m_l, 0].float()
        mod = modality_ids[m_b, m_l]
        order = torch.argsort(t_coord, stable=True)
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
            slots = self.gru(
                updates.reshape(-1, D), slots.reshape(-1, D).to(updates.dtype)
            ).view(G, S, D)
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
        return new_x, new_pos, new_valid
