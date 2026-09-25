"""Joint latent-token transformer: one attention over tokens and a latent grid.

An alternative to the ``[ViT blocks -> Perceiver reads]`` encoder that keeps the
register grid as the output but replaces both the joint ViT attention and the
Perceiver's separate read / latent blocks with ONE block type over the concatenated
sequence ``[patch tokens ; latents]`` and a structured attention pattern:

* a latent at grid cell ``(i, j)`` attends to every latent and to the input tokens of
  cell ``(i, j)`` (all timesteps, all modalities);
* an input token at cell ``(i, j)`` attends to every latent and to the other input
  tokens of cell ``(i, j)``.

Tokens therefore interact along time and modality within their cell, latents carry
spatial context, and the latent-to-token edges are the read, so no separate read
block exists. Attention is linear in the number of tokens (each query sees
``n_latents + tokens_per_cell`` keys) while every token still passes through the
block's linear layers; on a 16x16 / 12-timestep / S1+S2+L8 / patch-size-1 input a
block costs ~71 G MACs against ~196 G for a joint ViT block.

The pattern is block-sparse (a same-cell band plus a dense latent stripe), so on CUDA
it runs through FlexAttention with a compiled block mask and no quadratic memory. On
other devices the same rule is materialised as a dense boolean mask for SDPA, which
is exact and is what the tests compare against.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch
from einops import rearrange
from torch import Tensor, nn

from olmoearth_pretrain.config import Config
from olmoearth_pretrain.nn.attention import Block
from olmoearth_pretrain.nn.encodings import PositionEncoding

logger = logging.getLogger(__name__)

_COMPILED_FLEX: Callable[..., Tensor] | None = None


def flex_attention_cuda(q: Tensor, k: Tensor, v: Tensor, block_mask: Any) -> Tensor:
    """FlexAttention over ``[B, H, L, D]`` tensors, compiled once per process.

    The eager FlexAttention is a reference implementation (slow, and its backward is
    not guaranteed); the compiled kernel is the real one, so this is CUDA-only.
    """
    global _COMPILED_FLEX
    if _COMPILED_FLEX is None:
        from torch.nn.attention.flex_attention import flex_attention

        # dynamic=True: sequence lengths change every batch, and a recompile per
        # length would dwarf the attention itself.
        _COMPILED_FLEX = torch.compile(flex_attention, dynamic=True)
    return _COMPILED_FLEX(q, k, v, block_mask=block_mask)


_COMPILED_CREATE_BLOCK_MASK: Callable[..., Any] | None = None


def create_block_mask_cuda(
    mask_mod: Callable[[Tensor, Tensor, Tensor, Tensor], Tensor],
    batch: int,
    length: int,
    device: torch.device,
) -> Any:
    """``create_block_mask`` compiled once per process (CUDA only).

    The eager ``create_block_mask`` evaluates ``mask_mod`` on the full
    ``[B, L, L]`` grid -- with int64 gather intermediates, 8 bytes per element -- and
    only then reduces it to 128-blocks. At the eval shape (ws16, ps1, S1+S2+Landsat,
    12 timesteps: L ~ 9.5k, batch ~64) that is one 42.78 GiB allocation, which OOMed
    every joint-arm eval job (v1_3_vit0_rangerope_joint12_eval_step80000, 2026-09-22).
    Compiled, Inductor fuses the predicate into the block reduction, so peak memory is
    the block table rather than the dense grid. Training never hit this only because
    sampled patch sizes and rank-microbatch 32 keep L small.

    ``dynamic=True`` for the same reason as :func:`flex_attention_cuda`: L changes
    every batch. ``mask_mod`` is a fresh closure per call, but Dynamo guards on its
    code object and treats the captured tensors as graph inputs, so this does not
    recompile per call (the standard attention-gym pattern).
    """
    global _COMPILED_CREATE_BLOCK_MASK
    if _COMPILED_CREATE_BLOCK_MASK is None:
        from torch.nn.attention.flex_attention import create_block_mask

        _COMPILED_CREATE_BLOCK_MASK = torch.compile(create_block_mask, dynamic=True)
    return _COMPILED_CREATE_BLOCK_MASK(
        mask_mod, batch, None, length, length, device=device
    )


def build_register_grid_positions(
    patch_positions: Tensor, register_grid: tuple[int, int]
) -> Tensor:
    """Place an ``(n_h, n_w)`` grid evenly across the patch extent (GSD-scaled frame).

    Args:
        patch_positions: ``[B, N, 2]`` GSD-scaled ``(row, col)`` patch coordinates.
        register_grid: ``(n_h, n_w)`` grid to lay down.

    Returns:
        ``[B, n_h * n_w, 2]`` register coordinates spanning ``[0, max_patch_coord]``,
        row-major.
    """
    n_h, n_w = register_grid
    device = patch_positions.device
    # Patch coords are >= 0 (non-spatial tokens sit at 0), so amax gives the extent.
    max_pos = patch_positions.amax(dim=1)  # [B, 2]
    lin_h = torch.linspace(0.0, 1.0, n_h, device=device)
    lin_w = torch.linspace(0.0, 1.0, n_w, device=device)
    grid_h, grid_w = torch.meshgrid(lin_h, lin_w, indexing="ij")
    grid = torch.stack([grid_h, grid_w], dim=-1).reshape(-1, 2)  # [n_reg, 2] in [0, 1]
    return grid.unsqueeze(0) * max_pos.unsqueeze(1)  # [B, n_reg, 2]


def build_pixel_latent_positions(
    batch_size: int,
    latent_grid: tuple[int, int],
    patch_size: int,
    patch_spacing: float,
    device: torch.device,
) -> Tensor:
    """Pixel-centre latent coordinates in the patch RoPE frame.

    Patch ``i`` sits at ``i * patch_spacing``, so pixel ``p`` of an axis -- pixel
    ``p % patch_size`` of patch ``p // patch_size`` -- has its centre at
    ``((p + 0.5) / patch_size - 0.5) * patch_spacing``. At ``patch_size = 1`` this is
    exactly the patch coordinates. Same convention as the pixel registers of
    ``favyen/20260917-pixreg-v1_3``.

    Returns:
        ``[B, lat_h * lat_w, 2]`` row-major ``(row, col)`` coordinates.
    """
    lat_h, lat_w = latent_grid

    def axis(n: int) -> Tensor:
        pix = torch.arange(n, device=device, dtype=torch.float32)
        return ((pix + 0.5) / patch_size - 0.5) * patch_spacing

    grid_h, grid_w = torch.meshgrid(axis(lat_h), axis(lat_w), indexing="ij")
    grid = torch.stack([grid_h, grid_w], dim=-1).reshape(-1, 2)
    return grid.unsqueeze(0).expand(batch_size, -1, -1)


def joint_attention_allowed(
    cell_id: Tensor, is_latent: Tensor, valid: Tensor, latent_reads_all: bool = False
) -> Tensor:
    """Dense ``[B, L, L]`` boolean mask for the joint pattern (reference / CPU path).

    ``allowed[b, q, kv] = valid[b, kv] & (is_latent[b, kv] | cell_id[b, q] == cell_id[b, kv])``.
    Latents carry the cell id of their grid position, so the one rule gives both
    directions: latent -> (all latents + own cell), token -> (all latents + own cell).

    With ``latent_reads_all`` a latent QUERY additionally sees every valid key
    (``| is_latent[b, q]``): latents read the whole sample in one hop, as the pure
    Perceiver's reads do, while token rows keep the cell-local pattern.
    """
    same_cell = cell_id[:, :, None] == cell_id[:, None, :]
    allowed = is_latent[:, None, :] | same_cell
    if latent_reads_all:
        allowed = allowed | is_latent[:, :, None]
    return valid[:, None, :] & allowed


class JointLatentTransformer(nn.Module):
    """Latent grid + patch tokens under one structured attention (see module doc).

    Has the same call contract as :class:`olmoearth_pretrain.nn.flexi_vit.Perceiver`
    so it occupies the encoder's ``perceiver`` slot and everything downstream (student
    readout, supervision heads, decoder cross-attention) reads the same register grid.
    """

    def __init__(
        self,
        embedding_size: int,
        num_heads: int,
        mlp_ratio: float,
        joint_depth: int,
        latent_only_depth: int,
        token_mlp: bool,
        position_encoding: str,
        rope_base: float,
        rope_mixed_base: float,
        temporal_rope_dim_frac: float,
        rope_temporal_base: float | None,
        qk_norm: bool,
        drop_path: float = 0.0,
        latent_time_range: bool = False,
        latent_reads_all: bool = False,
        sort_by_cell: bool = True,
        pixel_latents: bool = False,
    ) -> None:
        """Initialize the joint transformer.

        Args:
            embedding_size: Width of tokens AND latents (they share one residual stream),
                which is also the register width the encoder ships.
            num_heads: Attention heads.
            mlp_ratio: MLP ratio of every block.
            joint_depth: Number of joint blocks over ``[tokens ; latents]``.
            latent_only_depth: Number of plain self-attention blocks over the latents
                alone, run after the joint blocks (tokens are frozen by then).
            token_mlp: If False, only the latent slice gets the block's MLP; tokens are
                updated by attention alone. Cuts a block's per-token cost from 12 d^2 to
                4 d^2 (the Q/K/V/out projections).
            position_encoding: The encoder's RoPE mode, applied to the whole joint
                sequence. Tokens carry ``(t, row, col)`` under 3D modes; latents get a
                fixed temporal coordinate (the mean of the valid tokens') and their grid
                ``(row, col)``.
            rope_base: Axial RoPE frequency base.
            rope_mixed_base: Mixed-RoPE frequency base.
            temporal_rope_dim_frac: Fraction of head dims given to the temporal axis.
            rope_temporal_base: Optional separate base for the temporal axis.
            qk_norm: QK normalisation in attention.
            drop_path: Stochastic depth rate.
            latent_time_range: If True, latents are encoded as temporal INTERVALS
                spanning the sample's visible-token time range (centred at its
                midpoint) instead of points at the mean time: their RoPE pairs are
                sinc-gated by how many turns the pair's temporal frequency makes across
                the window (see :func:`apply_3d_mixed_rope`), so a latent's attention
                over time is a soft box over its window rather than a peak at the
                centre. Requires a 3D mixed RoPE encoder. Tokens stay points.
            latent_reads_all: If True, latent queries attend to every valid token (and
                every latent), not only their own cell; token queries are unchanged.
                Restores the pure Perceiver's one-hop view of the whole sample for the
                latents while tokens keep cell-local attention. Costs
                ``M x (N + M)`` extra score pairs per block (~+5% MACs at the ws16/ps1
                eval shape, ~+1% at training shapes).
            sort_by_cell: If True, tokens are permuted so that each cell's tokens are
                contiguous before the joint blocks. Attention is permutation-equivariant
                given positions and mask move with the tokens, and only the latents are
                returned, so this changes nothing numerically; it only makes the
                FlexAttention block mask sparser (whole 128-blocks off the cell
                diagonal are skipped instead of computed).
            pixel_latents: If True, the latent grid is laid at PIXEL resolution -- one
                latent per pixel, ``patch_size**2`` per patch cell -- instead of one per
                patch. Each latent sits at its pixel centre inside the patch frame and
                carries the cell id of the patch that contains it, so the cell-local
                rule is unchanged: a pixel latent reads its own patch's tokens (or every
                token with ``latent_reads_all``), and a token sees every latent. At
                patch size 1 this is exactly the patch-grid model. Follows the
                pixel-register positions of ``favyen/20260917-pixreg-v1_3``.
        """
        super().__init__()
        if not PositionEncoding.is_rope(position_encoding):
            raise ValueError(
                "JointLatentTransformer needs a RoPE position_encoding: latents are "
                "clones of one vector and are told apart by their coordinates alone."
            )
        if joint_depth < 1:
            raise ValueError(
                "joint_depth must be >= 1 (otherwise nothing reads the tokens)"
            )
        self.register_dim = embedding_size
        self.embedding_size = embedding_size
        self.token_mlp = token_mlp
        self.position_encoding = position_encoding
        if latent_time_range and position_encoding != PositionEncoding.MIXED_3D_ROPE:
            raise ValueError(
                "latent_time_range needs position_encoding == MIXED_3D_ROPE, got "
                f"{position_encoding}"
            )
        self.latent_time_range = latent_time_range
        self.latent_reads_all = latent_reads_all
        self.sort_by_cell = sort_by_cell
        self.pixel_latents = pixel_latents
        self.register = nn.Parameter(torch.empty(1, embedding_size))
        nn.init.trunc_normal_(self.register, std=0.02)
        block_kwargs: dict[str, Any] = dict(
            qkv_bias=True,
            qk_norm=qk_norm,
            cross_attn=False,
            use_flash_attn=False,
            drop_path=drop_path,
            rope_base=rope_base,
            rope_mixed_base=rope_mixed_base,
            temporal_rope_dim_frac=temporal_rope_dim_frac,
            rope_temporal_base=rope_temporal_base,
        )
        self.joint_blocks = nn.ModuleList(
            [
                Block(
                    embedding_size,
                    num_heads,
                    mlp_ratio,
                    position_encoding=position_encoding,
                    **block_kwargs,
                )
                for _ in range(joint_depth)
            ]
        )
        # The latent grid is purely spatial, so the latent-only tail rotates over
        # (row, col) like the Perceiver's latent blocks.
        self.latent_blocks = nn.ModuleList(
            [
                Block(
                    embedding_size,
                    num_heads,
                    mlp_ratio,
                    position_encoding=PositionEncoding.AXIAL_2D_ROPE,
                    **block_kwargs,
                )
                for _ in range(latent_only_depth)
            ]
        )
        self.norm = nn.LayerNorm(embedding_size)

    @property
    def is_3d(self) -> bool:
        """Whether the joint blocks rotate over ``(t, row, col)``."""
        return PositionEncoding.is_3d_rope(self.position_encoding)

    @staticmethod
    def _sort_by_cell(
        patch_tokens: Tensor,
        patch_positions: Tensor,
        valid_tokens: Tensor,
        cell_ids: Tensor,
        n_latents: int,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Permute the token axis so each cell's tokens are contiguous.

        The encoder flattens each modality as ``(h, w, t[, bandset])`` and concatenates
        modalities, so a cell's tokens arrive as one run per modality, spread across
        the sequence. Sorting by cell (stable, so the within-cell modality/time order is
        kept) puts every run of a cell side by side: a 128-block of queries then needs
        only the key blocks covering the same handful of cells plus the latent blocks,
        and FlexAttention skips the rest outright. Non-spatial tokens (cell ``-1``)
        sort first as one run; invalid (padding) tokens are sent to the end so they
        cannot fragment a cell's run. Positions and validity move with the tokens, so
        the attention pattern and RoPE are unchanged, and the caller only consumes the
        latents, which never move.
        """
        sort_key = cell_ids.masked_fill(~valid_tokens, n_latents)
        order = torch.argsort(sort_key, dim=1, stable=True)
        gather_d = lambda x: torch.gather(  # noqa: E731
            x, 1, order[..., None].expand(-1, -1, x.shape[-1])
        )
        return (
            gather_d(patch_tokens),
            gather_d(patch_positions),
            torch.gather(valid_tokens, 1, order),
            torch.gather(cell_ids, 1, order),
        )

    def _attention_masks(
        self, cell_id: Tensor, is_latent: Tensor, valid: Tensor
    ) -> dict[str, Any]:
        """Attention-mask kwargs for :meth:`Attention.forward`.

        A FlexAttention block mask on CUDA, a dense ``[B, 1, L, L]`` SDPA mask elsewhere.
        """
        if cell_id.is_cuda:
            reads_all = self.latent_reads_all  # Python bool: specialised at trace time

            def mask_mod(b: Tensor, h: Tensor, q: Tensor, kv: Tensor) -> Tensor:
                allowed = is_latent[b, kv] | (cell_id[b, q] == cell_id[b, kv])
                if reads_all:
                    allowed = allowed | is_latent[b, q]
                return valid[b, kv] & allowed

            batch, length = cell_id.shape
            return {
                "block_mask": create_block_mask_cuda(
                    mask_mod, batch, length, cell_id.device
                )
            }
        return {
            "attn_mask": joint_attention_allowed(
                cell_id, is_latent, valid, latent_reads_all=self.latent_reads_all
            )[:, None]
        }

    def _joint_block(
        self,
        blk: Block,
        x: Tensor,
        n_tokens: int,
        rope_positions: Tensor,
        attn_kwargs: dict[str, Any],
        rope_extent: Tensor | None = None,
    ) -> Tensor:
        """One joint block.

        Masked attention over the whole sequence, then the MLP on all of it or on the
        latents only.
        """
        x = x + blk.drop_path(
            blk.ls1(
                blk.attn(
                    x=blk.norm1(x),
                    rope_positions=rope_positions,
                    rope_extent=rope_extent,
                    **attn_kwargs,
                )
            )
        )
        if self.token_mlp:
            return x + blk.drop_path(blk.ls2(blk.mlp(blk.norm2(x))))
        latents = x[:, n_tokens:]
        latents = latents + blk.drop_path(blk.ls2(blk.mlp(blk.norm2(latents))))
        return torch.cat([x[:, :n_tokens], latents], dim=1)

    def forward(
        self,
        patch_tokens: Tensor,
        patch_positions: Tensor,
        visible_mask: Tensor | None,
        cell_ids: Tensor,
        spatial_grid: tuple[int, int],
        grid_extent_positions: Tensor | None = None,
        patch_size: int = 1,
        patch_spacing: float | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Run the joint blocks and return the latent grid.

        Args:
            patch_tokens: ``[B, N, D]`` patch embeddings (already normed by the encoder).
                May include padding positions; ``visible_mask`` marks the real ones.
            patch_positions: ``[B, N, 3]`` ``(t, row, col)`` under 3D RoPE, or
                ``[B, N, 2]`` ``(row, col)`` under 2D RoPE, in the GSD-scaled frame.
            visible_mask: Bool ``[B, N]``, True where a token is real. None = all real.
            cell_ids: Long ``[B, N]``: row-major index of each token's patch cell in
                ``spatial_grid`` (``-1`` for non-spatial tokens, which then see the
                latents and each other but are read by no latent).
            spatial_grid: ``(n_h, n_w)`` patch grid the latent is cloned to.
            grid_extent_positions: Optional ``[B, M, >=2]`` positions whose LAST two
                coordinates give the full (pre-masking) patch extent the latent grid is
                laid over. Defaults to ``patch_positions``; pass the unmasked positions so
                a masking pattern that hides a whole edge row/column of cells cannot
                shrink the grid off the cells.
            patch_size: Patch size of this forward pass (pixel-latent mode only).
            patch_spacing: Distance between adjacent patch centres in the RoPE frame
                (the GSD ratio times ``rope_coordinate_scale``); required in pixel-latent
                mode to place the pixel centres.

        Returns:
            registers: ``[B, n_h, n_w, D]`` latent grid after the final norm, or
                ``[B, n_h * patch_size, n_w * patch_size, D]`` with pixel latents.
            register_positions: ``[B, n_h * n_w, 2]`` row-major ``(row, col)`` for the
                decoder's cross-attention.
        """
        batch_size, n_tokens, _ = patch_tokens.shape
        device = patch_tokens.device
        n_h, n_w = spatial_grid
        n_cells = n_h * n_w
        if self.pixel_latents:
            if patch_spacing is None:
                raise ValueError("pixel_latents requires patch_spacing")
            lat_h, lat_w = n_h * patch_size, n_w * patch_size
        else:
            lat_h, lat_w = n_h, n_w
        n_latents = lat_h * lat_w
        valid_tokens = (
            visible_mask.bool()
            if visible_mask is not None
            else torch.ones(batch_size, n_tokens, dtype=torch.bool, device=device)
        )
        cell_ids = cell_ids.long()
        if self.sort_by_cell:
            patch_tokens, patch_positions, valid_tokens, cell_ids = self._sort_by_cell(
                patch_tokens, patch_positions, valid_tokens, cell_ids, n_cells
            )

        # Latents: one vector cloned to the grid; identity comes from RoPE.
        latents = self.register.unsqueeze(0).expand(batch_size, n_latents, -1)
        extent_source = (
            grid_extent_positions
            if grid_extent_positions is not None
            else patch_positions
        )
        if self.pixel_latents:
            assert patch_spacing is not None
            latent_positions_2d = build_pixel_latent_positions(
                batch_size, (lat_h, lat_w), patch_size, patch_spacing, device
            )
        else:
            latent_positions_2d = build_register_grid_positions(
                extent_source[..., -2:], spatial_grid
            )
        rope_extent: Tensor | None = None
        if self.is_3d:
            t = patch_positions[..., 0]
            if self.latent_time_range:
                # Interval latents: centred on the midpoint of the visible tokens'
                # time range and spanning its full width; tokens stay points.
                big = torch.finfo(t.dtype).max
                t_min = torch.where(valid_tokens, t, torch.full_like(t, big)).amin(1)
                t_max = torch.where(valid_tokens, t, torch.full_like(t, -big)).amax(1)
                t_anchor = (t_min + t_max) / 2
                width = (t_max - t_min).clamp(min=0)
                rope_extent = torch.cat(
                    [
                        torch.zeros(batch_size, n_tokens, device=device, dtype=t.dtype),
                        width[:, None].expand(-1, n_latents),
                    ],
                    dim=1,
                )
            else:
                # Point latents: anchored at the mean valid token time so temporal
                # RoPE offsets to the tokens stay bounded.
                t_anchor = (t * valid_tokens).sum(1) / valid_tokens.sum(1).clamp(min=1)
            latent_positions = torch.cat(
                [
                    t_anchor[:, None, None].expand(-1, n_latents, 1),
                    latent_positions_2d,
                ],
                dim=-1,
            )
        else:
            latent_positions = latent_positions_2d

        x = torch.cat([patch_tokens, latents.to(patch_tokens.dtype)], dim=1)
        rope_positions = torch.cat([patch_positions, latent_positions], dim=1)
        if self.pixel_latents:
            # Each pixel latent belongs to the patch cell that contains it.
            rows = torch.arange(lat_h, device=device) // patch_size
            cols = torch.arange(lat_w, device=device) // patch_size
            latent_cell_ids = (rows[:, None] * n_w + cols[None, :]).reshape(1, -1)
            latent_cell_ids = latent_cell_ids.expand(batch_size, -1)
        else:
            latent_cell_ids = torch.arange(n_latents, device=device).expand(
                batch_size, -1
            )
        cell_id = torch.cat([cell_ids, latent_cell_ids], dim=1)
        is_latent = torch.cat(
            [
                torch.zeros(batch_size, n_tokens, dtype=torch.bool, device=device),
                torch.ones(batch_size, n_latents, dtype=torch.bool, device=device),
            ],
            dim=1,
        )
        valid = torch.cat(
            [
                valid_tokens,
                torch.ones(batch_size, n_latents, dtype=torch.bool, device=device),
            ],
            dim=1,
        )
        attn_kwargs = self._attention_masks(cell_id, is_latent, valid)

        for blk in self.joint_blocks:
            x = self._joint_block(
                blk, x, n_tokens, rope_positions, attn_kwargs, rope_extent=rope_extent
            )
        latents = x[:, n_tokens:]
        for blk in self.latent_blocks:
            latents = blk(x=latents, rope_positions=latent_positions_2d)
        out = self.norm(latents)
        out = rearrange(out, "b (h w) d -> b h w d", h=lat_h, w=lat_w)
        return out, latent_positions_2d


@dataclass
class JointLatentConfig(Config):
    """Configuration for :class:`JointLatentTransformer` in the encoder's perceiver slot.

    Set ``EncoderConfig.perceiver_config`` to one of these (with ``depth=0`` on the
    encoder, since the joint blocks replace the ViT blocks) to get a register grid
    produced by joint token-latent attention instead of ViT blocks + Perceiver reads.

    Args:
        register_dim: Width of the latent grid. Must equal the encoder
            ``embedding_size``: tokens and latents share one residual stream.
        joint_depth: Number of joint blocks over ``[tokens ; latents]``.
        latent_only_depth: Plain latent self-attention blocks after the joint ones.
        token_mlp: Whether tokens get the block MLP (True) or only the latents do.
        latent_time_range: Encode latents as temporal intervals over the sample's
            visible time range (sinc-gated RoPE) instead of points at the mean time.
            Requires ``position_encoding == rope_3d_mixed`` on the encoder.
        student_dims / student_output_norm: As on ``PerceiverConfig``: a detached
            low-dim student readout of the register grid.
        latent_reads_all: Latent queries attend to every valid token, not only their
            own cell (tokens unchanged). See :class:`JointLatentTransformer`.
        sort_by_cell: Cell-contiguous token layout inside the joint blocks. Numerically
            a no-op; makes the FlexAttention block mask sparser. Default True.
        pixel_latents: One latent per pixel instead of per patch (see
            :class:`JointLatentTransformer`); the register grid is then at pixel
            resolution, so pair it with ``SupervisionHeadConfig.spatial_unfold = 1``.
    """

    register_dim: int
    joint_depth: int = 12
    latent_only_depth: int = 0
    token_mlp: bool = True
    latent_time_range: bool = False
    student_dims: list[int] | None = None
    student_output_norm: bool = False
    latent_reads_all: bool = False
    sort_by_cell: bool = True
    pixel_latents: bool = False

    @property
    def sorted_student_dims(self) -> list[int] | None:
        """Student dims descending: ``[0]`` is the student width, the rest prefixes."""
        if not self.student_dims:
            return None
        return sorted(set(self.student_dims), reverse=True)

    def validate(self, *, encoder_num_heads: int, position_encoding: str) -> None:
        """Check the module against the encoder it will attach to."""
        if not PositionEncoding.is_rope(position_encoding):
            raise ValueError("JointLatentTransformer requires a RoPE position_encoding")
        if self.register_dim % encoder_num_heads != 0:
            raise ValueError(
                f"register_dim ({self.register_dim}) must be divisible by num_heads "
                f"({encoder_num_heads})"
            )
        if self.joint_depth < 1:
            raise ValueError("joint_depth must be >= 1")
        if (
            self.latent_time_range
            and position_encoding != PositionEncoding.MIXED_3D_ROPE
        ):
            raise ValueError(
                "latent_time_range requires the rope_3d_mixed position_encoding, got "
                f"{position_encoding}"
            )
        if self.latent_only_depth < 0:
            raise ValueError("latent_only_depth must be >= 0")
        if self.student_dims is not None:
            if len(self.student_dims) == 0 or any(d <= 0 for d in self.student_dims):
                raise ValueError(
                    "student_dims must be a non-empty list of positive ints, got "
                    f"{self.student_dims}"
                )

    def build(
        self,
        *,
        encoder_embedding_size: int,
        encoder_num_heads: int,
        mlp_ratio: float,
        position_encoding: str,
        rope_base: float,
        qk_norm: bool,
        rope_mixed_base: float = 10.0,
        temporal_rope_dim_frac: float = 0.25,
        rope_temporal_base: float | None = None,
        drop_path: float = 0.0,
    ) -> JointLatentTransformer:
        """Build the module for an encoder with these settings."""
        if self.register_dim != encoder_embedding_size:
            raise ValueError(
                "JointLatentConfig.register_dim must equal the encoder embedding_size "
                f"(tokens and latents share a residual stream): {self.register_dim} vs "
                f"{encoder_embedding_size}"
            )
        return JointLatentTransformer(
            embedding_size=encoder_embedding_size,
            num_heads=encoder_num_heads,
            mlp_ratio=mlp_ratio,
            joint_depth=self.joint_depth,
            latent_only_depth=self.latent_only_depth,
            token_mlp=self.token_mlp,
            position_encoding=position_encoding,
            rope_base=rope_base,
            rope_mixed_base=rope_mixed_base,
            temporal_rope_dim_frac=temporal_rope_dim_frac,
            rope_temporal_base=rope_temporal_base,
            qk_norm=qk_norm,
            drop_path=drop_path,
            latent_time_range=self.latent_time_range,
            latent_reads_all=self.latent_reads_all,
            sort_by_cell=self.sort_by_cell,
            pixel_latents=self.pixel_latents,
        )
