"""Dual-resolution encoder: coarse patch tokens + a lightweight per-pixel branch.

The coarse branch is the existing :class:`~olmoearth_pretrain.nn.flexi_vit.Encoder`
(full spatio-temporal attention at ``embedding_size``). Alongside it we keep a
per-pixel branch at a smaller ``pixel_embedding_size`` that, per transformer block,

1. runs *per-location* self-attention: at each individual pixel location, its tokens
   attend across all its ``(modality, band set, timestep)`` observations -- but never
   to *other* pixel locations. (Modality/band-set identity is carried by the separate
   per-``(modality, band set)`` projection weights + an additive temporal encoding.)
2. cross-attends to the coarse token of the ``(patch, timestep, band set)`` unit it
   belongs to.

Optionally the coarse branch also cross-attends to the (pooled) pixel features of
its unit, so the fine detail flows back into the coarse tokens that downstream
tasks consume.

**Masked-patch handling.** Masking is applied at the ``(patch, timestep, band set)``
*unit* granularity: within a unit either all ``P**2`` pixels are visible
(``ONLINE_ENCODER``) or all are masked. The encoder therefore operates on **ONLINE
units only** -- the coarse branch packs them via ``remove_masked_tokens`` and the
pixel branch gathers each ONLINE unit's ``P**2`` pixels into a packed
``[num_online_units, P**2, Dp]`` tensor. No pixel token is ever created for a masked
unit, so no compute is spent on (and no signal leaks from) masked patches.

This module follows the ``st_model.py`` pattern: it lives on its own and reuses the
shared building blocks from ``flexi_vit`` / ``attention`` rather than modifying them.

.. note::
    First-version limitations (documented, not silently ignored):

    * Attention uses SDPA with (small) padded group masks; flash-attn var-len packing
      is a later optimization. ``use_flash_attn`` and ``num_register_tokens`` are not
      supported yet.
    * Cross-attention uses absolute positions (no RoPE); the pixel self-attention gets
      its temporal ordering signal from an additive 1D sin/cos encoding. Fractional-
      coordinate RoPE for pixels is a future refinement.
    * The pixel branch only applies to spatial, multitemporal modalities, and
      cross-modality self-attention requires those modalities to share the same pixel
      grid (same ``G`` and ``P``).
"""

import logging
from dataclasses import dataclass
from typing import Any

import torch
from einops import rearrange
from torch import Tensor, nn
from torch.distributed.fsdp import fully_shard

from olmoearth_pretrain.data.constants import Modality, ModalitySpec
from olmoearth_pretrain.datatypes import (
    MaskedOlmoEarthSample,
    MaskValue,
    TokensAndMasks,
)
from olmoearth_pretrain.decorators import experimental
from olmoearth_pretrain.nn.attention import Attention, Mlp
from olmoearth_pretrain.nn.encodings import PositionEncoding, get_1d_sincos_pos_encoding
from olmoearth_pretrain.nn.flexi_vit import (
    BASE_GSD,
    Encoder,
    EncoderConfig,
    get_modalities_to_process,
)
from olmoearth_pretrain.nn.tokenization import TokenizationConfig

logger = logging.getLogger(__name__)


def get_pixel_branch_modalities(supported_modalities: list[ModalitySpec]) -> list[str]:
    """Return names of modalities that get a pixel branch (spatial + multitemporal)."""
    return [m.name for m in supported_modalities if m.is_spatial and m.is_multitemporal]


@dataclass
class PixelGroupings:
    """The set of ONLINE ``(patch, timestep, band set)`` units for one modality.

    Attributes:
        flat_idx: ``[num_online]`` indices of the ONLINE units into the flattened
            per-instance unit axis ``[B * U]`` (``U = G * G * T * band_sets``, ordered
            ``(g1, g2, t, band set)`` to match ``online_mask.reshape(B, U)``). This
            defines the packed order shared by the pixel tokens and the coarse tokens.
        p2: number of pixels per unit (``P**2``).
    """

    flat_idx: Tensor
    p2: int

    @property
    def num_online(self) -> int:
        """Number of ONLINE units for this modality."""
        return int(self.flat_idx.numel())


def build_pixel_groupings(online_mask: Tensor, p2: int) -> PixelGroupings:
    """Find the ONLINE ``(patch, timestep, band set)`` units for one modality.

    The coarse branch masks at the *unit* granularity -- a unit being one
    ``(spatial patch, timestep, band set)`` cell of the ``[B, G, G, T, band_sets]``
    mask. Every pixel inside an ONLINE unit is visible; every pixel inside a
    non-ONLINE unit is dropped. So all we need here is the flat list of ONLINE units;
    each of them expands to ``P**2`` pixels downstream.

    We flatten the mask to ``[B * U]`` with ``U = G * G * T * band_sets`` in row-major
    ``(instance, g1, g2, t, band set)`` order (the natural ``reshape`` order), then take
    the indices where the mask is ``ONLINE_ENCODER``. Keeping this canonical flat order
    means the pixel tokens (``pixels.reshape(B*U, ...)[flat_idx]``) and the coarse tokens
    (``coarse.reshape(B*U, ...)[flat_idx]``) line up index-for-index without any extra
    bookkeeping.

    Args:
        online_mask: ``[B, G, G, T, band_sets]`` bool, True where the unit is ONLINE.
        p2: pixels per unit (``P**2``), carried through for convenience.

    Returns:
        A :class:`PixelGroupings` holding the flat ONLINE-unit indices.
    """
    online_flat = online_mask.reshape(-1)  # [B * U], row-major (b, g1, g2, t, band set)
    flat_idx = torch.nonzero(online_flat, as_tuple=False).squeeze(-1)  # [num_online]
    return PixelGroupings(flat_idx=flat_idx, p2=p2)


@dataclass
class LocationGroupings:
    """Groups ONLINE units by *pixel location* for the pixel self-attention.

    A "location" is a single ``(instance, patch)`` cell; the ``P**2`` intra-patch
    offsets are handled as a batched dimension, so each physical pixel attends only to
    its own ``(modality, band set, timestep)`` observations. Units from *all* pixel
    modalities are pooled into these groups (that is what makes the self-attention span
    modalities), which requires every pixel modality to share the same ``G`` and ``P``.

    The fields describe how to scatter a packed ``[num_units, P**2, Dp]`` tensor (the
    concatenation of every modality's ONLINE units, in modality-iteration order) into a
    padded ``[num_locations, max_units, P**2, Dp]`` buffer and back.

    Attributes:
        order: ``[num_units]`` permutation sorting units by their location id.
        scatter_pos: ``[num_units]`` slot of each (sorted) unit in the flattened
            ``[num_locations * max_units]`` buffer.
        num_locations: number of distinct ``(instance, patch)`` cells with >=1 unit.
        max_units: max number of units at any single location (pad length).
        valid: ``[num_locations, max_units]`` bool mask of populated slots.
    """

    order: Tensor
    scatter_pos: Tensor
    num_locations: int
    max_units: int
    valid: Tensor


def build_location_groupings(location_ids: Tensor) -> LocationGroupings:
    """Group units by location id for padded per-location attention.

    Args:
        location_ids: ``[num_units]`` integer location id per unit (``b * G**2 + patch``),
            in the same order as the concatenated packed pixel tensor.

    Returns:
        A :class:`LocationGroupings`.
    """
    device = location_ids.device
    n = location_ids.numel()
    if n == 0:
        return LocationGroupings(
            order=location_ids.new_zeros(0),
            scatter_pos=location_ids.new_zeros(0),
            num_locations=0,
            max_units=0,
            valid=torch.zeros(0, 0, dtype=torch.bool, device=device),
        )
    # Sort units so that all units sharing a location are contiguous.
    order = torch.argsort(location_ids)
    sorted_ids = location_ids[order]
    # Consecutive-unique gives each location a 0..num_locations-1 group index.
    _, inv = torch.unique_consecutive(sorted_ids, return_inverse=True)
    num_locations = int(inv.max().item()) + 1
    counts = torch.bincount(inv, minlength=num_locations)
    max_units = int(counts.max().item())
    # Rank of each (sorted) unit within its location -> padded slot index.
    group_start = counts.cumsum(0) - counts
    rank = torch.arange(n, device=device) - group_start[inv]
    scatter_pos = inv * max_units + rank
    valid = torch.zeros(num_locations * max_units, dtype=torch.bool, device=device)
    valid[scatter_pos] = True
    return LocationGroupings(
        order=order,
        scatter_pos=scatter_pos,
        num_locations=num_locations,
        max_units=max_units,
        valid=valid.view(num_locations, max_units),
    )


class PixelPatchEmbed(nn.Module):
    """Per-pixel linear embedding for spatial, multitemporal modalities.

    Produces ``[B, H, W, T, band_sets, pixel_embedding_size]`` tokens (native input
    resolution -- one token per pixel, per timestep, per band set), plus a matching
    per-pixel mask. An additive 1D sin/cos temporal encoding is applied so the
    downstream self-attention has a timestep-ordering signal; modality/band-set identity
    is carried implicitly by the separate per-``(modality, band set)`` linear weights.
    """

    def __init__(
        self,
        supported_modality_names: list[str],
        pixel_embedding_size: int,
        tokenization_config: TokenizationConfig | None = None,
    ) -> None:
        """Initialize the pixel patch embedding.

        Args:
            supported_modality_names: Modalities to build a pixel branch for. Only
                spatial + multitemporal modalities are kept.
            pixel_embedding_size: Per-pixel embedding dimension.
            tokenization_config: Optional band-grouping config (shared with coarse).
        """
        super().__init__()
        self.pixel_embedding_size = pixel_embedding_size
        self.tokenization_config = tokenization_config or TokenizationConfig()
        specs = [Modality.get(n) for n in supported_modality_names]
        self.pixel_modality_names = get_pixel_branch_modalities(specs)

        self.per_modality_embeddings = nn.ModuleDict({})
        for modality in self.pixel_modality_names:
            bandset_indices = self.tokenization_config.get_bandset_indices(modality)
            self.per_modality_embeddings[modality] = nn.ModuleDict(
                {
                    self._embed_name(modality, idx): nn.Linear(
                        len(channel_set_idxs), pixel_embedding_size
                    )
                    for idx, channel_set_idxs in enumerate(bandset_indices)
                }
            )
            for idx, bandset in enumerate(bandset_indices):
                self.register_buffer(
                    self._buffer_name(modality, idx),
                    torch.tensor(bandset, dtype=torch.long),
                    persistent=False,
                )

    @staticmethod
    def _embed_name(modality: str, idx: int) -> str:
        return f"{modality}__{idx}"

    @staticmethod
    def _buffer_name(modality: str, idx: int) -> str:
        return f"{modality}__{idx}_pixel_buffer"

    def forward(
        self, input_data: MaskedOlmoEarthSample, patch_size: int
    ) -> dict[str, Tensor]:
        """Return per-pixel tokens and masks for each supported spatial modality.

        Args:
            input_data: The masked input sample.
            patch_size: Coarse patch size (unused for pixels; kept for symmetry).

        Returns:
            Dict mapping ``modality`` -> ``[B, H, W, T, band_sets, Dp]`` and
            ``modality_mask`` -> ``[B, H, W, T, band_sets]``.
        """
        output: dict[str, Tensor] = {}
        modalities = get_modalities_to_process(
            input_data.modalities, self.pixel_modality_names
        )
        for modality in modalities:
            modality_data = getattr(input_data, modality)
            modality_mask = getattr(
                input_data, input_data.get_masked_modality_name(modality)
            )
            num_bandsets = self.tokenization_config.get_num_bandsets(modality)
            tokens, masks = [], []
            for idx in range(num_bandsets):
                bands = getattr(self, self._buffer_name(modality, idx))
                inp = torch.index_select(modality_data, -1, bands)
                embed = self.per_modality_embeddings[modality][
                    self._embed_name(modality, idx)
                ]
                tokens.append(embed(inp))  # [B, H, W, T, Dp]
                masks.append(modality_mask[..., idx])  # [B, H, W, T]
            modality_tokens = torch.stack(tokens, dim=-2)  # [B, H, W, T, bs, Dp]
            modality_tokens = self._add_temporal_encoding(modality_tokens)
            output[modality] = modality_tokens
            output[input_data.get_masked_modality_name(modality)] = torch.stack(
                masks, dim=-1
            )  # [B, H, W, T, bs]
        return output

    def _add_temporal_encoding(self, tokens: Tensor) -> Tensor:
        """Add an additive 1D sin/cos temporal encoding over the time axis."""
        t = tokens.shape[3]
        enc = get_1d_sincos_pos_encoding(
            torch.arange(t, device=tokens.device, dtype=torch.float32),
            self.pixel_embedding_size,
        )  # [T, Dp]
        return tokens + enc[None, None, None, :, None, :]


class PixelTemporalBlock(nn.Module):
    """Transformer block that self-attends over a (padded) sequence axis with a mask.

    Used for the pixel branch's per-location attention: each row is one pixel location's
    ONLINE ``(modality, band set, timestep)`` observations (padded to ``max_units``), so
    tokens mix across observations of a single pixel, never across pixel locations.
    """

    def __init__(self, dim: int, num_heads: int, mlp_ratio: float) -> None:
        """Initialize the block.

        Args:
            dim: Pixel embedding dimension.
            num_heads: Number of attention heads.
            mlp_ratio: MLP hidden-dim ratio.
        """
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=True,
            position_encoding=PositionEncoding.ABSOLUTE,
        )
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = Mlp(dim, hidden_features=int(dim * mlp_ratio))

    def forward(self, x: Tensor, key_mask: Tensor | None) -> Tensor:
        """Apply self-attention over the sequence axis.

        Args:
            x: ``[N, S, D]`` tokens (S = padded sequence length).
            key_mask: Optional ``[N, S]`` bool (True = attend).

        Returns:
            ``[N, S, D]`` updated tokens.
        """
        x = x + self.attn(self.norm1(x), attn_mask=key_mask)
        x = x + self.mlp(self.norm2(x))
        return x


class CrossAttnBlock(nn.Module):
    """Generic cross-attention block: queries attend to (projected) key/value tokens."""

    def __init__(
        self, q_dim: int, kv_dim: int, num_heads: int, mlp_ratio: float
    ) -> None:
        """Initialize the cross-attention block.

        Args:
            q_dim: Query (and output) embedding dimension.
            kv_dim: Key/value source embedding dimension (projected to ``q_dim``).
            num_heads: Number of attention heads.
            mlp_ratio: MLP hidden-dim ratio.
        """
        super().__init__()
        self.kv_proj = nn.Linear(kv_dim, q_dim)
        self.norm_q = nn.LayerNorm(q_dim)
        self.norm_kv = nn.LayerNorm(q_dim)
        self.attn = Attention(
            q_dim,
            num_heads=num_heads,
            qkv_bias=True,
            cross_attn=True,
            position_encoding=PositionEncoding.ABSOLUTE,
        )
        self.norm_mlp = nn.LayerNorm(q_dim)
        self.mlp = Mlp(q_dim, hidden_features=int(q_dim * mlp_ratio))

    def forward(self, q: Tensor, kv: Tensor, key_mask: Tensor | None = None) -> Tensor:
        """Cross-attend ``q`` to ``kv``.

        Args:
            q: ``[B', N_q, q_dim]`` query tokens.
            kv: ``[B', N_k, kv_dim]`` key/value source tokens.
            key_mask: Optional ``[B', N_k]`` bool mask (True = attend).

        Returns:
            ``[B', N_q, q_dim]`` residual update (to be added by the caller).
        """
        kv = self.kv_proj(kv)
        out = self.attn(self.norm_q(q), y=self.norm_kv(kv), attn_mask=key_mask)
        out = out + self.mlp(self.norm_mlp(out))
        return out


@experimental("Dual-resolution (pixel branch) encoder is experimental.")
class DualResEncoder(Encoder):
    """Encoder with a coarse patch branch plus a lightweight per-pixel branch."""

    def __init__(
        self,
        *,
        pixel_embedding_size: int = 128,
        pixel_num_heads: int = 4,
        pixel_mlp_ratio: float = 4.0,
        pixel_cross_attn_to_coarse: bool = True,
        coarse_cross_attn_to_pixel: bool = True,
        **encoder_kwargs: Any,
    ) -> None:
        """Initialize the dual-resolution encoder.

        Args:
            pixel_embedding_size: Per-pixel embedding dimension (Dp).
            pixel_num_heads: Attention heads for the pixel branch.
            pixel_mlp_ratio: MLP ratio for the pixel branch.
            pixel_cross_attn_to_coarse: If True, pixels cross-attend to their unit's
                coarse token.
            coarse_cross_attn_to_pixel: If True, coarse tokens cross-attend to their
                unit's pixel features so fine detail flows into the output.
            **encoder_kwargs: Forwarded to :class:`Encoder` for the coarse branch.
        """
        if encoder_kwargs.get("use_flash_attn", False):
            raise NotImplementedError(
                "DualResEncoder does not support use_flash_attn yet."
            )
        if encoder_kwargs.get("num_register_tokens", 0):
            raise NotImplementedError(
                "DualResEncoder does not support register tokens yet."
            )
        super().__init__(**encoder_kwargs)

        self.pixel_embedding_size = pixel_embedding_size
        self.pixel_num_heads = pixel_num_heads
        self.pixel_cross_attn_to_coarse = pixel_cross_attn_to_coarse
        self.coarse_cross_attn_to_pixel = coarse_cross_attn_to_pixel
        self.pixel_modality_names = get_pixel_branch_modalities(
            self.supported_modalities
        )

        depth = len(self.blocks)
        coarse_dim = self.embedding_size
        coarse_heads = self.blocks[0].attn.num_heads

        self.pixel_embeddings = PixelPatchEmbed(
            self.supported_modality_names,
            pixel_embedding_size,
            tokenization_config=self.tokenization_config,
        )
        self.pixel_self_blocks = nn.ModuleList(
            [
                PixelTemporalBlock(
                    pixel_embedding_size, pixel_num_heads, pixel_mlp_ratio
                )
                for _ in range(depth)
            ]
        )
        self.pixel_to_coarse = (
            nn.ModuleList(
                [
                    CrossAttnBlock(
                        q_dim=pixel_embedding_size,
                        kv_dim=coarse_dim,
                        num_heads=pixel_num_heads,
                        mlp_ratio=pixel_mlp_ratio,
                    )
                    for _ in range(depth)
                ]
            )
            if pixel_cross_attn_to_coarse
            else None
        )
        self.coarse_to_pixel = (
            nn.ModuleList(
                [
                    CrossAttnBlock(
                        q_dim=coarse_dim,
                        kv_dim=pixel_embedding_size,
                        num_heads=coarse_heads,
                        mlp_ratio=1.0,
                    )
                    for _ in range(depth)
                ]
            )
            if coarse_cross_attn_to_pixel
            else None
        )

        for module in self._pixel_modules():
            module.apply(self._init_weights)

    def _pixel_modules(self) -> list[nn.Module]:
        modules: list[nn.Module] = [self.pixel_embeddings, self.pixel_self_blocks]
        if self.pixel_to_coarse is not None:
            modules.append(self.pixel_to_coarse)
        if self.coarse_to_pixel is not None:
            modules.append(self.coarse_to_pixel)
        return modules

    def forward(
        self,
        x: MaskedOlmoEarthSample,
        patch_size: int,
        input_res: int = BASE_GSD,
        token_exit_cfg: dict | None = None,
        fast_pass: bool = False,
    ) -> dict[str, Any]:
        """Process a masked sample into (pixel-enriched) coarse token representations."""
        if fast_pass and token_exit_cfg is not None:
            raise ValueError("token_exit_cfg cannot be set when fast_pass is True")

        coarse_patchified = self.patch_embeddings.forward(x, patch_size)
        pixel_patchified = self.pixel_embeddings.forward(x, patch_size)

        pixel_state: dict[str, dict[str, Any]] = {}
        if token_exit_cfg is None or any(
            exit_depth > 0 for exit_depth in token_exit_cfg.values()
        ):
            coarse_tokens_dict, pixel_state = self.apply_dual_attn(
                coarse_x=coarse_patchified,
                pixel_x=pixel_patchified,
                timestamps=x.timestamps,
                patch_size=patch_size,
                input_res=input_res,
                token_exit_cfg=token_exit_cfg,
                fast_pass=fast_pass,
            )
        else:
            coarse_tokens_dict = coarse_patchified

        output = TokensAndMasks(**coarse_tokens_dict)
        if self.embedding_projector is not None:
            output = self.embedding_projector(output)

        output_dict: dict[str, Any] = {"tokens_and_masks": output}
        if not fast_pass:
            output_dict["project_aggregated"] = self.project_and_aggregate(output)
        # The pixel branch is consumed by the pixel decoders (Phase 2+); it is NOT a
        # coarse-decoder kwarg, so the dual-res model wrapper must pop it before
        # calling the coarse decoder.
        output_dict["pixel_branch"] = pixel_state
        return output_dict

    def apply_dual_attn(
        self,
        coarse_x: dict[str, Tensor],
        pixel_x: dict[str, Tensor],
        timestamps: Tensor,
        patch_size: int,
        input_res: int,
        token_exit_cfg: dict[str, int] | None,
        fast_pass: bool,
    ) -> tuple[dict[str, Tensor], dict[str, dict[str, Any]]]:
        """Run interleaved coarse (packed full attn) + pixel (per-location) blocks.

        Returns the per-modality coarse tokens/masks and the pixel-branch state
        (final ONLINE-unit pixel reps + groupings) for the pixel decoders.
        """
        tokens_only_dict, original_masks_dict, dims_dict = (
            self.split_tokens_masks_and_dims(coarse_x)
        )
        exit_ids_seq = self.create_exit_seqs(
            tokens_only_dict, original_masks_dict, token_exit_cfg
        )
        exited_dense, _ = self.collapse_and_combine_hwtc(coarse_x)

        tokens_dict = self.composite_encodings.forward(
            tokens_only_dict, timestamps, patch_size, input_res
        )
        positions = self.build_rope_positions(
            tokens_only_dict,
            original_masks_dict,
            patch_size,
            input_res,
            timestamps=timestamps,
        )
        tokens_dict.update(original_masks_dict)
        coarse_dense, mask = self.collapse_and_combine_hwtc(tokens_dict)
        bool_mask = mask == MaskValue.ONLINE_ENCODER.value

        # Precompute the (fixed) coarse ONLINE packing.
        _, indices, new_mask, seq_lengths, max_seqlen = self.remove_masked_tokens(
            coarse_dense, bool_mask
        )
        positions_packed = None
        if positions is not None:
            positions_packed, _, _, _, _ = self.remove_masked_tokens(
                positions, bool_mask
            )
        attn_mask = self._maybe_get_attn_mask(new_mask, fast_pass)
        exit_packed = None
        exited_packed = None
        if exit_ids_seq is not None:
            exit_packed, _, _, _, _ = self.remove_masked_tokens(exit_ids_seq, bool_mask)
            exited_packed, _, _, _, _ = self.remove_masked_tokens(
                exited_dense, bool_mask
            )

        # Pixel branch bookkeeping: per-modality ONLINE units + a single cross-modality
        # per-location grouping used by the pixel self-attention.
        pixel_state = self._init_pixel_state(pixel_x, original_masks_dict)
        location_grouping = self._build_pixel_location_grouping(pixel_state)

        num_blocks = len(self.blocks)
        for i, blk in enumerate(self.blocks):
            packed = self._pack(coarse_dense, indices, new_mask, max_seqlen)
            if exit_ids_seq is not None and i > 0:
                exited_packed = torch.where(exit_packed == i, packed, exited_packed)
            packed = blk(x=packed, attn_mask=attn_mask, rope_positions=positions_packed)
            coarse_dense, _ = self.add_removed_tokens(packed, indices, new_mask)
            coarse_dense = self._interleave_pixels(
                block_idx=i,
                coarse_dense=coarse_dense,
                dims_dict=dims_dict,
                pixel_state=pixel_state,
                location_grouping=location_grouping,
            )

        packed = self._pack(coarse_dense, indices, new_mask, max_seqlen)
        if exit_ids_seq is not None:
            packed = torch.where(exit_packed == num_blocks, packed, exited_packed)
        packed = self.norm(packed)
        coarse_dense, _ = self.add_removed_tokens(packed, indices, new_mask)

        tokens_per_modality = self.split_and_expand_per_modality(
            coarse_dense, dims_dict
        )
        tokens_per_modality.update(original_masks_dict)
        return tokens_per_modality, pixel_state

    @staticmethod
    def _pack(
        dense: Tensor, indices: Tensor, new_mask: Tensor, max_seqlen: Tensor
    ) -> Tensor:
        """Gather ONLINE tokens into the packed [B, max_seqlen, D] layout."""
        d = dense.shape[-1]
        packed = dense.gather(1, indices[:, :, None].expand(-1, -1, d))[:, :max_seqlen]
        return packed * new_mask.unsqueeze(-1)

    def _init_pixel_state(
        self,
        pixel_x: dict[str, Tensor],
        original_masks_dict: dict[str, Tensor],
    ) -> dict[str, dict[str, Any]]:
        """Gather ONLINE-unit pixel tokens (per modality) into packed tensors."""
        state: dict[str, dict[str, Any]] = {}
        pixel_modalities = get_modalities_to_process(
            [m for m in pixel_x if not m.endswith("_mask")], self.pixel_modality_names
        )
        for modality in pixel_modalities:
            mask_name = MaskedOlmoEarthSample.get_masked_modality_name(modality)
            online = original_masks_dict[mask_name] == MaskValue.ONLINE_ENCODER.value
            b, g1, g2, t, bs = online.shape
            pixels = pixel_x[modality]  # [B, L, L, T, bs, Dp]
            p1 = pixels.shape[1] // g1
            p2 = pixels.shape[2] // g2
            # Group each unit's P**2 pixels: [B * U, P**2, Dp], U ordered (g1,g2,t,bs).
            units = rearrange(
                pixels,
                "b (g1 p1) (g2 p2) t bs d -> (b g1 g2 t bs) (p1 p2) d",
                g1=g1,
                g2=g2,
                p1=p1,
                p2=p2,
            )
            groupings = build_pixel_groupings(online, p1 * p2)
            packed_pixels = units[groupings.flat_idx]  # [num_online, P**2, Dp]
            state[modality] = {
                "groupings": groupings,
                "packed_pixels": packed_pixels,
                "shape": (b, g1, g2, t, bs),
                "p": (p1, p2),
            }
        return state

    def _build_pixel_location_grouping(
        self, pixel_state: dict[str, dict[str, Any]]
    ) -> LocationGroupings:
        """Build the cross-modality per-location grouping for pixel self-attention.

        Pools the ONLINE units of every pixel modality (in ``pixel_state`` iteration
        order, matching how :meth:`_interleave_pixels` concatenates their pixels) and
        groups them by location id ``b * G**2 + patch``. Requires all pixel modalities
        to share the same ``G`` and ``P``.
        """
        location_ids: list[Tensor] = []
        g2_count: int | None = None
        p2n: int | None = None
        for st in pixel_state.values():
            b, g1, g2, t, bs = st["shape"]
            p1, p2 = st["p"]
            if g2_count is None:
                g2_count, p2n = g1 * g2, p1 * p2
            elif g1 * g2 != g2_count or p1 * p2 != p2n:
                raise NotImplementedError(
                    "Cross-modality pixel self-attention requires all pixel modalities "
                    "to share the same spatial grid (G) and patch size (P)."
                )
            flat = st["groupings"].flat_idx
            u = g1 * g2 * t * bs
            within = flat % u
            b_of = flat // u
            patch = (within // bs) // t  # (g1 * g2) patch index within the instance
            location_ids.append(b_of * (g1 * g2) + patch)
        if not location_ids:
            return build_location_groupings(torch.empty(0, dtype=torch.long))
        return build_location_groupings(torch.cat(location_ids))

    def _interleave_pixels(
        self,
        block_idx: int,
        coarse_dense: Tensor,
        dims_dict: dict[str, tuple],
        pixel_state: dict[str, dict[str, Any]],
        location_grouping: LocationGroupings,
    ) -> Tensor:
        """Run the pixel branch + coarse<-pixel enrichment for one block."""
        if not pixel_state or location_grouping.num_locations == 0:
            return coarse_dense

        modalities = list(pixel_state.keys())

        # 1) Cross-modality per-location self-attention. Pool every modality's ONLINE
        #    units into one tensor (concat order == the location-grouping order), attend
        #    per pixel location across (modality, band set, timestep), then split back.
        all_pixels = torch.cat(
            [pixel_state[m]["packed_pixels"] for m in modalities], dim=0
        )
        all_pixels = self._pixel_location_attn(block_idx, all_pixels, location_grouping)
        offset = 0
        for modality in modalities:
            n = pixel_state[modality]["packed_pixels"].shape[0]
            pixel_state[modality]["packed_pixels"] = all_pixels[offset : offset + n]
            offset += n

        # 2/3) Per-modality cross-attention with the coarse tokens (per unit).
        coarse_per_mod = self.split_and_expand_per_modality(coarse_dense, dims_dict)
        for modality in modalities:
            st = pixel_state[modality]
            flat_idx = st["groupings"].flat_idx
            if flat_idx.numel() == 0:
                continue
            b, g1, g2, t, bs = st["shape"]
            coarse_dim = coarse_per_mod[modality].shape[-1]
            coarse_units = rearrange(
                coarse_per_mod[modality], "b g1 g2 t bs d -> (b g1 g2 t bs) d"
            )
            coarse_online = coarse_units[flat_idx]  # [num_online, Dc]
            pixels = st["packed_pixels"]  # [num_online, P**2, Dp]

            # Pixel <- coarse cross-attention (per unit; 1 coarse KV).
            if self.pixel_to_coarse is not None:
                pixels = pixels + self.pixel_to_coarse[block_idx](
                    pixels, coarse_online[:, None]
                )
                st["packed_pixels"] = pixels

            # Coarse <- pixel cross-attention (per unit; P**2 pixel KV), written back.
            # Placed last so every pixel module feeds the coarse output this block,
            # keeping all pixel params on the autograd graph (FSDP-safe).
            if self.coarse_to_pixel is not None:
                delta = self.coarse_to_pixel[block_idx](coarse_online[:, None], pixels)
                coarse_online = coarse_online + delta[:, 0]
                coarse_units = coarse_units.index_copy(0, flat_idx, coarse_online)
                coarse_per_mod[modality] = coarse_units.view(
                    b, g1, g2, t, bs, coarse_dim
                )

        return self._recombine_coarse(coarse_per_mod, dims_dict)

    def _pixel_location_attn(
        self, block_idx: int, pixels: Tensor, loc: LocationGroupings
    ) -> Tensor:
        """Self-attention over each pixel location's ONLINE observations.

        ``pixels`` is the concatenation of every modality's ONLINE units,
        ``[num_units, P**2, Dp]``. Units are scattered into a padded
        ``[num_locations, max_units, P**2, Dp]`` buffer; the ``P**2`` intra-patch offsets
        are then folded into the batch so each physical pixel attends across the
        ``(modality, band set, timestep)`` units at its location only.
        """
        num_units, p2, d = pixels.shape
        nl, mu = loc.num_locations, loc.max_units
        pix_sorted = pixels[loc.order]
        buf = pixels.new_zeros(nl * mu, p2, d)
        buf[loc.scatter_pos] = pix_sorted
        # -> [(nl * P**2), max_units, d]: one row per (location, pixel-offset).
        x = buf.view(nl, mu, p2, d).permute(0, 2, 1, 3).reshape(nl * p2, mu, d)
        key_mask = loc.valid.repeat_interleave(p2, dim=0)  # [nl*P**2, max_units]
        x = self.pixel_self_blocks[block_idx](x, key_mask)
        buf = x.reshape(nl, p2, mu, d).permute(0, 2, 1, 3).reshape(nl * mu, p2, d)
        out = pixels.new_empty(num_units, p2, d)
        out[loc.order] = buf[loc.scatter_pos]
        return out

    def _recombine_coarse(
        self, coarse_per_mod: dict[str, Tensor], dims_dict: dict[str, tuple]
    ) -> Tensor:
        """Collapse per-modality coarse tokens back to [B, N, D] (collapse order)."""
        chunks = [
            rearrange(coarse_per_mod[m], "b ... d -> b (...) d") for m in dims_dict
        ]
        return torch.cat(chunks, dim=1)

    def apply_fsdp(self, **fsdp_kwargs: Any) -> None:
        """Apply FSDP to the coarse and pixel branches."""
        for block in self.pixel_self_blocks:
            fully_shard(block, **fsdp_kwargs)
        if self.pixel_to_coarse is not None:
            for block in self.pixel_to_coarse:
                fully_shard(block, **fsdp_kwargs)
        if self.coarse_to_pixel is not None:
            for block in self.coarse_to_pixel:
                fully_shard(block, **fsdp_kwargs)
        super().apply_fsdp(**fsdp_kwargs)

    def apply_compile(self) -> None:
        """Apply torch.compile to the coarse and pixel blocks."""
        super().apply_compile()
        for block in self.pixel_self_blocks:
            block.compile(dynamic=False)


@dataclass
class DualResEncoderConfig(EncoderConfig):
    """Configuration for :class:`DualResEncoder`.

    Reuses every coarse-branch field from :class:`EncoderConfig` and adds the
    pixel-branch settings.
    """

    pixel_embedding_size: int = 128
    pixel_num_heads: int = 4
    pixel_mlp_ratio: float = 4.0
    pixel_cross_attn_to_coarse: bool = True
    coarse_cross_attn_to_pixel: bool = True

    def validate(self) -> None:
        """Validate the configuration."""
        super().validate()
        if self.pixel_embedding_size % self.pixel_num_heads != 0:
            raise ValueError(
                "pixel_embedding_size must be divisible by pixel_num_heads, got "
                f"{self.pixel_embedding_size} and {self.pixel_num_heads}"
            )
        if not any(
            m.is_spatial and m.is_multitemporal for m in self.supported_modalities
        ):
            raise ValueError(
                "DualResEncoder requires at least one spatial, multitemporal "
                "modality for the pixel branch."
            )

    def build(self) -> "DualResEncoder":
        """Build the dual-resolution encoder."""
        self.validate()
        kwargs = self.as_dict(exclude_none=True, recurse=False)
        kwargs.pop("supported_modality_names")
        kwargs["supported_modalities"] = self.supported_modalities
        logger.info(f"DualResEncoder kwargs: {kwargs}")
        return DualResEncoder(**kwargs)
