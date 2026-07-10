"""Dual-resolution encoder: coarse patch tokens + a lightweight per-pixel branch.

The coarse branch is the existing :class:`~olmoearth_pretrain.nn.flexi_vit.Encoder`
(full spatio-temporal attention at ``embedding_size``). Alongside it we keep a
per-pixel branch at a smaller ``pixel_embedding_size`` that, per transformer block,

runs *per-location* attention: at each individual pixel location, its tokens attend
across all its ``(modality, band set, timestep)`` pixel observations **and** (as extra
keys/values, projected to the pixel width) the coarse tokens of those same units --
but never across *other* pixel locations. Modality/band-set identity is carried by the
separate per-``(modality, band set)`` projection weights + an additive temporal
encoding.

Injecting the coarse context as extra keys of the location attention (rather than a
separate pixel -> coarse cross-attention) is deliberate: cross-attending to the single
coarse token of a pixel's own unit is degenerate -- softmax over one key is identically
1, so the "attention" reduces to a query-independent broadcast. As joint keys, the
pixel/coarse mixing is query-dependent even in the minimal fine-tuning configuration
(one modality, one timestep: two keys per pixel).

Optionally the coarse branch also cross-attends to the pixel features of its unit
(``P**2`` keys -- real attention), so the fine detail flows back into the coarse
tokens that downstream tasks consume.

**Vocabulary** (used consistently across this module and ``pixel_decoder.py``):

* A **unit** is one ``(spatial patch, timestep, band set)`` cell of one modality's
  ``[B, G1, G2, T, band_sets]`` grid. Its **flat index** is its position in the
  row-major ``reshape(B * U)`` flattening (``U = G1*G2*T*band_sets``); decode it with
  :func:`unit_grid_coords`. This canonical order is shared between the packed pixel
  tensors and the collapsed coarse token layout.
* A **location** is one ``(instance, spatial patch)`` cell, shared by all pixel
  modalities (they must live on the same grid).
* An **offset** is a pixel's position within its patch (``P**2`` per unit). Offsets
  are always a batched dimension: pixels never attend across offsets.

**Masked-patch handling.** Masking is applied at the unit granularity: within a unit
either all ``P**2`` pixels are visible (``ONLINE_ENCODER``) or all are masked. The
encoder therefore operates on **ONLINE units only** -- the coarse branch packs them via
``remove_masked_tokens`` and the pixel branch gathers each ONLINE unit's ``P**2``
pixels into a packed ``[num_online_units, P**2, Dp]`` tensor. No pixel token is ever
created for a masked unit, so no compute is spent on (and no signal leaks from) masked
patches.

**FSDP.** The pixel modules are intentionally *not* wrapped as individual FSDP units
(see :meth:`DualResEncoder.apply_fsdp`): they are invoked a data-dependent number of
times per step, which would desync the all-gather sequence across ranks and deadlock
NCCL. They live in the enclosing model's root FSDP unit instead.

This module follows the ``st_model.py`` pattern: it lives on its own and reuses the
shared building blocks from ``flexi_vit`` / ``attention`` rather than modifying them.

.. note::
    First-version limitations (documented, not silently ignored):

    * Attention uses SDPA with (small) padded group masks; flash-attn var-len packing
      is a later optimization. ``use_flash_attn``, ``num_register_tokens`` and
      ``torch.compile`` of the pixel blocks (their shapes change every batch) are not
      supported yet. Because CUDA caps a launch grid dimension at 65535 and the fused
      SDPA kernels map the attention batch onto that dimension, the pixel branch's
      large attention batches are split into chunks (see :func:`chunked_batch_attn`).
    * Cross-attention uses absolute positions (no RoPE); the pixel self-attention gets
      its temporal ordering signal from an additive 1D sin/cos encoding. Fractional-
      coordinate RoPE for pixels is a future refinement.
    * The pixel branch only applies to spatial, multitemporal modalities, and
      cross-modality self-attention requires those modalities to share the same pixel
      grid (same ``G`` and ``P``).
"""

import logging
import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch
import torch.utils.checkpoint
from einops import rearrange
from torch import Tensor, nn

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

# Max independent attention problems (batch rows) per kernel launch. CUDA caps the
# grid ``.y``/``.z`` dimensions at 65535, and both the fused SDPA kernels and flash-attn
# map the attention batch onto one of those dims -- so a single attention call with more
# than ~65k rows raises ``CUDA error: invalid configuration argument``. The pixel branch
# routinely blows past that (one row per pixel or per ONLINE unit), so we chunk it.
ATTN_BATCH_CHUNK = 32768


def chunked_batch_attn(
    fn: Callable[..., Tensor],
    *tensors: Tensor | None,
    chunk: int = ATTN_BATCH_CHUNK,
) -> Tensor:
    """Run an attention block ``fn`` over dim-0 chunks of ``tensors`` and concatenate.

    Every row of the attention batch is an independent problem (no row attends to
    another), so slicing the leading dimension and concatenating the outputs is exact.
    Chunking keeps each kernel launch under CUDA's 65535 grid-dimension cap. ``None``
    tensors (e.g. an absent mask) are passed through to every chunk unchanged.

    Args:
        fn: Callable applied to a (chunk of the) argument tensors, returning a tensor
            whose leading dimension matches the inputs'.
        *tensors: Batched tensors (or ``None``) sharing the same leading dimension.
        chunk: Max rows per call.

    Returns:
        The concatenation of ``fn``'s per-chunk outputs along dim 0.
    """
    batch = next(t.shape[0] for t in tensors if t is not None)
    if batch <= chunk:
        return fn(*tensors)
    outputs = [
        fn(*(None if t is None else t[start : start + chunk] for t in tensors))
        for start in range(0, batch, chunk)
    ]
    return torch.cat(outputs, dim=0)


def get_pixel_branch_modalities(supported_modalities: list[ModalitySpec]) -> list[str]:
    """Return names of modalities that get a pixel branch (spatial + multitemporal)."""
    return [m.name for m in supported_modalities if m.is_spatial and m.is_multitemporal]


@dataclass
class UnitCoords:
    """Grid coordinates of flat unit indices (see :func:`unit_grid_coords`)."""

    instance: Tensor
    """Instance (batch) index of each unit."""
    patch: Tensor
    """Flat spatial patch index within the instance, in ``[0, G1*G2)``."""
    t: Tensor
    """Timestep index."""
    bandset: Tensor
    """Band-set index."""
    within: Tensor
    """Flat unit index within the instance, in ``[0, U)``."""


def unit_grid_coords(
    flat_idx: Tensor, grid: tuple[int, int, int, int, int]
) -> UnitCoords:
    """Decode flat unit indices into their grid coordinates.

    ``flat_idx`` indexes the row-major ``reshape(B * U)`` flattening of a modality's
    ``[B, G1, G2, T, band_sets]`` unit grid (``U = G1*G2*T*band_sets``). Every consumer
    of the packed unit order (encoder groupings, pixel decoders) derives positions with
    this one helper, so the layout convention lives in a single place.

    Args:
        flat_idx: ``[N]`` flat unit indices.
        grid: The ``(B, G1, G2, T, band_sets)`` grid shape.

    Returns:
        The per-unit :class:`UnitCoords`.
    """
    _, g1, g2, t, band_sets = grid
    u = g1 * g2 * t * band_sets
    within = flat_idx % u
    return UnitCoords(
        instance=flat_idx // u,
        patch=within // (t * band_sets),
        t=(within // band_sets) % t,
        bandset=within % band_sets,
        within=within,
    )


@dataclass
class PixelModalityState:
    """Packed pixel tokens for one modality's ONLINE units.

    ``pixels`` holds the ``P**2`` pixels of every ONLINE unit in ascending
    flat-unit-index order. The encoder fills it with the final (post-block) pixel
    representations, which the pixel decoders (``pixel_decoder.py``) consume.
    """

    pixels: Tensor
    """``[num_online, P**2, Dp]`` pixel tokens of the ONLINE units."""
    flat_idx: Tensor
    """``[num_online]`` flat indices into the ``[B * U]`` unit axis (ascending)."""
    grid: tuple[int, int, int, int, int]
    """The ``(B, G1, G2, T, band_sets)`` unit grid shape."""
    patch_shape: tuple[int, int]
    """``(P1, P2)`` pixels per patch side."""

    @property
    def num_online(self) -> int:
        """Number of ONLINE units for this modality."""
        return int(self.flat_idx.numel())

    @property
    def pixels_per_unit(self) -> int:
        """Number of pixels per unit (``P**2``)."""
        return self.patch_shape[0] * self.patch_shape[1]


@dataclass
class LocationGroupings:
    """Groups ONLINE units by *pixel location* for the pixel self-attention.

    The fields describe how to scatter a packed ``[num_units, P**2, Dp]`` tensor (the
    concatenation of every modality's ONLINE units) into a padded
    ``[num_locations, max_units, P**2, Dp]`` buffer and back. Grouping by location is
    what makes the self-attention span modalities while never mixing pixel locations.

    Attributes:
        order: ``[num_units]`` permutation sorting units by their location id.
        scatter_pos: ``[num_units]`` slot of each (sorted) unit in the flattened
            ``[num_locations * max_units]`` buffer.
        num_locations: number of distinct locations with >= 1 unit.
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
        location_ids: ``[num_units]`` integer location id per unit
            (``instance * G1*G2 + patch``), in packed unit order.

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


@dataclass
class PixelBranchContext:
    """Fixed per-forward bookkeeping for the pixel branch.

    Everything here is computed once, before the transformer blocks run, so the
    per-block work in :meth:`DualResEncoder._interleave_pixels` is pure tensor compute.
    All cross-modality tensors concatenate the modalities in ``states`` iteration
    order; ``split_sizes`` splits them back.

    Attributes:
        states: Per-modality packed ONLINE pixel tokens (updated block by block).
        split_sizes: ``num_online`` per modality.
        loc: Per-location grouping of the concatenated units (self-attention).
        coarse_idx: ``[total_units]`` position of each unit's coarse token in the
            flattened ``[B * N]`` coarse token tensor (``N`` = tokens per instance),
            so cross-attention gathers/scatters coarse partners with one index op.
    """

    states: dict[str, PixelModalityState]
    split_sizes: list[int]
    loc: LocationGroupings
    coarse_idx: Tensor


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
        # Cast to the token dtype: under FSDP mixed precision the tokens are bf16 and
        # a float32 addition would silently promote the whole pixel branch to float32
        # (and then fail against the bf16 LayerNorm/Linear params downstream).
        enc = enc.to(tokens.dtype)
        return tokens + enc[None, None, None, :, None, :]


class PixelTemporalBlock(nn.Module):
    """Transformer block over a (padded) sequence axis with optional extra keys.

    Used for the pixel branch's per-location attention: each row is one pixel location's
    ONLINE ``(modality, band set, timestep)`` observations (padded to ``max_units``), so
    tokens mix across observations of a single pixel, never across pixel locations.
    ``extra_kv`` appends additional key/value tokens that the queries can attend to but
    that are not themselves updated -- the coarse tokens of the row's units.
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
        # cross_attn=True gives separate q vs k/v inputs; with extra_kv=None the k/v
        # input is the same normed tensor as q, which is exactly self-attention.
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=True,
            cross_attn=True,
            position_encoding=PositionEncoding.ABSOLUTE,
        )
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = Mlp(dim, hidden_features=int(dim * mlp_ratio))

    def forward(
        self, x: Tensor, key_mask: Tensor | None, extra_kv: Tensor | None = None
    ) -> Tensor:
        """Attend over the sequence axis (and optional extra keys).

        Args:
            x: ``[N, S, D]`` tokens (S = padded sequence length).
            key_mask: Optional ``[N, S + S_extra]`` bool (True = attend), covering the
                concatenated key sequence.
            extra_kv: Optional ``[N, S_extra, D]`` additional key/value tokens
                (already normalized/projected; appended after the ``x`` keys).

        Returns:
            ``[N, S, D]`` updated tokens.
        """
        # Broadcastable [N, 1, 1, S] form: N is huge here, so materializing the
        # [N, heads, S, S] mask (what Attention does with a 2-dim mask) would dominate
        # the pixel branch's memory.
        if key_mask is not None:
            key_mask = key_mask[:, None, None, :]
        xn = self.norm1(x)
        kv = xn if extra_kv is None else torch.cat([xn, extra_kv], dim=1)
        x = x + self.attn(xn, y=kv, attn_mask=key_mask)
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
            The residual update (to be added by the caller): ``[B', N_q, q_dim]``, or
            ``[B', 1, q_dim]`` (broadcastable against the queries) when the single-key
            fast path applies.
        """
        if kv.shape[1] == 1 and key_mask is None and self.attn.attn_drop.p == 0.0:
            return self._single_key_forward(kv)
        # Broadcastable [B', 1, 1, N_k] form; see PixelTemporalBlock.forward.
        if key_mask is not None:
            key_mask = key_mask[:, None, None, :]
        kv = self.kv_proj(kv)
        out = self.attn(self.norm_q(q), y=self.norm_kv(kv), attn_mask=key_mask)
        out = out + self.mlp(self.norm_mlp(out))
        return out

    def _single_key_forward(self, kv: Tensor) -> Tensor:
        """Exact fast path for a single key/value token per row.

        Softmax over one key is identically 1 whatever the query--key logits are, so
        the attention output is just the (projected) value: ``proj(v(norm(kv_proj)))``,
        independent of the queries, and identical for every query of the row. Compute
        it once per row at ``[B', 1, q_dim]`` and let the caller's residual add
        broadcast it, instead of running SDPA over every query token. This applies to
        the coarse <- pixel cross-attention at patch size 1 (one pixel per unit) and to
        single-key groups in the pixel reconstruction decoder. The query/key
        projections receive zero gradient on this path -- exactly as they do through
        the softmax in the general path.
        """
        out = self.attn.proj(self.attn.v(self.norm_kv(self.kv_proj(kv))))
        out = self.attn.proj_drop(out)
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
        pixel_grad_checkpointing: bool = True,
        **encoder_kwargs: Any,
    ) -> None:
        """Initialize the dual-resolution encoder.

        Args:
            pixel_embedding_size: Per-pixel embedding dimension (Dp).
            pixel_num_heads: Attention heads for the pixel branch.
            pixel_mlp_ratio: MLP ratio for the pixel branch.
            pixel_cross_attn_to_coarse: If True, the coarse tokens of each location's
                ONLINE units are added (projected to the pixel width) as extra
                keys/values of the per-location pixel attention, so pixels read coarse
                context with query-dependent weights. (A separate cross-attention to
                the unit's single coarse token would be degenerate -- softmax over one
                key -- hence this formulation.)
            coarse_cross_attn_to_pixel: If True, coarse tokens cross-attend to their
                unit's pixel features so fine detail flows into the output.
            pixel_grad_checkpointing: Recompute each pixel-branch block in backward
                instead of storing its activations. The pixel branch holds one token
                per pixel (up to ``P**2 = 64`` times the coarse token count), so
                storing all its intermediate activations OOMs at large patch sizes;
                checkpointing keeps only each block's inputs. The pixel branch has no
                dropout/drop-path, so recomputation is deterministic.
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
        self.pixel_grad_checkpointing = pixel_grad_checkpointing
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
        # Per-block projection of coarse tokens into the pixel width, used as extra
        # keys/values of the per-location pixel attention.
        self.coarse_kv_projs = (
            nn.ModuleList(
                [
                    nn.Sequential(
                        nn.LayerNorm(coarse_dim),
                        nn.Linear(coarse_dim, pixel_embedding_size),
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
        if self.coarse_kv_projs is not None:
            modules.append(self.coarse_kv_projs)
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

        pixel_state: dict[str, PixelModalityState] = {}
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
        # The pixel branch is consumed by the pixel decoders; it is NOT a coarse-decoder
        # kwarg, so the dual-res model wrapper must pop it before calling the decoder.
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
    ) -> tuple[dict[str, Tensor], dict[str, PixelModalityState]]:
        """Run interleaved coarse (packed full attn) + pixel (per-location) blocks.

        Returns the per-modality coarse tokens/masks and the pixel-branch state
        (final ONLINE-unit pixel reps per modality) for the pixel decoders.
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

        # Precompute the pixel-branch bookkeeping (packed ONLINE pixels + groupings).
        # ``pixels`` (all modalities' ONLINE units, concatenated) is carried through
        # the block loop alongside the coarse tokens and split back into per-modality
        # state once at the end.
        pixel_ctx = self._build_pixel_context(pixel_x, original_masks_dict, dims_dict)
        pixels = None
        if pixel_ctx is not None:
            pixels = torch.cat([st.pixels for st in pixel_ctx.states.values()], dim=0)

        num_blocks = len(self.blocks)
        for i, blk in enumerate(self.blocks):
            packed = self._pack(coarse_dense, indices, new_mask, max_seqlen)
            if exit_ids_seq is not None and i > 0:
                exited_packed = torch.where(exit_packed == i, packed, exited_packed)
            packed = blk(x=packed, attn_mask=attn_mask, rope_positions=positions_packed)
            coarse_dense, _ = self.add_removed_tokens(packed, indices, new_mask)
            if pixel_ctx is not None:
                if self.pixel_grad_checkpointing and torch.is_grad_enabled():
                    coarse_dense, pixels = torch.utils.checkpoint.checkpoint(
                        self._interleave_pixels,
                        i,
                        coarse_dense,
                        pixels,
                        pixel_ctx,
                        use_reentrant=False,
                    )
                else:
                    coarse_dense, pixels = self._interleave_pixels(
                        i, coarse_dense, pixels, pixel_ctx
                    )

        if pixel_ctx is not None:
            assert pixels is not None
            for state, updated in zip(
                pixel_ctx.states.values(), pixels.split(pixel_ctx.split_sizes)
            ):
                state.pixels = updated

        packed = self._pack(coarse_dense, indices, new_mask, max_seqlen)
        if exit_ids_seq is not None:
            packed = torch.where(exit_packed == num_blocks, packed, exited_packed)
        packed = self.norm(packed)
        coarse_dense, _ = self.add_removed_tokens(packed, indices, new_mask)

        tokens_per_modality = self.split_and_expand_per_modality(
            coarse_dense, dims_dict
        )
        tokens_per_modality.update(original_masks_dict)
        pixel_state = pixel_ctx.states if pixel_ctx is not None else {}
        return tokens_per_modality, pixel_state

    @staticmethod
    def _pack(
        dense: Tensor, indices: Tensor, new_mask: Tensor, max_seqlen: Tensor
    ) -> Tensor:
        """Gather ONLINE tokens into the packed [B, max_seqlen, D] layout."""
        d = dense.shape[-1]
        packed = dense.gather(1, indices[:, :, None].expand(-1, -1, d))[:, :max_seqlen]
        return packed * new_mask.unsqueeze(-1)

    def _build_pixel_context(
        self,
        pixel_x: dict[str, Tensor],
        original_masks_dict: dict[str, Tensor],
        dims_dict: dict[str, tuple],
    ) -> PixelBranchContext | None:
        """Pack each modality's ONLINE-unit pixels and precompute all groupings.

        For every pixel modality this gathers the ``P**2`` pixels of each ONLINE unit
        into a packed ``[num_online, P**2, Dp]`` tensor (ascending flat-unit-index
        order). Across modalities it additionally precomputes, in concatenation order:

        * the location id of every unit (for the per-location self-attention), and
        * the index of every unit's coarse token in the collapsed coarse layout
          (``dims_dict`` order matches ``collapse_and_combine_hwtc``), so the
          cross-attentions can gather/scatter coarse partners with a single index op.

        Returns ``None`` when no modality has any ONLINE unit.
        """
        # Where each modality's tokens start on the collapsed coarse token axis.
        coarse_offsets: dict[str, int] = {}
        tokens_per_instance = 0
        for modality, dims in dims_dict.items():
            coarse_offsets[modality] = tokens_per_instance
            tokens_per_instance += math.prod(dims[1:-1])

        states: dict[str, PixelModalityState] = {}
        location_ids: list[Tensor] = []
        coarse_idx: list[Tensor] = []
        shared_grid: tuple[int, int] | None = None  # (G1*G2, P**2)
        pixel_modalities = get_modalities_to_process(
            [m for m in pixel_x if not m.endswith("_mask")], self.pixel_modality_names
        )
        for modality in pixel_modalities:
            mask_name = MaskedOlmoEarthSample.get_masked_modality_name(modality)
            online = original_masks_dict[mask_name] == MaskValue.ONLINE_ENCODER.value
            b, g1, g2, t, bs = online.shape
            pixels = pixel_x[modality]  # [B, H, W, T, bs, Dp]
            p1 = pixels.shape[1] // g1
            p2 = pixels.shape[2] // g2
            if shared_grid is None:
                shared_grid = (g1 * g2, p1 * p2)
            elif shared_grid != (g1 * g2, p1 * p2):
                raise NotImplementedError(
                    "Cross-modality pixel self-attention requires all pixel modalities "
                    "to share the same spatial grid (G) and patch size (P)."
                )
            # Group each unit's P**2 pixels: [B * U, P**2, Dp], U ordered (g1,g2,t,bs).
            units = rearrange(
                pixels,
                "b (g1 p1) (g2 p2) t bs d -> (b g1 g2 t bs) (p1 p2) d",
                g1=g1,
                g2=g2,
                p1=p1,
                p2=p2,
            )
            flat_idx = torch.nonzero(online.reshape(-1), as_tuple=False).squeeze(-1)
            state = PixelModalityState(
                pixels=units[flat_idx],
                flat_idx=flat_idx,
                grid=(b, g1, g2, t, bs),
                patch_shape=(p1, p2),
            )
            coords = unit_grid_coords(flat_idx, state.grid)
            location_ids.append(coords.instance * (g1 * g2) + coords.patch)
            coarse_idx.append(
                coords.instance * tokens_per_instance
                + coarse_offsets[modality]
                + coords.within
            )
            states[modality] = state

        if not states or sum(st.num_online for st in states.values()) == 0:
            return None
        return PixelBranchContext(
            states=states,
            split_sizes=[st.num_online for st in states.values()],
            loc=build_location_groupings(torch.cat(location_ids)),
            coarse_idx=torch.cat(coarse_idx),
        )

    def _interleave_pixels(
        self,
        block_idx: int,
        coarse_dense: Tensor,
        pixels: Tensor,
        ctx: PixelBranchContext,
    ) -> tuple[Tensor, Tensor]:
        """Run one pixel-branch block and exchange information with the coarse tokens.

        All modalities' ONLINE units are processed as one concatenated tensor
        (``pixels``, in ``ctx.states`` order), so each module below runs exactly once
        per block:

        1. per-location attention across ``(modality, band set, timestep)`` pixel
           observations, with the units' coarse tokens (projected to the pixel width)
           as extra keys/values -- this is how pixels read coarse context;
        2. coarse <- pixel: each ONLINE coarse token queries its unit's pixels and the
           result is written back into ``coarse_dense``. Placed last so every pixel
           module feeds the coarse output each block (keeping all pixel params on the
           autograd graph).

        This method is (optionally) wrapped in gradient checkpointing by the caller,
        so it must stay a pure ``(coarse_dense, pixels) -> (coarse_dense, pixels)``
        function of its tensor inputs.
        """
        b, n, d = coarse_dense.shape
        coarse_flat = coarse_dense.reshape(b * n, d)
        coarse_units = coarse_flat[ctx.coarse_idx]  # [total_units, Dc]

        coarse_kv = None
        if self.coarse_kv_projs is not None:
            coarse_kv = self.coarse_kv_projs[block_idx](coarse_units)  # [units, Dp]
        pixels = self._pixel_location_attn(block_idx, pixels, ctx.loc, coarse_kv)

        if self.coarse_to_pixel is not None:
            delta = chunked_batch_attn(
                self.coarse_to_pixel[block_idx], coarse_units[:, None], pixels
            )
            coarse_flat = coarse_flat.index_copy(
                0, ctx.coarse_idx, coarse_units + delta[:, 0]
            )
            coarse_dense = coarse_flat.view(b, n, d)

        return coarse_dense, pixels

    def _pixel_location_attn(
        self,
        block_idx: int,
        pixels: Tensor,
        loc: LocationGroupings,
        coarse_kv: Tensor | None,
    ) -> Tensor:
        """Attention over each pixel location's ONLINE observations (+ coarse keys).

        ``pixels`` is the concatenation of every modality's ONLINE units,
        ``[num_units, P**2, Dp]``. Units are scattered into a padded
        ``[num_locations, max_units, P**2, Dp]`` buffer; the ``P**2`` intra-patch offsets
        are then folded into the batch so each physical pixel attends across the
        ``(modality, band set, timestep)`` units at its location only.

        ``coarse_kv`` (``[num_units, Dp]``, the units' coarse tokens projected to the
        pixel width) is scattered with the same grouping and appended as extra
        keys/values, so every pixel also attends to the coarse context of its location.
        The coarse keys are shared by the location's ``P**2`` offsets (expanded view,
        not materialized).
        """
        num_units, p2, d = pixels.shape
        nl, mu = loc.num_locations, loc.max_units
        pix_sorted = pixels[loc.order]
        buf = pixels.new_zeros(nl * mu, p2, d)
        buf[loc.scatter_pos] = pix_sorted
        # -> [(nl * P**2), max_units, d]: one row per (location, pixel-offset).
        x = buf.view(nl, mu, p2, d).permute(0, 2, 1, 3).reshape(nl * p2, mu, d)
        key_mask = loc.valid  # [nl, max_units]
        extra_kv = None
        if coarse_kv is not None:
            cbuf = coarse_kv.new_zeros(nl * mu, d)
            cbuf[loc.scatter_pos] = coarse_kv[loc.order]
            extra_kv = (
                cbuf.view(nl, 1, mu, d).expand(nl, p2, mu, d).reshape(nl * p2, mu, d)
            )
            # Coarse-key slots are valid exactly where the unit slots are.
            key_mask = torch.cat([key_mask, key_mask], dim=1)  # [nl, 2 * max_units]
        key_mask = key_mask.repeat_interleave(p2, dim=0)  # [nl*P**2, ...]
        x = chunked_batch_attn(self.pixel_self_blocks[block_idx], x, key_mask, extra_kv)
        buf = x.reshape(nl, p2, mu, d).permute(0, 2, 1, 3).reshape(nl * mu, p2, d)
        # Allocate ``out`` from ``buf`` (not ``pixels``): under autocast the attention
        # block returns a lower-precision dtype (e.g. bf16) than the float32 input, and
        # index_put requires the source and destination dtypes to match.
        out = buf.new_empty(num_units, p2, d)
        out[loc.order] = buf[loc.scatter_pos]
        return out

    def apply_fsdp(self, **fsdp_kwargs: Any) -> None:
        """Apply FSDP to the coarse blocks; keep pixel modules in the root unit.

        The pixel modules are deliberately NOT wrapped as their own FSDP units. FSDP2
        issues one all-gather per wrapped-module forward call, and the pixel modules
        are invoked a *data-dependent* number of times per step (once per
        :func:`chunked_batch_attn` chunk, and only when a rank has ONLINE pixel units).
        Ranks see different pixel counts, so per-module wrapping desyncs the collective
        sequence across ranks and deadlocks NCCL. Left unwrapped, the pixel params join
        the enclosing model's root FSDP unit (all-gathered once per model forward, like
        the patch embeddings), which is rank-uniform regardless of chunking.
        """
        super().apply_fsdp(**fsdp_kwargs)

    def apply_compile(self) -> None:
        """Apply torch.compile to the coarse blocks only.

        The pixel blocks see different shapes every batch (data-dependent unit counts
        and chunk sizes), so compiling them would trigger constant recompilation.
        """
        super().apply_compile()


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
    pixel_grad_checkpointing: bool = True

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
