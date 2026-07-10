"""Dual-resolution encoder: coarse patch tokens + a lightweight per-pixel branch.

The coarse branch is the existing :class:`~olmoearth_pretrain.nn.flexi_vit.Encoder`
(full spatio-temporal attention at ``embedding_size``). Alongside it we keep a
per-pixel branch at a smaller ``pixel_embedding_size``. Four pixel-branch designs are
available (``pixel_branch_type``), trading fine-branch expressivity against speed:

* ``"joint"`` (the original design; the slowest): per coarse block,

  1. conditions each unit's pixel tokens on its coarse token via gated FiLM
     (:class:`PixelFiLM`) -- the coarse branch does the contextual reasoning at full
     width; the pixels are told how to reinterpret their local detail;
  2. runs joint spatio-temporal self-attention *within each coarse patch*: every pixel
     attends over all ``(pixel offset, modality, band set, timestep)`` tokens at its
     patch -- but never across *other* patches (cross-patch reasoning is the coarse
     branch's job). Modality/band-set identity is carried by the separate
     per-``(modality, band set)`` projection weights; timestep and within-patch offset
     by additive sin/cos encodings;
  3. optionally lets each coarse token cross-attend to its unit's pixels.

* ``"conv"`` (:class:`ConvPixelStep`): a BiSeNet / MobileViT-style shallow
  high-resolution convolutional branch. Each step is a ConvNeXt-style unit (depthwise
  conv over each ``(instance, timestep, band set)`` frame + pointwise MLP) on the
  *dense* pixel grid, so spatial mixing crosses patch boundaries (non-ONLINE pixels
  are zeroed at input, so no masked data leaks). Fusion: coarse -> pixel by FiLM
  (per-patch modulation broadcast over the patch's pixels), pixel -> coarse by
  mean-pooling each patch's pixels through a zero-initialized linear, added
  residually. Temporal/cross-modal mixing stays in the coarse branch.

* ``"window"`` (:class:`WindowPixelStep`): ViTDet/Swin-style window attention where a
  window is exactly one unit's ``P**2`` pixels plus its (projected) coarse token as a
  per-window register -- sequence length ``P**2 + 1``, batched over units, no padding
  and no mask. Both fusion directions come free: pixels read the register, and the
  register's update is projected back up (zero-init) onto the coarse token.

* ``"perceiver"`` (:class:`PerceiverPixelStep`): the cheapest true branch -- pixel
  tokens never mix with each other. Each step FiLM-conditions the pixels on their
  coarse token and applies a pointwise MLP; every ``pixel_coarse_read_interval``
  steps (and always on the last) the coarse token attention-pools over its unit's
  pixels (:class:`CrossAttnBlock`) to read fine detail back up.

* ``"pixeldit"`` (:class:`PixelDiTStep` / :class:`PixelDiTFusion`): a post-trunk
  pathway following `PixelDiT <https://arxiv.org/abs/2511.20645>`_ -- the coarse
  trunk runs unmodified to completion, then ``pixel_dit_depth`` tiny PiT blocks
  refine the pixel tokens, conditioned on the final coarse tokens via *pixel-wise*
  AdaLN (each pixel offset gets its own modulation) with global context via token
  compaction attention. Unlike the paper (whose pixel pathway directly produces the
  diffusion output), the pixel tokens are finally re-aggregated per unit and added
  (zero-init) to the output tokens, so the latent-MIM loss and downstream evals
  consume pixel-refined tokens.

The interleaved non-``joint`` types can also run at a reduced cadence: with
``pixel_every_k_blocks = k`` the pixel branch executes only after every ``k``-th
coarse block (always including the last), cutting its cost by ``1/k``.

FiLM (rather than attention) for the pixel <- coarse direction is deliberate: a
pixel's coarse context is a *single* token (its own unit's), so there is nothing to
retrieve, and softmax over one key is identically 1 -- cross-attention degenerates to
a query-independent broadcast. FiLM's multiplicative term instead rescales each
pixel's own feature channels by the coarse context, so the shared per-unit
conditioning still acts differently on every pixel.

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
* An **offset** is a pixel's position within its patch (``P**2`` per unit), carried
  by an additive 2D sin/cos encoding of the integer offset.

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
    * Cross-attention uses absolute positions (no RoPE); the pixel attention gets its
      temporal and within-patch spatial signals from additive sin/cos encodings.
      RoPE for pixels is a future refinement.
    * The pixel branch only applies to spatial, multitemporal modalities, and
      cross-modality self-attention requires those modalities to share the same pixel
      grid (same ``G`` and ``P``).
"""

import logging
import math
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from einops import rearrange, reduce
from torch import Tensor, nn

from olmoearth_pretrain.data.constants import Modality, ModalitySpec
from olmoearth_pretrain.datatypes import (
    MaskedOlmoEarthSample,
    MaskValue,
    TokensAndMasks,
)
from olmoearth_pretrain.decorators import experimental
from olmoearth_pretrain.nn.attention import Attention, Mlp
from olmoearth_pretrain.nn.encodings import (
    PositionEncoding,
    get_1d_sincos_pos_encoding,
    get_2d_sincos_pos_encoding,
)
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
    """Groups ONLINE units by *location* (spatial patch) for the pixel attention.

    The fields describe how to scatter a packed ``[num_units, P**2, Dp]`` tensor (the
    concatenation of every modality's ONLINE units) into a padded
    ``[num_locations, max_units, P**2, Dp]`` buffer and back. Grouping by location is
    what makes the attention span modalities while never mixing patches.

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
    """Group units by location id for padded per-patch attention.

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
        loc: Per-location grouping of the concatenated units (``"joint"`` pixel
            self-attention only; ``None`` for the other branch types).
        coarse_idx: ``[total_units]`` position of each unit's coarse token in the
            flattened ``[B * N]`` coarse token tensor (``N`` = tokens per instance),
            so cross-attention gathers/scatters coarse partners with one index op.
        coarse_offsets: Start of each modality's token slab on the per-instance
            coarse token axis (``"conv"`` branch: whole-slab gather/scatter).
        frame_splits: Frames (``B * T * band_sets``) per modality in the concatenated
            dense frame tensor (``"conv"`` branch only).
    """

    states: dict[str, PixelModalityState]
    split_sizes: list[int]
    loc: LocationGroupings | None
    coarse_idx: Tensor
    coarse_offsets: dict[str, int] = field(default_factory=dict)
    frame_splits: list[int] = field(default_factory=list)


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
            patch_size: Coarse patch size (defines the within-patch offsets encoded
                into the tokens).

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
            modality_tokens = self._add_positional_encodings(
                modality_tokens, patch_size
            )
            output[modality] = modality_tokens
            output[input_data.get_masked_modality_name(modality)] = torch.stack(
                masks, dim=-1
            )  # [B, H, W, T, bs]
        return output

    def _add_positional_encodings(self, tokens: Tensor, patch_size: int) -> Tensor:
        """Add additive sin/cos encodings: temporal (1D over T) + within-patch (2D).

        The 2D encoding is over the *integer* pixel offset within the coarse patch
        ``(h % P, w % P)``, so a pixel pair one ground-sample apart looks identical at
        every patch size, and the per-patch joint attention can tell offsets apart.
        Patch-to-patch position is deliberately NOT encoded -- pixels never attend
        across patches, and their patch's location is carried by the coarse branch.
        """
        _, h, w, t, _, _ = tokens.shape
        temporal = get_1d_sincos_pos_encoding(
            torch.arange(t, device=tokens.device, dtype=torch.float32),
            self.pixel_embedding_size,
        )  # [T, Dp]
        offsets = torch.stack(
            torch.meshgrid(
                torch.arange(h, device=tokens.device) % patch_size,
                torch.arange(w, device=tokens.device) % patch_size,
                indexing="ij",
            ),
            dim=0,
        ).float()  # [2, H, W] within-patch integer offsets
        spatial = get_2d_sincos_pos_encoding(offsets, self.pixel_embedding_size).view(
            h, w, self.pixel_embedding_size
        )  # [H, W, Dp]
        # Cast to the token dtype: under FSDP mixed precision the tokens are bf16 and
        # a float32 addition would silently promote the whole pixel branch to float32
        # (and then fail against the bf16 LayerNorm/Linear params downstream).
        enc = (
            temporal[None, None, :, None, :] + spatial[:, :, None, None, :]
        )  # [H, W, T, bs=1, Dp]
        return tokens + enc.to(tokens.dtype)[None]


class PixelAttentionBlock(nn.Module):
    """Transformer block that self-attends over a (padded) sequence axis with a mask.

    Used for the pixel branch's per-patch attention: each row is one patch's ONLINE
    ``(pixel offset, modality, band set, timestep)`` tokens (padded to
    ``max_units * P**2``), so tokens mix within a single patch, never across patches.
    """

    def __init__(
        self, dim: int, num_heads: int, mlp_ratio: float, norm_affine: bool = True
    ) -> None:
        """Initialize the block.

        Args:
            dim: Pixel embedding dimension.
            num_heads: Number of attention heads.
            mlp_ratio: MLP hidden-dim ratio.
            norm_affine: Learn LayerNorm scale/shift. Pass False for pixel-scale
                inputs: the affine is redundant before the qkv/MLP linears, and its
                gamma/beta gradient reduction over millions of pixel rows dominates
                the backward otherwise. (Default True preserves the original "joint"
                branch behavior.)
        """
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=norm_affine)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=True,
            position_encoding=PositionEncoding.ABSOLUTE,
        )
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=norm_affine)
        self.mlp = Mlp(dim, hidden_features=int(dim * mlp_ratio))

    def forward(self, x: Tensor, key_mask: Tensor | None) -> Tensor:
        """Apply self-attention over the sequence axis.

        Args:
            x: ``[N, S, D]`` tokens (S = padded sequence length).
            key_mask: Optional ``[N, S]`` bool (True = attend).

        Returns:
            ``[N, S, D]`` updated tokens.
        """
        # Broadcastable [N, 1, 1, S] form: N is huge here, so materializing the
        # [N, heads, S, S] mask (what Attention does with a 2-dim mask) would dominate
        # the pixel branch's memory.
        if key_mask is not None:
            key_mask = key_mask[:, None, None, :]
        x = x + self.attn(self.norm1(x), attn_mask=key_mask)
        x = x + self.mlp(self.norm2(x))
        return x


class PixelFiLM(nn.Module):
    """Gated FiLM conditioning of a unit's pixel tokens on its coarse token.

    The coarse branch does the heavy contextual reasoning at the full embedding width;
    this module injects that context into the unit's ``P**2`` pixel tokens as an
    AdaLN-Zero-style modulation (`DiT <https://arxiv.org/abs/2212.09748>`_):

    ``pixels += gate * (norm(pixels) * (1 + scale) + shift)``

    with ``(scale, shift, gate)`` produced per unit from its coarse token. The
    conditioning is shared by the unit's pixels, but the *effect* is pixel-dependent
    through the multiplicative ``scale`` term (the coarse context rescales each pixel's
    own feature channels) -- unlike cross-attention to a single coarse token, whose
    softmax over one key collapses to a query-independent broadcast.

    The projection is zero-initialized (see :meth:`zero_init`), so training starts
    from the unconditioned pixel branch and the gate opens gradually.
    """

    def __init__(
        self, coarse_dim: int, pixel_dim: int, pixel_norm_affine: bool = True
    ) -> None:
        """Initialize the FiLM module.

        Args:
            coarse_dim: Coarse-token embedding dimension.
            pixel_dim: Pixel-token embedding dimension.
            pixel_norm_affine: Learn the pixel LayerNorm's scale/shift. Pass False for
                pixel-scale inputs -- the affine is redundant with the FiLM
                modulation, and its gamma/beta gradient reduction over millions of
                pixel rows is expensive in backward. (Default True preserves the
                original "joint" branch behavior.)
        """
        super().__init__()
        self.norm_coarse = nn.LayerNorm(coarse_dim)
        self.norm_pixel = nn.LayerNorm(pixel_dim, elementwise_affine=pixel_norm_affine)
        self.to_film = nn.Linear(coarse_dim, 3 * pixel_dim)

    def zero_init(self) -> None:
        """Zero the modulation projection (AdaLN-Zero): the module starts as identity.

        Call *after* any blanket weight init. With zeros, ``gate == 0`` so the pixel
        stream is untouched at step 0, while ``d(update)/d(gate weights)`` is nonzero,
        so the gate (and then scale/shift) can still learn.
        """
        nn.init.zeros_(self.to_film.weight)
        nn.init.zeros_(self.to_film.bias)

    def forward(self, pixels: Tensor, coarse: Tensor) -> Tensor:
        """Modulate ``pixels`` by their unit's coarse token.

        Args:
            pixels: ``[num_units, P**2, Dp]`` pixel tokens.
            coarse: ``[num_units, Dc]`` coarse token of each unit.

        Returns:
            ``[num_units, P**2, Dp]`` modulated pixel tokens.
        """
        film = self.to_film(self.norm_coarse(coarse))[:, None]  # [units, 1, 3 * Dp]
        scale, shift, gate = film.chunk(3, dim=-1)
        return pixels + gate * (self.norm_pixel(pixels) * (1 + scale) + shift)


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
        # Broadcastable [B', 1, 1, N_k] form; see PixelAttentionBlock.forward.
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


class ConvPixelStep(nn.Module):
    """One convolutional pixel-branch step with bidirectional coarse fusion.

    Runs on the *dense* pixel frames of every pixel modality (concatenated on the
    frame axis) as a single DiT-style adaptively-modulated ConvNeXt unit:

    ``frames += gate * mlp(dwconv(norm(frames) * (1 + scale) + shift))``

    with per-patch ``(scale, shift, gate)`` produced from the patch's coarse token
    (coarse -> pixel fusion, broadcast over the patch's pixels) -- so one step costs a
    single pixel-resolution LayerNorm, one depthwise conv (local spatial mixing,
    including across patch boundaries) and one pointwise MLP. Pixel -> coarse fusion
    mean-pools each patch's pixels through a zero-initialized linear that the caller
    adds residually to the coarse tokens.

    All pixel-resolution norms are affine-free (DiT-style): their scale/shift is
    subsumed by the FiLM modulation / following linear layer, and the LayerNorm
    gamma/beta gradient reduction over millions of pixel rows would otherwise dominate
    the whole branch's backward time.
    """

    def __init__(
        self, coarse_dim: int, pixel_dim: int, kernel_size: int, mlp_ratio: float
    ) -> None:
        """Initialize the step.

        Args:
            coarse_dim: Coarse-token embedding dimension.
            pixel_dim: Pixel embedding dimension.
            kernel_size: Depthwise convolution kernel size (odd).
            mlp_ratio: Pointwise MLP hidden-dim ratio.
        """
        super().__init__()
        self.norm_coarse = nn.LayerNorm(coarse_dim)
        self.to_film = nn.Linear(coarse_dim, 3 * pixel_dim)
        self.norm = nn.LayerNorm(pixel_dim, elementwise_affine=False)
        self.dwconv = nn.Conv2d(
            pixel_dim,
            pixel_dim,
            kernel_size,
            padding=kernel_size // 2,
            groups=pixel_dim,
        )
        self.mlp = Mlp(pixel_dim, hidden_features=int(pixel_dim * mlp_ratio))
        self.norm_pool = nn.LayerNorm(pixel_dim, elementwise_affine=False)
        self.to_coarse = nn.Linear(pixel_dim, coarse_dim)

    def zero_init(self) -> None:
        """Zero the fusion projections so the step starts as (near-)identity.

        The FiLM gate starts closed (the whole conv unit's output is gated to zero)
        and the pixel -> coarse projection starts at zero, so at step 0 both streams
        pass through unchanged while gradients still reach every parameter.
        """
        nn.init.zeros_(self.to_film.weight)
        nn.init.zeros_(self.to_film.bias)
        nn.init.zeros_(self.to_coarse.weight)
        nn.init.zeros_(self.to_coarse.bias)

    def forward(
        self, frames: Tensor, coarse: Tensor, patch_shape: tuple[int, int]
    ) -> tuple[Tensor, Tensor]:
        """Run the step.

        Args:
            frames: ``[F, H, W, Dp]`` dense pixel frames (all modalities, concatenated
                on the frame axis).
            coarse: ``[F, G1, G2, Dc]`` coarse token of each frame's patches.
            patch_shape: ``(P1, P2)`` pixels per patch side.

        Returns:
            ``(frames, coarse_delta)``: the updated frames and a ``[F, G1, G2, Dc]``
            residual update for the coarse tokens.
        """
        p1, p2 = patch_shape
        f, h, w, dp = frames.shape
        film = self.to_film(self.norm_coarse(coarse))  # [F, G1, G2, 3 * Dp]
        # Broadcast the per-patch params over the patch's pixels through a 6D view
        # (no [F, H, W, 3 * Dp] materialization).
        scale, shift, gate = film[:, :, None, :, None, :].chunk(3, dim=-1)
        grid = frames.view(f, h // p1, p1, w // p2, p2, dp)
        y = (self.norm(grid) * (1 + scale) + shift).view(f, h, w, dp)
        y = self.dwconv(y.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        y = self.mlp(y)
        frames = (grid + gate * y.view_as(grid)).view(f, h, w, dp)
        pooled = reduce(
            frames, "f (g1 p1) (g2 p2) d -> f g1 g2 d", "mean", p1=p1, p2=p2
        )
        return frames, self.to_coarse(self.norm_pool(pooled))


class WindowPixelStep(nn.Module):
    """Per-unit window attention with the coarse token as a per-window register.

    Each window is exactly one unit's ``P**2`` pixel tokens plus the unit's coarse
    token projected down to the pixel width (a learned register embedding marks it) --
    a ``[num_units, P**2 + 1, Dp]`` batched attention with no padding and no mask.
    Pixels never attend across units (cross-patch/temporal/modal reasoning stays in
    the coarse branch), and the fusion is bidirectional for free: pixels read the
    register, and the register's post-attention state is projected back up through a
    zero-initialized linear that the caller adds residually to the coarse token.
    """

    def __init__(
        self, coarse_dim: int, pixel_dim: int, num_heads: int, mlp_ratio: float
    ) -> None:
        """Initialize the step.

        Args:
            coarse_dim: Coarse-token embedding dimension.
            pixel_dim: Pixel embedding dimension.
            num_heads: Attention heads for the window attention.
            mlp_ratio: MLP hidden-dim ratio for the window attention block.
        """
        super().__init__()
        self.norm_coarse = nn.LayerNorm(coarse_dim)
        self.down = nn.Linear(coarse_dim, pixel_dim)
        self.register_embed = nn.Parameter(torch.zeros(pixel_dim))
        self.block = PixelAttentionBlock(
            pixel_dim, num_heads, mlp_ratio, norm_affine=False
        )
        self.up = nn.Linear(pixel_dim, coarse_dim)
        nn.init.normal_(self.register_embed, std=0.02)

    def zero_init(self) -> None:
        """Zero the register -> coarse projection: coarse tokens start unperturbed."""
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, pixels: Tensor, coarse: Tensor) -> tuple[Tensor, Tensor]:
        """Run the step.

        Args:
            pixels: ``[num_units, P**2, Dp]`` packed ONLINE pixel tokens.
            coarse: ``[num_units, Dc]`` coarse token of each unit.

        Returns:
            ``(pixels, coarse_delta)``: the updated pixel tokens and a
            ``[num_units, Dc]`` residual update for the coarse tokens.
        """
        reg = self.down(self.norm_coarse(coarse)) + self.register_embed.to(pixels.dtype)
        x = torch.cat([pixels, reg[:, None]], dim=1)  # [U, P**2 + 1, Dp]
        x = chunked_batch_attn(self.block, x, None)
        return x[:, :-1], self.up(x[:, -1])


class PixelReadPool(nn.Module):
    """Cheap coarse <- pixel attention pool, computed at the pixel width.

    The coarse token is projected DOWN to the pixel width and used as a single query
    over its unit's ``P**2`` pixels; the pooled result is projected back up through a
    zero-initialized linear that the caller adds residually to the coarse token. This
    keeps the per-pixel work at ``O(Dp**2)`` -- unlike a coarse-width
    :class:`CrossAttnBlock`, whose ``kv_proj`` alone costs ``Dp * Dc`` per pixel (with
    ``Dc = 512`` that one projection dwarfs the entire pixel branch).
    """

    def __init__(self, coarse_dim: int, pixel_dim: int, num_heads: int) -> None:
        """Initialize the pool.

        Args:
            coarse_dim: Coarse-token embedding dimension.
            pixel_dim: Pixel embedding dimension.
            num_heads: Attention heads (over the pixel width).
        """
        super().__init__()
        self.num_heads = num_heads
        self.norm_q = nn.LayerNorm(coarse_dim)
        self.to_q = nn.Linear(coarse_dim, pixel_dim)
        # Affine-free: redundant with to_kv, and the gamma/beta grad reduction over
        # millions of pixel rows is expensive in backward.
        self.norm_kv = nn.LayerNorm(pixel_dim, elementwise_affine=False)
        self.to_kv = nn.Linear(pixel_dim, 2 * pixel_dim)
        self.up = nn.Linear(pixel_dim, coarse_dim)

    def zero_init(self) -> None:
        """Zero the up-projection: coarse tokens start unperturbed."""
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, coarse: Tensor, pixels: Tensor) -> Tensor:
        """Pool each unit's pixels with its coarse token as the query.

        Args:
            coarse: ``[U, Dc]`` coarse token of each unit.
            pixels: ``[U, P**2, Dp]`` pixel tokens.

        Returns:
            ``[U, Dc]`` residual update for the coarse tokens.
        """

        def attend(q: Tensor, k: Tensor, v: Tensor) -> Tensor:
            heads = self.num_heads
            q = rearrange(q, "u s (h d) -> u h s d", h=heads)
            k = rearrange(k, "u s (h d) -> u h s d", h=heads)
            v = rearrange(v, "u s (h d) -> u h s d", h=heads)
            out = torch.nn.functional.scaled_dot_product_attention(q, k, v)
            return rearrange(out, "u h s d -> u s (h d)")

        q = self.to_q(self.norm_q(coarse))[:, None]  # [U, 1, Dp]
        k, v = self.to_kv(self.norm_kv(pixels)).chunk(2, dim=-1)
        pooled = chunked_batch_attn(attend, q, k, v)  # [U, 1, Dp]
        return self.up(pooled[:, 0])


class PerceiverPixelStep(nn.Module):
    """Cross-attention-only pixel step: pixels never mix with each other.

    Each step FiLM-conditions the unit's pixels on its coarse token and applies a
    pointwise MLP -- the pixel branch is a steered high-resolution "memory" that
    preserves and sharpens local evidence while the coarse branch does all the
    reasoning. On read steps the coarse token additionally attention-pools over its
    unit's pixels (:class:`PixelReadPool`, linear in the pixel count and computed at
    the pixel width) so fine detail flows back up.
    """

    def __init__(
        self,
        coarse_dim: int,
        pixel_dim: int,
        num_heads: int,
        mlp_ratio: float,
        with_read: bool,
    ) -> None:
        """Initialize the step.

        Args:
            coarse_dim: Coarse-token embedding dimension.
            pixel_dim: Pixel embedding dimension.
            num_heads: Attention heads for the coarse <- pixel read (pixel width).
            mlp_ratio: Hidden-dim ratio of the pointwise pixel MLP.
            with_read: Whether this step includes the coarse <- pixel read.
        """
        super().__init__()
        self.film = PixelFiLM(coarse_dim, pixel_dim, pixel_norm_affine=False)
        self.norm = nn.LayerNorm(pixel_dim, elementwise_affine=False)
        self.mlp = Mlp(pixel_dim, hidden_features=int(pixel_dim * mlp_ratio))
        self.read = (
            PixelReadPool(coarse_dim, pixel_dim, num_heads) if with_read else None
        )

    def zero_init(self) -> None:
        """Zero the FiLM modulation and the read's coarse update (identity at init)."""
        self.film.zero_init()
        if self.read is not None:
            self.read.zero_init()

    def forward(self, pixels: Tensor, coarse: Tensor) -> tuple[Tensor, Tensor | None]:
        """Run the step.

        Args:
            pixels: ``[num_units, P**2, Dp]`` packed ONLINE pixel tokens.
            coarse: ``[num_units, Dc]`` coarse token of each unit.

        Returns:
            ``(pixels, coarse_delta)``: the updated pixel tokens and, on read steps, a
            ``[num_units, Dc]`` residual update for the coarse tokens (else ``None``).
        """
        pixels = self.film(pixels, coarse)
        pixels = pixels + self.mlp(self.norm(pixels))
        if self.read is None:
            return pixels, None
        return pixels, self.read(coarse, pixels)


def within_patch_offset_index(
    patch_shape: tuple[int, int], max_patch_size: int, device: torch.device
) -> Tensor:
    """Rows of a ``max_patch_size**2``-entry per-offset table for a smaller patch.

    Per-offset parameter tables (PixelDiT AdaLN / compaction / expansion weights) are
    sized for the maximum patch; a ``(P1, P2)`` patch's pixel at within-patch offset
    ``(i, j)`` uses row ``i * max_patch_size + j``. The returned index follows the
    packed pixel order (row-major ``(p1, p2)``, as produced by
    :meth:`DualResEncoder._build_pixel_context`).

    Args:
        patch_shape: ``(P1, P2)`` pixels per patch side.
        max_patch_size: Table side length (the encoder's maximum patch size).
        device: Device for the index tensor.

    Returns:
        ``[P1 * P2]`` long tensor of table rows.
    """
    p1, p2 = patch_shape
    i = torch.arange(p1, device=device)
    j = torch.arange(p2, device=device)
    return (i[:, None] * max_patch_size + j[None, :]).reshape(-1)


class PixelDiTStep(nn.Module):
    """One PixelDiT "PiT" block: pixel-wise AdaLN + token-compaction attention + MLP.

    Follows `PixelDiT <https://arxiv.org/abs/2511.20645>`_'s pixel pathway, adapted to
    the packed ONLINE-unit layout and flexi patch sizes:

    1. **Pixel-wise AdaLN**: the unit's (final) coarse token is projected to
       ``P**2`` *distinct* per-offset ``(shift, scale, gate)`` pairs -- unlike
       :class:`PixelFiLM`, every pixel of the patch gets its own modulation. The
       projection weight covers ``max_patch_size**2`` offsets and is row-gathered for
       the actual patch size, so smaller patches compute proportionally less.
    2. **Token compaction attention**: each unit's ``P**2 x Dp`` pixel tokens are
       compacted by a learned per-offset flattening into ONE coarse-width token; the
       compacted tokens attend over the same packed per-instance sequence (same
       attention mask and RoPE positions) as the coarse trunk, then a learned
       per-offset expansion distributes the result back to the pixels (gated
       residual). This gives every pixel a global receptive field at
       patch-sequence attention cost.
    3. **Pointwise MLP** at the (tiny) pixel width, with its own AdaLN modulation.

    The step is conditioned on the coarse tokens but never writes to them (the fusion
    into the encoder output happens once, after all steps -- see
    :class:`PixelDiTFusion`). ``zero_init`` closes every gate so the step is exactly
    the identity on the pixel stream at initialization.
    """

    def __init__(
        self,
        coarse_dim: int,
        pixel_dim: int,
        max_patch_size: int,
        num_heads: int,
        mlp_ratio: float,
        position_encoding: str,
        rope_base: float,
        rope_mixed_base: float,
        temporal_rope_dim_frac: float,
        rope_temporal_base: float | None,
    ) -> None:
        """Initialize the step.

        Args:
            coarse_dim: Coarse-token embedding dimension (compaction/attention width).
            pixel_dim: Pixel embedding dimension (Dp).
            max_patch_size: Maximum patch side length (sizes the per-offset tables).
            num_heads: Attention heads for the compacted-token attention.
            mlp_ratio: Hidden-dim ratio of the pointwise pixel MLP.
            position_encoding: The coarse trunk's position-encoding mode (the
                compacted tokens attend with the same RoPE variant/positions).
            rope_base: RoPE frequency base (matches the trunk).
            rope_mixed_base: RoPE-Mixed initialization base (matches the trunk).
            temporal_rope_dim_frac: Temporal RoPE head-dim fraction (matches trunk).
            rope_temporal_base: Temporal RoPE base (matches the trunk).
        """
        super().__init__()
        self.max_patch_size = max_patch_size
        self.pixel_dim = pixel_dim
        n_off = max_patch_size**2
        self.norm_coarse = nn.LayerNorm(coarse_dim)
        # 6 groups per offset: (shift1, scale1, gate1, shift2, scale2, gate2).
        self.to_adaln = nn.Linear(coarse_dim, n_off * 6 * pixel_dim)
        self.norm1 = nn.LayerNorm(pixel_dim, elementwise_affine=False)
        self.norm2 = nn.LayerNorm(pixel_dim, elementwise_affine=False)
        self.compact = nn.Parameter(torch.empty(n_off, pixel_dim, coarse_dim))
        self.expand = nn.Parameter(torch.empty(n_off, coarse_dim, pixel_dim))
        self.norm_attn = nn.LayerNorm(coarse_dim)
        self.attn = Attention(
            coarse_dim,
            num_heads=num_heads,
            qkv_bias=True,
            position_encoding=position_encoding,
            rope_base=rope_base,
            rope_mixed_base=rope_mixed_base,
            temporal_rope_dim_frac=temporal_rope_dim_frac,
            rope_temporal_base=rope_temporal_base,
        )
        self.mlp = Mlp(pixel_dim, hidden_features=int(pixel_dim * mlp_ratio))
        # The compaction acts as a Linear over the flattened (offset, Dp) axis; the
        # expansion as a per-offset Linear from the coarse width. Xavier bounds with
        # the matching fan-in/fan-out (raw Parameters are skipped by the blanket
        # module init).
        bound_c = (6.0 / (n_off * pixel_dim + coarse_dim)) ** 0.5
        nn.init.uniform_(self.compact, -bound_c, bound_c)
        bound_e = (6.0 / (coarse_dim + pixel_dim)) ** 0.5
        nn.init.uniform_(self.expand, -bound_e, bound_e)

    def zero_init(self) -> None:
        """Zero the AdaLN projection: all gates closed, the step starts as identity.

        Both residual branches (compaction attention and MLP) are multiplied by the
        AdaLN gates, so with a zeroed projection the pixel stream passes through
        untouched while every parameter still receives gradient through the gates.
        """
        nn.init.zeros_(self.to_adaln.weight)
        nn.init.zeros_(self.to_adaln.bias)

    def adaln_params(
        self, coarse: Tensor, offset_idx: Tensor
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Compute the per-pixel AdaLN parameters for the given patch offsets.

        The projection weight is row-gathered BEFORE the matmul so only the actual
        ``P**2`` offsets are computed (a patch-size-1 unit computes 1/64th of the
        full table).

        Args:
            coarse: ``[num_units, Dc]`` coarse token per unit.
            offset_idx: ``[P**2]`` per-offset table rows
                (:func:`within_patch_offset_index`).

        Returns:
            Six ``[num_units, P**2, Dp]`` tensors:
            ``(shift1, scale1, gate1, shift2, scale2, gate2)``.
        """
        n_off = self.max_patch_size**2
        dp = self.pixel_dim
        w = self.to_adaln.weight.view(n_off, 6 * dp, -1)[offset_idx].flatten(0, 1)
        bias = self.to_adaln.bias.view(n_off, 6 * dp)[offset_idx].flatten()
        adaln = F.linear(self.norm_coarse(coarse), w, bias)
        adaln = adaln.view(coarse.shape[0], offset_idx.numel(), 6, dp)
        s1, g1, a1, s2, g2, a2 = adaln.unbind(dim=2)
        return s1, g1, a1, s2, g2, a2

    def forward(
        self,
        pixels: Tensor,
        coarse: Tensor,
        patch_shape: tuple[int, int],
        packed_pos: Tensor,
        batch: int,
        seqlen: int,
        attn_mask: Tensor | None,
        positions_packed: Tensor | None,
    ) -> Tensor:
        """Run the step.

        Args:
            pixels: ``[num_units, P**2, Dp]`` packed ONLINE pixel tokens.
            coarse: ``[num_units, Dc]`` final coarse token of each unit (fixed
                conditioning; not updated by this step).
            patch_shape: ``(P1, P2)`` pixels per patch side.
            packed_pos: ``[num_units]`` slot of each unit in the flattened
                ``[batch * seqlen]`` packed coarse layout.
            batch: Number of instances.
            seqlen: Packed sequence length (the trunk's ``max_seqlen``).
            attn_mask: The trunk's broadcastable attention mask (or ``None``).
            positions_packed: The trunk's packed RoPE positions (or ``None``).

        Returns:
            ``[num_units, P**2, Dp]`` updated pixel tokens.
        """
        offset_idx = within_patch_offset_index(
            patch_shape, self.max_patch_size, pixels.device
        )
        s1, g1, a1, s2, g2, a2 = self.adaln_params(coarse, offset_idx)

        # Compaction -> packed attention (same sequence/mask/RoPE as the trunk;
        # slots of non-pixel units keep zero content and are ordinary masked-or-zero
        # keys) -> expansion, gated residual.
        h = self.norm1(pixels) * (1 + g1) + s1
        comp = torch.einsum("upd,pdc->uc", h, self.compact[offset_idx])
        buf = comp.new_zeros(batch * seqlen, comp.shape[-1])
        buf[packed_pos] = comp
        x = buf.view(batch, seqlen, -1)
        x = x + self.attn(
            self.norm_attn(x), attn_mask=attn_mask, rope_positions=positions_packed
        )
        out_units = x.reshape(batch * seqlen, -1)[packed_pos]
        expanded = torch.einsum("uc,pcd->upd", out_units, self.expand[offset_idx])
        pixels = pixels + a1 * expanded

        # Pointwise MLP with its own pixel-wise modulation.
        pixels = pixels + a2 * self.mlp(self.norm2(pixels) * (1 + g2) + s2)
        return pixels


class PixelDiTFusion(nn.Module):
    """Fuse the pixel pathway's output into the encoder's final coarse tokens.

    This is the piece PixelDiT does not need (its pixel pathway directly produces the
    diffusion output): our downstream losses and evals consume patch tokens, so each
    unit's ``P**2`` pixel tokens are re-aggregated by a learned per-offset compaction
    and added residually to the unit's final coarse token. Zero-initialized: the
    encoder output starts exactly equal to the pixel-free coarse encoder's.
    """

    def __init__(self, coarse_dim: int, pixel_dim: int, max_patch_size: int) -> None:
        """Initialize the fusion.

        Args:
            coarse_dim: Coarse-token embedding dimension.
            pixel_dim: Pixel embedding dimension.
            max_patch_size: Maximum patch side length (sizes the per-offset table).
        """
        super().__init__()
        self.max_patch_size = max_patch_size
        self.norm = nn.LayerNorm(pixel_dim, elementwise_affine=False)
        self.compact = nn.Parameter(
            torch.zeros(max_patch_size**2, pixel_dim, coarse_dim)
        )

    def zero_init(self) -> None:
        """Zero the compaction: the encoder output starts unperturbed."""
        nn.init.zeros_(self.compact)

    def forward(self, pixels: Tensor, patch_shape: tuple[int, int]) -> Tensor:
        """Compute the per-unit residual update for the final coarse tokens.

        Args:
            pixels: ``[num_units, P**2, Dp]`` final pixel tokens.
            patch_shape: ``(P1, P2)`` pixels per patch side.

        Returns:
            ``[num_units, Dc]`` residual updates.
        """
        offset_idx = within_patch_offset_index(
            patch_shape, self.max_patch_size, pixels.device
        )
        return torch.einsum("upd,pdc->uc", self.norm(pixels), self.compact[offset_idx])


@experimental("Dual-resolution (pixel branch) encoder is experimental.")
class DualResEncoder(Encoder):
    """Encoder with a coarse patch branch plus a lightweight per-pixel branch."""

    def __init__(
        self,
        *,
        pixel_embedding_size: int = 128,
        pixel_num_heads: int = 4,
        pixel_mlp_ratio: float = 4.0,
        pixel_branch_type: str = "joint",
        pixel_every_k_blocks: int = 1,
        pixel_conv_kernel: int = 3,
        pixel_coarse_read_interval: int = 1,
        pixel_dit_depth: int = 4,
        pixel_film_from_coarse: bool = True,
        coarse_cross_attn_to_pixel: bool = True,
        pixel_grad_checkpointing: bool = True,
        **encoder_kwargs: Any,
    ) -> None:
        """Initialize the dual-resolution encoder.

        Args:
            pixel_embedding_size: Per-pixel embedding dimension (Dp).
            pixel_num_heads: Attention heads for the pixel branch.
            pixel_mlp_ratio: MLP ratio for the pixel branch.
            pixel_branch_type: Pixel-branch design: ``"joint"`` (original per-patch
                joint spatio-temporal attention), ``"conv"`` (ConvNeXt-style dense
                convolutional branch, :class:`ConvPixelStep`), ``"window"`` (per-unit
                window attention with a coarse register token,
                :class:`WindowPixelStep`), ``"perceiver"`` (cross-attention-only
                pixel tokens, :class:`PerceiverPixelStep`), or ``"pixeldit"``
                (post-trunk PixelDiT-style pathway with pixel-wise AdaLN and token
                compaction, fused into the output tokens; :class:`PixelDiTStep` /
                :class:`PixelDiTFusion`). See the module docstring.
            pixel_every_k_blocks: Run the pixel branch only after every ``k``-th
                coarse block (the last block always runs it). Requires
                ``depth % k == 0``; ``"joint"`` and ``"pixeldit"`` support only 1
                (``"pixeldit"`` runs after the trunk, not interleaved).
            pixel_conv_kernel: Depthwise kernel size (``"conv"`` type only).
            pixel_coarse_read_interval: For ``"perceiver"``: the coarse <- pixel
                attention-pool read runs every this-many pixel steps (the last pixel
                step always reads).
            pixel_dit_depth: For ``"pixeldit"``: number of post-trunk PiT blocks (M).
            pixel_film_from_coarse: (``"joint"`` only) If True, each block conditions
                a unit's pixel tokens on its coarse token via gated FiLM (see
                :class:`PixelFiLM`) -- the coarse branch reasons at full width, the
                pixels are told how to reinterpret their local detail.
                (Cross-attention formulations degenerate here: a pixel's coarse
                context is a single token, and softmax over one key ignores the
                query.)
            coarse_cross_attn_to_pixel: (``"joint"`` only) If True, coarse tokens
                cross-attend to their unit's pixel features so fine detail flows into
                the output.
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
        self.pixel_branch_type = pixel_branch_type
        self.pixel_every_k_blocks = pixel_every_k_blocks
        self.pixel_film_from_coarse = pixel_film_from_coarse
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
        self.pixel_self_blocks: nn.ModuleList | None = None
        self.pixel_film: nn.ModuleList | None = None
        self.coarse_to_pixel: nn.ModuleList | None = None
        self.pixel_steps: nn.ModuleList | None = None
        self.pixel_fusion: PixelDiTFusion | None = None
        if pixel_branch_type == "joint":
            self.pixel_self_blocks = nn.ModuleList(
                [
                    PixelAttentionBlock(
                        pixel_embedding_size, pixel_num_heads, pixel_mlp_ratio
                    )
                    for _ in range(depth)
                ]
            )
            self.pixel_film = (
                nn.ModuleList(
                    [PixelFiLM(coarse_dim, pixel_embedding_size) for _ in range(depth)]
                )
                if pixel_film_from_coarse
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
        else:
            num_steps = depth // pixel_every_k_blocks
            if pixel_branch_type == "conv":
                steps: list[nn.Module] = [
                    ConvPixelStep(
                        coarse_dim,
                        pixel_embedding_size,
                        kernel_size=pixel_conv_kernel,
                        mlp_ratio=pixel_mlp_ratio,
                    )
                    for _ in range(num_steps)
                ]
            elif pixel_branch_type == "window":
                steps = [
                    WindowPixelStep(
                        coarse_dim,
                        pixel_embedding_size,
                        num_heads=pixel_num_heads,
                        mlp_ratio=pixel_mlp_ratio,
                    )
                    for _ in range(num_steps)
                ]
            elif pixel_branch_type == "perceiver":
                steps = [
                    PerceiverPixelStep(
                        coarse_dim,
                        pixel_embedding_size,
                        num_heads=pixel_num_heads,
                        mlp_ratio=pixel_mlp_ratio,
                        # The last step always reads so fine detail reaches the coarse
                        # output (and every pixel param stays on the coarse loss path).
                        with_read=(
                            (s + 1) % pixel_coarse_read_interval == 0
                            or s == num_steps - 1
                        ),
                    )
                    for s in range(num_steps)
                ]
            elif pixel_branch_type == "pixeldit":
                steps = [
                    PixelDiTStep(
                        coarse_dim,
                        pixel_embedding_size,
                        max_patch_size=self.max_patch_size,
                        num_heads=coarse_heads,
                        mlp_ratio=pixel_mlp_ratio,
                        position_encoding=self.position_encoding,
                        rope_base=self.rope_base,
                        rope_mixed_base=self.rope_mixed_base,
                        temporal_rope_dim_frac=self.temporal_rope_dim_frac,
                        rope_temporal_base=self.rope_temporal_base,
                    )
                    for _ in range(pixel_dit_depth)
                ]
                self.pixel_fusion = PixelDiTFusion(
                    coarse_dim, pixel_embedding_size, self.max_patch_size
                )
            else:
                raise ValueError(f"Unknown pixel_branch_type {pixel_branch_type!r}")
            self.pixel_steps = nn.ModuleList(steps)

        for module in self._pixel_modules():
            module.apply(self._init_weights)
        # After the blanket init: fusion modulations start at zero (identity modules).
        if self.pixel_film is not None:
            for film in self.pixel_film:
                film.zero_init()
        if self.pixel_steps is not None:
            for step in self.pixel_steps:
                step.zero_init()
        if self.pixel_fusion is not None:
            self.pixel_fusion.zero_init()

    def _pixel_modules(self) -> list[nn.Module]:
        modules: list[nn.Module] = [self.pixel_embeddings]
        for maybe in (
            self.pixel_self_blocks,
            self.pixel_film,
            self.coarse_to_pixel,
            self.pixel_steps,
            self.pixel_fusion,
        ):
            if maybe is not None:
                modules.append(maybe)
        return modules

    def _is_pixel_block(self, block_idx: int) -> bool:
        """Whether the pixel branch runs after coarse block ``block_idx``."""
        if self.pixel_branch_type == "pixeldit":
            # The PixelDiT pathway runs once, after the whole trunk.
            return False
        return (block_idx + 1) % self.pixel_every_k_blocks == 0

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

        pixel_state: dict[str, PixelModalityState] = {}
        if token_exit_cfg is None or any(
            exit_depth > 0 for exit_depth in token_exit_cfg.values()
        ):
            # Only embed pixels when the blocks (and so the pixel branch) run at all:
            # the target-encoder path uses an all-zero token_exit_cfg, and embedding
            # the unmasked batch's pixels there would be pure waste.
            pixel_patchified = self.pixel_embeddings.forward(x, patch_size)
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
        """Run interleaved coarse (packed full attn) + pixel (per-patch) blocks.

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
        # ``pixels`` is carried through the block loop alongside the coarse tokens:
        # for the packed branch types it concatenates all modalities' ONLINE units
        # ([num_units, P**2, Dp]); for the "conv" type it is the dense frame tensor
        # ([F, H, W, Dp], non-ONLINE pixels zeroed). Split back into per-modality
        # state once at the end.
        pixel_ctx = self._build_pixel_context(pixel_x, original_masks_dict, dims_dict)
        pixels = None
        if pixel_ctx is not None:
            if self.pixel_branch_type == "conv":
                pixels = self._build_conv_frames(pixel_x, pixel_ctx)
            else:
                pixels = torch.cat(
                    [st.pixels for st in pixel_ctx.states.values()], dim=0
                )

        num_blocks = len(self.blocks)
        for i, blk in enumerate(self.blocks):
            packed = self._pack(coarse_dense, indices, new_mask, max_seqlen)
            if exit_ids_seq is not None and i > 0:
                exited_packed = torch.where(exit_packed == i, packed, exited_packed)
            packed = blk(x=packed, attn_mask=attn_mask, rope_positions=positions_packed)
            coarse_dense, _ = self.add_removed_tokens(packed, indices, new_mask)
            if pixel_ctx is not None and self._is_pixel_block(i):
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

        if pixel_ctx is not None and self.pixel_branch_type != "pixeldit":
            assert pixels is not None
            if self.pixel_branch_type == "conv":
                self._pack_conv_frames(pixels, pixel_ctx)
            else:
                self._assign_pixel_states(pixel_ctx, pixels)

        packed = self._pack(coarse_dense, indices, new_mask, max_seqlen)
        if exit_ids_seq is not None:
            packed = torch.where(exit_packed == num_blocks, packed, exited_packed)
        packed = self.norm(packed)
        coarse_dense, _ = self.add_removed_tokens(packed, indices, new_mask)

        if pixel_ctx is not None and self.pixel_branch_type == "pixeldit":
            assert pixels is not None
            coarse_dense, pixels = self._pixeldit_pathway(
                coarse_dense,
                pixels,
                pixel_ctx,
                indices,
                new_mask,
                max_seqlen,
                attn_mask,
                positions_packed,
            )
            self._assign_pixel_states(pixel_ctx, pixels)

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

        * the location id of every unit (for the per-patch joint attention), and
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
            flat_idx = torch.nonzero(online.reshape(-1), as_tuple=False).squeeze(-1)
            if self.pixel_branch_type == "conv":
                # The conv branch carries DENSE frames through the blocks (see
                # _build_conv_frames) and packs the ONLINE units only at the end, so
                # skip the packed gather here (pixels is a placeholder until then).
                packed_units = pixels.new_zeros(0, p1 * p2, pixels.shape[-1])
            else:
                # Group each unit's P**2 pixels: [B * U, P**2, Dp], U ordered
                # (g1, g2, t, bs).
                units = rearrange(
                    pixels,
                    "b (g1 p1) (g2 p2) t bs d -> (b g1 g2 t bs) (p1 p2) d",
                    g1=g1,
                    g2=g2,
                    p1=p1,
                    p2=p2,
                )
                packed_units = units[flat_idx]
            state = PixelModalityState(
                pixels=packed_units,
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
            # The per-location grouping is only used by the "joint" per-patch
            # self-attention; skip the sort/unique work for the other branch types.
            loc=(
                build_location_groupings(torch.cat(location_ids))
                if self.pixel_branch_type == "joint"
                else None
            ),
            coarse_idx=torch.cat(coarse_idx),
            coarse_offsets=coarse_offsets,
            frame_splits=[
                st.grid[0] * st.grid[3] * st.grid[4] for st in states.values()
            ],
        )

    def _build_conv_frames(
        self, pixel_x: dict[str, Tensor], ctx: PixelBranchContext
    ) -> Tensor:
        """Build the dense frame tensor for the convolutional pixel branch.

        Every modality's ``[B, H, W, T, band_sets, Dp]`` pixel tokens are flattened to
        per-``(instance, timestep, band set)`` frames and concatenated (in
        ``ctx.states`` order, matching ``ctx.frame_splits``). Non-ONLINE pixels are
        zeroed FIRST, so no masked-unit data ever enters the branch: whatever the
        depthwise convolutions later propagate across patch boundaries is derived from
        visible pixels only, and reconstruction of masked units cannot cheat.

        Args:
            pixel_x: The :class:`PixelPatchEmbed` output (tokens + pixel-level masks).
            ctx: The pixel-branch context.

        Returns:
            ``[F, H, W, Dp]`` dense frames (``F`` = total frames over modalities).
        """
        frames = []
        for modality in ctx.states:
            tokens = pixel_x[modality]  # [B, H, W, T, bs, Dp]
            mask_name = MaskedOlmoEarthSample.get_masked_modality_name(modality)
            online = pixel_x[mask_name] == MaskValue.ONLINE_ENCODER.value
            tokens = tokens * online[..., None].to(tokens.dtype)
            frames.append(rearrange(tokens, "b h w t bs d -> (b t bs) h w d"))
        return torch.cat(frames, dim=0)

    def _pack_conv_frames(self, frames: Tensor, ctx: PixelBranchContext) -> None:
        """Pack final dense frames into each modality's ONLINE-unit pixel state.

        Inverts :meth:`_build_conv_frames`: split the frame tensor per modality,
        regroup each patch's ``P**2`` pixels into units (canonical row-major
        ``(g1, g2, t, bs)`` order) and gather the ONLINE units, giving the same packed
        ``[num_online, P**2, Dp]`` layout the pixel decoders consume.

        Args:
            frames: ``[F, H, W, Dp]`` final dense frames.
            ctx: The pixel-branch context (its ``states`` are updated in place).
        """
        for (modality, st), frames_m in zip(
            ctx.states.items(), frames.split(ctx.frame_splits)
        ):
            b, g1, g2, t, bs = st.grid
            p1, p2 = st.patch_shape
            units = rearrange(
                frames_m,
                "(b t bs) (g1 p1) (g2 p2) d -> (b g1 g2 t bs) (p1 p2) d",
                b=b,
                t=t,
                bs=bs,
                g1=g1,
                g2=g2,
                p1=p1,
                p2=p2,
            )
            st.pixels = units[st.flat_idx]

    @staticmethod
    def _assign_pixel_states(ctx: PixelBranchContext, pixels: Tensor) -> None:
        """Split the concatenated packed pixel tensor back into per-modality states."""
        for state, updated in zip(ctx.states.values(), pixels.split(ctx.split_sizes)):
            state.pixels = updated

    def _pixeldit_pathway(
        self,
        coarse_dense: Tensor,
        pixels: Tensor,
        ctx: PixelBranchContext,
        indices: Tensor,
        new_mask: Tensor,
        max_seqlen: Tensor,
        attn_mask: Tensor | None,
        positions_packed: Tensor | None,
    ) -> tuple[Tensor, Tensor]:
        """Run the post-trunk PixelDiT pathway and fuse it into the output tokens.

        Every :class:`PixelDiTStep` is conditioned on the same post-norm final coarse
        tokens (PixelDiT's ``s_N``) and attends -- after token compaction -- over the
        same packed per-instance sequence (mask + RoPE positions) as the trunk. The
        coarse tokens are only modified once, at the very end, by the zero-initialized
        :class:`PixelDiTFusion` residual at the ONLINE units.

        Args:
            coarse_dense: ``[B, N, Dc]`` final (post-norm) coarse tokens.
            pixels: ``[num_units, P**2, Dp]`` packed ONLINE pixel tokens (as embedded;
                untouched during the trunk).
            ctx: The pixel-branch context.
            indices: The trunk's ONLINE-packing gather indices.
            new_mask: The trunk's packed-slot validity mask.
            max_seqlen: The trunk's packed sequence length.
            attn_mask: The trunk's broadcastable attention mask (or ``None``).
            positions_packed: The trunk's packed RoPE positions (or ``None``).

        Returns:
            ``(coarse_dense, pixels)``: the fused output tokens and final pixel
            tokens.
        """
        assert self.pixel_steps is not None and self.pixel_fusion is not None
        b, n, d = coarse_dense.shape
        s = int(max_seqlen)
        patch_shape = next(iter(ctx.states.values())).patch_shape

        # Packed slot of each ONLINE unit: invert the dense -> packed gather once so
        # the steps can scatter/gather compacted tokens without rebuilding the dense
        # layout. (Every ONLINE token sits within the first ``s`` slots of its row.)
        device = coarse_dense.device
        rows = torch.arange(b, device=device)[:, None]
        inv = torch.zeros(b * n, dtype=torch.long, device=device)
        inv[(rows * n + indices[:, :s]).reshape(-1)] = (
            rows * s + torch.arange(s, device=device)[None, :]
        ).reshape(-1)
        packed_pos = inv[ctx.coarse_idx]

        coarse_flat = coarse_dense.reshape(b * n, d)
        coarse_units = coarse_flat[ctx.coarse_idx]  # fixed s_N conditioning

        for step in self.pixel_steps:
            if self.pixel_grad_checkpointing and torch.is_grad_enabled():
                pixels = torch.utils.checkpoint.checkpoint(
                    step,
                    pixels,
                    coarse_units,
                    patch_shape,
                    packed_pos,
                    b,
                    s,
                    attn_mask,
                    positions_packed,
                    use_reentrant=False,
                )
            else:
                pixels = step(
                    pixels,
                    coarse_units,
                    patch_shape,
                    packed_pos,
                    b,
                    s,
                    attn_mask,
                    positions_packed,
                )

        delta = self.pixel_fusion(pixels, patch_shape)
        coarse_flat = coarse_flat.index_add(
            0, ctx.coarse_idx, delta.to(coarse_flat.dtype)
        )
        return coarse_flat.view(b, n, d), pixels

    def _interleave_pixels(
        self,
        block_idx: int,
        coarse_dense: Tensor,
        pixels: Tensor,
        ctx: PixelBranchContext,
    ) -> tuple[Tensor, Tensor]:
        """Run one pixel-branch step and exchange information with the coarse tokens.

        Dispatches on ``pixel_branch_type``; see :meth:`_interleave_joint` (the
        original design), :meth:`_interleave_conv`, and :meth:`_interleave_packed`
        (window/perceiver). This method is (optionally) wrapped in gradient
        checkpointing by the caller, so it must stay a pure
        ``(coarse_dense, pixels) -> (coarse_dense, pixels)`` function of its tensor
        inputs.
        """
        if self.pixel_branch_type == "joint":
            return self._interleave_joint(block_idx, coarse_dense, pixels, ctx)
        assert self.pixel_steps is not None
        step_idx = (block_idx + 1) // self.pixel_every_k_blocks - 1
        step = self.pixel_steps[step_idx]
        if self.pixel_branch_type == "conv":
            return self._interleave_conv(step, coarse_dense, pixels, ctx)
        return self._interleave_packed(step, coarse_dense, pixels, ctx)

    def _interleave_joint(
        self,
        block_idx: int,
        coarse_dense: Tensor,
        pixels: Tensor,
        ctx: PixelBranchContext,
    ) -> tuple[Tensor, Tensor]:
        """Run one ``"joint"`` pixel-branch block (the original design).

        All modalities' ONLINE units are processed as one concatenated tensor
        (``pixels``, in ``ctx.states`` order), so each module below runs exactly once
        per block:

        1. pixel <- coarse: gated FiLM conditioning of each unit's pixels on its
           coarse token (see :class:`PixelFiLM`);
        2. joint spatio-temporal attention over all (offset, modality, band set,
           timestep) tokens at the patch;
        3. coarse <- pixel: each ONLINE coarse token queries its unit's pixels and the
           result is written back into ``coarse_dense``. Placed last so every pixel
           module feeds the coarse output each block (keeping all pixel params on the
           autograd graph).
        """
        b, n, d = coarse_dense.shape
        coarse_flat = coarse_dense.reshape(b * n, d)
        coarse_units = coarse_flat[ctx.coarse_idx]  # [total_units, Dc]

        if self.pixel_film is not None:
            pixels = self.pixel_film[block_idx](pixels, coarse_units)
        assert ctx.loc is not None
        pixels = self._pixel_patch_attn(block_idx, pixels, ctx.loc)

        if self.coarse_to_pixel is not None:
            delta = chunked_batch_attn(
                self.coarse_to_pixel[block_idx], coarse_units[:, None], pixels
            )
            # index_add (indices are unique) rather than index_copy of
            # ``coarse_units + delta``: same result, far cheaper backward. The cast
            # matches the coarse dtype (under autocast the delta may be bf16 while
            # the dense coarse tokens are float32).
            coarse_flat = coarse_flat.index_add(
                0, ctx.coarse_idx, delta[:, 0].to(coarse_flat.dtype)
            )
            coarse_dense = coarse_flat.view(b, n, d)

        return coarse_dense, pixels

    def _interleave_packed(
        self,
        step: nn.Module,
        coarse_dense: Tensor,
        pixels: Tensor,
        ctx: PixelBranchContext,
    ) -> tuple[Tensor, Tensor]:
        """Run one packed (``"window"`` / ``"perceiver"``) pixel-branch step.

        The step consumes the packed ONLINE units and each unit's coarse token, and
        returns the updated pixels plus an optional residual update for the ONLINE
        coarse tokens (scattered back with one index op).
        """
        b, n, d = coarse_dense.shape
        coarse_flat = coarse_dense.reshape(b * n, d)
        coarse_units = coarse_flat[ctx.coarse_idx]  # [total_units, Dc]
        pixels, delta = step(pixels, coarse_units)
        if delta is not None:
            # index_add (indices are unique) rather than index_copy of
            # ``coarse_units + delta``: same result, far cheaper backward. The cast
            # matches the coarse dtype (under autocast the delta may be bf16 while
            # the dense coarse tokens are float32).
            coarse_flat = coarse_flat.index_add(
                0, ctx.coarse_idx, delta.to(coarse_flat.dtype)
            )
            coarse_dense = coarse_flat.view(b, n, d)
        return coarse_dense, pixels

    def _interleave_conv(
        self,
        step: nn.Module,
        coarse_dense: Tensor,
        frames: Tensor,
        ctx: PixelBranchContext,
    ) -> tuple[Tensor, Tensor]:
        """Run one ``"conv"`` pixel-branch step on the dense frames.

        Gathers each frame's coarse tokens from the collapsed coarse layout (whole
        per-modality slabs -- masked positions are zeros there, and their FiLM /
        pooled outputs land on masked pixels / coarse positions that are never read),
        runs :class:`ConvPixelStep`, and adds the pooled pixel -> coarse update back
        onto the same slabs.
        """
        b, n, d = coarse_dense.shape
        patch_shape = next(iter(ctx.states.values())).patch_shape

        coarse_frames = []
        for modality, st in ctx.states.items():
            _, g1, g2, t, bs = st.grid
            off = ctx.coarse_offsets[modality]
            u = g1 * g2 * t * bs
            slab = coarse_dense[:, off : off + u].view(b, g1, g2, t, bs, d)
            coarse_frames.append(rearrange(slab, "b g1 g2 t bs d -> (b t bs) g1 g2 d"))
        frames, pooled = step(frames, torch.cat(coarse_frames, dim=0), patch_shape)

        delta_full = torch.zeros_like(coarse_dense)
        for (modality, st), pooled_m in zip(
            ctx.states.items(), pooled.split(ctx.frame_splits)
        ):
            _, g1, g2, t, bs = st.grid
            off = ctx.coarse_offsets[modality]
            u = g1 * g2 * t * bs
            delta_full[:, off : off + u] = rearrange(
                pooled_m, "(b t bs) g1 g2 d -> b (g1 g2 t bs) d", b=b, t=t, bs=bs
            )
        return coarse_dense + delta_full, frames

    def _pixel_patch_attn(
        self, block_idx: int, pixels: Tensor, loc: LocationGroupings
    ) -> Tensor:
        """Joint spatio-temporal self-attention over each patch's ONLINE pixels.

        ``pixels`` is the concatenation of every modality's ONLINE units,
        ``[num_units, P**2, Dp]``. Units are scattered into a padded
        ``[num_locations, max_units, P**2, Dp]`` buffer whose last two axes are then
        flattened into one sequence: every pixel attends over ALL
        ``(offset, modality, band set, timestep)`` tokens at its patch -- spatial and
        temporal mixing in a single attention -- but never across other patches.
        Offsets are told apart by the within-patch positional encoding added in
        :class:`PixelPatchEmbed`.
        """
        assert self.pixel_self_blocks is not None
        num_units, p2, d = pixels.shape
        nl, mu = loc.num_locations, loc.max_units
        pix_sorted = pixels[loc.order]
        buf = pixels.new_zeros(nl * mu, p2, d)
        buf[loc.scatter_pos] = pix_sorted
        # -> [nl, max_units * P**2, d]: one row per location, all of its tokens as one
        # sequence. A unit's P**2 pixels are valid keys exactly iff the unit is.
        x = buf.view(nl, mu * p2, d)
        key_mask = loc.valid.repeat_interleave(p2, dim=1)  # [nl, max_units * P**2]
        x = chunked_batch_attn(self.pixel_self_blocks[block_idx], x, key_mask)
        buf = x.reshape(nl * mu, p2, d)
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
    pixel_branch_type: str = "joint"
    pixel_every_k_blocks: int = 1
    pixel_conv_kernel: int = 3
    pixel_coarse_read_interval: int = 1
    pixel_dit_depth: int = 4
    pixel_film_from_coarse: bool = True
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
        if self.pixel_branch_type not in (
            "joint",
            "conv",
            "window",
            "perceiver",
            "pixeldit",
        ):
            raise ValueError(
                f"Unknown pixel_branch_type {self.pixel_branch_type!r}; expected "
                "'joint', 'conv', 'window', 'perceiver' or 'pixeldit'."
            )
        if self.pixel_every_k_blocks < 1:
            raise ValueError("pixel_every_k_blocks must be >= 1")
        if (
            self.pixel_branch_type in ("joint", "pixeldit")
            and self.pixel_every_k_blocks != 1
        ):
            raise ValueError(
                "pixel_every_k_blocks > 1 is only supported for the interleaved "
                "non-'joint' pixel branch types ('pixeldit' runs after the trunk)."
            )
        if self.pixel_dit_depth < 1:
            raise ValueError("pixel_dit_depth must be >= 1")
        if self.depth % self.pixel_every_k_blocks != 0:
            raise ValueError(
                f"depth ({self.depth}) must be divisible by pixel_every_k_blocks "
                f"({self.pixel_every_k_blocks}) so the last block runs the pixel "
                "branch."
            )
        if self.pixel_coarse_read_interval < 1:
            raise ValueError("pixel_coarse_read_interval must be >= 1")
        if self.pixel_conv_kernel % 2 != 1:
            raise ValueError("pixel_conv_kernel must be odd")

    def build(self) -> "DualResEncoder":
        """Build the dual-resolution encoder."""
        self.validate()
        kwargs = self.as_dict(exclude_none=True, recurse=False)
        kwargs.pop("supported_modality_names")
        kwargs["supported_modalities"] = self.supported_modalities
        logger.info(f"DualResEncoder kwargs: {kwargs}")
        return DualResEncoder(**kwargs)
