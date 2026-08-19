"""Convolutional pixel branch that initializes pixel-resolution registers.

Ported from the dual-resolution encoder program (``origin/favyen/20260709-pixel-branch``,
``dual_res_encoder.py``): of the pixel-branch designs explored there, the **conv**
variant (:class:`ConvPixelStep` -- a FiLM-modulated ConvNeXt-style unit on the dense
pixel grid, interleaved with the coarse trunk at a reduced cadence, zero-init fusion in
both directions) was the clear winner on cost (~1.5x a plain step vs ~8.7x for joint
per-patch attention, which OOM'd at width 128). This module ports exactly that variant
-- plus an unconditioned "thin" version -- with ONE consumer change: instead of feeding
per-pixel decoders, the branch's final per-pixel features initialize the pixel-resolution
register grid of the :class:`~olmoearth_pretrain.nn.flexi_vit.SpatialRegisterBottleneck`
(``register_pixel_grid``), through a zero-initialized projection.

Two branch types:

* ``"conv"``: :class:`ConvPixelStep` interleaved with the coarse trunk -- after every
  ``k``-th encoder block the dense pixel frames are refined, conditioned on the coarse
  tokens via per-patch FiLM, and a mean-pooled zero-init residual flows back into the
  coarse tokens. This is the old top pick, verbatim.
* ``"thinconv"``: :class:`PlainConvStep` x ``depth`` run ONCE, before/independent of the
  trunk -- no FiLM, no fusion back, no coarse interaction of any kind. Isolates whether
  the pixel branch benefits from talking to the coarse trunk versus merely supplying an
  independent high-resolution register init.

**Leakage guard.** Non-ONLINE pixels are zeroed BEFORE the first convolution, so
whatever the depthwise convolutions later propagate across patch boundaries is derived
from visible pixels only, and reconstruction of masked units cannot cheat. The final
register-init pooling is additionally ONLINE-only (mask-weighted mean over the
``(timestep, band set, modality)`` axes at each pixel), so a pixel whose every unit is
masked contributes exactly zero.

**Init equivalence.** ``zero_init`` closes every fusion path (FiLM gates, pixel->coarse
projections, the register-init projection), so at initialization a model with this
branch is EXACTLY the model without it -- the pixel parameters receive gradients but
contribute nothing at step 0.

Dense frames follow the old vocabulary: each modality's ``[B, H, W, T, band_sets, Dp]``
pixel tokens are flattened to per-``(instance, timestep, band set)`` **frames** and
concatenated across modalities, ``[F, H, W, Dp]``. Masking granularity is the **unit**
(one ``(spatial patch, timestep, band set)`` cell): within a unit all ``P**2`` pixels
share one mask value.
"""

import logging
import math
from dataclasses import dataclass, field

import torch
import torch.utils.checkpoint
from einops import rearrange, reduce
from torch import Tensor, nn

from olmoearth_pretrain.data.constants import Modality, ModalitySpec
from olmoearth_pretrain.datatypes import MaskedOlmoEarthSample, MaskValue
from olmoearth_pretrain.nn.attention import Mlp
from olmoearth_pretrain.nn.encodings import (
    get_1d_sincos_pos_encoding,
    get_2d_sincos_pos_encoding,
)
from olmoearth_pretrain.nn.tokenization import TokenizationConfig

logger = logging.getLogger(__name__)


def get_pixel_branch_modalities(supported_modalities: list[ModalitySpec]) -> list[str]:
    """Return names of modalities that get a pixel branch (spatial + multitemporal)."""
    return [m.name for m in supported_modalities if m.is_spatial and m.is_multitemporal]


def _modalities_to_process(available: list[str], supported: list[str]) -> list[str]:
    """Intersect available and supported modality names (order-stable)."""
    supported_set = set(supported)
    return [m for m in available if m in supported_set]


@dataclass
class PixelModalityFrames:
    """Bookkeeping for one modality's dense pixel frames."""

    grid: tuple[int, int, int, int, int]
    """The ``(B, G1, G2, T, band_sets)`` unit grid shape."""
    patch_shape: tuple[int, int]
    """``(P1, P2)`` pixels per patch side."""
    online: Tensor
    """``[B, H, W, T, band_sets]`` bool, True at ONLINE pixels."""


@dataclass
class PixelFrameContext:
    """Fixed per-forward bookkeeping for the pixel branch.

    All cross-modality tensors concatenate the modalities in ``states`` iteration
    order; ``frame_splits`` splits them back.

    Attributes:
        states: Per-modality frame bookkeeping.
        frame_splits: Frames (``B * T * band_sets``) per modality in the concatenated
            dense frame tensor.
        pixel_hw: ``(H, W)`` shared dense pixel grid.
        coarse_offsets: Start of each modality's token slab on the per-instance
            collapsed coarse token axis. Filled by the encoder (which owns the
            collapsed layout) before any ``"conv"`` fusion step runs.
    """

    states: dict[str, PixelModalityFrames]
    frame_splits: list[int]
    pixel_hw: tuple[int, int]
    coarse_offsets: dict[str, int] = field(default_factory=dict)


class PixelPatchEmbed(nn.Module):
    """Per-pixel linear embedding for spatial, multitemporal modalities.

    Produces ``[B, H, W, T, band_sets, pixel_embedding_size]`` tokens (native input
    resolution -- one token per pixel, per timestep, per band set). An additive 1D
    sin/cos temporal encoding plus a 2D sin/cos encoding of the *within-patch* pixel
    offset are applied; modality/band-set identity is carried implicitly by the
    separate per-``(modality, band set)`` linear weights.
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
        modalities = _modalities_to_process(
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
        every patch size. Patch-to-patch position is deliberately NOT encoded -- the
        convolutions are translation-equivariant and the patch's location is carried
        by the coarse branch / register RoPE coordinates.
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
        # Cast to the token dtype: under mixed precision the tokens are bf16 and a
        # float32 addition would silently promote the whole pixel branch to float32.
        enc = (
            temporal[None, None, :, None, :] + spatial[:, :, None, None, :]
        )  # [H, W, T, bs=1, Dp]
        return tokens + enc.to(tokens.dtype)[None]


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


class PlainConvStep(nn.Module):
    """One unconditioned ConvNeXt-style unit on the dense pixel frames.

    The ``"thinconv"`` building block: ``frames += mlp(dwconv(norm(frames)))`` -- the
    :class:`ConvPixelStep` minus every coarse coupling (no FiLM modulation, no gate,
    no pooled residual back to the coarse tokens). Affine-free LayerNorm for the same
    pixel-row-count reason.
    """

    def __init__(self, pixel_dim: int, kernel_size: int, mlp_ratio: float) -> None:
        """Initialize the step.

        Args:
            pixel_dim: Pixel embedding dimension.
            kernel_size: Depthwise convolution kernel size (odd).
            mlp_ratio: Pointwise MLP hidden-dim ratio.
        """
        super().__init__()
        self.norm = nn.LayerNorm(pixel_dim, elementwise_affine=False)
        self.dwconv = nn.Conv2d(
            pixel_dim,
            pixel_dim,
            kernel_size,
            padding=kernel_size // 2,
            groups=pixel_dim,
        )
        self.mlp = Mlp(pixel_dim, hidden_features=int(pixel_dim * mlp_ratio))

    def forward(self, frames: Tensor) -> Tensor:
        """Run the step on ``[F, H, W, Dp]`` dense frames."""
        y = self.norm(frames)
        y = self.dwconv(y.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        return frames + self.mlp(y)


class PixelRegisterBranch(nn.Module):
    """Dense convolutional pixel branch whose output initializes pixel registers.

    Owns the pixel patch embedding, the conv steps, and the zero-initialized
    ``pixel -> register_dim`` projection. The encoder drives it:

    1. :meth:`build_frames` embeds the sample's pixels, zeroes non-ONLINE pixels
       (the leakage guard) and returns the concatenated dense frames + bookkeeping.
    2. ``"conv"`` only: :meth:`run_conv_step` is invoked after every ``k``-th coarse
       block with the CURRENT dense coarse tokens; it refines the frames (FiLM from
       the coarse tokens) and returns a residual update for them.
       ``"thinconv"`` only: :meth:`run_thin_steps` runs the whole stack once,
       independent of the trunk.
    3. :meth:`register_init` pools the final frames per pixel over the
       ``(timestep, band set, modality)`` axes -- ONLINE-only -- and projects them
       (zero-init) to the register width, giving the additive initialization for the
       cloned register latent.
    """

    def __init__(
        self,
        supported_modality_names: list[str],
        coarse_dim: int,
        register_dim: int,
        pixel_dim: int = 128,
        branch_type: str = "conv",
        num_steps: int = 3,
        kernel_size: int = 3,
        mlp_ratio: float = 4.0,
        tokenization_config: TokenizationConfig | None = None,
        grad_checkpointing: bool = True,
    ) -> None:
        """Initialize the branch.

        Args:
            supported_modality_names: Encoder modalities; spatial + multitemporal ones
                get pixel frames.
            coarse_dim: Coarse-token embedding dimension (``"conv"`` FiLM/fusion only).
            register_dim: Width of the register latent the branch initializes.
            pixel_dim: Per-pixel embedding dimension (Dp).
            branch_type: ``"conv"`` (interleaved, FiLM + fusion -- the old top pick) or
                ``"thinconv"`` (standalone unconditioned stack, no coarse interaction).
            num_steps: Number of conv steps. For ``"conv"`` the encoder derives this as
                ``depth // pixel_every_k_blocks``; for ``"thinconv"`` it is the thin
                net's depth.
            kernel_size: Depthwise convolution kernel size (odd).
            mlp_ratio: Pointwise MLP hidden-dim ratio.
            tokenization_config: Band-grouping config (shared with the coarse embed).
            grad_checkpointing: Recompute each step in backward instead of storing its
                (pixel-resolution) activations. The steps have no dropout, so
                recomputation is deterministic.
        """
        super().__init__()
        if branch_type not in ("conv", "thinconv"):
            raise ValueError(
                f"branch_type must be 'conv' or 'thinconv', got {branch_type!r}"
            )
        self.branch_type = branch_type
        self.pixel_dim = pixel_dim
        self.grad_checkpointing = grad_checkpointing
        self.embed = PixelPatchEmbed(
            supported_modality_names,
            pixel_dim,
            tokenization_config=tokenization_config,
        )
        self.pixel_modality_names = self.embed.pixel_modality_names
        if branch_type == "conv":
            self.steps = nn.ModuleList(
                [
                    ConvPixelStep(coarse_dim, pixel_dim, kernel_size, mlp_ratio)
                    for _ in range(num_steps)
                ]
            )
        else:
            self.steps = nn.ModuleList(
                [
                    PlainConvStep(pixel_dim, kernel_size, mlp_ratio)
                    for _ in range(num_steps)
                ]
            )
        # Affine-free: the projection consumes pixel-resolution rows, and the affine
        # is redundant with the (zero-init) linear that follows.
        self.norm_register = nn.LayerNorm(pixel_dim, elementwise_affine=False)
        self.to_register = nn.Linear(pixel_dim, register_dim)

    def zero_init(self) -> None:
        """Zero every fusion path: the model equals the branch-free model at init.

        Call AFTER any blanket weight init. Closes the conv steps' FiLM gates and
        pixel -> coarse projections (``"conv"`` only) and the register-init
        projection, so at step 0 the coarse trunk and the register grid are exactly
        what they would be without the branch, while gradients still reach every
        pixel parameter through the zeroed projections.
        """
        for step in self.steps:
            if isinstance(step, ConvPixelStep):
                step.zero_init()
        nn.init.zeros_(self.to_register.weight)
        nn.init.zeros_(self.to_register.bias)

    def build_frames(
        self, input_data: MaskedOlmoEarthSample, patch_size: int
    ) -> tuple[Tensor | None, PixelFrameContext | None]:
        """Embed the sample's pixels into dense frames (non-ONLINE pixels zeroed).

        Returns ``(None, None)`` when no pixel modality is present. The zeroing BEFORE
        any convolution is the leakage guard: nothing derived from masked units can
        ever enter the branch.

        Args:
            input_data: The masked input sample.
            patch_size: Coarse patch size.

        Returns:
            ``(frames, ctx)``: ``[F, H, W, Dp]`` dense frames (concatenated over
            modalities) and the per-forward bookkeeping.
        """
        pixel_x = self.embed(input_data, patch_size)
        states: dict[str, PixelModalityFrames] = {}
        frames: list[Tensor] = []
        pixel_hw: tuple[int, int] | None = None
        for modality in self.pixel_modality_names:
            if modality not in pixel_x:
                continue
            tokens = pixel_x[modality]  # [B, H, W, T, bs, Dp]
            mask_name = MaskedOlmoEarthSample.get_masked_modality_name(modality)
            online = pixel_x[mask_name] == MaskValue.ONLINE_ENCODER.value
            b, h, w, t, bs, _ = tokens.shape
            if h % patch_size != 0 or w % patch_size != 0:
                raise ValueError(
                    f"pixel grid ({h}x{w}) of {modality} is not divisible by the "
                    f"patch size ({patch_size})"
                )
            if pixel_hw is None:
                pixel_hw = (h, w)
            elif pixel_hw != (h, w):
                raise NotImplementedError(
                    "the dense conv pixel branch requires all pixel modalities to "
                    f"share one pixel grid, got {pixel_hw} and {(h, w)}"
                )
            tokens = tokens * online[..., None].to(tokens.dtype)
            frames.append(rearrange(tokens, "b h w t bs d -> (b t bs) h w d"))
            states[modality] = PixelModalityFrames(
                grid=(b, h // patch_size, w // patch_size, t, bs),
                patch_shape=(patch_size, patch_size),
                online=online,
            )
        if not states:
            return None, None
        assert pixel_hw is not None
        return torch.cat(frames, dim=0), PixelFrameContext(
            states=states,
            frame_splits=[
                st.grid[0] * st.grid[3] * st.grid[4] for st in states.values()
            ],
            pixel_hw=pixel_hw,
        )

    def run_thin_steps(self, frames: Tensor) -> Tensor:
        """Run the whole ``"thinconv"`` stack once (no coarse interaction)."""
        assert self.branch_type == "thinconv"
        for step in self.steps:
            if self.grad_checkpointing and torch.is_grad_enabled():
                frames = torch.utils.checkpoint.checkpoint(
                    step, frames, use_reentrant=False
                )
            else:
                frames = step(frames)
        return frames

    def run_conv_step(
        self,
        step_idx: int,
        frames: Tensor,
        coarse_dense: Tensor,
        ctx: PixelFrameContext,
    ) -> tuple[Tensor, Tensor]:
        """Run one ``"conv"`` step against the CURRENT dense coarse tokens.

        Gathers each frame's coarse tokens from the collapsed coarse layout (whole
        per-modality slabs -- masked positions are zeros there, and their FiLM /
        pooled outputs land on masked pixels / coarse positions that are never read),
        runs :class:`ConvPixelStep`, and adds the pooled pixel -> coarse update back
        onto the same slabs.

        Args:
            step_idx: Which conv step to run.
            frames: ``[F, H, W, Dp]`` current dense frames.
            coarse_dense: ``[B, N, Dc]`` coarse tokens in the collapsed (dense,
                pre-packing) layout, masked positions zeroed.
            ctx: The frame context; ``ctx.coarse_offsets`` must be filled.

        Returns:
            ``(coarse_dense, frames)``: both updated.
        """
        assert self.branch_type == "conv"
        step = self.steps[step_idx]
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

    def register_init(self, frames: Tensor, ctx: PixelFrameContext) -> Tensor:
        """Pool the final frames per pixel (ONLINE-only) into the register init.

        Mask-weighted mean over the ``(timestep, band set, modality)`` axes at each
        pixel, then the zero-initialized projection to the register width. A pixel
        with no ONLINE unit anywhere contributes exactly zero (its register cell
        starts from the bare learned latent).

        Args:
            frames: ``[F, H, W, Dp]`` final dense frames.
            ctx: The frame context.

        Returns:
            ``[B, H * W, register_dim]`` additive register initialization, rows in
            row-major ``(h, w)`` order (matching the pixel register grid layout).
        """
        h, w = ctx.pixel_hw
        first = next(iter(ctx.states.values()))
        b = first.grid[0]
        total = frames.new_zeros(b, h, w, self.pixel_dim)
        count = frames.new_zeros(b, h, w, 1)
        for st, frames_m in zip(ctx.states.values(), frames.split(ctx.frame_splits)):
            _, _, _, t, bs = st.grid
            fr = rearrange(frames_m, "(b t bs) h w d -> b t bs h w d", b=b, t=t, bs=bs)
            # [B, T, bs, H, W, 1] ONLINE indicator at pixel level.
            m = rearrange(st.online, "b h w t bs -> b t bs h w")[..., None]
            m = m.to(frames.dtype)
            total = total + (fr * m).sum(dim=(1, 2))
            count = count + m.sum(dim=(1, 2))
        pooled = total / count.clamp(min=1)
        init = self.to_register(self.norm_register(pooled))  # [B, H, W, D_reg]
        return init.flatten(1, 2)


def compute_coarse_offsets(dims_dict: dict[str, tuple]) -> dict[str, int]:
    """Start of each modality's token slab on the collapsed coarse token axis.

    ``dims_dict`` is the encoder's ``modalities_to_dims_dict`` (modality -> per-modality
    token shape) in the SAME iteration order that ``collapse_and_combine_hwtc``
    concatenates, so cumulative products of the middle dims give the slab offsets.
    """
    offsets: dict[str, int] = {}
    tokens_per_instance = 0
    for modality, dims in dims_dict.items():
        offsets[modality] = tokens_per_instance
        tokens_per_instance += math.prod(dims[1:-1])
    return offsets
