"""Convolutional pixel branch that initializes the Perceiver's sub-patch latents.

Ported (``"thinconv"`` variant) from ``origin/favyen/20260917-pixreg-v1_3``, where a
stack of unconditioned ConvNeXt-style units ran on the dense PIXEL grid of the
time-series input modalities and its final per-pixel features initialized the
pixel-resolution register grid through a zero-initialized projection.

Here the branch runs at the resolution of the Perceiver's latent grid
(``PerceiverConfig.pixel_latents``): with latents every ``s`` pixels (``s`` divides
the patch size, drawn per forward pass in training, 1 at eval), each
``(instance, timestep, band set)`` frame is pooled to ``(H / s, W / s)`` cells before
the convolutions, and the final per-cell features initialize the latent covering
that cell. At ``s = 1`` this is the original pixel-resolution branch.

**Mixing** (``mixing``). Each step is one ConvNeXt-style unit,
``x += mlp(mix(norm(x)))``, where ``mix`` is a depthwise spatial convolution over each
frame (``"space"``, the ported branch), that convolution followed by a depthwise 1D
convolution over the timesteps of each ``(modality, band set, cell)`` series
(``"space_time"``), or the temporal convolution alone (``"time"``). Nothing ever mixes
band sets or modalities.

**Register init** (``register_pool``). ``"mean"`` (the ported branch) averages the final
features over the ONLINE ``(timestep, band set, modality)`` units of each cell and
projects them. ``"modality_concat"`` averages over the ONLINE ``(timestep, band set)``
units of each modality, applies a per-modality ``Linear + GELU``, concatenates the
modalities (a modality with no ONLINE unit at the cell fills its slot with zeros) and
projects the concatenation.

**Leakage guard.** Masking is per token (``P x P`` pixels) and ``s`` divides ``P``, so
every ``s x s`` cell lies inside one token. The pooling averages ONLINE pixels only and
every cell with no ONLINE pixel is zeroed BEFORE the first convolution, so nothing the
convolutions propagate -- through space or time -- is derived from masked values. The
register-init pooling is ONLINE-only, so a masked band set or timestep contributes
exactly nothing.

**Mask-normalized convolutions** (``mask_normalized``). Random-mode masking leaves
token-shaped holes (zeros) in a frame that inference never has. With this option each
depthwise convolution sees only the ONLINE cells of its window and rescales by
``k**2 / (number of ONLINE cells in the window)`` (partial convolution), so an ONLINE
cell's output does not depend on how many of its neighbours were masked. The mask is
held fixed through the stack (holes are never filled), and the zero padding at the
frame border is treated as missing too, so border cells are renormalized as well.
Spatial mixing only.

**Init equivalence.** ``zero_init`` zeroes the register-init projection, so at
initialization a model with this branch is EXACTLY the model without it.
"""

import logging
from dataclasses import dataclass

import torch
import torch.nn.functional as F
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

PIXEL_BRANCH_TYPES = ("thinconv",)
PIXEL_BRANCH_MIXINGS = ("space", "space_time", "time")
PIXEL_BRANCH_REGISTER_POOLS = ("mean", "modality_concat")


def get_pixel_branch_modalities(supported_modalities: list[ModalitySpec]) -> list[str]:
    """Return names of modalities that get a pixel branch (spatial + multitemporal)."""
    return [m.name for m in supported_modalities if m.is_spatial and m.is_multitemporal]


def pool_online_pixels(x: Tensor, online: Tensor, stride: int) -> tuple[Tensor, Tensor]:
    """Mean of the ONLINE pixels of every ``stride x stride`` cell.

    Args:
        x: ``[B, H, W, T, C]`` pixel values.
        online: ``[B, H, W, T]`` bool, True at ONLINE pixels.
        stride: Cell side in pixels (divides ``H`` and ``W``).

    Returns:
        ``([B, H / s, W / s, T, C]`` pooled values (0 at cells with no ONLINE pixel),
        ``[B, H / s, W / s, T]`` bool, True at cells with an ONLINE pixel).
    """
    x = torch.where(online[..., None], x, torch.zeros_like(x))
    if stride == 1:
        return x, online
    total = reduce(x, "b (h i) (w j) t c -> b h w t c", "sum", i=stride, j=stride)
    count = reduce(
        online.to(x.dtype), "b (h i) (w j) t -> b h w t", "sum", i=stride, j=stride
    )
    return total / count.clamp(min=1)[..., None], count > 0


@dataclass
class PixelModalityFrames:
    """Bookkeeping for one modality's frames."""

    grid: tuple[int, int, int, int, int]
    """The ``(B, H / s, W / s, T, band_sets)`` cell grid shape."""
    online: Tensor
    """``[B, H / s, W / s, T, band_sets]`` bool, True at cells with an ONLINE pixel."""


@dataclass
class PixelFrameContext:
    """Fixed per-forward bookkeeping for the pixel branch.

    The frame tensor is ``[N, T, H / s, W / s, Dp]``: one ``T``-frame series per
    ``(modality, instance, band set)``, the modalities concatenated along ``N`` in
    ``states`` iteration order; ``series_splits`` splits them back.

    Attributes:
        states: Per-modality frame bookkeeping.
        series_splits: Series (``B * band_sets``) per modality along ``N``.
        cell_hw: ``(H / s, W / s)`` shared cell grid.
        batch_size: ``B``.
        frame_online: ``[N, T, H / s, W / s, 1]`` ONLINE indicator of every frame
            cell, in the frame tensor's dtype.
    """

    states: dict[str, PixelModalityFrames]
    series_splits: list[int]
    cell_hw: tuple[int, int]
    batch_size: int
    frame_online: Tensor


class PixelPatchEmbed(nn.Module):
    """Per-cell linear embedding for spatial, multitemporal modalities.

    Produces ``[B, H / s, W / s, T, band_sets, pixel_embedding_size]`` tokens: the
    ONLINE pixels of each ``s x s`` cell are averaged (a linear map commutes with the
    mean, so this equals embedding every pixel and pooling) and embedded by a
    per-``(modality, band set)`` linear. An additive 1D sin/cos temporal encoding plus a
    2D sin/cos encoding of the cell centre's offset *within its patch* (in pixels) are
    applied.
    """

    def __init__(
        self,
        supported_modality_names: list[str],
        pixel_embedding_size: int,
        tokenization_config: TokenizationConfig | None = None,
    ) -> None:
        """Initialize the embedding.

        Args:
            supported_modality_names: Modalities to build a pixel branch for. Only
                spatial + multitemporal modalities are kept.
            pixel_embedding_size: Per-cell embedding dimension.
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
        self, input_data: MaskedOlmoEarthSample, patch_size: int, stride: int
    ) -> dict[str, Tensor]:
        """Return per-cell tokens and ONLINE indicators for each pixel modality.

        Args:
            input_data: The masked input sample.
            patch_size: Patch size of this forward pass.
            stride: Cell side in pixels (divides ``patch_size``).

        Returns:
            Dict mapping ``modality`` -> ``[B, H / s, W / s, T, band_sets, Dp]`` and
            ``modality_mask`` -> ``[B, H / s, W / s, T, band_sets]`` bool (True at
            cells with an ONLINE pixel).
        """
        output: dict[str, Tensor] = {}
        supported = set(self.pixel_modality_names)
        for modality in input_data.modalities:
            if modality not in supported:
                continue
            modality_data = getattr(input_data, modality)
            mask_name = input_data.get_masked_modality_name(modality)
            modality_mask = getattr(input_data, mask_name)
            num_bandsets = self.tokenization_config.get_num_bandsets(modality)
            tokens, masks = [], []
            for idx in range(num_bandsets):
                bands = getattr(self, self._buffer_name(modality, idx))
                inp = torch.index_select(modality_data, -1, bands)
                online = modality_mask[..., idx] == MaskValue.ONLINE_ENCODER.value
                pooled, cell_online = pool_online_pixels(inp, online, stride)
                embed = self.per_modality_embeddings[modality][
                    self._embed_name(modality, idx)
                ]
                tokens.append(embed(pooled))  # [B, H/s, W/s, T, Dp]
                masks.append(cell_online)  # [B, H/s, W/s, T]
            modality_tokens = torch.stack(tokens, dim=-2)  # [B, H/s, W/s, T, bs, Dp]
            output[modality] = self._add_positional_encodings(
                modality_tokens, patch_size, stride
            )
            output[mask_name] = torch.stack(masks, dim=-1)
        return output

    def _add_positional_encodings(
        self, tokens: Tensor, patch_size: int, stride: int
    ) -> Tensor:
        """Add additive sin/cos encodings: temporal (1D over T) + within-patch (2D).

        The 2D encoding is over the cell centre's offset within its patch, in pixels:
        ``(k * s) % P + (s - 1) / 2`` for cell ``k``. At ``s = 1`` this is the integer
        pixel offset ``h % P`` of the pixel-resolution branch. Patch-to-patch position
        is deliberately NOT encoded -- the convolutions are translation-equivariant and
        the latent's location is carried by its RoPE coordinates.
        """
        _, h, w, t, _, _ = tokens.shape
        device = tokens.device
        temporal = get_1d_sincos_pos_encoding(
            torch.arange(t, device=device, dtype=torch.float32),
            self.pixel_embedding_size,
        )  # [T, Dp]
        centre = (stride - 1) / 2

        def axis(n: int) -> Tensor:
            return (torch.arange(n, device=device) * stride) % patch_size + centre

        offsets = torch.stack(
            torch.meshgrid(axis(h), axis(w), indexing="ij"), dim=0
        ).float()  # [2, H/s, W/s]
        spatial = get_2d_sincos_pos_encoding(offsets, self.pixel_embedding_size).view(
            h, w, self.pixel_embedding_size
        )
        # Cast to the token dtype: under mixed precision the tokens are bf16 and a
        # float32 addition would silently promote the whole branch to float32.
        enc = temporal[None, None, :, None, :] + spatial[:, :, None, None, :]
        return tokens + enc.to(tokens.dtype)[None]


class ThinConvStep(nn.Module):
    """One unconditioned ConvNeXt-style unit on the ``[N, T, h, w, Dp]`` frames.

    ``frames += mlp(mix(norm(frames)))`` with ``mix`` a depthwise spatial convolution
    (``"space"``), that convolution then a depthwise temporal convolution
    (``"space_time"``), or the temporal convolution alone (``"time"``). One MLP per
    unit whatever the mixing. Affine-free LayerNorm: the affine is redundant before the
    convolution / MLP that follow.

    The spatial convolution runs on every ``(series, timestep)`` frame; the temporal
    one on every ``(series, cell)`` sequence of ``T`` frames, zero-padded at both ends.

    With ``mask_normalized`` the spatial convolution is a partial convolution: it reads
    only the ONLINE cells of its window and rescales by ``k**2 / count`` before the bias
    (``scale`` below), see the module docstring.
    """

    def __init__(
        self,
        pixel_dim: int,
        kernel_size: int,
        mlp_ratio: float,
        mixing: str = "space",
        time_kernel: int = 3,
        mask_normalized: bool = False,
    ) -> None:
        """Initialize the step.

        Args:
            pixel_dim: Pixel embedding dimension.
            kernel_size: Spatial depthwise convolution kernel size (odd).
            mlp_ratio: Pointwise MLP hidden-dim ratio.
            mixing: One of :data:`PIXEL_BRANCH_MIXINGS`.
            time_kernel: Temporal depthwise convolution kernel size (odd).
            mask_normalized: Use the mask-normalized (partial) spatial convolution.
        """
        super().__init__()
        if mixing not in PIXEL_BRANCH_MIXINGS:
            raise ValueError(
                f"mixing must be one of {PIXEL_BRANCH_MIXINGS}, got {mixing!r}"
            )
        if kernel_size % 2 != 1:
            raise ValueError(f"kernel_size must be odd, got {kernel_size}")
        if time_kernel % 2 != 1:
            raise ValueError(f"time_kernel must be odd, got {time_kernel}")
        if mask_normalized and mixing != "space":
            raise ValueError("mask_normalized supports only mixing='space'")
        self.mask_normalized = mask_normalized
        self.norm = nn.LayerNorm(pixel_dim, elementwise_affine=False)
        self.dwconv: nn.Conv2d | None = None
        self.dwconv_t: nn.Conv1d | None = None
        if mixing in ("space", "space_time"):
            self.dwconv = nn.Conv2d(
                pixel_dim,
                pixel_dim,
                kernel_size,
                padding=kernel_size // 2,
                groups=pixel_dim,
            )
        if mixing in ("space_time", "time"):
            self.dwconv_t = nn.Conv1d(
                pixel_dim,
                pixel_dim,
                time_kernel,
                padding=time_kernel // 2,
                groups=pixel_dim,
            )
        self.mlp = Mlp(pixel_dim, hidden_features=int(pixel_dim * mlp_ratio))

    def _spatial(self, y: Tensor, valid: Tensor | None, scale: Tensor | None) -> Tensor:
        assert self.dwconv is not None
        n = y.shape[0]
        y = rearrange(y, "n t h w d -> (n t) d h w")
        if not self.mask_normalized:
            y = self.dwconv(y)
            return rearrange(y, "(n t) d h w -> n t h w d", n=n)
        assert valid is not None and scale is not None
        y = F.conv2d(
            y * rearrange(valid, "n t h w 1 -> (n t) 1 h w"),
            self.dwconv.weight,
            None,
            padding=self.dwconv.padding,
            groups=self.dwconv.groups,
        )
        y = rearrange(y, "(n t) d h w -> n t h w d", n=n)
        return y * scale + self.dwconv.bias

    def _temporal(self, y: Tensor) -> Tensor:
        assert self.dwconv_t is not None
        n, _, h, w, _ = y.shape
        y = self.dwconv_t(rearrange(y, "n t h w d -> (n h w) d t"))
        return rearrange(y, "(n h w) d t -> n t h w d", n=n, h=h, w=w)

    def forward(
        self,
        frames: Tensor,
        valid: Tensor | None = None,
        scale: Tensor | None = None,
    ) -> Tensor:
        """Run the step on ``[N, T, h, w, Dp]`` frames.

        Args:
            frames: ``[N, T, h, w, Dp]`` frames.
            valid: ``[N, T, h, w, 1]`` ONLINE indicator (``mask_normalized`` only).
            scale: ``[N, T, h, w, 1]`` partial-convolution rescale ``k**2 / count``
                (``mask_normalized`` only).
        """
        y = self.norm(frames)
        if self.dwconv is not None:
            y = self._spatial(y, valid, scale)
        if self.dwconv_t is not None:
            y = self._temporal(y)
        return frames + self.mlp(y)


class PixelRegisterBranch(nn.Module):
    """Convolutional branch whose output initializes the Perceiver's latent grid.

    Owns the cell embedding, the conv steps, and the register-init head ending in the
    zero-initialized projection. :meth:`forward` runs the three stages:

    1. :meth:`build_frames` pools and embeds the sample's ONLINE pixels per
       ``s x s`` cell, zeroes cells with no ONLINE pixel (the leakage guard) and
       returns the ``[N, T, h, w, Dp]`` frames + bookkeeping.
    2. :meth:`run_thin_steps` runs the whole conv stack once.
    3. :meth:`register_init` pools the final frames per cell -- ONLINE-only -- and
       maps them (zero-init) to the register width, see ``register_pool``.
    """

    def __init__(
        self,
        supported_modality_names: list[str],
        register_dim: int,
        pixel_dim: int = 128,
        branch_type: str = "thinconv",
        num_steps: int = 4,
        kernel_size: int = 3,
        mlp_ratio: float = 4.0,
        mask_normalized: bool = False,
        mixing: str = "space",
        time_kernel: int = 3,
        register_pool: str = "mean",
        tokenization_config: TokenizationConfig | None = None,
        grad_checkpointing: bool = True,
    ) -> None:
        """Initialize the branch.

        Args:
            supported_modality_names: Encoder modalities; spatial + multitemporal ones
                get frames.
            register_dim: Width of the latent the branch initializes.
            pixel_dim: Per-cell embedding dimension (Dp).
            branch_type: Only ``"thinconv"`` (standalone unconditioned stack).
            num_steps: Depth of the conv stack.
            kernel_size: Spatial depthwise convolution kernel size (odd).
            mlp_ratio: Pointwise MLP hidden-dim ratio.
            mask_normalized: Mask-normalized (partial) spatial convolutions.
            mixing: One of :data:`PIXEL_BRANCH_MIXINGS`, see the module docstring.
            time_kernel: Temporal depthwise convolution kernel size (odd).
            register_pool: One of :data:`PIXEL_BRANCH_REGISTER_POOLS`, see the module
                docstring.
            tokenization_config: Band-grouping config (shared with the coarse embed).
            grad_checkpointing: Recompute each step in backward instead of storing its
                activations. The steps have no dropout, so recomputation is
                deterministic.
        """
        super().__init__()
        if branch_type not in PIXEL_BRANCH_TYPES:
            raise ValueError(
                f"branch_type must be one of {PIXEL_BRANCH_TYPES}, got {branch_type!r}"
            )
        if register_pool not in PIXEL_BRANCH_REGISTER_POOLS:
            raise ValueError(
                f"register_pool must be one of {PIXEL_BRANCH_REGISTER_POOLS}, got "
                f"{register_pool!r}"
            )
        if num_steps < 1:
            raise ValueError(f"num_steps must be >= 1, got {num_steps}")
        self.branch_type = branch_type
        self.pixel_dim = pixel_dim
        self.kernel_size = kernel_size
        self.mask_normalized = mask_normalized
        self.register_pool = register_pool
        self.grad_checkpointing = grad_checkpointing
        self.embed = PixelPatchEmbed(
            supported_modality_names,
            pixel_dim,
            tokenization_config=tokenization_config,
        )
        self.pixel_modality_names = self.embed.pixel_modality_names
        self.steps = nn.ModuleList(
            [
                ThinConvStep(
                    pixel_dim,
                    kernel_size,
                    mlp_ratio,
                    mixing=mixing,
                    time_kernel=time_kernel,
                    mask_normalized=mask_normalized,
                )
                for _ in range(num_steps)
            ]
        )
        self.norm_register = nn.LayerNorm(pixel_dim, elementwise_affine=False)
        self.per_modality_proj: nn.ModuleDict | None = None
        if register_pool == "modality_concat":
            self.per_modality_proj = nn.ModuleDict(
                {m: nn.Linear(pixel_dim, pixel_dim) for m in self.pixel_modality_names}
            )
            register_in = pixel_dim * len(self.pixel_modality_names)
        else:
            register_in = pixel_dim
        self.to_register = nn.Linear(register_in, register_dim)

    def zero_init(self) -> None:
        """Zero the register-init projection: the model equals the branch-free one at init.

        Call AFTER any blanket weight init. Gradients still reach every branch
        parameter through the zeroed projection once its weight moves off zero.
        """
        nn.init.zeros_(self.to_register.weight)
        nn.init.zeros_(self.to_register.bias)

    def build_frames(
        self, input_data: MaskedOlmoEarthSample, patch_size: int, stride: int
    ) -> tuple[Tensor | None, PixelFrameContext | None]:
        """Embed the sample into ``[N, T, H / s, W / s, Dp]`` frames (masked cells zeroed).

        Returns ``(None, None)`` when no pixel modality is present.
        """
        if patch_size % stride != 0:
            raise ValueError(
                f"latent stride {stride} does not divide patch size {patch_size}"
            )
        pixel_x = self.embed(input_data, patch_size, stride)
        states: dict[str, PixelModalityFrames] = {}
        frames: list[Tensor] = []
        onlines: list[Tensor] = []
        grid: tuple[int, int, int, int] | None = None
        for modality in self.pixel_modality_names:
            if modality not in pixel_x:
                continue
            tokens = pixel_x[modality]  # [B, H/s, W/s, T, bs, Dp]
            online = pixel_x[MaskedOlmoEarthSample.get_masked_modality_name(modality)]
            b, h, w, t, bs, _ = tokens.shape
            if grid is None:
                grid = (b, h, w, t)
            elif grid != (b, h, w, t):
                raise NotImplementedError(
                    "the pixel branch requires all pixel modalities to share one "
                    f"(B, H / s, W / s, T) grid, got {grid} and {(b, h, w, t)}"
                )
            tokens = tokens * online[..., None].to(tokens.dtype)
            frames.append(rearrange(tokens, "b h w t bs d -> (b bs) t h w d"))
            onlines.append(rearrange(online, "b h w t bs -> (b bs) t h w"))
            states[modality] = PixelModalityFrames(grid=(b, h, w, t, bs), online=online)
        if not states:
            return None, None
        assert grid is not None
        frame_tensor = torch.cat(frames, dim=0)
        return frame_tensor, PixelFrameContext(
            states=states,
            series_splits=[st.grid[0] * st.grid[4] for st in states.values()],
            cell_hw=(grid[1], grid[2]),
            batch_size=grid[0],
            frame_online=torch.cat(onlines, dim=0)[..., None].to(frame_tensor.dtype),
        )

    def _partial_conv_scale(self, valid: Tensor) -> Tensor:
        """``k**2 / count`` per cell (0 where no ONLINE cell is in the window)."""
        k = self.kernel_size
        n = valid.shape[0]
        ones = valid.new_ones(1, 1, k, k)
        count = F.conv2d(
            rearrange(valid, "n t h w 1 -> (n t) 1 h w"), ones, padding=k // 2
        )
        count = rearrange(count, "(n t) 1 h w -> n t h w 1", n=n)
        return torch.where(
            count > 0, (k * k) / count.clamp(min=1), torch.zeros_like(count)
        )

    def run_thin_steps(self, frames: Tensor, ctx: PixelFrameContext) -> Tensor:
        """Run the whole conv stack once (no coarse interaction)."""
        valid: Tensor | None = None
        scale: Tensor | None = None
        if self.mask_normalized:
            valid = ctx.frame_online
            scale = self._partial_conv_scale(valid)
        for step in self.steps:
            if self.grad_checkpointing and torch.is_grad_enabled():
                frames = torch.utils.checkpoint.checkpoint(
                    step, frames, valid, scale, use_reentrant=False
                )
            else:
                frames = step(frames, valid, scale)
        return frames

    def _modality_sums(
        self, frames: Tensor, ctx: PixelFrameContext
    ) -> dict[str, tuple[Tensor, Tensor]]:
        """Per modality, the ONLINE-masked sum and count over ``(timestep, band set)``.

        Returns ``{modality: ([B, h, w, Dp] sum, [B, h, w, 1] count)}``.
        """
        b = ctx.batch_size
        out = {}
        for (name, st), frames_m in zip(
            ctx.states.items(), frames.split(ctx.series_splits)
        ):
            bs = st.grid[4]
            fr = rearrange(frames_m, "(b bs) t h w d -> b t bs h w d", b=b, bs=bs)
            m = rearrange(st.online, "b h w t bs -> b t bs h w")[..., None]
            m = m.to(frames.dtype)
            out[name] = ((fr * m).sum(dim=(1, 2)), m.sum(dim=(1, 2)))
        return out

    def register_init(self, frames: Tensor, ctx: PixelFrameContext) -> Tensor:
        """Pool the final frames per cell (ONLINE-only) into the latent init.

        ``"mean"``: mask-weighted mean over the ``(timestep, band set, modality)``
        units of each cell, then the zero-initialized projection.
        ``"modality_concat"``: per modality the mask-weighted mean over its
        ``(timestep, band set)`` units, ``Linear + GELU``, zeroed where the modality
        has no ONLINE unit; concatenated over :attr:`pixel_modality_names` (absent
        modalities give zero slots), then the zero-initialized projection. Either way a
        cell with no ONLINE unit anywhere contributes exactly zero.

        Returns:
            ``[B, (H / s) * (W / s), register_dim]`` additive init, rows in row-major
            ``(h, w)`` order (matching the Perceiver's latent grid layout).
        """
        sums = self._modality_sums(frames, ctx)
        if self.per_modality_proj is None:
            total = sum(s for s, _ in sums.values())
            count = sum(c for _, c in sums.values())
            assert isinstance(total, Tensor) and isinstance(count, Tensor)
            pooled = total / count.clamp(min=1)
            init = self.to_register(self.norm_register(pooled))
            return init.flatten(1, 2)
        h, w = ctx.cell_hw
        slots = []
        for name in self.pixel_modality_names:
            if name not in sums:
                slots.append(frames.new_zeros(ctx.batch_size, h, w, self.pixel_dim))
                continue
            total_m, count_m = sums[name]
            pooled = self.norm_register(total_m / count_m.clamp(min=1))
            slot = F.gelu(self.per_modality_proj[name](pooled))
            slots.append(slot * (count_m > 0).to(slot.dtype))
        init = self.to_register(torch.cat(slots, dim=-1))  # [B, h, w, D_reg]
        return init.flatten(1, 2)

    def forward(
        self, input_data: MaskedOlmoEarthSample, patch_size: int, stride: int
    ) -> Tensor | None:
        """The additive latent init ``[B, (H / s) * (W / s), register_dim]`` (or None)."""
        frames, ctx = self.build_frames(input_data, patch_size, stride)
        if frames is None or ctx is None:
            return None
        return self.register_init(self.run_thin_steps(frames, ctx), ctx)
