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

**Leakage guard.** Masking is per token (``P x P`` pixels) and ``s`` divides ``P``, so
every ``s x s`` cell lies inside one token. The pooling averages ONLINE pixels only and
every cell with no ONLINE pixel is zeroed BEFORE the first convolution, so nothing the
depthwise convolutions propagate is derived from masked values. Frames are convolved
independently (no mixing across timesteps, band sets or modalities), and the final
register-init pooling over ``(timestep, band set, modality)`` is ONLINE-only, so a
masked band set or timestep contributes exactly nothing.

**Mask-normalized convolutions** (``mask_normalized``). Random-mode masking leaves
token-shaped holes (zeros) in a frame that inference never has. With this option each
depthwise convolution sees only the ONLINE cells of its window and rescales by
``k**2 / (number of ONLINE cells in the window)`` (partial convolution), so an ONLINE
cell's output does not depend on how many of its neighbours were masked. The mask is
held fixed through the stack (holes are never filled), and the zero padding at the
frame border is treated as missing too, so border cells are renormalized as well.

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

    All cross-modality tensors concatenate the modalities in ``states`` iteration
    order; ``frame_splits`` splits them back.

    Attributes:
        states: Per-modality frame bookkeeping.
        frame_splits: Frames (``B * T * band_sets``) per modality in the concatenated
            frame tensor.
        cell_hw: ``(H / s, W / s)`` shared cell grid.
        frame_online: ``[F, H / s, W / s, 1]`` ONLINE indicator of every frame cell,
            in the frame tensor's dtype.
    """

    states: dict[str, PixelModalityFrames]
    frame_splits: list[int]
    cell_hw: tuple[int, int]
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


class PlainConvStep(nn.Module):
    """One unconditioned ConvNeXt-style unit on the frames.

    ``frames += mlp(dwconv(norm(frames)))``: a depthwise spatial convolution followed
    by a pointwise MLP. Affine-free LayerNorm: the affine is redundant before the
    convolution / MLP that follow.

    With ``mask_normalized`` the depthwise convolution is a partial convolution: it
    reads only the ONLINE cells of its window and rescales by ``k**2 / count`` before
    the bias (``scale`` below), see the module docstring.
    """

    def __init__(
        self,
        pixel_dim: int,
        kernel_size: int,
        mlp_ratio: float,
        mask_normalized: bool = False,
    ) -> None:
        """Initialize the step.

        Args:
            pixel_dim: Pixel embedding dimension.
            kernel_size: Depthwise convolution kernel size (odd).
            mlp_ratio: Pointwise MLP hidden-dim ratio.
            mask_normalized: Use the mask-normalized (partial) depthwise convolution.
        """
        super().__init__()
        if kernel_size % 2 != 1:
            raise ValueError(f"kernel_size must be odd, got {kernel_size}")
        self.mask_normalized = mask_normalized
        self.norm = nn.LayerNorm(pixel_dim, elementwise_affine=False)
        self.dwconv = nn.Conv2d(
            pixel_dim,
            pixel_dim,
            kernel_size,
            padding=kernel_size // 2,
            groups=pixel_dim,
        )
        self.mlp = Mlp(pixel_dim, hidden_features=int(pixel_dim * mlp_ratio))

    def forward(
        self,
        frames: Tensor,
        valid: Tensor | None = None,
        scale: Tensor | None = None,
    ) -> Tensor:
        """Run the step on ``[F, h, w, Dp]`` frames.

        Args:
            frames: ``[F, h, w, Dp]`` frames.
            valid: ``[F, h, w, 1]`` ONLINE indicator (``mask_normalized`` only).
            scale: ``[F, h, w, 1]`` partial-convolution rescale ``k**2 / count``
                (``mask_normalized`` only).
        """
        y = self.norm(frames)
        if not self.mask_normalized:
            y = self.dwconv(y.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
            return frames + self.mlp(y)
        assert valid is not None and scale is not None
        y = F.conv2d(
            (y * valid).permute(0, 3, 1, 2),
            self.dwconv.weight,
            None,
            padding=self.dwconv.padding,
            groups=self.dwconv.groups,
        ).permute(0, 2, 3, 1)
        y = y * scale + self.dwconv.bias
        return frames + self.mlp(y)


class PixelRegisterBranch(nn.Module):
    """Convolutional branch whose output initializes the Perceiver's latent grid.

    Owns the cell embedding, the conv steps, and the zero-initialized
    ``pixel -> register_dim`` projection. :meth:`forward` runs the three stages:

    1. :meth:`build_frames` pools and embeds the sample's ONLINE pixels per
       ``s x s`` cell, zeroes cells with no ONLINE pixel (the leakage guard) and
       returns the concatenated frames + bookkeeping.
    2. :meth:`run_thin_steps` runs the whole conv stack once.
    3. :meth:`register_init` pools the final frames per cell over the
       ``(timestep, band set, modality)`` axes -- ONLINE-only -- and projects them
       (zero-init) to the register width.
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
            kernel_size: Depthwise convolution kernel size (odd).
            mlp_ratio: Pointwise MLP hidden-dim ratio.
            mask_normalized: Mask-normalized (partial) depthwise convolutions.
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
        if num_steps < 1:
            raise ValueError(f"num_steps must be >= 1, got {num_steps}")
        self.branch_type = branch_type
        self.pixel_dim = pixel_dim
        self.kernel_size = kernel_size
        self.mask_normalized = mask_normalized
        self.grad_checkpointing = grad_checkpointing
        self.embed = PixelPatchEmbed(
            supported_modality_names,
            pixel_dim,
            tokenization_config=tokenization_config,
        )
        self.pixel_modality_names = self.embed.pixel_modality_names
        self.steps = nn.ModuleList(
            [
                PlainConvStep(pixel_dim, kernel_size, mlp_ratio, mask_normalized)
                for _ in range(num_steps)
            ]
        )
        self.norm_register = nn.LayerNorm(pixel_dim, elementwise_affine=False)
        self.to_register = nn.Linear(pixel_dim, register_dim)

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
        """Embed the sample into ``[F, H / s, W / s, Dp]`` frames (masked cells zeroed).

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
        cell_hw: tuple[int, int] | None = None
        for modality in self.pixel_modality_names:
            if modality not in pixel_x:
                continue
            tokens = pixel_x[modality]  # [B, H/s, W/s, T, bs, Dp]
            online = pixel_x[MaskedOlmoEarthSample.get_masked_modality_name(modality)]
            b, h, w, t, bs, _ = tokens.shape
            if cell_hw is None:
                cell_hw = (h, w)
            elif cell_hw != (h, w):
                raise NotImplementedError(
                    "the pixel branch requires all pixel modalities to share one "
                    f"pixel grid, got {cell_hw} and {(h, w)} cells"
                )
            tokens = tokens * online[..., None].to(tokens.dtype)
            frames.append(rearrange(tokens, "b h w t bs d -> (b t bs) h w d"))
            onlines.append(rearrange(online, "b h w t bs -> (b t bs) h w"))
            states[modality] = PixelModalityFrames(grid=(b, h, w, t, bs), online=online)
        if not states:
            return None, None
        assert cell_hw is not None
        frame_tensor = torch.cat(frames, dim=0)
        return frame_tensor, PixelFrameContext(
            states=states,
            frame_splits=[
                st.grid[0] * st.grid[3] * st.grid[4] for st in states.values()
            ],
            cell_hw=cell_hw,
            frame_online=torch.cat(onlines, dim=0)[..., None].to(frame_tensor.dtype),
        )

    def _partial_conv_scale(self, valid: Tensor) -> Tensor:
        """``k**2 / count`` per cell (0 where no ONLINE cell is in the window)."""
        k = self.kernel_size
        ones = valid.new_ones(1, 1, k, k)
        count = F.conv2d(valid.permute(0, 3, 1, 2), ones, padding=k // 2).permute(
            0, 2, 3, 1
        )
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

    def register_init(self, frames: Tensor, ctx: PixelFrameContext) -> Tensor:
        """Pool the final frames per cell (ONLINE-only) into the latent init.

        Mask-weighted mean over the ``(timestep, band set, modality)`` axes at each
        cell, then the zero-initialized projection. A cell with no ONLINE unit
        anywhere contributes exactly zero (its latent starts from the bare learned
        latent).

        Returns:
            ``[B, (H / s) * (W / s), register_dim]`` additive init, rows in row-major
            ``(h, w)`` order (matching the Perceiver's latent grid layout).
        """
        h, w = ctx.cell_hw
        b = next(iter(ctx.states.values())).grid[0]
        total = frames.new_zeros(b, h, w, self.pixel_dim)
        count = frames.new_zeros(b, h, w, 1)
        for st, frames_m in zip(ctx.states.values(), frames.split(ctx.frame_splits)):
            _, _, _, t, bs = st.grid
            fr = rearrange(frames_m, "(b t bs) h w d -> b t bs h w d", b=b, t=t, bs=bs)
            m = rearrange(st.online, "b h w t bs -> b t bs h w")[..., None]
            m = m.to(frames.dtype)
            total = total + (fr * m).sum(dim=(1, 2))
            count = count + m.sum(dim=(1, 2))
        pooled = total / count.clamp(min=1)
        init = self.to_register(self.norm_register(pooled))  # [B, h, w, D_reg]
        return init.flatten(1, 2)

    def forward(
        self, input_data: MaskedOlmoEarthSample, patch_size: int, stride: int
    ) -> Tensor | None:
        """The additive latent init ``[B, (H / s) * (W / s), register_dim]`` (or None)."""
        frames, ctx = self.build_frames(input_data, patch_size, stride)
        if frames is None or ctx is None:
            return None
        return self.register_init(self.run_thin_steps(frames, ctx), ctx)
