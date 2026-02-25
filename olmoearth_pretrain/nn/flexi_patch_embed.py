"""Flexible patch embedding Module.

Extended from: https://github.com/huggingface/pytorch-image-models/blob/main/timm/layers/patch_embed.py#L24
by https://github.com/bwconrad/flexivit/
"""

import logging
from collections.abc import Iterable
from typing import Any

import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor

from olmoearth_pretrain.data.constants import ModalitySpec

logger = logging.getLogger(__name__)


class FlexiPatchEmbed(nn.Module):
    """Flexible patch embedding nn.Module."""

    def __init__(
        self,
        modality_spec: ModalitySpec,
        patch_size_at_16: int | tuple[int, int],
        in_chans: int = 3,
        embedding_size: int = 128,
        norm_layer: nn.Module | None = None,
        bias: bool = True,
        interpolation: str = "bicubic",
        antialias: bool = True,
        use_linear_patch_embed: bool = True,
    ) -> None:
        """2D image to patch embedding w/ flexible patch sizes.

        Extended from: https://github.com/huggingface/pytorch-image-models/blob/main/timm/layers/patch_embed.py#L24
        by https://github.com/bwconrad/flexivit/

        Args:
            modality_spec: The modality spec for this modality
            patch_size_at_16: Base patch size. i.e the size of the parameter buffer at a resolution of 16
            in_chans: Number of input image channels
            embedding_size: Network embedding dimension size
            norm_layer: Optional normalization layer
            bias: Whether to use bias in convolution
            interpolation: Resize interpolation type
            antialias: Whether to apply antialiasing resizing
            use_linear_patch_embed: If True, use nn.Linear (reshape + matmul via cuBLAS GEMM).
                If False, use nn.Conv2d (required to load checkpoints trained before this flag existed).
        """
        super().__init__()

        self.embedding_size = embedding_size
        self.use_linear_patch_embed = use_linear_patch_embed

        self.modality_spec = modality_spec
        self.patch_size = self.to_2tuple(
            patch_size_at_16 * modality_spec.image_tile_size_factor
        )

        p_h, p_w = self.patch_size
        if use_linear_patch_embed:
            # Reshape patches to (p1 p2 c) then project — hits cuBLAS GEMM (always fast
            # on TensorCores) vs Conv2d which hits slow cuDNN paths for small in_chans.
            self.proj = nn.Linear(in_chans * p_h * p_w, embedding_size, bias=bias)
            # Keep PyTorch's default nn.Linear initialization (kaiming_uniform_) for
            # patch projection to match prior Conv2d behavior; overriding this with
            # encoder-level Xavier init correlated with a PASTIS regression.
            self.proj._skip_custom_init = True
        else:
            self.proj = nn.Conv2d(
                in_chans,
                embedding_size,
                kernel_size=self.patch_size,
                stride=self.patch_size,
                bias=bias,
            )
        self.norm = norm_layer(embedding_size) if norm_layer else nn.Identity()
        self.interpolation = interpolation
        self.antialias = antialias

    @staticmethod
    def to_2tuple(x: Any) -> Any:
        """Convert a scalar or 2-element iterable to a (h, w) tuple."""
        if isinstance(x, Iterable) and not isinstance(x, str):
            assert len(list(x)) == 2, "x must be a 2-tuple"
            return tuple(x)
        return (x, x)

    def _resolve_patch_size(
        self, patch_size: int | tuple[int, int] | None
    ) -> tuple[int, int]:
        """Resolve the effective patch size, applying the modality tile size factor."""
        if not patch_size:
            return self.patch_size
        if isinstance(patch_size, tuple):
            patch_size = (
                patch_size[0] * self.modality_spec.image_tile_size_factor,
                patch_size[1] * self.modality_spec.image_tile_size_factor,
            )
        else:
            patch_size = patch_size * self.modality_spec.image_tile_size_factor
        resolved = self.to_2tuple(patch_size)
        assert isinstance(resolved, tuple) and len(resolved) == 2
        return resolved

    def _project_linear(
        self,
        x: Tensor,
        h_patches: int,
        w_patches: int,
        batch_size: int,
        has_time_dim: bool,
        num_timesteps: int,
    ) -> Tensor:
        """Project patches using nn.Linear (reshape → cuBLAS GEMM → reshape)."""
        p_h, p_w = self.patch_size
        x = rearrange(x, "b c (h p1) (w p2) -> b (h w) (p1 p2 c)", p1=p_h, p2=p_w)
        x = self.proj(x)
        if has_time_dim:
            return rearrange(
                x,
                "(b t) (h w) d -> b h w t d",
                b=batch_size,
                t=num_timesteps,
                h=h_patches,
                w=w_patches,
            )
        return rearrange(x, "b (h w) d -> b h w d", h=h_patches, w=w_patches)

    def _project_conv(
        self,
        x: Tensor,
        batch_size: int,
        has_time_dim: bool,
        num_timesteps: int,
    ) -> Tensor:
        """Project patches using nn.Conv2d (for loading pre-linear checkpoints)."""
        x = self.proj(x)  # b c h w -> b d h_out w_out
        if has_time_dim:
            _, d, h, w = x.shape
            return rearrange(
                x,
                "(b t) d h w -> b h w t d",
                b=batch_size,
                t=num_timesteps,
                h=h,
                w=w,
            )
        return rearrange(x, "b d h w -> b h w d")

    def forward(
        self,
        x: Tensor,
        patch_size: int | tuple[int, int] | None = None,
    ) -> Tensor:
        """Forward pass for the FlexiPatchEmbed module.

        Args:
            x: Input tensor with shape [b, h, w, (t), c]
            patch_size: Patch size to use for the embedding. If None, uses the base patch size.
        """
        batch_size = x.shape[0]
        has_time_dim = len(x.shape) == 5
        num_timesteps = x.shape[3] if has_time_dim else 0

        if has_time_dim:
            x = rearrange(x, "b h w t c -> (b t) c h w")
        else:
            x = rearrange(x, "b h w c -> b c h w")

        patch_size = self._resolve_patch_size(patch_size)

        if patch_size != self.patch_size:
            shape = x.shape[-2:]
            new_shape = (
                shape[0] // patch_size[0] * self.patch_size[0],
                shape[1] // patch_size[1] * self.patch_size[1],
            )
            x = F.interpolate(
                x, size=new_shape, mode=self.interpolation, antialias=self.antialias
            )

        p_h, p_w = self.patch_size
        h_patches, w_patches = x.shape[2] // p_h, x.shape[3] // p_w

        if self.use_linear_patch_embed:
            x = self._project_linear(
                x, h_patches, w_patches, batch_size, has_time_dim, num_timesteps
            )
        else:
            x = self._project_conv(x, batch_size, has_time_dim, num_timesteps)

        return self.norm(x)


class FlexiPatchReconstruction(nn.Module):
    """Flexible patch reconstruction nn.Module."""

    def __init__(
        self,
        max_patch_size: int | tuple[int, int],
        out_chans: int = 3,
        embedding_size: int = 128,
        norm_layer: nn.Module | None = None,
        bias: bool = True,
        interpolation: str = "bicubic",
        antialias: bool = True,
    ) -> None:
        """Patch embeding to 2d image reconstruction w/ flexible patch sizes.

        Args:
            max_patch_size: Base patch size. i.e the size of the parameter buffer
            out_chans: Number of out image channels
            embedding_size: Network embedding dimension size
            norm_layer: Optional normalization layer
            bias: Whether to use bias in convolution
            interpolation: Resize interpolation type
            antialias: Whether to apply antialiasing resizing
        """
        super().__init__()

        self.embedding_size = embedding_size

        self.max_patch_size = self.to_2tuple(max_patch_size)

        self.proj = nn.ConvTranspose2d(
            embedding_size,
            out_chans,
            kernel_size=max_patch_size,
            stride=max_patch_size,
            bias=bias,
        )
        self.norm = norm_layer(embedding_size) if norm_layer else nn.Identity()
        self.interpolation = interpolation
        self.antialias = antialias

    @staticmethod
    def to_2tuple(x: Any) -> Any:
        """Convert a scalar or 2-element iterable to a (h, w) tuple."""
        if isinstance(x, Iterable) and not isinstance(x, str):
            assert len(list(x)) == 2, "x must be a 2-tuple"
            return tuple(x)
        return (x, x)

    def _resize(self, x: Tensor, shape: tuple[int, int]) -> Tensor:
        """Resize the input tensor to the target shape.

        Args:
            x: Input tensor
            shape: Target shape

        Returns:
            Resized tensor
        """
        x_resized = F.interpolate(
            x[None, None, ...],
            shape,
            mode=self.interpolation,
            antialias=self.antialias,
        )
        return x_resized[0, 0, ...]

    def forward(
        self,
        x: Tensor,
        patch_size: int | tuple[int, int] | None = None,
    ) -> Tensor | tuple[Tensor, tuple[int, int]]:
        """Forward pass for the FlexiPatchReconstruction module.

        Args:
            x: Input tensor with shape [b, h, w, (t), d]
            patch_size: Patch size to use for the reconstruction. If None, the base patch size
                will be used.
        """
        # x has input shape [b, h, w, (t), d]
        if len(x.shape) == 4:
            has_time_dimension = False
            b, h, w, d = x.shape
            t = 1
        else:
            has_time_dimension = True
            b, h, w, t, d = x.shape

        if not patch_size:
            # During evaluation use base patch size if not specified
            patch_size = self.max_patch_size

        patch_size = self.to_2tuple(patch_size)

        if has_time_dimension:
            x = rearrange(x, "b h w t d -> (b t) d h w", b=b, t=t)
        else:
            x = rearrange(x, "b h w d -> b d h w")

        x = self.proj(x)

        if patch_size != self.max_patch_size:
            x = rearrange(
                x,
                "b c (h p_h) (w p_w) -> b h w c p_h p_w",
                p_h=self.max_patch_size[0],
                p_w=self.max_patch_size[1],
            )
            bl, hl, wl, cl = x.shape[:4]
            x = rearrange(x, "b h w c p_h p_w -> (b h w) c p_h p_w")
            x = F.interpolate(
                x, patch_size, mode=self.interpolation, antialias=self.antialias
            )
            x = rearrange(
                x, "(b h w) c p_h p_w -> b c (h p_h) (w p_w)", b=bl, h=hl, w=wl
            )

        if has_time_dimension:
            x = rearrange(x, "(b t) c h w -> b h w t c", b=b, t=t)
        else:
            x = rearrange(x, "b c h w -> b h w c")

        x = self.norm(x)

        return x
