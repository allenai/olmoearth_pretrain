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

logger = logging.getLogger(__name__)


class FlexiPatchEmbed(nn.Module):
    """Flexible patch embedding nn.Module."""

    def __init__(
        self,
        patch_size: int | tuple[int, int],
        in_chans: int = 3,
        embedding_size: int = 128,
        norm_layer: nn.Module | None = None,
        bias: bool = True,
        interpolation: str = "bicubic",
        antialias: bool = True,
    ) -> None:
        """2D image to patch embedding w/ flexible patch sizes.

        Extended from: https://github.com/huggingface/pytorch-image-models/blob/main/timm/layers/patch_embed.py#L24
        by https://github.com/bwconrad/flexivit/

        Args:
            patch_size: Base patch size. i.e the size of the parameter buffer
            in_chans: Number of input image channels
            embedding_size: Network embedding dimension size
            norm_layer: Optional normalization layer
            bias: Whether to use bias in convolution
            interpolation: Resize interpolation type
            antialias: Whether to apply antialiasing resizing (TODO: Add a link or more info)
        """
        super().__init__()

        self.embedding_size = embedding_size

        self.patch_size = self.to_2tuple(patch_size)

        self.proj = nn.Conv2d(
            in_chans,
            embedding_size,
            kernel_size=patch_size,
            stride=patch_size,
            bias=bias,
        )
        self.norm = norm_layer(embedding_size) if norm_layer else nn.Identity()

        # Flexi specific attributes
        self.interpolation = interpolation
        self.antialias = antialias

    @staticmethod
    def to_2tuple(x: Any) -> Any:
        """Convert a value to a 2-tuple by either converting an iterable or repeating a scalar.

        This is used to handle patch sizes that can be specified either as:
        - A single integer (e.g. 16) which gets converted to (16, 16) for square patches
        - A tuple/list of 2 integers (e.g. (16, 32)) for rectangular patches

        Args:
            x: Value to convert to a 2-tuple. Can be an iterable (list/tuple) of 2 elements,
               or a single value to repeat twice.

        Returns:
            A 2-tuple containing either the original iterable values or the input repeated twice.
        """
        if isinstance(x, Iterable) and not isinstance(x, str):
            assert len(list(x)) == 2, "x must be a 2-tuple"
            return tuple(x)
        return (x, x)

    def forward(
        self,
        x: Tensor,
        patch_size: int | tuple[int, int] | None = None,
    ) -> Tensor | tuple[Tensor, tuple[int, int]]:
        """Forward pass for the FlexiPatchEmbed module.

        Args:
            x: Input tensor with shape [b, h, w, (t), c]
            patch_size: Patch size to use for the embedding. If None, the base patch size
                will be used.
        """
        # x has input shape [b, h, w, (t), c]
        batch_size = x.shape[0]
        has_time_dimension = False
        num_timesteps = 0  # ignored if has_time_dimension is False

        if len(x.shape) == 5:
            has_time_dimension = True
            num_timesteps = x.shape[3]
            x = rearrange(x, "b h w t c -> (b t) c h w")
        else:
            x = rearrange(x, "b h w c -> b c h w")

        if not patch_size:
            # During evaluation use base patch size if not specified
            patch_size = self.patch_size

        patch_size = self.to_2tuple(patch_size)
        assert (
            isinstance(patch_size, tuple) and len(patch_size) == 2
        ), "patch_size must be a 2-tuple"
        # Resize input
        if patch_size != self.patch_size:
            shape = x.shape[-2:]
            new_shape = (
                shape[0] // patch_size[0] * self.patch_size[0],
                shape[1] // patch_size[1] * self.patch_size[1],
            )
            x = F.interpolate(
                x,
                size=new_shape,
                mode=self.interpolation,
                antialias=self.antialias,
            )
        # Apply conv with resized weights
        x = self.proj(x)
        # At this point x has embedding dim sized channel dimension
        if has_time_dimension:
            _, d, h, w = x.shape
            x = rearrange(
                x,
                "(b t) d h w -> b h w t d",
                b=batch_size,
                t=num_timesteps,
                d=d,
                h=h,
                w=w,
            )
        else:
            x = rearrange(x, "b d h w -> b h w d")

        x = self.norm(x)

        return x


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

        # Flexi specific attributes
        self.interpolation = interpolation
        self.antialias = antialias

    @staticmethod
    def to_2tuple(x: Any) -> Any:
        """Convert a value to a 2-tuple by either converting an iterable or repeating a scalar.

        This is used to handle patch sizes that can be specified either as:
        - A single integer (e.g. 16) which gets converted to (16, 16) for square patches
        - A tuple/list of 2 integers (e.g. (16, 32)) for rectangular patches

        Args:
            x: Value to convert to a 2-tuple. Can be an iterable (list/tuple) of 2 elements,
               or a single value to repeat twice.

        Returns:
            A 2-tuple containing either the original iterable values or the input repeated twice.
        """
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
