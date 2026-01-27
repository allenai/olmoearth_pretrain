"""Flexible patch embedding Module.

Extended from: https://github.com/huggingface/pytorch-image-models/blob/main/timm/layers/patch_embed.py#L24
by https://github.com/bwconrad/flexivit/
"""

import logging
from collections.abc import Iterable, Sequence
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor, vmap

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
        use_pseudoinverse: bool = False,
        patch_size_seq: Sequence[int] = (1, 2, 3, 4, 5, 6, 7, 8),
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
            use_pseudoinverse: If True, use pseudoinverse to resize patch embedding weights
                instead of resizing the input image. This preserves more information.
            patch_size_seq: Sequence of patch sizes to precompute pseudoinverse matrices for.
                Only used when use_pseudoinverse=True.
        """
        super().__init__()

        self.embedding_size = embedding_size

        self.modality_spec = modality_spec
        self.patch_size = self.to_2tuple(
            patch_size_at_16 * modality_spec.image_tile_size_factor
        )

        self.proj = nn.Conv2d(
            in_chans,
            embedding_size,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            bias=bias,
        )
        self.norm = norm_layer(embedding_size) if norm_layer else nn.Identity()

        # Flexi specific attributes
        self.interpolation = interpolation
        self.antialias = antialias
        self.use_pseudoinverse = use_pseudoinverse
        self.patch_size_seq = patch_size_seq

        # Pre-calculate pinvs for pseudoinverse resizing
        if self.use_pseudoinverse:
            self.pinvs: dict[tuple[int, int], Tensor] = self._cache_pinvs()
        else:
            self.pinvs = {}

    def _cache_pinvs(self) -> dict[tuple[int, int], Tensor]:
        """Pre-calculate all pinv matrices for patch sizes in patch_size_seq."""
        pinvs = {}
        for ps in self.patch_size_seq:
            # Account for modality's image_tile_size_factor
            actual_ps = ps * self.modality_spec.image_tile_size_factor
            tuple_ps = self.to_2tuple(actual_ps)
            if tuple_ps != self.patch_size:
                pinvs[tuple_ps] = self._calculate_pinv(self.patch_size, tuple_ps)
        return pinvs

    def _resize_for_pinv(self, x: Tensor, shape: tuple[int, int]) -> Tensor:
        """Resize tensor for pinv calculation."""
        x_resized = F.interpolate(
            x[None, None, ...],
            shape,
            mode=self.interpolation,
            antialias=self.antialias,
        )
        return x_resized[0, 0, ...]

    def _calculate_pinv(
        self, old_shape: tuple[int, int], new_shape: tuple[int, int]
    ) -> Tensor:
        """Calculate pseudoinverse matrix for resizing from old_shape to new_shape.

        Creates a resize matrix by applying interpolation to basis vectors,
        then computes its pseudoinverse.
        """
        mat = []
        for i in range(int(np.prod(old_shape))):
            basis_vec = torch.zeros(old_shape)
            basis_vec[np.unravel_index(i, old_shape)] = 1.0
            mat.append(self._resize_for_pinv(basis_vec, new_shape).reshape(-1))
        resize_matrix = torch.stack(mat)
        return torch.linalg.pinv(resize_matrix)

    def resize_patch_embed(
        self, patch_embed: Tensor, new_patch_size: tuple[int, int]
    ) -> Tensor:
        """Resize patch_embed to target resolution via pseudo-inverse resizing.

        Args:
            patch_embed: Original patch embedding weights [out_ch, in_ch, H, W]
            new_patch_size: Target patch size (H', W')

        Returns:
            Resized patch embedding weights [out_ch, in_ch, H', W']
        """
        # Return original kernel if no resize is necessary
        if self.patch_size == new_patch_size:
            return patch_embed

        # Get or calculate pseudo-inverse of resize matrix
        if new_patch_size not in self.pinvs:
            self.pinvs[new_patch_size] = self._calculate_pinv(
                self.patch_size, new_patch_size
            )
        pinv = self.pinvs[new_patch_size].to(device=patch_embed.device)

        # Store original dtype for casting back after matmul
        orig_dtype = patch_embed.dtype

        def resample_patch_embed(patch_embed: Tensor) -> Tensor:
            h, w = new_patch_size
            # Cast to float32 for numerical stability, then back to original dtype
            resampled_kernel = pinv @ patch_embed.float().reshape(-1)
            return rearrange(resampled_kernel, "(h w) -> h w", h=h, w=w).to(orig_dtype)

        # Use vmap for efficient batched application over out_ch and in_ch dims
        v_resample_patch_embed = vmap(vmap(resample_patch_embed, 0, 0), 1, 1)

        return v_resample_patch_embed(patch_embed)

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
                will be used, at an image_tile_size_factor of 16
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
        else:
            if isinstance(patch_size, tuple):
                patch_size = (
                    patch_size[0] * self.modality_spec.image_tile_size_factor,
                    patch_size[1] * self.modality_spec.image_tile_size_factor,
                )
            else:
                patch_size = patch_size * self.modality_spec.image_tile_size_factor
        patch_size = self.to_2tuple(patch_size)
        assert isinstance(patch_size, tuple) and len(patch_size) == 2, (
            "patch_size must be a 2-tuple"
        )

        # Apply patch embedding
        if patch_size == self.patch_size:
            # Use original weights directly
            x = self.proj(x)
        elif self.use_pseudoinverse:
            # Resize weights using pseudoinverse, then apply conv
            weight = self.resize_patch_embed(self.proj.weight, patch_size)
            x = F.conv2d(x, weight, bias=self.proj.bias, stride=patch_size)
        else:
            # Original approach: resize input image, then apply conv
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
