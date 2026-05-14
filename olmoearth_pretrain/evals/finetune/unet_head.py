"""UNet-style decoder head for segmentation finetuning."""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class _ConvBnRelu(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UNetDecoder(nn.Module):
    """Progressive upsampling decoder for segmentation on top of ViT patch tokens.

    Accepts spatial patch embeddings (B, H_p, W_p, D) and produces per-pixel
    logits (B, num_classes, H, W) where H = H_p * patch_size.

    The number of upsampling stages is log2(patch_size), so patch_size must be
    a power of two (4, 8, 16, ...).
    """

    def __init__(
        self,
        in_dim: int,
        num_classes: int,
        patch_size: int,
        decoder_dims: tuple[int, ...] = (256, 128, 64),
    ) -> None:
        """Initialize UNetDecoder."""
        if patch_size < 1 or (patch_size & (patch_size - 1)) != 0:
            raise ValueError(f"patch_size must be a power of two, got {patch_size}")
        super().__init__()
        n_stages = int(math.log2(patch_size))

        # Build per-stage channel widths, padding or truncating decoder_dims to fit.
        dims: list[int] = list(decoder_dims)
        while len(dims) < n_stages + 1:
            dims.append(max(dims[-1] // 2, 16))
        dims = dims[: n_stages + 1]

        # 1×1 projection from ViT embedding dim to first decoder width.
        self.input_proj = nn.Conv2d(in_dim, dims[0], kernel_size=1)

        # One block per upsampling stage: bilinear 2× upsample then conv.
        self.up_blocks = nn.ModuleList()
        for i in range(n_stages):
            self.up_blocks.append(
                nn.Sequential(
                    nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
                    _ConvBnRelu(dims[i], dims[i + 1]),
                )
            )

        self.head = nn.Conv2d(dims[n_stages], num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass from patch tokens (B, H_p, W_p, D) to per-pixel logits (B, C, H, W)."""
        x = x.permute(0, 3, 1, 2).contiguous()  # (B, D, H_p, W_p)
        x = self.input_proj(x)
        for block in self.up_blocks:
            x = block(x)
        return self.head(x)  # (B, num_classes, H, W)
