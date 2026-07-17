"""UNet-style decoder head for segmentation/regression finetuning.

This mirrors ``rslearn.models.unet.UNetDecoder`` for the single-scale case. The
OlmoEarth ViT backbone returns a *single* feature map (at 1/patch_size
resolution), so — exactly as in rslearn when its ``in_channels`` has one entry —
there are no multi-resolution skip connections; this is a plain progressive
upsampler, not a true U-Net. To match rslearn's behavior (and its results) we
keep the same choices it uses there:

* full-width (``in_dim``) channels through every stage — no tapering,
* one 3x3 conv + ReLU per upsample stage (``conv_layers_per_resolution``), with
  one leading conv at the lowest resolution and a 3x3 output conv,
* nearest-neighbor upsampling,
* **no normalization** (no BatchNorm).
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class UNetDecoder(nn.Module):
    """Progressive upsampling decoder on top of ViT patch tokens.

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
        conv_layers_per_resolution: int = 1,
    ) -> None:
        """Initialize UNetDecoder."""
        if patch_size < 1 or (patch_size & (patch_size - 1)) != 0:
            raise ValueError(f"patch_size must be a power of two, got {patch_size}")
        super().__init__()
        n_stages = int(math.log2(patch_size))

        def conv() -> list[nn.Module]:
            # 3x3 conv (padding=same for stride 1) + ReLU, keeping channels at in_dim.
            return [
                nn.Conv2d(in_dim, in_dim, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
            ]

        # Leading conv at the lowest (1/patch_size) resolution.
        layers: list[nn.Module] = conv()
        # One nearest-upsample + conv(s) per stage, mirroring rslearn.
        for _ in range(n_stages):
            layers.append(nn.Upsample(scale_factor=2, mode="nearest"))
            for _ in range(conv_layers_per_resolution):
                layers.extend(conv())
        # 3x3 output projection to per-pixel logits.
        layers.append(nn.Conv2d(in_dim, num_classes, kernel_size=3, padding=1))
        self.decoder = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass from patch tokens (B, H_p, W_p, D) to per-pixel logits (B, C, H, W)."""
        x = x.permute(0, 3, 1, 2).contiguous()  # (B, D, H_p, W_p)
        return self.decoder(x)
