"""Multi-layer decoder heads for finetuning.

These mirror the decoder architectures used in rslearn (PoolingDecoder and UNetDecoder)
to provide more capacity than the default single-linear-layer head.
"""

from __future__ import annotations

from enum import StrEnum

import torch
import torch.nn as nn


class HeadType(StrEnum):
    """Type of head to use for finetuning."""

    LINEAR = "linear"
    MULTI_LAYER = "multi_layer"


class MultiLayerClassificationHead(nn.Module):
    """Multi-layer classification head with spatial conv before pooling.

    Mirrors rslearn's PoolingDecoder(num_conv_layers=1, num_fc_layers=1):
    3x3 conv on spatial features -> global max pool -> FC + ReLU -> output.

    Expects spatial (B, H, W, D) input (not pre-pooled).
    """

    def __init__(
        self,
        emb_dim: int,
        num_classes: int,
        conv_channels: int = 128,
        fc_channels: int = 512,
    ) -> None:
        """Initialize multi-layer classification head."""
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(emb_dim, conv_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.fc = nn.Sequential(
            nn.Linear(conv_channels, fc_channels),
            nn.ReLU(inplace=True),
        )
        self.output_layer = nn.Linear(fc_channels, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass. Input (B, H, W, D) -> output (B, num_classes)."""
        x = x.permute(0, 3, 1, 2)  # (B, D, H, W)
        x = self.conv(x)
        x = torch.amax(x, dim=(2, 3))  # global max pool -> (B, C)
        x = self.fc(x)
        return self.output_layer(x)


class MultiLayerSegmentationHead(nn.Module):
    """Multi-layer segmentation head with upsampling.

    Mirrors rslearn's UNetDecoder on single-scale features at patch_size=4:
    conv+ReLU blocks interleaved with 2x upsamples, progressively reducing
    channel width until pixel resolution is reached.
    """

    def __init__(
        self,
        emb_dim: int,
        num_classes: int,
        channels: tuple[int, ...] = (512, 256, 128),
    ) -> None:
        """Initialize multi-layer segmentation head."""
        super().__init__()
        layers: list[nn.Module] = []
        prev_ch = emb_dim
        for i, ch in enumerate(channels):
            if i > 0:
                layers.append(nn.Upsample(scale_factor=2))
            layers.extend(
                [
                    nn.Conv2d(prev_ch, ch, kernel_size=3, padding=1),
                    nn.ReLU(inplace=True),
                ]
            )
            prev_ch = ch
        layers.append(nn.Conv2d(prev_ch, num_classes, kernel_size=3, padding=1))
        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass. Input (B, H, W, D) -> output (B, C, H_full, W_full)."""
        x = x.permute(0, 3, 1, 2)
        return self.layers(x)
