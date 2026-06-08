"""FPN / multi-scale decoder head for segmentation/regression finetuning.

The OlmoEarth ViT backbone returns a single feature map at one resolution, so the
plain UNet head (``unet_head.py``) has no multi-scale skip connections. Following
ViTDet's "simple feature pyramid", this head synthesizes a multi-scale pyramid
from that single feature map (strided convs to downsample, transposed convs to
upsample), runs a top-down FPN fusion across scales, then decodes to per-pixel
logits. This gives the decoder genuine multi-scale context without touching the
encoder or eval wrapper.

GroupNorm is used (not BatchNorm) so it is stable at the small finetune batch
sizes (1-2) used here.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def _gn(channels: int, max_groups: int = 32) -> nn.GroupNorm:
    groups = math.gcd(channels, max_groups)
    return nn.GroupNorm(max(1, groups), channels)


class _ConvNormAct(nn.Module):
    def __init__(
        self, in_ch: int, out_ch: int, stride: int = 1, transpose: bool = False
    ):
        super().__init__()
        if transpose:
            conv: nn.Module = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
        else:
            conv = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1)
        self.block = nn.Sequential(conv, _gn(out_ch), nn.ReLU(inplace=True))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class FPNDecoder(nn.Module):
    """Simple-feature-pyramid + FPN decoder on top of ViT patch tokens.

    Accepts spatial patch embeddings (B, H_p, W_p, D) and produces per-pixel
    logits (B, num_classes, H, W) where H = H_p * patch_size. patch_size must be
    a power of two.
    """

    def __init__(
        self,
        in_dim: int,
        num_classes: int,
        patch_size: int,
        fpn_dim: int = 256,
        pyramid_levels: tuple[int, ...] = (1, 0, -1, -2),
    ) -> None:
        """Initialize FPNDecoder.

        Args:
            in_dim: ViT embedding dimension.
            num_classes: output channels (num_classes=1 -> dense regression).
            patch_size: token spatial patch size (power of two).
            fpn_dim: common channel width for the pyramid / FPN.
            pyramid_levels: scales relative to the token grid, as powers of two.
                e.g. 1 -> 2x upsample, 0 -> same, -1 -> 2x downsample. Ordered
                fine-to-coarse.
        """
        if patch_size < 1 or (patch_size & (patch_size - 1)) != 0:
            raise ValueError(f"patch_size must be a power of two, got {patch_size}")
        super().__init__()
        self.levels = list(pyramid_levels)

        # Build each pyramid level from the single token feature map.
        self.stems = nn.ModuleList()
        for lvl in self.levels:
            layers: list[nn.Module] = [nn.Conv2d(in_dim, fpn_dim, kernel_size=1)]
            if lvl > 0:  # upsample by 2**lvl via transposed convs
                for _ in range(lvl):
                    layers.append(_ConvNormAct(fpn_dim, fpn_dim, transpose=True))
            elif lvl < 0:  # downsample by 2**(-lvl) via strided convs
                for _ in range(-lvl):
                    layers.append(_ConvNormAct(fpn_dim, fpn_dim, stride=2))
            self.stems.append(nn.Sequential(*layers))

        # FPN smoothing convs (one per level, applied after top-down merge).
        self.smooth = nn.ModuleList(_ConvNormAct(fpn_dim, fpn_dim) for _ in self.levels)

        # Decode the finest fused level up to full (patch_size) resolution.
        n_up = int(math.log2(patch_size)) - max(self.levels)  # remaining 2x stages
        dec: list[nn.Module] = []
        for _ in range(max(0, n_up)):
            dec.append(_ConvNormAct(fpn_dim, fpn_dim, transpose=True))
        self.decode = nn.Sequential(*dec)
        self.head = nn.Conv2d(fpn_dim, num_classes, kernel_size=1)
        self.patch_size = patch_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward from patch tokens (B, H_p, W_p, D) to per-pixel logits (B, C, H, W)."""
        x = x.permute(0, 3, 1, 2).contiguous()  # (B, D, H_p, W_p)
        feats = [stem(x) for stem in self.stems]  # fine -> coarse
        # top-down FPN fusion: add each coarser (upsampled) map into the finer one
        for i in range(len(feats) - 2, -1, -1):
            up = F.interpolate(
                feats[i + 1],
                size=feats[i].shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
            feats[i] = self.smooth[i](feats[i] + up)
        out = self.decode(feats[0])  # upsample finest level to full resolution
        # guard against off-by-one from odd grids: match exact target size
        target = (x.shape[-2] * self.patch_size, x.shape[-1] * self.patch_size)
        if out.shape[-2:] != target:
            out = F.interpolate(out, size=target, mode="bilinear", align_corners=False)
        return self.head(out)
