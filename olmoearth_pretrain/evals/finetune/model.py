"""Model components for finetuning."""

from __future__ import annotations

from typing import cast

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from olmoearth_pretrain.evals.datasets.configs import TaskType
from olmoearth_pretrain.evals.eval_wrapper import get_eval_wrapper
from olmoearth_pretrain.evals.finetune.heads import (
    HeadType,
    MultiLayerClassificationHead,
    MultiLayerSegmentationHead,
)
from olmoearth_pretrain.train.masking import MaskedOlmoEarthSample


class BackboneWithHead(nn.Module):
    """Backbone model with a classification or segmentation head."""

    def __init__(
        self,
        model: nn.Module,
        task_type: TaskType,
        patch_size: int,
        pooling_type: str,
        num_classes: int,
        use_pooled_tokens: bool = False,
        head_type: HeadType = HeadType.LINEAR,
    ) -> None:
        """Initialize the backbone with head."""
        super().__init__()
        self.backbone = model
        self.wrapper = get_eval_wrapper(
            model,
            task_type=task_type,
            patch_size=patch_size,
            pooling_type=pooling_type,
            concat_features=False,
            use_pooled_tokens=use_pooled_tokens,
        )
        self.task_type = task_type
        self.patch_size = patch_size
        self.num_classes = num_classes
        self.head_type = head_type

        # Multi-layer heads need spatial features even for classification,
        # since they apply convolutions before pooling.
        if head_type == HeadType.MULTI_LAYER:
            self.wrapper.spatial_pool = True

        # placeholder head; real in_dim discovered on first forward
        self._head: nn.Module = nn.Linear(1, 1, bias=True)
        self._inited = False

    def _init_head(self, emb_dim: int, device: torch.device) -> None:
        """Initialize the head based on the embedding dimension."""
        if self.head_type == HeadType.MULTI_LAYER:
            if self.task_type == TaskType.CLASSIFICATION:
                self._head = MultiLayerClassificationHead(emb_dim, self.num_classes)
            else:
                self._head = MultiLayerSegmentationHead(emb_dim, self.num_classes)
        else:
            if self.task_type == TaskType.CLASSIFICATION:
                self._head = nn.Linear(emb_dim, self.num_classes, bias=True)
            else:
                logits_per_patch = int(
                    self.num_classes * self.patch_size * self.patch_size
                )
                self._head = nn.Linear(emb_dim, logits_per_patch, bias=True)

        self._head = self._head.to(device=device)
        self._inited = True

    def forward(
        self, batch: MaskedOlmoEarthSample, labels: torch.Tensor, is_train: bool = True
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through the model and head.

        Returns:
            (logits, labels) where logits are:
              - classification: (B, num_classes)
              - segmentation: (B, C, H, W) at label resolution
        """
        dev = next(self.wrapper.parameters()).device
        emb, labels = self.wrapper(batch, labels, is_train=is_train)
        emb = cast(torch.Tensor, emb)
        emb_dim = emb.shape[-1]
        if not self._inited:
            self._init_head(emb_dim, dev)
        if emb.device != dev:
            emb = emb.to(dev, non_blocking=True)

        logits = self._head(emb)

        if self.task_type == TaskType.SEGMENTATION:
            if self.head_type == HeadType.LINEAR:
                # Linear head outputs (B, H, W, C*p*p) -> rearrange to (B, C, H_full, W_full)
                H, W = logits.shape[1], logits.shape[2]
                logits = rearrange(
                    logits,
                    "b h w (c i j) -> b c (h i) (w j)",
                    h=H,
                    w=W,
                    c=self.num_classes,
                    i=self.patch_size,
                    j=self.patch_size,
                )
            # Multi-layer head already outputs (B, C, H_full, W_full)

            if logits.shape[-2:] != labels.shape[-2:]:
                logits = F.interpolate(
                    logits.float(),
                    size=labels.shape[-2:],
                    mode="bilinear",
                    align_corners=True,
                )

        return logits, labels


def to_device(
    masked: MaskedOlmoEarthSample, device: torch.device
) -> MaskedOlmoEarthSample:
    """Move a MaskedOlmoEarthSample to a device with appropriate dtypes."""
    d = masked.as_dict()
    for k, v in d.items():
        if k == "timestamps":
            d[k] = v.to(device=device)
        else:
            d[k] = v.to(device=device, dtype=torch.bfloat16)
    return MaskedOlmoEarthSample.from_dict(d)


def snapshot_state_dict(module: nn.Module) -> dict[str, torch.Tensor]:
    """Clone a module's state dict onto CPU for later restoration."""
    return {k: v.detach().cpu().clone() for k, v in module.state_dict().items()}


def set_backbone_trainable(backbone: nn.Module, requires_grad: bool) -> None:
    """Toggle gradient computation for backbone parameters."""
    for param in backbone.parameters():
        param.requires_grad = requires_grad
