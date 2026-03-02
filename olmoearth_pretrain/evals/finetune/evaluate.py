"""Evaluation functions for finetuning."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from einops import rearrange
from torch.utils.data import DataLoader

from olmoearth_pretrain.evals.finetune.model import BackboneWithHead, to_device
from olmoearth_pretrain.evals.metrics import (
    EvalMetric,
    EvalResult,
    classification_metrics,
    segmentation_metrics,
)


@torch.no_grad()
def eval_cls(
    module: BackboneWithHead,
    loader: DataLoader,
    device: torch.device,
    is_multilabel: bool,
    primary_metric: EvalMetric | None = None,
    primary_metric_class: int | None = None,
) -> EvalResult:
    """Evaluate classification metrics."""
    module.eval()
    logits_all, labels_all = [], []
    for masked, label in loader:
        label = label.to(device=device)
        masked = to_device(masked, device)
        with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16):
            logits, _ = module(masked, label, is_train=False)  # (B, C)
        logits_all.append(logits.float().cpu())
        labels_all.append(label.cpu())
    logits = torch.cat(logits_all, 0)
    labels = torch.cat(labels_all, 0)
    if is_multilabel:
        preds = torch.sigmoid(logits).gt(0.5).int()
    else:
        preds = torch.argmax(logits, dim=-1)
    return classification_metrics(
        preds,
        labels,
        is_multilabel=is_multilabel,
        primary_metric=primary_metric,
        primary_metric_class=primary_metric_class,
    )


@torch.no_grad()
def eval_seg(
    module: BackboneWithHead,
    loader: DataLoader,
    device: torch.device,
    num_classes: int,
    patch_size: int,
    primary_metric: EvalMetric | None = None,
    primary_metric_class: int | None = None,
) -> EvalResult:
    """Evaluate segmentation metrics."""
    module.eval()
    preds_all, labels_all = [], []
    for masked, label in loader:
        label = label.to(device=device)
        masked = to_device(masked, device)
        with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16):
            logits, _ = module(masked, label, is_train=False)  # (B, H, W, C*p*p)
            H, W = logits.shape[1], logits.shape[2]
            logits = rearrange(
                logits,
                "b h w (c i j) -> b c (h i) (w j)",
                h=H,
                w=W,
                c=num_classes,
                i=patch_size,
                j=patch_size,
            )
            if logits.shape[-2:] != label.shape[-2:]:
                logits = F.interpolate(
                    logits.float(),
                    size=label.shape[-2:],
                    mode="bilinear",
                    align_corners=True,
                )
        preds_all.append(torch.argmax(logits, dim=1).cpu())
        labels_all.append(label.cpu())
    preds = torch.cat(preds_all, 0)
    labels = torch.cat(labels_all, 0)
    return segmentation_metrics(
        preds,
        labels,
        num_classes=num_classes,
        ignore_label=-1,
        primary_metric=primary_metric,
        primary_metric_class=primary_metric_class,
    )
