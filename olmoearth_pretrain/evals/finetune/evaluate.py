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
    regression_metrics,
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
def eval_reg(
    module: BackboneWithHead,
    loader: DataLoader,
    device: torch.device,
    primary_metric: EvalMetric | None = None,
) -> EvalResult:
    """Evaluate regression metrics (scalar targets per sample)."""
    module.eval()
    preds_all, labels_all = [], []
    for masked, label in loader:
        label = label.to(device=device)
        masked = to_device(masked, device)
        with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16):
            logits, _ = module(masked, label, is_train=False)
            preds = logits.squeeze(-1).float()
        preds_all.append(preds.cpu())
        labels_all.append(label.float().cpu())
    preds = torch.cat(preds_all, 0)
    labels = torch.cat(labels_all, 0)
    return regression_metrics(preds, labels, primary_metric=primary_metric)


def _seg_logits_to_pixel(
    logits: torch.Tensor,
    label: torch.Tensor,
    pixel_space_output: bool,
    num_classes: int,
    patch_size: int,
) -> torch.Tensor:
    """Pixel-shuffle patch-space logits and resize to label resolution."""
    if not pixel_space_output:
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
    return logits


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
            logits, _ = module(masked, label, is_train=False)
            logits = _seg_logits_to_pixel(
                logits, label, module.pixel_space_output, num_classes, patch_size
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
