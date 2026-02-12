"""Evaluation functions for finetuning."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from einops import rearrange
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader

from olmoearth_pretrain.evals.finetune.model import BackboneWithHead, to_device
from olmoearth_pretrain.evals.metrics import EvalResult, segmentation_metrics


@torch.no_grad()
def eval_cls(
    module: BackboneWithHead,
    loader: DataLoader,
    device: torch.device,
    is_multilabel: bool,
) -> EvalResult:
    """Evaluate classification metrics.

    Returns:
        EvalResult with metrics: micro f1 (if multilabel) or accuracy.
    """
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
        labels_np = labels.numpy().astype(int)
        preds_np = preds.numpy()
        acc = accuracy_score(labels_np, preds_np)  # subset/exact match accuracy
        f1 = f1_score(labels_np, preds_np, average="micro", zero_division=0)
        return EvalResult.from_classification(acc, f1=f1)
    else:
        preds = torch.argmax(logits, dim=-1)
        acc = accuracy_score(labels.numpy(), preds.numpy())
        return EvalResult.from_classification(acc)


@torch.no_grad()
def eval_seg(
    module: BackboneWithHead,
    loader: DataLoader,
    device: torch.device,
    num_classes: int,
    patch_size: int,
) -> EvalResult:
    """Evaluate segmentation metrics.

    Returns:
        EvalResult with metrics: miou, overall_acc, macro_acc, macro_f1
    """
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
    return segmentation_metrics(preds, labels, num_classes=num_classes, ignore_label=-1)
