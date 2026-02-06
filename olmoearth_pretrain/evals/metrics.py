"""Eval metrics."""

from __future__ import annotations

from dataclasses import dataclass

import torch

# Label value used to mark invalid/ignored pixels in segmentation targets.
# Pixels with this label are excluded from loss and metric calculations.
SEGMENTATION_IGNORE_LABEL = -1


@dataclass
class EvalResult:
    """Result from evaluation - handles both classification and segmentation."""

    # Primary metric (used for model selection, backward compat logging)
    primary: float

    # All metrics as dict (superset including primary)
    metrics: dict[str, float]

    @classmethod
    def from_classification(cls, accuracy: float) -> EvalResult:
        """Create EvalResult from classification accuracy."""
        return cls(primary=accuracy, metrics={"accuracy": accuracy})

    @classmethod
    def from_segmentation(
        cls,
        miou: float,
        overall_acc: float,
        macro_acc: float,
        macro_f1: float,
        per_class_f1: dict[int, float] | None = None,
    ) -> EvalResult:
        """Create EvalResult from segmentation metrics."""
        metrics = {
            "miou": miou,
            "overall_acc": overall_acc,
            "macro_acc": macro_acc,
            "macro_f1": macro_f1,
        }
        if per_class_f1:
            for cls_idx, val in per_class_f1.items():
                metrics[f"f1_cls_{cls_idx}"] = val
        return cls(primary=miou, metrics=metrics)


def _build_confusion_matrix(
    predictions: torch.Tensor,
    labels: torch.Tensor,
    num_classes: int,
    ignore_label: int = SEGMENTATION_IGNORE_LABEL,
) -> torch.Tensor:
    """Build confusion matrix from predictions and labels.

    Args:
        predictions: Predicted segmentation masks of shape (N, H, W), integer class indices
        labels: Ground truth segmentation masks of shape (N, H, W), integer class indices
        num_classes: Number of classes in the segmentation task
        ignore_label: Label value to ignore (default: SEGMENTATION_IGNORE_LABEL)

    Returns:
        Confusion matrix of shape (num_classes, num_classes)

    Raises:
        TypeError: If predictions or labels are not integer tensors
    """
    # Validate tensor dtypes
    if predictions.dtype not in (torch.int32, torch.int64, torch.long):
        raise TypeError(
            f"predictions must be integer class indices, got {predictions.dtype}"
        )
    if labels.dtype not in (torch.int32, torch.int64, torch.long):
        raise TypeError(f"labels must be integer class indices, got {labels.dtype}")

    device = predictions.device
    labels = labels.to(device)

    valid_mask = labels != ignore_label
    predictions_valid = predictions[valid_mask]
    labels_valid = labels[valid_mask]

    n = num_classes
    confusion = torch.bincount(
        n * labels_valid + predictions_valid, minlength=n**2
    ).reshape(n, n)

    return confusion


def segmentation_metrics(
    predictions: torch.Tensor,
    labels: torch.Tensor,
    num_classes: int,
    ignore_label: int = SEGMENTATION_IGNORE_LABEL,
) -> EvalResult:
    """Compute all segmentation metrics from predictions and labels.

    Args:
        predictions: Predicted segmentation masks of shape (N, H, W), integer class indices
        labels: Ground truth segmentation masks of shape (N, H, W), integer class indices
        num_classes: Number of classes in the segmentation task
        ignore_label: Label value to ignore (default: -1)

    Returns:
        EvalResult with metrics: miou, overall_acc, macro_acc, macro_f1
    """
    confusion = _build_confusion_matrix(predictions, labels, num_classes, ignore_label)

    # Per-class statistics from confusion matrix
    # confusion[i, j] = number of pixels with true label i predicted as j
    tp = confusion.diagonal().float()  # True positives per class
    fp = confusion.sum(dim=0).float() - tp  # False positives per class
    fn = confusion.sum(dim=1).float() - tp  # False negatives per class

    # IoU per class
    union = tp + fp + fn
    iou = tp / (union + 1e-8)
    valid_classes = union > 0
    miou = iou[valid_classes].mean().item()

    # Overall accuracy: total correct / total pixels
    total_correct = tp.sum()
    total_pixels = confusion.sum()
    overall_acc = (total_correct / (total_pixels + 1e-8)).item()

    # Macro accuracy (mean recall): mean of TP_c / (TP_c + FN_c) per class
    class_totals = tp + fn  # Total pixels per class (ground truth)
    per_class_acc = tp / (class_totals + 1e-8)
    valid_acc_classes = class_totals > 0
    macro_acc = per_class_acc[valid_acc_classes].mean().item()

    # Macro F1: mean of per-class F1 scores
    per_class_precision = tp / (tp + fp + 1e-8)
    per_class_recall = tp / (tp + fn + 1e-8)
    per_class_f1 = (
        2
        * per_class_precision
        * per_class_recall
        / (per_class_precision + per_class_recall + 1e-8)
    )
    # Only average over classes that have ground truth samples
    valid_f1_classes = class_totals > 0
    macro_f1 = per_class_f1[valid_f1_classes].mean().item()

    per_class_f1_dict = {
        i: per_class_f1[i].item()
        for i in range(num_classes)
        if class_totals[i] > 0
    }

    return EvalResult.from_segmentation(
        miou=miou,
        overall_acc=overall_acc,
        macro_acc=macro_acc,
        macro_f1=macro_f1,
        per_class_f1=per_class_f1_dict,
    )
