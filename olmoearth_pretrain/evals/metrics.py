"""Eval metrics."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

import torch


class EvalMetric(StrEnum):
    """Available eval metrics."""

    ACCURACY = "accuracy"
    F1 = "f1"
    CLASS_F1 = "class_f1"
    MICRO_F1 = "micro_f1"
    MIOU = "miou"
    OVERALL_ACC = "overall_acc"
    MACRO_ACC = "macro_acc"
    MACRO_F1 = "macro_f1"


@dataclass
class EvalTaskResult:
    """Result from an evaluation task (knn, linear probe, finetune)."""

    val_result: EvalResult | None
    test_result: EvalResult | None
    bootstrap_stats: dict[str, Any] = field(default_factory=dict)
    eval_time: float | None = None


@dataclass
class EvalResult:
    """Result from evaluation - handles both classification and segmentation."""

    # Primary metric value (used for model selection, backward compat logging)
    primary: float

    # Which metric enum is primary
    primary_metric: EvalMetric

    # The exact key in `metrics` for the primary metric
    primary_metric_key: str

    # All metrics as dict (superset including primary)
    metrics: dict[str, float]

    @staticmethod
    def _resolve_metric_key(metric: EvalMetric, class_idx: int | None = None) -> str:
        """Resolve an EvalMetric enum to the actual metrics dict key."""
        if metric == EvalMetric.CLASS_F1:
            if class_idx is None:
                raise ValueError("class_idx is required when metric is CLASS_F1")
            return f"f1_class_{class_idx}"
        return metric.value

    def with_primary_metric(
        self, metric: EvalMetric, class_idx: int | None = None
    ) -> EvalResult:
        """Return a copy with a different primary metric selected.

        For CLASS_F1, class_idx specifies which class's F1 to use.
        """
        key = self._resolve_metric_key(metric, class_idx)
        if key not in self.metrics:
            raise ValueError(
                f"primary_metric '{key}' not found in metrics: {list(self.metrics.keys())}"
            )
        return EvalResult(
            primary=self.metrics[key],
            primary_metric=metric,
            primary_metric_key=key,
            metrics=self.metrics,
        )

    @classmethod
    def from_classification(
        cls,
        accuracy: float,
        f1: float | None = None,
        macro_f1: float | None = None,
        per_class_f1: list[float] | None = None,
    ) -> EvalResult:
        """Create EvalResult from classification metrics.

        Args:
            accuracy: Classification accuracy (exact match for multilabel)
            f1: Optional F1 score (micro-averaged, typically for multilabel tasks)
            macro_f1: Optional macro-averaged F1 score
            per_class_f1: Optional per-class F1 scores (index = class index)
        """
        metrics: dict[str, float] = {EvalMetric.ACCURACY.value: accuracy}
        if f1 is not None:
            metrics[EvalMetric.F1.value] = f1
        if macro_f1 is not None:
            metrics[EvalMetric.MACRO_F1.value] = macro_f1
        if per_class_f1 is not None:
            for i, score in enumerate(per_class_f1):
                metrics[f"f1_class_{i}"] = score
        return cls(
            primary=accuracy,
            primary_metric=EvalMetric.ACCURACY,
            primary_metric_key=EvalMetric.ACCURACY.value,
            metrics=metrics,
        )

    @classmethod
    def from_segmentation(
        cls,
        miou: float,
        overall_acc: float,
        macro_acc: float,
        macro_f1: float,
        micro_f1: float,
    ) -> EvalResult:
        """Create EvalResult from segmentation metrics."""
        return cls(
            primary=miou,
            primary_metric=EvalMetric.MIOU,
            primary_metric_key=EvalMetric.MIOU.value,
            metrics={
                EvalMetric.MIOU.value: miou,
                EvalMetric.OVERALL_ACC.value: overall_acc,
                EvalMetric.MACRO_ACC.value: macro_acc,
                EvalMetric.MACRO_F1.value: macro_f1,
                EvalMetric.MICRO_F1.value: micro_f1,
            },
        )


def _build_confusion_matrix(
    predictions: torch.Tensor,
    labels: torch.Tensor,
    num_classes: int,
    ignore_label: int = -1,
) -> torch.Tensor:
    """Build confusion matrix from predictions and labels.

    Args:
        predictions: Predicted segmentation masks of shape (N, H, W), integer class indices
        labels: Ground truth segmentation masks of shape (N, H, W), integer class indices
        num_classes: Number of classes in the segmentation task
        ignore_label: Label value to ignore (default: -1)

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
    ignore_label: int = -1,
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

    # Micro F1: global TP / (TP + 0.5*(FP+FN))
    tp_sum = tp.sum()
    micro_f1 = (2 * tp_sum / (2 * tp_sum + fp.sum() + fn.sum() + 1e-8)).item()

    return EvalResult.from_segmentation(
        miou=miou,
        overall_acc=overall_acc,
        macro_acc=macro_acc,
        macro_f1=macro_f1,
        micro_f1=micro_f1,
    )
