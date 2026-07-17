"""Eval metrics."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    roc_auc_score,
)


class EvalMetric(StrEnum):
    """Available eval metrics."""

    ACCURACY = "accuracy"
    F1 = "f1"
    CLASS_F1 = "class_f1"
    MICRO_F1 = "micro_f1"
    # Macro one-vs-rest multiclass AUROC (mean of per-class ROC AUC, skipping
    # classes that are all-positive/all-negative). Threshold-free and robust to
    # class imbalance, computed on softmax/sigmoid scores.
    AUROC = "auroc"
    # Macro one-vs-rest area under the precision-recall curve (mean of per-class
    # average precision, skipping classes that are all-positive/all-negative).
    # Threshold-free and, unlike AUROC, sensitive to class imbalance; computed
    # on softmax/sigmoid scores.
    PRAUC = "prauc"
    # Micro-averaged mean Average Precision (threshold-free, ranking-based).
    # This is GeoBench-2 / torchgeo-bench's reported metric for multilabel
    # classification (e.g. TreeSatAI, BigEarthNet), computed on sigmoid scores.
    MICRO_MAP = "micro_map"
    MIOU = "miou"
    OVERALL_ACC = "overall_acc"
    MACRO_ACC = "macro_acc"
    MACRO_F1 = "macro_f1"
    MAE = "mae"
    RMSE = "rmse"
    NEG_RMSE = "neg_rmse"
    R2 = "r2"


# Error metrics where a smaller value is better. Every other metric (accuracy,
# F1, mIoU, R2, neg_rmse, ...) is higher-is-better.
LOWER_IS_BETTER_METRICS = frozenset({EvalMetric.MAE, EvalMetric.RMSE})


def metric_higher_is_better(metric: EvalMetric) -> bool:
    """Whether a larger value of ``metric`` indicates a better model."""
    return metric not in LOWER_IS_BETTER_METRICS


# Label value used to mark invalid/ignored pixels in segmentation targets.
# Pixels with this label are excluded from loss and metric calculations.
SEGMENTATION_IGNORE_LABEL = -1


@dataclass
class EvalTaskResult:
    """Container for one task's outputs: validation/test results, optional bootstrap stats, and eval runtime."""

    val_result: EvalResult | None
    test_result: EvalResult | None
    bootstrap_stats: dict[str, Any] = field(default_factory=dict)
    eval_time: float | None = None
    embedding_diagnostics: dict[str, float] = field(default_factory=dict)
    pca_rgb: np.ndarray | None = None


@dataclass
class EvalResult:
    """Result from evaluation - handles both classification and segmentation."""

    # Primary metric value (used for model selection, backward compat logging)
    primary: float

    # Which metric enum is primary
    primary_metric: EvalMetric

    # The exact key in `metrics` for the primary metric, ie if primary_metric is EvalMetric.CLASS_F1, primary_metric_key may be "f1_class_0"
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
        micro_map: float | None = None,
        auroc: float | None = None,
        prauc: float | None = None,
        is_multilabel: bool = False,
        primary_metric: EvalMetric | None = None,
        primary_metric_class: int | None = None,
    ) -> EvalResult:
        """Create EvalResult from classification metrics.

        Primary metric defaults to F1 for multilabel, ACCURACY for single-label.
        """
        metrics: dict[str, float] = {EvalMetric.ACCURACY.value: accuracy}
        if f1 is not None:
            metrics[EvalMetric.F1.value] = f1
        if macro_f1 is not None:
            metrics[EvalMetric.MACRO_F1.value] = macro_f1
        if micro_map is not None:
            metrics[EvalMetric.MICRO_MAP.value] = micro_map
        if auroc is not None:
            metrics[EvalMetric.AUROC.value] = auroc
        if prauc is not None:
            metrics[EvalMetric.PRAUC.value] = prauc
        if per_class_f1 is not None:
            for i, score in enumerate(per_class_f1):
                metrics[f"f1_class_{i}"] = score

        if primary_metric is None:
            primary_metric = EvalMetric.F1 if is_multilabel else EvalMetric.ACCURACY
        resolved_key = cls._resolve_metric_key(primary_metric, primary_metric_class)
        if resolved_key not in metrics:
            raise ValueError(
                f"primary_metric '{resolved_key}' not found in computed metrics: "
                f"{list(metrics.keys())}"
            )
        return cls(
            primary=metrics[resolved_key],
            primary_metric=primary_metric,
            primary_metric_key=resolved_key,
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
        per_class_f1: list[float] | None = None,
        auroc: float | None = None,
        prauc: float | None = None,
        primary_metric: EvalMetric | None = None,
        primary_metric_class: int | None = None,
    ) -> EvalResult:
        """Create EvalResult from segmentation metrics. Primary defaults to MIOU."""
        metrics = {
            EvalMetric.MIOU.value: miou,
            EvalMetric.OVERALL_ACC.value: overall_acc,
            EvalMetric.MACRO_ACC.value: macro_acc,
            EvalMetric.MACRO_F1.value: macro_f1,
            EvalMetric.MICRO_F1.value: micro_f1,
        }
        if auroc is not None:
            metrics[EvalMetric.AUROC.value] = auroc
        if prauc is not None:
            metrics[EvalMetric.PRAUC.value] = prauc
        if per_class_f1 is not None:
            for i, score in enumerate(per_class_f1):
                metrics[f"f1_class_{i}"] = score
        if primary_metric is None:
            primary_metric = EvalMetric.MIOU
        resolved_key = cls._resolve_metric_key(primary_metric, primary_metric_class)
        if resolved_key not in metrics:
            raise ValueError(
                f"primary_metric '{resolved_key}' not found in computed metrics: "
                f"{list(metrics.keys())}"
            )
        return cls(
            primary=metrics[resolved_key],
            primary_metric=primary_metric,
            primary_metric_key=resolved_key,
            metrics=metrics,
        )

    @classmethod
    def from_regression(
        cls,
        mae: float,
        rmse: float,
        r2: float,
        primary_metric: EvalMetric | None = None,
    ) -> EvalResult:
        """Create EvalResult from regression metrics. Primary defaults to RMSE."""
        metrics = {
            EvalMetric.MAE.value: mae,
            EvalMetric.RMSE.value: rmse,
            EvalMetric.NEG_RMSE.value: -rmse,
            EvalMetric.R2.value: r2,
        }
        if primary_metric is None:
            primary_metric = EvalMetric.NEG_RMSE
        resolved_key = cls._resolve_metric_key(primary_metric)
        if resolved_key not in metrics:
            raise ValueError(
                f"primary_metric '{resolved_key}' not found in computed metrics: "
                f"{list(metrics.keys())}"
            )
        return cls(
            primary=metrics[resolved_key],
            primary_metric=primary_metric,
            primary_metric_key=resolved_key,
            metrics=metrics,
        )


def _indices_to_one_hot(labels_idx: np.ndarray, num_classes: int) -> np.ndarray:
    """Convert integer class indices of shape (M,) to a one-hot (M, num_classes)."""
    one_hot = np.zeros((labels_idx.shape[0], num_classes), dtype=int)
    if labels_idx.size:
        one_hot[np.arange(labels_idx.shape[0]), labels_idx] = 1
    return one_hot


def _macro_binary_score(
    scores: np.ndarray,
    binary_labels: np.ndarray,
    score_fn: Callable[[np.ndarray, np.ndarray], float],
) -> float:
    """Macro-average a per-class binary score (e.g. ROC AUC or average precision).

    Args:
        scores: Per-class continuous scores of shape (M, num_classes).
        binary_labels: Multi-hot / one-hot ground-truth of shape
            (M, num_classes). For single-label tasks pass a one-hot encoding.
        score_fn: Binary scoring function ``score_fn(y_true, y_score) -> float``,
            such as ``roc_auc_score`` or ``average_precision_score``.

    Returns:
        Mean of the per-class score, skipping columns that are all-positive or
        all-negative (score undefined). Returns NaN if every column is
        degenerate or if there are no samples.
    """
    if binary_labels.shape[0] == 0:
        return float("nan")
    values = []
    for c in range(binary_labels.shape[1]):
        y_true_c = binary_labels[:, c]
        # The score is undefined when a class has no positives or no negatives.
        if y_true_c.min() == y_true_c.max():
            continue
        values.append(score_fn(y_true_c, scores[:, c]))
    return float(np.mean(values)) if values else float("nan")


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
    scores: torch.Tensor,
    ignore_label: int = SEGMENTATION_IGNORE_LABEL,
    primary_metric: EvalMetric | None = None,
    primary_metric_class: int | None = None,
) -> EvalResult:
    """Compute all segmentation metrics from predictions and labels.

    Args:
        predictions: Predicted segmentation masks of shape (N, H, W), integer class indices
        labels: Ground truth segmentation masks of shape (N, H, W), integer class indices
        num_classes: Number of classes in the segmentation task
        scores: Per-pixel per-class softmax scores of shape (N, C, H, W). Used to
            compute the macro one-vs-rest ``auroc``.
        ignore_label: Label value to ignore (default: -1)
        primary_metric: Override the default primary metric (None = MIOU)
        primary_metric_class: Class index for CLASS_F1 primary metric

    Returns:
        EvalResult with metrics: miou, overall_acc, macro_acc, macro_f1
    """
    confusion = _build_confusion_matrix(predictions, labels, num_classes, ignore_label)

    # Flatten to (M, C) / (M,) over valid pixels, then macro OVR AUROC / PR-AUC.
    scores_np = scores.detach().cpu().float().numpy()
    scores_np = scores_np.transpose(0, 2, 3, 1).reshape(-1, num_classes)
    labels_np = labels.detach().cpu().numpy().reshape(-1)
    valid_mask = labels_np != ignore_label
    valid_scores = scores_np[valid_mask]
    one_hot = _indices_to_one_hot(labels_np[valid_mask], num_classes)
    auroc = _macro_binary_score(valid_scores, one_hot, roc_auc_score)
    prauc = _macro_binary_score(valid_scores, one_hot, average_precision_score)

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
        per_class_f1=per_class_f1.tolist(),
        auroc=auroc,
        prauc=prauc,
        primary_metric=primary_metric,
        primary_metric_class=primary_metric_class,
    )


def classification_metrics(
    predictions: torch.Tensor,
    labels: torch.Tensor,
    scores: torch.Tensor,
    is_multilabel: bool = False,
    primary_metric: EvalMetric | None = None,
    primary_metric_class: int | None = None,
) -> EvalResult:
    """Compute classification metrics from predictions and labels.

    Args:
        predictions: Hard predictions (thresholded multi-hot or argmax indices).
        labels: Ground-truth labels (multi-hot for multilabel, indices otherwise).
        scores: Per-class continuous scores (softmax/sigmoid probabilities),
            shape (N, num_classes). Used to compute the macro one-vs-rest
            ``auroc`` and ``prauc`` (and ``micro_map`` for multilabel tasks).
        is_multilabel: Whether this is a multi-label task (sigmoid/multi-hot)
            rather than single-label (softmax/argmax).
        primary_metric: Override the default primary metric (None = task default).
        primary_metric_class: Class index for the CLASS_F1 primary metric.
    """
    preds_np = predictions.detach().cpu().numpy()
    labels_np = labels.detach().cpu().numpy()
    scores_np = scores.detach().cpu().float().numpy()

    if is_multilabel:
        preds_np = preds_np.astype(int)
        labels_np = labels_np.astype(int)
        accuracy = accuracy_score(labels_np, preds_np)
        micro_f1 = f1_score(labels_np, preds_np, average="micro", zero_division=0)
        macro_f1 = f1_score(labels_np, preds_np, average="macro", zero_division=0)
        per_class_f1 = f1_score(
            labels_np, preds_np, average=None, zero_division=0
        ).tolist()
        micro_map = float(
            average_precision_score(labels_np, scores_np, average="micro")
        )
        auroc = _macro_binary_score(scores_np, labels_np, roc_auc_score)
        prauc = _macro_binary_score(scores_np, labels_np, average_precision_score)
        return EvalResult.from_classification(
            accuracy,
            f1=micro_f1,
            macro_f1=macro_f1,
            per_class_f1=per_class_f1,
            micro_map=micro_map,
            auroc=auroc,
            prauc=prauc,
            is_multilabel=True,
            primary_metric=primary_metric,
            primary_metric_class=primary_metric_class,
        )

    accuracy = accuracy_score(labels_np, preds_np)
    macro_f1 = f1_score(labels_np, preds_np, average="macro", zero_division=0)
    per_class_f1 = f1_score(labels_np, preds_np, average=None, zero_division=0).tolist()
    one_hot = _indices_to_one_hot(labels_np, scores_np.shape[-1])
    auroc = _macro_binary_score(scores_np, one_hot, roc_auc_score)
    prauc = _macro_binary_score(scores_np, one_hot, average_precision_score)
    return EvalResult.from_classification(
        accuracy,
        macro_f1=macro_f1,
        per_class_f1=per_class_f1,
        auroc=auroc,
        prauc=prauc,
        primary_metric=primary_metric,
        primary_metric_class=primary_metric_class,
    )


def regression_metrics(
    predictions: torch.Tensor,
    labels: torch.Tensor,
    primary_metric: EvalMetric | None = None,
) -> EvalResult:
    """Compute regression metrics from continuous predictions and labels."""
    predictions = predictions.float()
    labels = labels.float().to(predictions.device)
    valid_mask = torch.isfinite(labels)
    predictions = predictions[valid_mask]
    labels = labels[valid_mask]
    if labels.numel() == 0:
        raise ValueError("No finite labels available for regression metrics")
    errors = predictions - labels
    mae = errors.abs().mean().item()
    rmse = torch.sqrt(errors.pow(2).mean()).item()
    total = (labels - labels.mean()).pow(2).sum()
    residual = errors.pow(2).sum()
    r2 = (1.0 - residual / (total + 1e-8)).item()
    return EvalResult.from_regression(
        mae=mae,
        rmse=rmse,
        r2=r2,
        primary_metric=primary_metric,
    )
