"""Eval metrics."""

import torch


def _build_confusion_matrix(
    predictions: torch.Tensor,
    labels: torch.Tensor,
    num_classes: int,
    ignore_label: int = -1,
) -> torch.Tensor:
    """Build confusion matrix from predictions and labels.

    Args:
        predictions: Predicted segmentation masks of shape (N, H, W)
        labels: Ground truth segmentation masks of shape (N, H, W)
        num_classes: Number of classes in the segmentation task
        ignore_label: Label value to ignore (default: -1)

    Returns:
        Confusion matrix of shape (num_classes, num_classes)
    """
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
) -> dict[str, float]:
    """Compute all segmentation metrics from predictions and labels.

    Args:
        predictions: Predicted segmentation masks of shape (N, H, W)
        labels: Ground truth segmentation masks of shape (N, H, W)
        num_classes: Number of classes in the segmentation task
        ignore_label: Label value to ignore (default: -1)

    Returns:
        Dictionary containing:
            - miou: Mean Intersection over Union
            - overall_acc: Overall pixel accuracy (correct pixels / total pixels)
            - macro_acc: Mean of per-class accuracies (recall per class)
            - micro_f1: F1 computed from global TP/FP/FN
            - macro_f1: Mean of per-class F1 scores
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

    # Micro F1: global precision and recall
    total_tp = tp.sum()
    total_fp = fp.sum()
    total_fn = fn.sum()
    micro_precision = total_tp / (total_tp + total_fp + 1e-8)
    micro_recall = total_tp / (total_tp + total_fn + 1e-8)
    micro_f1 = (
        2 * micro_precision * micro_recall / (micro_precision + micro_recall + 1e-8)
    ).item()

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

    return {
        "miou": miou,
        "overall_acc": overall_acc,
        "macro_acc": macro_acc,
        "micro_f1": micro_f1,
        "macro_f1": macro_f1,
    }


def mean_iou(
    predictions: torch.Tensor,
    labels: torch.Tensor,
    num_classes: int,
    ignore_label: int = -1,
) -> float:
    """Calculate mean IoU given prediction and label tensors, ignoring pixels with a specific label.

    Args:
    predictions (torch.Tensor): Predicted segmentation masks of shape (N, H, W)
    labels (torch.Tensor): Ground truth segmentation masks of shape (N, H, W)
    num_classes (int): Number of classes in the segmentation task
    ignore_label (int): Label value to ignore in IoU calculation (default: -1)

    Returns:
    float: Mean IoU across all classes
    """
    device = predictions.device
    labels = labels.to(device)

    valid_mask = labels != ignore_label

    predictions_valid = predictions[valid_mask]
    labels_valid = labels[valid_mask]

    n = num_classes
    confusion = torch.bincount(
        n * labels_valid + predictions_valid, minlength=n**2
    ).reshape(n, n)

    # Calculate intersection (diagonal) and union
    intersection = confusion.diagonal()
    union = confusion.sum(dim=1) + confusion.sum(dim=0) - intersection

    # Calculate IoU for each class
    iou = intersection.float() / (union.float() + 1e-8)

    # Calculate mean IoU (excluding classes with zero union)
    valid_classes = union > 0
    mean_iou_value = iou[valid_classes].mean()

    return mean_iou_value.item()
