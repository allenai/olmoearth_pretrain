"""Eval metrics."""

import torch


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
