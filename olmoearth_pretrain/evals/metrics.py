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
    # Ensure inputs are on the same device
    device = predictions.device
    labels = labels.to(device)

    # Initialize tensors to store intersection and union for each class
    intersection = torch.zeros(num_classes, device=device)
    union = torch.zeros(num_classes, device=device)

    # Create a mask for valid pixels (i.e., not ignore_label)
    valid_mask = labels != ignore_label

    # Iterate through each class
    for class_id in range(num_classes):
        # Create binary masks for the current class
        pred_mask = (predictions == class_id) & valid_mask
        label_mask = (labels == class_id) & valid_mask

        # Calculate intersection and union
        intersection[class_id] = (pred_mask & label_mask).sum().float()
        union[class_id] = (pred_mask | label_mask).sum().float()

    # Calculate IoU for each class
    iou = intersection / (union + 1e-8)  # Add small epsilon to avoid division by zero

    # Calculate mean IoU (excluding classes with zero union)
    valid_classes = union > 0
    mean_iou = iou[valid_classes].mean()

    return mean_iou.item()
