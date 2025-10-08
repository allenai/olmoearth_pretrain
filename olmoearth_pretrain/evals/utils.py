"""Utility functions for the evals."""

import math

import torch


def adjust_learning_rate(
    optimizer: torch.optim.Optimizer,
    epoch: float,
    warmup_epochs: int,
    total_epochs: int,
    max_lr: float,
    min_lr: float,
) -> float:
    """Decay the learning rate with half-cycle cosine after warmup."""
    if epoch < warmup_epochs:
        lr = max_lr * epoch / warmup_epochs
    else:
        lr = min_lr + (max_lr - min_lr) * 0.5 * (
            1.0
            + math.cos(
                math.pi * (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
            )
        )
    for group in optimizer.param_groups:
        group["lr"] = lr
    return lr
