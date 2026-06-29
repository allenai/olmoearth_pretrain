"""Checkpoint save/load utilities for finetuning."""

from __future__ import annotations

import os
import shutil
import tempfile
from logging import getLogger

import torch

logger = getLogger(__name__)


def save_training_checkpoint(
    path: str,
    epoch: int,
    model_state: dict[str, torch.Tensor],
    optimizer_state: dict,
    scheduler_state: dict,
    best_state: dict[str, torch.Tensor],
    best_val_metric: float,
    backbone_unfrozen: bool,
) -> None:
    """Save a resumable training checkpoint atomically."""
    dir_path = os.path.dirname(path)
    if dir_path:
        os.makedirs(dir_path, exist_ok=True)

    checkpoint = {
        "epoch": epoch,
        "model_state": model_state,
        "optimizer_state": optimizer_state,
        "scheduler_state": scheduler_state,
        "best_state": best_state,
        "best_val_metric": best_val_metric,
        "backbone_unfrozen": backbone_unfrozen,
    }

    # Write to temp file first, then atomic rename
    tmp_fd, tmp_path = tempfile.mkstemp(
        dir=dir_path if dir_path else ".",
    )
    try:
        torch.save(checkpoint, tmp_path)
        os.close(tmp_fd)
        shutil.move(tmp_path, path)  # Atomic on POSIX
        logger.info(f"Saved training checkpoint to {path} at epoch {epoch}")
    except Exception:
        os.close(tmp_fd)
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise


def load_training_checkpoint(path: str, device: torch.device) -> dict:
    """Load a training checkpoint."""
    try:
        ckpt = torch.load(path, map_location=device, weights_only=False)
    except Exception as e:
        raise RuntimeError(f"Failed to load checkpoint {path}: {e}") from e
    required = [
        "epoch",
        "model_state",
        "optimizer_state",
        "scheduler_state",
        "best_state",
        "best_val_metric",
        "backbone_unfrozen",
    ]
    missing = [k for k in required if k not in ckpt]
    if missing:
        raise ValueError(f"Checkpoint {path} missing keys: {missing}")
    logger.info(f"Loaded training checkpoint from {path}")
    return ckpt
