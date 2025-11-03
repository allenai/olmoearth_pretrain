"""Finetune the OlmoEarth Pretrain and other models on a downstream task."""

from __future__ import annotations

import math
import os
import random
from logging import getLogger
from typing import Any, cast

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from olmo_core.train.trainer import Trainer
from sklearn.metrics import accuracy_score, f1_score
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from olmoearth_pretrain.evals.datasets.configs import EvalDatasetConfig, TaskType
from olmoearth_pretrain.evals.eval_wrapper import get_eval_wrapper
from olmoearth_pretrain.evals.metrics import mean_iou
from olmoearth_pretrain.train.callbacks.wandb import OlmoEarthWandBCallback
from olmoearth_pretrain.train.masking import MaskedOlmoEarthSample

logger = getLogger(__name__)


def _get_wandb_logger(trainer: Trainer) -> Any | None:
    """Return the wandb module from the OlmoEarth callback, if available."""
    for callback in trainer._iter_callbacks():
        if isinstance(callback, OlmoEarthWandBCallback) and callback.enabled:
            return callback.wandb
    return None


class BackboneWithHead(nn.Module):
    """Backbone model with a classification or segmentation head."""

    def __init__(
        self,
        model: nn.Module,
        task_type: TaskType,
        patch_size: int,
        pooling_type: str,
        num_classes: int,
        use_pooled_tokens: bool = False,
    ) -> None:
        """Initialize the backbone with head."""
        super().__init__()
        self.backbone = model
        self.wrapper = get_eval_wrapper(
            model,
            task_type=task_type,
            patch_size=patch_size,
            pooling_type=pooling_type,
            concat_features=False,
            use_pooled_tokens=use_pooled_tokens,
        )
        self.task_type = task_type
        self.patch_size = patch_size
        self.num_classes = num_classes
        # placeholder head; real in_dim discovered on first forward
        self._head = nn.Linear(1, 1, bias=True)
        self._inited = False

    def _init_head(self, emb_dim: int, device: torch.device) -> None:
        """Initialize the head based on the embedding dimension."""
        if self.task_type == TaskType.CLASSIFICATION:
            self._head = nn.Linear(emb_dim, self.num_classes, bias=True)
        else:
            logits_per_patch = int(self.num_classes * self.patch_size * self.patch_size)
            self._head = nn.Linear(emb_dim, logits_per_patch, bias=True)

        self._head = self._head.to(device=device)
        self._inited = True

    def forward(
        self, batch: MaskedOlmoEarthSample, labels: torch.Tensor, is_train: bool = True
    ) -> torch.Tensor:
        """Forward pass through the model and head."""
        dev = next(self.wrapper.parameters()).device
        emb, labels = self.wrapper(batch, labels, is_train=is_train)
        emb = cast(torch.Tensor, emb)
        emb_dim = emb.shape[-1]
        if not self._inited:
            self._init_head(emb_dim, dev)
        if emb.device != dev:
            emb = emb.to(dev, non_blocking=True)
        return self._head(emb), labels


def _to_device(
    masked: MaskedOlmoEarthSample, device: torch.device
) -> MaskedOlmoEarthSample:
    """Move a MaskedOlmoEarthSample to a device with appropriate dtypes."""
    d = masked.as_dict(return_none=False)
    for k, v in d.items():
        if k == "timestamps":
            d[k] = v.to(device=device)
        else:
            d[k] = v.to(device=device, dtype=torch.bfloat16)
    return MaskedOlmoEarthSample.from_dict(d)


@torch.no_grad()
def _eval_cls(
    module: BackboneWithHead,
    loader: DataLoader,
    device: torch.device,
    is_multilabel: bool,
) -> float:
    """Evaluate classification metric (micro F1 for multilabel, accuracy otherwise)."""
    module.eval()
    logits_all, labels_all = [], []
    for masked, label in loader:
        label = label.to(device=device)
        masked = _to_device(masked, device)
        with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16):
            logits, _ = module(masked, label, is_train=False)  # (B, C)
        logits_all.append(logits.float().cpu())
        labels_all.append(label.cpu())
    logits = torch.cat(logits_all, 0)
    labels = torch.cat(labels_all, 0)
    if is_multilabel:
        preds = torch.sigmoid(logits).gt(0.5).int()
        return f1_score(
            labels.numpy().astype(int),
            preds.numpy(),
            average="micro",
            zero_division=0,
        )
    else:
        preds = torch.argmax(logits, dim=-1)
        return accuracy_score(labels.numpy(), preds.numpy())


@torch.no_grad()
def _eval_seg(
    module: BackboneWithHead,
    loader: DataLoader,
    device: torch.device,
    num_classes: int,
    patch_size: int,
) -> float:
    """Evaluate segmentation mIoU."""
    module.eval()
    preds_all, labels_all = [], []
    for masked, label in loader:
        label = label.to(device=device)
        masked = _to_device(masked, device)
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
    return mean_iou(preds, labels, num_classes=num_classes, ignore_label=-1)


def count_params(backbone: nn.Module, head: nn.Module) -> tuple[int, int, int, int]:
    """Count total and trainable parameters separately for the backbone and the linear head."""
    total_backbone = sum(p.numel() for p in backbone.parameters())
    trainable_backbone = sum(
        p.numel() for p in backbone.parameters() if p.requires_grad
    )

    total_head = sum(p.numel() for p in head.parameters())
    trainable_head = sum(p.numel() for p in head.parameters() if p.requires_grad)

    return total_backbone, trainable_backbone, total_head, trainable_head


def _snapshot_state_dict(module: nn.Module) -> dict[str, torch.Tensor]:
    """Clone a module's state dict onto CPU for later restoration."""
    return {k: v.detach().cpu().clone() for k, v in module.state_dict().items()}


def _set_backbone_trainable(backbone: nn.Module, requires_grad: bool) -> None:
    """Toggle gradient computation for backbone parameters."""
    for param in backbone.parameters():
        param.requires_grad = requires_grad


def run_finetune_eval(
    task_name: str,
    task_config: EvalDatasetConfig,
    trainer: Trainer,
    model: nn.Module,
    device: torch.device,
    lr: float,
    epochs: int,
    patch_size: int,
    pooling_type: str,
    use_pooled_tokens: bool,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader | None,
    seed: int | None = None,
    best_checkpoint_path: str | None = None,
) -> dict[str, Any]:
    """Finetune the model on a downstream task and evaluate.

    Returns:
        Dictionary with keys:
            - val_score: Validation score
            - test_score: Test score (0.0 if no test loader)
            - bootstrap_stats: Empty dict (bootstrap not supported for finetune)
    """
    if seed is not None:
        logger.info(f"Setting finetune random seed to {seed}")
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    ft = BackboneWithHead(
        model=model,
        task_type=task_config.task_type,
        patch_size=patch_size,
        pooling_type=pooling_type,
        num_classes=task_config.num_classes,
        use_pooled_tokens=use_pooled_tokens,
    ).to(device)

    # Trigger _init_head once with a tiny dry pass
    with torch.no_grad(), torch.autocast(device_type=device.type, dtype=torch.bfloat16):
        sample_batch, label = next(iter(train_loader))
        _, _ = ft(_to_device(sample_batch, device), label.to(device))

    # Freeze the backbone for the first 20% of the epochs
    freeze_epochs = math.ceil(0.2 * epochs) if epochs > 0 else 0
    backbone_unfrozen = freeze_epochs == 0
    if not backbone_unfrozen:
        _set_backbone_trainable(ft.backbone, False)
        logger.info(
            f"Freezing backbone for the first {freeze_epochs} epoch(s) before unfreezing."
        )

    total_backbone, trainable_backbone, total_head, trainable_head = count_params(
        ft.backbone, ft._head
    )
    logger.info(f"Total backbone parameters: {total_backbone:,}")
    logger.info(f"Trainable backbone parameters: {trainable_backbone:,}")
    logger.info(f"Total head parameters: {total_head:,}")
    logger.info(f"Trainable head parameters: {trainable_head:,}")

    current_lr = lr
    opt = torch.optim.AdamW(ft.parameters(), lr=current_lr)
    scheduler = ReduceLROnPlateau(
        opt,
        mode="max",
        factor=0.2,
        patience=2,
        min_lr=0.0,
        cooldown=10,
    )
    if task_config.task_type == TaskType.CLASSIFICATION:
        loss_fn: nn.Module = (
            nn.MultiLabelSoftMarginLoss()
            if task_config.is_multilabel
            else nn.CrossEntropyLoss()
        )
    else:
        loss_fn = nn.CrossEntropyLoss(ignore_index=-1)

    best_state = _snapshot_state_dict(ft)
    best_val_metric = float("-inf")

    ft.train()
    wandb_logger = _get_wandb_logger(trainer)
    num_batches = len(train_loader)

    for epoch in range(epochs):
        # Reset epoch and global step
        trainer.global_step = epoch * len(train_loader)
        trainer.epoch = epoch + 1

        if not backbone_unfrozen and epoch >= freeze_epochs:
            _set_backbone_trainable(ft.backbone, True)
            backbone_unfrozen = True
            current_lr = lr / 10.0
            for group in opt.param_groups:
                group["lr"] = current_lr
            logger.info(
                "Backbone unfrozen; reducing optimizer learning rate to "
                f"{current_lr:.3e} for remaining epochs."
            )

        for i, (masked, label) in enumerate(train_loader):
            label = label.to(device=device)
            masked = _to_device(masked, device)
            with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16):
                logits, label = ft(masked, label)
                if task_config.task_type == TaskType.SEGMENTATION:
                    H, W = logits.shape[1], logits.shape[2]
                    logits = rearrange(
                        logits,
                        "b h w (c i j) -> b c (h i) (w j)",
                        h=H,
                        w=W,
                        c=task_config.num_classes,
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
                loss = loss_fn(logits, label)
                if wandb_logger is not None:
                    wandb_logger.log(
                        {
                            f"{task_name}_step": epoch * num_batches + i,
                            f"{task_name}/train_loss": loss.item(),
                        }
                    )
                logger.info(
                    f"Finetune Epoch [{epoch + 1}/{epochs}] Step [{i + 1}/{len(train_loader)}] Loss: {loss.item():.4f}"
                )
            loss.backward()
            opt.step()
            opt.zero_grad()

        if task_config.task_type == TaskType.CLASSIFICATION:
            val_metric = _eval_cls(ft, val_loader, device, task_config.is_multilabel)
        else:
            val_metric = _eval_seg(
                ft,
                val_loader,
                device,
                task_config.num_classes,
                patch_size,
            )
        if wandb_logger is not None:
            wandb_logger.log(
                {
                    f"{task_name}_step": (epoch + 1) * num_batches,
                    f"{task_name}/val_metric": val_metric,
                }
            )
        logger.info(
            f"Finetune Epoch [{epoch + 1}/{epochs}] Validation Metric: {val_metric:.4f}"
        )
        scheduler.step(val_metric)

        if val_metric > best_val_metric:
            best_val_metric = val_metric
            best_state = _snapshot_state_dict(ft)
            logger.info(
                f"New best validation metric {best_val_metric:.4f} at epoch {epoch + 1}"
            )
        ft.train()

    if best_val_metric == float("-inf"):
        if task_config.task_type == TaskType.CLASSIFICATION:
            best_val_metric = _eval_cls(
                ft, val_loader, device, task_config.is_multilabel
            )
        else:
            best_val_metric = _eval_seg(
                ft,
                val_loader,
                device,
                task_config.num_classes,
                patch_size,
            )

    ft.load_state_dict(best_state)
    if best_checkpoint_path is not None:
        os.makedirs(os.path.dirname(best_checkpoint_path), exist_ok=True)
        torch.save(best_state, best_checkpoint_path)
        logger.info(f"Saved best checkpoint to {best_checkpoint_path}")
    else:
        logger.info("No best checkpoint path provided, skipping saving best checkpoint")

    if task_config.task_type == TaskType.CLASSIFICATION:
        val_score = best_val_metric
        test_score = (
            _eval_cls(ft, test_loader, device, task_config.is_multilabel)
            if test_loader is not None
            else 0.0
        )
    else:
        val_score = best_val_metric
        test_score = (
            _eval_seg(ft, test_loader, device, task_config.num_classes, patch_size)
            if test_loader is not None
            else 0.0
        )

    return {
        "val_score": val_score,
        "test_score": test_score,
        "bootstrap_stats": {},
    }
