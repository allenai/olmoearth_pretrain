"""Main finetuning training loop."""

from __future__ import annotations

import math
import os
import random
from logging import getLogger
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from olmo_core.train.trainer import Trainer
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from olmoearth_pretrain.evals.datasets.configs import EvalDatasetConfig, TaskType
from olmoearth_pretrain.evals.finetune.checkpoint import (
    load_training_checkpoint,
    save_training_checkpoint,
)
from olmoearth_pretrain.evals.finetune.constants import (
    FREEZE_EPOCH_FRACTION,
    SCHEDULER_COOLDOWN,
    SCHEDULER_FACTOR,
    SCHEDULER_MIN_LR,
    SCHEDULER_PATIENCE,
    UNFREEZE_LR_FACTOR,
)
from olmoearth_pretrain.evals.finetune.evaluate import eval_cls, eval_reg, eval_seg
from olmoearth_pretrain.evals.finetune.model import (
    BackboneWithHead,
    HeadType,
    set_backbone_trainable,
    snapshot_state_dict,
    to_device,
)
from olmoearth_pretrain.evals.linear_probe import weighted_dice_loss
from olmoearth_pretrain.evals.metrics import EvalMetric, EvalResult, EvalTaskResult


def _primary_metric_higher_is_better(
    task_type: TaskType, primary_metric: EvalMetric | None
) -> bool:
    """Whether validation primary should be maximized (scheduler / best checkpoint)."""
    if task_type == TaskType.REGRESSION:
        return primary_metric == EvalMetric.R2
    return True


logger = getLogger(__name__)


def _get_wandb_logger(trainer: Trainer) -> Any | None:
    """Return the wandb module from the OlmoEarth callback, if available."""
    from olmoearth_pretrain.train.callbacks.wandb import OlmoEarthWandBCallback

    for callback in trainer._iter_callbacks():
        if isinstance(callback, OlmoEarthWandBCallback) and callback.enabled:
            return callback.wandb
    return None


def _save_best_and_cleanup(
    best_state: dict[str, torch.Tensor],
    best_checkpoint_path: str | None,
    resume_checkpoint_path: str | None,
) -> None:
    """Save the best model checkpoint and remove the resume checkpoint."""
    if best_checkpoint_path is not None:
        dir_path = os.path.dirname(best_checkpoint_path)
        if dir_path:
            os.makedirs(dir_path, exist_ok=True)
        torch.save(best_state, best_checkpoint_path)
        logger.info(f"Saved best checkpoint to {best_checkpoint_path}")
    else:
        logger.info("No best checkpoint path provided, skipping saving best checkpoint")
    if resume_checkpoint_path and os.path.exists(resume_checkpoint_path):
        os.remove(resume_checkpoint_path)
        logger.info(f"Removed resume checkpoint {resume_checkpoint_path}")


def compute_eval_metrics(
    ft: nn.Module,
    task_config: EvalDatasetConfig,
    val_loader: DataLoader,
    test_loader: DataLoader | None,
    device: torch.device,
    patch_size: int,
    primary_metric: EvalMetric | None = None,
    primary_metric_class: int | None = None,
) -> EvalTaskResult:
    """Evaluate a finetuned model on val and test sets."""
    ft.eval()

    if task_config.task_type == TaskType.CLASSIFICATION:
        val_result = eval_cls(
            ft,
            val_loader,
            device,
            task_config.is_multilabel,
            primary_metric=primary_metric,
            primary_metric_class=primary_metric_class,
        )
    elif task_config.task_type == TaskType.REGRESSION:
        val_result = eval_reg(
            ft,
            val_loader,
            device,
            primary_metric=primary_metric,
        )
    else:
        val_result = eval_seg(
            ft,
            val_loader,
            device,
            task_config.num_classes,
            patch_size,
            primary_metric=primary_metric,
            primary_metric_class=primary_metric_class,
        )

    test_result: EvalResult | None = None
    if test_loader is not None:
        if task_config.task_type == TaskType.CLASSIFICATION:
            test_result = eval_cls(
                ft,
                test_loader,
                device,
                task_config.is_multilabel,
                primary_metric=primary_metric,
                primary_metric_class=primary_metric_class,
            )
        elif task_config.task_type == TaskType.REGRESSION:
            test_result = eval_reg(
                ft,
                test_loader,
                device,
                primary_metric=primary_metric,
            )
        else:
            test_result = eval_seg(
                ft,
                test_loader,
                device,
                task_config.num_classes,
                patch_size,
                primary_metric=primary_metric,
                primary_metric_class=primary_metric_class,
            )

    return EvalTaskResult(val_result=val_result, test_result=test_result)


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
    resume_checkpoint_path: str | None = None,
    primary_metric: EvalMetric | None = None,
    primary_metric_class: int | None = None,
    ft_grad_accum_steps: int = 1,
    head_type: HeadType = "linear",
    use_dice_loss: bool = False,
) -> EvalTaskResult:
    """Finetune the model on a downstream task and evaluate."""
    accum_steps = max(1, ft_grad_accum_steps)
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
        head_type=head_type,
    ).to(device)

    # Trigger _init_head once with a tiny dry pass which initializes the head with the correct dimension.
    with torch.no_grad(), torch.autocast(device_type=device.type, dtype=torch.bfloat16):
        sample_batch, label = next(iter(train_loader))
        _, _ = ft(to_device(sample_batch, device), label.to(device))

    # If best checkpoint exists, load it and evaluate directly
    if best_checkpoint_path and os.path.exists(best_checkpoint_path):
        logger.info(f"Loading existing best checkpoint from {best_checkpoint_path}")
        state = torch.load(best_checkpoint_path, map_location=device)
        ft.load_state_dict(state)
        return compute_eval_metrics(
            ft,
            task_config,
            val_loader,
            test_loader,
            device,
            patch_size,
            primary_metric=primary_metric,
            primary_metric_class=primary_metric_class,
        )

    # Freeze the backbone for the first portion of epochs
    freeze_epochs = math.ceil(FREEZE_EPOCH_FRACTION * epochs) if epochs > 0 else 0
    backbone_unfrozen = freeze_epochs == 0
    if not backbone_unfrozen:
        set_backbone_trainable(ft.backbone, False)
        logger.info(
            f"Freezing backbone for the first {freeze_epochs} epoch(s) before unfreezing."
        )

    current_lr = lr
    opt = torch.optim.AdamW(ft.parameters(), lr=current_lr)
    higher_is_better = _primary_metric_higher_is_better(
        task_config.task_type, primary_metric
    )
    scheduler = ReduceLROnPlateau(
        opt,
        mode="max" if higher_is_better else "min",
        factor=SCHEDULER_FACTOR,
        patience=SCHEDULER_PATIENCE,
        min_lr=SCHEDULER_MIN_LR,
        cooldown=SCHEDULER_COOLDOWN,
    )
    if task_config.task_type == TaskType.CLASSIFICATION:
        loss_fn: nn.Module = (
            nn.MultiLabelSoftMarginLoss()
            if task_config.is_multilabel
            else nn.CrossEntropyLoss()
        )
    elif task_config.task_type == TaskType.REGRESSION:
        loss_fn = nn.MSELoss()
    elif use_dice_loss:
        ce_fn = nn.CrossEntropyLoss(ignore_index=-1)
        num_classes = task_config.num_classes

        class _CombinedLoss(nn.Module):
            def forward(
                self, logits: torch.Tensor, targets: torch.Tensor
            ) -> torch.Tensor:
                return ce_fn(logits, targets) + weighted_dice_loss(
                    logits, targets, num_classes
                )

        loss_fn = _CombinedLoss()
    else:
        loss_fn = nn.CrossEntropyLoss(ignore_index=-1)

    best_state = snapshot_state_dict(ft)
    best_val_metric = float("-inf") if higher_is_better else float("inf")
    start_epoch = 0

    # Resume from checkpoint if it exists
    if resume_checkpoint_path and os.path.exists(resume_checkpoint_path):
        ckpt = load_training_checkpoint(resume_checkpoint_path, device)
        start_epoch = ckpt["epoch"] + 1
        ft.load_state_dict(ckpt["model_state"])
        opt.load_state_dict(ckpt["optimizer_state"])
        scheduler.load_state_dict(ckpt["scheduler_state"])
        best_state = ckpt["best_state"]
        best_val_metric = ckpt["best_val_metric"]
        backbone_unfrozen = ckpt["backbone_unfrozen"]
        # Handle backbone freeze state on resume
        set_backbone_trainable(ft.backbone, backbone_unfrozen)
        logger.info(
            f"Resumed from epoch {start_epoch}, best_val_metric={best_val_metric:.4f}, "
            f"backbone_unfrozen={backbone_unfrozen}"
        )

        # All epochs already completed in a previous run — save best, clean up, evaluate.
        if start_epoch >= epochs:
            logger.info(
                "All epochs already completed before preemption. "
                "Saving best checkpoint and evaluating."
            )
            ft.load_state_dict(best_state)
            _save_best_and_cleanup(
                best_state, best_checkpoint_path, resume_checkpoint_path
            )
            return compute_eval_metrics(
                ft,
                task_config,
                val_loader,
                test_loader,
                device,
                patch_size,
                primary_metric=primary_metric,
                primary_metric_class=primary_metric_class,
            )

    ft.train()
    wandb_logger = _get_wandb_logger(trainer)
    num_batches = len(train_loader)
    if accum_steps > 1:
        eff_bs = train_loader.batch_size
        if eff_bs is not None:
            logger.info(
                "Finetune grad accumulation: "
                f"batch_size={eff_bs}, accum_steps={accum_steps}, "
                f"effective batch_size={eff_bs * accum_steps}"
            )

    for epoch in range(start_epoch, epochs):
        # Reset epoch and global step
        trainer.global_step = epoch * len(train_loader)
        trainer.epoch = epoch + 1

        if not backbone_unfrozen and epoch >= freeze_epochs:
            set_backbone_trainable(ft.backbone, True)
            backbone_unfrozen = True
            current_lr = lr * UNFREEZE_LR_FACTOR
            for group in opt.param_groups:
                group["lr"] = current_lr
            logger.info(
                "Backbone unfrozen; reducing optimizer learning rate to "
                f"{current_lr:.3e} for remaining epochs."
            )

        for i, (masked, label) in enumerate(train_loader):
            label = label.to(device=device)
            masked = to_device(masked, device)
            with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16):
                logits, label = ft(masked, label)
                if task_config.task_type == TaskType.SEGMENTATION:
                    if not ft.pixel_space_output:
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
                if task_config.task_type == TaskType.REGRESSION:
                    raw_loss = loss_fn(logits, label.float())
                else:
                    raw_loss = loss_fn(logits, label)
                loss = raw_loss / accum_steps
                if wandb_logger is not None:
                    wandb_logger.log(
                        {
                            f"{task_name}_step": epoch * num_batches + i,
                            f"{task_name}/train_loss": raw_loss.item(),
                        }
                    )
                logger.info(
                    f"Finetune Epoch [{epoch + 1}/{epochs}] Step [{i + 1}/{len(train_loader)}] Loss: {raw_loss.item():.4f}"
                )
            loss.backward()
            if (i + 1) % accum_steps == 0 or (i + 1) == num_batches:
                opt.step()
                opt.zero_grad()

        if task_config.task_type == TaskType.CLASSIFICATION:
            val_result = eval_cls(
                ft,
                val_loader,
                device,
                task_config.is_multilabel,
                primary_metric=primary_metric,
                primary_metric_class=primary_metric_class,
            )
        elif task_config.task_type == TaskType.REGRESSION:
            val_result = eval_reg(
                ft,
                val_loader,
                device,
                primary_metric=primary_metric,
            )
        else:
            val_result = eval_seg(
                ft,
                val_loader,
                device,
                task_config.num_classes,
                patch_size,
                primary_metric=primary_metric,
                primary_metric_class=primary_metric_class,
            )

        if device.type == "cuda":
            torch.cuda.empty_cache()

        if wandb_logger is not None:
            wandb_logger.log(
                {
                    f"{task_name}_step": (epoch + 1) * num_batches,
                    f"{task_name}/val_metric": val_result.primary,
                }
            )
        logger.info(
            f"Finetune Epoch [{epoch + 1}/{epochs}] Validation Metric: {val_result.primary:.4f}"
        )
        scheduler.step(val_result.primary)

        improved = (
            val_result.primary > best_val_metric
            if higher_is_better
            else val_result.primary < best_val_metric
        )
        if improved:
            best_val_metric = val_result.primary
            best_state = snapshot_state_dict(ft)
            logger.info(
                f"New best validation metric {best_val_metric:.4f} at epoch {epoch + 1}"
            )

        # Save resumable checkpoint at end of each epoch
        if resume_checkpoint_path:
            save_training_checkpoint(
                path=resume_checkpoint_path,
                epoch=epoch,
                model_state=snapshot_state_dict(ft),
                optimizer_state=opt.state_dict(),
                scheduler_state=scheduler.state_dict(),
                best_state=best_state,
                best_val_metric=best_val_metric,
                backbone_unfrozen=backbone_unfrozen,
            )

        ft.train()

    ft.load_state_dict(best_state)
    _save_best_and_cleanup(best_state, best_checkpoint_path, resume_checkpoint_path)
    return compute_eval_metrics(
        ft,
        task_config,
        val_loader,
        test_loader,
        device,
        patch_size,
        primary_metric=primary_metric,
        primary_metric_class=primary_metric_class,
    )
