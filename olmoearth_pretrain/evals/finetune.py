"""Finetune the OlmoEarth Pretrain and other models on a downstream task."""

from __future__ import annotations

from logging import getLogger
from typing import cast

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader

from olmoearth_pretrain.evals.datasets.configs import EvalDatasetConfig, TaskType
from olmoearth_pretrain.evals.eval_wrapper import get_eval_wrapper
from olmoearth_pretrain.evals.metrics import mean_iou
from olmoearth_pretrain.evals.utils import adjust_learning_rate
from olmoearth_pretrain.train.masking import MaskedOlmoEarthSample

logger = getLogger(__name__)


class _BackboneWithHead(nn.Module):
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
        self, batch: MaskedOlmoEarthSample, labels: torch.Tensor
    ) -> torch.Tensor:
        """Forward pass through the model and head."""
        dev = next(self.wrapper.parameters()).device
        # classification: (B, D), segmentation: (B, H, W, D)
        emb, _ = self.wrapper(batch, None)
        emb = cast(torch.Tensor, emb)
        emb_dim = emb.shape[-1]
        if not self._inited:
            self._init_head(emb_dim, dev)
        if emb.device != dev:
            emb = emb.to(dev, non_blocking=True)
        return self._head(emb)


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
    module: _BackboneWithHead,
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
            logits = module(masked, label)  # (B, C)
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
    module: _BackboneWithHead,
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
            logits = module(masked, label)  # (B, H, W, C*p*p)
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


def count_params(module: nn.Module) -> tuple[int, int]:
    """Count total and trainable parameters in a module."""
    total = sum(p.numel() for p in module.parameters())
    trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
    return total, trainable


def _snapshot_state_dict(module: nn.Module) -> dict[str, torch.Tensor]:
    """Clone a module's state dict onto CPU for later restoration."""
    return {k: v.detach().cpu().clone() for k, v in module.state_dict().items()}


def run_finetune_eval(
    task_config: EvalDatasetConfig,
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
) -> tuple[float, float]:
    """Finetune the model on a downstream task and evaluate."""
    ft = _BackboneWithHead(
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
        _ = ft(_to_device(sample_batch, device), label.to(device))

    total, trainable = count_params(ft)
    logger.info(f"Total parameters: {total:,}")
    logger.info(f"Trainable parameters: {trainable:,}")

    opt = torch.optim.AdamW(ft.parameters(), lr=lr)
    if task_config.task_type == TaskType.CLASSIFICATION:
        loss_fn: nn.Module = (
            nn.MultiLabelSoftMarginLoss()
            if task_config.is_multilabel
            else nn.CrossEntropyLoss()
        )
    else:
        loss_fn = nn.CrossEntropyLoss(ignore_index=-1)

    patience = max(1, int(0.1 * epochs)) if epochs > 0 else 1
    logger.info(f"Using early stopping patience of {patience} epochs")

    best_state = _snapshot_state_dict(ft)
    best_val_metric = float("-inf")
    epochs_without_improvement = 0
    should_stop = False

    ft.train()
    for epoch in range(epochs):
        for i, (masked, label) in enumerate(train_loader):
            label = label.to(device=device)
            masked = _to_device(masked, device)
            with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16):
                logits = ft(masked, label)
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
                logger.info(
                    f"Finetune Epoch [{epoch + 1}/{epochs}] Step [{i + 1}/{len(train_loader)}] Loss: {loss.item():.4f}"
                )
            loss.backward()
            adjust_learning_rate(
                optimizer=opt,
                epoch=epoch + (i / max(1, len(train_loader))),
                total_epochs=epochs,
                warmup_epochs=max(1, int(0.1 * epochs)),
                max_lr=lr,
                min_lr=1.0e-5,
            )
            # torch.nn.utils.clip_grad_norm_(ft.parameters(), 1.0)
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
        logger.info(
            f"Finetune Epoch [{epoch + 1}/{epochs}] Validation Metric: {val_metric:.4f}"
        )

        if val_metric > best_val_metric:
            best_val_metric = val_metric
            best_state = _snapshot_state_dict(ft)
            epochs_without_improvement = 0
            logger.info(
                f"New best validation metric {best_val_metric:.4f} at epoch {epoch + 1}"
            )
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                logger.info(
                    "Early stopping triggered after "
                    f"{epochs_without_improvement} epochs without improvement"
                )
                should_stop = True

        if should_stop:
            break
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

    if task_config.task_type == TaskType.CLASSIFICATION:
        val_acc = best_val_metric
        test_acc = (
            _eval_cls(ft, test_loader, device, task_config.is_multilabel)
            if test_loader is not None
            else 0.0
        )
        return val_acc, test_acc
    else:
        val_miou = best_val_metric
        test_miou = (
            _eval_seg(ft, test_loader, device, task_config.num_classes, patch_size)
            if test_loader is not None
            else 0.0
        )
        return val_miou, test_miou
