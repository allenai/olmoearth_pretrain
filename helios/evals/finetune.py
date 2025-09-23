"""Finetune the Helios and other models on a downstream task."""

from __future__ import annotations

from logging import getLogger

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader

from helios.evals.datasets.configs import EvalDatasetConfig, TaskType
from helios.evals.eval_wrapper import get_eval_wrapper
from helios.evals.metrics import mean_iou
from helios.evals.utils import adjust_learning_rate
from helios.train.masking import MaskedHeliosSample

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

    def forward(self, batch: MaskedHeliosSample) -> torch.Tensor:
        """Forward pass through the model and head."""
        dev = next(self.wrapper.parameters()).device
        emb = self.wrapper(batch)  # CLS: (B, D)  |  SEG: (B, H, W, D)
        emb_dim = emb.shape[-1]
        if not self._inited:
            self._init_head(emb_dim, dev)
        if emb.device != dev:
            emb = emb.to(dev, non_blocking=True)
        return self._head(emb)


def _to_device(masked: MaskedHeliosSample, device: torch.device) -> MaskedHeliosSample:
    """Move a MaskedHeliosSample to a device with appropriate dtypes."""
    d = masked.as_dict(return_none=False)
    for k, v in d.items():
        if k == "timestamps":
            d[k] = v.to(device=device)
        else:
            d[k] = v.to(device=device, dtype=torch.bfloat16)
    return MaskedHeliosSample.from_dict(d)


@torch.no_grad()
def _eval_cls(
    module: _BackboneWithHead, loader: DataLoader, device: torch.device
) -> float:
    """Evaluate classification accuracy."""
    module.eval()
    logits_all, labels_all = [], []
    for masked, label in loader:
        label = label.to(device=device)
        masked = _to_device(masked, device)
        with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16):
            logits = module(masked)  # (B, C)
        logits_all.append(logits.float().cpu())
        labels_all.append(label.cpu())
    logits = torch.cat(logits_all, 0)
    labels = torch.cat(labels_all, 0)
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
            logits = module(masked)  # (B, H, W, C*p*p)
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
) -> float:
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
        sample_batch, _ = next(iter(train_loader))
        _ = ft(_to_device(sample_batch, device))

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

    ft.train()
    for epoch in range(epochs):
        for i, (masked, label) in enumerate(train_loader):
            label = label.to(device=device)
            masked = _to_device(masked, device)
            with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16):
                logits = ft(masked)
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
        return _eval_cls(ft, val_loader, device)
    else:
        return _eval_seg(ft, val_loader, device, task_config.num_classes, patch_size)
