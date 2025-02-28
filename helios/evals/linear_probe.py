"""Linear probing for evals."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from sklearn.metrics import accuracy_score

from helios.evals.datasets import EvalType

from .metrics import mean_iou

PROBING_LRs = {
    "LP": [
        1e-4,
        3e-4,
        5e-4,
        8e-4,
        1e-3,
        3e-3,
        5e-3,
        8e-3,
        1e-2,
        3e-2,
        5e-2,
        8e-2,
        1e-1,
        3e-1,
        5e-1,
        8e-1,
    ],
}


def _adjust_learning_rate(
    optimizer: torch.optim.Optimizer,
    epoch: int,
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


def train_probe(
    data_loader: torch.utils.data.DataLoader,
    lr: float,
    epochs: int,
    in_features: int,
    num_classes: int,
    device: torch.device,
    eval_type: EvalType,
    patch_size: int,
) -> nn.Module:
    """Train a linear probe."""
    if eval_type == EvalType.classifciaton:
        probe = nn.Sequential(
            nn.BatchNorm1d(in_features), nn.Linear(in_features, num_classes)
        ).to(device)
    else:
        logits_per_patch = int(num_classes * patch_size * patch_size)
        probe = nn.Sequential(nn.Linear(in_features, logits_per_patch)).to(device)

    opt = torch.optim.AdamW(probe.parameters(), lr=lr)

    sched_config = {
        "lr": lr,
        "warmup_epochs": int(epochs * 0.1),
        "min_lr": 1.0e-5,
        "epochs": int(epochs),
    }
    probe = probe.train()

    loss_function = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        for i, batch in enumerate(data_loader):
            batch_emb, batch_labels = batch  # (bsz, dim), (bsz)
            batch_emb = batch_emb.to(device)
            spatial_patches_per_dim = int(batch_emb.shape[1] ** 0.5)

            with torch.amp.autocast(dtype=torch.bfloat16):
                logits = probe(batch_emb)  # (bsz, num_classes)
                if eval_type == EvalType.segmentation:
                    logits = rearrange(
                        logits,
                        "b (h w) (c i j) -> b c (h i) (w j)",
                        h=spatial_patches_per_dim,
                        w=spatial_patches_per_dim,
                        c=num_classes,
                        i=patch_size,
                        j=patch_size,
                    )
                    if logits.shape[-2] != batch_labels.shape[-2]:
                        logits = F.interpolate(
                            logits,
                            size=(batch_labels.shape[-2], batch_labels.shape[-1]),
                            mode="bilinear",
                            align_corners=True,
                        )  # (bsz, num_classes, H, W)

                loss = loss_function(logits, batch_labels.to(device))

            loss.backward()
            _adjust_learning_rate(
                optimizer=opt,
                epoch=int(epoch + (i / len(data_loader))),
                total_epochs=int(sched_config["epochs"]),
                warmup_epochs=int(sched_config["warmup_epochs"]),
                max_lr=sched_config["lr"],
                min_lr=sched_config["min_lr"],
            )

            opt.step()
            opt.zero_grad()

    return probe


def evaluate_probe(
    data_loader: torch.utils.data.DataLoader,
    probe: nn.Module,
    num_classes: int,
    patch_size: int,
    device: torch.device,
    eval_type: EvalType,
) -> float:
    """Evaluate a trained linear probe."""
    probe = probe.eval()

    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch in data_loader:
            batch_emb, batch_labels = batch  # (bsz, num_patches, dim), (bsz, H, W)
            spatial_patches_per_dim = int(batch_emb.shape[1] ** 0.5)
            batch_emb = batch_emb.to(device)

            with torch.amp.autocast(dtype=torch.bfloat16):
                logits = probe(batch_emb)  # (bsz, num_patches, logits_per_patch)
                if eval_type == EvalType.segmentation:
                    logits = rearrange(
                        logits,
                        "b (h w) (c i j) -> b c (h i) (w j)",
                        h=spatial_patches_per_dim,
                        w=spatial_patches_per_dim,
                        c=num_classes,
                        i=patch_size,
                        j=patch_size,
                    )
                    if logits.shape[-2] != batch_labels.shape[-2]:
                        logits = F.interpolate(
                            logits,
                            size=(batch_labels.shape[-2], batch_labels.shape[-1]),
                            mode="bilinear",
                            align_corners=True,
                        )  # (bsz, num_classes, H, W)

            preds = torch.argmax(logits, dim=1).cpu()
            all_preds.append(preds)
            all_labels.append(batch_labels)

    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)
    if eval_type == EvalType.segmentation:
        return mean_iou(all_preds, all_labels, num_classes=num_classes, ignore_label=-1)
    else:
        return accuracy_score(all_labels, all_preds)
