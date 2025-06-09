"""Train and evaluate a linear probe."""

import math
from logging import getLogger

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader, TensorDataset

from helios.evals.datasets.configs import EvalDatasetConfig, TaskType
from helios.evals.metrics import mean_iou
from helios.evals.utils import adjust_learning_rate

logger = getLogger(__name__)


def train_and_eval_probe(
    config: EvalDatasetConfig,
    lr: float,
    train_embeddings: torch.Tensor,
    train_labels: torch.Tensor,
    test_embeddings: torch.Tensor,
    test_labels: torch.Tensor,
    device: torch.device,
    patch_size: int,
    batch_size: int,
    epochs: int = 50,
    eval_interval: int = 1,
) -> float:
    """Run a linear probe on the Helios model."""
    if train_embeddings.shape[-1] != test_embeddings.shape[-1]:
        raise ValueError("Embedding dims don't match.")
    in_features = train_embeddings.shape[-1]

    if config.task_type == TaskType.SEGMENTATION:
        logits_per_patch = int(config.num_classes * patch_size * patch_size)
        probe = nn.Sequential(nn.Linear(in_features, logits_per_patch)).to(device)
    else:
        probe = nn.Sequential(
            nn.BatchNorm1d(in_features), nn.Linear(in_features, config.num_classes)
        ).to(device)

    num_times_to_run_eval = math.ceil(epochs / eval_interval)
    data_loader = None
    eval_mious = []
    for i in range(num_times_to_run_eval):
        start_epoch = i * eval_interval
        end_epoch = min(start_epoch + eval_interval, epochs)

        probe, data_loader = train_probe(
            task_type=config.task_type,
            probe=probe,
            data_loader=(
                DataLoader(
                    TensorDataset(train_embeddings, train_labels),
                    batch_size=batch_size,
                    shuffle=True,
                )
                if data_loader is None
                else data_loader
            ),
            lr=lr,
            epochs=end_epoch,
            total_epochs=epochs,
            current_epoch=start_epoch,
            num_classes=config.num_classes,
            patch_size=patch_size,
            device=device,
        )
        eval_miou = evaluate_probe(
            data_loader=DataLoader(
                TensorDataset(test_embeddings, test_labels),
                batch_size=batch_size,
                shuffle=False,
            ),
            probe=probe,
            num_classes=config.num_classes,
            patch_size=patch_size,
            device=device,
            task_type=config.task_type,
        )
        eval_mious.append(eval_miou)
    for i in range(len(eval_mious)):
        logger.debug(f"Epoch {(i + 1) * eval_interval}, MIoU: {eval_mious[i]}")
    max_miou = max(eval_mious)
    max_epoch = (eval_mious.index(max_miou) + 1) * eval_interval
    logger.debug(f"Max MIoU: {max_miou} at epoch {max_epoch}")
    final_miou = eval_mious[-1]
    if final_miou < max_miou:
        logger.warning(
            f"Final MIoU: {final_miou} at epoch {epochs} is less than max MIoU: {max_miou} at epoch {max_epoch}"
        )
    return final_miou


def train_probe(
    data_loader: DataLoader,
    probe: nn.Module,
    lr: float,
    current_epoch: int,
    epochs: int,
    total_epochs: int,
    num_classes: int,
    patch_size: int,
    device: torch.device,
    task_type: TaskType,
) -> nn.Module:
    """Train a linear probe on a segmentation task."""
    opt = torch.optim.AdamW(probe.parameters(), lr=lr)

    probe = probe.train()
    loss_function = nn.CrossEntropyLoss(ignore_index=-1)  # for MADOS, but ok for others
    start_epoch = current_epoch
    for epoch in range(start_epoch, epochs):
        for i, batch in enumerate(data_loader):
            batch_emb, batch_labels = batch  # (bsz, t_h, t_w, dim), (bsz, H, W)
            spatial_patches_per_dim = batch_emb.shape[1]
            batch_emb = batch_emb.to(device)

            with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16):
                logits = probe(
                    batch_emb
                )  # (bsz, num_patches, logits_per_patch) or (bsz, n_cls)
                if task_type == TaskType.SEGMENTATION:
                    logits = rearrange(
                        logits,
                        "b h w (c i j) -> b c (h i) (w j)",
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
                logger.debug(f"Epoch {epoch}, Step {i}, Loss: {loss.item()}")

            loss.backward()
            adjust_learning_rate(
                optimizer=opt,
                epoch=epoch + (i / len(data_loader)),
                total_epochs=total_epochs,
                warmup_epochs=int(total_epochs * 0.1),
                max_lr=lr,
                min_lr=1.0e-5,
            )

            opt.step()
            opt.zero_grad()

    return probe, data_loader


def evaluate_probe(
    data_loader: DataLoader,
    probe: nn.Module,
    num_classes: int,
    patch_size: int,
    device: torch.device,
    task_type: TaskType,
) -> float:
    """Evaluate a trained linear probe on a segmentation or classification task."""
    probe = probe.eval()

    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch in data_loader:
            batch_emb, batch_labels = batch  # (bsz, num_patches, dim), (bsz, H, W)
            batch_emb = batch_emb.to(device)

            with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16):
                logits = probe(batch_emb)  # (bsz, num_patches, logits_per_patch)
                if task_type == TaskType.SEGMENTATION:
                    spatial_patches_per_dim = batch_emb.shape[1]
                    logits = rearrange(
                        logits,
                        "b h w (c i j) -> b c (h i) (w j)",
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
    if task_type == TaskType.SEGMENTATION:
        metric = mean_iou(
            all_preds, all_labels, num_classes=num_classes, ignore_label=-1
        )
    else:
        metric = accuracy_score(all_labels, all_preds)
    return metric
