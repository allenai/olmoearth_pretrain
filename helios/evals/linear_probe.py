"""Train and evaluate a linear probe."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch.utils.data import DataLoader, TensorDataset

from helios.evals.datasets.configs import EvalDatasetConfig, TaskType
from helios.evals.metrics import mean_iou
from helios.evals.utils import adjust_learning_rate

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


def train_and_eval_probe(
    config: EvalDatasetConfig,
    lr: float,
    train_embeddings: torch.Tensor,
    train_labels: torch.Tensor,
    test_embeddings: torch.Tensor,
    test_labels: torch.Tensor,
    device: torch.device,
    grid_size: int,
    batch_size: int,
    epochs: int = 50,
    eval_interval: int = 1
) -> float:
    """Run a linear probe on the Helios model."""
    if config.task_type != TaskType.SEGMENTATION:
        raise RuntimeError("Unsupported task type for linear probe.")
    if train_embeddings.shape[-1] != test_embeddings.shape[-1]:
        raise ValueError("Embedding dims don't match.")
    in_features = train_embeddings.shape[-1]

    # we test this is the case for segmentation task configs.
    assert config.height_width is not None
    output_patch_size = math.ceil(config.height_width / grid_size)
    logits_per_patch = int(config.num_classes * output_patch_size * output_patch_size)
    probe = nn.Sequential(nn.Linear(in_features, logits_per_patch)).to(device)
    num_eval_groups = math.ceil(epochs / eval_interval)
    data_loader= None
    eval_mious = []
    for i in range(num_eval_groups):
        start_epoch = i * eval_interval
        end_epoch = min(start_epoch + eval_interval, epochs)

        probe, data_loader = train_probe(
            probe=probe,
            data_loader=DataLoader(
                TensorDataset(train_embeddings, train_labels),
                batch_size=batch_size,
                shuffle=True,
            ) if data_loader is None else data_loader,
            lr=lr,
            epochs=end_epoch,
            total_epochs=epochs,
            current_epoch=start_epoch,
            in_features=in_features,
            num_classes=config.num_classes,
            patch_size=output_patch_size,
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
            patch_size=output_patch_size,
            device=device,
        )
        print(f"Epoch {end_epoch}, MIoU: {eval_miou}")
        eval_mious.append(eval_miou)
    for i in range(len(eval_mious)):
        print(f"Epoch {(i + 1) * eval_interval}, MIoU: {eval_mious[i]}")
    max_miou = max(eval_mious)
    max_epoch = (eval_mious.index(max_miou) + 1) * eval_interval
    print(f"Max MIoU: {max_miou} at epoch {max_epoch}")
    return max(eval_mious)


def train_probe(
    data_loader: DataLoader,
    probe: nn.Module,
    lr: float,
    current_epoch: int,
    epochs: int,
    total_epochs: int,
    in_features: int,
    num_classes: int,
    patch_size: int,
    device: torch.device,
) -> nn.Module:
    """Train a linear probe on a segmentation task."""
    logits_per_patch = int(num_classes * patch_size * patch_size)

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
                logits = probe(batch_emb)  # (bsz, num_patches, logits_per_patch)

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
                    # we should log when we are interpolating
                    logits = F.interpolate(
                        logits,
                        size=(batch_labels.shape[-2], batch_labels.shape[-1]),
                        mode="bilinear",
                        align_corners=True,
                    )  # (bsz, num_classes, H, W)
                loss = loss_function(logits, batch_labels.to(device))
                print(f"Epoch {epoch}, Step {i}, Loss: {loss.item()}")

            loss.backward()
            adjust_learning_rate(
                optimizer=opt,
                epoch=epoch + (i / len(data_loader)),
                total_epochs=total_epochs,
                warmup_epochs=0, #1, #int(total_epochs * 0.1),
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
) -> float:
    """Evaluate a trained linear probe on a segmentation task."""
    probe = probe.eval()

    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch in data_loader:
            batch_emb, batch_labels = batch  # (bsz, num_patches, dim), (bsz, H, W)
            spatial_patches_per_dim = batch_emb.shape[1]
            batch_emb = batch_emb.to(device)

            with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16):
                logits = probe(batch_emb)  # (bsz, num_patches, logits_per_patch)
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
    miou = mean_iou(all_preds, all_labels, num_classes=num_classes, ignore_label=-1)
    return miou
