"""Train and evaluate a linear probe."""

import math
from enum import StrEnum
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


class ProbeType(StrEnum):
    """Enumeration of probe types for linear probing."""

    ATTNPOOL = "attnpool"
    LINEAR = "linear"


class AttnPoolLinearProbe(nn.Module):
    """Attention Pooling Linear Probe for segmentation tasks.

    Args:
        in_dim (int): Input feature dimension. Must be divisible by 64.
        out_dim (int): Output dimension (typically num_classes * patch_size * patch_size).

    Attributes:
        query_token (nn.Parameter): Learnable query token for attention pooling.
        num_heads (int): Number of attention heads.
        kv (nn.Linear): Linear layer to produce keys and values.
        linear (nn.Linear): Final linear layer for output logits.
    """

    def __init__(self, in_dim: int, out_dim: int) -> None:
        """Initialize the attention pooling linear probe."""
        super().__init__()
        assert in_dim % 64 == 0, "in_dim must be divisible by 64"
        self.query_token: nn.Parameter = nn.Parameter(torch.empty(in_dim))
        self.num_heads: int = in_dim // 64
        self.kv: nn.Linear = nn.Linear(in_dim, in_dim * 2)
        self.linear: nn.Linear = nn.Linear(in_dim, out_dim)
        self.init_weights()

    def init_weights(self) -> None:
        """Initialize weights for the probe."""
        nn.init.trunc_normal_(self.query_token, std=0.02)
        nn.init.trunc_normal_(self.kv.weight, std=0.02)
        nn.init.zeros_(self.kv.bias)
        nn.init.trunc_normal_(self.linear.weight, std=0.02)
        nn.init.zeros_(self.linear.bias)

    def forward(self, feat_tokens: torch.Tensor) -> dict:
        """Forward pass for attention pooling linear probe.

        Args:
            feat_tokens (torch.Tensor): Input feature tokens of shape (B, H, W, N, D).

        Returns:
            tuple[torch.Tensor, torch.Tensor]:
                - Output logits after linear layer, shape (B, H, W, out_dim).
                - Attention weights, shape (B*H*W, num_heads, 1, N).
        """
        B, H, W, N, D = feat_tokens.shape
        feat_tokens = rearrange(feat_tokens, "b h w n d -> (b h w) n d")
        collapsed_dim = B * H * W
        q = self.query_token.expand(collapsed_dim, 1, -1)
        q = q.reshape(
            collapsed_dim, 1, self.num_heads, D // self.num_heads
        )  # [B, 1, head, D_head]
        q = rearrange(q, "b h n d -> b n h d")
        kv = self.kv(feat_tokens).reshape(
            collapsed_dim, N, 2, self.num_heads, D // self.num_heads
        )  # [B, N, 2, head, D_head]
        kv = rearrange(kv, "b n two h d -> two b h n d")
        k, v = torch.unbind(kv, dim=0)  # 2 * [B, head, N, D_head]
        # Compute attention scores
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(
            D // self.num_heads
        )
        attn_weights = F.softmax(attn_scores, dim=-1)
        x = torch.matmul(attn_weights, v)  # [B, head, 1, D_head]
        x = x.reshape(B, H, W, D)
        return {"logits": self.linear(x), "attn_weights": attn_weights}


class LinearProbe(nn.Module):
    """Linear Probe for classification tasks."""

    def __init__(self, in_dim: int, out_dim: int, use_batchnorm: bool = False) -> None:
        """Initialize the linear probe."""
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        if use_batchnorm:
            self.batchnorm = nn.BatchNorm1d(in_dim)
        else:
            self.batchnorm = nn.Identity()

    def forward(self, x: torch.Tensor) -> dict:
        """Forward pass for linear probe."""
        return {"logits": self.linear(self.batchnorm(x))}


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
    probe_type: ProbeType = ProbeType.LINEAR,
) -> float:
    """Run a linear probe on the Helios model."""
    logger.info(f"Probe type {probe_type}")
    if train_embeddings.shape[-1] != test_embeddings.shape[-1]:
        raise ValueError("Embedding dims don't match.")
    in_features = train_embeddings.shape[-1]

    if config.task_type == TaskType.SEGMENTATION:
        logits_per_patch = int(config.num_classes * patch_size * patch_size)
        if probe_type == ProbeType.ATTNPOOL:
            probe = AttnPoolLinearProbe(
                in_dim=in_features, out_dim=logits_per_patch
            ).to(device)
        elif probe_type == ProbeType.LINEAR:
            probe = LinearProbe(
                in_dim=in_features, out_dim=logits_per_patch, use_batchnorm=False
            ).to(device)
        else:
            raise ValueError(f"Probe type {probe_type} not supported for segmentation.")
    else:
        if probe_type == ProbeType.LINEAR:
            probe = LinearProbe(
                in_dim=in_features, out_dim=config.num_classes, use_batchnorm=True
            ).to(device)
        else:
            raise ValueError(
                f"Probe type {probe_type} not supported for classification."
            )

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
            probe_type=probe_type,
        )
        logger.info(f"Epoch {end_epoch}, MIoU: {eval_miou}")
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
                outputs = probe(
                    batch_emb
                )  # (bsz, num_patches, logits_per_patch) or (bsz, n_cls)
                logits = outputs["logits"]
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

            loss.backward()
            adjust_learning_rate(
                optimizer=opt,
                epoch=epoch + (i / len(data_loader)),
                total_epochs=total_epochs,
                warmup_epochs=int(total_epochs * 0.1),
                max_lr=lr,
                min_lr=1.0e-5,  # maybe this is too low and should just be 10x smaller
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
    probe_type: ProbeType,
) -> float:
    """Evaluate a trained linear probe on a segmentation or classification task."""
    probe = probe.eval()

    all_preds = []
    all_labels = []
    all_attn_weights = []
    with torch.no_grad():
        for batch in data_loader:
            batch_emb, batch_labels = batch  # (bsz, num_patches, dim), (bsz, H, W)
            batch_emb = batch_emb.to(device)

            with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16):
                outputs = probe(batch_emb)  # (bsz, num_patches, logits_per_patch)
                logits = outputs["logits"]
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
            if probe_type == ProbeType.ATTNPOOL:
                all_attn_weights.append(outputs["attn_weights"])

    if probe_type == ProbeType.ATTNPOOL:
        all_attn_weights_tensor = torch.cat(all_attn_weights)
        per_head = all_attn_weights_tensor.mean(dim=(0, 2))  # → [heads, Num_bandsets]
        overall = all_attn_weights_tensor.mean(dim=(0, 1, 2))  # → [Num_bandsets]
        logger.info(f"overall: {overall.tolist()}")
        logger.info(f"per_head: {per_head.tolist()}")

    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)
    if task_type == TaskType.SEGMENTATION:
        metric = mean_iou(
            all_preds, all_labels, num_classes=num_classes, ignore_label=-1
        )
    else:
        metric = accuracy_score(all_labels, all_preds)
    return metric
