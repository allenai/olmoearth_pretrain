"""Train and evaluate a linear probe."""

import copy
import math
from enum import StrEnum
from logging import getLogger

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from olmo_core.data.utils import get_rng
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from olmoearth_pretrain.evals.datasets.configs import EvalDatasetConfig, TaskType
from olmoearth_pretrain.evals.metrics import mean_iou
from olmoearth_pretrain.evals.utils import adjust_learning_rate

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
    val_embeddings: torch.Tensor,
    val_labels: torch.Tensor,
    test_embeddings: torch.Tensor | None,
    test_labels: torch.Tensor | None,
    device: torch.device,
    batch_size: int,
    epochs: int = 50,
    eval_interval: int = 50,
    probe_type: ProbeType = ProbeType.LINEAR,
    select_final_test_miou_based_on_epoch_of_max_val_miou: bool = False,
    n_bootstrap: int = 0,
    bootstrap_seed: int = 42,
) -> dict[str, float | dict]:
    """Run a linear probe on the OlmoEarth Pretrain model.

    Returns:
        Dictionary with keys:
            - val_score: Validation score
            - test_score: Test score (0.0 if no test set)
            - bootstrap_stats: Bootstrap statistics dict (empty dict if n_bootstrap == 0)
    """
    logger.info(f"Probe type {probe_type}")
    if train_embeddings.shape[-1] != val_embeddings.shape[-1]:
        raise ValueError("Embedding dims don't match.")
    if test_embeddings is not None:
        if train_embeddings.shape[-1] != test_embeddings.shape[-1]:
            raise ValueError("Embedding dims don't match.")
    in_features = train_embeddings.shape[-1]
    output_pixels_per_side_of_patch = None
    if config.task_type == TaskType.SEGMENTATION:
        assert config.height_width is not None, (
            "Height width is required for segmentation"
        )
        # if the image is resized the patch size will correspond to a different number of pixels in the labels
        # This normalizes the number of logits per patch to the number of label pixels each patch corresponds to
        num_patches = train_embeddings.shape[1] * train_embeddings.shape[2]
        output_pixels_per_side_of_patch = int(
            (config.height_width**2 / num_patches) ** 0.5
        )
        num_output_pixels = config.num_classes * output_pixels_per_side_of_patch**2
        logits_per_patch = num_output_pixels
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
    val_mious = []
    best_probe_state = None
    best_val_miou = float("-inf")
    best_epoch = 0

    data_loader = DataLoader(
        TensorDataset(train_embeddings, train_labels),
        batch_size=batch_size,
        shuffle=True,
    )
    # Training loop: only evaluate on validation set
    for i in range(num_times_to_run_eval):
        start_epoch = i * eval_interval
        end_epoch = min(start_epoch + eval_interval, epochs)

        probe = train_probe(
            task_type=config.task_type,
            probe=probe,
            data_loader=data_loader,
            lr=lr,
            epochs=end_epoch,
            total_epochs=epochs,
            current_epoch=start_epoch,
            num_classes=config.num_classes,
            num_output_pixels_per_side_of_patch=output_pixels_per_side_of_patch,
            device=device,
        )
        val_miou = evaluate_probe(
            data_loader=DataLoader(
                TensorDataset(val_embeddings, val_labels),
                batch_size=batch_size,
                shuffle=False,
            ),
            probe=probe,
            num_classes=config.num_classes,
            num_output_pixels_per_side_of_patch=output_pixels_per_side_of_patch,
            device=device,
            task_type=config.task_type,
            probe_type=probe_type,
        )
        logger.info(f"Epoch {end_epoch}, Val MIoU: {val_miou}")
        val_mious.append(val_miou)

        # Save best probe state
        if val_miou > best_val_miou:
            best_val_miou = val_miou
            best_epoch = end_epoch
            best_probe_state = copy.deepcopy(probe.state_dict())

    # Log all validation results
    for i, val_miou in enumerate(val_mious):
        logger.debug(f"Epoch {(i + 1) * eval_interval}, Val MIoU: {val_miou}")
    logger.debug(f"Best Val MIoU: {best_val_miou} at epoch {best_epoch}")

    # Determine final validation MIoU
    if select_final_test_miou_based_on_epoch_of_max_val_miou:
        val_miou = best_val_miou
    else:
        val_miou = val_mious[-1]
        if val_miou < best_val_miou:
            logger.warning(
                f"Final Val MIoU: {val_miou} at epoch {epochs} is less than best Val MIoU: "
                f"{best_val_miou} at epoch {best_epoch}"
            )

    # Evaluate test set only once with the best probe
    if test_embeddings is not None:
        if test_labels is None:
            raise ValueError("Can't have test embeddings without test labels")

        # Load best probe state
        if best_probe_state is not None:
            probe.load_state_dict(best_probe_state)
            logger.info(f"Evaluating test set with best probe (epoch {best_epoch})")

        # Compute predictions once (regardless of bootstrap)
        logger.info(
            f"Computing predictions for {test_embeddings.shape[0]} test samples..."
        )
        test_data_loader = DataLoader(
            TensorDataset(test_embeddings, test_labels),
            batch_size=batch_size,
            shuffle=False,
        )
        all_preds, all_labels = get_probe_predictions(
            data_loader=test_data_loader,
            probe=probe,
            num_classes=config.num_classes,
            device=device,
            task_type=config.task_type,
            probe_type=probe_type,
            num_output_pixels_per_side_of_patch=output_pixels_per_side_of_patch,
        )

        bootstrap_stats = None
        if n_bootstrap > 0:
            # Bootstrap resample the predictions (very fast!)
            rng = get_rng(bootstrap_seed)
            n_test_samples = all_preds.shape[0]
            bootstrap_scores = []

            logger.info(
                f"Running {n_bootstrap} bootstrap iterations on precomputed predictions..."
            )

            for i in tqdm(range(n_bootstrap), desc="Bootstrapping", leave=False):
                # Resample indices only - no model forward pass!
                bootstrap_indices = rng.choice(
                    n_test_samples, size=n_test_samples, replace=True
                )

                bootstrap_preds = all_preds[bootstrap_indices]
                bootstrap_labels = all_labels[bootstrap_indices]

                # Compute metric on resampled predictions
                score = compute_metric(
                    bootstrap_preds,
                    bootstrap_labels,
                    num_classes=config.num_classes,
                    task_type=config.task_type,
                )
                bootstrap_scores.append(score)

                if (i + 1) % 100 == 0:
                    logger.debug(
                        f"Bootstrap iteration {i + 1}/{n_bootstrap}, current mean: {np.mean(bootstrap_scores):.4f}"
                    )

            bootstrap_scores_array = np.array(bootstrap_scores)
            test_miou = float(np.mean(bootstrap_scores_array))
            std_metric = float(np.std(bootstrap_scores_array))
            ci_lower = float(np.percentile(bootstrap_scores_array, 2.5))
            ci_upper = float(np.percentile(bootstrap_scores_array, 97.5))
            bootstrap_stats = {
                "bootstrap_scores": bootstrap_scores_array.tolist(),
                "mean": test_miou,
                "std": std_metric,
                "ci_lower": ci_lower,
                "ci_upper": ci_upper,
            }
            logger.info(
                f"Bootstrap test MIoU: {test_miou:.4f} ± {std_metric:.4f} "
                f"[{ci_lower:.4f}, {ci_upper:.4f}]"
            )
        else:
            # No bootstrap - just compute metric from predictions
            test_miou = compute_metric(
                all_preds,
                all_labels,
                num_classes=config.num_classes,
                task_type=config.task_type,
            )
            bootstrap_stats = {}
            logger.info(f"Test MIoU: {test_miou}")
    else:
        test_miou = 0.0
        bootstrap_stats = {}

    return {
        "val_score": val_miou,
        "test_score": test_miou,
        "bootstrap_stats": bootstrap_stats,
    }


def train_probe(
    data_loader: DataLoader,
    probe: nn.Module,
    lr: float,
    current_epoch: int,
    epochs: int,
    total_epochs: int,
    num_classes: int,
    device: torch.device,
    task_type: TaskType,
    num_output_pixels_per_side_of_patch: int | None = None,
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
                # logger.info(f"logits: {logits.shape}")
                if task_type == TaskType.SEGMENTATION:
                    assert num_output_pixels_per_side_of_patch is not None, (
                        "num_output_pixels_per_side_of_patch is required for segmentation"
                    )
                    # This is effectively nearest neighbor interpolation
                    logits = rearrange(
                        logits,
                        "b h w (c i j) -> b c (h i) (w j)",
                        h=spatial_patches_per_dim,
                        w=spatial_patches_per_dim,
                        c=num_classes,
                        i=num_output_pixels_per_side_of_patch,
                        j=num_output_pixels_per_side_of_patch,
                    )
                    if logits.shape[-2] != batch_labels.shape[-2]:
                        logger.debug(
                            f"Logits shape {logits.shape} does not match batch_labels shape {batch_labels.shape} interpolating to labels shape"
                        )
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

    return probe


def get_probe_predictions(
    data_loader: DataLoader,
    probe: nn.Module,
    num_classes: int,
    device: torch.device,
    task_type: TaskType,
    probe_type: ProbeType,
    num_output_pixels_per_side_of_patch: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Get predictions from a trained linear probe.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: (predictions, labels)
    """
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
                    assert num_output_pixels_per_side_of_patch is not None, (
                        "num_output_pixels_per_side_of_patch is required for segmentation"
                    )
                    spatial_patches_per_dim = batch_emb.shape[1]
                    logits = rearrange(
                        logits,
                        "b h w (c i j) -> b c (h i) (w j)",
                        h=spatial_patches_per_dim,
                        w=spatial_patches_per_dim,
                        c=num_classes,
                        i=num_output_pixels_per_side_of_patch,
                        j=num_output_pixels_per_side_of_patch,
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
    return all_preds, all_labels


def compute_metric(
    preds: torch.Tensor,
    labels: torch.Tensor,
    num_classes: int,
    task_type: TaskType,
) -> float:
    """Compute metric from predictions and labels.

    Args:
        preds: Predictions tensor
        labels: Labels tensor
        num_classes: Number of classes
        task_type: Type of task (classification or segmentation)

    Returns:
        float: Computed metric (accuracy for classification, mIoU for segmentation)
    """
    if task_type == TaskType.SEGMENTATION:
        metric = mean_iou(preds, labels, num_classes=num_classes, ignore_label=-1)
    else:
        metric = accuracy_score(labels.numpy(), preds.numpy())
    return metric


def evaluate_probe(
    data_loader: DataLoader,
    probe: nn.Module,
    num_classes: int,
    device: torch.device,
    task_type: TaskType,
    probe_type: ProbeType,
    num_output_pixels_per_side_of_patch: int | None = None,
) -> float:
    """Evaluate a trained linear probe on a segmentation or classification task."""
    preds, labels = get_probe_predictions(
        data_loader=data_loader,
        probe=probe,
        num_classes=num_classes,
        device=device,
        task_type=task_type,
        probe_type=probe_type,
        num_output_pixels_per_side_of_patch=num_output_pixels_per_side_of_patch,
    )
    return compute_metric(preds, labels, num_classes, task_type)
