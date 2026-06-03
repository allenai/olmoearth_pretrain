"""Train and evaluate a linear probe."""

from __future__ import annotations

import copy
import functools
import math
from enum import StrEnum
from logging import getLogger

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from olmo_core.data.utils import get_rng
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from olmoearth_pretrain.evals.datasets.configs import EvalDatasetConfig, TaskType
from olmoearth_pretrain.evals.metrics import (
    SEGMENTATION_IGNORE_LABEL,
    EvalMetric,
    EvalResult,
    EvalTaskResult,
    classification_metrics,
    regression_metrics,
    segmentation_metrics,
)
from olmoearth_pretrain.evals.utils import adjust_learning_rate

logger = getLogger(__name__)


class MaskedMSELoss(nn.Module):
    """MSE loss that ignores NaN targets (used for sparse regression labels)."""

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute MSE only over finite (non-NaN) target pixels."""
        mask = torch.isfinite(targets)
        if not mask.any():
            return torch.tensor(0.0, device=predictions.device, requires_grad=True)
        return (predictions[mask] - targets[mask]).pow(2).mean()


def _compute_regression_target_stats(
    labels: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor] | tuple[None, None]:
    """Compute mean/std of finite regression labels for target normalization.

    Returns (None, None) when there are no finite labels to fit on, signalling
    callers to skip normalization entirely.
    """
    finite = labels[torch.isfinite(labels)]
    if finite.numel() == 0:
        return None, None
    mean = finite.mean()
    # Clamp std away from zero so degenerate constant-label datasets don't
    # produce inf/NaN after dividing.
    std = finite.std().clamp_min(1e-6)
    return mean, std


def _unnormalize_regression(
    t: torch.Tensor,
    mean: torch.Tensor | None,
    std: torch.Tensor | None,
) -> torch.Tensor:
    """Map a normalized regression tensor back to original target units."""
    if mean is None or std is None:
        return t
    return t * std.to(t.device, dtype=t.dtype) + mean.to(t.device, dtype=t.dtype)


class ProbeType(StrEnum):
    """Enumeration of probe types for linear probing."""

    ATTNPOOL = "attnpool"
    LINEAR = "linear"
    BILINEAR_CONV = "bilinear_conv"


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


class BilinearConvProbe(nn.Module):
    """Bilinear upsample + shared 1×1 conv probe for per-pixel regression.

    Instead of a per-patch linear that reshapes outputs blockwise (no spatial
    mixing, separate weights per sub-pixel slot), this probe:
      1. Rearranges (B, H, W, D) patch embeddings to (B, D, H, W).
      2. Bilinearly upsamples to the full label resolution (mixing neighbors).
      3. Applies a single Conv2d(D, out_channels, 1) shared across all pixels.

    This mirrors the rslearn Upsample+Conv head and is better regularized for
    sparse point-label regression tasks like LFMC.
    """

    def __init__(self, in_dim: int, scale_factor: int, out_channels: int = 1) -> None:
        """Initialize the bilinear conv probe.

        Args:
            in_dim: Embedding dimension per patch.
            scale_factor: Spatial upsample factor (typically patch_size).
            out_channels: Number of output channels (1 for regression).
        """
        super().__init__()
        self.upsample = nn.Upsample(
            scale_factor=scale_factor, mode="bilinear", align_corners=True
        )
        self.conv = nn.Conv2d(in_dim, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> dict:
        """Forward pass.

        Args:
            x: Patch embeddings of shape (B, H, W, D).

        Returns:
            Dict with 'logits' of shape (B, H*scale, W*scale) for single-channel
            regression, or (B, C, H*scale, W*scale) for multi-channel.
        """
        x = x.permute(0, 3, 1, 2)  # (B, D, H, W)
        x = self.upsample(x)  # (B, D, H', W')
        x = self.conv(x)  # (B, out_channels, H', W')
        if x.shape[1] == 1:
            x = x.squeeze(1)  # (B, H', W') for regression
        return {"logits": x}


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
    select_best_by_primary_metric: bool = False,
    n_bootstrap: int = 0,
    bootstrap_seed: int = 42,
    use_dice_loss: bool = False,
    primary_metric: EvalMetric | None = None,
    primary_metric_class: int | None = None,
) -> EvalTaskResult:
    """Run a linear probe on the OlmoEarth Pretrain model.

    Returns:
        Dictionary with keys:
            - val_score: EvalResult for validation
            - test_score: EvalResult for test, or None if no test set
            - bootstrap_stats: Bootstrap statistics dict (empty dict if n_bootstrap == 0)
    """
    logger.info(f"Probe type {probe_type}")
    if train_embeddings.shape[-1] != val_embeddings.shape[-1]:
        raise ValueError("Embedding dims don't match.")
    if test_embeddings is not None:
        if train_embeddings.shape[-1] != test_embeddings.shape[-1]:
            raise ValueError("Embedding dims don't match.")
    in_features = train_embeddings.shape[-1]

    # Z-score regression targets using train-set statistics so the MSE probe
    # is well-conditioned regardless of target units (e.g. raw LFMC values in
    # [0, 302]). Predictions and labels are un-normalized before any metric
    # is computed, so reported RMSE/MAE/R2 stay in original units. Stats are
    # kept on CPU; NaN-masked invalid pixels stay NaN through the affine map.
    target_mean: torch.Tensor | None = None
    target_std: torch.Tensor | None = None
    if config.task_type == TaskType.REGRESSION:
        target_mean, target_std = _compute_regression_target_stats(train_labels)
        if target_mean is not None and target_std is not None:
            logger.info(
                f"Normalizing regression targets with mean={target_mean.item():.4f}, "
                f"std={target_std.item():.4f}"
            )
            train_labels = (train_labels - target_mean) / target_std
            val_labels = (val_labels - target_mean) / target_std
            if test_labels is not None:
                test_labels = (test_labels - target_mean) / target_std
        else:
            logger.warning(
                "No finite regression labels found in train set; "
                "skipping target normalization."
            )
    output_pixels_per_side_of_patch = None
    if config.task_type in (TaskType.SEGMENTATION, TaskType.REGRESSION):
        assert config.height_width is not None, (
            "Height width is required for spatial probe tasks"
        )
        # if the image is resized the patch size will correspond to a different number of pixels in the labels
        # This normalizes the number of logits per patch to the number of label pixels each patch corresponds to
        num_patches = train_embeddings.shape[1] * train_embeddings.shape[2]
        output_pixels_per_side_of_patch = int(
            (config.height_width**2 / num_patches) ** 0.5
        )
        output_channels = (
            1 if config.task_type == TaskType.REGRESSION else config.num_classes
        )
        num_output_pixels = output_channels * output_pixels_per_side_of_patch**2
        logits_per_patch = num_output_pixels
        if config.task_type == TaskType.REGRESSION and probe_type == ProbeType.LINEAR:
            probe_type = ProbeType.BILINEAR_CONV
            logger.info(
                "Auto-upgrading probe from LINEAR to BILINEAR_CONV for regression "
                "(shared 1×1 conv with bilinear upsampling is better regularized "
                "for sparse per-pixel regression targets)."
            )
        if probe_type == ProbeType.BILINEAR_CONV:
            probe = BilinearConvProbe(
                in_dim=in_features,
                scale_factor=output_pixels_per_side_of_patch,
                out_channels=output_channels,
            ).to(device)
        elif probe_type == ProbeType.ATTNPOOL:
            if config.task_type == TaskType.REGRESSION:
                raise ValueError("Attention pooling is not supported for regression.")
            probe = AttnPoolLinearProbe(
                in_dim=in_features, out_dim=logits_per_patch
            ).to(device)
        elif probe_type == ProbeType.LINEAR:
            probe = LinearProbe(
                in_dim=in_features, out_dim=logits_per_patch, use_batchnorm=False
            ).to(device)
        else:
            raise ValueError(
                f"Probe type {probe_type} not supported for spatial tasks."
            )
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
    val_results: list[EvalResult] = []
    best_probe_state = None
    best_val_score = float("-inf")
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
            use_dice_loss=use_dice_loss,
        )
        val_result = evaluate_probe(
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
            primary_metric=primary_metric,
            primary_metric_class=primary_metric_class,
            target_mean=target_mean,
            target_std=target_std,
        )
        logger.info(f"Epoch {end_epoch}, Val Score: {val_result.primary}")
        val_results.append(val_result)

        # Save best probe state based on primary metric
        if val_result.primary > best_val_score:
            best_val_score = val_result.primary
            best_epoch = end_epoch
            best_probe_state = copy.deepcopy(probe.state_dict())

    # Log all validation results
    for i, val_result in enumerate(val_results):
        logger.debug(
            f"Epoch {(i + 1) * eval_interval}, Val Score: {val_result.primary}"
        )
    logger.debug(f"Best Val Score: {best_val_score} at epoch {best_epoch}")

    # Determine final validation result
    if select_best_by_primary_metric:
        # Find the result corresponding to best epoch
        best_idx = (best_epoch // eval_interval) - 1
        if best_idx < 0:
            best_idx = 0
        final_val_result = val_results[best_idx]
    else:
        final_val_result = val_results[-1]
        if final_val_result.primary < best_val_score:
            logger.warning(
                f"Final Val Score: {final_val_result.primary} at epoch {epochs} is less than best Val Score: "
                f"{best_val_score} at epoch {best_epoch}"
            )

    # Evaluate test set only once with the best probe
    test_result: EvalResult | None = None
    bootstrap_stats: dict = {}

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

        # Map regression preds/labels back to original target units so the
        # downstream metrics (and bootstrap resamples of them) are reported
        # on the same scale as the raw labels.
        if config.task_type == TaskType.REGRESSION:
            all_preds = _unnormalize_regression(all_preds, target_mean, target_std)
            all_labels = _unnormalize_regression(all_labels, target_mean, target_std)

        if n_bootstrap > 0:
            # Bootstrap resample the predictions (very fast!)
            rng = get_rng(bootstrap_seed)
            n_test_samples = all_preds.shape[0]
            bootstrap_scores: list[float] = []

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
                result = compute_metric(
                    bootstrap_preds,
                    bootstrap_labels,
                    num_classes=config.num_classes,
                    task_type=config.task_type,
                    primary_metric=primary_metric,
                    primary_metric_class=primary_metric_class,
                )
                bootstrap_scores.append(result.primary)

                if (i + 1) % 100 == 0:
                    logger.debug(
                        f"Bootstrap iteration {i + 1}/{n_bootstrap}, current mean: {np.mean(bootstrap_scores):.4f}"
                    )

            bootstrap_scores_array = np.array(bootstrap_scores)
            bootstrap_mean = float(np.mean(bootstrap_scores_array))
            std_metric = float(np.std(bootstrap_scores_array))
            ci_lower = float(np.percentile(bootstrap_scores_array, 2.5))
            ci_upper = float(np.percentile(bootstrap_scores_array, 97.5))
            bootstrap_stats = {
                "bootstrap_scores": bootstrap_scores_array.tolist(),
                "mean": bootstrap_mean,
                "std": std_metric,
                "ci_lower": ci_lower,
                "ci_upper": ci_upper,
            }
            logger.info(
                f"Bootstrap test score: {bootstrap_mean:.4f} ± {std_metric:.4f} "
                f"[{ci_lower:.4f}, {ci_upper:.4f}]"
            )
        # Compute full metrics for the actual test result
        test_result = compute_metric(
            all_preds,
            all_labels,
            num_classes=config.num_classes,
            task_type=config.task_type,
            primary_metric=primary_metric,
            primary_metric_class=primary_metric_class,
        )
        if n_bootstrap == 0:
            logger.info(f"Test result: {test_result}")

    return EvalTaskResult(
        val_result=final_val_result,
        test_result=test_result,
        bootstrap_stats=bootstrap_stats,
    )


def weighted_dice_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int,
    ignore_index: int = SEGMENTATION_IGNORE_LABEL,
    smooth: float = 1.0,
) -> torch.Tensor:
    """Compute class-weighted dice loss for segmentation.

    Args:
        logits: Model predictions of shape (N, C, ...) where C is num_classes.
        targets: Ground truth labels of shape (N, ...) with integer class indices.
        num_classes: Number of classes.
        ignore_index: Label value to ignore when computing loss.
        smooth: Smoothing term to avoid division by zero.

    Returns:
        Scalar weighted dice loss.
    """
    valid_mask = targets != ignore_index
    targets_masked = targets.clone()
    targets_masked[~valid_mask] = 0

    probs = F.softmax(logits, dim=1)
    one_hot = (
        F.one_hot(targets_masked, num_classes)
        .permute(0, -1, *range(1, targets.ndim))
        .float()
    )

    # Zero out ignored pixels in both probs and one_hot
    valid_mask_expanded = valid_mask.unsqueeze(1).expand_as(one_hot)
    probs = probs * valid_mask_expanded
    one_hot = one_hot * valid_mask_expanded

    # Per-class dice: sum over batch and spatial dims
    dims = (0,) + tuple(range(2, probs.ndim))
    intersection = (probs * one_hot).sum(dim=dims)
    cardinality = probs.sum(dim=dims) + one_hot.sum(dim=dims)

    dice_per_class = (2.0 * intersection + smooth) / (cardinality + smooth)

    # Class weights: inverse frequency of valid pixels per class
    class_counts = one_hot.sum(dim=dims)
    total = class_counts.sum()
    weights = torch.where(
        class_counts > 0,
        total / (num_classes * class_counts),
        torch.zeros_like(class_counts),
    )
    weights = weights / (weights.sum() + 1e-8)

    loss = 1.0 - (weights * dice_per_class).sum()
    return loss


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
    use_dice_loss: bool = False,
) -> nn.Module:
    """Train a linear probe on a classification or segmentation task."""
    opt = torch.optim.AdamW(probe.parameters(), lr=lr)

    probe = probe.train()
    loss_function: nn.Module | functools.partial
    if task_type == TaskType.REGRESSION:
        loss_function = MaskedMSELoss()
    elif use_dice_loss:
        loss_function = functools.partial(weighted_dice_loss, num_classes=num_classes)
    else:
        loss_function = nn.CrossEntropyLoss(ignore_index=SEGMENTATION_IGNORE_LABEL)
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
                    if not isinstance(probe, BilinearConvProbe):
                        assert num_output_pixels_per_side_of_patch is not None, (
                            "num_output_pixels_per_side_of_patch is required for segmentation"
                        )
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
                            )
                    targets = batch_labels.to(device)
                elif task_type == TaskType.REGRESSION:
                    if (
                        logits.shape[-2] != batch_labels.shape[-2]
                        or logits.shape[-1] != batch_labels.shape[-1]
                    ):
                        logits = F.interpolate(
                            logits.unsqueeze(1),
                            size=(batch_labels.shape[-2], batch_labels.shape[-1]),
                            mode="bilinear",
                            align_corners=True,
                        ).squeeze(1)
                    targets = batch_labels.to(device).float()
                else:
                    targets = batch_labels.to(device)
                loss = loss_function(logits, targets)

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
                is_bilinear_conv = isinstance(probe, BilinearConvProbe)
                if task_type == TaskType.SEGMENTATION and not is_bilinear_conv:
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
                        )

            if task_type == TaskType.REGRESSION:
                if (
                    logits.shape[-2] != batch_labels.shape[-2]
                    or logits.shape[-1] != batch_labels.shape[-1]
                ):
                    logits = F.interpolate(
                        logits.unsqueeze(1),
                        size=(batch_labels.shape[-2], batch_labels.shape[-1]),
                        mode="bilinear",
                        align_corners=True,
                    ).squeeze(1)
                preds = logits.float().cpu()
            else:
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
    primary_metric: EvalMetric | None = None,
    primary_metric_class: int | None = None,
) -> EvalResult:
    """Compute metric from predictions and labels."""
    if task_type == TaskType.SEGMENTATION:
        return segmentation_metrics(
            preds,
            labels,
            num_classes=num_classes,
            ignore_label=SEGMENTATION_IGNORE_LABEL,
            primary_metric=primary_metric,
            primary_metric_class=primary_metric_class,
        )
    if task_type == TaskType.REGRESSION:
        return regression_metrics(
            predictions=preds,
            labels=labels,
            primary_metric=primary_metric,
        )
    return classification_metrics(
        predictions=preds,
        labels=labels,
        primary_metric=primary_metric,
        primary_metric_class=primary_metric_class,
    )


def evaluate_probe(
    data_loader: DataLoader,
    probe: nn.Module,
    num_classes: int,
    device: torch.device,
    task_type: TaskType,
    probe_type: ProbeType,
    num_output_pixels_per_side_of_patch: int | None = None,
    primary_metric: EvalMetric | None = None,
    primary_metric_class: int | None = None,
    target_mean: torch.Tensor | None = None,
    target_std: torch.Tensor | None = None,
) -> EvalResult:
    """Evaluate a trained linear probe on a segmentation or classification task.

    For regression, ``target_mean``/``target_std`` are the train-set stats used
    to normalize labels before training; they're applied in reverse here so the
    reported metrics stay in original target units.
    """
    preds, labels = get_probe_predictions(
        data_loader=data_loader,
        probe=probe,
        num_classes=num_classes,
        device=device,
        task_type=task_type,
        probe_type=probe_type,
        num_output_pixels_per_side_of_patch=num_output_pixels_per_side_of_patch,
    )
    if task_type == TaskType.REGRESSION:
        preds = _unnormalize_regression(preds, target_mean, target_std)
        labels = _unnormalize_regression(labels, target_mean, target_std)
    return compute_metric(
        preds,
        labels,
        num_classes,
        task_type,
        primary_metric=primary_metric,
        primary_metric_class=primary_metric_class,
    )
