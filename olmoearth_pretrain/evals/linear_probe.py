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
    metric_higher_is_better,
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
    INTERPOLATE_LINEAR = "interpolate_linear"
    BILINEAR_CONV = "bilinear_conv"


class AttnPoolLinearProbe(nn.Module):
    """Attention Pooling Linear Probe for segmentation tasks.

    Args:
        in_dim (int): Input feature dimension. Must be divisible by 64.
        num_classes (int): Number of output classes.
        task_type (TaskType): Must be SEGMENTATION.
        num_output_pixels_per_side_of_patch (int | None): Number of output pixels per side of each patch.

    Attributes:
        query_token (nn.Parameter): Learnable query token for attention pooling.
        num_heads (int): Number of attention heads.
        kv (nn.Linear): Linear layer to produce keys and values.
        linear (nn.Linear): Final linear layer for output logits.
    """

    def __init__(
        self,
        in_dim: int,
        num_classes: int,
        task_type: TaskType,
        num_output_pixels_per_side_of_patch: int | None = None,
    ) -> None:
        """Initialize the attention pooling linear probe."""
        super().__init__()
        if task_type != TaskType.SEGMENTATION:
            raise ValueError("AttnPoolLinearProbe only supports segmentation")
        if num_output_pixels_per_side_of_patch is None:
            raise ValueError(
                "num_output_pixels_per_side_of_patch is required for AttnPoolLinearProbe"
            )
        assert in_dim % 64 == 0, "in_dim must be divisible by 64"
        out_dim = num_classes * num_output_pixels_per_side_of_patch**2
        self.num_classes = num_classes
        self.num_output_pixels_per_side_of_patch = num_output_pixels_per_side_of_patch
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
            dict with:
                - "logits": Output logits, shape (B, C, H_out, W_out).
                - "attn_weights": Attention weights, shape (B*H*W, num_heads, 1, N).
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
        logits = self.linear(x)  # (B, H, W, out_dim)
        logits = rearrange(
            logits,
            "b h w (c i j) -> b c (h i) (w j)",
            c=self.num_classes,
            i=self.num_output_pixels_per_side_of_patch,
            j=self.num_output_pixels_per_side_of_patch,
        )
        return {"logits": logits, "attn_weights": attn_weights}


class InterpolateLinearProbe(nn.Module):
    """Probe that bilinear-interpolates embeddings to full resolution then applies a per-pixel linear layer.

    For segmentation only. Takes (B, H_p, W_p, D) embeddings, upsamples to
    (B, H_p * num_output_pixels_per_side_of_patch, ..., D), then applies Linear(D, num_classes).
    """

    def __init__(
        self,
        in_dim: int,
        num_classes: int,
        task_type: TaskType,
        num_output_pixels_per_side_of_patch: int | None = None,
    ) -> None:
        """Initialize the interpolate linear probe."""
        super().__init__()
        if task_type != TaskType.SEGMENTATION:
            raise ValueError("InterpolateLinearProbe only supports segmentation")
        if num_output_pixels_per_side_of_patch is None:
            raise ValueError(
                "num_output_pixels_per_side_of_patch is required for InterpolateLinearProbe"
            )
        self.linear = nn.Linear(in_dim, num_classes)
        self.num_output_pixels_per_side_of_patch = num_output_pixels_per_side_of_patch

    def forward(self, x: torch.Tensor) -> dict:
        """Forward pass: bilinear upsample embeddings, then per-pixel linear.

        Args:
            x: Embedding tensor of shape (B, H_p, W_p, D).

        Returns:
            dict with "logits" of shape (B, C, H, W).
        """
        B, H_p, W_p, D = x.shape
        target_hw = H_p * self.num_output_pixels_per_side_of_patch
        x = rearrange(x, "b h w d -> b d h w")
        x = F.interpolate(
            x,
            size=(target_hw, target_hw),
            mode="bilinear",
            align_corners=True,
        )
        x = rearrange(x, "b d h w -> b h w d")
        logits = self.linear(x)  # (B, target_hw, target_hw, C)
        logits = rearrange(logits, "b h w c -> b c h w")
        return {"logits": logits}


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

    def __init__(
        self,
        in_dim: int,
        num_classes: int,
        task_type: TaskType,
        num_output_pixels_per_side_of_patch: int | None = None,
    ) -> None:
        """Initialize the bilinear conv probe.

        Args:
            in_dim: Embedding dimension per patch.
            num_classes: Number of output classes (used for non-regression tasks).
            task_type: Task type; REGRESSION produces a single output channel.
            num_output_pixels_per_side_of_patch: Spatial upsample factor (typically
                patch_size).
        """
        super().__init__()
        if num_output_pixels_per_side_of_patch is None:
            raise ValueError(
                "num_output_pixels_per_side_of_patch is required for BilinearConvProbe"
            )
        out_channels = 1 if task_type == TaskType.PER_PIXEL_REGRESSION else num_classes
        self.upsample = nn.Upsample(
            scale_factor=num_output_pixels_per_side_of_patch,
            mode="bilinear",
            align_corners=True,
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
    """Linear Probe for classification and segmentation tasks.

    For classification: applies BatchNorm1d then Linear(D, num_classes).
    For segmentation: applies Linear(D, num_classes * ps^2) then rearranges to (B, C, H, W).
    """

    def __init__(
        self,
        in_dim: int,
        num_classes: int,
        task_type: TaskType,
        num_output_pixels_per_side_of_patch: int | None = None,
    ) -> None:
        """Initialize the linear probe."""
        super().__init__()
        self.task_type = task_type
        self.num_classes = num_classes
        self.num_output_pixels_per_side_of_patch = num_output_pixels_per_side_of_patch
        if task_type == TaskType.SEGMENTATION:
            assert num_output_pixels_per_side_of_patch is not None, (
                "num_output_pixels_per_side_of_patch is required for segmentation"
            )
            out_dim = num_classes * num_output_pixels_per_side_of_patch**2
            self.batchnorm: nn.Module = nn.Identity()
        else:
            out_dim = num_classes
            self.batchnorm = nn.BatchNorm1d(in_dim)
        self.linear = nn.Linear(in_dim, out_dim)

    def forward(self, x: torch.Tensor) -> dict:
        """Forward pass for linear probe."""
        logits = self.linear(self.batchnorm(x))
        if self.task_type == TaskType.SEGMENTATION:
            logits = rearrange(
                logits,
                "b h w (c i j) -> b c (h i) (w j)",
                c=self.num_classes,
                i=self.num_output_pixels_per_side_of_patch,
                j=self.num_output_pixels_per_side_of_patch,
            )
        elif self.task_type == TaskType.WINDOW_REGRESSION:
            # Single value per sample: (B, 1) -> (B,).
            logits = logits.squeeze(-1)
        return {"logits": logits}


PROBE_TYPE_TO_CLASS: dict[ProbeType, type[nn.Module]] = {
    ProbeType.LINEAR: LinearProbe,
    ProbeType.ATTNPOOL: AttnPoolLinearProbe,
    ProbeType.INTERPOLATE_LINEAR: InterpolateLinearProbe,
    ProbeType.BILINEAR_CONV: BilinearConvProbe,
}


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
    if config.task_type in (TaskType.PER_PIXEL_REGRESSION, TaskType.WINDOW_REGRESSION):
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
    if config.task_type in (TaskType.SEGMENTATION, TaskType.PER_PIXEL_REGRESSION):
        assert config.height_width is not None, (
            "Height width is required for spatial probe tasks"
        )
        # if the image is resized the patch size will correspond to a different number of pixels in the labels
        # This normalizes the number of logits per patch to the number of label pixels each patch corresponds to
        num_patches = train_embeddings.shape[1] * train_embeddings.shape[2]
        output_pixels_per_side_of_patch = int(
            (config.height_width**2 / num_patches) ** 0.5
        )

    # Regression auto-upgrades a plain LINEAR probe to the spatially-aware
    # BilinearConvProbe; attention pooling is not supported for regression.
    if (
        config.task_type == TaskType.PER_PIXEL_REGRESSION
        and probe_type == ProbeType.LINEAR
    ):
        probe_type = ProbeType.BILINEAR_CONV
        logger.info(
            "Auto-upgrading probe from LINEAR to BILINEAR_CONV for regression "
            "(shared 1×1 conv with bilinear upsampling is better regularized "
            "for sparse per-pixel regression targets)."
        )
    if probe_type == ProbeType.ATTNPOOL and config.task_type in (
        TaskType.PER_PIXEL_REGRESSION,
        TaskType.WINDOW_REGRESSION,
    ):
        raise ValueError("Attention pooling is not supported for regression.")

    probe_cls = PROBE_TYPE_TO_CLASS[probe_type]
    probe = probe_cls(
        in_dim=in_features,
        num_classes=config.num_classes,
        task_type=config.task_type,
        num_output_pixels_per_side_of_patch=output_pixels_per_side_of_patch,
    ).to(device)

    num_times_to_run_eval = math.ceil(epochs / eval_interval)
    val_results: list[EvalResult] = []
    best_probe_state = None
    # Direction is resolved from the actual primary metric below (lower is better
    # for error metrics like RMSE/MAE, higher for everything else).
    higher_is_better = True
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
            probe=probe,
            data_loader=data_loader,
            lr=lr,
            epochs=end_epoch,
            total_epochs=epochs,
            current_epoch=start_epoch,
            device=device,
            task_type=config.task_type,
            num_classes=config.num_classes,
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

        # Save best probe state based on primary metric (respecting its direction:
        # e.g. minimize RMSE/MAE, maximize accuracy/F1/R2).
        higher_is_better = metric_higher_is_better(val_result.primary_metric)
        if higher_is_better:
            improved = val_result.primary > best_val_score
        else:
            improved = val_result.primary < best_val_score
        if best_probe_state is None or improved:
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
        final_is_worse = (
            final_val_result.primary < best_val_score
            if higher_is_better
            else final_val_result.primary > best_val_score
        )
        if final_is_worse:
            logger.warning(
                f"Final Val Score: {final_val_result.primary} at epoch {epochs} is worse than best Val Score: "
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
        all_preds, all_labels, all_scores = get_probe_predictions(
            data_loader=test_data_loader,
            probe=probe,
            device=device,
            probe_type=probe_type,
            task_type=config.task_type,
        )

        # Map regression preds/labels back to original target units so the
        # downstream metrics (and bootstrap resamples of them) are reported
        # on the same scale as the raw labels.
        if config.task_type in (
            TaskType.PER_PIXEL_REGRESSION,
            TaskType.WINDOW_REGRESSION,
        ):
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
                bootstrap_iter_scores = (
                    all_scores[bootstrap_indices] if all_scores is not None else None
                )

                # Compute metric on resampled predictions
                result = compute_metric(
                    bootstrap_preds,
                    bootstrap_labels,
                    num_classes=config.num_classes,
                    task_type=config.task_type,
                    primary_metric=primary_metric,
                    primary_metric_class=primary_metric_class,
                    scores=bootstrap_iter_scores,
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
            scores=all_scores,
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
    """Compute class-weighted dice + cross-entropy loss for segmentation.

    Args:
        logits: Model predictions of shape (N, C, ...) where C is num_classes.
        targets: Ground truth labels of shape (N, ...) with integer class indices.
        num_classes: Number of classes.
        ignore_index: Label value to ignore when computing loss.
        smooth: Smoothing term to avoid division by zero.

    Returns:
        Scalar combined dice + CE loss.
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

    # Class weights: inverse frequency of valid pixels per class.
    # Use uniform weight for absent classes so the CE term still provides
    # gradient signal on batches where a rare class (e.g. flood) is absent.
    class_counts = one_hot.sum(dim=dims)
    total = class_counts.sum()
    weights = torch.where(
        class_counts > 0,
        total / (num_classes * class_counts),
        torch.ones_like(class_counts),
    )
    weights = weights / (weights.sum() + 1e-8)

    dice_loss = 1.0 - (weights * dice_per_class).sum()

    # CE loss provides gradient even when the minority class is absent from
    # the batch, preventing the model from collapsing to all-background.
    ce_loss = F.cross_entropy(logits, targets, ignore_index=ignore_index)

    return dice_loss + ce_loss


def train_probe(
    data_loader: DataLoader,
    probe: nn.Module,
    lr: float,
    current_epoch: int,
    epochs: int,
    total_epochs: int,
    device: torch.device,
    task_type: TaskType,
    num_classes: int,
    use_dice_loss: bool = False,
) -> nn.Module:
    """Train a linear probe on a classification or segmentation task."""
    opt = torch.optim.AdamW(probe.parameters(), lr=lr)

    probe = probe.train()
    loss_function: nn.Module | functools.partial
    if task_type in (TaskType.PER_PIXEL_REGRESSION, TaskType.WINDOW_REGRESSION):
        loss_function = MaskedMSELoss()
    elif use_dice_loss:
        loss_function = functools.partial(weighted_dice_loss, num_classes=num_classes)
    else:
        loss_function = nn.CrossEntropyLoss(ignore_index=SEGMENTATION_IGNORE_LABEL)
    start_epoch = current_epoch
    for epoch in range(start_epoch, epochs):
        for i, batch in enumerate(data_loader):
            batch_emb, batch_labels = batch
            batch_emb = batch_emb.to(device)

            with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16):
                # Probes return final-shaped logits (rearrange happens inside
                # each probe). Only interpolate to label resolution when the
                # probe output and label spatial shapes disagree (e.g. models
                # that require a fixed input image size).
                outputs = probe(batch_emb)
                logits = outputs["logits"]
                if task_type == TaskType.SEGMENTATION:
                    if logits.shape[-2:] != batch_labels.shape[-2:]:
                        logits = F.interpolate(
                            logits,
                            size=(batch_labels.shape[-2], batch_labels.shape[-1]),
                            mode="bilinear",
                            align_corners=True,
                        )
                    targets = batch_labels.to(device)
                elif task_type == TaskType.PER_PIXEL_REGRESSION:
                    if logits.shape[-2:] != batch_labels.shape[-2:]:
                        logits = F.interpolate(
                            logits.unsqueeze(1),
                            size=(batch_labels.shape[-2], batch_labels.shape[-1]),
                            mode="bilinear",
                            align_corners=True,
                        ).squeeze(1)
                    targets = batch_labels.to(device).float()
                elif task_type == TaskType.WINDOW_REGRESSION:
                    # Pooled (B,) prediction vs (B,) scalar target; no interp.
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
    device: torch.device,
    probe_type: ProbeType,
    task_type: TaskType,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Get predictions from a trained linear probe.

    Returns:
        tuple of (predictions, labels, scores). ``scores`` holds per-class
        softmax scores (shape (N, C) for classification, (N, C, H, W) for
        segmentation) and is None for regression.
    """
    probe = probe.eval()

    all_preds = []
    all_labels = []
    all_scores: list[torch.Tensor] = []
    all_attn_weights = []
    collect_scores = task_type not in (
        TaskType.PER_PIXEL_REGRESSION,
        TaskType.WINDOW_REGRESSION,
    )
    with torch.no_grad():
        for batch in data_loader:
            batch_emb, batch_labels = batch
            batch_emb = batch_emb.to(device)

            with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16):
                # Probes return final-shaped logits; only interpolate to the
                # label resolution when the spatial shapes disagree.
                outputs = probe(batch_emb)
                logits = outputs["logits"]
                if (
                    task_type == TaskType.SEGMENTATION
                    and logits.shape[-2:] != batch_labels.shape[-2:]
                ):
                    logits = F.interpolate(
                        logits,
                        size=(batch_labels.shape[-2], batch_labels.shape[-1]),
                        mode="bilinear",
                        align_corners=True,
                    )

            if task_type == TaskType.WINDOW_REGRESSION:
                preds = logits.float().cpu()
            elif task_type == TaskType.PER_PIXEL_REGRESSION:
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
                if collect_scores:
                    all_scores.append(torch.softmax(logits.float(), dim=1).cpu())
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
    scores = torch.cat(all_scores) if collect_scores else None
    return all_preds, all_labels, scores


def compute_metric(
    preds: torch.Tensor,
    labels: torch.Tensor,
    num_classes: int,
    task_type: TaskType,
    primary_metric: EvalMetric | None = None,
    primary_metric_class: int | None = None,
    scores: torch.Tensor | None = None,
) -> EvalResult:
    """Compute metric from predictions and labels."""
    if task_type == TaskType.SEGMENTATION:
        return segmentation_metrics(
            preds,
            labels,
            num_classes=num_classes,
            ignore_label=SEGMENTATION_IGNORE_LABEL,
            scores=scores,
            primary_metric=primary_metric,
            primary_metric_class=primary_metric_class,
        )
    if (
        task_type == TaskType.PER_PIXEL_REGRESSION
        or task_type == TaskType.WINDOW_REGRESSION
    ):
        return regression_metrics(
            predictions=preds,
            labels=labels,
            primary_metric=primary_metric,
        )
    return classification_metrics(
        predictions=preds,
        labels=labels,
        scores=scores,
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
    preds, labels, scores = get_probe_predictions(
        data_loader=data_loader,
        probe=probe,
        device=device,
        probe_type=probe_type,
        task_type=task_type,
    )
    if task_type in (TaskType.PER_PIXEL_REGRESSION, TaskType.WINDOW_REGRESSION):
        preds = _unnormalize_regression(preds, target_mean, target_std)
        labels = _unnormalize_regression(labels, target_mean, target_std)
    return compute_metric(
        preds,
        labels,
        num_classes,
        task_type,
        primary_metric=primary_metric,
        primary_metric_class=primary_metric_class,
        scores=scores,
    )
