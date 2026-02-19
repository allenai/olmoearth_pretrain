"""Train and evaluate a linear probe."""

from __future__ import annotations

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
from olmoearth_pretrain.evals.metrics import (
    EvalResult,
    EvalTaskResult,
    segmentation_metrics,
)
from olmoearth_pretrain.evals.utils import adjust_learning_rate

logger = getLogger(__name__)


class ProbeType(StrEnum):
    """Enumeration of probe types for linear probing."""

    ATTNPOOL = "attnpool"
    LINEAR = "linear"
    INTERPOLATE_LINEAR = "interpolate_linear"


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
        return {"logits": logits}


PROBE_TYPE_TO_CLASS: dict[ProbeType, type[nn.Module]] = {
    ProbeType.LINEAR: LinearProbe,
    ProbeType.ATTNPOOL: AttnPoolLinearProbe,
    ProbeType.INTERPOLATE_LINEAR: InterpolateLinearProbe,
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
    select_final_test_miou_based_on_epoch_of_max_val_miou: bool = False,
    n_bootstrap: int = 0,
    bootstrap_seed: int = 42,
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
    if select_final_test_miou_based_on_epoch_of_max_val_miou:
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
            device=device,
            probe_type=probe_type,
        )

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
        )
        if n_bootstrap == 0:
            logger.info(f"Test result: {test_result}")

    return EvalTaskResult(
        val_result=final_val_result,
        test_result=test_result,
        bootstrap_stats=bootstrap_stats,
    )


def train_probe(
    data_loader: DataLoader,
    probe: nn.Module,
    lr: float,
    current_epoch: int,
    epochs: int,
    total_epochs: int,
    device: torch.device,
) -> nn.Module:
    """Train a linear probe on a classification or segmentation task."""
    opt = torch.optim.AdamW(probe.parameters(), lr=lr)

    probe = probe.train()
    loss_function = nn.CrossEntropyLoss(ignore_index=-1)  # for MADOS, but ok for others
    start_epoch = current_epoch
    for epoch in range(start_epoch, epochs):
        for i, batch in enumerate(data_loader):
            batch_emb, batch_labels = batch
            batch_emb = batch_emb.to(device)

            with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16):
                outputs = probe(batch_emb)
                logits = outputs["logits"]
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
    device: torch.device,
    probe_type: ProbeType,
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
            batch_emb, batch_labels = batch
            batch_emb = batch_emb.to(device)

            with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16):
                outputs = probe(batch_emb)
                logits = outputs["logits"]

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
) -> EvalResult:
    """Compute metric from predictions and labels.

    Args:
        preds: Predictions tensor
        labels: Labels tensor
        num_classes: Number of classes
        task_type: Type of task (classification or segmentation)

    Returns:
        EvalResult with computed metrics
    """
    if task_type == TaskType.SEGMENTATION:
        return segmentation_metrics(
            preds, labels, num_classes=num_classes, ignore_label=-1
        )
    else:
        acc = accuracy_score(labels.numpy(), preds.numpy())
        return EvalResult.from_classification(acc)


def evaluate_probe(
    data_loader: DataLoader,
    probe: nn.Module,
    num_classes: int,
    device: torch.device,
    task_type: TaskType,
    probe_type: ProbeType,
) -> EvalResult:
    """Evaluate a trained linear probe on a segmentation or classification task.

    Returns:
        EvalResult with computed metrics
    """
    preds, labels = get_probe_predictions(
        data_loader=data_loader,
        probe=probe,
        device=device,
        probe_type=probe_type,
    )
    return compute_metric(preds, labels, num_classes, task_type)
