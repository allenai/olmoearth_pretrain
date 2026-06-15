"""Per-task supervised heads for the daily ERA5 encoder.

Each head consumes a pooled per-sample embedding produced by
`Era5DailyEncoder` and emits logits / predictions for one downstream task.
Heads are intentionally tiny (single Linear) — the encoder is the part that
actually learns transferable representations across tasks.

The `embedding_dim` is discovered on first forward (mirrors the pattern used
by `BackboneWithHead._init_head` in `evals/finetune/model.py`) so the
encoder's embedding size doesn't need to leak into the per-task configs.

`SupervisedHeadRegistry` groups heads by `TaskType` and exposes a single
`compute_loss` entry-point that the multi-objective train module calls per
microbatch / task.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from olmoearth_pretrain.evals.task_types import TaskType

logger = logging.getLogger(__name__)


SEGMENTATION_IGNORE_LABEL = -100


class _LazyLinearHead(nn.Module, ABC):
    """Base class for heads that defer the input dim to first forward."""

    def __init__(self, num_outputs: int) -> None:
        super().__init__()
        self.num_outputs = num_outputs
        # Placeholder; replaced on first call to `_ensure_initialized`.
        self._linear = nn.Linear(1, 1, bias=True)
        self._initialized = False

    def _ensure_initialized(self, embedding_dim: int, device: torch.device) -> None:
        if self._initialized:
            return
        self._linear = nn.Linear(embedding_dim, self.num_outputs, bias=True).to(device)
        self._initialized = True

    def forward(self, pooled: Tensor) -> Tensor:
        """Project a pooled embedding ``[B, D]`` to ``[B, num_outputs]``."""
        if pooled.ndim != 2:
            raise ValueError(
                f"Expected [B, D] pooled embedding, got shape {tuple(pooled.shape)}"
            )
        self._ensure_initialized(pooled.shape[-1], pooled.device)
        return self._linear(pooled)


class SupervisedHead(_LazyLinearHead, ABC):
    """A head that knows how to compute its own loss + a few simple metrics."""

    task_type: TaskType

    @abstractmethod
    def compute_loss(
        self, pooled: Tensor, labels: Tensor
    ) -> tuple[Tensor, dict[str, Tensor]]:
        """Compute loss and per-batch metrics for this head.

        Args:
            pooled: ``[B, D]`` pooled embedding from the encoder.
            labels: Task-specific labels (shape varies by task type).

        Returns:
            ``(loss, metrics_dict)`` where ``loss`` is a 0-d tensor and
            ``metrics_dict`` maps metric name -> 0-d tensor (already detached
            from the autograd graph).
        """


class RegressionHead(SupervisedHead):
    """Per-sample scalar / multi-output regression head.

    When ``target_mean`` and ``target_std`` are provided, labels are z-scored
    before computing the loss so that the gradient magnitude is independent of
    the raw target scale.  Auxiliary ``mse_raw`` / ``mae_raw`` metrics are
    reported in the original target units for interpretability.
    """

    task_type = TaskType.REGRESSION

    def __init__(
        self,
        num_outputs: int = 1,
        loss: str = "l2",
        target_mean: float | None = None,
        target_std: float | None = None,
    ) -> None:
        """Initialize with output dimension, loss type, and optional target stats."""
        super().__init__(num_outputs=num_outputs)
        if loss not in {"l1", "l2"}:
            raise ValueError(f"Unsupported regression loss: {loss}")
        self.loss = loss
        self.register_buffer(
            "_target_mean",
            torch.tensor(target_mean, dtype=torch.float32)
            if target_mean is not None
            else None,
        )
        self.register_buffer(
            "_target_std",
            torch.tensor(target_std, dtype=torch.float32)
            if target_std is not None
            else None,
        )

    @property
    def _normalizes_targets(self) -> bool:
        return self._target_mean is not None and self._target_std is not None

    def compute_loss(
        self, pooled: Tensor, labels: Tensor
    ) -> tuple[Tensor, dict[str, Tensor]]:
        """Mean-squared (or absolute) error over finite-valued labels."""
        preds = self.forward(pooled)
        if labels.ndim == 1:
            labels = labels.unsqueeze(-1)
        if labels.shape != preds.shape:
            raise ValueError(
                f"Regression label shape {tuple(labels.shape)} does not match "
                f"prediction shape {tuple(preds.shape)}"
            )
        labels = labels.to(preds.dtype)
        valid = torch.isfinite(labels)
        if not valid.any():
            zero = torch.zeros((), device=preds.device, dtype=preds.dtype)
            return zero, {"valid_fraction": zero.detach()}

        if self._normalizes_targets:
            labels = (labels - self._target_mean) / self._target_std

        diff = (preds - labels)[valid]
        if self.loss == "l1":
            loss = diff.abs().mean()
        else:
            loss = (diff**2).mean()

        with torch.no_grad():
            metrics: dict[str, Tensor] = {
                "mse": (diff**2).mean().detach(),
                "mae": diff.abs().mean().detach(),
                "valid_fraction": valid.float().mean().detach(),
            }
            if self._normalizes_targets:
                raw_diff = diff * self._target_std
                metrics["mse_raw"] = (raw_diff**2).mean().detach()
                metrics["mae_raw"] = raw_diff.abs().mean().detach()

        return loss, metrics


class ClassificationHead(SupervisedHead):
    """Multi-class classification head (single label per sample)."""

    task_type = TaskType.CLASSIFICATION

    def __init__(self, num_classes: int) -> None:
        """Initialize with the number of target classes (>= 2)."""
        if num_classes < 2:
            raise ValueError("ClassificationHead requires num_classes >= 2")
        super().__init__(num_outputs=num_classes)
        self.num_classes = num_classes

    def compute_loss(
        self, pooled: Tensor, labels: Tensor
    ) -> tuple[Tensor, dict[str, Tensor]]:
        """Standard cross-entropy with an ignore-index for invalid samples."""
        logits = self.forward(pooled)
        if labels.ndim > 1:
            labels = labels.view(-1)
        labels = labels.long()
        loss = F.cross_entropy(logits, labels, ignore_index=SEGMENTATION_IGNORE_LABEL)
        with torch.no_grad():
            valid = labels != SEGMENTATION_IGNORE_LABEL
            if valid.any():
                preds = logits.argmax(dim=-1)
                acc = (preds[valid] == labels[valid]).float().mean().detach()
            else:
                acc = torch.zeros((), device=logits.device)
            metrics = {
                "accuracy": acc,
                "valid_fraction": valid.float().mean().detach(),
            }
        return loss, metrics


class MultiLabelClassificationHead(SupervisedHead):
    """Multi-label classification (sigmoid per label, BCE-with-logits)."""

    task_type = TaskType.CLASSIFICATION

    def __init__(self, num_classes: int) -> None:
        """Initialize with the number of label classes (>= 1)."""
        if num_classes < 1:
            raise ValueError("MultiLabelClassificationHead requires num_classes >= 1")
        super().__init__(num_outputs=num_classes)
        self.num_classes = num_classes

    def compute_loss(
        self, pooled: Tensor, labels: Tensor
    ) -> tuple[Tensor, dict[str, Tensor]]:
        """BCE-with-logits over multi-hot labels.

        Ignores rows where the label tensor is all NaN (used to mark missing
        ground truth).
        """
        logits = self.forward(pooled)
        if labels.shape != logits.shape:
            raise ValueError(
                f"Multi-label label shape {tuple(labels.shape)} does not match "
                f"logits shape {tuple(logits.shape)}"
            )
        labels = labels.to(logits.dtype)
        # A row is "valid" if at least one element is finite.
        row_valid = torch.isfinite(labels).any(dim=-1)
        if not row_valid.any():
            zero = torch.zeros((), device=logits.device, dtype=logits.dtype)
            return zero, {"valid_fraction": zero.detach()}
        logits = logits[row_valid]
        labels = labels[row_valid]
        # Cells that are non-finite are replaced with 0 and masked out of the loss.
        cell_valid = torch.isfinite(labels)
        labels = torch.where(cell_valid, labels, torch.zeros_like(labels))
        loss = F.binary_cross_entropy_with_logits(
            logits, labels, weight=cell_valid.to(logits.dtype)
        )
        with torch.no_grad():
            probs = torch.sigmoid(logits)
            preds = (probs >= 0.5).to(labels.dtype)
            acc = (preds == labels)[cell_valid].float().mean().detach()
            metrics = {
                "accuracy": acc,
                "valid_fraction": row_valid.float().mean().detach(),
            }
        return loss, metrics


def build_head(
    task_type: TaskType | str,
    num_classes: int | None,
    is_multilabel: bool = False,
    regression_loss: str = "l2",
    target_mean: float | None = None,
    target_std: float | None = None,
) -> SupervisedHead:
    """Construct the right head for a given (task_type, is_multilabel) pair."""
    task_type = TaskType(task_type)
    if task_type == TaskType.REGRESSION:
        num_outputs = num_classes or 1
        return RegressionHead(
            num_outputs=num_outputs,
            loss=regression_loss,
            target_mean=target_mean,
            target_std=target_std,
        )
    if task_type == TaskType.CLASSIFICATION:
        if num_classes is None:
            raise ValueError("ClassificationHead requires num_classes")
        if is_multilabel:
            return MultiLabelClassificationHead(num_classes=num_classes)
        return ClassificationHead(num_classes=num_classes)
    raise ValueError(
        f"Unsupported task type for ERA5 supervised pretraining: {task_type}. "
        "Per-pixel segmentation tasks are not supported by the time-only "
        "ERA5 encoder."
    )


class SupervisedHeadRegistry(nn.Module):
    """A registry of per-task supervised heads.

    The registry is itself an `nn.Module` so the heads are tracked by the
    train module's optimizer / FSDP / DDP plumbing. Each head is registered
    by task name (any string the user picks — typically the ingest registry
    entry name like ``"lfmc"``).
    """

    def __init__(self) -> None:
        """Initialize an empty head registry."""
        super().__init__()
        self.heads = nn.ModuleDict()
        # Module-state mappings are stored separately so that ModuleDict
        # remains the source of truth for tracked parameters.
        self._task_types: dict[str, TaskType] = {}

    def register(self, task_name: str, head: SupervisedHead) -> None:
        """Register a head under ``task_name``."""
        if task_name in self.heads:
            raise KeyError(f"Task {task_name!r} already registered")
        self.heads[task_name] = head
        self._task_types[task_name] = head.task_type
        logger.info(
            "Registered supervised head for task=%s (type=%s, num_outputs=%d)",
            task_name,
            head.task_type.value,
            head.num_outputs,
        )

    def __contains__(self, task_name: str) -> bool:
        """Check whether a head is registered for *task_name*."""
        return task_name in self.heads

    def __getitem__(self, task_name: str) -> SupervisedHead:
        """Retrieve the head registered under *task_name*."""
        return self.heads[task_name]  # type: ignore[return-value]

    def task_type(self, task_name: str) -> TaskType:
        """Return the TaskType associated with ``task_name``."""
        return self._task_types[task_name]

    def compute_loss(
        self, task_name: str, pooled: Tensor, labels: Tensor
    ) -> tuple[Tensor, dict[str, Tensor]]:
        """Forward into the right head and compute (loss, metrics)."""
        if task_name not in self.heads:
            raise KeyError(
                f"No head registered for task {task_name!r}. "
                f"Available: {list(self.heads.keys())}"
            )
        head: SupervisedHead = self.heads[task_name]  # type: ignore[assignment]
        return head.compute_loss(pooled, labels)
