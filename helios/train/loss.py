"""Loss functions for training."""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from class_registry import ClassRegistry
from olmo_core.config import Config
from torch import Tensor

from helios.nn.flexihelios import TokensAndMasks
from helios.train.masking import MaskValue

logger = logging.getLogger(__name__)


class Loss(ABC):
    """Abstract base class for loss functions."""

    name: str

    @abstractmethod
    def compute(self, predictions: Any, targets: Any, **kwargs: Any) -> Tensor:
        """Compute the loss between predictions and targets."""
        pass

    @staticmethod
    def _expand_and_reciprocate(t: Tensor) -> Tensor:
        """As described in the name.

        >>> _expand_and_reciprocate(torch.tensor([1, 2, 3]))
        tensor([1.0000, 0.5000, 0.5000, 0.3333, 0.3333, 0.3333])
        """
        reciprocals = torch.reciprocal(t.float())
        return torch.repeat_interleave(reciprocals, t)


LOSS_REGISTRY = ClassRegistry[Loss]()


@LOSS_REGISTRY.register("patch_discrimination")
class PatchDiscriminationLoss(Loss):
    """Loss function for patch discrimination task."""

    name = "PatchDisc"

    def __init__(
        self,
        tau: float = 0.1,
        pred2unit: bool = False,
        mask_other_samples: bool = True,
    ):
        """Initialize patch discrimination loss.

        Args:
            tau: the softmax temperature
            pred2unit: whether to standardize the predictions using batch statistics
            mask_other_samples: whether to apply the contrastive loss drawing samples
                from within a sample (True) or using all other instances in a batch (False).
                If this is False, then this is the AllDisc loss from the Galileo paper
        """
        self.tau = tau
        self.pred2unit = pred2unit
        self.mask_other_samples = mask_other_samples

    def compute(
        self, predictions: TokensAndMasks, targets: TokensAndMasks, **kwargs: Any
    ) -> Tensor:
        """Compute patch discrimination loss between predictions and targets.

        Args:
            predictions: Model predictions.
            targets: Ground truth targets.
            **kwargs: Additional keyword arguments.

        Returns:
            The computed loss value.
        """
        all_preds, all_masks = predictions.flatten_tokens_and_masks()
        all_targets = targets.flatten_tokens_and_masks()[0]

        # Samples may have different number of tokens
        # TODO: Skip unqueeze and the for loop when mask_other_samples is True
        pred = all_preds[all_masks == MaskValue.DECODER.value].unsqueeze(dim=0)
        target = all_targets[all_masks == MaskValue.DECODER.value].unsqueeze(dim=0)
        bs, nt, _ = pred.shape

        if self.pred2unit:
            pred_mu = pred.mean(1, keepdims=True)
            pred_std = pred.std(1, keepdims=True)
            pred = (pred - pred_mu) / (pred_std + 1e-4)

        pred = F.normalize(pred, p=2, dim=-1)
        target = F.normalize(target, p=2, dim=-1)

        count = (all_masks == MaskValue.DECODER.value).sum(dim=-1)

        if self.mask_other_samples:
            # Compute scores per sample to reduce memory usage
            losses = []
            start = 0
            for c in count:
                end = start + c
                pred_sample = pred[:, start:end, :]
                target_sample = target[:, start:end, :]
                score_sample = (
                    torch.einsum("npd,nqd->npq", pred_sample, target_sample) / self.tau
                )
                labels = torch.arange(c, dtype=torch.long, device=pred.device)[None]
                loss = F.cross_entropy(
                    score_sample.flatten(0, 1),
                    labels.flatten(0, 1),
                    reduction="none",
                ) * (self.tau * 2)
                loss = loss.mean()
                losses.append(loss)
                start = end
            loss = torch.stack(losses).mean()
        else:
            # Compute scores for all samples
            scores = torch.einsum("npd,nqd->npq", pred, target) / self.tau
            logger.info(f"scores: {scores.shape}")
            labels = torch.arange(nt, dtype=torch.long, device=pred.device)[
                None
            ].repeat(bs, 1)
            loss = F.cross_entropy(
                scores.flatten(0, 1),
                labels.flatten(0, 1),
                reduction="none",
            ) * (self.tau * 2)
            # emulate averaging across the batch dimension
            loss_multiplier = self._expand_and_reciprocate(count)
            # can't use bs here since this is after the unsqueezing, so bs == 1
            loss = (loss * loss_multiplier).sum() / all_preds.shape[0]
        return loss


@LOSS_REGISTRY.register("l1")
class L1Loss(Loss):
    """Loss function for L1 (mean average error)."""

    name = "L1"

    def compute(
        self, predictions: TokensAndMasks, targets: TokensAndMasks, **kwargs: Any
    ) -> Tensor:
        """Compute L1 loss between predictions and targets.

        Args:
            predictions: Model predictions.
            targets: Ground truth targets.
            **kwargs: Additional keyword arguments.

        Returns:
            The computed loss value.
        """
        all_preds, all_masks = predictions.flatten_tokens_and_masks()
        all_targets = targets.flatten_tokens_and_masks()[0]
        pred = all_preds[all_masks == MaskValue.DECODER.value]
        target = all_targets[all_masks == MaskValue.DECODER.value]

        return F.l1_loss(pred, target)


@LOSS_REGISTRY.register("l2")
class L2Loss(Loss):
    """Loss function for L2 (mean squared error)."""

    name = "L2"

    def compute(
        self, predictions: TokensAndMasks, targets: TokensAndMasks, **kwargs: Any
    ) -> float:
        """Compute L2 loss between predictions and targets.

        Args:
            predictions: Model predictions.
            targets: Ground truth targets.
            **kwargs: Additional keyword arguments.

        Returns:
            The computed loss value.
        """
        all_preds, all_masks = predictions.flatten_tokens_and_masks()
        all_targets = targets.flatten_tokens_and_masks()[0]
        pred = all_preds[all_masks == MaskValue.DECODER.value]
        target = all_targets[all_masks == MaskValue.DECODER.value]
        return F.mse_loss(pred, target)


@LOSS_REGISTRY.register("cross_entropy")
class CrossEntropyLoss(Loss):
    """Loss function for cross entropy."""

    name = "CrossEntropy"

    def compute(
        self, predictions: TokensAndMasks, targets: TokensAndMasks, **kwargs: Any
    ) -> Tensor:
        """Compute cross entropy between predictions and targets.

        Args:
            predictions: Model predictions.
            targets: Ground truth targets.
            **kwargs: Additional keyword arguments.

        Returns:
            The computed loss value.
        """
        all_preds, all_masks = predictions.flatten_tokens_and_masks()
        all_targets = targets.flatten_tokens_and_masks()[0]
        pred = all_preds[all_masks == MaskValue.DECODER.value]
        target = all_targets[all_masks == MaskValue.DECODER.value]

        return F.cross_entropy(pred, target.squeeze())


@dataclass
class LossConfig(Config):
    """Configuration for loss functions.

    Args:
        loss_config: Loss config in the format of
        e.g.
        {
            "type": "patch_discrimination",
            # rest of init kwargs
    """

    loss_config: dict[str, Any]  # List of loss configs

    def build(self) -> Loss:
        """Build a Loss from the config."""
        loss_key = self.loss_config.pop("type")
        return LOSS_REGISTRY.get_class(loss_key)(**self.loss_config)
