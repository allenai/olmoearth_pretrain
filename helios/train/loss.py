"""Loss functions for training."""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from class_registry import ClassRegistry
from einops import rearrange
from olmo_core.config import Config
from torch import Tensor

from helios.nn.flexihelios import TokensAndMasks
from helios.train.masking import MaskedHeliosSample, MaskValue

logger = logging.getLogger(__name__)


class Loss(ABC):
    """Abstract base class for loss functions."""

    @staticmethod
    def _flatten(x: Tensor) -> Tensor:
        return rearrange(x, "b ... d -> b (...) d")

    @abstractmethod
    def compute(self, predictions: Any, targets: Any, **kwargs: Any) -> float:
        """Compute the loss between predictions and targets."""
        pass

    @classmethod
    def _flatten_tokens_or_masks(
        cls, x: TokensAndMasks, is_masks: bool = False
    ) -> Tensor:
        flattened_attrs = []
        for attr_name in x.modalities:
            if is_masks:
                attr_name = MaskedHeliosSample.get_masked_modality_name(attr_name)
            attr = getattr(x, attr_name)
            if attr is not None:
                if is_masks:
                    attr = attr.unsqueeze(dim=-1)
                flattened_attr = cls._flatten(attr)
                flattened_attrs.append(flattened_attr)

        flattened_x = torch.cat(flattened_attrs, dim=1)
        return flattened_x[:, :, 0] if is_masks else flattened_x

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
    ) -> float:
        """Compute patch discrimination loss between predictions and targets.

        Args:
            predictions: Model predictions.
            targets: Ground truth targets.
            **kwargs: Additional keyword arguments.

        Returns:
            The computed loss value.
        """
        # TODO: write a function that deals with this
        all_preds = self._flatten_tokens_or_masks(predictions)
        all_masks = self._flatten_tokens_or_masks(predictions, is_masks=True)
        all_targets = self._flatten_tokens_or_masks(targets)

        pred = all_preds[all_masks == MaskValue.DECODER.value].unsqueeze(dim=0)
        target = all_targets[all_masks == MaskValue.DECODER.value].unsqueeze(dim=0)
        bs, nt, _ = pred.shape

        if self.pred2unit:
            pred_mu = pred.mean(1, keepdims=True)
            pred_std = pred.std(1, keepdims=True)
            pred = (pred - pred_mu) / (pred_std + 1e-4)

        pred = F.normalize(pred, p=2, dim=-1)
        target = F.normalize(target, p=2, dim=-1)

        scores = torch.einsum("npd,nqd->npq", pred, target) / self.tau
        count = (all_masks == MaskValue.DECODER.value).sum(dim=-1)

        if self.mask_other_samples:
            logit_mask = torch.full_like(scores, -torch.finfo(scores.dtype).max)
            start = 0
            for c in count:
                end = start + c
                logit_mask[:, start:end, start:end] = 0
                start += c
            logger.info(f"logit_mask: {logit_mask.shape}")
            logger.info(f"scores: {scores.shape}")
            scores = scores + logit_mask

        labels = torch.arange(nt, dtype=torch.long, device=pred.device)[None].repeat(
            bs, 1
        )
        loss = F.cross_entropy(
            scores.flatten(0, 1), labels.flatten(0, 1), reduction="none"
        ) * (self.tau * 2)

        # emulate averaging across the batch dimension
        loss_multiplier = self._expand_and_reciprocate(count)
        # can't use bs here since this is after the unsqueezing, so bs == 1
        loss = (loss * loss_multiplier).sum() / all_preds.shape[0]
        return loss


@LOSS_REGISTRY.register("l1")
class L1Loss(Loss):
    """Loss function for L1 (mean average error)."""

    def compute(
        self, predictions: TokensAndMasks, targets: TokensAndMasks, **kwargs: Any
    ) -> float:
        """Compute L1 loss between predictions and targets.

        Args:
            predictions: Model predictions.
            targets: Ground truth targets.
            **kwargs: Additional keyword arguments.

        Returns:
            The computed loss value.
        """
        all_preds = self._flatten_tokens_or_masks(predictions)
        all_masks = self._flatten_tokens_or_masks(predictions, is_masks=True)
        all_targets = self._flatten_tokens_or_masks(targets)
        pred = all_preds[all_masks == MaskValue.DECODER.value]
        target = all_targets[all_masks == MaskValue.DECODER.value]

        return F.l1_loss(pred, target)


@LOSS_REGISTRY.register("l2")
class L2Loss(Loss):
    """Loss function for L2 (mean squared error)."""

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
        all_preds = self._flatten_tokens_or_masks(predictions)
        all_masks = self._flatten_tokens_or_masks(predictions, is_masks=True)
        all_targets = self._flatten_tokens_or_masks(targets)
        pred = all_preds[all_masks == MaskValue.DECODER.value]
        target = all_targets[all_masks == MaskValue.DECODER.value]

        return F.mse_loss(pred, target)


@LOSS_REGISTRY.register("cross_entropy")
class CrossEntropyLoss(Loss):
    """Loss function for cross entropy."""

    def compute(
        self, predictions: TokensAndMasks, targets: TokensAndMasks, **kwargs: Any
    ) -> float:
        """Compute cross entropy between predictions and targets.

        Args:
            predictions: Model predictions.
            targets: Ground truth targets.
            **kwargs: Additional keyword arguments.

        Returns:
            The computed loss value.
        """
        all_preds = self._flatten_tokens_or_masks(predictions)
        all_masks = self._flatten_tokens_or_masks(predictions, is_masks=True)
        all_targets = self._flatten_tokens_or_masks(targets)
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
