"""Loss functions for training."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from class_registry import ClassRegistry
from einops import rearrange
from olmo_core.config import Config
from torch import Tensor

from helios.train.masking import MaskValue
from helios.train.model import TokensAndMasks


class Loss(ABC):
    """Abstract base class for loss functions."""

    @abstractmethod
    def compute(self, predictions: Any, targets: Any, **kwargs: Any) -> float:
        """Compute the loss between predictions and targets."""
        pass


LOSS_REGISTRY = ClassRegistry[Loss]()


@LOSS_REGISTRY.register("patch_discrimination")
class PatchDiscriminationLoss(Loss):
    """Loss function for patch discrimination task."""

    def __init__(
        self,
        tau: float = 0.07,
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

    @staticmethod
    def _flatten(x: Tensor) -> Tensor:
        if x.dim() == 6:
            # (B, C_G, T, P_H, P_W, D)
            return rearrange(x, "b c t h w d -> b (h w t c) d")
        elif x.dim() == 5:
            # (B, C_G, P_H, P_W, D)
            return rearrange(x, "b c h w d -> b (h w c) d")
        elif x.dim() == 4:
            # (B, C_G, T, D)
            return rearrange(x, "b c t d -> b (t c) d")
        elif x.dim() == 3:
            # (B, C_G, D)
            return x

    @staticmethod
    def _expand_and_reciprocate(t: Tensor) -> Tensor:
        """As described in the name.

        >>> _expand_and_reciprocate(torch.tensor([1, 2, 3]))
        tensor([1.0000, 0.5000, 0.5000, 0.3333, 0.3333, 0.3333])
        """
        reciprocals = torch.reciprocal(t.float())
        return torch.repeat_interleave(reciprocals, t)

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
        all_preds = torch.cat(
            [self._flatten(getattr(predictions, d)) for d in predictions.data_fields],
            dim=1,
        )
        all_masks = torch.cat(
            [
                self._flatten(getattr(predictions, f"{d}_mask").unsqueeze(dim=-1))
                for d in predictions.data_fields
            ],
            dim=1,
        )[:, :, 0]
        all_targets = torch.cat(
            [self._flatten(getattr(targets, d)) for d in predictions.data_fields], dim=1
        )
        pred = all_preds[all_masks == MaskValue.DECODER_ONLY.value].unsqueeze(dim=0)
        target = all_targets[all_masks == MaskValue.DECODER_ONLY.value].unsqueeze(dim=0)
        bs, nt, _ = pred.shape

        if self.pred2unit:
            pred_mu = pred.mean(1, keepdims=True)
            pred_std = pred.std(1, keepdims=True)
            pred = (pred - pred_mu) / (pred_std + 1e-4)

        pred = F.normalize(pred, p=2, dim=-1)
        target = F.normalize(target, p=2, dim=-1)

        scores = torch.einsum("npd,nqd->npq", pred, target) / self.tau
        count = (all_masks == 2).sum(dim=-1)

        if self.mask_other_samples:
            logit_mask = torch.full_like(scores, -torch.finfo(scores.dtype).max)
            start = 0
            for c in count:
                end = start + c
                logit_mask[:, start:end, start:end] = 0
                start += c

            scores = scores + logit_mask

        labels = torch.arange(nt, dtype=torch.long, device=pred.device)[None].repeat(
            bs, 1
        )
        loss = F.cross_entropy(
            scores.flatten(0, 1), labels.flatten(0, 1), reduction="none"
        ) * (self.tau * 2)

        # emulate averaging across the batch dimension
        loss_multiplier = self._expand_and_reciprocate(count)
        loss = (loss * loss_multiplier).sum() / bs
        return loss


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

    def build(self) -> type[Loss]:
        """Build a Loss from the config."""
        return LOSS_REGISTRY[self.loss_config["type"]](**self.loss_config)
