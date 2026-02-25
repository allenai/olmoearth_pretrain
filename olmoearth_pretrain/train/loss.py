"""Loss functions for training."""

import logging
import math
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from class_registry import ClassRegistry
from einops import rearrange, repeat
from torch import Tensor

from olmoearth_pretrain.config import Config
from olmoearth_pretrain.nn.flexi_vit import TokensAndMasks
from olmoearth_pretrain.nn.pooling import PoolingType
from olmoearth_pretrain.nn.tokenization import TokenizationConfig
from olmoearth_pretrain.train.masking import MaskedOlmoEarthSample, MaskValue

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


@LOSS_REGISTRY.register("all_discrimination")
class AllDiscriminationLoss(Loss):
    """Loss function for all discrimination task.

    Discriminates across patches using all samples in a batch.
    """

    name = "AllDisc"

    def __init__(self, tau: float = 0.1, pred2unit: bool = False):
        """Initialize all patch discrimination loss.

        Args:
            tau: the softmax temperature
            pred2unit: whether to standardize the predictions using batch statistics
        """
        self.tau = tau
        self.pred2unit = pred2unit

    def compute(
        self, predictions: TokensAndMasks, targets: TokensAndMasks, **kwargs: Any
    ) -> Tensor:
        """Compute all patch discrimination loss between predictions and targets.

        Args:
            predictions: Model predictions.
            targets: Ground truth targets.
            **kwargs: Additional keyword arguments.

        Returns:
            The computed loss value.
        """
        all_preds, all_masks = predictions.flatten_all_tokens_and_masks()
        all_targets = targets.flatten_all_tokens_and_masks()[0]

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


@LOSS_REGISTRY.register("modality_all_discrimination")
class ModalityAllDiscriminationLoss(Loss):
    """Loss function for all discrimination task.

    Discriminates across patches using all samples in a batch.
    """

    name = "ModalityAllDisc"

    def __init__(self, tau: float = 0.1, pred2unit: bool = False):
        """Initialize all patch discrimination loss.

        Args:
            tau: the softmax temperature
            pred2unit: whether to standardize the predictions using batch statistics
        """
        self.tau = tau
        self.pred2unit = pred2unit

    def compute(
        self, predictions: TokensAndMasks, targets: TokensAndMasks, **kwargs: Any
    ) -> Tensor:
        """Compute all patch discrimination loss between predictions and targets.

        Args:
            predictions: Model predictions.
            targets: Ground truth targets.
            **kwargs: Additional keyword arguments.

        Returns:
            The computed loss value.
        """
        modality_preds, modality_masks = (
            predictions.flatten_tokens_and_masks_per_modality()
        )
        modality_targets = targets.flatten_tokens_and_masks_per_modality()[0]

        total_loss = 0
        for all_preds, all_masks, all_targets in zip(
            modality_preds, modality_masks, modality_targets
        ):
            pred = all_preds[all_masks == MaskValue.DECODER.value].unsqueeze(dim=0)
            target = all_targets[all_masks == MaskValue.DECODER.value].unsqueeze(dim=0)
            bs, nt, _ = pred.shape
            if nt == 0:
                # If no decoded values, skip this modality
                logger.warning("No decoded values for this modality")
                continue
            if self.pred2unit:
                pred_mu = pred.mean(1, keepdims=True)
                pred_std = pred.std(1, keepdims=True)
                pred = (pred - pred_mu) / (pred_std + 1e-4)

            pred = F.normalize(pred, p=2, dim=-1)
            target = F.normalize(target, p=2, dim=-1)

            scores = torch.einsum("npd,nqd->npq", pred, target) / self.tau
            count = (all_masks == MaskValue.DECODER.value).sum(dim=-1)

            labels = torch.arange(nt, dtype=torch.long, device=pred.device)[
                None
            ].repeat(bs, 1)
            loss = F.cross_entropy(
                scores.flatten(0, 1), labels.flatten(0, 1), reduction="none"
            ) * (self.tau * 2)

            # emulate averaging across the batch dimension
            loss_multiplier = self._expand_and_reciprocate(count)
            # can't use bs here since this is after the unsqueezing, so bs == 1
            loss = (loss * loss_multiplier).sum() / all_preds.shape[0]
            total_loss += loss

        return total_loss


@LOSS_REGISTRY.register("patch_discrimination")
class PatchDiscriminationLoss(Loss):
    """Loss function for patch discrimination task.

    Memory-efficient per-sample contrastive loss. Computes similarity matrices
    per sample rather than across the full batch.
    """

    name = "PatchDisc"

    def __init__(self, tau: float = 0.1, pred2unit: bool = False, weight: float = 1.0):
        """Initialize patch discrimination loss.

        Args:
            tau: the softmax temperature
            pred2unit: whether to standardize the predictions using batch statistics
            mask_other_samples: whether to apply the contrastive loss drawing samples
                from within a sample (True) or using all other instances in a batch (False).
                If this is False, then this is the AllDisc loss from the Galileo paper
            weight: the weight to apply to this loss
        """
        self.tau = tau
        self.pred2unit = pred2unit
        self.weight = weight

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
        all_preds, all_masks = predictions.flatten_all_tokens_and_masks()
        all_targets = targets.flatten_all_tokens_and_masks()[0]

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
        losses = []
        start = 0
        for c in count:
            end = start + c
            if c == 0:
                # we will occasionally get a sample with no decoded values due to missing data this will let us skip it
                logger.warning("No decoded values for this sample")
                continue
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
        return self.weight * loss


@LOSS_REGISTRY.register("modality_patch_discrimination")
class ModalityPatchDiscriminationLoss(Loss):
    """Loss function for per-modality patch discrimination task.

    Memory-efficient per-sample contrastive loss. Computes similarity matrices
    per sample rather than across the full batch, independently for each modality.
    """

    name = "ModalityPatchDisc"

    def __init__(
        self,
        tau: float = 0.1,
        pred2unit: bool = False,
        weight: float = 1.0,
        modality_weights: dict[str, float] | None = None,
    ):
        """Initialize patch discrimination loss.

        Args:
            tau: the softmax temperature
            pred2unit: whether to standardize the predictions using batch statistics
            mask_other_samples: whether to apply the contrastive loss drawing samples
                from within a sample (True) or using all other instances in a batch (False).
                If this is False, then this is the AllDisc loss from the Galileo paper
            weight: the weight to apply to this loss
            modality_weights: the weights to apply to each modality
        """
        self.tau = tau
        self.pred2unit = pred2unit
        self.weight = weight
        self.modality_weights = modality_weights

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
        modality_preds, modality_masks = (
            predictions.flatten_tokens_and_masks_per_modality()
        )
        modality_targets = targets.flatten_tokens_and_masks_per_modality()[0]

        # Accumulate to the total loss
        total_loss = 0
        for all_preds, all_masks, all_targets, modality in zip(
            modality_preds, modality_masks, modality_targets, targets.modalities
        ):
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
            losses = []
            start = 0
            for c in count:
                end = start + c
                if c == 0:
                    # we will occasionally get a sample with no decoded values due to missing data this will let us skip it
                    # logger.warning("No decoded values for this sample")
                    continue
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
            if len(losses) == 0:
                # If no losses were computed, skip this modality
                # logger.warning("No decoded values for this modality")
                continue
            loss = torch.stack(losses).mean()
            if self.modality_weights is not None:
                loss = loss * self.modality_weights[modality]
            total_loss += loss

        return self.weight * total_loss


@LOSS_REGISTRY.register("modality_patch_discrimination_masked_negatives")
class ModalityPatchDiscriminationMaskedNegatives(Loss):
    """Patch discrimination that masks out same-target negatives.

    Useful for map modalities where many tokens may have the same class/embedding.
    When computing contrastive loss, tokens with identical target embeddings
    are not treated as negatives (they are masked out from the denominator).
    """

    name = "ModalityPatchDiscMasked"

    def __init__(
        self,
        tau: float = 0.1,
        pred2unit: bool = False,
        weight: float = 1.0,
        modality_weights: dict[str, float] | None = None,
        same_target_threshold: float = 0.999,
        mask_negatives_for_modalities: list[str] | None = None,
    ):
        """Initialize masked negatives patch discrimination loss.

        Args:
            tau: the softmax temperature
            pred2unit: whether to standardize the predictions using batch statistics
            weight: the weight to apply to this loss
            modality_weights: the weights to apply to each modality
            same_target_threshold: cosine similarity threshold to consider targets as same
            mask_negatives_for_modalities: list of modality names to apply masking for.
                If None, applies to all modalities.
        """
        self.tau = tau
        self.pred2unit = pred2unit
        self.weight = weight
        self.modality_weights = modality_weights
        self.same_target_threshold = same_target_threshold
        self.mask_negatives_for_modalities = mask_negatives_for_modalities

    def compute(
        self, predictions: TokensAndMasks, targets: TokensAndMasks, **kwargs: Any
    ) -> Tensor:
        """Compute patch discrimination loss with masked same-target negatives."""
        modality_preds, modality_masks = predictions.flatten_tokens_and_masks(
            return_lists=True
        )
        modality_targets = targets.flatten_tokens_and_masks(return_lists=True)[0]

        total_loss = 0
        for all_preds, all_masks, all_targets, modality in zip(
            modality_preds, modality_masks, modality_targets, targets.modalities
        ):
            pred = all_preds[all_masks == MaskValue.DECODER.value].unsqueeze(dim=0)
            target = all_targets[all_masks == MaskValue.DECODER.value].unsqueeze(dim=0)
            pred = pred.float()
            target = target.float()


            all_preds, all_masks = predictions.flatten_all_tokens_and_masks()
            all_targets = targets.flatten_all_tokens_and_masks()[0]
            decoder_mask = all_masks == MaskValue.DECODER.value
            pred = all_preds[decoder_mask].unsqueeze(dim=0)
            target = all_targets[decoder_mask].unsqueeze(dim=0)
            bs, nt, _ = pred.shape
            if nt == 0:
                continue

            if self.pred2unit:
                pred_mu = pred.mean(1, keepdims=True)
                pred_std = pred.std(1, keepdims=True)
                pred = (pred - pred_mu) / (pred_std + 1e-4)

            pred = F.normalize(pred, p=2, dim=-1)
            target = F.normalize(target, p=2, dim=-1)

            # Check if we should mask negatives for this modality
            should_mask = (
                self.mask_negatives_for_modalities is None
                or modality in self.mask_negatives_for_modalities
            )

            count = (all_masks == MaskValue.DECODER.value).sum(dim=-1)
            losses = []
            start = 0

            for c in count:
                c_val = c.item() if hasattr(c, "item") else int(c)
                end = start + c_val
                if c_val == 0:
                    continue

                pred_sample = pred[:, start:end, :]
                target_sample = target[:, start:end, :]

                # Compute similarity scores
                score_sample = (
                    torch.einsum("npd,nqd->npq", pred_sample, target_sample) / self.tau
                )

                # Apply same-target masking if enabled for this modality
                if should_mask and c_val > 1:
                    target_flat = target_sample.squeeze(0)  # [c, dim]
                    target_sim = target_flat @ target_flat.T  # [c, c]
                    same_target = target_sim > self.same_target_threshold

                    # Mask: same target but not self (diagonal)
                    diagonal = torch.eye(
                        c_val, dtype=torch.bool, device=target_flat.device
                    )
                    invalid_negatives = same_target & ~diagonal

                    # Check if any token has valid negatives
                    valid_neg_count = (~same_target).sum(dim=-1)
                    if valid_neg_count.min() == 0:
                        # Some tokens have no valid negatives - skip this sample
                        start = end
                        continue

                    # Apply mask to scores: set invalid negatives to -inf
                    score_sample = score_sample.masked_fill(
                        invalid_negatives[None, :, :], float("-inf")
                    )

                # Standard cross-entropy
                labels = torch.arange(c_val, dtype=torch.long, device=pred.device)[None]
                loss = F.cross_entropy(
                    score_sample.flatten(0, 1),
                    labels.flatten(0, 1),
                    reduction="none",
                ) * (self.tau * 2)

                loss = loss.mean()
                losses.append(loss)
                start = end

            if len(losses) == 0:
                continue

            loss = torch.stack(losses).mean()
            if self.modality_weights is not None:
                loss = loss * self.modality_weights.get(modality, 1.0)

            total_loss += loss

        return self.weight * total_loss


# --- Deprecated aliases for backward compatibility ---


class _DeprecatedPatchDiscriminationLossNew(PatchDiscriminationLoss):
    def __init__(self, *args: Any, **kwargs: Any):
        warnings.warn(
            '"patch_discrimination_new" is deprecated, use "patch_discrimination" instead.',
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__(*args, **kwargs)


LOSS_REGISTRY.register("patch_discrimination_new")(
    _DeprecatedPatchDiscriminationLossNew
)


class _DeprecatedModalityPatchDiscriminationLossNew(ModalityPatchDiscriminationLoss):
    def __init__(self, *args: Any, **kwargs: Any):
        warnings.warn(
            '"modality_patch_discrimination_new" is deprecated, '
            'use "modality_patch_discrimination" instead.',
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__(*args, **kwargs)


LOSS_REGISTRY.register("modality_patch_discrimination_new")(
    _DeprecatedModalityPatchDiscriminationLossNew
)

# Backward-compat class name aliases
PatchDiscriminationLossNew = PatchDiscriminationLoss
ModalityPatchDiscriminationLossNew = ModalityPatchDiscriminationLoss


@LOSS_REGISTRY.register("adjusted_patch_discrimination")
class AdjustedPatchDiscriminationLoss(Loss):
    """Loss function for adjusted patch discrimination task.

    Reference: https://proceedings.neurips.cc/paper_files/paper/2023/file/48aaa5ea741ae8430bd58e25917d267d-Paper-Conference.pdf
    """

    name = "AdjustedPatchDisc"

    def __init__(
        self,
        tau: float = 0.1,
        mu: float = 0.7,
        sigma: float = 1.0,
        pred2unit: bool = False,
    ):
        """Initialize adjusted patch discrimination loss.

        Args:
            tau: the softmax temperature
            mu: the mean of the Gaussian distribution
            sigma: the standard deviation of the Gaussian distribution
            pred2unit: whether to standardize the predictions using batch statistics
        """
        self.tau = tau
        self.mu = mu
        self.sigma = sigma
        self.pred2unit = pred2unit

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
        all_preds, all_masks = predictions.flatten_all_tokens_and_masks()
        all_targets = targets.flatten_all_tokens_and_masks()[0]

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

        losses = []
        start = 0
        for c in count:
            end = start + c
            pred_sample = pred[:, start:end, :]  # (1, c, d)
            target_sample = target[:, start:end, :]  # (1, c, d)

            sim_matrix = torch.einsum(
                "npd,nqd->npq", pred_sample, target_sample
            )  # (1, c, c)

            pos_scores = torch.diagonal(sim_matrix, dim1=-2, dim2=-1)  # (1, c)
            pos_scores = pos_scores / self.tau

            # Mask out diagonal (positives) to get negatives
            mask = ~torch.eye(c, dtype=torch.bool, device=pred.device)
            neg_scores = sim_matrix.masked_select(mask).view(1, c, c - 1)  # (1, c, c-1)
            neg_scores = neg_scores / self.tau

            # Apply Gaussian-based weights to negatives
            # Weight is computed based on the neg_scores from a sample
            weight = (
                1.0
                / (self.sigma * math.sqrt(2 * math.pi))
                * torch.exp(
                    -((neg_scores * self.tau - self.mu) ** 2)
                    / (2 * math.pow(self.sigma, 2))
                )
            )  # (1, c, c-1)
            # Normalize the weights per query
            weight = weight / weight.mean(dim=-1, keepdim=True)
            neg_scores = neg_scores * weight.detach()

            # Reconstruct the sim_matrix
            sim_matrix = torch.zeros(
                1, c, c, device=pred.device, dtype=neg_scores.dtype
            )
            sim_matrix.diagonal(dim1=-2, dim2=-1).copy_(pos_scores)
            sim_matrix.masked_scatter_(mask, neg_scores)

            labels = torch.arange(c, dtype=torch.long, device=pred.device)[None]
            loss = F.cross_entropy(
                sim_matrix.flatten(0, 1),
                labels.flatten(0, 1),
                reduction="none",
            ) * (self.tau * 2)
            loss = loss.mean()
            losses.append(loss)
            start = end

        loss = torch.stack(losses).mean()
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
        all_preds, all_masks = predictions.flatten_all_tokens_and_masks()
        all_targets = targets.flatten_all_tokens_and_masks()[0]
        pred = all_preds[all_masks == MaskValue.DECODER.value]
        target = all_targets[all_masks == MaskValue.DECODER.value]

        return F.l1_loss(pred, target)


@LOSS_REGISTRY.register("l2")
class L2Loss(Loss):
    """Loss function for L2 (mean squared error)."""

    name = "L2"

    def compute(
        self, predictions: TokensAndMasks, targets: TokensAndMasks, **kwargs: Any
    ) -> Tensor:
        """Compute L2 loss between predictions and targets.

        Args:
            predictions: Model predictions.
            targets: Ground truth targets.
            **kwargs: Additional keyword arguments.

        Returns:
            The computed loss value.
        """
        all_preds, all_masks = predictions.flatten_all_tokens_and_masks()
        all_targets = targets.flatten_all_tokens_and_masks()[0]
        pred = all_preds[all_masks == MaskValue.DECODER.value]
        target = all_targets[all_masks == MaskValue.DECODER.value]
        return F.mse_loss(pred, target)


@LOSS_REGISTRY.register("mae")
class MAELoss(Loss):
    """Loss function masked auto-encoding (reconstruction)."""

    name = "MAE"

    def __init__(
        self,
        loss_function: str = "MSELoss",
        only_decode: bool = True,
        weight: float = 1.0,
        tokenization_config: TokenizationConfig | None = None,
        **kwargs: Any,
    ):
        """Initialize MAE loss.

        Args:
            loss_function: pytorch loss to use
            only_decode: only calculate loss on DECODER masked tokens, otherwise all
            weight: the weight to apply to this loss
            tokenization_config: Optional config for custom band groupings
            **kwargs: arguments for pytorch loss constructor
        """
        self.only_decode = only_decode
        self.loss = getattr(torch.nn, loss_function)(reduction="sum", **kwargs)
        self.weight = weight
        self.tokenization_config = tokenization_config or TokenizationConfig()

    # data: [B, H, W, T, C]
    def _flatten_spatiotemporal_data(
        self, data: TokensAndMasks
    ) -> tuple[Tensor, Tensor]:
        masks = []
        datas = []
        for modality in data.modalities:
            pred = getattr(data, modality)
            if pred is not None:
                mask = getattr(data, data.get_masked_modality_name(modality))
                for idx, channel_set_idxs in enumerate(
                    self.tokenization_config.get_bandset_indices(modality)
                ):
                    bs_mask = mask[..., idx]
                    bs_mask = repeat(
                        bs_mask, "b h w t -> b h w t c", c=len(channel_set_idxs)
                    )
                    bs_mask = rearrange(bs_mask, "b h w t c -> b (h w t c)")
                    masks.append(bs_mask)
                    bs_data = pred[..., channel_set_idxs]
                    bs_data = rearrange(bs_data, "b h w t c -> b (h w t c)")
                    datas.append(bs_data)
        return torch.cat(datas, dim=1), torch.cat(masks, dim=1)

    def compute(
        self, predictions: TokensAndMasks, targets: MaskedOlmoEarthSample, **kwargs: Any
    ) -> Tensor:
        """Compute MAE loss between predictions and targets.

        Args:
            predictions: Model predictions.
            targets: Ground truth targets.
            **kwargs: Additional keyword arguments.

        Returns:
            The computed loss value.
        """
        data, masks = self._flatten_spatiotemporal_data(predictions)
        valid_dict = {}
        for modality in predictions.modalities:
            if getattr(predictions, modality) is not None:
                masked_name = predictions.get_masked_modality_name(modality)
                valid_dict[modality] = getattr(targets, modality)
                valid_dict[masked_name] = getattr(targets, masked_name)
        valid_targets = TokensAndMasks(**valid_dict)
        labels, label_masks = self._flatten_spatiotemporal_data(valid_targets)
        if self.only_decode:
            decode = label_masks == MaskValue.DECODER.value
        else:
            decode = label_masks != MaskValue.MISSING.value
        data = data * decode
        labels = labels * decode
        return self.weight * self.loss(data, labels) / torch.count_nonzero(decode)


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
        all_preds, all_masks = predictions.flatten_all_tokens_and_masks()
        all_targets = targets.flatten_all_tokens_and_masks()[0]
        pred = all_preds[all_masks == MaskValue.DECODER.value]
        target = all_targets[all_masks == MaskValue.DECODER.value]

        return F.cross_entropy(pred, target.squeeze())


@LOSS_REGISTRY.register("InfoNCE")
class InfoNCELoss(Loss):
    """Loss function for InfoNCE."""

    name = "InfoNCE"

    def __init__(self, tau: float = 0.1, weight: float = 1):
        """Initialize InfoNCE loss.

        Args:
            tau: the softmax temperature
            weight: the weight to apply to this loss
        """
        self.tau = tau
        self.weight = weight

    def compute(
        self, predictions: torch.Tensor, targets: torch.Tensor, **kwargs: Any
    ) -> Tensor:
        """Compute InfoNCE between predictions and targets.

        Args:
            predictions: Model predictions.
            targets: Ground truth targets.
            **kwargs: Additional keyword arguments.

        Returns:
            The computed loss value.
        """
        predictions = F.normalize(predictions, p=2, dim=-1)
        targets = F.normalize(targets, p=2, dim=-1)
        logits = predictions @ targets.transpose(-2, -1)

        # Positive keys are the entries on the diagonal
        labels = torch.arange(len(predictions), device=predictions.device)
        return self.weight * F.cross_entropy(logits / self.tau, labels)


@LOSS_REGISTRY.register("KoLeo")
class KoLeoLoss(Loss):
    """Loss function for cross entropy.

    The KoLeo regularizer derives from the
    Kozachenko-Leonenko differential entropy estimator and
    encourages a uniform span of the features within a batch.

    https://github.com/facebookresearch/dinov2/blob/main/dinov2/loss/koleo_loss.py
    """

    name = "KoLeo"

    def __init__(
        self,
        weight: float = 0.1,
        mode: str = "instance",
        eps: float = 1e-8,
    ) -> None:
        """Initialize KoLeo regularizer.

        Args:
            weight: a weight to apply to the regularization value. Default value follows Dinov2
            eps: small value to avoid division by zero.
            mode: one of "instance" or "patch" - whether to compute
                nearest neighbourst at the instance or patch level
        """
        self.eps = eps
        self.pdist = torch.nn.PairwiseDistance(2, eps=eps)
        if mode not in ["instance", "patch"]:
            raise ValueError(f"Unsupported mode {mode}")
        self.mode = mode
        self.weight = weight

    @staticmethod
    def pairwise_nearest_neighbours(x: torch.Tensor) -> torch.Tensor:
        """Pairwise nearest neighbors for L2-normalized vectors.

        Uses Torch rather than Faiss to remain on GPU.

        Args:
            x: embeddings against which we want to compute nearest neighbours.

        Returns:
            indices: indices of nearest neighbour (i.e. indices[i] will return
            the index for the nearest neighbour of the ith embedding).
        """
        # parwise dot products (= inverse distance)
        dots = torch.mm(x, x.t())
        n = x.shape[0]
        dots.view(-1)[:: (n + 1)].fill_(-1)  # Trick to fill diagonal with -1
        # max inner prod -> min distance
        _, indices = torch.max(dots, dim=1)
        return indices

    def compute(
        self, predictions: TokensAndMasks, targets: None, **kwargs: Any
    ) -> Tensor:
        """Compute the KoLeo regularization term.

        Args:
            predictions: Model predictions. Unlike other losses, these are
                _online encoder outputs_, not decoder outputs.
            targets: Unused, and only kept for consistency.
            **kwargs: Additional keyword arguments.

        Returns:
            The computed loss value.
        """
        if isinstance(predictions, TokensAndMasks):
            if self.mode == "patch":
                if not isinstance(predictions, TokensAndMasks):
                    raise ValueError(
                        "predictions must be TokensAndMasks for patch mode"
                    )
                all_preds, all_masks = predictions.flatten_all_tokens_and_masks()
                online_encodings = all_preds[
                    all_masks == MaskValue.ONLINE_ENCODER.value
                ]
            else:
                online_encodings = predictions.pool_unmasked_tokens(
                    PoolingType.MEAN, spatial_pooling=False
                )
        else:
            online_encodings = predictions

        # apply l2 norm
        online_encodings = F.normalize(online_encodings, eps=self.eps, p=2, dim=-1)
        idx_of_nn = self.pairwise_nearest_neighbours(online_encodings)
        distances_to_nn = self.pdist(online_encodings, online_encodings[idx_of_nn])
        return self.weight * -torch.log(distances_to_nn + self.eps).mean()


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
