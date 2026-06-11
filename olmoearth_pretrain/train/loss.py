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
from olmoearth_pretrain.data.constants import Modality
from olmoearth_pretrain.datatypes import (
    MaskedOlmoEarthSample,
    MaskValue,
    TokensAndMasks,
)
from olmoearth_pretrain.nn.pooling import PoolingType, pool_unmasked_tokens
from olmoearth_pretrain.nn.tokenization import TokenizationConfig

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


@LOSS_REGISTRY.register("clip_patch_discrimination")
class ClipPatchDiscriminationLoss(Loss):
    """Loss function for configurable CLIP-like patch discrimination task.

    Token-level symmetric CLIP-style cross-entropy between decoded predictions
    and target-encoder tokens, closer to the original loss from the CLIP paper.
    Instead of a fixed tau temperature, the learned logit scale is passed to
    ``compute`` on every call (already clamped and exponentiated by the caller).

    Differences from the original implementation (PR #376), each fixing a
    defect found in review:
        - Predictions and targets are L2-normalized by default
          (``prediction_norm``/``target_norm`` default to 1.0; ``None``
          disables normalization).
        - Scores and cross-entropy are computed in float32 regardless of the
          input dtype, so a logit scale of ~100 does not collapse logit
          differences in bf16.
        - ``logit_scale`` is threaded through as a local variable rather than
          stored as mutable state on the loss object.
        - The two symmetric directions (pred->target and target->pred) are
          averaged in pairs rather than appended as separate loss entries, so
          enabling ``symmetric`` does not silently reweight modalities or
          grouping variants in the final reduction.
        - Optional same-target negative masking
          (``same_target_threshold``/``mask_negatives_for_modalities``) so
          map-like modalities (worldcover/srtm/osm/cdl/worldcereal/canopy)
          do not produce false negatives. Only supported for the
          ``modality_loss`` grouping.
    """

    name = "ClipPatchDisc"

    def __init__(
        self,
        label_smoothing: float = 0.0,
        prediction_norm: float | None = 1.0,
        target_norm: float | None = 1.0,
        modality_loss: bool = True,
        symmetric: bool = True,
        batch_loss: bool = False,
        bandset_loss: bool = False,
        spatial_loss: bool = False,
        time_loss: bool = False,
        mean_of_modalities: bool = True,
        sum_of_modalities: bool = False,
        decode_only: bool = True,
        weight: float = 1.0,
        same_target_threshold: float | None = None,
        mask_negatives_for_modalities: list[str] | None = None,
    ):
        """Initialize CLIP-style patch discrimination loss.

        Args:
            label_smoothing: label smoothing [0,1], 0=none, 1=too much
            prediction_norm: if not None, predictions are L2-normalized and
                multiplied by this value
            target_norm: if not None, targets are L2-normalized and multiplied
                by this value
            modality_loss: contrast each token against the other tokens of the
                same sample and modality
            symmetric: average the pred->target and target->pred directions
            batch_loss: contrast each token against the same position across
                the batch
            bandset_loss: contrast within (sample, band-set) groups
            spatial_loss: contrast within (sample, timestep) groups for
                space-time varying modalities (falls back to modality_loss
                otherwise)
            time_loss: contrast within (sample, spatial position) groups for
                space-time varying modalities (falls back to modality_loss
                otherwise)
            mean_of_modalities: mean of per-(modality, grouping) means instead
                of a mean over all per-token losses
            sum_of_modalities: sum of per-(modality, grouping) means instead
                of a mean over all per-token losses
            decode_only: only allow targets at DECODER positions as
                candidates (prevents cheating maybe?)
            weight: the weight to apply to this loss
            same_target_threshold: if not None, negatives whose target
                embedding has cosine similarity above this threshold with the
                positive's target are masked out of the softmax denominator
                (diagonal excluded). Only supported with the modality_loss
                grouping.
            mask_negatives_for_modalities: list of modality names to apply
                same-target masking for. If None, applies to all modalities.
        """
        if same_target_threshold is not None and (
            batch_loss or bandset_loss or spatial_loss or time_loss
        ):
            raise ValueError(
                "same_target_threshold is only supported with the "
                "modality_loss grouping; disable batch_loss/bandset_loss/"
                "spatial_loss/time_loss or set same_target_threshold=None."
            )
        self.label_smoothing = label_smoothing
        self.weight = weight
        self.prediction_norm = prediction_norm
        self.target_norm = target_norm
        self.modality_loss = modality_loss
        self.symmetric = symmetric
        self.batch_loss = batch_loss
        self.bandset_loss = bandset_loss
        self.mean_of_modalities = mean_of_modalities
        self.sum_of_modalities = sum_of_modalities
        self.decode_only = decode_only
        self.spatial_loss = spatial_loss
        self.time_loss = time_loss
        self.same_target_threshold = same_target_threshold
        self.mask_negatives_for_modalities = mask_negatives_for_modalities

    def _grouped_loss(
        self,
        rows: Tensor,
        cols: Tensor,
        masks: Tensor,
        logit_scale: Tensor,
        invalid_negatives: Tensor | None = None,
    ) -> Tensor:
        """Token-level CLIP cross-entropy for one grouping of tokens.

        Args:
            rows: (groups, n, d) query embeddings (already normalized).
            cols: (groups, n, d) candidate embeddings; cols[g, i] is the
                positive for rows[g, i].
            masks: (groups, n) mask values; only rows at DECODER positions
                contribute to the loss.
            logit_scale: final multiplicative scale for the logits.
            invalid_negatives: optional (groups, n, n) bool tensor; True
                entries are same-target negatives to exclude from the softmax
                denominator.

        Returns:
            1-D tensor of per-token losses at DECODER positions.
        """
        # Compute in float32: bf16 logits at scale ~100 collapse differences.
        rows = rows.float()
        cols = cols.float()
        score = torch.einsum("gxd,gyd->gxy", rows, cols) * logit_scale
        neg_inf = -torch.finfo(score.dtype).max
        decoder_cols = masks == MaskValue.DECODER.value  # (groups, n)
        if self.decode_only:
            score = score.masked_fill(~decoder_cols.unsqueeze(1), neg_inf)
        if invalid_negatives is not None:
            score = score.masked_fill(invalid_negatives, neg_inf)
        num_tokens = score.shape[-1]
        labels = torch.arange(num_tokens, dtype=torch.long, device=score.device)
        loss = F.cross_entropy(
            score.flatten(0, 1),
            labels.repeat(score.shape[0]),
            reduction="none",
            label_smoothing=self.label_smoothing,
        )
        row_select = decoder_cols.reshape(-1)
        if invalid_negatives is not None:
            # Rows whose negatives were all masked carry no contrastive
            # signal; exclude them (mirrors the skip in the masked-negatives
            # losses).
            diagonal = torch.eye(
                num_tokens, dtype=torch.bool, device=score.device
            ).unsqueeze(0)
            valid_negatives = ~invalid_negatives & ~diagonal
            if self.decode_only:
                valid_negatives = valid_negatives & decoder_cols.unsqueeze(1)
            row_select = row_select & valid_negatives.any(dim=-1).reshape(-1)
        return loss[row_select]

    def _calculate_modality_loss(
        self,
        rows: Tensor,
        cols: Tensor,
        masks: Tensor,
        logit_scale: Tensor,
        invalid_negatives: Tensor | None = None,
    ) -> Tensor:
        """Contrast each token against the other tokens of the same sample."""
        return self._grouped_loss(
            rearrange(rows, "b ... d -> b (...) d"),
            rearrange(cols, "b ... d -> b (...) d"),
            rearrange(masks, "b ... -> b (...)"),
            logit_scale,
            invalid_negatives,
        )

    def _calculate_batch_loss(
        self,
        rows: Tensor,
        cols: Tensor,
        masks: Tensor,
        logit_scale: Tensor,
        invalid_negatives: Tensor | None = None,
    ) -> Tensor:
        """Contrast each token against the same position across the batch."""
        return self._grouped_loss(
            rearrange(rows, "b ... d -> (...) b d"),
            rearrange(cols, "b ... d -> (...) b d"),
            rearrange(masks, "b ... -> (...) b"),
            logit_scale,
            invalid_negatives,
        )

    def _calculate_bandset_loss(
        self,
        rows: Tensor,
        cols: Tensor,
        masks: Tensor,
        logit_scale: Tensor,
        invalid_negatives: Tensor | None = None,
    ) -> Tensor:
        """Contrast within (sample, band-set) groups."""
        return self._grouped_loss(
            rearrange(rows, "b ... bs d -> (b bs) (...) d"),
            rearrange(cols, "b ... bs d -> (b bs) (...) d"),
            rearrange(masks, "b ... bs -> (b bs) (...)"),
            logit_scale,
            invalid_negatives,
        )

    def _calculate_spatial_loss(
        self,
        rows: Tensor,
        cols: Tensor,
        masks: Tensor,
        logit_scale: Tensor,
        invalid_negatives: Tensor | None = None,
    ) -> Tensor:
        """Contrast within (sample, timestep) groups across spatial positions."""
        if rows.ndim == 6:
            embed_rearrange = "b ph pw t bs d -> (b t) (ph pw bs) d"
            mask_rearrange = "b ph pw t bs -> (b t) (ph pw bs)"
        elif rows.ndim == 5:
            embed_rearrange = "b ph pw t d -> (b t) (ph pw) d"
            mask_rearrange = "b ph pw t -> (b t) (ph pw)"
        else:
            raise ValueError(
                f"spatial_loss requires space-time varying tokens of rank 5 "
                f"or 6, got rank {rows.ndim}"
            )
        return self._grouped_loss(
            rearrange(rows, embed_rearrange),
            rearrange(cols, embed_rearrange),
            rearrange(masks, mask_rearrange),
            logit_scale,
            invalid_negatives,
        )

    def _calculate_time_loss(
        self,
        rows: Tensor,
        cols: Tensor,
        masks: Tensor,
        logit_scale: Tensor,
        invalid_negatives: Tensor | None = None,
    ) -> Tensor:
        """Contrast within (sample, spatial position) groups across time."""
        if rows.ndim == 6:
            embed_rearrange = "b ph pw t bs d -> (b ph pw) (t bs) d"
            mask_rearrange = "b ph pw t bs -> (b ph pw) (t bs)"
        elif rows.ndim == 5:
            embed_rearrange = "b ph pw t d -> (b ph pw) t d"
            mask_rearrange = "b ph pw t -> (b ph pw) t"
        else:
            raise ValueError(
                f"time_loss requires space-time varying tokens of rank 5 "
                f"or 6, got rank {rows.ndim}"
            )
        return self._grouped_loss(
            rearrange(rows, embed_rearrange),
            rearrange(cols, embed_rearrange),
            rearrange(masks, mask_rearrange),
            logit_scale,
            invalid_negatives,
        )

    def _symmetric_loss(
        self,
        loss_fn: Any,
        preds: Tensor,
        targs: Tensor,
        masks: Tensor,
        logit_scale: Tensor,
        invalid_negatives: Tensor | None = None,
    ) -> Tensor:
        """Average the two contrastive directions into a single loss vector.

        The same-target ``invalid_negatives`` matrix is built from target
        cosine similarity, which is symmetric, so it applies unchanged to both
        directions.
        """
        forward = loss_fn(preds, targs, masks, logit_scale, invalid_negatives)
        if not self.symmetric:
            return forward
        backward = loss_fn(targs, preds, masks, logit_scale, invalid_negatives)
        return 0.5 * (forward + backward)

    def _build_same_target_negatives(self, targs: Tensor) -> Tensor:
        """Mark same-target negatives, excluding the diagonal.

        Args:
            targs: target tokens of shape (b, ..., d).

        Returns:
            Bool tensor of shape (b, n, n) where True marks pairs of distinct
            tokens whose target embeddings have cosine similarity above
            ``same_target_threshold``.
        """
        targs_flat = rearrange(targs, "b ... d -> b (...) d").float()
        normed = F.normalize(targs_flat, p=2, dim=-1)
        similarity = torch.bmm(normed, normed.transpose(1, 2))
        same_target = similarity > self.same_target_threshold
        diagonal = torch.eye(
            same_target.shape[-1], dtype=torch.bool, device=same_target.device
        ).unsqueeze(0)
        return same_target & ~diagonal

    def compute(
        self,
        predictions: TokensAndMasks,
        targets: TokensAndMasks,
        logit_scale: Tensor | None = None,
        **kwargs: Any,
    ) -> Tensor:
        """Compute CLIP-style patch discrimination loss.

        Args:
            predictions: Decoded predictions. Masks are read from here:
                DECODER positions mark which tokens participate in the loss.
            targets: Target-encoder tokens.
            logit_scale: The final multiplicative scale applied to the logits.
                The caller is responsible for clamping and exponentiating the
                learned parameter, i.e. pass
                ``model.logit_scale.clamp(max=...).exp()``, not the raw
                log-space parameter.
            **kwargs: Additional keyword arguments (ignored).

        Returns:
            The computed loss value.
        """
        assert logit_scale is not None, (
            "ClipPatchDiscriminationLoss.compute() requires a logit_scale "
            "kwarg: the final multiplicative logit scale, post clamp().exp() "
            "(e.g. model.logit_scale.clamp(max=math.log(100.0)).exp())."
        )
        losses: list[Tensor] = []
        for modality_name in predictions.modalities:
            # Cast to float32 up front (matches the masked-negatives-vec
            # convention) so normalization and scoring are stable.
            preds = getattr(predictions, modality_name).float()
            targs = getattr(targets, modality_name).float()
            masks = getattr(
                predictions, predictions.get_masked_modality_name(modality_name)
            )
            if self.target_norm is not None:
                targs = self.target_norm * F.normalize(targs, p=2, dim=-1)
            if self.prediction_norm is not None:
                preds = self.prediction_norm * F.normalize(preds, p=2, dim=-1)

            invalid_negatives = None
            if self.same_target_threshold is not None and (
                self.mask_negatives_for_modalities is None
                or modality_name in self.mask_negatives_for_modalities
            ):
                invalid_negatives = self._build_same_target_negatives(targs)

            if self.modality_loss:
                losses.append(
                    self._symmetric_loss(
                        self._calculate_modality_loss,
                        preds,
                        targs,
                        masks,
                        logit_scale,
                        invalid_negatives,
                    )
                )

            if self.batch_loss:
                losses.append(
                    self._symmetric_loss(
                        self._calculate_batch_loss, preds, targs, masks, logit_scale
                    )
                )

            if self.bandset_loss:
                losses.append(
                    self._symmetric_loss(
                        self._calculate_bandset_loss, preds, targs, masks, logit_scale
                    )
                )

            if self.time_loss:
                time_fn = (
                    self._calculate_time_loss
                    if Modality.get(modality_name).is_spacetime_varying
                    else self._calculate_modality_loss
                )
                losses.append(
                    self._symmetric_loss(time_fn, preds, targs, masks, logit_scale)
                )

            if self.spatial_loss:
                spatial_fn = (
                    self._calculate_spatial_loss
                    if Modality.get(modality_name).is_spacetime_varying
                    else self._calculate_modality_loss
                )
                losses.append(
                    self._symmetric_loss(spatial_fn, preds, targs, masks, logit_scale)
                )

        nonempty = [loss for loss in losses if loss.numel() > 0]
        if len(nonempty) == 0:
            return torch.tensor(0.0, device=predictions.device)
        if self.mean_of_modalities:
            total_loss = torch.stack([loss.mean() for loss in nonempty]).mean()
        elif self.sum_of_modalities:
            total_loss = torch.stack([loss.mean() for loss in nonempty]).sum()
        else:
            total_loss = torch.cat([loss.flatten() for loss in nonempty]).mean()

        return self.weight * total_loss


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
        modality_preds, modality_masks = (
            predictions.flatten_tokens_and_masks_per_modality()
        )
        modality_targets = targets.flatten_tokens_and_masks_per_modality()[0]

        total_loss = 0
        for all_preds, all_masks, all_targets, modality in zip(
            modality_preds, modality_masks, modality_targets, targets.modalities
        ):
            pred = all_preds[all_masks == MaskValue.DECODER.value].unsqueeze(dim=0)
            target = all_targets[all_masks == MaskValue.DECODER.value].unsqueeze(dim=0)
            pred = pred.float()
            target = target.float()
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


@LOSS_REGISTRY.register("modality_patch_discrimination_masked_negatives_vec")
class ModalityPatchDiscriminationMaskedNegativesVec(Loss):
    """Vectorized patch discrimination with same-target negative masking.

    Equivalent to ModalityPatchDiscriminationMaskedNegatives but fully batched:
    no per-sample Python loops, no .item() syncs, no repeated torch.eye allocations.
    """

    name = "ModalityPatchDiscMaskedVec"

    def __init__(
        self,
        tau: float = 0.1,
        pred2unit: bool = False,
        weight: float = 1.0,
        modality_weights: dict[str, float] | None = None,
        same_target_threshold: float = 0.999,
        mask_negatives_for_modalities: list[str] | None = None,
    ) -> None:
        """Initialize with same params as ModalityPatchDiscriminationMaskedNegatives."""
        self.tau = tau
        self.pred2unit = pred2unit
        self.weight = weight
        self.modality_weights = modality_weights
        self.same_target_threshold = same_target_threshold
        self.mask_negatives_for_modalities = mask_negatives_for_modalities

    def _compute_modality_loss_parallel(
        self,
        all_preds: Tensor,
        all_masks: Tensor,
        all_targets: Tensor,
        modality: str,
    ) -> Tensor:
        batch_size, num_tokens, dim = all_preds.shape
        decoder_mask = all_masks == MaskValue.DECODER.value
        count = decoder_mask.sum(dim=-1)  # (batch,)

        # Sort so decoder tokens come first per sample
        _, sort_indices = decoder_mask.long().sort(dim=1, descending=True, stable=True)
        sort_expanded = sort_indices.unsqueeze(-1).expand(-1, -1, dim)
        sorted_preds = all_preds.gather(1, sort_expanded).float()
        sorted_targets = all_targets.gather(1, sort_expanded).float()

        # valid_mask[b, i] = True iff position i is a decoder token for sample b
        range_tensor = torch.arange(num_tokens, device=count.device)
        valid_mask = range_tensor.unsqueeze(0) < count.unsqueeze(1)  # (batch, T)

        if self.pred2unit:
            mask_float = valid_mask.unsqueeze(-1).float()
            total_decoder = mask_float.sum().clamp(min=1)
            pred_mu = (sorted_preds * mask_float).sum(
                dim=(0, 1), keepdim=True
            ) / total_decoder
            centered = sorted_preds - pred_mu
            pred_var = (centered**2 * mask_float).sum(dim=(0, 1), keepdim=True) / (
                total_decoder - 1
            ).clamp(min=1)
            sorted_preds = (sorted_preds - pred_mu) / (pred_var.sqrt() + 1e-4)

        sorted_preds = F.normalize(sorted_preds, p=2, dim=-1)
        sorted_targets = F.normalize(sorted_targets, p=2, dim=-1)

        # Score matrix: (batch, T, T) — each sample independent
        scores = torch.bmm(sorted_preds, sorted_targets.transpose(1, 2)) / self.tau

        should_mask = (
            self.mask_negatives_for_modalities is None
            or modality in self.mask_negatives_for_modalities
        )

        # Track which samples to skip (default: none)
        sample_skip = torch.zeros(batch_size, dtype=torch.bool, device=scores.device)

        if should_mask:
            # Target self-similarity per sample: (batch, T, T)
            target_sim = torch.bmm(sorted_targets, sorted_targets.transpose(1, 2))
            same_target = target_sim > self.same_target_threshold

            # Only consider valid token pairs
            valid_2d = valid_mask.unsqueeze(1) & valid_mask.unsqueeze(
                2
            )  # (batch, T, T)

            # Diagonal (self) is never an invalid negative
            diag = torch.eye(num_tokens, dtype=torch.bool, device=scores.device)
            invalid_negatives = same_target & ~diag.unsqueeze(0) & valid_2d

            # The original only applies masking when c_val > 1, so restrict
            # invalid_negatives and skip-detection to samples with count > 1.
            multi_token = (count > 1).unsqueeze(1).unsqueeze(2)  # (batch, 1, 1)
            invalid_negatives = invalid_negatives & multi_token

            # Skip samples where any valid token has zero valid negatives
            valid_neg_count = (~same_target & valid_2d).sum(dim=-1)  # (batch, T)
            has_zero_neg = (
                (valid_neg_count == 0) & valid_mask & (count > 1).unsqueeze(1)
            )
            sample_skip = has_zero_neg.any(dim=1)

            scores = scores.masked_fill(invalid_negatives, float("-inf"))

        # Mask out non-decoder columns
        col_mask = valid_mask.unsqueeze(1).expand_as(scores)
        scores = scores.masked_fill(~col_mask, -torch.finfo(scores.dtype).max)

        # Mask rows for zero-count samples to prevent NaN
        row_mask = valid_mask.unsqueeze(2).expand_as(scores)
        scores = scores.masked_fill(~row_mask, 0.0)

        # Labels: diagonal (token i matches target i)
        labels = range_tensor.unsqueeze(0).expand(batch_size, -1)

        loss_per_pos = F.cross_entropy(
            scores.reshape(-1, num_tokens),
            labels.reshape(-1),
            reduction="none",
        ) * (self.tau * 2)
        loss_per_pos = loss_per_pos.reshape(batch_size, num_tokens)

        # Zero out invalid positions and skipped samples
        sample_contributes = (count > 0) & ~sample_skip
        effective_valid = valid_mask.float() * sample_contributes.unsqueeze(1).float()
        effective_count = count.float() * sample_contributes.float()
        num_contributing = sample_contributes.sum()

        loss_per_sample = (loss_per_pos * effective_valid).sum(
            dim=1
        ) / effective_count.clamp(min=1)
        loss = loss_per_sample.sum() / num_contributing.float().clamp(min=1)

        return loss

    def compute(
        self, predictions: TokensAndMasks, targets: TokensAndMasks, **kwargs: Any
    ) -> Tensor:
        """Compute patch discrimination loss with masked same-target negatives (vectorized)."""
        modality_preds, modality_masks = (
            predictions.flatten_tokens_and_masks_per_modality()
        )
        modality_targets = targets.flatten_tokens_and_masks_per_modality()[0]

        total_loss = 0
        for all_preds, all_masks, all_targets, modality in zip(
            modality_preds, modality_masks, modality_targets, targets.modalities
        ):
            loss = self._compute_modality_loss_parallel(
                all_preds, all_masks, all_targets, modality
            )
            if self.modality_weights is not None:
                loss = loss * self.modality_weights.get(modality, 1.0)
            total_loss += loss

        return self.weight * total_loss


@LOSS_REGISTRY.register("modality_patch_discrimination_vec")
class ModalityPatchDiscriminationLossVec(Loss):
    """Loss function for per-modality patch discrimination task.

    This is a fully parallelized implementation with no for loops over samples.
    It does not support all discrimination loss.
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
            weight: the weight to apply to this loss
            modality_weights: the weights to apply to each modality
        """
        self.tau = tau
        self.pred2unit = pred2unit
        self.weight = weight
        self.modality_weights = modality_weights

    def _compute_modality_loss_parallel(
        self,
        all_preds: Tensor,
        all_masks: Tensor,
        all_targets: Tensor,
    ) -> Tensor:
        """Compute patch discrimination loss for a single modality in parallel.

        Uses sort-based token reordering and pure masking to avoid all
        boolean indexing (nonzero) and GPU→CPU sync points.

        Args:
            all_preds: Predictions tensor of shape (batch, tokens, dim)
            all_masks: Mask tensor of shape (batch, tokens)
            all_targets: Targets tensor of shape (batch, tokens, dim)

        Returns:
            The computed loss value for this modality. Zero if no decoder tokens are present.
        """
        batch_size, num_tokens, dim = all_preds.shape
        decoder_mask = all_masks == MaskValue.DECODER.value
        count = decoder_mask.sum(dim=-1)  # (batch,)
        num_valid = (count > 0).sum()  # stays as tensor, no sync

        # Sort tokens so decoder tokens come first, preserving relative order.
        _, sort_indices = decoder_mask.long().sort(dim=1, descending=True, stable=True)
        sort_expanded = sort_indices.unsqueeze(-1).expand(-1, -1, dim)
        sorted_preds = all_preds.gather(1, sort_expanded)
        sorted_targets = all_targets.gather(1, sort_expanded)

        # Validity mask: first count[b] positions per sample are decoder tokens.
        range_tensor = torch.arange(num_tokens, device=count.device)
        valid_mask = range_tensor.unsqueeze(0) < count.unsqueeze(
            1
        )  # (batch, num_tokens)

        if self.pred2unit:
            # Global mean/std across all decoder tokens (matches original flat behavior)
            mask_float = valid_mask.unsqueeze(-1).float()  # (batch, tokens, 1)
            total_decoder = mask_float.sum().clamp(min=1)
            pred_mu = (sorted_preds * mask_float).sum(
                dim=(0, 1), keepdim=True
            ) / total_decoder
            centered = sorted_preds - pred_mu
            pred_var = (centered**2 * mask_float).sum(dim=(0, 1), keepdim=True) / (
                total_decoder - 1
            ).clamp(min=1)
            pred_std = pred_var.sqrt()
            sorted_preds = (sorted_preds - pred_mu) / (pred_std + 1e-4)

        sorted_preds = F.normalize(sorted_preds, p=2, dim=-1)
        sorted_targets = F.normalize(sorted_targets, p=2, dim=-1)

        # Compute scores: (batch, num_tokens, num_tokens)
        scores = torch.bmm(sorted_preds, sorted_targets.transpose(1, 2)) / self.tau

        # Mask out non-decoder columns with -inf so they don't affect softmax
        col_mask = valid_mask.unsqueeze(1).expand_as(scores)
        scores = scores.masked_fill(~col_mask, -torch.finfo(scores.dtype).max)

        # Also mask rows for zero-count samples to prevent NaN from all-inf softmax
        row_mask = valid_mask.unsqueeze(2).expand_as(scores)
        scores = scores.masked_fill(~row_mask, 0.0)

        # Labels: diagonal (decoder token i should match decoder token i)
        labels = range_tensor.unsqueeze(0).expand(batch_size, -1)

        # Cross entropy per position
        loss_per_pos = F.cross_entropy(
            scores.reshape(-1, num_tokens),
            labels.reshape(-1),
            reduction="none",
        ) * (self.tau * 2)
        loss_per_pos = loss_per_pos.reshape(batch_size, num_tokens)

        # Zero out invalid positions, average per sample, then over valid samples only
        valid_mask_float = valid_mask.float()
        loss_per_sample = (loss_per_pos * valid_mask_float).sum(
            dim=1
        ) / count.float().clamp(min=1)
        loss = loss_per_sample.sum() / num_valid.float().clamp(min=1)

        return loss

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
            loss = self._compute_modality_loss_parallel(
                all_preds, all_masks, all_targets
            )
            if self.modality_weights is not None:
                loss = loss * self.modality_weights[modality]
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


@LOSS_REGISTRY.register("cosine_similarity")
class CosineSimilarityLoss(Loss):
    """Negative mean cosine similarity between predicted and target decoder tokens."""

    name = "CosineSim"

    def compute(
        self, predictions: TokensAndMasks, targets: TokensAndMasks, **kwargs: Any
    ) -> Tensor:
        """Compute negative cosine similarity loss between predictions and targets.

        Args:
            predictions: Model predictions.
            targets: Ground truth targets.
            **kwargs: Additional keyword arguments.

        Returns:
            The computed loss value (negative mean cosine similarity).
        """
        all_preds, all_masks = predictions.flatten_all_tokens_and_masks()
        all_targets = targets.flatten_all_tokens_and_masks()[0]
        pred = all_preds[all_masks == MaskValue.DECODER.value]
        target = all_targets[all_masks == MaskValue.DECODER.value]
        return -F.cosine_similarity(pred, target, dim=-1).mean()


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


@LOSS_REGISTRY.register("clip_infonce")
class ClipInfoNCELoss(Loss):
    """CLIP-style instance contrastive loss on pooled [B, D] embeddings.

    Differences from :class:`InfoNCELoss`:
      * Learnable temperature: ``compute`` receives ``logit_scale`` — the
        final multiplicative scale, post ``clamp().exp()`` — instead of a
        fixed ``1/tau``. Gradient flows through it, so the optimizer tunes
        the softmax sharpness.
      * Symmetric: cross-entropy is averaged over both directions
        (predictions as queries and targets as queries), as in CLIP.
      * float32: logits and CE are computed in float32 regardless of input
        dtype (a scale near 100 collapses bf16 logit differences).
    """

    name = "ClipInfoNCE"

    def __init__(self, weight: float = 1.0):
        """Initialize the CLIP-style InfoNCE loss.

        Args:
            weight: the weight to apply to this loss
        """
        self.weight = weight

    def compute(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        logit_scale: Tensor | None = None,
        **kwargs: Any,
    ) -> Tensor:
        """Compute symmetric temperature-scaled InfoNCE between [B, D] embeddings.

        Args:
            predictions: Pooled embeddings of one view, shape [B, D].
            targets: Pooled embeddings of the other view, shape [B, D].
            logit_scale: The final multiplicative scale (post clamp().exp()).
            **kwargs: Additional keyword arguments (ignored).

        Returns:
            The computed loss value.
        """
        assert logit_scale is not None, (
            "ClipInfoNCELoss requires logit_scale — the final multiplicative "
            "scale, post clamp().exp() — passed by the train module."
        )
        predictions = F.normalize(predictions.float(), p=2, dim=-1)
        targets = F.normalize(targets.float(), p=2, dim=-1)
        logits = logit_scale.float() * (predictions @ targets.transpose(-2, -1))

        labels = torch.arange(len(predictions), device=predictions.device)
        loss = 0.5 * (
            F.cross_entropy(logits, labels)
            + F.cross_entropy(logits.transpose(-2, -1), labels)
        )
        return self.weight * loss


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
                online_encodings = pool_unmasked_tokens(
                    predictions, PoolingType.MEAN, spatial_pooling=False
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
