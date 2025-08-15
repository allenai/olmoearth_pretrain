"""Loss functions for training."""

import logging
import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from class_registry import ClassRegistry
from einops import rearrange, repeat
from olmo_core.config import Config
from torch import Tensor

from helios.data.constants import Modality
from helios.nn.flexihelios import PoolingType, TokensAndMasks
from helios.train.masking import MaskedHeliosSample, MaskValue

logger = logging.getLogger(__name__)

def get_spatial_grid(H, W, device="cpu"):
    rows = torch.arange(H, device=device).repeat_interleave(W)   # [0,0,...,1,1,...]
    cols = torch.arange(W, device=device).repeat(H)              # [0,1,2,...,0,1,2,...]
    return torch.stack([rows, cols], dim=-1)

def chebyshev_within_mask(H, W, n, device="cpu"):
    N = H * W
    rows = torch.arange(H, device=device).repeat_interleave(W)   # [0,0,...,1,1,...]
    cols = torch.arange(W, device=device).repeat(H)              # [0,1,2,...,0,1,2,...]

    dr = (rows[:, None] - rows[None, :]).abs()  # (N, N)
    dc = (cols[:, None] - cols[None, :]).abs()  # (N, N)
    return ((dr <= n) & (dc <= n)).bool()


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
        all_preds, all_masks = predictions.flatten_tokens_and_masks()
        all_targets = targets.flatten_tokens_and_masks()[0]

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


@LOSS_REGISTRY.register("patch_discrimination_new")
class PatchDiscriminationLossNew(Loss):
    """Loss function for patch discrimination task.

    This has lower memory consumption than the old patch discrimination loss.
    It does not support all discrimination loss.
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


#  The global embedding is two similar to individual embeddings
# how can we add a regularizar
# maybe just a single term that adds a penalty if the variance of the embeddings is too low

        # Compute the squared norm of the difference between each encoding and the mean encoding

@LOSS_REGISTRY.register("anti_anisotropy_top_pc")
class AntiAnisotropyTopPC(Loss):
    """
    Penalizes variance concentrated along the top principal component.
    Equivalent to penalizing the top eigenvalue of the centered covariance.

    Loss = weight * ( lambda_max(C) / (trace(C) + eps_ref) )
    where C = cov(centered, unit-norm features)

    This discourages a single dominant direction without fully computing all eigenvalues.
    """

    name = "AntiAnisotropyTopPC"

    def __init__(
        self,
        weight: float = 1e-3,
        mode: str = "patch",       # "patch" or "instance", same semantics as KoLeo
        eps: float = 1e-8,
        power_iter: int = 10,      # iterations for power method (top eigen)
        normalize_trace: bool = True,
    ) -> None:
        self.weight = weight
        if mode not in ["instance", "patch"]:
            raise ValueError(f"Unsupported mode {mode}")
        self.mode = mode
        self.eps = eps
        self.power_iter = power_iter
        self.normalize_trace = normalize_trace

    def _extract_encodings(self, predictions):
        if isinstance(predictions, TokensAndMasks):
            if self.mode == "patch":
                all_preds, all_masks = predictions.flatten_tokens_and_masks()
                encoded_token_count = (all_masks == MaskValue.ONLINE_ENCODER.value).sum(dim=-1)
                enc = all_preds[all_masks == MaskValue.ONLINE_ENCODER.value]
            else:
                enc = predictions.pool_unmasked_tokens(
                    PoolingType.MEAN, spatial_pooling=False
                )
        else:
            enc = predictions
        return enc, encoded_token_count

    @staticmethod
    def _centered_cov(X: Tensor, eps: float) -> Tensor:
        # X: (N, D), already unit-normalized
        N = X.size(0)
        if N <= 1:
            # Degenerate batch; return tiny isotropic cov for stability
            D = X.size(1)
            return eps * torch.eye(D, device=X.device, dtype=X.dtype)
        Xc = X - X.mean(dim=0, keepdim=True)
        C = (Xc.T @ Xc) / (N - 1)
        return C

    def _top_eigenvalue_power(self, C: Tensor) -> Tensor:
        # Power iteration for largest eigenvalue; differentiable through C
        D = C.size(0)
        v = torch.randn(D, device=C.device, dtype=C.dtype)
        v = v / (v.norm() + self.eps)
        for _ in range(self.power_iter):
            v = C @ v
            v = v / (v.norm() + self.eps)
        # Rayleigh quotient
        lam1 = (v @ (C @ v))
        return lam1

    def compute(self, predictions: TokensAndMasks, targets: None, **kwargs: Any) -> Tensor:
        enc, encoded_token_count = self._extract_encodings(predictions)
        # L2 normalize features to prevent scale cheating
        Z = F.normalize(enc, p=2, dim=-1)
        start = 0
        losses = []
        for c in encoded_token_count:
            end = start + c
            if c == 0:
                continue
            Z_sample = Z[start:end, :]
            C = self._centered_cov(Z_sample, self.eps)
            lam1 = self._top_eigenvalue_power(C)

            if self.normalize_trace:
                tr = torch.trace(C)
                # Normalize by total variance to get a shape-free ratio
                loss_val = lam1 / (tr + self.eps)
            else:
                loss_val = lam1
            losses.append(loss_val)
            start = end
        loss = torch.stack(losses).mean()
        return self.weight * loss

@LOSS_REGISTRY.register("negative_variance")
class NegativeVariance(Loss):
    """
    Subtracts (scaled) variance of embeddings from the loss:
      Loss = - weight * mean_diag(C)
    where C is centered covariance of unit-normalized features.

    This encourages larger spread without managing anisotropy explicitly.
    """

    name = "NegativeVariance"

    def __init__(
        self,
        weight: float = 1e-3,     # note the sign: this will be subtracted from total loss
        mode: str = "patch",
        eps: float = 1e-8,
        use_mean_per_dim: bool = True,  # average variance per-dim for scale-invariance
    ) -> None:
        self.weight = weight
        if mode not in ["instance", "patch"]:
            raise ValueError(f"Unsupported mode {mode}")
        self.mode = mode
        self.eps = eps
        self.use_mean_per_dim = use_mean_per_dim

    def _extract_encodings(self, predictions):
        if isinstance(predictions, TokensAndMasks):
            if self.mode == "patch":
                all_preds, all_masks = predictions.flatten_tokens_and_masks()
                encoded_token_count = (all_masks == MaskValue.ONLINE_ENCODER.value).sum(dim=-1)
                enc = all_preds[all_masks == MaskValue.ONLINE_ENCODER.value]
            else:
                enc = predictions.pool_unmasked_tokens(
                    PoolingType.MEAN, spatial_pooling=False
                )
        else:
            enc = predictions
        return enc, encoded_token_count

    def compute(self, predictions: TokensAndMasks, targets: None, **kwargs: Any) -> Tensor:
        enc, encoded_token_count = self._extract_encodings(predictions)
        Z = F.normalize(enc, p=2, dim=-1)
        start = 0
        losses = []
        for c in encoded_token_count:
            end = start + c
            if c == 0:
                continue
            Z_sample = Z[start:end, :]
            N, D = Z_sample.shape
            if N <= 1:
                return Z_sample.new_tensor(0.0)

            Zc = Z_sample - Z_sample.mean(dim=0, keepdim=True)
            C = (Zc.T @ Zc) / (N - 1)
            # total variance = trace(C); mean per-dim variance = trace(C) / D
            var_val = torch.trace(C)
            if self.use_mean_per_dim:
                var_val = var_val / (D + self.eps)
            losses.append(var_val)
            start = end
        loss = torch.stack(losses).mean()
        return -self.weight * loss

# WIP Stuck on how to make sure the right par mask is passed and aligned to each sample
@LOSS_REGISTRY.register("patch_discrimination_no_neighbors")
class PatchDiscriminationLossNoNeighbors(Loss):
    """Loss function for patch discrimination task.

    This has lower memory consumption than the old patch discrimination loss.
    It does not support all discrimination loss.
    """

    name = "PatchDiscNoNeighbors"

    def __init__(self, tau: float = 0.1, pred2unit: bool = False, weight: float = 1.0, neighbor_radius: int = 1):
        """Initialize patch discrimination loss.

        Args:
            tau: the softmax temperature
            pred2unit: whether to standardize the predictions using batch statistics
            mask_other_samples: whether to apply the contrastive loss drawing samples
                from within a sample (True) or using all other instances in a batch (False).
                If this is False, then this is the AllDisc loss from the Galileo paper
            weight: the weight to apply to this loss
            neighbor_radius: the radius of the neighborhood to not consider  as negatives for the loss
        """
        self.tau = tau
        self.pred2unit = pred2unit
        self.weight = weight
        self.neighbor_radius = neighbor_radius

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
        # before we flatten this we need to figure out what the nearby tokens are in a way that we can use it to mask out the nearby negatives
        # for each modality  do we want a different mask?
        # What if we just create a spatial grid for each modality and then creat pariswise distance matrix for all the decoder tokens
        # Then we can  threshold that mask so that we filter out any negatives that are within the neighbor_radius

        # I need a mask for each modality
        neighbors_masks_dict = {}
        decoder_pair_masks = {}
        for modality in predictions.modalities:
            modality_spec = Modality.get(modality)
            if not modality_spec.is_spatial:
                continue
            logger.info(f"modality: {modality}")
            data = getattr(predictions, modality)
            logger.info(f"data.shape: {data.shape}")
            B, H, W, T, B_S, _ = data.shape

            masked_name  = TokensAndMasks.get_masked_modality_name(modality)
            mask_data = getattr(predictions, masked_name)
            neighbors_decoder_mask = mask_data == MaskValue.DECODER.value
            # num decoded tokens before collapse
            num_decoded_tokens = (neighbors_decoder_mask == MaskValue.DECODER.value).sum()
            logger.info(f"num_decoded_tokens.shape: {num_decoded_tokens.shape} {num_decoded_tokens}")
            collapsed_hw = rearrange(neighbors_decoder_mask, "b h w ... -> b (...) (h w) ").long()
            # num decoded tokens after collapse
            num_decoded_tokens_collapsed = (collapsed_hw == MaskValue.DECODER.value).sum()
            logger.info(f"num_decoded_tokens_collapsed.shape: {num_decoded_tokens_collapsed.shape} {num_decoded_tokens_collapsed}")
            logger.info(f"collapsed_hw.shape: {collapsed_hw.shape}")
            pair_mask = collapsed_hw.unsqueeze(-1) & collapsed_hw.unsqueeze(-2)

            # Rearrange the pair mask
            pair_mask = rearrange(pair_mask, "b (b_s t) n1 n2 -> b n1 n2 b_s t", b_s=B_S, t=T)

            # Hacky redundancy for now
            decoder_pair_masks[masked_name] = pair_mask.bool()
            decoder_pair_masks[modality] = data

            spatial_grid = get_spatial_grid(H, W, device=predictions.device)
            logger.info(f"spatial_grid.shape: {spatial_grid.shape}")
            # IF both of the positions for which we have a pairwise distance are decoder only then we say it is decoder only
            # make this a long tensor
            neighbors_masks = chebyshev_within_mask(H, W, self.neighbor_radius, device=predictions.device)

            logger.info(f"neighbors_masks.shape: {neighbors_masks.shape}")
            # Repeat in T and B_S dimensions so we have the mask for each timestep and modalit
            # the issue here is we need to calculate a new decoder mask for the neighborhood masks

            neighbors_masks = repeat(neighbors_masks, "h w -> b h w t b_s", t=T, b_s=B_S, b=B)
            logger.info(f"neighbors_masks.shape: {neighbors_masks.shape}")
            # for each neighborhood masks we
            neighbors_masks_dict[masked_name] = neighbors_masks
            neighbors_masks_dict[modality] = data

        all_neighbors_masks = TokensAndMasks(**neighbors_masks_dict).flatten_tokens_and_masks()[1]
        logger.info(f"all_neighbors_masks.shape: {all_neighbors_masks.shape}")
        decoder_pair_masks = TokensAndMasks(**decoder_pair_masks).flatten_tokens_and_masks()[1]
        logger.info(f"decoder_pair_masks.shape: {decoder_pair_masks.shape}")


        all_preds, all_masks = predictions.flatten_tokens_and_masks()
        all_targets = targets.flatten_tokens_and_masks()[0]
        logger.info(f"all_preds.shape: {all_preds.shape}")
        logger.info(f"all_masks.shape: {all_masks.shape}")
        logger.info(f"all_targets.shape: {all_targets.shape}")
        # create a grid of spatial locations for each modality

        # Samples may have different number of tokens
        # TODO: Skip unqueeze and the for loop when mask_other_samples is True

        pred = all_preds[all_masks == MaskValue.DECODER.value].unsqueeze(dim=0)
        target = all_targets[all_masks == MaskValue.DECODER.value].unsqueeze(dim=0)

        # get the shape of all
        neighbors_masks = all_neighbors_masks[decoder_pair_masks].unsqueeze(dim=0)
        # get how many of these belong to each sample
        # I would need the decoder sampling
        pair_count = (decoder_pair_masks).sum(dim=-1)
        logger.info(f"pair_count.shape: {pair_count.shape} {pair_count}")
        logger.info(f"neighbors_masks.shape: {neighbors_masks.shape}")
        bs, nt, _ = pred.shape


        # for each token we need a decode don't decode boolean mask so we can use it to mask out the nearby negatives

        if self.pred2unit:
            pred_mu = pred.mean(1, keepdims=True)
            pred_std = pred.std(1, keepdims=True)
            pred = (pred - pred_mu) / (pred_std + 1e-4)

        pred = F.normalize(pred, p=2, dim=-1)
        target = F.normalize(target, p=2, dim=-1)

        count = (all_masks == MaskValue.DECODER.value).sum(dim=-1)
        losses = []
        start = 0
        start_pair = 0
        for c, pair_count in zip(count, pair_count):
            end = start + c
            end_pair = start_pair + pair_count
            if c == 0:
                # we will occasionally get a sample with no decoded values due to missing data this will let us skip it
                logger.warning("No decoded values for this sample")
                continue
            pred_sample   = pred[:, start:end, :]          # [N, c, D]
            target_sample = target[:, start:end, :]        # [N, c, D]
            mask_sample   = neighbors_masks[:, start_pair:end_pair]  # [N, c, C]
            N, c, D = pred_sample.shape
            logger.info(f"mask_sample.shape: {mask_sample.shape} c {c} pair_count {pair_count}")
            _, _, C = mask_sample.shape  # typically C == original sequence length; here we want c classes

            # similarity logits: [N, c, c]
            score_sample = torch.einsum("npd,nqd->npq", pred_sample, target_sample) / self.tau

            # restrict to the same c columns (q in start:end) so scores and masks match
            score_sample = score_sample[:, :, :c]          # [N, c, c]
            mask_sample  = mask_sample[:, :, start:end]    # [N, c, c]

            # ensure the positive (diagonal) is always included
            eye = torch.eye(c, dtype=torch.bool, device=pred.device)  # [c, c]
            mask = mask_sample | eye.unsqueeze(0)                      # [N, c, c]

            # remove neighbor logits from the softmax support
            masked_scores = score_sample.masked_fill(mask, float("-inf"))  # no grad flows through invalid entries

            # labels: for each (n, p) the correct class is q == p
            labels = torch.arange(c, device=pred.device).expand(N, -1).reshape(-1)  # [N*c]

            # standard CE over the masked logits (only neighbors contribute)
            loss = F.cross_entropy(masked_scores.reshape(-1, c), labels)
            losses.append(loss)
            start = end
        loss = torch.stack(losses).mean()
        return self.weight * loss

# Top N patch discrimination loss

@LOSS_REGISTRY.register("patch_discrimination_top_n")
class PatchDiscriminationLossTopN(Loss):
    """Loss function for patch discrimination task.

    This has lower memory consumption than the old patch discrimination loss.
    It does not support all discrimination loss.
    """

    name = "PatchDiscTopN"

    def __init__(self, tau: float = 0.1, pred2unit: bool = False, weight: float = 1.0, top_n: int  | None = 2000):
        """Initialize patch discrimination loss.

        Args:
            tau: the softmax temperature
            pred2unit: whether to standardize the predictions using batch statistics
            mask_other_samples: whether to apply the contrastive loss drawing samples
                from within a sample (True) or using all other instances in a batch (False).
                If this is False, then this is the AllDisc loss from the Galileo paper
            weight: the weight to apply to this loss
            top_n: the number of negatives to use for the loss
        """
        self.tau = tau
        self.pred2unit = pred2unit
        self.weight = weight
        self.top_n = top_n

    def compute(self, predictions: TokensAndMasks, targets: TokensAndMasks, **kwargs: Any) -> Tensor:
        all_preds, all_masks = predictions.flatten_tokens_and_masks()
        all_targets = targets.flatten_tokens_and_masks()[0]

        # keep only decoder tokens; no dummy batch dim
        keep = (all_masks == MaskValue.DECODER.value)
        pred = all_preds[keep]      # (N_tokens, D)
        target = all_targets[keep]  # (N_tokens, D)

        # optional standardization across tokens in this batch
        if self.pred2unit:
            mu = pred.mean(dim=0, keepdim=True)
            std = pred.std(dim=0, keepdim=True)
            pred = (pred - mu) / (std + 1e-4)

        pred = F.normalize(pred, p=2, dim=-1)
        target = F.normalize(target, p=2, dim=-1)

        counts = (all_masks == MaskValue.DECODER.value).sum(dim=-1).tolist()

        device = pred.device
        losses = []
        start = 0
        for c in counts:
            end = start + c
            if c == 0:
                logger.warning("No decoded values for this sample")
                start = end
                continue

            pred_sample = pred[start:end]      # (c, D)
            target_sample = target[start:end]  # (c, D)

            # logits (c, c); positive is the diagonal
            score_sample = (pred_sample @ target_sample.T) / self.tau

            if self.top_n is None:
                labels = torch.arange(c, dtype=torch.long, device=device)
                loss = F.cross_entropy(score_sample, labels, reduction="mean") * (self.tau * 2)
                losses.append(loss)
            else:
                if c == 1:
                    losses.append(score_sample.new_zeros(()))
                    start = end
                    continue

                k = min(self.top_n, c - 1)
                logger.info(f"k: {k}")
                # log c as well
                logger.info(f"c: {c}")
                neg_only = score_sample.masked_fill(
                    torch.eye(c, dtype=torch.bool, device=device), -torch.inf
                )
                topk_vals, _ = torch.topk(neg_only, k=k, dim=1)  # (c, k)


                pos_vals = score_sample.diag().unsqueeze(1)      # (c, 1)
                subset_scores = torch.cat([pos_vals, topk_vals], dim=1)  # (c, 1+k)

                labels = torch.zeros(c, dtype=torch.long, device=device)  # positive at col 0
                loss = F.cross_entropy(subset_scores, labels, reduction="mean") * (self.tau * 2)
                losses.append(loss)

            start = end

        if not losses:
            return pred.new_zeros(())

        return self.weight * torch.stack(losses).mean()



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

    def compute(self, predictions: TokensAndMasks, targets: TokensAndMasks, **kwargs: Any
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
        decoder_mask = all_masks == MaskValue.DECODER.value
        pred = all_preds[decoder_mask].unsqueeze(dim=0)
        target = all_targets[decoder_mask].unsqueeze(dim=0)
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
        all_preds, all_masks = predictions.flatten_tokens_and_masks()
        all_targets = targets.flatten_tokens_and_masks()[0]

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
            sim_matrix = torch.zeros(1, c, c, device=pred.device)
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
    ) -> Tensor:
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


@LOSS_REGISTRY.register("mae")
class MAELoss(Loss):
    """Loss function masked auto-encoding (reconstruction)."""

    name = "MAE"

    def __init__(
        self,
        loss_function: str = "MSELoss",
        only_decode: bool = True,
        weight: float = 1.0,
        **kwargs: Any,
    ):
        """Initialize MAE loss.

        Args:
            loss_function: pytorch loss to use
            only_decode: only calculate loss on DECODER masked tokens, otherwise all
            weight: the weight to apply to this loss
            **kwargs: arguments for pytorch loss constructor
        """
        self.only_decode = only_decode
        self.loss = getattr(torch.nn, loss_function)(reduction="sum", **kwargs)
        self.weight = weight

    # data: [B, H, W, T, C]
    def _flatten_helios_data(self, data: TokensAndMasks) -> tuple[Tensor, Tensor]:
        masks = []
        datas = []
        for modality in data.modalities:
            modality_spec = Modality.get(modality)
            pred = getattr(data, modality)
            if pred is not None:
                mask = getattr(data, data.get_masked_modality_name(modality))
                for idx, channel_set_idxs in enumerate(
                    modality_spec.bandsets_as_indices()
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
        self, predictions: TokensAndMasks, targets: MaskedHeliosSample, **kwargs: Any
    ) -> Tensor:
        """Compute MAE loss between predictions and targets.

        Args:
            predictions: Model predictions.
            targets: Ground truth targets.
            **kwargs: Additional keyword arguments.

        Returns:
            The computed loss value.
        """
        data, masks = self._flatten_helios_data(predictions)
        valid_dict = {}
        for modality in predictions.modalities:
            if getattr(predictions, modality) is not None:
                masked_name = predictions.get_masked_modality_name(modality)
                valid_dict[modality] = getattr(targets, modality)
                valid_dict[masked_name] = getattr(targets, masked_name)
        valid_targets = TokensAndMasks(**valid_dict)
        labels, label_masks = self._flatten_helios_data(valid_targets)
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
        all_preds, all_masks = predictions.flatten_tokens_and_masks()
        all_targets = targets.flatten_tokens_and_masks()[0]
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
        # online_encodings_a = predictions.pool_unmasked_tokens(
        #     PoolingType.MEAN, spatial_pooling=False
        # )
        # online_encodings_b = predictions.pool_unmasked_tokens(
        #     PoolingType.MEAN, spatial_pooling=False
        # )

        predictions = F.normalize(predictions, p=2, dim=-1)
        targets = F.normalize(targets, p=2, dim=-1)
        logits = predictions @ targets.transpose(-2, -1)

        logger.warning(logits.shape)

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
                all_preds, all_masks = predictions.flatten_tokens_and_masks()
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
