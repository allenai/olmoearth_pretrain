"""Supervision heads for direct supervision of decode-only modalities.

Every head reads the encoder's register grid (the Perceiver bottleneck), so the
supervision gradient flows straight into the representation the decoder and the
downstream probes consume. Per-modality linear heads predict a
max_patch_size x max_patch_size sub-patch grid per register cell, unfolded and
then bilinearly resized to the target's pixel resolution; non-spatial modalities
read the mean-pooled grid and predict one vector per sample.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import StrEnum

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor

from olmoearth_pretrain.config import Config
from olmoearth_pretrain.data.constants import MISSING_VALUE, Modality
from olmoearth_pretrain.datatypes import MaskedOlmoEarthSample

logger = logging.getLogger(__name__)


class SupervisionTaskType(StrEnum):
    """Type of supervision task for a modality."""

    CLASSIFICATION = "classification"
    BINARY_CLASSIFICATION = "binary_classification"
    REGRESSION = "regression"


@dataclass
class SupervisionModalityConfig(Config):
    """Configuration for supervising a single modality.

    Args:
        task_type: The type of supervision task.
        num_output_channels: For classification: number of classes.
            For binary_classification: number of bands (each gets BCE).
            For regression: number of output channels (typically 1).
        weight: Loss weight for this modality.
        class_values: For classification only: list of normalized pixel values
            that map to class indices 0..N-1. Used to convert normalized targets
            to integer class labels.
        norm_pix_loss: For regression only. If True, apply MAE-style per-patch
            normalization to the target before computing MSE (mean/var pooled
            over the (max_patch_size*max_patch_size*C) values in each patch).
        pos_weight: For binary_classification only. If True, compute per-channel
            positive frequency from the batch's valid pixels and pass
            pos_weight = (1 - p) / p to BCE. Shifts the loss-minimizing constant
            solution off the class prior so the model has to learn spatial
            structure rather than predict-prior.
        regression_loss_type: For regression only. "mse" (default) uses
            F.mse_loss; "l1" uses F.l1_loss. L1 is more robust to long-tail
            targets like SRTM/canopy where MSE overweights extreme outliers.
            Matches AlphaEarth's choice (Table S2 of arXiv:2507.22291) of L1
            across all continuous reconstruction targets.
    """

    task_type: str  # stored as str for OmegaConf compat; coerced to SupervisionTaskType in __post_init__
    num_output_channels: int
    weight: float = 1.0
    class_values: list[float] | None = None
    norm_pix_loss: bool = False
    pos_weight: bool = False
    regression_loss_type: str = "mse"

    def __post_init__(self) -> None:
        """Validate and coerce task_type."""
        if isinstance(self.task_type, str):
            self.task_type = SupervisionTaskType(self.task_type)
        if (
            self.task_type == SupervisionTaskType.CLASSIFICATION
            and self.class_values is None
        ):
            raise ValueError("class_values must be provided for classification tasks")
        if self.regression_loss_type not in ("mse", "l1"):
            raise ValueError(
                f"regression_loss_type must be 'mse' or 'l1', got "
                f"{self.regression_loss_type!r}"
            )


@dataclass
class SupervisionHeadConfig(Config):
    """Configuration for the supervision head.

    Args:
        modality_configs: Mapping from modality name to its supervision config.
        spatial_unfold: Override for the spatial heads' sub-cell unfold factor (the
            ``max_patch_size**2`` grid each register cell predicts). With a
            PIXEL-resolution register grid the cells already sit at target resolution,
            so set it to 1 (one value per cell). None keeps ``max_patch_size``. Same
            field as on ``favyen/20260917-pixreg-v1_3``.
    """

    modality_configs: dict[str, SupervisionModalityConfig] = field(default_factory=dict)
    spatial_unfold: int | None = None

    def __post_init__(self) -> None:
        """Coerce raw dicts in modality_configs to SupervisionModalityConfig instances."""
        self.modality_configs = {
            name: SupervisionModalityConfig(**cfg) if isinstance(cfg, dict) else cfg
            for name, cfg in self.modality_configs.items()
        }
        if self.spatial_unfold is not None and self.spatial_unfold < 1:
            raise ValueError(f"spatial_unfold must be >= 1, got {self.spatial_unfold}")

    def build(self, embedding_dim: int, max_patch_size: int) -> SupervisionHead:
        """Build the supervision head.

        Args:
            embedding_dim: Width of the register grid the heads read (the bottleneck's
                register dim, resolved by LatentMIMConfig).
            max_patch_size: Maximum patch size; each register cell predicts a
                max_patch_size x max_patch_size sub-patch grid.
        """
        return SupervisionHead(
            modality_configs=self.modality_configs,
            embedding_dim=embedding_dim,
            max_patch_size=(
                self.spatial_unfold
                if self.spatial_unfold is not None
                else max_patch_size
            ),
        )


class SupervisionHead(nn.Module):
    """Per-modality linear heads on the encoder register grid.

    Forward path (per supervised spatial modality):
      1. Read the shared register grid ``[B, n_h, n_w, D]``.
      2. Linear head predicting max_patch_size^2 * C values per cell.
      3. Unfold to ``[B, n_h * max_ps, n_w * max_ps, 1, C]``.
      4. Bilinearly resize to the target's pixel resolution.

    Non-spatial modalities read the mean-pooled grid and predict ``[B, C]``.
    """

    def __init__(
        self,
        modality_configs: dict[str, SupervisionModalityConfig],
        embedding_dim: int,
        max_patch_size: int,
    ) -> None:
        """Initialize the supervision head."""
        super().__init__()
        self.modality_configs = modality_configs
        self.max_patch_size = max_patch_size
        self._non_spatial_modalities: set[str] = set()
        self.heads = nn.ModuleDict()
        for name, cfg in modality_configs.items():
            modality_spec = Modality.get(name)
            if modality_spec.is_spatial:
                # The max_patch_size^2 unfold predates register supervision (each decoder
                # token was one real patch of up to max_patch_size px). The registers are
                # a coarse latent grid and the output is interpolated to the target
                # resolution regardless, so the factor is not strictly needed; it is kept
                # because the shipped checkpoints were trained with these head shapes.
                out_dim = cfg.num_output_channels * max_patch_size * max_patch_size
            else:
                out_dim = cfg.num_output_channels
                self._non_spatial_modalities.add(name)
            self.heads[name] = nn.Linear(embedding_dim, out_dim)

        for name, cfg in modality_configs.items():
            if cfg.class_values is not None:
                self.register_buffer(
                    f"_class_values_{name}",
                    torch.tensor(cfg.class_values, dtype=torch.float32),
                )

    def get_class_values(self, name: str) -> Tensor:
        """Retrieve the cached class_values buffer for a modality."""
        return getattr(self, f"_class_values_{name}")

    @staticmethod
    def _maybe_interpolate_to_target(
        output: Tensor, raw_target: Tensor | None
    ) -> Tensor:
        """Bilinearly resize ``[B, H, W, T, C]`` predictions to the target's (H, W)."""
        if raw_target is None:
            return output
        target_h, target_w = raw_target.shape[1], raw_target.shape[2]
        if output.shape[1] == target_h and output.shape[2] == target_w:
            return output
        orig_dtype = output.dtype
        b, h, w, t, c = output.shape
        output = rearrange(output, "b h w t c -> (b t) c h w")
        output = F.interpolate(
            output.float(),
            size=(target_h, target_w),
            mode="bilinear",
            align_corners=False,
        ).to(orig_dtype)
        return rearrange(output, "(b t) c h w -> b h w t c", b=b, t=t)

    def forward(
        self, register_grid: Tensor, batch: MaskedOlmoEarthSample
    ) -> dict[str, Tensor]:
        """Produce per-supervised-modality predictions at pixel resolution.

        Every head runs on every call (FSDP needs each parameter touched on every
        rank), whether or not the batch carries that modality's target.

        Args:
            register_grid: The encoder register grid ``[B, n_h, n_w, register_dim]``.
            batch: The original batch (used to determine target spatial dims).

        Returns:
            Dictionary mapping supervised modality name to predictions: ``[B, H, W,
            1, C]`` for spatial modalities, ``[B, C]`` for non-spatial ones.
        """
        mps = self.max_patch_size
        predictions: dict[str, Tensor] = {}
        for sup_name, head in self.heads.items():
            if sup_name in self._non_spatial_modalities:
                output = head(register_grid.mean(dim=(1, 2)))  # [B, C]
            else:
                num_channels = self.modality_configs[sup_name].num_output_channels
                raw = head(register_grid.unsqueeze(3))  # [B, n_h, n_w, 1, mps^2 * C]
                output = rearrange(
                    raw,
                    "b ph pw t (c i j) -> b (ph i) (pw j) t c",
                    c=num_channels,
                    i=mps,
                    j=mps,
                )  # [B, n_h*mps, n_w*mps, 1, C]

                output = self._maybe_interpolate_to_target(
                    output, getattr(batch, sup_name, None)
                )

            predictions[sup_name] = output

        return predictions


# ---------------------------------------------------------------------------
# Loss computation
# ---------------------------------------------------------------------------


def _compute_per_modality_losses(
    predictions: dict[str, Tensor],
    batch: MaskedOlmoEarthSample,
    supervision_head: SupervisionHead,
) -> dict[str, Tensor]:
    """Compute per-modality supervision losses (non-detached, unweighted).

    Returns a dict of raw loss values suitable for external weighting.
    Modalities whose target is absent or entirely missing-valued contribute
    ``0 * pred.sum()`` so that FSDP still sees gradients for all parameters.
    """
    modality_configs = supervision_head.modality_configs
    first_pred = next(iter(predictions.values()))
    dtype = first_pred.dtype
    per_modality_losses: dict[str, Tensor] = {}

    for name, pred in predictions.items():
        cfg = modality_configs[name]
        raw_target = getattr(batch, name, None)

        if raw_target is None:
            per_modality_losses[name] = (0 * pred.sum()).to(dtype)
            continue

        # No early exit on an all-missing target (that needed a host sync): the losses
        # below are masked means, which give 0 with zero gradients when nothing is
        # valid -- the same value and gradient as the old ``0 * pred.sum()``.
        valid_mask = _build_valid_mask(raw_target)

        if cfg.task_type == SupervisionTaskType.CLASSIFICATION:
            class_values = supervision_head.get_class_values(name)
            loss = _classification_loss(pred, raw_target, valid_mask, class_values)
        elif cfg.task_type == SupervisionTaskType.BINARY_CLASSIFICATION:
            loss = _binary_classification_loss(
                pred, raw_target, valid_mask, pos_weight=cfg.pos_weight
            )
        elif cfg.task_type == SupervisionTaskType.REGRESSION:
            loss = _regression_loss(
                pred,
                raw_target,
                valid_mask,
                norm_pix_loss=cfg.norm_pix_loss,
                max_patch_size=supervision_head.max_patch_size,
                regression_loss_type=cfg.regression_loss_type,
            )
        else:
            raise ValueError(f"Unknown task type: {cfg.task_type}")

        per_modality_losses[name] = loss

    return per_modality_losses


def compute_supervision_loss(
    predictions: dict[str, Tensor],
    batch: MaskedOlmoEarthSample,
    supervision_head: SupervisionHead,
) -> tuple[Tensor, dict[str, Tensor]]:
    """Compute the combined supervision loss across all supervised modalities.

    Args:
        predictions: Per-modality predictions from SupervisionHead.forward.
        batch: The original batch containing raw pixel targets.
        supervision_head: The supervision head (used for configs and cached buffers).

    Returns:
        total_loss: Weighted sum of per-modality losses.
        per_modality_losses: Dict of unweighted per-modality loss values (detached).
    """
    raw_losses = _compute_per_modality_losses(predictions, batch, supervision_head)
    modality_configs = supervision_head.modality_configs
    first_pred = next(iter(predictions.values()))
    device = first_pred.device
    dtype = first_pred.dtype
    total_loss = torch.zeros([], device=device, dtype=dtype)
    per_modality_losses: dict[str, Tensor] = {}

    for name, loss in raw_losses.items():
        per_modality_losses[name] = loss.detach()
        total_loss = total_loss + modality_configs[name].weight * loss

    return total_loss, per_modality_losses


def _masked_mean(values: Tensor, mask: Tensor) -> Tensor:
    """Mean of ``values`` over ``mask`` (same shape) without boolean indexing.

    Equal to ``values[mask].mean()`` up to summation order, and 0 (not NaN) when the
    mask is empty. Masked-out entries are zeroed with ``where`` first so a non-finite
    value there cannot leak into the sum or the gradient.
    """
    maskf = mask.to(values.dtype)
    total = torch.where(mask, values, torch.zeros_like(values)).sum()
    return total / maskf.sum().clamp(min=1.0)


def _build_valid_mask(raw_target: Tensor) -> Tensor:
    """Bool mask that is True where all bands are non-missing [B, H, W]."""
    return (raw_target != MISSING_VALUE).all(dim=-1)


def _classification_loss(
    pred: Tensor,
    raw_target: Tensor,
    valid_mask: Tensor,
    class_values: Tensor,
) -> Tensor:
    """Cross-entropy loss for single-band categorical modalities.

    Converts normalized float targets to integer class indices using
    class_values as the lookup table (nearest-value matching).
    """
    class_values = class_values.to(dtype=raw_target.dtype)
    # raw_target: [B, H, W, 1] -> [B, H, W]
    target_vals = raw_target[..., 0]
    distances = (target_vals.unsqueeze(-1) - class_values).abs()
    target_indices = distances.argmin(dim=-1)  # [B, H, W]

    num_classes = pred.shape[-1]
    per_pixel = F.cross_entropy(
        pred.float().reshape(-1, num_classes),
        target_indices.reshape(-1),
        reduction="none",
    ).reshape(target_indices.shape)
    return _masked_mean(per_pixel, valid_mask).to(pred.dtype)


def _binary_classification_loss(
    pred: Tensor,
    raw_target: Tensor,
    valid_mask: Tensor,
    pos_weight: bool = False,
) -> Tensor:
    """BCE loss for multi-band binary modalities (e.g., OSM raster, WorldCereal).

    If pos_weight is True, computes per-channel positive frequency over this
    batch's valid pixels and applies pos_weight = (1 - p) / p in BCE. The
    loss-minimizing constant solution moves from sigmoid(z) = p (loss = entropy
    of prior) to sigmoid(z) = 0.5 (loss = log(2)), so the model can't get away
    with predict-prior collapse.
    """
    # pred, raw_target: [B, H, W, T, C]; valid_mask: [B, H, W, T]
    valid_expanded = valid_mask.unsqueeze(-1).expand_as(pred)
    # Missing targets are zeroed before the loss so they stay finite; they are then
    # excluded by the masked mean.
    safe_target = torch.where(
        valid_expanded,
        raw_target.float(),
        torch.zeros_like(raw_target, dtype=torch.float32),
    )

    if not pos_weight:
        elementwise = F.binary_cross_entropy_with_logits(
            pred.float(), safe_target, reduction="none"
        )
        return _masked_mean(elementwise, valid_expanded).to(pred.dtype)

    # Per-channel positive rate from the batch's valid pixels.
    valid_mask_f = valid_mask.float().unsqueeze(-1)  # [B, H, W, T, 1]
    valid_count = valid_mask_f.sum().clamp(min=1.0)
    pos_count_per_ch = (raw_target.float() * valid_mask_f).sum(dim=(0, 1, 2, 3))  # [C]
    p_pos = (pos_count_per_ch / valid_count).clamp(min=1e-3, max=1.0 - 1e-3)
    pos_weight_tensor = (1.0 - p_pos) / p_pos  # [C], broadcasts on last dim

    elementwise_loss = F.binary_cross_entropy_with_logits(
        pred.float(),
        safe_target,
        pos_weight=pos_weight_tensor,
        reduction="none",
    )  # [B, H, W, T, C]
    return _masked_mean(elementwise_loss, valid_expanded).to(pred.dtype)


def _regression_loss(
    pred: Tensor,
    raw_target: Tensor,
    valid_mask: Tensor,
    norm_pix_loss: bool = False,
    max_patch_size: int = 1,
    regression_loss_type: str = "mse",
) -> Tensor:
    """Regression loss for continuous modalities.

    regression_loss_type selects MSE (default) or L1. L1 is more robust to
    long-tail targets — matches AlphaEarth's choice across all continuous
    reconstruction targets in arXiv:2507.22291 Table S2.

    If norm_pix_loss is True, apply MAE-style per-patch normalization to the
    target before computing the loss. The image is grouped into
    max_patch_size x max_patch_size patches at target resolution; for each
    patch, mean and variance are pooled over the (max_patch_size^2 * C) values
    (across valid pixels only) and used to normalize that patch's target.
    """
    if not norm_pix_loss:
        valid_expanded = valid_mask.unsqueeze(-1).expand_as(pred)
        safe_target = torch.where(
            valid_expanded,
            raw_target.float(),
            torch.zeros_like(raw_target, dtype=torch.float32),
        )
        loss_fn = F.l1_loss if regression_loss_type == "l1" else F.mse_loss
        elementwise = loss_fn(pred.float(), safe_target, reduction="none")
        return _masked_mean(elementwise, valid_expanded).to(pred.dtype)

    b, h, w, t, c = pred.shape
    mps = max_patch_size
    if h % mps != 0 or w % mps != 0:
        raise ValueError(
            f"norm_pix_loss requires target H, W ({h}, {w}) divisible by "
            f"max_patch_size ({mps})"
        )

    pred_p = rearrange(pred, "b (ph i) (pw j) t c -> b ph pw t (i j c)", i=mps, j=mps)
    target_p = rearrange(
        raw_target, "b (ph i) (pw j) t c -> b ph pw t (i j c)", i=mps, j=mps
    )
    # Lift the spatial valid mask to per-(pixel, channel) and broadcast over T.
    valid_p = rearrange(valid_mask, "b (ph i) (pw j) -> b ph pw (i j)", i=mps, j=mps)
    valid_p_c = valid_p.unsqueeze(-1).expand(-1, -1, -1, -1, c)
    valid_p_c = rearrange(valid_p_c, "b ph pw n c -> b ph pw (n c)").unsqueeze(3)

    target_p_f = target_p.float()
    n_valid = valid_p_c.sum(dim=-1, keepdim=True).clamp(min=1).to(target_p_f.dtype)
    target_p_zeroed = target_p_f.masked_fill(~valid_p_c, 0.0)
    mean = target_p_zeroed.sum(dim=-1, keepdim=True) / n_valid
    diff = (target_p_f - mean).masked_fill(~valid_p_c, 0.0)
    var = (diff * diff).sum(dim=-1, keepdim=True) / n_valid
    target_normalized = (target_p_f - mean) / (var + 1e-6).sqrt()

    valid_full = valid_p_c.expand_as(pred_p)
    diff = pred_p.float() - target_normalized
    elementwise = diff.abs() if regression_loss_type == "l1" else diff * diff
    return _masked_mean(elementwise, valid_full).to(pred.dtype)
