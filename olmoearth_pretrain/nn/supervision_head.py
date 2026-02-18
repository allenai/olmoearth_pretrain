"""Supervision heads for direct supervision of decode-only modalities.

Applied to encoder output: pool T and BandSets (masked mean over ONLINE_ENCODER
tokens), combine across encoded modalities, bilinearly upsample to pixel resolution,
then per-modality linear heads produce predictions.
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
from olmoearth_pretrain.datatypes import MaskedOlmoEarthSample, MaskValue
from olmoearth_pretrain.nn.flexi_vit import TokensAndMasks

logger = logging.getLogger(__name__)


class SupervisionTaskType(StrEnum):
    """Type of supervision task for a modality."""

    CLASSIFICATION = "classification"
    BINARY_CLASSIFICATION = "binary_classification"
    REGRESSION = "regression"


@dataclass
class SupervisionModalityConfig:
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
    """

    task_type: SupervisionTaskType
    num_output_channels: int
    weight: float = 1.0
    class_values: list[float] | None = None

    def __post_init__(self) -> None:
        """Validate and coerce task_type."""
        if isinstance(self.task_type, str):
            self.task_type = SupervisionTaskType(self.task_type)
        if (
            self.task_type == SupervisionTaskType.CLASSIFICATION
            and self.class_values is None
        ):
            raise ValueError("class_values must be provided for classification tasks")


@dataclass
class SupervisionHeadConfig(Config):
    """Configuration for the supervision head.

    Args:
        modality_configs: Mapping from modality name to its supervision config.
    """

    modality_configs: dict[str, SupervisionModalityConfig] = field(default_factory=dict)

    def build(self, embedding_dim: int) -> SupervisionHead:
        """Build the supervision head.

        Args:
            embedding_dim: Dimension of the encoder output embeddings.
        """
        return SupervisionHead(
            modality_configs=self.modality_configs,
            embedding_dim=embedding_dim,
        )


class SupervisionHead(nn.Module):
    """Per-modality linear heads applied to spatially-upsampled encoder features.

    Forward path:
      1. For each encoded spatial modality in the encoder output, compute a masked
         mean over T and BandSets (only ONLINE_ENCODER tokens) -> ``[B, P_H, P_W, D]``
      2. Bilinear-upsample each to the target pixel resolution and average across
         encoded modalities -> ``[B, H, W, D]``
      3. Per-supervised-modality linear head -> ``[B, H, W, C]``
    """

    def __init__(
        self,
        modality_configs: dict[str, SupervisionModalityConfig],
        embedding_dim: int,
    ) -> None:
        """Initialize the supervision head."""
        super().__init__()
        self.modality_configs = modality_configs
        self.heads = nn.ModuleDict()
        for name, cfg in modality_configs.items():
            self.heads[name] = nn.Linear(embedding_dim, cfg.num_output_channels)

    def forward(
        self,
        latent: TokensAndMasks,
        batch: MaskedOlmoEarthSample,
    ) -> dict[str, Tensor]:
        """Produce per-supervised-modality predictions at pixel resolution.

        Args:
            latent: Encoder output ``TokensAndMasks``.
            batch: The original batch (used to determine target spatial dims).

        Returns:
            Dictionary mapping supervised modality name -> predictions ``[B, H, W, C]``.
        """
        # Step 1+2: build a combined spatial feature map from all encoded modalities,
        # cached by target resolution so we only upsample once per unique (H, W).
        combined_features: dict[tuple[int, int], Tensor] = {}
        for sup_name in self.heads:
            raw_target = getattr(batch, sup_name, None)
            if raw_target is None:
                continue
            target_h, target_w = raw_target.shape[1], raw_target.shape[2]
            res_key = (target_h, target_w)
            if res_key in combined_features:
                continue
            feat = _pool_and_upsample_encoder_features(latent, target_h, target_w)
            if feat is not None:
                combined_features[res_key] = feat

        # Step 3: apply per-modality heads
        predictions: dict[str, Tensor] = {}
        for sup_name, head in self.heads.items():
            raw_target = getattr(batch, sup_name, None)
            if raw_target is None:
                continue
            target_h, target_w = raw_target.shape[1], raw_target.shape[2]
            res_key = (target_h, target_w)
            if res_key not in combined_features:
                continue
            predictions[sup_name] = head(combined_features[res_key])

        return predictions


def _pool_and_upsample_encoder_features(
    latent: TokensAndMasks,
    target_h: int,
    target_w: int,
) -> Tensor | None:
    """Pool encoder tokens across T/BandSets and upsample to target resolution.

    For each spatial modality with ONLINE_ENCODER tokens:
      - Masked mean over T and BandSets -> ``[B, P_H, P_W, D]``
      - Bilinear upsample to ``(target_h, target_w)``
    Then average across contributing modalities.

    Returns:
        ``[B, target_h, target_w, D]`` or None if no encoded spatial modalities.
    """
    upsampled: list[Tensor] = []
    for modality_name in latent.modalities:
        modality_spec = Modality.get(modality_name)
        if not modality_spec.is_spatial:
            continue

        tokens = getattr(latent, modality_name)  # [B, P_H, P_W, T, BS, D]
        mask_name = TokensAndMasks.get_masked_modality_name(modality_name)
        mask = getattr(latent, mask_name)  # [B, P_H, P_W, T, BS]

        encoder_mask = mask == MaskValue.ONLINE_ENCODER.value  # bool
        if not encoder_mask.any():
            continue

        pooled = _masked_mean_over_t_bs(tokens, encoder_mask)  # [B, P_H, P_W, D]

        pooled = rearrange(pooled, "b h w d -> b d h w")
        pooled = F.interpolate(
            pooled.float(),
            size=(target_h, target_w),
            mode="bilinear",
            align_corners=False,
        )
        pooled = rearrange(pooled, "b d h w -> b h w d")
        upsampled.append(pooled)

    if not upsampled:
        return None

    return torch.stack(upsampled).mean(dim=0)


def _masked_mean_over_t_bs(tokens: Tensor, mask: Tensor) -> Tensor:
    """Compute masked mean over the T and BandSets dimensions.

    Args:
        tokens: ``[B, P_H, P_W, T, BandSets, D]``
        mask: bool ``[B, P_H, P_W, T, BandSets]``

    Returns:
        ``[B, P_H, P_W, D]``
    """
    mask_expanded = mask.unsqueeze(-1)  # [B, P_H, P_W, T, BS, 1]
    masked_tokens = tokens * mask_expanded  # zero out non-encoder tokens
    summed = masked_tokens.sum(dim=(-3, -2))  # [B, P_H, P_W, D]
    count = mask.sum(dim=(-2, -1)).unsqueeze(-1).clamp(min=1)  # [B, P_H, P_W, 1]
    return summed / count


# ---------------------------------------------------------------------------
# Loss computation
# ---------------------------------------------------------------------------


def compute_supervision_loss(
    predictions: dict[str, Tensor],
    batch: MaskedOlmoEarthSample,
    modality_configs: dict[str, SupervisionModalityConfig],
) -> tuple[Tensor, dict[str, Tensor]]:
    """Compute the combined supervision loss across all supervised modalities.

    Args:
        predictions: Per-modality predictions from ``SupervisionHead.forward``.
        batch: The original batch containing raw pixel targets.
        modality_configs: Per-modality supervision configurations.

    Returns:
        total_loss: Weighted sum of per-modality losses.
        per_modality_losses: Dict of unweighted per-modality loss values (detached).
    """
    device = next(iter(predictions.values())).device
    total_loss = torch.zeros([], device=device)
    per_modality_losses: dict[str, Tensor] = {}

    for name, pred in predictions.items():
        cfg = modality_configs[name]
        raw_target = getattr(batch, name)
        # raw_target: [B, H, W, 1, num_bands] -> squeeze T dim
        raw_target = raw_target[:, :, :, 0, :]  # [B, H, W, num_bands]

        valid_mask = _build_valid_mask(raw_target)

        if not valid_mask.any():
            per_modality_losses[name] = torch.zeros([], device=device)
            continue

        if cfg.task_type == SupervisionTaskType.CLASSIFICATION:
            loss = _classification_loss(pred, raw_target, valid_mask, cfg)
        elif cfg.task_type == SupervisionTaskType.BINARY_CLASSIFICATION:
            loss = _binary_classification_loss(pred, raw_target, valid_mask)
        elif cfg.task_type == SupervisionTaskType.REGRESSION:
            loss = _regression_loss(pred, raw_target, valid_mask)
        else:
            raise ValueError(f"Unknown task type: {cfg.task_type}")

        per_modality_losses[name] = loss.detach()
        total_loss = total_loss + cfg.weight * loss

    return total_loss, per_modality_losses


def _build_valid_mask(raw_target: Tensor) -> Tensor:
    """Bool mask that is True where all bands are non-missing ``[B, H, W]``."""
    return (raw_target != MISSING_VALUE).all(dim=-1)


def _classification_loss(
    pred: Tensor,
    raw_target: Tensor,
    valid_mask: Tensor,
    cfg: SupervisionModalityConfig,
) -> Tensor:
    """Cross-entropy loss for single-band categorical modalities.

    Converts normalized float targets to integer class indices using
    ``cfg.class_values`` as the lookup table (nearest-value matching).
    """
    assert cfg.class_values is not None
    class_values = torch.tensor(
        cfg.class_values, device=pred.device, dtype=raw_target.dtype
    )
    # raw_target: [B, H, W, 1] -> [B, H, W]
    target_vals = raw_target[..., 0]
    distances = (target_vals.unsqueeze(-1) - class_values).abs()
    target_indices = distances.argmin(dim=-1)  # [B, H, W]

    pred_flat = pred[valid_mask]  # [N, num_classes]
    target_flat = target_indices[valid_mask]  # [N]
    return F.cross_entropy(pred_flat, target_flat)


def _binary_classification_loss(
    pred: Tensor,
    raw_target: Tensor,
    valid_mask: Tensor,
) -> Tensor:
    """BCE loss for multi-band binary modalities (e.g., OSM raster, WorldCereal)."""
    valid_expanded = valid_mask.unsqueeze(-1).expand_as(pred)
    pred_flat = pred[valid_expanded]
    target_flat = raw_target[valid_expanded]
    return F.binary_cross_entropy_with_logits(pred_flat, target_flat.float())


def _regression_loss(
    pred: Tensor,
    raw_target: Tensor,
    valid_mask: Tensor,
) -> Tensor:
    """MSE loss for continuous modalities."""
    valid_expanded = valid_mask.unsqueeze(-1).expand_as(pred)
    pred_flat = pred[valid_expanded]
    target_flat = raw_target[valid_expanded]
    return F.mse_loss(pred_flat, target_flat.float())
