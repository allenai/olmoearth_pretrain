"""Supervision heads for direct supervision of decode-only modalities.

Applied to decoder output: pool T and BandSets (over all non-MISSING tokens),
combine across modalities, then per-modality linear heads predict
max_patch_size x max_patch_size sub-patch grids that are unfolded to pixel
resolution.  When the actual patch_size < max_patch_size the predictions are
*downsampled* to match the target -- never upsampled.

Unlike supervision-v2 which operates on encoder output, this version operates
on decoder output to avoid pressuring the encoder to encode spatial details
at the expense of global semantic features.
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

    def build(self, embedding_dim: int, max_patch_size: int) -> SupervisionHead:
        """Build the supervision head.

        Args:
            embedding_dim: Dimension of the decoder output embeddings.
            max_patch_size: Maximum patch size; each token predicts a
                max_patch_size x max_patch_size sub-patch grid.
        """
        return SupervisionHead(
            modality_configs=self.modality_configs,
            embedding_dim=embedding_dim,
            max_patch_size=max_patch_size,
        )


class SupervisionHead(nn.Module):
    """Per-modality linear heads on pooled decoder features.

    Forward path:
      1. For each spatial modality in the decoder output, compute a masked
         mean over T and BandSets (only non-MISSING tokens) -> [B, P_H, P_W, D]
      2. Average across contributing modalities -> [B, P_H, P_W, D]
      3. Per-supervised-modality linear head predicting max_patch_size^2 * C
         values per token -> [B, P_H, P_W, max_ps^2 * C]
      4. Unfold to [B, P_H * max_ps, P_W * max_ps, C]
      5. Downsample to target pixel resolution when prediction > target.
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
        self.heads = nn.ModuleDict()
        for name, cfg in modality_configs.items():
            out_dim = cfg.num_output_channels * max_patch_size * max_patch_size
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

    def forward(
        self,
        decoded: TokensAndMasks,
        batch: MaskedOlmoEarthSample,
    ) -> dict[str, Tensor]:
        """Produce per-supervised-modality predictions at pixel resolution.

        Under FSDP the linear heads are sharded across ranks, so every rank
        **must** call every head's forward pass even when a target modality is
        absent from the local batch. We achieve this by always running every
        head and only including results whose targets actually exist.

        Args:
            decoded: Decoder output TokensAndMasks.
            batch: The original batch (used to determine target spatial dims).

        Returns:
            Dictionary mapping supervised modality name -> predictions [B, H, W, C].
        """
        pooled_features = _pool_decoder_features(decoded)

        if pooled_features is None:
            first_head = next(iter(self.heads.values()))
            batch_size = 1
            for modality_name in decoded.modalities:
                t = getattr(decoded, modality_name)
                if t is not None:
                    batch_size = t.shape[0]
                    break
            pooled_features = torch.zeros(
                batch_size,
                1,
                1,
                first_head.in_features,
                device=next(self.parameters()).device,
                dtype=next(self.parameters()).dtype,
            )

        mps = self.max_patch_size
        predictions: dict[str, Tensor] = {}
        for sup_name, head in self.heads.items():
            num_channels = self.modality_configs[sup_name].num_output_channels
            raw = head(pooled_features)  # [B, P_H, P_W, mps^2 * C]

            output = rearrange(
                raw,
                "b ph pw (c i j) -> b (ph i) (pw j) c",
                c=num_channels,
                i=mps,
                j=mps,
            )  # [B, P_H*mps, P_W*mps, C]

            raw_target = getattr(batch, sup_name, None)
            if raw_target is not None:
                target_h, target_w = raw_target.shape[1], raw_target.shape[2]
                if output.shape[1] != target_h or output.shape[2] != target_w:
                    orig_dtype = output.dtype
                    output = rearrange(output, "b h w c -> b c h w")
                    output = F.interpolate(
                        output.float(),
                        size=(target_h, target_w),
                        mode="bilinear",
                        align_corners=False,
                    ).to(orig_dtype)
                    output = rearrange(output, "b c h w -> b h w c")

            predictions[sup_name] = output

        return predictions


def _pool_decoder_features(
    decoded: TokensAndMasks,
) -> Tensor | None:
    """Pool decoder tokens across T/BandSets at patch resolution.

    For each spatial modality with non-MISSING tokens:
      - Masked mean over T and BandSets -> [B, P_H, P_W, D]
    Then average across contributing modalities.

    Returns:
        [B, P_H, P_W, D] or None if no valid spatial modalities.
    """
    pooled_list: list[Tensor] = []
    for modality_name in decoded.modalities:
        modality_spec = Modality.get(modality_name)
        if not modality_spec.is_spatial:
            continue

        tokens = getattr(decoded, modality_name)  # [B, P_H, P_W, T, BS, D]
        mask_name = TokensAndMasks.get_masked_modality_name(modality_name)
        mask = getattr(decoded, mask_name)  # [B, P_H, P_W, T, BS]

        valid_mask = mask != MaskValue.MISSING.value
        if not valid_mask.any():
            continue

        pooled_list.append(
            _masked_mean_over_t_bs(tokens, valid_mask)
        )  # [B, P_H, P_W, D]

    if not pooled_list:
        return None

    return torch.stack(pooled_list).mean(dim=0)


def _masked_mean_over_t_bs(tokens: Tensor, mask: Tensor) -> Tensor:
    """Compute masked mean over the T and BandSets dimensions.

    Args:
        tokens: [B, P_H, P_W, T, BandSets, D]
        mask: bool [B, P_H, P_W, T, BandSets]

    Returns:
        [B, P_H, P_W, D]
    """
    mask_expanded = mask.unsqueeze(-1)  # [B, P_H, P_W, T, BS, 1]
    masked_tokens = tokens * mask_expanded
    summed = masked_tokens.sum(dim=(-3, -2))  # [B, P_H, P_W, D]
    count = mask.sum(dim=(-2, -1)).unsqueeze(-1).clamp(min=1)  # [B, P_H, P_W, 1]
    return summed / count


# ---------------------------------------------------------------------------
# Loss computation
# ---------------------------------------------------------------------------


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
    modality_configs = supervision_head.modality_configs
    first_pred = next(iter(predictions.values()))
    device = first_pred.device
    dtype = first_pred.dtype
    total_loss = torch.zeros([], device=device, dtype=dtype)
    per_modality_losses: dict[str, Tensor] = {}

    for name, pred in predictions.items():
        cfg = modality_configs[name]
        raw_target = getattr(batch, name, None)

        if raw_target is None:
            total_loss = total_loss + 0 * pred.sum()
            continue

        # raw_target: [B, H, W, 1, num_bands] -> squeeze T dim
        raw_target = raw_target[:, :, :, 0, :]  # [B, H, W, num_bands]

        valid_mask = _build_valid_mask(raw_target)

        if not valid_mask.any():
            total_loss = total_loss + 0 * pred.sum()
            per_modality_losses[name] = torch.zeros([], device=device)
            continue

        if cfg.task_type == SupervisionTaskType.CLASSIFICATION:
            class_values = supervision_head.get_class_values(name)
            loss = _classification_loss(pred, raw_target, valid_mask, class_values)
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

    pred_flat = pred[valid_mask]  # [N, num_classes]
    target_flat = target_indices[valid_mask]  # [N]
    return F.cross_entropy(pred_flat.float(), target_flat).to(pred.dtype)


def _binary_classification_loss(
    pred: Tensor,
    raw_target: Tensor,
    valid_mask: Tensor,
) -> Tensor:
    """BCE loss for multi-band binary modalities (e.g., OSM raster, WorldCereal)."""
    valid_expanded = valid_mask.unsqueeze(-1).expand_as(pred)
    pred_flat = pred[valid_expanded]
    target_flat = raw_target[valid_expanded]
    return F.binary_cross_entropy_with_logits(
        pred_flat.float(), target_flat.float()
    ).to(pred.dtype)


def _regression_loss(
    pred: Tensor,
    raw_target: Tensor,
    valid_mask: Tensor,
) -> Tensor:
    """MSE loss for continuous modalities."""
    valid_expanded = valid_mask.unsqueeze(-1).expand_as(pred)
    pred_flat = pred[valid_expanded]
    target_flat = raw_target[valid_expanded]
    return F.mse_loss(pred_flat.float(), target_flat.float()).to(pred.dtype)
