"""Supervision heads for direct supervision of decode-only modalities.

Each supervision modality uses its own decoder tokens directly (no cross-
modality pooling).  Decode-only spatial modalities have T=1 and BS=1, so
the T and BandSet dimensions are indexed directly.  Per-modality linear
heads predict max_patch_size x max_patch_size sub-patch grids that are
unfolded to pixel resolution.  When the actual patch_size < max_patch_size
the predictions are *downsampled* to match the target -- never upsampled.

Operates on decoder output to avoid pressuring the encoder to encode spatial
details at the expense of global semantic features.
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
from olmoearth_pretrain.data.constants import MISSING_VALUE
from olmoearth_pretrain.datatypes import MaskedOlmoEarthSample
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
    """Per-modality linear heads on per-modality decoder tokens.

    Forward path (per supervised modality):
      1. Grab the modality's own decoder tokens [B, P_H, P_W, T, 1, D]
         and squeeze BS -> [B, P_H, P_W, T, D]
      2. Linear head predicting max_patch_size^2 * C values per token
         (broadcasts over T)
      3. Unfold to [B, P_H * max_ps, P_W * max_ps, T, C]
      4. Downsample spatial dims to target pixel resolution when needed.

    For non-multitemporal modalities T=1.  For multitemporal modalities
    (e.g. NDVI) a separate prediction is produced for each timestep.
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

    def _get_batch_size(self, decoded: TokensAndMasks) -> int:
        for modality_name in decoded.modalities:
            t = getattr(decoded, modality_name)
            if t is not None:
                return t.shape[0]
        return 1

    def forward(
        self,
        decoded: TokensAndMasks,
        batch: MaskedOlmoEarthSample,
    ) -> dict[str, Tensor]:
        """Produce per-supervised-modality predictions at pixel resolution.

        Each modality head operates on that modality's own decoder tokens.
        Under FSDP every head must run on every rank, so we use dummy zero
        features when a modality's tokens are absent from the decoder output.

        Args:
            decoded: Decoder output TokensAndMasks.
            batch: The original batch (used to determine target spatial dims).

        Returns:
            Dictionary mapping supervised modality name to predictions.
            Shape is [B, H, W, T, C] (T preserved from decoder tokens).
        """
        mps = self.max_patch_size
        device = next(self.parameters()).device
        dtype = next(self.parameters()).dtype

        predictions: dict[str, Tensor] = {}
        for sup_name, head in self.heads.items():
            tokens = getattr(decoded, sup_name, None)  # [B, P_H, P_W, T, BS, D] | None

            if tokens is not None:
                features = tokens.mean(dim=-2)  # [B, P_H, P_W, T, D]
            else:
                batch_size = self._get_batch_size(decoded)
                features = torch.zeros(
                    batch_size, 1, 1, 1, head.in_features, device=device, dtype=dtype
                )

            num_channels = self.modality_configs[sup_name].num_output_channels
            raw = head(features)  # [B, P_H, P_W, T, mps^2 * C]

            output = rearrange(
                raw,
                "b ph pw t (c i j) -> b (ph i) (pw j) t c",
                c=num_channels,
                i=mps,
                j=mps,
            )  # [B, P_H*mps, P_W*mps, T, C]

            raw_target = getattr(batch, sup_name, None)
            if raw_target is not None:
                target_h, target_w = raw_target.shape[1], raw_target.shape[2]
                if output.shape[1] != target_h or output.shape[2] != target_w:
                    orig_dtype = output.dtype
                    b, h, w, t, c = output.shape
                    output = rearrange(output, "b h w t c -> (b t) c h w")
                    output = F.interpolate(
                        output.float(),
                        size=(target_h, target_w),
                        mode="bilinear",
                        align_corners=False,
                    ).to(orig_dtype)
                    output = rearrange(output, "(b t) c h w -> b h w t c", b=b, t=t)

            predictions[sup_name] = output

        return predictions


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

        # raw_target: [B, H, W, T, num_bands] -- keep all timesteps
        valid_mask = _build_valid_mask(raw_target)  # [B, H, W, T]

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
