"""Supervision heads for direct supervision of decode-only modalities.

Each supervision modality uses its own decoder tokens directly (no cross-
modality pooling).  Decode-only spatial modalities have T=1 and BS=1, so
the T and BandSet dimensions are indexed directly.  Per-modality linear
heads predict max_patch_size x max_patch_size sub-patch grids that are
unfolded to pixel resolution.  When the actual patch_size < max_patch_size
the predictions are *downsampled* to match the target -- never upsampled.

Operates on decoder output to avoid pressuring the encoder to encode spatial
details at the expense of global semantic features.

The ``latlon`` modality is a special case: the sample's (lat, lon) is regressed
as cartesian coordinates on the unit sphere (``LATLON_TARGET_DIM = 3``) from the
pooled features, so the target has no dateline discontinuity or pole degeneracy.
Since latlon is never a decoder modality, it is only supervisable with
``register_supervision=True`` (the non-spatial register path mean-pools the grid).
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
from olmoearth_pretrain.nn.flexi_vit import TokensAndMasks

logger = logging.getLogger(__name__)

# The latlon supervision target is a point on the unit sphere: (x, y, z).
LATLON_TARGET_DIM = 3


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
    """

    modality_configs: dict[str, SupervisionModalityConfig] = field(default_factory=dict)
    # When True, the heads read the encoder register grid (the Perceiver bottleneck)
    # instead of the per-modality decoder tokens, providing a spatial-salience signal to
    # the registers. embedding_dim is then the register dim (resolved by LatentMIMConfig).
    register_supervision: bool = False

    def __post_init__(self) -> None:
        """Coerce raw dicts in modality_configs to SupervisionModalityConfig instances."""
        self.modality_configs = {
            name: SupervisionModalityConfig(**cfg) if isinstance(cfg, dict) else cfg
            for name, cfg in self.modality_configs.items()
        }

    def build(self, embedding_dim: int, max_patch_size: int) -> SupervisionHead:
        """Build the supervision head.

        Args:
            embedding_dim: Dimension of the feature source (decoder embeddings, or the
                register dim when register_supervision is True).
            max_patch_size: Maximum patch size; each token predicts a
                max_patch_size x max_patch_size sub-patch grid.
        """
        return SupervisionHead(
            modality_configs=self.modality_configs,
            embedding_dim=embedding_dim,
            max_patch_size=max_patch_size,
            register_supervision=self.register_supervision,
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
        register_supervision: bool = False,
    ) -> None:
        """Initialize the supervision head."""
        super().__init__()
        self.modality_configs = modality_configs
        self.max_patch_size = max_patch_size
        self.register_supervision = register_supervision
        self._non_spatial_modalities: set[str] = set()
        self.heads = nn.ModuleDict()
        for name, cfg in modality_configs.items():
            if name == Modality.LATLON.name and (
                cfg.task_type != SupervisionTaskType.REGRESSION
                or cfg.num_output_channels != LATLON_TARGET_DIM
            ):
                raise ValueError(
                    "latlon supervision must be a regression onto unit-sphere xyz "
                    f"(num_output_channels={LATLON_TARGET_DIM}), got "
                    f"task_type={cfg.task_type} num_output_channels="
                    f"{cfg.num_output_channels}"
                )
            modality_spec = Modality.get(name)
            if modality_spec.is_spatial:
                # TODO: the max_patch_size^2 unfold is a holdover from decoder-token
                # supervision (each token = one real patch of up to max_patch_size px).
                # For register_supervision the registers are a coarse latent grid, not
                # patches, and the output is bilinearly interpolated to the target
                # resolution regardless — so this factor isn't needed. With register grids
                # finer than the patch grid (e.g. n=16/32 vs ~13) it over-produces
                # (n*max_patch_size > target) then downsamples, wasting head params.
                # Consider making it configurable (e.g. 1, or ceil(max_target / n)) in
                # register_supervision mode. Expected downstream effect: negligible (low-
                # weight nudge, interpolated output, head discarded after pretraining).
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
        register_grid: Tensor | None = None,
    ) -> dict[str, Tensor]:
        """Produce per-supervised-modality predictions at pixel resolution.

        Each modality head operates on that modality's own decoder tokens.
        Under FSDP every head must run on every rank, so we use dummy zero
        features when a modality's tokens are absent from the decoder output.

        Args:
            decoded: Decoder output TokensAndMasks.
            batch: The original batch (used to determine target spatial dims).
            register_grid: When register_supervision is True, the encoder register grid
                ``[B, n_h, n_w, register_dim]`` that all heads read from instead of the
                per-modality decoder tokens.

        Returns:
            Dictionary mapping supervised modality name to predictions.
            Shape is [B, H, W, T, C] (T preserved from decoder tokens).
        """
        mps = self.max_patch_size
        device = next(self.parameters()).device
        dtype = next(self.parameters()).dtype

        if self.register_supervision and register_grid is None:
            raise ValueError(
                "register_grid must be provided when register_supervision is True"
            )

        predictions: dict[str, Tensor] = {}
        for sup_name, head in self.heads.items():
            tokens = getattr(decoded, sup_name, None)

            if sup_name in self._non_spatial_modalities:
                # Non-spatial modality: features [B, D]
                if self.register_supervision:
                    assert register_grid is not None
                    features = register_grid.mean(dim=(1, 2))  # [B, D]
                elif tokens is not None:
                    features = tokens.mean(dim=-2)  # [B, D]
                else:
                    batch_size = self._get_batch_size(decoded)
                    features = torch.zeros(
                        batch_size, head.in_features, device=device, dtype=dtype
                    )
                output = head(features)  # [B, C]
            else:
                # Spatial modality: features [B, P_H, P_W, T, D]. In register-supervision
                # mode all heads read the shared register grid [B, n_h, n_w, D] (T=1).
                if self.register_supervision:
                    assert register_grid is not None
                    features = register_grid.unsqueeze(3)  # [B, n_h, n_w, 1, D]
                elif tokens is not None:
                    features = tokens.mean(dim=-2)  # [B, P_H, P_W, T, D]
                else:
                    batch_size = self._get_batch_size(decoded)
                    features = torch.zeros(
                        batch_size,
                        1,
                        1,
                        1,
                        head.in_features,
                        device=device,
                        dtype=dtype,
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

        # latlon targets are [B, 2] (lat, lon) but the prediction is [B, 3]
        # unit-sphere xyz, so the generic shape-matched paths below don't apply.
        if name == Modality.LATLON.name:
            per_modality_losses[name] = _latlon_regression_loss(
                pred, raw_target, regression_loss_type=cfg.regression_loss_type
            )
            continue

        valid_mask = _build_valid_mask(raw_target)

        if not valid_mask.any():
            per_modality_losses[name] = (0 * pred.sum()).to(dtype)
            continue

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


def _build_valid_mask(raw_target: Tensor) -> Tensor:
    """Bool mask that is True where all bands are non-missing [B, H, W]."""
    return (raw_target != MISSING_VALUE).all(dim=-1)


def _latlon_unit_xyz_target(raw_latlon: Tensor) -> Tensor:
    """Convert normalized (lat, lon) [B, 2] to unit-sphere xyz [B, 3].

    The dataloader normalizes latlon with the predefined min/max config
    (norm_configs/predefined.json: lat [-90, 90] -> [0, 1], lon [-180, 180] ->
    [0, 1]); undo that, then map to cartesian coordinates on the unit sphere.
    xyz is bounded in [-1, 1] and, unlike raw lat/lon, has no +-180 dateline
    discontinuity or pole degeneracy, so it is well-scaled for MSE/L1.
    """
    lat = torch.deg2rad(raw_latlon[..., 0].float() * 180.0 - 90.0)
    lon = torch.deg2rad(raw_latlon[..., 1].float() * 360.0 - 180.0)
    cos_lat = torch.cos(lat)
    return torch.stack(
        (cos_lat * torch.cos(lon), cos_lat * torch.sin(lon), torch.sin(lat)),
        dim=-1,
    )


def _latlon_regression_loss(
    pred: Tensor,
    raw_latlon: Tensor,
    regression_loss_type: str = "mse",
) -> Tensor:
    """Regression loss between predicted [B, 3] xyz and the sample's location.

    Samples whose latlon is missing-valued are excluded; if none are valid the
    loss is ``0 * pred.sum()`` so DDP/FSDP still see gradients for the head.
    """
    valid = (raw_latlon != MISSING_VALUE).all(dim=-1)  # [B]
    if not valid.any():
        return (0 * pred.sum()).to(pred.dtype)
    target = _latlon_unit_xyz_target(raw_latlon[valid])
    loss_fn = F.l1_loss if regression_loss_type == "l1" else F.mse_loss
    return loss_fn(pred[valid].float(), target).to(pred.dtype)


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

    if not pos_weight:
        pred_flat = pred[valid_expanded]
        target_flat = raw_target[valid_expanded]
        return F.binary_cross_entropy_with_logits(
            pred_flat.float(), target_flat.float()
        ).to(pred.dtype)

    # Per-channel positive rate from the batch's valid pixels.
    valid_mask_f = valid_mask.float().unsqueeze(-1)  # [B, H, W, T, 1]
    valid_count = valid_mask_f.sum().clamp(min=1.0)
    pos_count_per_ch = (raw_target.float() * valid_mask_f).sum(dim=(0, 1, 2, 3))  # [C]
    p_pos = (pos_count_per_ch / valid_count).clamp(min=1e-3, max=1.0 - 1e-3)
    pos_weight_tensor = (1.0 - p_pos) / p_pos  # [C], broadcasts on last dim

    elementwise_loss = F.binary_cross_entropy_with_logits(
        pred.float(),
        raw_target.float(),
        pos_weight=pos_weight_tensor,
        reduction="none",
    )  # [B, H, W, T, C]
    return elementwise_loss[valid_expanded].mean().to(pred.dtype)


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
        pred_flat = pred[valid_expanded]
        target_flat = raw_target[valid_expanded]
        loss_fn = F.l1_loss if regression_loss_type == "l1" else F.mse_loss
        return loss_fn(pred_flat.float(), target_flat.float()).to(pred.dtype)

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
    return elementwise[valid_full].mean().to(pred.dtype)
