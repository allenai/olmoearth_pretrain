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
from olmoearth_pretrain.datatypes import MaskedOlmoEarthSample, MaskValue
from olmoearth_pretrain.nn.encodings import timestamps_to_day_of_year

logger = logging.getLogger(__name__)


def _day_of_year_encoding(timestamps: Tensor, num_harmonics: int) -> Tensor:
    """Fixed sincos day-of-year basis for the time-conditioned heads.

    ``phi(t) = [sin(2*pi*k*doy/365.25), cos(2*pi*k*doy/365.25)] for k = 1..K``:
    periodic across year boundaries and year-invariant. A learned linear map over
    a Fourier basis IS a learned continuous-time embedding, so the MLP's first
    layer provides the mixing and nothing here needs to be learned -- which also
    means exact generalization to observation dates never seen in training.

    Args:
        timestamps: ``[B, T, 3]`` ``(day, month, year)`` timestamps.
        num_harmonics: Number of annual harmonics K.

    Returns:
        ``[B, T, 2 * num_harmonics]`` float tensor.
    """
    doy = timestamps_to_day_of_year(timestamps)  # [B, T]
    k = torch.arange(
        1, num_harmonics + 1, device=timestamps.device, dtype=torch.float32
    )
    angles = 2.0 * torch.pi * doy.unsqueeze(-1) * k / 365.25  # [B, T, K]
    return torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)


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
        time_conditioned: For MULTITEMPORAL spatial targets (e.g. the raw S2/S1
            bands). The register grid is a time-free 2D map, so a plain linear
            head can only produce one prediction per cell; a time-conditioned
            head instead predicts a value per (cell, timestep) by evaluating a
            small MLP on ``[register_cell ; phi(t)]``, where ``phi(t)`` is a
            fixed day-of-year sincos basis built from the sample's own
            timestamps. Because the prediction for cell (i, j) can only read
            ``z[i, j]``, the loss forces each cell to store its own trajectory,
            decodable given time -- exactly what a frozen per-cell probe needs.
            Variable timestep counts need no fixed output layer (the head is
            queried at exactly the observed times; per-timestep validity is
            handled by the MISSING_VALUE mask). Regression only.
        time_harmonics: For time_conditioned only. Number of annual harmonics
            K in the day-of-year encoding: ``phi(t) = [sin(2*pi*k*doy/365.25),
            cos(...)] for k = 1..K`` (2K features). K=4 spans phenology-scale
            temporal structure; the MLP's first layer learns the mixing.
        time_mlp_hidden_dim: For time_conditioned only. Hidden width of the
            two-layer MLP head. Kept small on purpose: the point of the loss
            is to force the REGISTER to store the trajectory, not to let a
            clever head reconstruct it from weak features.
        masked_timesteps_only: For time_conditioned only. If True, the loss scores
            only the ``(pixel, timestep)`` targets whose input unit the ONLINE
            encoder did NOT see (mask value ``DECODER`` or
            ``TARGET_ENCODER_ONLY``; ``MISSING`` stays excluded). Without it about
            half the targets are timesteps the register reads saw directly, making
            the head a partial copy task; with it the objective is temporal
            inpainting, aligned with the masked-modelling loss. The prediction is
            still computed at every observed timestep; only the loss mask changes.
            Requires the modality's mask tensor on the batch.
    """

    task_type: str  # stored as str for OmegaConf compat; coerced to SupervisionTaskType in __post_init__
    num_output_channels: int
    weight: float = 1.0
    class_values: list[float] | None = None
    norm_pix_loss: bool = False
    pos_weight: bool = False
    regression_loss_type: str = "mse"
    time_conditioned: bool = False
    time_harmonics: int = 4
    time_mlp_hidden_dim: int = 64
    masked_timesteps_only: bool = False

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
        if self.time_conditioned:
            if self.task_type != SupervisionTaskType.REGRESSION:
                raise ValueError(
                    "time_conditioned supervision only supports regression, got "
                    f"{self.task_type}"
                )
            if self.time_harmonics < 1:
                raise ValueError(
                    f"time_harmonics must be >= 1, got {self.time_harmonics}"
                )
            if self.time_mlp_hidden_dim < 1:
                raise ValueError(
                    f"time_mlp_hidden_dim must be >= 1, got {self.time_mlp_hidden_dim}"
                )
        elif self.masked_timesteps_only:
            raise ValueError(
                "masked_timesteps_only requires time_conditioned=True (only the "
                "time-conditioned heads predict per timestep)"
            )


@dataclass
class SupervisionHeadConfig(Config):
    """Configuration for the supervision head.

    Args:
        modality_configs: Mapping from modality name to its supervision config.
        spatial_unfold: Override for the spatial heads' sub-cell unfold factor (the
            ``max_patch_size**2`` grid each register cell predicts). With PIXEL-
            resolution register grids the cells already sit at the target
            resolution, so the default unfold would over-produce (``H*mps x W*mps``
            predictions immediately downsampled back to ``H x W``) -- set 1 there.
            ``None`` (default) keeps the ``max_patch_size`` unfold the shipped
            checkpoints were trained with.
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
                max_patch_size x max_patch_size sub-patch grid (overridden by
                ``spatial_unfold`` when set).
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

    Time-conditioned modalities run a small MLP on ``[register_cell ; phi(t)]`` at
    every observed timestep and predict ``[B, n_h, n_w, T, C]`` (per cell, no unfold),
    then resize to the target resolution like the other spatial heads.
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
        self._time_conditioned_modalities: set[str] = set()
        self.heads = nn.ModuleDict()
        for name, cfg in modality_configs.items():
            modality_spec = Modality.get(name)
            if cfg.time_conditioned:
                if not (modality_spec.is_spatial and modality_spec.is_multitemporal):
                    raise ValueError(
                        f"time_conditioned supervision requires a spatial "
                        f"multitemporal modality, got {name}"
                    )
                # Two-layer MLP on [register_cell ; phi(t)] -> C. Per-cell (no
                # max_patch_size^2 unfold): the output is bilinearly interpolated
                # to the target resolution like the other spatial heads.
                self._time_conditioned_modalities.add(name)
                self.heads[name] = nn.Sequential(
                    nn.Linear(
                        embedding_dim + 2 * cfg.time_harmonics,
                        cfg.time_mlp_hidden_dim,
                    ),
                    nn.GELU(),
                    nn.Linear(cfg.time_mlp_hidden_dim, cfg.num_output_channels),
                )
                continue
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
            if sup_name in self._time_conditioned_modalities:
                # Time-conditioned head: MLP([register_cell ; phi(t)]) evaluated at
                # every (cell, observed timestep). The prediction for cell (i, j)
                # can only read register_grid[:, i, j], so the fitted trajectory is
                # guaranteed to live in the cell the frozen probes read.
                if batch.timestamps is None:
                    raise ValueError(
                        f"time_conditioned supervision ({sup_name}) requires batch "
                        "timestamps to build the day-of-year encoding"
                    )
                cfg = self.modality_configs[sup_name]
                phi = _day_of_year_encoding(batch.timestamps, cfg.time_harmonics).to(
                    register_grid.dtype
                )  # [B, T, 2K]
                b, n_h, n_w, d = register_grid.shape
                t = phi.shape[1]
                features = torch.cat(
                    [
                        register_grid[:, :, :, None, :].expand(b, n_h, n_w, t, d),
                        phi[:, None, None, :, :].expand(b, n_h, n_w, t, -1),
                    ],
                    dim=-1,
                )
                output = head(features)  # [B, n_h, n_w, T, C]
                predictions[sup_name] = self._maybe_interpolate_to_target(
                    output, getattr(batch, sup_name, None)
                )
                continue

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

        valid_mask = _build_valid_mask(raw_target)
        if cfg.masked_timesteps_only:
            valid_mask = valid_mask & _build_non_online_mask(batch, name)

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


def _build_non_online_mask(batch: MaskedOlmoEarthSample, name: str) -> Tensor:
    """Bool ``[B, H, W, T]`` mask, True where the online encoder did NOT see the unit.

    Reads the modality's ``[B, H, W, T, band_sets]`` mask tensor and keeps the
    ``DECODER`` / ``TARGET_ENCODER_ONLY`` units (``MISSING`` is excluded here too,
    although the MISSING_VALUE target check already drops it). A pixel-timestep
    counts as masked if ANY of its band sets was hidden from the online encoder.
    """
    mask = getattr(batch, batch.get_masked_modality_name(name), None)
    if mask is None:
        raise ValueError(
            f"masked_timesteps_only supervision ({name}) requires the batch to carry "
            f"{batch.get_masked_modality_name(name)}"
        )
    non_online = (mask != MaskValue.ONLINE_ENCODER.value) & (
        mask != MaskValue.MISSING.value
    )
    return non_online.any(dim=-1)


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
