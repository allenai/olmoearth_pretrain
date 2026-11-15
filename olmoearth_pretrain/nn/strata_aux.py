"""Auxiliary register-grid losses ported from olmoearth_strata.

Two training-only additions to the v1.3 recipe, each built only when its config is set:

* **recdisc** (:class:`RecDiscHead`): per-pixel discriminative temporal reconstruction.
  A small MLP sees ONE register (the location token at a pixel) plus a query date and
  must pick out that pixel's held-out acquisition among the sample's other held-out
  (pixel, date) queries (within-sample InfoNCE). The decoder cross-attends the whole
  register grid, so nothing else forces a single location token to carry its own
  pixel's time series. Queries are the sensor tokens the encoder did not see
  (TARGET_ENCODER_ONLY or DECODER; ``random_time_with_decode`` hides about half of every
  present sensor's tokens either way), and targets are the main MIM's frozen-projection
  pixel targets for them (computed each step on the unmasked input), centered per sample.

* **emainv** (:class:`EmaInvariance`): an EMA copy of the encoder (incl. the Perceiver)
  encodes the same sample with every encoder token visible, at the student's drawn
  latent stride; the student's register grid from the masked view, through a predictor
  MLP, regresses the teacher's grid with ``1 - cos`` (stop-grad, BYOL-style). No
  prototypes, centering or uniformity terms.
"""

import math
from copy import deepcopy
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from olmoearth_pretrain.config import Config
from olmoearth_pretrain.datatypes import MaskedOlmoEarthSample, MaskValue
from olmoearth_pretrain.nn.attention import DropPath
from olmoearth_pretrain.train.loss import ModalityPatchDiscriminationMaskedNegativesVec

_MASK_SUFFIX = "_mask"


# =========================================================================================
# recdisc
# =========================================================================================


def timestamp_features(timestamps: Tensor, num_harmonics: int) -> Tensor:
    """Day-of-year harmonics plus the year offset from the sample's mean year.

    Args:
        timestamps: ``[B, T, 3]`` ``(day, month - 1, year)`` (see ``encodings.py``).
        num_harmonics: Number of sin/cos day-of-year harmonics.

    Returns:
        ``[B, T, 2 * num_harmonics + 1]`` float features.
    """
    ts = timestamps.float()
    day, month, year = ts[..., 0], ts[..., 1], ts[..., 2]
    doy = month * (365.25 / 12.0) + day - 1.0
    k = torch.arange(1, num_harmonics + 1, device=ts.device, dtype=ts.dtype)
    angle = 2 * math.pi * doy[..., None] * k / 365.25
    year_offset = (year - year.mean(dim=1, keepdim=True))[..., None]
    return torch.cat([angle.sin(), angle.cos(), year_offset], dim=-1)


class RecDiscHead(nn.Module):
    """``MLP([LN(register) ; time features]) -> target space``."""

    def __init__(
        self,
        register_dim: int,
        target_dim: int,
        hidden_dim: int,
        num_harmonics: int,
    ) -> None:
        """Build the head."""
        super().__init__()
        self.num_harmonics = num_harmonics
        time_dim = 2 * num_harmonics + 1
        self.norm = nn.LayerNorm(register_dim)
        self.mlp = nn.Sequential(
            nn.Linear(register_dim + time_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, target_dim),
        )

    def forward(self, registers: Tensor, time_feats: Tensor) -> Tensor:
        """``[B, K, D_reg]``, ``[B, K, F]`` -> ``[B, K, D_target]``."""
        x = torch.cat([self.norm(registers), time_feats.to(registers.dtype)], dim=-1)
        return self.mlp(x)


@dataclass
class RecDiscHeadConfig(Config):
    """Configuration for :class:`RecDiscHead` and its loss.

    Args:
        modalities: Sensor modalities whose held-out tokens are queried.
        num_queries: Max held-out (pixel, date) queries per sample per modality.
        weight: Loss weight.
        tau: InfoNCE temperature.
        hidden_dim: Head MLP width.
        num_harmonics: Day-of-year harmonics in the time features.
        center_targets: Subtract each sample's mean target before normalising (the raw
            frozen-projection targets are near-collinear, cos ~0.985 within a sample).
        min_centered_norm_ratio: Queries whose centered target norm falls below this
            fraction of the mean raw norm are dropped (they equal the sample mean).
    """

    modalities: tuple[str, ...] = ("sentinel2_l2a",)
    num_queries: int = 64
    weight: float = 1.0
    tau: float = 0.1
    hidden_dim: int = 1024
    num_harmonics: int = 4
    center_targets: bool = True
    min_centered_norm_ratio: float = 1e-3

    def build(self, register_dim: int, target_dim: int) -> RecDiscHead:
        """Build the head."""
        return RecDiscHead(
            register_dim=register_dim,
            target_dim=target_dim,
            hidden_dim=self.hidden_dim,
            num_harmonics=self.num_harmonics,
        )


def recdisc_loss(
    head: RecDiscHead,
    config: RecDiscHeadConfig,
    register_grid: Tensor,
    latent_stride: int,
    patch_size: int,
    pixel_offsets: Tensor | None,
    targets: Tensor,
    token_mask: Tensor,
    timestamps: Tensor,
) -> tuple[Tensor, dict[str, Tensor]]:
    """Recdisc loss for one modality.

    Args:
        head: The head.
        config: Its config.
        register_grid: Student registers ``[B, H_r, W_r, D]`` (with grad).
        latent_stride: Pixels per register along each axis in this forward pass.
        patch_size: Patch size of this forward pass.
        pixel_offsets: ``[B, h, w, 2]`` ``(row, col)`` offset of each cell's drawn
            target pixel (``sample_pixel_offsets``), or None at patch size 1.
        targets: Frozen-projection targets ``[B, h, w, T, 1, D_t]`` (one band set).
        token_mask: Token-level mask ``[B, h, w, T, 1]`` (the decoder output's).
        timestamps: ``[B, T, 3]``.

    Returns:
        (weighted loss, metrics)
    """
    if targets.shape[-2] != 1:
        raise ValueError(f"recdisc needs one band set, got {targets.shape[-2]}")
    targets = targets[..., 0, :]
    token_mask = token_mask[..., 0]
    batch, h, w, num_t, target_dim = targets.shape
    _, h_r, w_r, reg_dim = register_grid.shape
    device = register_grid.device
    if h_r * latent_stride != h * patch_size or w_r * latent_stride != w * patch_size:
        raise ValueError(
            f"register grid {h_r}x{w_r} at stride {latent_stride} does not cover the "
            f"{h}x{w} token grid at patch size {patch_size}"
        )

    # Pixel coordinates of each cell's drawn target pixel -> the register holding it.
    rows = torch.arange(h, device=device).view(1, h, 1) * patch_size
    cols = torch.arange(w, device=device).view(1, 1, w) * patch_size
    if pixel_offsets is not None:
        rows = rows + pixel_offsets[..., 0].long()
        cols = cols + pixel_offsets[..., 1].long()
    else:
        rows = rows.expand(batch, h, w)
        cols = cols.expand(batch, h, w)
    reg_index = (rows // latent_stride) * w_r + (cols // latent_stride)  # [B, h, w]

    # Pick up to K held-out (cell, t) queries per sample, uniformly at random: tokens
    # the encoder did not see (TARGET_ENCODER_ONLY when the sensor was an encode band
    # set, DECODER when it was the decode one), never MISSING.
    valid = (token_mask == MaskValue.TARGET_ENCODER_ONLY.value) | (
        token_mask == MaskValue.DECODER.value
    )  # [B, h, w, T]
    valid_flat = valid.reshape(batch, h * w * num_t)
    k = min(config.num_queries, h * w * num_t)
    scores = torch.rand(valid_flat.shape, device=device)
    scores = scores.masked_fill(~valid_flat, -1.0)
    query_idx = scores.topk(k, dim=1).indices  # [B, K]
    query_valid = valid_flat.gather(1, query_idx)
    cell = query_idx // num_t
    t_idx = query_idx % num_t

    reg_flat = register_grid.reshape(batch, h_r * w_r, reg_dim)
    reg_of_query = reg_index.reshape(batch, h * w).gather(1, cell)
    query_registers = reg_flat.gather(
        1, reg_of_query.unsqueeze(-1).expand(-1, -1, reg_dim)
    )
    time_feats = timestamp_features(timestamps, config.num_harmonics)
    query_time = time_feats.gather(
        1, t_idx.unsqueeze(-1).expand(-1, -1, time_feats.shape[-1])
    )
    query_targets = (
        targets.reshape(batch, h * w * num_t, target_dim)
        .gather(1, query_idx.unsqueeze(-1).expand(-1, -1, target_dim))
        .float()
    )

    if config.center_targets:
        weights = query_valid.float().unsqueeze(-1)
        mean = (query_targets * weights).sum(1, keepdim=True) / weights.sum(
            1, keepdim=True
        ).clamp(min=1)
        raw_norm = (query_targets.norm(dim=-1) * query_valid.float()).sum(
            1, keepdim=True
        ) / query_valid.float().sum(1, keepdim=True).clamp(min=1)
        query_targets = query_targets - mean
        query_valid = query_valid & (
            query_targets.norm(dim=-1) > config.min_centered_norm_ratio * raw_norm
        )

    preds = head(query_registers, query_time)
    masks = torch.where(
        query_valid,
        torch.full_like(query_idx, MaskValue.DECODER.value),
        torch.full_like(query_idx, MaskValue.MISSING.value),
    )
    loss_fn = ModalityPatchDiscriminationMaskedNegativesVec(tau=config.tau)
    with torch.autocast(device.type, enabled=False):
        loss = loss_fn._compute_modality_loss_parallel(
            preds.float(), masks, query_targets, "recdisc"
        )
    metrics = {"queries_per_sample": query_valid.float().sum(1).mean().detach()}
    return config.weight * loss, metrics


# =========================================================================================
# emainv
# =========================================================================================


class EmaInvariancePredictor(nn.Module):
    """BYOL-style predictor ``LN -> Linear -> GELU -> Linear`` on the register grid."""

    def __init__(self, register_dim: int, hidden_dim: int) -> None:
        """Build the predictor."""
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(register_dim),
            nn.Linear(register_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, register_dim),
        )

    def forward(self, x: Tensor) -> Tensor:
        """``[..., D] -> [..., D]``."""
        return self.net(x)


@dataclass
class EmaInvarianceConfig(Config):
    """Configuration for the EMA-teacher register-grid invariance loss.

    Args:
        weight: Loss weight.
        ema_start: Teacher momentum at step 0.
        ema_end: Teacher momentum at the final step (linear in between, like
            ``LatentMIMTrainModule``'s target-encoder schedule).
        predictor_hidden_dim: Predictor MLP width.
    """

    weight: float = 1.0
    ema_start: float = 0.996
    ema_end: float = 1.0
    predictor_hidden_dim: int = 2048

    def build_predictor(self, register_dim: int) -> EmaInvariancePredictor:
        """Build the predictor."""
        return EmaInvariancePredictor(register_dim, self.predictor_hidden_dim)


def build_ema_teacher(encoder: nn.Module) -> nn.Module:
    """Frozen copy of ``encoder`` with every stochastic train-mode op disabled.

    The teacher stays in train mode: on this branch the encoder drops its attention mask
    in eval mode (the ragged-batch bug fixed by bc8321ab8 elsewhere), so eval mode would
    let padding leak into the teacher. Drop path, attention/MLP dropout and band dropout
    are zeroed instead, so the train-mode forward is deterministic.
    """
    teacher = deepcopy(encoder)
    for module in teacher.modules():
        if isinstance(module, DropPath):
            module.drop_prob = 0.0
        elif isinstance(module, nn.Dropout):
            module.p = 0.0
        if hasattr(module, "band_dropout_rate"):
            module.band_dropout_rate = 0.0
        if hasattr(module, "attn_drop") and isinstance(module.attn_drop, nn.Dropout):
            module.attn_drop.p = 0.0
    for p in teacher.parameters():
        p.requires_grad = False
    return teacher


def unmask_encoder_modalities(
    batch: MaskedOlmoEarthSample, decode_only_modalities: set[str]
) -> MaskedOlmoEarthSample:
    """Every non-missing token of an encoder modality becomes ONLINE_ENCODER.

    Decode-only modalities (the maps) keep their masks, so the encoder still skips them.
    """
    updates: dict[str, Any] = {}
    for name in batch.modalities:
        if name in decode_only_modalities:
            continue
        mask_name = f"{name}{_MASK_SUFFIX}"
        mask = getattr(batch, mask_name, None)
        if mask is None:
            continue
        missing = mask == MaskValue.MISSING.value
        updates[mask_name] = torch.where(
            missing, mask, torch.full_like(mask, MaskValue.ONLINE_ENCODER.value)
        )
    return batch._replace(**updates)


def ema_invariance_loss(
    predictor: EmaInvariancePredictor,
    student_grid: Tensor,
    teacher_grid: Tensor,
    weight: float,
) -> tuple[Tensor, dict[str, Tensor]]:
    """``weight * mean(1 - cos(predictor(student), stop_grad(teacher)))`` over cells."""
    if student_grid.shape != teacher_grid.shape:
        raise ValueError(
            f"student grid {tuple(student_grid.shape)} != teacher grid "
            f"{tuple(teacher_grid.shape)}"
        )
    pred = predictor(student_grid).float()
    target = teacher_grid.detach().float()
    cos = F.cosine_similarity(pred, target, dim=-1)
    loss = (1.0 - cos).mean()
    with torch.no_grad():
        raw_cos = F.cosine_similarity(student_grid.float(), target, dim=-1).mean()
        teacher_std = F.normalize(target.flatten(0, -2), dim=-1).std(dim=0).mean()
    metrics = {
        "loss_unweighted": loss.detach(),
        "student_teacher_cos": raw_cos,
        "teacher_unit_std": teacher_std,
    }
    return weight * loss, metrics


def ema_update(student: nn.Module, teacher: nn.Module, momentum: float) -> None:
    """``teacher = momentum * teacher + (1 - momentum) * student`` (params only)."""
    with torch.no_grad():
        for p, tp in zip(student.parameters(), teacher.parameters(), strict=True):
            tp.data.mul_(momentum).add_(p.data, alpha=1.0 - momentum)
