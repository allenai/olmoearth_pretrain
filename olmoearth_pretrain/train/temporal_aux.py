"""Time-hidden temporal auxiliary loss for multi-year samples (train-module side).

The run mixes 12-step monthly one-year samples with quarterly four-year samples. For
every multi-year sample of a microbatch this module

1. derives a year-chunk id for each present timestep from its timestamp
   (``round((days - d0) / 91) // 4``, ``d0`` = the sample's earliest present timestep;
   a sample is multi-year iff its present timesteps span more than 340 days),
2. crops one random spatial window of at most ``max_grid x max_grid`` tokens (the same
   window for every chunk of the sample),
3. builds an aux batch in which EACH YEAR CHUNK IS ITS OWN BATCH ELEMENT holding only
   that year's timesteps (padded with MISSING to the longest chunk), with a random
   token dropout as augmentation (two independent dropout views for
   ``gated_contrast``).

``LatentMIM.forward`` encodes the aux batch with the online encoder (so gradients
flow) and the :class:`~olmoearth_pretrain.nn.temporal_aux.TemporalAuxHead` projects
one embedding per (chunk, location). Because chunks are encoded separately the encoder
cannot see cross-year time gaps (3D RoPE is relative within a chunk; the additive
month embedding is seasonal only). This module then computes one of two losses:

* ``ordering`` (OPTIMUS-style): for every pair of year chunks (a earlier than b) of a
  sample and every location, BCE-with-logits on the antisymmetric head's
  ``s(a, b)`` with target 1. Stable or seasonal locations cannot be ordered, so their
  optimum is ``s = 0`` and they get no pressure.
* ``gated_contrast`` (CaCo-style): weighted InfoNCE between the two dropout views of
  each (sample, location, year). Negatives are the other samples (weight 1) and the
  same location in other years of the same sample, weighted by a stop-grad change
  gate ``g`` in [0, 1] from the ratio of cross-year to within-year (view) distance,
  standardized with an EMA of its mean / std. Other locations of the same sample are
  excluded (spatial autocorrelation). Optional TINC-style pull of ungated year pairs.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor

from olmoearth_pretrain.config import Config
from olmoearth_pretrain.data.constants import Modality
from olmoearth_pretrain.datatypes import MaskedOlmoEarthSample, MaskValue
from olmoearth_pretrain.nn.encodings import timestamps_to_days
from olmoearth_pretrain.nn.temporal_aux import TemporalAuxHead, TemporalAuxInput

logger = logging.getLogger(__name__)

VARIANTS = ("ordering", "gated_contrast")
METRIC_NAMESPACE = "temporal_aux"


# =========================================================================================
# Year chunks
# =========================================================================================


def spacetime_modalities(batch: MaskedOlmoEarthSample) -> list[str]:
    """The batch's spatial multitemporal modalities that have a mask."""
    return [
        m
        for m in batch.modalities
        if Modality.get(m).is_spacetime_varying
        and getattr(batch, batch.get_masked_modality_name(m)) is not None
    ]


def year_chunk_ids(
    timestamps: Tensor,
    present: Tensor,
    quarter_days: float = 91.0,
    quarters_per_year: int = 4,
) -> tuple[Tensor, Tensor]:
    """Year-chunk id of every present timestep, and each sample's time span.

    Args:
        timestamps: ``[B, T, 3]`` ``(day, month - 1, year)``.
        present: ``[B, T]`` bool presence of each timestep.
        quarter_days: Days per quarter step.
        quarters_per_year: Quarter steps per year chunk.

    Returns:
        ``chunk [B, T]`` long (``round((d - d0) / quarter_days) // quarters_per_year``
        with ``d0`` the earliest present day of the sample; -1 where not present) and
        ``span_days [B]`` float (latest minus earliest present day; 0 if none).
    """
    days = timestamps_to_days(timestamps)
    inf = torch.full_like(days, float("inf"))
    d0 = torch.where(present, days, inf).amin(dim=1)
    d1 = torch.where(present, days, -inf).amax(dim=1)
    any_present = present.any(dim=1)
    span = torch.where(any_present, d1 - d0, torch.zeros_like(d0))
    d0 = torch.where(any_present, d0, torch.zeros_like(d0))
    steps = torch.round((days - d0[:, None]) / quarter_days).long()
    chunk = torch.div(steps, quarters_per_year, rounding_mode="floor")
    chunk = torch.where(present, chunk, torch.full_like(chunk, -1))
    return chunk, span


# =========================================================================================
# Plan + aux batch
# =========================================================================================


@dataclass
class TemporalAuxPlan:
    """Which samples, chunks, timesteps and crop windows the aux pass uses.

    ``S`` aux samples, ``N`` chunk elements (one per (sample, year chunk), view 0),
    ``L`` = longest chunk, ``Q`` ordered chunk pairs (earlier, later) of the same
    sample. Element / pair indices refer to view 0; view ``v`` element ``n`` is aux
    batch row ``v * N + n``.
    """

    batch_index: Tensor  # [S] microbatch index of each aux sample
    crop_row0: Tensor  # [S] crop offset (tokens)
    crop_col0: Tensor  # [S]
    grid: tuple[int, int]  # (gh, gw) crop size (tokens)
    elem_sample: Tensor  # [N] aux sample id (0..S-1)
    elem_year: Tensor  # [N] year-chunk id
    elem_tidx: Tensor  # [N, L] timestep indices (padded with the chunk's last one)
    elem_tvalid: Tensor  # [N, L] bool
    pair_a: Tensor  # [Q] element index of the earlier chunk
    pair_b: Tensor  # [Q] element index of the later chunk

    @property
    def num_samples(self) -> int:
        """Number of multi-year samples in the aux pass."""
        return int(self.batch_index.numel())

    @property
    def num_elements(self) -> int:
        """Number of (sample, year chunk) elements (one view)."""
        return int(self.elem_sample.numel())

    def to(self, device: torch.device) -> TemporalAuxPlan:
        """Move the index tensors to ``device``."""
        return TemporalAuxPlan(
            batch_index=self.batch_index.to(device),
            crop_row0=self.crop_row0.to(device),
            crop_col0=self.crop_col0.to(device),
            grid=self.grid,
            elem_sample=self.elem_sample.to(device),
            elem_year=self.elem_year.to(device),
            elem_tidx=self.elem_tidx.to(device),
            elem_tvalid=self.elem_tvalid.to(device),
            pair_a=self.pair_a.to(device),
            pair_b=self.pair_b.to(device),
        )

    @staticmethod
    def empty(grid: tuple[int, int] = (0, 0)) -> TemporalAuxPlan:
        """A plan with no aux sample."""
        z = torch.zeros(0, dtype=torch.long)
        return TemporalAuxPlan(
            batch_index=z,
            crop_row0=z,
            crop_col0=z,
            grid=grid,
            elem_sample=z,
            elem_year=z,
            elem_tidx=torch.zeros(0, 1, dtype=torch.long),
            elem_tvalid=torch.zeros(0, 1, dtype=torch.bool),
            pair_a=z,
            pair_b=z,
        )


def _token_presence(
    batch: MaskedOlmoEarthSample, modalities: list[str], patch_size: int
) -> Tensor:
    """``[B, n_h, n_w, T]`` bool: some aux modality is present at the token."""
    present: Tensor | None = None
    for m in modalities:
        mask = getattr(batch, batch.get_masked_modality_name(m))
        factor = max(Modality.get(m).image_tile_size_factor, 1)
        step = patch_size * factor
        tok = (mask[:, ::step, ::step] != MaskValue.MISSING.value).any(dim=-1)
        present = tok if present is None else present | tok
    assert present is not None
    return present


def plan_temporal_aux(
    batch: MaskedOlmoEarthSample,
    patch_size: int,
    config: TemporalAuxConfig,
    modalities: list[str],
    generator: torch.Generator | None = None,
) -> TemporalAuxPlan:
    """Pick the eligible multi-year samples, their crop windows and year chunks.

    Runs on CPU (the batch is moved there if needed: only ``[B, T]`` timestamps and
    a token-level presence grid are read).

    Args:
        batch: The (masked) microbatch; only timestamps and masks are read.
        patch_size: Patch size of this step.
        config: The aux loss configuration.
        modalities: Modalities of the aux batch (spacetime, in the batch).
        generator: Optional CPU generator for the crop / sample draws.
    """
    if not modalities:
        return TemporalAuxPlan.empty()
    timestamps = batch.timestamps.detach().cpu()
    # Masks are constant within a patch: read them on the token grid only (cheap on
    # the CPU copy of the microbatch).
    present_any = (
        _token_presence(batch, spacetime_modalities(batch), patch_size)
        .any(dim=(1, 2))
        .cpu()
    )
    chunk, span = year_chunk_ids(
        timestamps, present_any, config.quarter_days, config.quarters_per_year
    )
    days = timestamps_to_days(timestamps)
    tok_present = _token_presence(batch, modalities, patch_size).cpu()
    _, n_h, n_w, _ = tok_present.shape
    gh, gw = min(config.max_grid, n_h), min(config.max_grid, n_w)
    if gh < 1 or gw < 1:
        return TemporalAuxPlan.empty()

    candidates = []
    for b in range(chunk.shape[0]):
        if span[b] <= config.multiyear_min_span_days:
            continue
        ids = chunk[b][chunk[b] >= 0]
        if torch.unique(ids).numel() >= config.min_year_chunks:
            candidates.append(b)
    if config.max_aux_samples is not None and len(candidates) > config.max_aux_samples:
        keep = torch.randperm(len(candidates), generator=generator)
        candidates = sorted(candidates[int(i)] for i in keep[: config.max_aux_samples])

    batch_index: list[int] = []
    row0s: list[int] = []
    col0s: list[int] = []
    elem_sample: list[int] = []
    elem_year: list[int] = []
    elem_t: list[list[int]] = []
    pair_a: list[int] = []
    pair_b: list[int] = []
    for b in candidates:
        r0 = int(torch.randint(0, n_h - gh + 1, (1,), generator=generator))
        c0 = int(torch.randint(0, n_w - gw + 1, (1,), generator=generator))
        crop_present = tok_present[b, r0 : r0 + gh, c0 : c0 + gw].any(dim=(0, 1))
        chunks = []
        for c in torch.unique(chunk[b][chunk[b] >= 0]).tolist():
            ts = ((chunk[b] == c) & crop_present).nonzero().flatten()
            if ts.numel() == 0:
                continue
            ts = ts[torch.argsort(days[b, ts], stable=True)]
            chunks.append((int(c), ts.tolist()))
        if len(chunks) < config.min_year_chunks:
            continue
        s = len(batch_index)
        batch_index.append(b)
        row0s.append(r0)
        col0s.append(c0)
        first = len(elem_sample)
        for c, ts in chunks:
            elem_sample.append(s)
            elem_year.append(c)
            elem_t.append(ts)
        for i in range(len(chunks)):
            for j in range(i + 1, len(chunks)):
                pair_a.append(first + i)
                pair_b.append(first + j)

    if not batch_index:
        return TemporalAuxPlan.empty(grid=(gh, gw))
    max_len = max(len(ts) for ts in elem_t)
    tidx = torch.tensor(
        [ts + [ts[-1]] * (max_len - len(ts)) for ts in elem_t], dtype=torch.long
    )
    tvalid = torch.tensor(
        [[True] * len(ts) + [False] * (max_len - len(ts)) for ts in elem_t],
        dtype=torch.bool,
    )
    as_long = lambda v: torch.tensor(v, dtype=torch.long)  # noqa: E731
    return TemporalAuxPlan(
        batch_index=as_long(batch_index),
        crop_row0=as_long(row0s),
        crop_col0=as_long(col0s),
        grid=(gh, gw),
        elem_sample=as_long(elem_sample),
        elem_year=as_long(elem_year),
        elem_tidx=tidx,
        elem_tvalid=tvalid,
        pair_a=as_long(pair_a),
        pair_b=as_long(pair_b),
    )


def build_temporal_aux_batch(
    batch: MaskedOlmoEarthSample,
    plan: TemporalAuxPlan,
    patch_size: int,
    modalities: list[str],
    keep_ratio: float,
    num_views: int = 1,
) -> MaskedOlmoEarthSample:
    """Gather the aux batch: one element per (view, year chunk), cropped, with token dropout.

    Every present token of the chunk is ONLINE_ENCODER except a random ``1 -
    keep_ratio`` fraction (independently per view, modality, location, timestep and
    band set), which becomes MISSING; padding timesteps are MISSING. An element whose
    dropout removed every token keeps all of its present tokens. Static and
    non-aux modalities are left out.
    """
    device = batch.timestamps.device
    plan = plan.to(device)
    gh, gw = plan.grid
    n = plan.num_elements
    big_n = n * num_views
    batch_rows = plan.batch_index[plan.elem_sample]  # [N]
    tidx = plan.elem_tidx  # [N, L]
    seq_len = tidx.shape[1]

    present_by_mod: dict[str, Tensor] = {}
    keep_by_mod: dict[str, Tensor] = {}
    data_by_mod: dict[str, Tensor] = {}
    for m in modalities:
        data = getattr(batch, m)
        mask = getattr(batch, batch.get_masked_modality_name(m))
        step = patch_size * Modality.get(m).image_tile_size_factor
        rows = (plan.crop_row0[plan.elem_sample] * step)[:, None] + torch.arange(
            gh * step, device=device
        )
        cols = (plan.crop_col0[plan.elem_sample] * step)[:, None] + torch.arange(
            gw * step, device=device
        )
        index = (
            batch_rows[:, None, None, None],
            rows[:, :, None, None],
            cols[:, None, :, None],
            tidx[:, None, None, :],
        )
        x = data[index]  # [N, gh*step, gw*step, L, C]
        mk = mask[index]  # [N, gh*step, gw*step, L, BS]
        present = (mk != MaskValue.MISSING.value) & plan.elem_tvalid[
            :, None, None, :, None
        ]
        x = x.repeat(num_views, *([1] * (x.dim() - 1)))
        present = present.repeat(num_views, *([1] * (present.dim() - 1)))
        if keep_ratio < 1.0:
            keep_tok = (
                torch.rand(big_n, gh, gw, seq_len, mk.shape[-1], device=device)
                < keep_ratio
            )
            keep = keep_tok.repeat_interleave(step, dim=1).repeat_interleave(
                step, dim=2
            )
        else:
            keep = torch.ones_like(present)
        data_by_mod[m] = x
        present_by_mod[m] = present
        keep_by_mod[m] = keep

    visible_any = torch.zeros(big_n, dtype=torch.bool, device=device)
    for m in modalities:
        visible_any |= (present_by_mod[m] & keep_by_mod[m]).flatten(1).any(dim=1)
    fallback = ~visible_any  # keep every present token of these elements

    fields: dict[str, Any] = {}
    for m in modalities:
        mask_dtype = getattr(batch, batch.get_masked_modality_name(m)).dtype
        visible = present_by_mod[m] & (
            keep_by_mod[m] | fallback.view(-1, *([1] * (keep_by_mod[m].dim() - 1)))
        )
        new_mask = torch.where(
            visible,
            torch.full_like(visible, MaskValue.ONLINE_ENCODER.value, dtype=mask_dtype),
            torch.full_like(visible, MaskValue.MISSING.value, dtype=mask_dtype),
        )
        fields[m] = data_by_mod[m]
        fields[batch.get_masked_modality_name(m)] = new_mask
    timestamps = batch.timestamps[batch_rows[:, None], tidx]  # [N, L, 3]
    fields["timestamps"] = timestamps.repeat(num_views, 1, 1)
    return MaskedOlmoEarthSample(**fields)


# =========================================================================================
# Losses
# =========================================================================================


def ordering_loss(logits: Tensor, valid: Tensor) -> tuple[Tensor, dict[str, Tensor]]:
    """BCE-with-logits on "b is later than a" (target 1) over the valid (pair, location)s.

    Args:
        logits: ``[Q, P]`` antisymmetric ordering logits ``s(a, b)``.
        valid: ``[Q, P]`` bool.
    """
    logits = logits.float()
    w = valid.float()
    denom = w.sum().clamp(min=1.0)
    bce = F.binary_cross_entropy_with_logits(
        logits, torch.ones_like(logits), reduction="none"
    )
    loss = (bce * w).sum() / denom
    with torch.no_grad():
        acc = ((logits > 0).float() * w).sum() / denom
        abs_logit = (logits.abs() * w).sum() / denom
    return loss, {"order_acc": acc, "order_abs_logit": abs_logit}


def weighted_info_nce(
    sim: Tensor,
    weights: Tensor,
    anchor_valid: Tensor,
    temperature: float,
) -> Tensor:
    """Mean over valid anchors of ``-log(e^{s+/t} / (e^{s+/t} + sum_j w_j e^{s_j/t}))``.

    Args:
        sim: ``[M, M]`` cosine similarities; row i is anchor i, column i its positive.
        weights: ``[M, M]`` non-negative negative weights (the diagonal is ignored;
            zero excludes a column).
        anchor_valid: ``[M]`` bool rows to average over.
        temperature: InfoNCE temperature.
    """
    m = sim.shape[0]
    eye = torch.eye(m, dtype=torch.bool, device=sim.device)
    logits = sim / temperature
    w = torch.where(eye, torch.ones_like(weights), weights)
    keep = w > 0
    log_w = torch.log(torch.where(keep, w, torch.ones_like(w)))
    masked = torch.where(keep, logits + log_w, torch.full_like(logits, -math.inf))
    per_anchor = torch.logsumexp(masked, dim=1) - logits.diagonal()
    a = anchor_valid.float()
    return (per_anchor * a).sum() / a.sum().clamp(min=1.0)


@dataclass
class GateStats:
    """Batch statistics of the log distance ratio (sum, sum of squares, count)."""

    total: Tensor
    total_sq: Tensor
    count: Tensor


class TemporalAuxLoss:
    """Builds the aux batch of a microbatch and computes the temporal aux loss."""

    def __init__(self, config: TemporalAuxConfig):
        """Build from a :class:`TemporalAuxConfig`."""
        config.validate()
        self.config = config
        self.num_views = 2 if config.variant == "gated_contrast" else 1

    def aux_modalities(
        self, batch: MaskedOlmoEarthSample, encoder_modalities: list[str] | None
    ) -> list[str]:
        """The batch's spacetime modalities the aux pass encodes."""
        mods = spacetime_modalities(batch)
        if encoder_modalities is not None:
            mods = [m for m in mods if m in encoder_modalities]
        if self.config.modalities is not None:
            mods = [m for m in mods if m in self.config.modalities]
        return [m for m in mods if Modality.get(m).image_tile_size_factor >= 1]

    def plan(
        self,
        batch: MaskedOlmoEarthSample,
        patch_size: int,
        encoder_modalities: list[str] | None = None,
    ) -> TemporalAuxPlan:
        """CPU plan of the aux pass for this microbatch."""
        return plan_temporal_aux(
            batch,
            patch_size,
            self.config,
            self.aux_modalities(batch, encoder_modalities),
        )

    def model_input(
        self,
        batch: MaskedOlmoEarthSample,
        plan: TemporalAuxPlan,
        patch_size: int,
        encoder_modalities: list[str] | None = None,
    ) -> TemporalAuxInput:
        """The ``temporal_aux`` argument of ``LatentMIM.forward`` (batch None if empty)."""
        if plan.num_elements == 0:
            return TemporalAuxInput(batch=None)
        device = batch.timestamps.device
        aux_batch = build_temporal_aux_batch(
            batch,
            plan,
            patch_size,
            self.aux_modalities(batch, encoder_modalities),
            keep_ratio=self.config.keep_ratio,
            num_views=self.num_views,
        )
        return TemporalAuxInput(
            batch=aux_batch,
            pair_a=plan.pair_a.to(device),
            pair_b=plan.pair_b.to(device),
        )

    # ------------------------------------------------------------------ gate
    @torch.no_grad()
    def _update_gate_stats(
        self,
        head: TemporalAuxHead,
        stats: GateStats,
        process_group: dist.ProcessGroup | None,
    ) -> None:
        """EMA of the mean / std of the log ratio, synchronized across ranks if asked.

        Every rank must call this on every microbatch (the all-reduce is a
        collective), with zero counts when it has no eligible sample.
        """
        packed = torch.stack([stats.total, stats.total_sq, stats.count]).float()
        if (
            self.config.sync_gate_stats
            and dist.is_available()
            and dist.is_initialized()
            and dist.get_world_size(process_group) > 1
        ):
            dist.all_reduce(packed, group=process_group)
        total, total_sq, count = packed.unbind()
        has = count > 0
        mean = total / count.clamp(min=1.0)
        var = (total_sq / count.clamp(min=1.0) - mean**2).clamp(min=0.0)
        std = var.sqrt()
        momentum = self.config.gate_ema_momentum
        init = head.gate_stats_initialized > 0
        new_mean = torch.where(
            init, momentum * head.gate_log_ratio_mean + (1 - momentum) * mean, mean
        )
        new_std = torch.where(
            init, momentum * head.gate_log_ratio_std + (1 - momentum) * std, std
        )
        head.gate_log_ratio_mean.copy_(
            torch.where(has, new_mean, head.gate_log_ratio_mean)
        )
        head.gate_log_ratio_std.copy_(
            torch.where(has, new_std, head.gate_log_ratio_std)
        )
        head.gate_stats_initialized.copy_(
            torch.where(has, torch.ones_like(count), head.gate_stats_initialized)
        )

    def gate(self, head: TemporalAuxHead, log_ratio: Tensor) -> Tensor:
        """``sigmoid((log r - mu - k sigma) / temp)`` with the EMA mean / std (detached)."""
        mu = head.gate_log_ratio_mean
        sigma = head.gate_log_ratio_std
        return torch.sigmoid(
            (log_ratio - mu - self.config.gate_k * sigma) / self.config.gate_temp
        ).detach()

    @staticmethod
    def log_distance_ratio(
        u1: Tensor, u2: Tensor, pair_a: Tensor, pair_b: Tensor, eps: float
    ) -> Tensor:
        """CaCo-style ``log(d_cross + eps) - log(d_within + eps)`` per (pair, location).

        Args:
            u1: ``[N, P, D]`` L2-normalized embeddings of view 1.
            u2: ``[N, P, D]`` L2-normalized embeddings of view 2.
            pair_a: ``[Q]`` element indices of the earlier year of each pair.
            pair_b: ``[Q]`` element indices of the later year.
            eps: Added to both distances inside the logs.

        ``d_cross = 1 - cos(mean-view z_a, mean-view z_b)``; ``d_within`` = mean over the
        two years of ``1 - cos(view 1, view 2)``.
        """
        with torch.no_grad():
            zbar = F.normalize(u1 + u2, dim=-1)
            d_cross = 1.0 - (zbar[pair_a] * zbar[pair_b]).sum(-1)
            d_view = 1.0 - (u1 * u2).sum(-1)  # [N, P]
            d_within = 0.5 * (d_view[pair_a] + d_view[pair_b])
            return torch.log(d_cross.clamp(min=0.0) + eps) - torch.log(
                d_within.clamp(min=0.0) + eps
            )

    # ------------------------------------------------------------------ compute
    def _zero_metrics(self, device: torch.device) -> dict[str, Tensor]:
        names = ["loss", "num_samples", "num_chunks"]
        if self.config.variant == "ordering":
            names += ["order_acc", "order_abs_logit"]
        else:
            names += [
                "info_nce",
                "tinc",
                "gate_mean",
                "gate_frac_open",
                "log_ratio_mean",
                "gate_ema_mean",
                "gate_ema_std",
                "pos_cos",
                "neg_cos",
                "cross_year_cos",
            ]
        return {n: torch.zeros((), device=device) for n in names}

    def compute(
        self,
        outputs: dict[str, Any],
        plan: TemporalAuxPlan,
        head: TemporalAuxHead,
        process_group: dist.ProcessGroup | None = None,
    ) -> tuple[Tensor, dict[str, Any]]:
        """Weighted aux loss term (incl. the zero parameter anchor) and its metrics.

        Always returns the same metric keys (metrics must match across ranks).

        Args:
            outputs: ``LatentMIM.temporal_aux_forward`` outputs.
            plan: The plan the aux batch was built from.
            head: The model's temporal aux head (its gate buffers are updated).
            process_group: Data-parallel group for the gate statistics.
        """
        anchor = outputs["param_anchor"].float()
        device = anchor.device
        metrics = self._zero_metrics(device)
        aux = torch.zeros((), device=device)
        proj = outputs.get("proj")
        n = plan.num_elements
        empty = proj is None or n == 0
        metrics["num_samples"] = torch.tensor(float(plan.num_samples), device=device)
        metrics["num_chunks"] = torch.tensor(float(n), device=device)

        if self.config.variant == "ordering":
            logits = outputs.get("order_logits")
            if not empty and logits is not None:
                plan_d = plan.to(device)
                valid = outputs["valid"]
                pair_valid = valid[plan_d.pair_a] & valid[plan_d.pair_b]
                aux, order_metrics = ordering_loss(logits, pair_valid)
                metrics.update(order_metrics)
        else:
            if empty:
                zero = torch.zeros((), device=device)
                self._update_gate_stats(
                    head, GateStats(zero, zero, zero), process_group
                )
            else:
                aux = self._gated_contrast(
                    proj,
                    outputs["valid"],
                    plan.to(device),
                    head,
                    process_group,
                    metrics,
                )
        metrics["loss"] = aux.detach()
        return self.config.weight * aux + anchor, {
            METRIC_NAMESPACE: {k: v.detach() for k, v in metrics.items()}
        }

    def _gated_contrast(
        self,
        proj: Tensor,
        valid: Tensor,
        plan: TemporalAuxPlan,
        head: TemporalAuxHead,
        process_group: dist.ProcessGroup | None,
        metrics: dict[str, Tensor],
    ) -> Tensor:
        cfg = self.config
        n = plan.num_elements
        u = F.normalize(proj.float(), dim=-1)
        u1, u2 = u[:n], u[n:]
        p = u.shape[1]
        loc_valid = valid[:n] & valid[n:]  # [N, P]

        # ---- change gate per (year pair, location)
        pa, pb = plan.pair_a, plan.pair_b
        log_ratio = self.log_distance_ratio(u1, u2, pa, pb, cfg.gate_eps)  # [Q, P]
        pair_valid = (loc_valid[pa] & loc_valid[pb]).float()
        stats = GateStats(
            total=(log_ratio * pair_valid).sum(),
            total_sq=(log_ratio**2 * pair_valid).sum(),
            count=pair_valid.sum(),
        )
        self._update_gate_stats(head, stats, process_group)
        g = self.gate(head, log_ratio)  # [Q, P]

        # ---- weighted InfoNCE over the flattened (element, location) grid
        m = n * p
        elem = torch.arange(n, device=u.device).repeat_interleave(p)
        samp = plan.elem_sample[elem]
        a_flat = u1.reshape(m, -1)
        c_flat = u2.reshape(m, -1)
        sim = a_flat @ c_flat.T  # [M, M]
        flat_valid = loc_valid.reshape(m)
        other_sample = samp[:, None] != samp[None, :]
        weights = other_sample.float()
        rows = (pa[:, None] * p + torch.arange(p, device=u.device)).reshape(-1)
        cols = (pb[:, None] * p + torch.arange(p, device=u.device)).reshape(-1)
        g_flat = g.reshape(-1)
        weights = weights.index_put((rows, cols), g_flat)
        weights = weights.index_put((cols, rows), g_flat)
        weights = weights * flat_valid[None, :].float()
        loss = weighted_info_nce(sim, weights, flat_valid, cfg.temperature)
        if cfg.symmetric_contrast:
            loss = 0.5 * (
                loss + weighted_info_nce(sim.T, weights.T, flat_valid, cfg.temperature)
            )
        aux = loss
        metrics["info_nce"] = loss.detach()

        # ---- optional TINC-style pull of ungated year pairs
        if cfg.tinc_weight > 0:
            zbar = F.normalize(u1 + u2, dim=-1)
            dist2 = ((zbar[pa] - zbar[pb]) ** 2).sum(-1)  # [Q, P]
            dyears = (plan.elem_year[pb] - plan.elem_year[pa]).abs().float()[:, None]
            pull = F.relu(dist2 - cfg.tinc_margin * dyears) * (1.0 - g)
            tinc = (pull * pair_valid).sum() / pair_valid.sum().clamp(min=1.0)
            aux = aux + cfg.tinc_weight * tinc
            metrics["tinc"] = tinc.detach()

        # ---- diagnostics
        with torch.no_grad():
            denom_pairs = pair_valid.sum().clamp(min=1.0)
            metrics["gate_mean"] = (g * pair_valid).sum() / denom_pairs
            metrics["gate_frac_open"] = ((g > 0.5).float() * pair_valid).sum() / (
                denom_pairs
            )
            metrics["log_ratio_mean"] = stats.total / denom_pairs
            metrics["gate_ema_mean"] = head.gate_log_ratio_mean.detach().clone()
            metrics["gate_ema_std"] = head.gate_log_ratio_std.detach().clone()
            fv = flat_valid.float()
            metrics["pos_cos"] = (sim.diagonal() * fv).sum() / fv.sum().clamp(min=1.0)
            neg = other_sample.float() * fv[:, None] * fv[None, :]
            metrics["neg_cos"] = (sim * neg).sum() / neg.sum().clamp(min=1.0)
            cross = sim[rows, cols]
            pv = pair_valid.reshape(-1)
            metrics["cross_year_cos"] = (cross * pv).sum() / pv.sum().clamp(min=1.0)
        return aux


@dataclass
class TemporalAuxConfig(Config):
    """Configuration of the time-hidden temporal aux loss (``LatentMIMTrainModuleConfig``).

    The model needs a matching ``LatentMIMConfig.temporal_aux_head_config`` (with
    ``ordering_dims`` for the ``ordering`` variant).

    Args:
        variant: ``"ordering"`` (OPTIMUS-style antisymmetric pair ordering) or
            ``"gated_contrast"`` (CaCo-style change-gated weighted InfoNCE).
        weight: Weight of the aux loss in the total loss.
        max_grid: Crop at most ``max_grid x max_grid`` tokens per multi-year sample.
        keep_ratio: Token keep probability of the aux pass's random token dropout.
        quarter_days, quarters_per_year: Year-chunk id =
            ``round((days - d0) / quarter_days) // quarters_per_year``.
        multiyear_min_span_days: A sample is multi-year iff its present timesteps span
            more than this many days.
        min_year_chunks: Minimum number of year chunks for a sample to contribute.
        max_aux_samples: Optional cap on aux samples per microbatch (random subset).
        modalities: Optional subset of spacetime modalities to encode in the aux pass
            (e.g. S2 only, to reduce missingness/sensor-era shortcuts). None = all.
        temperature: InfoNCE temperature (gated_contrast).
        gate_k, gate_temp: Gate ``sigmoid((log r - mu - k sigma) / temp)``.
        gate_ema_momentum: EMA momentum of the log-ratio mean / std.
        gate_eps: Epsilon inside the log distances.
        sync_gate_stats: All-reduce the per-microbatch log-ratio statistics across
            data-parallel ranks before the EMA update (identical gates on all ranks).
        symmetric_contrast: Average the InfoNCE over both view directions.
        tinc_weight: Weight of the TINC-style pull ``relu(||z_a - z_b||^2 -
            margin |dyears|) (1 - g)`` (0 = off).
        tinc_margin: Squared-distance margin per year (unit-sphere embeddings).
    """

    variant: str = "ordering"
    weight: float = 0.1
    max_grid: int = 4
    keep_ratio: float = 0.75
    quarter_days: float = 91.0
    quarters_per_year: int = 4
    multiyear_min_span_days: float = 340.0
    min_year_chunks: int = 2
    max_aux_samples: int | None = None
    modalities: list[str] | None = None
    temperature: float = 0.1
    gate_k: float = 1.0
    gate_temp: float = 0.25
    gate_ema_momentum: float = 0.99
    gate_eps: float = 1e-4
    sync_gate_stats: bool = True
    symmetric_contrast: bool = True
    tinc_weight: float = 0.0
    tinc_margin: float = 0.1

    def validate(self) -> None:
        """Validate the configuration."""
        if self.variant not in VARIANTS:
            raise ValueError(f"variant must be one of {VARIANTS}, got {self.variant}")
        if self.max_grid < 1:
            raise ValueError("max_grid must be >= 1")
        if not 0.0 < self.keep_ratio <= 1.0:
            raise ValueError("keep_ratio must be in (0, 1]")
        if self.min_year_chunks < 2:
            raise ValueError("min_year_chunks must be >= 2")
        if self.temperature <= 0 or self.gate_temp <= 0:
            raise ValueError("temperatures must be positive")

    def build(self) -> TemporalAuxLoss:
        """Build the :class:`TemporalAuxLoss`."""
        return TemporalAuxLoss(self)
