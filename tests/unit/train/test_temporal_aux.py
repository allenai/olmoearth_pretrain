"""Tests for the time-hidden temporal aux loss on multi-year samples."""

import datetime
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn.functional as F
from olmo_core.optim.adamw import AdamWConfig

from olmoearth_pretrain.config import Config
from olmoearth_pretrain.data.constants import Modality
from olmoearth_pretrain.data.transform import TransformConfig
from olmoearth_pretrain.datatypes import MaskedOlmoEarthSample, MaskValue
from olmoearth_pretrain.nn.flexi_vit import (
    AttentionPoolerConfig,
    EncoderConfig,
    PredictorConfig,
)
from olmoearth_pretrain.nn.latent_mim import LatentMIM, LatentMIMConfig
from olmoearth_pretrain.nn.location_token_distillation_head import (
    LocationTokenDistillationHeadConfig,
)
from olmoearth_pretrain.nn.temporal_aux import (
    TEMPORAL_AUX_OUTPUT_KEY,
    TemporalAuxHead,
    TemporalAuxHeadConfig,
    TemporalAuxInput,
)
from olmoearth_pretrain.nn.tokenization import (
    ModalityTokenization,
    TokenizationConfig,
)
from olmoearth_pretrain.train.loss import LossConfig
from olmoearth_pretrain.train.masking import MaskingConfig
from olmoearth_pretrain.train.temporal_aux import (
    GateStats,
    TemporalAuxConfig,
    TemporalAuxPlan,
    build_temporal_aux_batch,
    plan_temporal_aux,
    weighted_info_nce,
    year_chunk_ids,
)
from olmoearth_pretrain.train.train_module.latent_mim import (
    LatentMIMTrainModule,
    LatentMIMTrainModuleConfig,
)

ONLINE = MaskValue.ONLINE_ENCODER.value
DECODER = MaskValue.DECODER.value
MISSING = MaskValue.MISSING.value

S2 = Modality.SENTINEL2_L2A.name
S1 = Modality.SENTINEL1.name
WC = Modality.WORLDCOVER.name
MODALITIES = [S2, S1, WC]
# v1.2/v1.3 single-bandset S2 tokenization.
TOKENIZATION = TokenizationConfig(
    overrides={
        S2: ModalityTokenization(band_groups=[list(Modality.SENTINEL2_L2A.band_order)])
    }
)


def _ts(date: datetime.date) -> list[int]:
    return [date.day, date.month - 1, date.year]


def quarterly_dates(
    start: datetime.date, years: list[int], quarters: list[int]
) -> list[list[int]]:
    """Year-major (year, quarter) grid of a 16-step quarterly series from ``start``."""
    return [
        _ts(start + datetime.timedelta(days=91 * (4 * y + q)))
        for y in years
        for q in quarters
    ]


def monthly_dates(year: int, n: int) -> list[list[int]]:
    """``n`` monthly timestamps (15th of each month) of ``year``."""
    return [[15, m, year] for m in range(n)]


# =========================================================================================
# (a) year-chunk ids
# =========================================================================================


def test_year_chunk_ids_quarterly_and_monthly() -> None:
    """Quarterly 4-year subsets map to year chunks; monthly samples span < 340 days."""
    start = datetime.date(2021, 1, 20)
    # Sample 0: years {0, 1, 3} x quarters {1, 3}; slot 3 (year 1, quarter 3) missing.
    s0 = quarterly_dates(start, [0, 1, 3], [1, 3])
    present0 = [True, True, True, False, True, True]
    # Sample 1: all 16 quarters, take the first 6.
    s1 = quarterly_dates(start, [0, 1, 2, 3], [0, 1, 2, 3])[:6]
    # Sample 2: monthly, 6 present months; padded slots edge-copied and absent.
    s2 = monthly_dates(2019, 6)
    present2 = [True] * 6
    timestamps = torch.tensor([s0, s1, s2], dtype=torch.long)
    present = torch.tensor([present0, [True] * 6, present2])
    chunk, span = year_chunk_ids(timestamps, present)
    assert chunk[0].tolist() == [0, 0, 1, -1, 3, 3]
    assert chunk[1].tolist() == [0, 0, 0, 0, 1, 1]
    assert span[0] > 340 and span[1] > 340
    assert span[2] < 340
    # The earliest PRESENT timestep is the anchor: dropping year 0 renumbers.
    present_late = torch.tensor([[False, False, True, True, True, True]])
    chunk_late, span_late = year_chunk_ids(timestamps[:1], present_late)
    assert chunk_late[0].tolist() == [-1, -1, 0, 0, 2, 2]
    assert span_late[0] > 340
    # No present timestep at all: span 0, all -1.
    chunk_none, span_none = year_chunk_ids(timestamps[:1], torch.zeros(1, 6).bool())
    assert (chunk_none == -1).all() and span_none[0] == 0


def test_year_chunk_ids_full_16_quarters() -> None:
    """16 quarterly steps -> chunks 0..3, four each (the drift of ~91.3-day quarters)."""
    start = datetime.date(2022, 10, 15)
    dates = [_ts(start + datetime.timedelta(days=round(91.31 * k))) for k in range(16)]
    chunk, _ = year_chunk_ids(
        torch.tensor([dates], dtype=torch.long), torch.ones(1, 16).bool()
    )
    assert chunk[0].tolist() == [i // 4 for i in range(16)]


# =========================================================================================
# Synthetic batches
# =========================================================================================


def _expand_token_mask(tok: torch.Tensor, patch_size: int) -> torch.Tensor:
    """[B, h, w, T, BS] token mask -> pixel mask."""
    return tok.repeat_interleave(patch_size, dim=1).repeat_interleave(patch_size, dim=2)


def make_batch(
    patch_size: int = 4,
    hw_tokens: int = 4,
    seed: int = 0,
    include_multiyear: bool = True,
    s2_bandsets: int = 1,
) -> MaskedOlmoEarthSample:
    """B=3: [multi-year (2 years x 4 quarters, one missing), monthly, multi-year (3 years x 2)].

    Masks: present tokens are randomly ONLINE_ENCODER / DECODER at token level (a
    stand-in for the masking strategy); missing timesteps are MISSING.
    """
    g = torch.Generator().manual_seed(seed)
    t = 8
    start = datetime.date(2019, 2, 1)
    ts0 = quarterly_dates(start, [0, 2], [0, 1, 2, 3])
    present0 = [True, True, True, True, True, False, True, True]
    ts1 = monthly_dates(2020, 6) + [[15, 5, 2020]] * 2
    present1 = [True] * 6 + [False] * 2
    ts2 = quarterly_dates(start, [1, 2, 3], [1, 3]) + [[1, 0, 2023]] * 2
    present2 = [True] * 6 + [False] * 2
    timestamps = [ts0, ts1, ts2]
    presents = [present0, present1, present2]
    if not include_multiyear:
        timestamps = [ts1, ts1, ts1]
        presents = [present1, present1, present1]
    b = len(timestamps)
    h = w = hw_tokens * patch_size
    present = torch.tensor(presents)  # [B, T]
    fields: dict[str, Any] = {"timestamps": torch.tensor(timestamps, dtype=torch.long)}
    for name, bands, bandsets in [(S2, 12, s2_bandsets), (S1, 2, 1)]:
        data = torch.randn(b, h, w, t, bands, generator=g)
        tok = torch.where(
            torch.rand(b, hw_tokens, hw_tokens, t, bandsets, generator=g) < 0.5,
            ONLINE,
            DECODER,
        )
        tok = torch.where(present[:, None, None, :, None], tok, MISSING)
        fields[name] = data
        fields[f"{name}_mask"] = _expand_token_mask(tok, patch_size).float()
    fields[WC] = torch.rand(b, h, w, 1, 1, generator=g)
    fields[f"{WC}_mask"] = torch.full((b, h, w, 1, 1), float(DECODER))
    return MaskedOlmoEarthSample(**fields)


# =========================================================================================
# (b) plan + aux batch construction
# =========================================================================================


def test_plan_picks_multiyear_samples_and_chunks() -> None:
    """Only multi-year samples with >= 2 chunks; chunks hold that year's present steps."""
    batch = make_batch()
    config = TemporalAuxConfig(max_grid=2)
    plan = plan_temporal_aux(batch, 4, config, [S2, S1])
    assert plan.batch_index.tolist() == [0, 2]
    assert plan.grid == (2, 2)
    # Sample 0: year 0 = slots 0-3, year 2 = slots 4, 6, 7 (slot 5 missing).
    # Sample 2: years 1, 2, 3 relative to its own d0 -> chunks 0, 1, 2.
    assert plan.elem_sample.tolist() == [0, 0, 1, 1, 1]
    assert plan.elem_year.tolist() == [0, 2, 0, 1, 2]
    assert plan.elem_tidx.shape == (5, 4)
    assert plan.elem_tidx[0].tolist() == [0, 1, 2, 3]
    assert plan.elem_tidx[1, :3].tolist() == [4, 6, 7]
    assert plan.elem_tvalid.sum(dim=1).tolist() == [4, 3, 2, 2, 2]
    assert plan.elem_tidx[2, :2].tolist() == [0, 1]
    # Ordered pairs (earlier, later) within each sample only.
    pairs = list(zip(plan.pair_a.tolist(), plan.pair_b.tolist()))
    assert pairs == [(0, 1), (2, 3), (2, 4), (3, 4)]
    assert (plan.crop_row0 >= 0).all() and (plan.crop_row0 <= 2).all()


def test_plan_empty_without_multiyear() -> None:
    """A microbatch of monthly samples gives an empty plan."""
    batch = make_batch(include_multiyear=False)
    plan = plan_temporal_aux(batch, 4, TemporalAuxConfig(), [S2, S1])
    assert plan.num_elements == 0 and plan.num_samples == 0


def test_plan_max_aux_samples() -> None:
    """``max_aux_samples`` caps the number of aux samples."""
    batch = make_batch()
    plan = plan_temporal_aux(batch, 4, TemporalAuxConfig(max_aux_samples=1), [S2, S1])
    assert plan.num_samples == 1


def test_aux_batch_shapes_masks_and_crop() -> None:
    """Shapes, MISSING padding and the same crop window for every chunk of a sample."""
    ps = 4
    batch = make_batch(patch_size=ps)
    plan = plan_temporal_aux(batch, ps, TemporalAuxConfig(max_grid=2), [S2, S1])
    aux = build_temporal_aux_batch(batch, plan, ps, [S2, S1], keep_ratio=1.0)
    n, length = plan.elem_tidx.shape
    assert aux.sentinel2_l2a.shape == (n, 2 * ps, 2 * ps, length, 12)
    assert aux.sentinel2_l2a_mask.shape == (n, 2 * ps, 2 * ps, length, 1)
    assert aux.sentinel1.shape == (n, 2 * ps, 2 * ps, length, 2)
    assert aux.timestamps.shape == (n, length, 3)
    assert aux.worldcover is None and aux.latlon is None
    # keep_ratio=1: every present token is ONLINE_ENCODER (whatever the MIM mask was),
    # padding is MISSING.
    m = aux.sentinel2_l2a_mask
    valid = plan.elem_tvalid[:, None, None, :, None].expand_as(m)
    assert (m[valid] == ONLINE).all()
    assert (m[~valid] == MISSING).all()
    for e in range(n):
        b = int(plan.batch_index[plan.elem_sample[e]])
        r0 = int(plan.crop_row0[plan.elem_sample[e]]) * ps
        c0 = int(plan.crop_col0[plan.elem_sample[e]]) * ps
        for k in range(length):
            if not plan.elem_tvalid[e, k]:
                continue
            t = int(plan.elem_tidx[e, k])
            expected = batch.sentinel2_l2a[b, r0 : r0 + 2 * ps, c0 : c0 + 2 * ps, t]
            assert torch.equal(aux.sentinel2_l2a[e, :, :, k], expected)
            assert torch.equal(aux.timestamps[e, k], batch.timestamps[b, t])


def test_aux_batch_token_dropout_two_views() -> None:
    """Dropout is token-level (patch-constant), independent per view, roughly keep_ratio."""
    ps = 2
    batch = make_batch(patch_size=ps, hw_tokens=8, seed=1)
    plan = plan_temporal_aux(batch, ps, TemporalAuxConfig(max_grid=8), [S2, S1])
    n = plan.num_elements
    aux = build_temporal_aux_batch(
        batch, plan, ps, [S2, S1], keep_ratio=0.5, num_views=2
    )
    m = aux.sentinel2_l2a_mask
    assert m.shape[0] == 2 * n
    # patch-constant
    assert torch.equal(m, _expand_token_mask(m[:, ::ps, ::ps], ps))
    valid = plan.elem_tvalid[:, None, None, :, None].repeat(2, 1, 1, 1, 1).expand_as(m)
    frac = (m[valid] == ONLINE).float().mean().item()
    assert 0.35 < frac < 0.65
    assert not torch.equal(m[:n], m[n:])
    # Both views gather the same data.
    assert torch.equal(aux.sentinel2_l2a[:n], aux.sentinel2_l2a[n:])


def test_aux_batch_never_empty() -> None:
    """An element whose dropout removed every token keeps its present tokens."""
    ps = 4
    batch = make_batch(patch_size=ps)
    plan = plan_temporal_aux(batch, ps, TemporalAuxConfig(max_grid=1), [S2, S1])
    aux = build_temporal_aux_batch(batch, plan, ps, [S2, S1], keep_ratio=1e-6)
    visible = (aux.sentinel2_l2a_mask == ONLINE).flatten(1).any(1) | (
        aux.sentinel1_mask == ONLINE
    ).flatten(1).any(1)
    assert visible.all()


# =========================================================================================
# (c) ordering head antisymmetry
# =========================================================================================


def test_ordering_head_is_antisymmetric() -> None:
    """s(a, b) == -s(b, a) exactly, and s(a, a) == 0."""
    torch.manual_seed(0)
    head = TemporalAuxHeadConfig(
        projector_dims=[32, 16], ordering_dims=[16, 8, 16]
    ).build(in_dim=24)
    z_a = torch.randn(7, 5, 16)
    z_b = torch.randn(7, 5, 16)
    s_ab = head.order_logits(z_a, z_b)
    s_ba = head.order_logits(z_b, z_a)
    assert s_ab.shape == (7, 5)
    assert torch.equal(s_ab, -s_ba)
    assert torch.equal(head.order_logits(z_a, z_a), torch.zeros(7, 5))


# =========================================================================================
# (d) change gate
# =========================================================================================


def _gate_population(
    num_samples: int, p: int, d: int, changed_frac: float, seed: int
) -> tuple[torch.Tensor, torch.Tensor, TemporalAuxPlan, torch.Tensor]:
    """Two-year samples; a ``changed_frac`` share of (sample, location) changes."""
    g = torch.Generator().manual_seed(seed)
    base = torch.randn(num_samples, p, d, generator=g)
    changed = torch.rand(num_samples, p, generator=g) < changed_frac
    year1 = torch.where(
        changed[..., None], torch.randn(num_samples, p, d, generator=g), base
    )
    z = torch.stack([base, year1], dim=1).reshape(2 * num_samples, p, d)
    noise = 0.15
    u1 = F.normalize(z + noise * torch.randn(z.shape, generator=g), dim=-1)
    u2 = F.normalize(z + noise * torch.randn(z.shape, generator=g), dim=-1)
    plan = TemporalAuxPlan.empty()
    plan.elem_sample = torch.arange(num_samples).repeat_interleave(2)
    plan.elem_year = torch.tensor([0, 1]).repeat(num_samples)
    plan.pair_a = torch.arange(num_samples) * 2
    plan.pair_b = torch.arange(num_samples) * 2 + 1
    return u1, u2, plan, changed


def test_gate_closed_for_stable_open_for_changed() -> None:
    """After EMA warm-up, stable year pairs get g ~ 0 and changed ones g ~ 1."""
    config = TemporalAuxConfig(variant="gated_contrast")
    loss = config.build()
    head = TemporalAuxHeadConfig(projector_dims=[8]).build(in_dim=8)
    for step in range(200):
        u1, u2, plan, changed = _gate_population(32, 4, 16, 0.2, seed=step)
        log_ratio = loss.log_distance_ratio(
            u1, u2, plan.pair_a, plan.pair_b, config.gate_eps
        )
        stats = GateStats(
            log_ratio.sum(),
            (log_ratio**2).sum(),
            torch.tensor(float(log_ratio.numel())),
        )
        loss._update_gate_stats(head, stats, None)
    assert head.gate_stats_initialized.item() == 1
    u1, u2, plan, changed = _gate_population(64, 4, 16, 0.2, seed=999)
    log_ratio = loss.log_distance_ratio(
        u1, u2, plan.pair_a, plan.pair_b, config.gate_eps
    )
    gate = loss.gate(head, log_ratio)
    assert gate[~changed].mean() < 0.05
    assert gate[changed].mean() > 0.95
    # Identical years (no change anywhere) -> gate closed.
    u1, u2, plan, _ = _gate_population(64, 4, 16, 0.0, seed=1000)
    log_ratio = loss.log_distance_ratio(
        u1, u2, plan.pair_a, plan.pair_b, config.gate_eps
    )
    assert loss.gate(head, log_ratio).mean() < 0.05


def test_gate_stats_ignore_empty_microbatch() -> None:
    """Zero counts leave the EMA untouched (and it initializes on the first data)."""
    loss = TemporalAuxConfig(variant="gated_contrast").build()
    head = TemporalAuxHeadConfig(projector_dims=[8]).build(in_dim=8)
    zero = torch.zeros(())
    loss._update_gate_stats(head, GateStats(zero, zero, zero), None)
    assert head.gate_stats_initialized.item() == 0
    vals = torch.tensor([1.0, 3.0])
    loss._update_gate_stats(
        head, GateStats(vals.sum(), (vals**2).sum(), torch.tensor(2.0)), None
    )
    assert head.gate_log_ratio_mean.item() == pytest.approx(2.0)
    assert head.gate_log_ratio_std.item() == pytest.approx(1.0)
    loss._update_gate_stats(head, GateStats(zero, zero, zero), None)
    assert head.gate_log_ratio_mean.item() == pytest.approx(2.0)


# =========================================================================================
# (e) weighted InfoNCE
# =========================================================================================


def test_weighted_info_nce_matches_cross_entropy_with_unit_weights() -> None:
    """All weights 1 = the standard InfoNCE (cross-entropy on the diagonal)."""
    torch.manual_seed(0)
    a = F.normalize(torch.randn(10, 8), dim=-1)
    c = F.normalize(torch.randn(10, 8), dim=-1)
    sim = a @ c.T
    tau = 0.1
    ours = weighted_info_nce(sim, torch.ones(10, 10), torch.ones(10).bool(), tau)
    ref = F.cross_entropy(sim / tau, torch.arange(10))
    assert ours.item() == pytest.approx(ref.item(), rel=1e-5)


def test_weighted_info_nce_weights() -> None:
    """Weight 0 removes a negative (no gradient); weight w adds log w to its logit."""
    torch.manual_seed(1)
    sim = torch.randn(6, 6).clamp(-1, 1).requires_grad_(True)
    tau = 0.2
    w = torch.ones(6, 6)
    w[0, 3] = 0.0
    w[1, 2] = 0.25
    valid = torch.ones(6).bool()
    loss = weighted_info_nce(sim, w, valid, tau)
    loss.backward()
    assert sim.grad is not None
    assert sim.grad[0, 3] == 0
    # Reference: explicit per-anchor formula.
    ref = 0.0
    for i in range(6):
        pos = torch.exp(sim[i, i] / tau)
        neg = sum(w[i, j] * torch.exp(sim[i, j] / tau) for j in range(6) if j != i)
        ref += -torch.log(pos / (pos + neg))
    assert loss.item() == pytest.approx((ref / 6).item(), rel=1e-5)
    # A lower weight on a negative lowers the loss.
    w2 = w.clone()
    w2[1, 2] = 1.0
    assert weighted_info_nce(sim, w2, valid, tau) > loss
    # Invalid anchors are excluded from the mean.
    only0 = torch.zeros(6).bool()
    only0[0] = True
    pos = torch.exp(sim[0, 0] / tau)
    neg = sum(w[0, j] * torch.exp(sim[0, j] / tau) for j in range(1, 6))
    expected0 = -torch.log(pos / (pos + neg))
    assert weighted_info_nce(sim, w, only0, tau).item() == pytest.approx(
        expected0.item(), rel=1e-5
    )


# =========================================================================================
# (f) tiny model forward + backward through the train module
# =========================================================================================


def tiny_model_config(
    variant: str, embedding_source: str = "location_tokens"
) -> LatentMIMConfig:
    """A tiny v1.3-shaped model: 3D mixed RoPE, AttentionPooler + student, frozen target."""
    encoder = EncoderConfig(
        supported_modality_names=MODALITIES,
        embedding_size=32,
        num_heads=2,
        depth=2,
        mlp_ratio=2.0,
        max_patch_size=8,
        min_patch_size=1,
        max_sequence_length=12,
        drop_path=0.0,
        position_encoding="rope_3d_mixed",
        rope_temporal_coordinate_scale=1.0 / 30.0,
        tokenization_config=TOKENIZATION,
        attention_pooler_config=AttentionPoolerConfig(
            pooler_dim=16,
            pooler_depth=1,
            attn_dim=32,
            student_dims=[8, 4],
            student_output_norm=True,
        ),
    )
    decoder = PredictorConfig(
        supported_modality_names=MODALITIES,
        encoder_embedding_size=32,
        decoder_embedding_size=16,
        num_heads=2,
        depth=1,
        mlp_ratio=2.0,
        max_sequence_length=12,
        drop_path=0.0,
        position_encoding="rope",
        tokenization_config=TOKENIZATION,
        use_attention_pooler=True,
        pooler_dim=16,
    )
    return LatentMIMConfig(
        encoder_config=encoder,
        decoder_config=decoder,
        projection_only_target=True,
        location_token_distillation_head_config=LocationTokenDistillationHeadConfig(),
        temporal_aux_head_config=TemporalAuxHeadConfig(
            embedding_source=embedding_source,
            projector_dims=[32, 16],
            ordering_dims=[16, 8, 16] if variant == "ordering" else None,
        ),
    )


def build_train_module(
    model: LatentMIM, aux_config: TemporalAuxConfig
) -> LatentMIMTrainModule:
    """A CPU LatentMIM train module (pixel targets) with the temporal aux loss."""
    config = LatentMIMTrainModuleConfig(
        optim_config=AdamWConfig(lr=1e-4, weight_decay=0.0),
        rank_microbatch_size=8,
        loss_config=LossConfig(loss_config={"type": "patch_discrimination"}),
        masking_config=MaskingConfig(strategy_config={"type": "random"}),
        token_exit_cfg={m: 0 for m in Modality.names()},
        ema_decay=(1.0, 1.0),
        max_grad_norm=1.0,
        transform_config=TransformConfig(transform_type="no_transform"),
        pixel_targets=True,
        temporal_aux_config=aux_config,
    )
    with patch("olmoearth_pretrain.train.train_module.train_module.build_world_mesh"):
        return config.build(model, device=torch.device("cpu"))


class _MockTrainer:
    def __init__(self) -> None:
        self.metrics: dict[str, Any] = {}
        self.global_step = 0
        self.max_steps = 100

    def record_metric(
        self, name: str, value: Any, reduce_type: Any = None, namespace: Any = None
    ) -> None:  # noqa: E501
        self.metrics[name] = value


@pytest.mark.parametrize(
    "variant,embedding_source",
    [
        ("ordering", "location_tokens"),
        ("gated_contrast", "location_tokens"),
        ("ordering", "mean_tokens"),
    ],
)
def test_train_module_forward_backward(variant: str, embedding_source: str) -> None:
    """Finite loss, aux metrics, and gradients reach the aux head and the encoder."""
    torch.manual_seed(0)
    model = tiny_model_config(variant, embedding_source).build()
    aux_config = TemporalAuxConfig(variant=variant, max_grid=2, tinc_weight=0.1)
    tm = build_train_module(model, aux_config)
    batch = make_batch(patch_size=4)
    loss, *_, extra = tm.model_forward(batch, 4, tm.token_exit_cfg)
    assert torch.isfinite(loss)
    assert TEMPORAL_AUX_OUTPUT_KEY not in extra
    metrics = extra["temporal_aux"]
    assert metrics["num_samples"].item() == 2
    assert metrics["num_chunks"].item() == 5
    assert metrics["loss"].item() > 0
    loss.backward()
    head = model.temporal_aux_head
    assert head is not None
    for name, p in head.named_parameters():
        assert p.grad is not None, name
    assert head.projector[0].weight.grad.abs().sum() > 0
    if variant == "ordering":
        assert head.order_g[0].weight.grad.abs().sum() > 0
        assert 0.0 <= metrics["order_acc"].item() <= 1.0
    else:
        assert head.gate_stats_initialized.item() == 1
        assert 0.0 <= metrics["gate_mean"].item() <= 1.0
        assert metrics["tinc"].item() >= 0.0
    # The aux pass trains the encoder itself (the aux loss alone).
    model.zero_grad()
    tm.temporal_aux.config.weight = 1.0
    plan = tm._plan_temporal_aux(batch, 4)
    aux_in = tm.temporal_aux.model_input(batch, plan, 4, tm._encoder_modalities())
    out = model.temporal_aux_forward(aux_in, 4)
    aux_loss, _ = tm.temporal_aux.compute(out, plan, head)
    aux_loss.backward()
    block_grad = model.encoder.blocks[0].attn.q.weight.grad
    assert block_grad is not None and block_grad.abs().sum() > 0


@pytest.mark.parametrize("variant", ["ordering", "gated_contrast"])
def test_train_module_without_multiyear_samples(variant: str) -> None:
    """No eligible sample: zero aux loss, same metric keys, every head param has a grad."""
    torch.manual_seed(0)
    model = tiny_model_config(variant).build()
    tm = build_train_module(model, TemporalAuxConfig(variant=variant, max_grid=2))
    with_my = tm.model_forward(make_batch(), 4, tm.token_exit_cfg)[-1]["temporal_aux"]
    loss, *_, extra = tm.model_forward(
        make_batch(include_multiyear=False), 4, tm.token_exit_cfg
    )
    assert set(extra["temporal_aux"]) == set(with_my)
    assert extra["temporal_aux"]["loss"].item() == 0
    assert extra["temporal_aux"]["num_samples"].item() == 0
    model.zero_grad()
    loss.backward()
    for name, p in model.temporal_aux_head.named_parameters():
        assert p.grad is not None and torch.all(p.grad == 0), name


@pytest.mark.parametrize("variant", ["ordering", "gated_contrast"])
def test_train_batch_logs_aux_metrics(variant: str) -> None:
    """``train_batch`` (CPU plan path) runs and logs the namespaced aux metrics."""
    torch.manual_seed(0)
    model = tiny_model_config(variant).build()
    tm = build_train_module(model, TemporalAuxConfig(variant=variant, max_grid=2))
    trainer = _MockTrainer()
    tm.on_attach = MagicMock(return_value=None)  # type: ignore[method-assign]
    tm._attach_trainer(trainer)
    tm.train_batch((4, make_batch(), None))
    assert "temporal_aux/loss" in trainer.metrics
    assert any(k.startswith("train/") and "temporal_aux" in k for k in trainer.metrics)
    assert torch.isfinite(torch.as_tensor(trainer.metrics["temporal_aux/loss"]))


def test_latent_mim_forward_contract_unchanged() -> None:
    """Without ``temporal_aux`` the 7-tuple and extra_metrics are unchanged."""
    torch.manual_seed(0)
    model = tiny_model_config("ordering").build()
    out = model(make_batch(), patch_size=4)
    assert len(out) == 7
    assert TEMPORAL_AUX_OUTPUT_KEY not in out[4]
    # With an empty aux input, only the anchor comes back.
    out = model(make_batch(), patch_size=4, temporal_aux=TemporalAuxInput(batch=None))
    aux = out[4][TEMPORAL_AUX_OUTPUT_KEY]
    assert aux["proj"] is None and aux["param_anchor"].item() == 0


def test_config_round_trip_and_validation() -> None:
    """The head config serializes with the model config; bad combinations are rejected."""
    config = tiny_model_config("ordering")
    rebuilt = Config.from_dict(config.as_config_dict())
    assert isinstance(rebuilt.temporal_aux_head_config, TemporalAuxHeadConfig)
    model = rebuilt.build()
    assert isinstance(model.temporal_aux_head, TemporalAuxHead)
    assert model.temporal_aux_head.has_ordering_head
    # The ordering variant needs the ordering head.
    no_order = tiny_model_config("gated_contrast").build()
    with pytest.raises(ValueError, match="ordering_dims"):
        build_train_module(no_order, TemporalAuxConfig(variant="ordering"))
    with pytest.raises(ValueError, match="variant"):
        TemporalAuxConfig(variant="nope").validate()
    # location tokens need the AttentionPooler.
    bad = tiny_model_config("ordering")
    bad.encoder_config.attention_pooler_config = None
    bad.decoder_config.use_attention_pooler = False
    with pytest.raises(ValueError, match="AttentionPooler"):
        bad.validate()
