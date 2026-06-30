"""Unit tests for ERA5 objectives A (supervised) and B (reconstruction).

Covers:
  - Supervised: classification e2e, task routing, invalid labels
  - Reconstruction: SWT transform, masking invariants, per-group loss gating,
    loss computation correctness, end-to-end backward
"""

from __future__ import annotations

import math
from typing import Any

import pytest
import torch
import torch.nn.functional as F
from torch import Tensor, nn

import olmoearth_pretrain.train.train_module.era5_multiobjective as era5_multiobjective
from olmoearth_pretrain.data.constants import ERA5_INPUT_SEQUENCE_LENGTH, Modality
from olmoearth_pretrain.data.multi_task_era5_dataset import Era5SupervisedBatch
from olmoearth_pretrain.nn.era5_decoder import Era5TimeQueryDecoderConfig
from olmoearth_pretrain.nn.era5_encoder import Era5DailyEncoderConfig
from olmoearth_pretrain.nn.transforms.era5_corruption import (
    DEFAULT_VARIABLE_GROUPS,
    MaskPolicy,
    NaiveMaskPolicy,
    corrupt_era5,
)
from olmoearth_pretrain.nn.transforms.era5_swt import StationaryWaveletTransform1d
from olmoearth_pretrain.train.train_module.era5_multiobjective import (
    Era5MultiObjectiveModelConfig,
    ReconstructionObjectiveConfig,
    SupervisedObjectiveConfig,
    SupervisedTaskConfig,
    _parse_recon_mode,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

T = ERA5_INPUT_SEQUENCE_LENGTH  # 448
V = Modality.ERA5L_DAY_10.num_bands  # 14
D = 64
B = 4
SWT_BUFFER = 83

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _make_batch(
    task_name: str = "smoke_task",
    labels: Tensor | None = None,
    device: torch.device = torch.device("cpu"),
) -> Era5SupervisedBatch:
    """Synthetic ERA5 batch with configurable labels."""
    era5 = torch.randn(B, T, V, device=device)
    timestamps = torch.zeros(B, T, 3, dtype=torch.long, device=device)
    timestamps[..., 0] = torch.arange(1, T + 1).unsqueeze(0)
    timestamps[..., 1] = (timestamps[..., 0] - 1) * 12 // 365
    timestamps[..., 2] = 2020
    if labels is None:
        labels = torch.zeros(B, dtype=torch.long, device=device)
    return Era5SupervisedBatch(
        era5=era5,
        timestamps=timestamps,
        labels=labels,
        task_name=task_name,
    )


def _small_encoder_cfg(**overrides: Any) -> Era5DailyEncoderConfig:
    defaults: dict[str, Any] = dict(
        embedding_size=D,
        depth=1,
        num_heads=4,
        max_sequence_length=T,
        modality_name=Modality.ERA5L_DAY_10.name.lower(),
        use_mask_embed=True,
        use_conv_stem=True,
    )
    defaults.update(overrides)
    return Era5DailyEncoderConfig(**defaults)


def _small_decoder_cfg(**overrides: Any) -> Era5TimeQueryDecoderConfig:
    defaults: dict[str, Any] = dict(
        embedding_size=D,
        depth=1,
        num_heads=4,
        max_sequence_length=T,
        num_output_channels=V,
    )
    defaults.update(overrides)
    return Era5TimeQueryDecoderConfig(**defaults)


class _FixedEncoder:
    """Tiny encoder double for deterministic reconstruction objective tests."""

    def __call__(
        self,
        era5: Tensor,
        timestamps: Tensor,
        corruption_mask: Tensor | None = None,
    ) -> dict[str, Tensor]:
        del timestamps, corruption_mask
        b = era5.shape[0]
        return {
            "tokens": torch.zeros(b, 1, D, device=era5.device, dtype=era5.dtype),
        }


class _FixedDecoder(nn.Module):
    """Decoder double that returns a precomputed full-sequence prediction."""

    def __init__(self, prediction: Tensor) -> None:
        super().__init__()
        self.register_buffer("prediction", prediction)

    def forward(
        self,
        tokens: Tensor,
        timestamps: Tensor,
    ) -> Tensor:
        del timestamps
        return self.prediction.to(device=tokens.device, dtype=tokens.dtype)


# ===================================================================
# Supervised Objective (A) — 3 tests
# ===================================================================


class TestSupervisedClassificationE2E:
    """End-to-end classification: forward + backward + metrics."""

    def test_loss_and_gradients(self):
        model_cfg = Era5MultiObjectiveModelConfig(
            encoder_config=_small_encoder_cfg(),
            supervised_objective=SupervisedObjectiveConfig(
                tasks=[
                    SupervisedTaskConfig(
                        name="smoke_task",
                        task_type="classification",
                        num_classes=3,
                    )
                ],
            ),
        )
        model = model_cfg.build()
        obj = model.objective_list[0]
        batch = _make_batch(
            labels=torch.randint(0, 3, (B,)),
        )

        loss, metrics = obj.compute(model.encoder, batch)

        assert loss.ndim == 0
        assert torch.isfinite(loss)
        assert loss.item() > 0

        loss.backward()
        encoder_grads = sum(
            p.grad.abs().sum().item()
            for p in model.encoder.parameters()
            if p.grad is not None
        )
        assert encoder_grads > 0, "Gradients should flow to encoder"

        head_grads = sum(
            p.grad.abs().sum().item()
            for p in obj.registry.parameters()
            if p.grad is not None
        )
        assert head_grads > 0, "Gradients should flow to head"

        assert "supervised/smoke_task/loss" in metrics
        assert "supervised/smoke_task/accuracy" in metrics
        acc = metrics["supervised/smoke_task/accuracy"].item()
        assert 0.0 <= acc <= 1.0


class TestSupervisedTaskRouting:
    """Two heads registered; only the addressed head receives gradients."""

    def test_only_active_head_gets_gradients(self):
        model_cfg = Era5MultiObjectiveModelConfig(
            encoder_config=_small_encoder_cfg(),
            supervised_objective=SupervisedObjectiveConfig(
                tasks=[
                    SupervisedTaskConfig(
                        name="task_a",
                        task_type="classification",
                        num_classes=2,
                    ),
                    SupervisedTaskConfig(
                        name="task_b",
                        task_type="regression",
                    ),
                ],
            ),
        )
        model = model_cfg.build()
        obj = model.objective_list[0]

        batch = _make_batch(
            task_name="task_a",
            labels=torch.randint(0, 2, (B,)),
        )

        # Trigger lazy init on both heads so parameters exist
        dummy_pooled = torch.randn(1, D)
        obj.registry.heads["task_a"](dummy_pooled)
        obj.registry.heads["task_b"](dummy_pooled)

        model.zero_grad()
        loss, metrics = obj.compute(model.encoder, batch)
        loss.backward()

        assert torch.isfinite(loss)

        head_a = obj.registry.heads["task_a"]
        head_b = obj.registry.heads["task_b"]

        a_grads = sum(
            p.grad.abs().sum().item() for p in head_a.parameters() if p.grad is not None
        )
        b_grads = sum(
            p.grad.abs().sum().item() for p in head_b.parameters() if p.grad is not None
        )
        assert a_grads > 0, "Active head (task_a) should have gradients"
        assert b_grads == 0, "Inactive head (task_b) should have no gradients"

        assert any("task_a" in k for k in metrics)
        assert not any("task_b" in k for k in metrics)


class TestSupervisedInvalidLabels:
    """All-invalid supervised labels fail loudly instead of producing no-op loss."""

    def test_classification_all_ignored(self):
        model_cfg = Era5MultiObjectiveModelConfig(
            encoder_config=_small_encoder_cfg(),
            supervised_objective=SupervisedObjectiveConfig(
                tasks=[
                    SupervisedTaskConfig(
                        name="cls_task",
                        task_type="classification",
                        num_classes=3,
                    )
                ],
            ),
        )
        model = model_cfg.build()
        obj = model.objective_list[0]

        labels = torch.full((B,), -100, dtype=torch.long)
        batch = _make_batch(task_name="cls_task", labels=labels)

        with pytest.raises(ValueError, match="only ignore-index labels"):
            obj.compute(model.encoder, batch)

    def test_regression_all_nan(self):
        model_cfg = Era5MultiObjectiveModelConfig(
            encoder_config=_small_encoder_cfg(),
            supervised_objective=SupervisedObjectiveConfig(
                tasks=[
                    SupervisedTaskConfig(
                        name="reg_task",
                        task_type="regression",
                    )
                ],
            ),
        )
        model = model_cfg.build()
        obj = model.objective_list[0]

        labels = torch.full((B,), float("nan"))
        batch = _make_batch(task_name="reg_task", labels=labels)

        with pytest.raises(ValueError, match="no finite labels"):
            obj.compute(model.encoder, batch)

    def test_multilabel_all_nan(self):
        model_cfg = Era5MultiObjectiveModelConfig(
            encoder_config=_small_encoder_cfg(),
            supervised_objective=SupervisedObjectiveConfig(
                tasks=[
                    SupervisedTaskConfig(
                        name="multilabel_task",
                        task_type="classification",
                        num_classes=3,
                        is_multilabel=True,
                    )
                ],
            ),
        )
        model = model_cfg.build()
        obj = model.objective_list[0]

        labels = torch.full((B, 3), float("nan"))
        batch = _make_batch(task_name="multilabel_task", labels=labels)

        with pytest.raises(ValueError, match="no rows with any finite"):
            obj.compute(model.encoder, batch)


# ===================================================================
# Reconstruction Objective (B) — 5 tests
# ===================================================================

# ---------------------------------------------------------------------------
# Test-only inverse Haar SWT helper
# ---------------------------------------------------------------------------


def _inverse_haar_swt(
    bands: list[tuple[Tensor, Tensor]],
) -> Tensor:
    """Reconstruct a signal from undecimated Haar SWT bands (test-only).

    For the undecimated Haar SWT at level j with dilation d = 2^j::

        approx[t] = (in[t-d] + in[t]) / sqrt(2)
        detail[t] = (in[t-d] - in[t]) / sqrt(2)

    Subtracting: ``in[t] = (approx[t] - detail[t]) / sqrt(2)``

    The cascaded SWT feeds ``approx_{j-1}`` into level ``j``, so the
    input to level 0 is the original signal ``x``.  Level 0 alone is
    sufficient for reconstruction; higher levels verify the cascade.
    """
    s2 = math.sqrt(2.0)
    approx_0, detail_0 = bands[0]
    return (approx_0 - detail_0) / s2


class TestSwtTransform:
    """SWT shapes, inverse reconstruction, coefficient correctness."""

    def test_shapes_and_cropping(self):
        """Full and cropped SWT output shapes; cropped == full tail."""
        x = torch.randn(2, 4, T)
        swt = StationaryWaveletTransform1d(num_channels=4, max_levels=6, wavelet="haar")
        levels = list(range(6))

        bands_full = swt(x, levels=levels, target_start=0)
        assert len(bands_full) == 6
        for i, (a, d) in enumerate(bands_full):
            assert a.shape == (2, 4, T), f"Level {i} full approx shape mismatch"
            assert d.shape == (2, 4, T), f"Level {i} full detail shape mismatch"

        t_win = T - SWT_BUFFER
        bands_crop = swt(x, levels=levels, target_start=SWT_BUFFER)
        assert len(bands_crop) == 6
        for i, (a, d) in enumerate(bands_crop):
            assert a.shape == (2, 4, t_win), f"Level {i} cropped shape mismatch"
            assert d.shape == (2, 4, t_win)
            assert torch.allclose(a, bands_full[i][0][:, :, SWT_BUFFER:])
            assert torch.allclose(d, bands_full[i][1][:, :, SWT_BUFFER:])

    def test_inverse_reconstruction(self):
        """ISWT(SWT(x)) recovers the original signal in the target window."""
        torch.manual_seed(42)
        x = torch.randn(2, 4, T)
        swt = StationaryWaveletTransform1d(num_channels=4, max_levels=6, wavelet="haar")
        levels = list(range(6))

        bands = swt(x, levels=levels, target_start=0)
        x_recon = _inverse_haar_swt(bands)

        # For undecimated Haar, the (approx - detail)/sqrt(2) formula is
        # exact at every position (even boundary) — verify everywhere.
        err = (x_recon - x).abs().max().item()
        assert err < 1e-5, f"Reconstruction error {err} exceeds tolerance"

    def test_cascade_consistency(self):
        """Verify that each level's inverse recovers the input to that level."""
        torch.manual_seed(42)
        x = torch.randn(2, 4, T)
        swt = StationaryWaveletTransform1d(num_channels=4, max_levels=6, wavelet="haar")
        bands = swt(x, levels=list(range(6)), target_start=0)

        s2 = math.sqrt(2.0)
        for i in range(len(bands) - 1, 0, -1):
            recovered_prev_approx = (bands[i][0] - bands[i][1]) / s2
            err = (recovered_prev_approx - bands[i - 1][0]).abs().max().item()
            assert err < 1e-5, (
                f"Level {i} inverse doesn't recover level {i - 1} approx: {err}"
            )

    def test_constant_signal_zero_detail(self):
        """Haar detail of a constant signal is exactly zero."""
        c = 3.14
        x = torch.full((1, 2, T), c)
        swt = StationaryWaveletTransform1d(num_channels=2, max_levels=6, wavelet="haar")
        bands = swt(x, levels=list(range(6)), target_start=SWT_BUFFER)

        for i, (_, detail) in enumerate(bands):
            assert detail.abs().max().item() < 1e-6, (
                f"Level {i} detail should be ~0 for constant signal"
            )

    def test_step_function_localized_detail(self):
        """Haar detail of a step function is non-zero only near the edge."""
        x = torch.zeros(1, 1, T)
        step_t = 200
        x[:, :, step_t:] = 1.0
        swt = StationaryWaveletTransform1d(num_channels=1, max_levels=6, wavelet="haar")
        bands = swt(x, levels=[0], target_start=0)

        _, detail = bands[0]
        far_from_step = torch.cat(
            [detail[:, :, : step_t - 5], detail[:, :, step_t + 5 :]], dim=2
        )
        assert far_from_step.abs().max().item() < 1e-6, (
            "Detail should be ~0 far from the step"
        )
        near_step = detail[:, :, step_t - 1 : step_t + 2]
        assert near_step.abs().max().item() > 0.1, (
            "Detail should be non-zero near the step"
        )

    def test_multiscale_loss_includes_deepest_approx(self):
        """Deepest-level approximation is included in the returned total."""
        from olmoearth_pretrain.nn.transforms.era5_swt import multiscale_swt_loss

        swt = StationaryWaveletTransform1d(num_channels=4, max_levels=6, wavelet="haar")
        x = torch.zeros(2, T, 4)
        x_hat = torch.ones_like(x)

        levels = [0, 1, 2]
        total, metrics = multiscale_swt_loss(x_hat, x, swt, levels)
        assert torch.isfinite(total) and total.item() > 0

        deepest = max(levels)
        assert f"swt_level_{deepest}_approx_loss" in metrics
        approx = metrics[f"swt_level_{deepest}_approx_loss"]
        assert approx.item() > 0

        # A constant offset leaves only tiny numerical detail terms after the
        # target buffer.  The total must include those detail terms plus the
        # deepest approximation term.
        detail_total = sum(metrics[f"swt_level_{lvl}_loss"] for lvl in levels)
        assert approx.item() > detail_total.item() * 1000
        assert torch.allclose(total, detail_total + approx, rtol=1e-6, atol=1e-6)


class TestMaskingInvariants:
    """Masking respects buffer, group exclusions, and padding."""

    def test_buffer_never_masked(self):
        """Buffer region [0, 83) is never masked across many seeds."""
        era5 = torch.randn(B, T, V)

        for seed in range(50):
            torch.manual_seed(seed)
            mask = corrupt_era5(era5, MaskPolicy(), DEFAULT_VARIABLE_GROUPS, SWT_BUFFER)
            assert not mask[:, :SWT_BUFFER, :].any(), f"Buffer masked at seed {seed}"
            assert mask[:, SWT_BUFFER:, :].any(), f"Nothing masked at seed {seed}"

    def test_naive_policy_buffer_clean(self):
        """NaiveMaskPolicy also respects the buffer."""
        era5 = torch.randn(B, T, V)

        for seed in range(20):
            torch.manual_seed(seed)
            mask = corrupt_era5(
                era5, NaiveMaskPolicy(), DEFAULT_VARIABLE_GROUPS, SWT_BUFFER
            )
            assert not mask[:, :SWT_BUFFER, :].any()
            assert mask[:, SWT_BUFFER:, :].any(), f"Nothing masked at seed {seed}"

    def test_stage1_excludes_wind_hydro(self):
        """With Stage 2 disabled, wind and hydro_flux bands are never masked."""
        era5 = torch.randn(B, T, V)
        policy = MaskPolicy(cross_variable_prob=0.0)

        wind_bands = DEFAULT_VARIABLE_GROUPS["wind"]
        hydro_bands = DEFAULT_VARIABLE_GROUPS["hydro_flux"]
        excluded = wind_bands + hydro_bands
        eligible = [idx for idx in range(V) if idx not in excluded]

        for seed in range(30):
            torch.manual_seed(seed)
            mask = corrupt_era5(era5, policy, DEFAULT_VARIABLE_GROUPS, SWT_BUFFER)
            assert not mask[:, :, excluded].any(), (
                f"Excluded bands masked at seed {seed}"
            )
            assert mask[:, SWT_BUFFER:, eligible].any(), (
                f"No eligible Stage 1 bands masked at seed {seed}"
            )

    def test_masked_fraction_reasonable(self):
        """Masked fraction stays in a sane range."""
        era5 = torch.randn(B, T, V)

        fracs = []
        for seed in range(50):
            torch.manual_seed(seed)
            mask = corrupt_era5(era5, MaskPolicy(), DEFAULT_VARIABLE_GROUPS, SWT_BUFFER)
            frac = mask[:, SWT_BUFFER:, :].float().mean().item()
            fracs.append(frac)

        avg_frac = sum(fracs) / len(fracs)
        assert 0.01 < avg_frac < 0.5, f"Average masked fraction {avg_frac} out of range"


class TestPerGroupLossGating:
    """group_recon_mode gates which groups contribute to raw vs SWT loss."""

    def test_pressure_excluded_from_raw(self):
        """Pressure (lowpass_plus_slow_swt) contributes no raw loss."""
        inc_raw, lvls, inc_lowpass = _parse_recon_mode(
            "lowpass_plus_slow_swt", [0, 1, 2, 3, 4, 5]
        )
        assert inc_raw is False
        assert inc_lowpass is True
        assert 0 not in lvls
        assert all(lv >= 2 for lv in lvls)

    def test_raw_loss_respects_group_recon_mode(self, monkeypatch):
        """Pressure-only raw loss is zero unless the mode explicitly enables raw."""
        target = torch.zeros(B, T, V)
        pressure = torch.linspace(-1.0, 1.0, T)
        target[:, :, 4] = pressure.unsqueeze(0)
        prediction = target.clone()
        prediction[:, SWT_BUFFER:, 4] += 0.5

        mask = torch.zeros(B, T, V, dtype=torch.bool)
        mask[:, SWT_BUFFER:, 4] = True
        monkeypatch.setattr(
            era5_multiobjective,
            "corrupt_era5",
            lambda era5, policy, variable_groups, target_start: mask,
        )

        batch = _make_batch()._replace(era5=target)

        def _build_obj(mode: str):
            model = Era5MultiObjectiveModelConfig(
                encoder_config=_small_encoder_cfg(),
                reconstruction_objective=ReconstructionObjectiveConfig(
                    decoder=_small_decoder_cfg(),
                    variable_groups={"pressure": [4]},
                    group_recon_mode={"pressure": mode},
                    swt_lambda=0.0,
                ),
            ).build()
            obj = model.objective_list[0]
            assert isinstance(obj, era5_multiobjective.ReconstructionObjective)
            obj._module.decoder = _FixedDecoder(prediction)
            return obj

        slow_wavelet = _build_obj("lowpass_plus_slow_swt")
        raw_enabled = _build_obj("raw_plus_slow_swt")

        loss_slow, metrics_slow = slow_wavelet.compute(_FixedEncoder(), batch)
        loss_raw, metrics_raw = raw_enabled.compute(_FixedEncoder(), batch)

        assert torch.isfinite(loss_slow) and torch.isfinite(loss_raw)
        assert metrics_slow["reconstruction/raw_loss"].item() == 0.0
        assert loss_slow.item() == 0.0
        assert metrics_raw["reconstruction/raw_loss"].item() > 0.0
        assert loss_raw.item() == metrics_raw["reconstruction/raw_loss"].item()


class TestLossComputationCorrectness:
    """Verify _group_huber and band normalization on hand-crafted tensors."""

    def _make_objective(self) -> object:
        """Build a minimal ReconstructionObjective for accessing _group_huber."""
        model_cfg = Era5MultiObjectiveModelConfig(
            encoder_config=_small_encoder_cfg(),
            reconstruction_objective=ReconstructionObjectiveConfig(
                decoder=_small_decoder_cfg(),
            ),
        )
        model = model_cfg.build()
        return model.objective_list[0]

    def test_perfect_prediction_zero_loss(self):
        obj = self._make_objective()
        pred = torch.tensor([1.0, 2.0, 3.0])
        targ = torch.tensor([1.0, 2.0, 3.0])
        loss = obj._group_huber(pred, targ, None)
        assert loss.item() == 0.0

    def test_known_huber_value(self):
        obj = self._make_objective()
        pred = torch.tensor([0.0])
        targ = torch.tensor([1.0])
        loss = obj._group_huber(pred, targ, None)
        expected = F.huber_loss(pred, targ, reduction="mean", delta=1.0).item()
        assert abs(loss.item() - expected) < 1e-6

    def test_empty_mask_zero_loss(self):
        obj = self._make_objective()
        pred = torch.tensor([0.0, 5.0])
        targ = torch.tensor([10.0, 20.0])
        mask = torch.zeros(2, dtype=torch.bool)
        loss = obj._group_huber(pred, targ, mask)
        assert loss.item() == 0.0

    def test_raw_band_normalization_matches_production_loss(self, monkeypatch):
        """Raw reconstruction loss uses per-channel std normalization."""
        torch.manual_seed(0)
        # Channel 0: std ~1, Channel 1: std ~100
        target = torch.zeros(B, T, V)
        target[:, :, 0] = torch.randn(B, T)
        target[:, :, 1] = torch.randn(B, T) * 100.0
        # Error proportional to signal scale (~10%)
        prediction = target.clone()
        prediction[:, :, [0, 1]] = target[:, :, [0, 1]] * 1.1

        # Without normalization: channel 1 dominates
        raw_loss_ch0 = F.huber_loss(
            prediction[:, SWT_BUFFER:, 0],
            target[:, SWT_BUFFER:, 0],
            reduction="mean",
            delta=1.0,
        )
        raw_loss_ch1 = F.huber_loss(
            prediction[:, SWT_BUFFER:, 1],
            target[:, SWT_BUFFER:, 1],
            reduction="mean",
            delta=1.0,
        )
        assert raw_loss_ch1 > raw_loss_ch0 * 5, "Channel 1 should dominate raw"

        mask = torch.zeros(B, T, V, dtype=torch.bool)
        mask[:, SWT_BUFFER:, [0, 1]] = True
        monkeypatch.setattr(
            era5_multiobjective,
            "corrupt_era5",
            lambda era5, policy, variable_groups, target_start: mask,
        )

        model = Era5MultiObjectiveModelConfig(
            encoder_config=_small_encoder_cfg(),
            reconstruction_objective=ReconstructionObjectiveConfig(
                decoder=_small_decoder_cfg(),
                variable_groups={"scaled": [0, 1]},
                group_recon_mode={"scaled": "raw_plus_wavelet"},
                swt_lambda=0.0,
            ),
        ).build()
        obj = model.objective_list[0]
        assert isinstance(obj, era5_multiobjective.ReconstructionObjective)
        obj._module.decoder = _FixedDecoder(prediction)
        batch = _make_batch()._replace(era5=target)

        loss, metrics = obj.compute(_FixedEncoder(), batch)

        g_pred = prediction[:, SWT_BUFFER:, [0, 1]]
        g_targ = target[:, SWT_BUFFER:, [0, 1]]
        with torch.no_grad():
            std = g_targ.std(dim=(0, 1)).clamp(min=1e-6)
        expected = F.huber_loss(
            g_pred / std[None, None, :],
            g_targ / std[None, None, :],
            reduction="mean",
            delta=1.0,
        )
        assert torch.allclose(loss, expected, rtol=1e-6, atol=1e-6)
        assert torch.allclose(
            metrics["reconstruction/raw_loss"], expected.detach(), rtol=1e-6, atol=1e-6
        )


class TestReconstructionE2EBackward:
    """Full forward+backward integration with gradient and metric checks."""

    def test_default_policy(self):
        torch.manual_seed(0)
        model_cfg = Era5MultiObjectiveModelConfig(
            encoder_config=_small_encoder_cfg(),
            reconstruction_objective=ReconstructionObjectiveConfig(
                decoder=_small_decoder_cfg(),
                swt_levels=[0, 1, 2],
                swt_lambda=0.1,
            ),
        )
        model = model_cfg.build()
        obj = model.objective_list[0]
        assert isinstance(obj, era5_multiobjective.ReconstructionObjective)
        batch = _make_batch()

        loss, metrics = obj.compute(model.encoder, batch)

        assert loss.ndim == 0
        assert torch.isfinite(loss)
        assert loss.item() > 0

        loss.backward()

        encoder_grads = sum(
            p.grad.abs().sum().item()
            for p in model.encoder.parameters()
            if p.grad is not None
        )
        assert encoder_grads > 0, "Gradients should flow to encoder"

        decoder_grads = sum(
            p.grad.abs().sum().item()
            for p in obj._module.decoder.parameters()
            if p.grad is not None
        )
        assert decoder_grads > 0, "Gradients should flow to decoder"

        assert "reconstruction/raw_loss" in metrics
        assert "reconstruction/swt_loss" in metrics
        assert "reconstruction/masked_fraction" in metrics
        for lvl in [0, 1, 2]:
            assert f"reconstruction/swt_level_{lvl}_loss" in metrics
        assert "reconstruction/swt_deepest_approx_loss" in metrics
        assert metrics["reconstruction/swt_deepest_approx_loss"].item() > 0

        frac = metrics["reconstruction/masked_fraction"].item()
        assert 0 < frac < 0.5

    def test_naive_policy(self):
        torch.manual_seed(0)
        model_cfg = Era5MultiObjectiveModelConfig(
            encoder_config=_small_encoder_cfg(),
            reconstruction_objective=ReconstructionObjectiveConfig(
                decoder=_small_decoder_cfg(),
                mask_policy=NaiveMaskPolicy(),
                swt_levels=[0, 1],
                swt_lambda=0.1,
            ),
        )
        model = model_cfg.build()
        obj = model.objective_list[0]
        batch = _make_batch()

        loss, metrics = obj.compute(model.encoder, batch)

        assert loss.ndim == 0
        assert torch.isfinite(loss)
        assert loss.item() > 0

        loss.backward()
        encoder_grads = sum(
            p.grad.abs().sum().item()
            for p in model.encoder.parameters()
            if p.grad is not None
        )
        assert encoder_grads > 0
