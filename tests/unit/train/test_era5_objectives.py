"""Unit tests for ERA5 objectives A (supervised) and B (reconstruction).

Covers:
  - Supervised: classification e2e, task routing, invalid labels
  - Reconstruction: SWT transform, masking invariants, per-group loss gating,
    loss computation correctness, end-to-end backward
"""

from __future__ import annotations

import importlib.util
import json
import math
import pickle
import sys
from contextlib import nullcontext
from dataclasses import dataclass, field, replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock

import pytest
import torch
import torch.nn.functional as F
from olmo_core.config import Config
from torch import Tensor, nn

import olmoearth_pretrain.nn.era5_encoder as era5_encoder_mod
import olmoearth_pretrain.train.callbacks.era5_evaluator_callback as era5_evaluator_callback
import olmoearth_pretrain.train.train_module.era5_multiobjective as era5_multiobjective
from olmoearth_pretrain.data.constants import ERA5_INPUT_SEQUENCE_LENGTH, Modality
from olmoearth_pretrain.data.multi_task_era5_dataset import (
    LABEL_EXTRACTORS,
    Era5SslBatch,
    Era5SupervisedBatch,
    Era5TaskSpec,
    make_regression_extractor,
)
from olmoearth_pretrain.nn.attention import Mlp
from olmoearth_pretrain.nn.era5_decoder import Era5TimeQueryDecoderConfig
from olmoearth_pretrain.nn.era5_encoder import Era5DailyEncoderConfig
from olmoearth_pretrain.nn.transforms.era5_corruption import (
    Era5CorruptionMasks,
    SwtHaloSpanMaskPolicy,
    SwtNaiveMaskPolicy,
    corrupt_era5_swt,
)
from olmoearth_pretrain.nn.transforms.era5_swt import (
    StationaryWaveletTransform1d,
    swt_band_supports,
    swt_bands_to_channels,
)
from olmoearth_pretrain.train.train_module.era5_multiobjective import (
    Era5MultiObjectiveModelConfig,
    ReconstructionObjectiveConfig,
    SupervisedObjectiveConfig,
    SupervisedTaskConfig,
    _instance_infonce,
    _parse_recon_mode,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Real stats file, resolved relative to the repo root (the dir with ``scripts/``).
_REPO_ROOT = Path(era5_encoder_mod.__file__).resolve().parents[2]
SWT_STATS_REL = "scripts/era5_supervised/v0/norm_configs/swt_input_stats.json"
SWT_STATS_PATH = _REPO_ROOT / SWT_STATS_REL

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
        valid_mask=torch.ones(B, T, V, dtype=torch.bool, device=device),
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
    """Tiny SWT-input encoder double for deterministic reconstruction tests.

    The reconstruction objective requires an ``is_swt_input`` encoder; this
    double advertises the SWT attributes the objective reads while ignoring the
    actual input (it returns fixed tokens).
    """

    is_swt_input = True
    swt_num_bands = 7
    swt_input_levels = [0, 1, 2, 3, 4, 5]
    swt_input_include_approx = True

    def __call__(
        self,
        era5: Tensor,
        timestamps: Tensor,
        corruption_mask: Tensor | None = None,
        valid_mask: Tensor | None = None,
    ) -> dict[str, Tensor]:
        del timestamps, corruption_mask, valid_mask
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
        swt = StationaryWaveletTransform1d(num_channels=4, max_levels=6)
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
        swt = StationaryWaveletTransform1d(num_channels=4, max_levels=6)
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
        swt = StationaryWaveletTransform1d(num_channels=4, max_levels=6)
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
        swt = StationaryWaveletTransform1d(num_channels=2, max_levels=6)
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
        swt = StationaryWaveletTransform1d(num_channels=1, max_levels=6)
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

        swt = StationaryWaveletTransform1d(num_channels=4, max_levels=6)
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
    """SWT band-space masking respects the buffer and halo arithmetic."""

    N_BANDS = 7  # levels [0..5] detail + deepest approx
    SUPPORTS = swt_band_supports([0, 1, 2, 3, 4, 5], include_approx=True)

    def test_naive_buffer_never_masked(self):
        """Naive budget masking never touches the buffer, both masks."""
        for seed in range(20):
            torch.manual_seed(seed)
            masks = corrupt_era5_swt(
                B,
                T,
                V,
                self.N_BANDS,
                self.SUPPORTS,
                SWT_BUFFER,
                torch.device("cpu"),
                SwtNaiveMaskPolicy(budget=0.5),
            )
            assert not masks.band_mask[:, :SWT_BUFFER, :].any()
            assert not masks.raw_loss_mask[:, :SWT_BUFFER, :].any()
            assert masks.band_mask[:, SWT_BUFFER:, :].any()

    def test_naive_masked_fraction_tracks_budget(self):
        """Naive band-mask fraction in the target window matches the budget."""
        torch.manual_seed(0)
        budget = 0.7
        masks = corrupt_era5_swt(
            B,
            T,
            V,
            self.N_BANDS,
            self.SUPPORTS,
            SWT_BUFFER,
            torch.device("cpu"),
            SwtNaiveMaskPolicy(budget=budget),
        )
        frac = masks.band_mask[:, SWT_BUFFER:, :].float().mean().item()
        assert abs(frac - budget) < 0.02, f"Band-mask fraction {frac} != {budget}"

    def test_naive_raw_loss_mask_reduce(self):
        """'all' reduce supervises a strict subset of 'any' reduce."""
        torch.manual_seed(0)
        any_masks = corrupt_era5_swt(
            B,
            T,
            V,
            self.N_BANDS,
            self.SUPPORTS,
            SWT_BUFFER,
            torch.device("cpu"),
            SwtNaiveMaskPolicy(budget=0.5, raw_loss_mask_reduce="any"),
        )
        torch.manual_seed(0)
        all_masks = corrupt_era5_swt(
            B,
            T,
            V,
            self.N_BANDS,
            self.SUPPORTS,
            SWT_BUFFER,
            torch.device("cpu"),
            SwtNaiveMaskPolicy(budget=0.5, raw_loss_mask_reduce="all"),
        )
        # Same band mask (same seed), but "all" supervises <= "any".
        assert torch.equal(any_masks.band_mask, all_masks.band_mask)
        assert (all_masks.raw_loss_mask & ~any_masks.raw_loss_mask).sum() == 0
        assert all_masks.raw_loss_mask.sum() < any_masks.raw_loss_mask.sum()

    def test_halo_span_buffer_never_masked(self):
        """Halo span masking never touches the buffer, both masks."""
        for seed in range(20):
            torch.manual_seed(seed)
            masks = corrupt_era5_swt(
                B,
                T,
                V,
                self.N_BANDS,
                self.SUPPORTS,
                SWT_BUFFER,
                torch.device("cpu"),
                SwtHaloSpanMaskPolicy(),
            )
            assert not masks.band_mask[:, :SWT_BUFFER, :].any()
            assert not masks.raw_loss_mask[:, :SWT_BUFFER, :].any()
            assert masks.band_mask[:, SWT_BUFFER:, :].any()

    def test_halo_span_arithmetic(self):
        """Each band's mask equals the span extended by exactly support-1 days.

        With a single span over a single variable, band ``s`` must be masked
        over ``[start, start+L + support_s - 1)`` while the raw loss mask covers
        only the span days ``[start, start+L)``.
        """
        policy = SwtHaloSpanMaskPolicy(
            num_spans=(1, 1), span_days=(20, 20), num_variables=(1, 1)
        )
        for seed in range(20):
            torch.manual_seed(seed)
            masks = corrupt_era5_swt(
                1,
                T,
                V,
                self.N_BANDS,
                self.SUPPORTS,
                SWT_BUFFER,
                torch.device("cpu"),
                policy,
            )
            band4d = masks.band_mask.view(1, T, V, self.N_BANDS)
            # Identify the single masked variable / span from the raw loss mask.
            raw = masks.raw_loss_mask[0]  # [T, V]
            var = int(raw.any(dim=0).nonzero()[0])
            days = raw[:, var].nonzero().flatten()
            start, end = int(days[0]), int(days[-1]) + 1
            assert end - start == 20
            for s, support in enumerate(self.SUPPORTS):
                col = band4d[0, :, var, s]
                masked = col.nonzero().flatten()
                assert int(masked[0]) == start
                expected_end = min(end + support - 1, T)
                assert int(masked[-1]) + 1 == expected_end
            # Untouched variables have no band mask at all.
            other_vars = [j for j in range(V) if j != var]
            assert not band4d[0, :, other_vars, :].any()

    def test_halo_span_loss_mask_is_span_only(self):
        """Raw loss mask covers strictly fewer positions than the band mask."""
        torch.manual_seed(0)
        masks = corrupt_era5_swt(
            B,
            T,
            V,
            self.N_BANDS,
            self.SUPPORTS,
            SWT_BUFFER,
            torch.device("cpu"),
            SwtHaloSpanMaskPolicy(),
        )
        band4d = masks.band_mask.view(B, T, V, self.N_BANDS)
        band_any = band4d.any(dim=-1)  # [B, T, V]
        # Every supervised raw position is band-masked; halos add strictly more.
        assert (masks.raw_loss_mask & ~band_any).sum() == 0
        assert masks.raw_loss_mask.sum() < band_any.sum()

    @pytest.mark.parametrize(
        "policy",
        [
            SwtHaloSpanMaskPolicy(),
            # halo75: the v1.3.4 visibility-matched rung.
            SwtHaloSpanMaskPolicy(
                num_spans=(4, 10), span_days=(30, 120), num_variables=(9, 14)
            ),
        ],
    )
    def test_halo_span_inside_matches_reference_loop(self, policy):
        """Vectorized "inside" sampler matches the original loop in distribution."""
        n = 512
        torch.manual_seed(0)
        masks = corrupt_era5_swt(
            n,
            T,
            V,
            self.N_BANDS,
            self.SUPPORTS,
            SWT_BUFFER,
            torch.device("cpu"),
            policy,
        )
        torch.manual_seed(0)
        ref_band, ref_raw = _reference_halo_span_loop(
            n, T, V, self.SUPPORTS, SWT_BUFFER, policy
        )
        band4d = masks.band_mask.view(n, T, V, self.N_BANDS)
        got_raw = masks.raw_loss_mask[:, SWT_BUFFER:].float().mean()
        ref_raw_frac = ref_raw[:, SWT_BUFFER:].float().mean()
        assert abs(got_raw - ref_raw_frac) < 0.02
        # Per-band fractions (halo ordering) match the loop too.
        torch.testing.assert_close(
            band4d[:, SWT_BUFFER:].float().mean(dim=(0, 1, 2)),
            ref_band[:, SWT_BUFFER:].float().mean(dim=(0, 1, 2)),
            atol=0.02,
            rtol=0,
        )

    @pytest.mark.parametrize("placement", ["inside", "pin"])
    def test_halo_span_has_no_host_sync(self, placement):
        """Sampling runs on the meta device, so no value is read back to host."""
        masks = corrupt_era5_swt(
            B,
            T,
            V,
            self.N_BANDS,
            self.SUPPORTS,
            SWT_BUFFER,
            torch.device("meta"),
            SwtHaloSpanMaskPolicy(placement=placement),
        )
        assert masks.band_mask.shape == (B, T, V * self.N_BANDS)
        assert masks.raw_loss_mask.shape == (B, T, V)

    @pytest.mark.parametrize("target_start", [0, SWT_BUFFER])
    def test_halo_span_pin_keeps_full_contiguous_length(self, target_start):
        """Pinned spans are one contiguous run of exactly L days in the window."""
        span = 60
        torch.manual_seed(0)
        masks = corrupt_era5_swt(
            2000,
            T,
            1,
            self.N_BANDS,
            self.SUPPORTS,
            target_start,
            torch.device("cpu"),
            SwtHaloSpanMaskPolicy(
                num_spans=(1, 1),
                span_days=(span, span),
                num_variables=(1, 1),
                placement="pin",
            ),
        )
        raw = masks.raw_loss_mask[:, :, 0]  # [N, T]
        assert (raw.sum(dim=1) == span).all()
        assert not raw[:, :target_start].any()
        rising = (raw[:, 1:] & ~raw[:, :-1]).sum(dim=1) + raw[:, 0].long()
        assert (rising == 1).all()
        # Both edges are reached: flush-left and flush-right spans occur.
        assert raw[:, target_start].any() and raw[:, -1].any()

    def test_halo_span_pin_coverage_profile(self):
        """Pin: edge days match interior coverage, with a <2x bump L days wide.

        One span of length L on ``[0, T)`` with starts uniform over
        ``[-L+1, T-1]`` (N = T + L - 1 draws): each edge-flush position gets L
        draws, so every day is covered with probability L/N except days just
        inside an edge, which peak at (2L - 1)/N one span length in. "inside"
        instead covers the edge day with probability 1/(T - L + 1).
        """
        span, n = 60, 40_000

        def coverage(placement: str) -> Tensor:
            torch.manual_seed(0)
            masks = corrupt_era5_swt(
                n,
                T,
                1,
                self.N_BANDS,
                self.SUPPORTS,
                0,
                torch.device("cpu"),
                SwtHaloSpanMaskPolicy(
                    num_spans=(1, 1),
                    span_days=(span, span),
                    num_variables=(1, 1),
                    placement=placement,
                ),
            )
            return masks.raw_loss_mask[:, :, 0].float().mean(dim=0)

        pin = coverage("pin")
        interior = pin[2 * span : T - 2 * span].mean()
        assert abs(interior - span / (T + span - 1)) < 0.005
        for day in (0, T - 1):
            assert abs(pin[day] / interior - 1) < 0.1
        for day in (span - 1, T - span):
            assert abs(pin[day] / interior - (2 * span - 1) / span) < 0.1
        inside = coverage("inside")
        assert inside[0] / inside[2 * span : T - 2 * span].mean() < 0.05

    def test_halo_span_invalid_placement_raises(self):
        with pytest.raises(ValueError, match="placement"):
            corrupt_era5_swt(
                B,
                T,
                V,
                self.N_BANDS,
                self.SUPPORTS,
                SWT_BUFFER,
                torch.device("cpu"),
                SwtHaloSpanMaskPolicy(placement="clip"),
            )


def _reference_halo_span_loop(
    b: int,
    t: int,
    v: int,
    supports: list[int],
    target_start: int,
    policy: SwtHaloSpanMaskPolicy,
) -> tuple[Tensor, Tensor]:
    """Original per-sample, per-span "inside" sampler, kept as a distribution oracle."""

    def randint(lo: int, hi: int) -> int:
        return lo if hi <= lo else int(torch.randint(lo, hi + 1, (1,)))

    band4d = torch.zeros(b, t, v, len(supports), dtype=torch.bool)
    raw = torch.zeros(b, t, v, dtype=torch.bool)
    window = t - target_start
    nvar_hi = min(policy.num_variables[1], v)
    for i in range(b):
        for _ in range(randint(*policy.num_spans)):
            length = min(randint(*policy.span_days), window)
            start = randint(target_start, t - length)
            end = start + length
            var_idx = torch.randperm(v)[: randint(policy.num_variables[0], nvar_hi)]
            raw[i, start:end][:, var_idx] = True
            for s, support in enumerate(supports):
                band4d[i, start : min(end + support - 1, t)][:, var_idx, s] = True
    return band4d, raw


class TestReconstructionConfigMerge:
    """The objective config must survive ``Config.merge`` (the launch path).

    ``build_config`` in ``internal/experiment.py`` calls ``config.merge(overrides)``
    on the whole experiment config, which OmegaConf structures recursively.
    OmegaConf rejects unions of dataclasses, so the mask policy must be exposed
    as flat knobs on the config rather than as a ``MaskPolicy`` field.
    """

    @dataclass
    class _FakeExperimentConfig(Config):
        """Mirror ``OlmoEarthExperimentConfig``: ``model`` typed as base ``Config``."""

        run_name: str = "test"
        model: Config = field(default_factory=Config)

    def _model_cfg(self, **recon_overrides: Any) -> Era5MultiObjectiveModelConfig:
        return Era5MultiObjectiveModelConfig(
            encoder_config=_small_encoder_cfg(),
            reconstruction_objective=ReconstructionObjectiveConfig(
                decoder=_small_decoder_cfg(), **recon_overrides
            ),
        )

    def test_merge_with_no_overrides_on_experiment_config(self):
        """Regression: an empty merge used to crash on the MaskPolicy union."""
        cfg = self._FakeExperimentConfig(model=self._model_cfg())
        merged = cfg.merge([])
        assert isinstance(merged.model, Era5MultiObjectiveModelConfig)

    def test_merge_halo_knobs_builds_halo_policy(self):
        cfg = self._FakeExperimentConfig(model=self._model_cfg())
        merged = cfg.merge(
            [
                "model.reconstruction_objective.mask_policy=swt_halo_span",
                "model.reconstruction_objective.span_num_spans=[4,10]",
                "model.reconstruction_objective.span_days=[30,120]",
                "model.reconstruction_objective.span_num_variables=[9,14]",
            ]
        )
        policy = merged.model.reconstruction_objective.build_mask_policy()
        assert isinstance(policy, SwtHaloSpanMaskPolicy)
        assert policy.num_spans == (4, 10)
        assert policy.span_days == (30, 120)
        assert policy.num_variables == (9, 14)
        assert policy.placement == "inside"
        assert merged.model.reconstruction_objective.mask_buffer is False
        merged = merged.merge(
            [
                "model.reconstruction_objective.span_placement=pin",
                "model.reconstruction_objective.mask_buffer=True",
            ]
        )
        recon = merged.model.reconstruction_objective
        assert recon.build_mask_policy().placement == "pin"
        assert recon.build().mask_buffer is True

    def test_merge_group_recon_mode_override(self):
        """Per-group loss gating is overridable from the CLI without code changes."""
        cfg = self._FakeExperimentConfig(model=self._model_cfg())
        merged = cfg.merge(
            [
                "model.reconstruction_objective.group_recon_mode.pressure=raw_plus_all_swt"
            ]
        )
        modes = merged.model.reconstruction_objective.group_recon_mode
        assert modes["pressure"] == "raw_plus_all_swt"
        # Other groups untouched.
        assert modes["thermo"] == "raw_plus_all_swt"
        assert modes["soil_moisture"] == "raw_plus_slow_swt"

    def test_naive_policy_default_and_knobs(self):
        policy = ReconstructionObjectiveConfig().build_mask_policy()
        assert isinstance(policy, SwtNaiveMaskPolicy)
        assert policy.budget == 0.5
        policy = ReconstructionObjectiveConfig(
            swt_naive_budget=0.7, swt_naive_raw_loss_mask_reduce="all"
        ).build_mask_policy()
        assert isinstance(policy, SwtNaiveMaskPolicy)
        assert policy.budget == 0.7
        assert policy.raw_loss_mask_reduce == "all"

    def test_invalid_policy_and_pairs_raise(self):
        with pytest.raises(ValueError, match="Unknown mask_policy"):
            ReconstructionObjectiveConfig(mask_policy="bogus").build_mask_policy()
        with pytest.raises(ValueError, match="span_days must be a"):
            ReconstructionObjectiveConfig(
                mask_policy="swt_halo_span", span_days=[7]
            ).build_mask_policy()
        with pytest.raises(ValueError, match="lo <= hi"):
            ReconstructionObjectiveConfig(
                mask_policy="swt_halo_span", span_num_spans=[5, 1]
            ).build_mask_policy()


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
            "corrupt_era5_swt",
            lambda *args, **kwargs: Era5CorruptionMasks(
                band_mask=torch.zeros(B, T, V * 7, dtype=torch.bool),
                raw_loss_mask=mask,
            ),
        )

        batch = replace(_make_batch(), era5=target)

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
            "corrupt_era5_swt",
            lambda *args, **kwargs: Era5CorruptionMasks(
                band_mask=torch.zeros(B, T, V * 7, dtype=torch.bool),
                raw_loss_mask=mask,
            ),
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
        batch = replace(_make_batch(), era5=target)

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


@pytest.mark.skipif(
    not SWT_STATS_PATH.is_file(),
    reason=f"swt_input_stats.json not found at {SWT_STATS_PATH}",
)
class TestReconstructionE2EBackward:
    """Full forward+backward integration with gradient and metric checks."""

    def _swt_model(self, **recon_overrides: Any):
        return Era5MultiObjectiveModelConfig(
            encoder_config=_small_encoder_cfg(
                is_swt_input=True,
                swt_input_stats_path=SWT_STATS_REL,
            ),
            reconstruction_objective=ReconstructionObjectiveConfig(
                decoder=_small_decoder_cfg(),
                **recon_overrides,
            ),
        ).build()

    def test_naive_policy(self):
        torch.manual_seed(0)
        model = self._swt_model(
            mask_policy="swt_naive",
            swt_naive_budget=0.7,
            swt_levels=[0, 1, 2],
            swt_lambda=0.1,
        )
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
        assert "reconstruction/band_masked_fraction" in metrics
        for lvl in [0, 1, 2]:
            assert f"reconstruction/swt_level_{lvl}_loss" in metrics
        assert "reconstruction/swt_deepest_approx_loss" in metrics
        assert metrics["reconstruction/swt_deepest_approx_loss"].item() > 0

    def test_halo_span_policy(self):
        torch.manual_seed(0)
        model = self._swt_model(
            mask_policy="swt_halo_span",
            swt_levels=[0, 1],
            swt_lambda=0.1,
        )
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
        # Halo band mask supervises strictly fewer raw positions than it hides.
        assert (
            metrics["reconstruction/masked_fraction"].item()
            < metrics["reconstruction/band_masked_fraction"].item()
        )


# ---------------------------------------------------------------------------
# SWT-input fixed normalization
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not SWT_STATS_PATH.is_file(),
    reason=f"swt_input_stats.json not found at {SWT_STATS_PATH}",
)
class TestSwtInputNormalization:
    """Fixed per-channel standardization of the encoder's SWT bands."""

    def _swt_encoder(self, **overrides: Any):
        overrides.setdefault("swt_input_stats_path", SWT_STATS_REL)
        cfg = _small_encoder_cfg(is_swt_input=True, **overrides)
        return cfg.build()

    def test_buffers_match_stats_file(self):
        """Registered buffers equal the JSON mean/std, shaped [1, 1, C]."""
        with SWT_STATS_PATH.open() as f:
            payload = json.load(f)
        enc = self._swt_encoder()

        assert not hasattr(enc, "swt_norm"), "BatchNorm swt_norm should be gone"
        c = enc.swt_channels
        assert c == len(payload["mean"]) == payload["num_channels"]
        assert enc.swt_norm_mean.shape == (1, 1, c)
        assert enc.swt_norm_std.shape == (1, 1, c)
        torch.testing.assert_close(
            enc.swt_norm_mean.view(-1),
            torch.tensor(payload["mean"], dtype=torch.float32),
        )
        torch.testing.assert_close(
            enc.swt_norm_std.view(-1),
            torch.tensor(payload["std"], dtype=torch.float32).clamp_(min=1e-6),
        )

    def test_apply_swt_standardizes_bands(self):
        """_apply_swt == (unnormalized bands - mean) / std, channel-aligned."""
        torch.manual_seed(0)
        enc = self._swt_encoder()
        era5 = torch.randn(B, T, V)

        out = enc._apply_swt(era5)

        # Recompute the unnormalized bands independently and standardize.
        swt = StationaryWaveletTransform1d(num_channels=V, max_levels=6)
        bands = swt(era5.transpose(1, 2), levels=enc.swt_input_levels, target_start=0)
        raw_bands = swt_bands_to_channels(bands, include_approx=True)
        expected = (raw_bands - enc.swt_norm_mean) / enc.swt_norm_std

        assert out.shape == (B, T, enc.swt_channels)
        torch.testing.assert_close(out, expected)

    def test_reference_channels_are_standardized(self):
        """Bands drawn from the reference dist normalize to ~0 mean / ~1 std.

        We synthesize unnormalized bands per channel as ``mean_c + std_c * z``
        (z ~ N(0, 1)); applying the fixed normalization must recover ~unit
        statistics per channel.
        """
        torch.manual_seed(0)
        enc = self._swt_encoder()
        mean = enc.swt_norm_mean.view(1, -1)
        std = enc.swt_norm_std.view(1, -1)
        z = torch.randn(20000, enc.swt_channels)
        raw_bands = mean + std * z
        normed = (raw_bands - mean) / std
        assert normed.mean(dim=0).abs().max().item() < 0.1
        assert (normed.std(dim=0) - 1.0).abs().max().item() < 0.1

    def test_channel_count_mismatch_raises(self):
        """include_approx=False -> 84 channels != 98 in stats -> ValueError."""
        with pytest.raises(ValueError, match="channels|include_approx"):
            self._swt_encoder(swt_input_include_approx=False)

    def test_levels_mismatch_raises(self):
        """Fewer levels changes the channel layout and must be rejected."""
        with pytest.raises(ValueError, match="channels|levels"):
            self._swt_encoder(swt_input_levels=[0, 1, 2])

    def test_missing_stats_path_raises(self):
        """is_swt_input without a stats path fails validation."""
        with pytest.raises(ValueError, match="swt_input_stats_path"):
            _small_encoder_cfg(is_swt_input=True).validate()

    def test_bad_stats_path_raises(self):
        """A non-existent stats path raises a clear FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            self._swt_encoder(swt_input_stats_path="does/not/exist.json")


# ---------------------------------------------------------------------------
# No-data (-9999) handling in SWT space (conservative whole-variable drop)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not SWT_STATS_PATH.is_file(),
    reason=f"swt_input_stats.json not found at {SWT_STATS_PATH}",
)
class TestSwtInputNoDataHandling:
    """No-data variable handling in SWT space.

    A variable with any no-data is 0-filled in SWT band space inside the encoder
    (universal, all objectives) and dropped from the reconstruction loss.
    """

    _NODATA_VAR = 7  # str: the only ERA5-Land variable with real no-data
    _NODATA_SAMPLE = 0

    def _build(self):
        model_cfg = Era5MultiObjectiveModelConfig(
            encoder_config=_small_encoder_cfg(
                is_swt_input=True,
                swt_input_stats_path=SWT_STATS_REL,
            ),
            reconstruction_objective=ReconstructionObjectiveConfig(
                decoder=_small_decoder_cfg(),
                mask_policy="swt_naive",
                swt_naive_budget=0.5,
                swt_levels=[0, 1, 2],
                swt_lambda=1.0,
                raw_lambda=1.0,
            ),
        )
        model = model_cfg.build()
        obj = model.objective_list[0]
        assert isinstance(obj, era5_multiobjective.ReconstructionObjective)
        return model, obj

    def _nodata_batch(self) -> Era5SupervisedBatch:
        """Batch whose `str` variable has a 27-day gap in one sample."""
        batch = _make_batch()
        valid = torch.ones(B, T, V, dtype=torch.bool)
        valid[self._NODATA_SAMPLE, 200:227, self._NODATA_VAR] = False
        era5 = batch.era5.clone()
        era5[self._NODATA_SAMPLE, 200:227, self._NODATA_VAR] = 0.0  # mirror impute
        return replace(batch, era5=era5, valid_mask=valid)

    def test_nodata_variable_zero_filled_in_swt_space(self):
        """`_apply_swt` 0-fills every band channel of a no-data variable."""
        torch.manual_seed(0)
        model, _obj = self._build()
        enc = model.encoder
        n_bands = enc.swt_num_bands
        batch = self._nodata_batch()

        bands = enc._apply_swt(batch.era5, batch.valid_mask)  # [B, T, C_swt]
        assert bands.shape == (B, T, V * n_bands)
        j = self._NODATA_VAR
        # The no-data variable's bands are entirely zero for the affected sample.
        nodata_bands = bands[self._NODATA_SAMPLE, :, j * n_bands : (j + 1) * n_bands]
        assert torch.count_nonzero(nodata_bands) == 0
        # A valid sample's same variable is present (non-zero).
        other_sample = bands[1, :, j * n_bands : (j + 1) * n_bands]
        assert torch.count_nonzero(other_sample) > 0
        # An unaffected variable in the no-data sample is untouched.
        other_var = bands[self._NODATA_SAMPLE, :, 0:n_bands]
        assert torch.count_nonzero(other_var) > 0

    def test_encoder_ignores_nodata_variable_content(self):
        """Encoder output is invariant to a fully-no-data variable's raw values.

        Uses the no-corruption_mask path (objective A / eval probe), proving the
        0-fill is universal, not reconstruction-specific.
        """
        torch.manual_seed(0)
        model, _obj = self._build()
        enc = model.encoder
        batch = _make_batch()
        valid = torch.ones(B, T, V, dtype=torch.bool)
        valid[:, :, self._NODATA_VAR] = False

        era5_a = batch.era5.clone()
        era5_a[:, :, self._NODATA_VAR] = 0.0
        era5_b = era5_a.clone()
        era5_b[:, :, self._NODATA_VAR] = 12345.0  # arbitrary garbage in dropped var

        enc.eval()
        with torch.no_grad():
            out_a = enc(era5=era5_a, timestamps=batch.timestamps, valid_mask=valid)
            out_b = enc(era5=era5_b, timestamps=batch.timestamps, valid_mask=valid)
        torch.testing.assert_close(out_a["pooled"], out_b["pooled"])
        torch.testing.assert_close(out_a["tokens"], out_b["tokens"])

    def test_nodata_fraction_metric(self):
        torch.manual_seed(0)
        model, obj = self._build()
        batch = self._nodata_batch()
        _loss, metrics = obj.compute(model.encoder, batch)
        assert "reconstruction/nodata_fraction" in metrics
        expected = 27.0 / (B * T * V)
        assert metrics["reconstruction/nodata_fraction"].item() == pytest.approx(
            expected, rel=1e-5
        )

    def test_dropped_variable_not_scored(self):
        """A fully-no-data variable must not contribute to the loss.

        Perturbing its predictions must not change the loss (it is excluded from
        both the raw and SWT terms).
        """
        torch.manual_seed(0)
        model, obj = self._build()
        # Make the target variable no-data across ALL samples/timesteps so it is
        # globally dropped, then confirm its decoder output does not affect loss.
        batch = _make_batch()
        valid = torch.ones(B, T, V, dtype=torch.bool)
        valid[:, :, self._NODATA_VAR] = False
        era5 = batch.era5.clone()
        era5[:, :, self._NODATA_VAR] = 0.0
        batch = replace(batch, era5=era5, valid_mask=valid)

        j = self._NODATA_VAR
        decoder = obj._module.decoder
        real_forward = decoder.forward

        def make_pred(scale: float):
            def fwd(tokens: Tensor, timestamps: Tensor) -> Tensor:
                out = real_forward(tokens, timestamps)
                perturbed = out.clone()
                perturbed[:, :, j] = perturbed[:, :, j] + scale
                return perturbed

            return fwd

        torch.manual_seed(1)
        decoder.forward = make_pred(0.0)  # type: ignore[method-assign]
        loss_a, _ = obj.compute(model.encoder, batch)
        torch.manual_seed(1)
        decoder.forward = make_pred(1000.0)  # type: ignore[method-assign]
        loss_b, _ = obj.compute(model.encoder, batch)
        decoder.forward = real_forward  # type: ignore[method-assign]

        torch.testing.assert_close(loss_a, loss_b)

    def test_loss_finite_and_backward(self):
        torch.manual_seed(0)
        model, obj = self._build()
        batch = self._nodata_batch()
        loss, _ = obj.compute(model.encoder, batch)
        assert loss.ndim == 0 and torch.isfinite(loss) and loss.item() > 0
        loss.backward()
        enc_grads = sum(
            p.grad.abs().sum().item()
            for p in model.encoder.parameters()
            if p.grad is not None
        )
        assert enc_grads > 0


# ===================================================================
# Pooled instance contrastive objective
# ===================================================================


def _contrastive_model(
    *, supervised: bool = False, pooling: str = "mean", **overrides: Any
) -> era5_multiobjective.Era5MultiObjectiveModel:
    settings: dict[str, Any] = dict(
        decoder=_small_decoder_cfg(),
        raw_lambda=1.0,
        swt_lambda=0.0,
        mask_policy="swt_halo_span",
        contrastive_lambda=0.1,
        contrastive_projector_hidden_dim=D,
        contrastive_projector_output_dim=16,
    )
    settings.update(overrides)
    return Era5MultiObjectiveModelConfig(
        encoder_config=_small_encoder_cfg(
            is_swt_input=True, swt_input_stats_path=SWT_STATS_REL, pooling=pooling
        ),
        reconstruction_objective=ReconstructionObjectiveConfig(**settings),
        supervised_objective=(
            SupervisedObjectiveConfig(
                tasks=[
                    SupervisedTaskConfig(
                        name="smoke_task", task_type="classification", num_classes=2
                    )
                ]
            )
            if supervised
            else None
        ),
    ).build()


class TestInstanceInfoNCE:
    """Verify loss semantics, both gradient branches, and FP32 computation."""

    def test_symmetric_reference_and_gradients(self):
        from olmoearth_pretrain.train.loss import InfoNCELoss

        a = torch.randn(B, 16, requires_grad=True)
        b = torch.randn(B, 16, requires_grad=True)
        loss, metrics = _instance_infonce(a, b, 0.2)
        reference = InfoNCELoss(tau=0.2)
        expected = (reference.compute(a, b) + reference.compute(b, a)) / 2
        torch.testing.assert_close(loss, expected)
        torch.testing.assert_close(loss, _instance_infonce(b, a, 0.2)[0])
        loss.backward()
        for x in (a, b):
            assert torch.isfinite(x.grad).all() and x.grad.abs().sum() > 0
        assert all(not value.requires_grad for value in metrics.values())
        assert metrics["contrastive_batch_size"] == B

    def test_pairing_and_uniform_logits(self):
        embeddings = torch.eye(B)
        paired, metrics = _instance_infonce(embeddings, embeddings, 0.1)
        shuffled, _ = _instance_infonce(embeddings, embeddings.roll(1, 0), 0.1)
        assert paired < shuffled
        assert metrics["contrastive_accuracy"] == 1
        uniform, _ = _instance_infonce(torch.ones(B, 8), torch.ones(B, 8), 0.1)
        torch.testing.assert_close(uniform, torch.tensor(math.log(B)))

    def test_autocast_and_zero_vectors(self):
        a = torch.zeros(B, 16, dtype=torch.bfloat16, requires_grad=True)
        b = torch.randn(B, 16, dtype=torch.bfloat16, requires_grad=True)
        with torch.autocast("cpu", dtype=torch.bfloat16):
            loss, metrics = _instance_infonce(a, b, 0.1)
        expected, _ = _instance_infonce(a.float(), b.float(), 0.1)
        assert loss.dtype == torch.float32
        torch.testing.assert_close(loss, expected, rtol=0, atol=0)
        assert all(torch.isfinite(value) for value in metrics.values())
        loss.backward()
        assert torch.isfinite(a.grad).all() and torch.isfinite(b.grad).all()


class TestContrastiveConfiguration:
    """Validate launch overrides and preserve disabled model compatibility."""

    @pytest.mark.parametrize(
        "settings",
        [
            {"contrastive_lambda": -0.1},
            {"contrastive_lambda": float("nan")},
            {"contrastive_lambda": float("inf")},
            {"contrastive_temperature": 0},
            {"contrastive_temperature": float("nan")},
            {"contrastive_temperature": float("inf")},
            {"num_views": 0},
            {"num_views": 3},
            {"num_views": True},
            {"contrastive_lambda": 0.1, "num_views": 1},
            {"contrastive_lambda": 0.1, "contrastive_projector_hidden_dim": 0},
            {"contrastive_lambda": 0.1, "contrastive_projector_output_dim": -1},
        ],
    )
    def test_invalid_settings(self, settings):
        with pytest.raises(ValueError):
            ReconstructionObjectiveConfig(**settings).validate()

    @pytest.mark.parametrize(
        "pooling,pooled_dim", [("mean", D), ("cls_mean_concat", 2 * D)]
    )
    def test_projector_dimensions_registration_and_checkpoint(
        self, pooling, pooled_dim
    ):
        model = _contrastive_model(pooling=pooling)
        projector = model.objectives["reconstruction"].projector
        assert isinstance(projector, Mlp)
        assert projector.fc1.in_features == pooled_dim
        assert projector.fc1.out_features == D
        assert projector.fc2.out_features == 16
        assert projector.drop1.p == projector.drop2.p == 0
        optimizer = torch.optim.AdamW(model.parameters())
        optimizer_params = {
            id(p) for group in optimizer.param_groups for p in group["params"]
        }
        assert all(id(p) in optimizer_params for p in projector.parameters())
        restored = _contrastive_model(pooling=pooling)
        restored.load_state_dict(model.state_dict(), strict=True)
        model.eval()
        restored.eval()
        batch = _make_batch()
        with torch.no_grad():
            first = model.encoder(batch.era5, batch.timestamps)["pooled"]
            second = restored.encoder(batch.era5, batch.timestamps)["pooled"]
        assert first.shape == (B, pooled_dim)
        torch.testing.assert_close(first, second, atol=0, rtol=0)
        torch.testing.assert_close(
            projector(first), restored.objectives["reconstruction"].projector(second)
        )

    def test_zero_lambda_preserves_state_and_rng(self):
        torch.manual_seed(97)
        model = _contrastive_model(contrastive_lambda=0)
        after_build = torch.get_rng_state()
        torch.manual_seed(97)
        direct = _contrastive_model(
            contrastive_lambda=0, contrastive_use_projector=False
        )
        assert torch.equal(after_build, torch.get_rng_state())
        direct.load_state_dict(model.state_dict(), strict=True)
        assert not any("projector" in name for name in model.state_dict())
        obj = model.objective_list[0]
        assert obj.num_views == 1 and obj._module.projector is None
        batch = _make_batch()
        rng = torch.get_rng_state()
        loss, metrics = obj.compute(model.encoder, batch)
        loss.backward()
        after_forward = torch.get_rng_state()
        torch.set_rng_state(rng)
        expected, original_metrics, _ = direct.objective_list[0]._compute_view(
            direct.encoder, batch
        )
        expected.backward()
        torch.testing.assert_close(loss, expected, rtol=0, atol=0)
        assert torch.equal(after_forward, torch.get_rng_state())
        for name, value in original_metrics.items():
            torch.testing.assert_close(metrics[name], value, rtol=0, atol=0)
        for (_, param), (_, reference) in zip(
            model.named_parameters(), direct.named_parameters()
        ):
            torch.testing.assert_close(param.grad, reference.grad, rtol=0, atol=0)
        assert not any("contrastive" in key for key in metrics)


class TestContrastiveReconstruction:
    """Shared two-view forwards, reconstruction averaging, and combined training."""

    @pytest.mark.parametrize("coefficient,views", [(0.0, None), (0.0, 2), (0.1, None)])
    def test_forward_counts_and_masks(self, monkeypatch, coefficient, views):
        model = _contrastive_model(contrastive_lambda=coefficient, num_views=views)
        obj = model.objective_list[0]
        masker = Mock(wraps=era5_multiobjective.corrupt_era5_swt)
        monkeypatch.setattr(era5_multiobjective, "corrupt_era5_swt", masker)
        masks, decoder_calls = [], []
        model.encoder.register_forward_pre_hook(
            lambda module, args, kwargs: masks.append(
                kwargs["corruption_mask"].clone()
            ),
            with_kwargs=True,
        )
        obj._module.decoder.register_forward_hook(lambda *args: decoder_calls.append(1))
        _, metrics = obj.compute(model.encoder, _make_batch())
        expected_views = 2 if coefficient > 0 or views == 2 else 1
        assert masker.call_count == len(masks) == len(decoder_calls) == expected_views
        assert metrics["reconstruction/num_views"] == expected_views
        if expected_views == 2:
            assert not torch.equal(masks[0], masks[1])
        if coefficient == 0:
            assert obj._module.projector is None

    @pytest.mark.parametrize("coefficient", [0.0, 0.1, 0.7])
    def test_loss_and_metric_averaging(self, monkeypatch, coefficient):
        model = _contrastive_model(
            contrastive_lambda=coefficient, contrastive_use_projector=False, num_views=2
        )
        obj = model.objective_list[0]
        a, b = (
            torch.randn(B, D, requires_grad=True),
            torch.randn(B, D, requires_grad=True),
        )
        outputs = iter(
            [
                (
                    torch.tensor(2.0),
                    {"reconstruction/raw_loss": torch.tensor(2.0)},
                    {"pooled": a},
                ),
                (
                    torch.tensor(6.0),
                    {"reconstruction/raw_loss": torch.tensor(6.0)},
                    {"pooled": b},
                ),
            ]
        )
        monkeypatch.setattr(obj, "_compute_view", lambda *args: next(outputs))
        loss, metrics = obj.compute(model.encoder, _make_batch())
        info, _ = _instance_infonce(a, b, obj.contrastive_temperature)
        torch.testing.assert_close(loss, 4 + coefficient * info)
        assert (
            metrics["reconstruction/raw_loss"]
            == metrics["reconstruction/recon_loss"]
            == 4
        )
        if coefficient:
            torch.testing.assert_close(
                metrics["reconstruction/contrastive_weighted_loss"], coefficient * info
            )

    @pytest.mark.parametrize("supervised", [False, True])
    @pytest.mark.parametrize("projector", [False, True])
    def test_training_backward(self, supervised, projector):
        # Turn reconstruction weights off to isolate InfoNCE's encoder gradients.
        model = _contrastive_model(
            supervised=supervised,
            contrastive_use_projector=projector,
            raw_lambda=0,
            swt_lambda=0,
        )
        batch = _make_batch()
        if not supervised:
            batch = Era5SslBatch(
                era5=batch.era5,
                timestamps=batch.timestamps,
                valid_mask=batch.valid_mask,
                task_name="ssl",
            )
        pooled_outputs, input_masks = [], []

        def record_output(module, args, kwargs, output):
            output["pooled"].retain_grad()
            pooled_outputs.append(output["pooled"])
            input_masks.append(kwargs.get("corruption_mask"))

        handle = model.encoder.register_forward_hook(record_output, with_kwargs=True)
        obj = model.objective_list[-1]
        with torch.autocast("cpu", dtype=torch.bfloat16):
            loss, metrics = obj.compute(model.encoder, batch)
        assert loss.dtype == torch.float32 and torch.isfinite(loss)
        loss.backward()
        for pooled in pooled_outputs:
            assert pooled.grad is not None and torch.isfinite(pooled.grad).all()
            assert pooled.grad.abs().sum() > 0
        assert any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in model.encoder.parameters()
        )
        if projector:
            assert all(
                p.grad is not None
                and torch.isfinite(p.grad).all()
                and p.grad.abs().sum() > 0
                for p in obj._module.projector.parameters()
            )
        assert all(
            torch.isfinite(value) and not value.requires_grad
            for value in metrics.values()
        )
        # Exercise the normal combined A+B backward and A's clean-input pass.
        if supervised:
            model.zero_grad(set_to_none=True)
            a_loss, _ = model.objective_list[0].compute(model.encoder, batch)
            assert input_masks[-1] is None
            b_loss, _ = obj.compute(model.encoder, batch)
            (a_loss + b_loss * obj.weight).backward()
        handle.remove()

    def test_single_sample_rejected_before_forward(self, monkeypatch):
        model = _contrastive_model()
        obj = model.objective_list[0]
        forward = Mock()
        monkeypatch.setattr(obj, "_compute_view", forward)
        with pytest.raises(ValueError, match="at least two samples"):
            obj.compute(model.encoder, _make_batch().microbatch(0, 1))
        forward.assert_not_called()

    def test_train_batch_microbatch_weighting(self):
        model = _contrastive_model(weight=2.5)
        obj = model.objective_list[0]
        batch = _make_batch()
        record_metric = Mock()
        harness = SimpleNamespace(
            model=model,
            objectives=[obj],
            device=torch.device("cpu"),
            _split_batch=lambda batch: [batch.microbatch(0, 2), batch.microbatch(2, 4)],
            _to_device=lambda batch: batch,
            _train_microbatch_context=lambda *args: nullcontext(),
            _model_forward_context=nullcontext,
            trainer=SimpleNamespace(record_metric=record_metric),
        )
        rng = torch.get_rng_state()
        era5_multiobjective.MultiObjectiveEra5TrainModule.train_batch(harness, batch)
        recorded = {call.args[0]: call.args[1] for call in record_metric.call_args_list}
        assert recorded["train/reconstruction/contrastive_batch_size"] == 2
        assert recorded["train/reconstruction/num_views"] == 2
        torch.testing.assert_close(
            recorded["train/reconstruction/loss"],
            2.5
            * (
                recorded["train/reconstruction/recon_loss"]
                + recorded["train/reconstruction/contrastive_weighted_loss"]
            ),
        )
        # The train loop must scale the whole B loss once and average gradients.
        actual_grads = {
            name: p.grad.clone()
            for name, p in model.named_parameters()
            if p.grad is not None
        }
        model.zero_grad(set_to_none=True)
        torch.set_rng_state(rng)
        for microbatch in harness._split_batch(batch):
            loss, _ = obj.compute(model.encoder, microbatch)
            (loss * 2.5 / 2).backward()
        for name, param in model.named_parameters():
            if name in actual_grads:
                torch.testing.assert_close(
                    param.grad, actual_grads[name], atol=0, rtol=0
                )

    def test_compile_projector_and_loss(self, monkeypatch):
        model = _contrastive_model()
        module = model.objectives["reconstruction"]
        decoder_compile, projector_compile = Mock(), Mock()
        with monkeypatch.context() as patch:
            patch.setattr(module.decoder, "apply_compile", decoder_compile)
            patch.setattr(module.projector, "compile", projector_compile)
            module.apply_compile()
        decoder_compile.assert_called_once_with()
        projector_compile.assert_called_once_with(dynamic=True)
        # CPU Dynamo smoke; production CUDA/Inductor is exercised on the cluster.
        module.projector.compile(backend="eager", dynamic=True)
        compiled_loss = torch.compile(_instance_infonce, backend="eager")
        a, b = module.projector(torch.randn(B, D)), module.projector(torch.randn(B, D))
        eager, _ = _instance_infonce(a, b, 0.1)
        traced, _ = compiled_loss(a, b, 0.1)
        torch.testing.assert_close(eager, traced)
        traced.backward()


class TestBufferMasking:
    """``mask_buffer`` moves the mask start to day 0 but never the loss start."""

    @pytest.mark.parametrize("mask_buffer", [False, True])
    def test_mask_start_and_metric(self, monkeypatch, mask_buffer):
        model = _contrastive_model(contrastive_lambda=0.0, mask_buffer=mask_buffer)
        obj = model.objective_list[0]
        masker = Mock(wraps=era5_multiobjective.corrupt_era5_swt)
        monkeypatch.setattr(era5_multiobjective, "corrupt_era5_swt", masker)
        _, metrics = obj.compute(model.encoder, _make_batch())
        assert masker.call_args.args[5] == (0 if mask_buffer else SWT_BUFFER)
        key = "reconstruction/buffer_band_masked_fraction"
        assert (key in metrics) is mask_buffer

    def test_buffer_masks_are_never_scored(self, monkeypatch):
        buffer_only = torch.zeros(B, T, V, dtype=torch.bool)
        buffer_only[:, 10:SWT_BUFFER, :] = True
        monkeypatch.setattr(
            era5_multiobjective,
            "corrupt_era5_swt",
            lambda *args: Era5CorruptionMasks(
                band_mask=buffer_only.repeat_interleave(7, dim=-1),
                raw_loss_mask=buffer_only,
            ),
        )
        model = _contrastive_model(contrastive_lambda=0.0, mask_buffer=True)
        _, metrics = model.objective_list[0].compute(model.encoder, _make_batch())
        assert metrics["reconstruction/raw_loss"] == 0
        assert metrics["reconstruction/masked_fraction"] == 0
        assert metrics["reconstruction/buffer_band_masked_fraction"] > 0


@pytest.fixture
def era5_launch_script(monkeypatch):
    """Import the actual launcher without resolving datasets or submitting jobs."""
    directory = _REPO_ROOT / "scripts/era5_supervised/v0"
    monkeypatch.syspath_prepend(str(directory))
    spec = importlib.util.spec_from_file_location(
        "era5_launch_test", directory / "script.py"
    )
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "_resolve_task_specs", lambda common: [])
    return module


class TestContrastiveLauncher:
    """Exercise real common knobs through OmegaConf and the model builder."""

    def test_merge_and_wiring(self, era5_launch_script):
        script = era5_launch_script
        common = script.Era5SupervisedCommonComponents(
            run_name="test",
            save_folder="unused",
            training_modalities=[],
            enable_supervised=False,
            enable_reconstruction=True,
        )
        common = common.merge(
            [
                "recon_contrastive_lambda=0.25",
                "recon_contrastive_temperature=0.3",
                "recon_contrastive_use_projector=False",
                "recon_contrastive_projector_hidden_dim=48",
                "recon_contrastive_projector_output_dim=24",
                "recon_num_views=2",
                "recon_mask_policy=swt_halo_span",
                "recon_span_placement=pin",
                "recon_mask_buffer=True",
            ]
        )
        cfg = script.build_model_config(common).merge([])
        recon = cfg.reconstruction_objective
        assert recon.mask_buffer is True
        assert recon.build_mask_policy().placement == "pin"
        assert recon.contrastive_lambda == 0.25
        assert recon.contrastive_temperature == 0.3
        assert recon.contrastive_use_projector is False
        assert recon.contrastive_projector_hidden_dim == 48
        assert recon.contrastive_projector_output_dim == 24
        assert recon.num_views == 2
        cfg = cfg.merge(["reconstruction_objective.contrastive_use_projector=True"])
        assert cfg.reconstruction_objective.contrastive_use_projector is True

    @pytest.mark.parametrize("reconstruction,microbatch", [(False, 4), (True, 1)])
    def test_invalid_launcher_combination(
        self, era5_launch_script, reconstruction, microbatch
    ):
        common = era5_launch_script.Era5SupervisedCommonComponents(
            run_name="test",
            save_folder="unused",
            training_modalities=[],
            enable_supervised=False,
            enable_reconstruction=reconstruction,
            recon_contrastive_lambda=0.1,
            rank_microbatch_size=microbatch,
        )
        with pytest.raises(
            ValueError, match="enable_reconstruction|rank_microbatch_size"
        ):
            era5_launch_script.build_model_config(common)


def test_regression_label_extractor_is_picklable() -> None:
    """Spawned eval DataLoader workers must be able to pickle the dataset's extractor.

    Regression: the closure returned by ``make_regression_extractor`` failed with
    ``Can't pickle local object`` the first time a window-level RegressionTask
    (CY-Bench yield) ran through the eval callback on Beaker.
    """
    spec = Era5TaskSpec(name="t", task_type="regression", num_classes=1)
    for fn in (
        spec.get_label_extractor(),
        make_regression_extractor("value"),
        LABEL_EXTRACTORS["default_regression"],
    ):
        restored = pickle.loads(pickle.dumps(fn))
        out = restored({"value": torch.tensor(3.5), "valid": torch.tensor(1.0)})
        assert float(out) == 3.5 and out.shape == ()


class TestLearnedPositionEmbedding:
    """The encoder's optional learned position embedding (end-aligned)."""

    def test_default_adds_no_parameters(self):
        encoder = _small_encoder_cfg().build()
        assert encoder.pos_embed is None
        assert not any("pos_embed" in key for key in encoder.state_dict())

    def test_learned_shape_and_forward(self):
        encoder = _small_encoder_cfg(position_embedding="learned").build()
        num_tokens = (T - encoder.patch_kernel_size) // encoder.patch_stride + 1
        assert encoder.pos_embed.shape == (1, num_tokens, D)
        batch = _make_batch()
        out = encoder(era5=batch.era5, timestamps=batch.timestamps)
        assert out["tokens"].shape == (B, num_tokens, D)
        out["pooled"].sum().backward()
        assert encoder.pos_embed.grad is not None
        assert encoder.pos_embed.grad.abs().sum() > 0

    def test_end_aligned_rows(self):
        """A shorter sequence uses the last rows; only the used rows matter."""
        torch.manual_seed(0)
        plain = _small_encoder_cfg().build().eval()
        learned = _small_encoder_cfg(position_embedding="learned").build().eval()
        missing, _ = learned.load_state_dict(plain.state_dict(), strict=False)
        assert missing == ["pos_embed"]
        batch = _make_batch()
        with torch.no_grad():
            learned.pos_embed.zero_()
            # Only the first (oldest) row; random, since the LayerNorms would
            # cancel a constant shift across features.
            learned.pos_embed[0, 0] = torch.randn(D)
        short = T - learned.patch_stride * 2  # two tokens fewer
        for t, should_differ in ((T, True), (short, False)):
            with torch.no_grad():
                a = plain(era5=batch.era5[:, :t], timestamps=batch.timestamps[:, :t])
                b = learned(era5=batch.era5[:, :t], timestamps=batch.timestamps[:, :t])
            assert (not torch.allclose(a["pooled"], b["pooled"])) is should_differ

    def test_invalid_value_raises(self):
        with pytest.raises(ValueError, match="position_embedding"):
            _small_encoder_cfg(position_embedding="sincos").build()

    def test_launcher_knob(self, era5_launch_script):
        common = era5_launch_script.Era5SupervisedCommonComponents(
            run_name="test",
            save_folder="unused",
            training_modalities=[],
            enable_supervised=False,
            enable_reconstruction=True,
        ).merge(["encoder_position_embedding=learned"])
        cfg = era5_launch_script.build_model_config(common)
        assert cfg.encoder_config.position_embedding == "learned"


class TestPooledLayerNorm:
    """The encoder's optional parameter-free LayerNorm on the pooled embedding."""

    @pytest.mark.parametrize("pooling", ["mean", "cls", "attention", "cls_mean_concat"])
    def test_each_chunk_is_normalized(self, pooling):
        encoder = _small_encoder_cfg(pooling=pooling, pooled_norm="layernorm").build()
        batch = _make_batch()
        pooled = encoder(era5=batch.era5, timestamps=batch.timestamps)["pooled"]
        chunks = pooled.reshape(B, -1, D)
        assert chunks.shape[1] == (2 if pooling == "cls_mean_concat" else 1)
        torch.testing.assert_close(
            chunks.mean(-1), torch.zeros(B, chunks.shape[1]), atol=1e-4, rtol=0
        )
        torch.testing.assert_close(
            chunks.var(-1, correction=0),
            torch.ones(B, chunks.shape[1]),
            atol=1e-3,
            rtol=0,
        )

    def test_adds_no_parameters_and_loads_old_checkpoints(self):
        plain = _small_encoder_cfg().build()
        normed = _small_encoder_cfg(pooled_norm="layernorm").build()
        assert set(plain.state_dict()) == set(normed.state_dict())
        normed.load_state_dict(plain.state_dict(), strict=True)

    def test_invalid_value_raises(self):
        with pytest.raises(ValueError, match="pooled_norm"):
            _small_encoder_cfg(pooled_norm="batchnorm").build()

    def test_launcher_knob(self, era5_launch_script):
        common = era5_launch_script.Era5SupervisedCommonComponents(
            run_name="test",
            save_folder="unused",
            training_modalities=[],
            enable_supervised=False,
            enable_reconstruction=True,
        ).merge(["encoder_pooled_norm=layernorm"])
        cfg = era5_launch_script.build_model_config(common)
        assert cfg.encoder_config.pooled_norm == "layernorm"


class TestCollapseMetrics:
    """Training-step and eval-time collapse monitors."""

    def test_reconstruction_only_logs_pooled_geometry(self):
        model = _contrastive_model(contrastive_lambda=0.0)
        _, metrics = model.objective_list[0].compute(model.encoder, _make_batch())
        for key in ("pooled_std_r", "pooled_mean_cos"):
            value = metrics[f"reconstruction/{key}"]
            assert torch.isfinite(value) and not value.requires_grad
        assert 0.0 <= float(metrics["reconstruction/pooled_std_r"]) <= 1.0 + 1e-6
        assert not any(k.endswith(("pooled_std", "projected_std")) for k in metrics)
        assert "reconstruction/projected_std_r" not in metrics

    def test_contrastive_logs_projected_geometry(self):
        model = _contrastive_model(contrastive_lambda=0.1)
        _, metrics = model.objective_list[0].compute(model.encoder, _make_batch())
        for key in (
            "pooled_std_r",
            "pooled_mean_cos",
            "projected_std_r",
            "projected_mean_cos",
        ):
            assert torch.isfinite(metrics[f"reconstruction/{key}"])

    def test_evaluator_geometry_metrics(self):
        torch.manual_seed(0)
        out = era5_evaluator_callback._embedding_geometry_metrics(
            "lfmc_woody_eval", torch.randn(500, 16)
        )
        prefix = "eval_other/lfmc_woody_eval"
        assert set(out) == {
            f"{prefix}/{k}"
            for k in (
                "pooled_std_r",
                "pooled_mean_cos",
                "effective_rank",
                "top10pc_var_share",
            )
        }
        assert 1.0 <= out[f"{prefix}/effective_rank"] <= 16.0
        assert (
            era5_evaluator_callback._embedding_geometry_metrics("x", torch.randn(1, 16))
            == {}
        )


class TestRawLossByRegion:
    """Temporary diagnostic: raw loss on the last 83 days vs earlier target days."""

    RECENT = "reconstruction/raw_loss_last83d"
    EARLIER = "reconstruction/raw_loss_earlier"

    def _mask_only(self, monkeypatch, start: int, end: int) -> None:
        raw = torch.zeros(B, T, V, dtype=torch.bool)
        raw[:, start:end, :] = True
        monkeypatch.setattr(
            era5_multiobjective,
            "corrupt_era5_swt",
            lambda *args: Era5CorruptionMasks(
                band_mask=raw.repeat_interleave(7, dim=-1), raw_loss_mask=raw
            ),
        )

    def test_both_regions_logged(self):
        model = _contrastive_model(contrastive_lambda=0.0)
        _, metrics = model.objective_list[0].compute(model.encoder, _make_batch())
        for key in (self.RECENT, self.EARLIER):
            assert torch.isfinite(metrics[key]) and not metrics[key].requires_grad

    @pytest.mark.parametrize("recent_only", [True, False])
    def test_region_without_scored_cells_is_omitted(self, monkeypatch, recent_only):
        if recent_only:
            self._mask_only(monkeypatch, T - SWT_BUFFER, T)
        else:
            self._mask_only(monkeypatch, SWT_BUFFER, T - SWT_BUFFER)
        model = _contrastive_model(contrastive_lambda=0.0, num_views=2)
        _, metrics = model.objective_list[0].compute(model.encoder, _make_batch())
        assert (self.RECENT in metrics) is recent_only
        assert (self.EARLIER in metrics) is not recent_only

    def test_absent_without_raw_loss(self):
        model = _contrastive_model(
            contrastive_lambda=0.0, raw_lambda=0.0, swt_lambda=1.0
        )
        _, metrics = model.objective_list[0].compute(model.encoder, _make_batch())
        assert self.RECENT not in metrics and self.EARLIER not in metrics
