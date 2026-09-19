"""Unit tests for the supervision head module."""

import pytest
import torch

from olmoearth_pretrain.data.constants import MISSING_VALUE
from olmoearth_pretrain.datatypes import MaskedOlmoEarthSample, MaskValue
from olmoearth_pretrain.nn.supervision_head import (
    SupervisionHead,
    SupervisionHeadConfig,
    SupervisionModalityConfig,
    SupervisionTaskType,
    _build_valid_mask,
    _day_of_year_encoding,
    compute_supervision_loss,
)

B, P_H, P_W, D = 2, 4, 4, 8
# Output width used for the non-spatial (per-sample) head tests; latlon is the
# example non-spatial modality but any [B, C] head behaves the same.
NON_SPATIAL_CHANNELS = 3
MAX_PATCH_SIZE = 8
H_PIX, W_PIX = P_H * MAX_PATCH_SIZE, P_W * MAX_PATCH_SIZE  # 32, 32


def _make_register_grid() -> torch.Tensor:
    """A register grid [B, n_h, n_w, D] the heads read from."""
    return torch.randn(B, P_H, P_W, D, requires_grad=True)


def _make_batch_with_worldcover() -> MaskedOlmoEarthSample:
    """Batch with worldcover raw pixels [B, H, W, 1, 1]."""
    wc_values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0]
    wc = torch.tensor(wc_values)[torch.randint(0, len(wc_values), (B, H_PIX, W_PIX))]
    wc = wc.unsqueeze(-1).unsqueeze(-1)  # [B, H, W, 1, 1]
    wc_mask = torch.full((B, H_PIX, W_PIX, 1, 1), MaskValue.DECODER.value)
    timestamps = torch.tensor([[1, 1, 2023]], dtype=torch.long).expand(B, -1, -1)
    return MaskedOlmoEarthSample(
        timestamps=timestamps,
        worldcover=wc,
        worldcover_mask=wc_mask,
    )


def _make_batch_with_srtm() -> MaskedOlmoEarthSample:
    """Batch with srtm raw pixels [B, H, W, 1, 1] for regression."""
    srtm = torch.rand(B, H_PIX, W_PIX, 1, 1)
    srtm_mask = torch.full((B, H_PIX, W_PIX, 1, 1), MaskValue.DECODER.value)
    timestamps = torch.tensor([[1, 1, 2023]], dtype=torch.long).expand(B, -1, -1)
    return MaskedOlmoEarthSample(
        timestamps=timestamps,
        srtm=srtm,
        srtm_mask=srtm_mask,
    )


class TestSupervisionHead:
    """Test SupervisionHead forward pass on the register grid."""

    @pytest.fixture
    def worldcover_config(self) -> dict[str, SupervisionModalityConfig]:
        """WorldCover classification config fixture."""
        return {
            "worldcover": SupervisionModalityConfig(
                task_type=SupervisionTaskType.CLASSIFICATION,
                num_output_channels=11,
                class_values=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0],
            ),
        }

    def test_forward_shape(
        self, worldcover_config: dict[str, SupervisionModalityConfig]
    ) -> None:
        """Spatial head output is resized to the target: [B, H, W, 1, C]."""
        head = SupervisionHead(
            worldcover_config, embedding_dim=D, max_patch_size=MAX_PATCH_SIZE
        )
        preds = head(_make_register_grid(), _make_batch_with_worldcover())
        assert "worldcover" in preds
        assert preds["worldcover"].shape == (B, H_PIX, W_PIX, 1, 11)

    def test_forward_downsample(
        self, worldcover_config: dict[str, SupervisionModalityConfig]
    ) -> None:
        """When the unfolded grid is larger than the target, output is downsampled."""
        small_h, small_w = 16, 16
        head = SupervisionHead(
            worldcover_config, embedding_dim=D, max_patch_size=MAX_PATCH_SIZE
        )
        wc_values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0]
        wc = torch.tensor(wc_values)[
            torch.randint(0, len(wc_values), (B, small_h, small_w))
        ]
        wc = wc.unsqueeze(-1).unsqueeze(-1)
        timestamps = torch.tensor([[1, 1, 2023]], dtype=torch.long).expand(B, -1, -1)
        batch = MaskedOlmoEarthSample(
            timestamps=timestamps,
            worldcover=wc,
            worldcover_mask=torch.full(
                (B, small_h, small_w, 1, 1), MaskValue.DECODER.value
            ),
        )
        preds = head(_make_register_grid(), batch)
        assert preds["worldcover"].shape == (B, small_h, small_w, 1, 11)

    def test_missing_target_still_produces_output(
        self, worldcover_config: dict[str, SupervisionModalityConfig]
    ) -> None:
        """Heads run even when the batch has no target for them (FSDP)."""
        head = SupervisionHead(
            worldcover_config, embedding_dim=D, max_patch_size=MAX_PATCH_SIZE
        )
        timestamps = torch.tensor([[1, 1, 2023]], dtype=torch.long).expand(B, -1, -1)
        preds = head(
            _make_register_grid(), MaskedOlmoEarthSample(timestamps=timestamps)
        )
        assert "worldcover" in preds
        # No target to resize to: the unfolded grid is returned as is.
        assert preds["worldcover"].shape == (
            B,
            P_H * MAX_PATCH_SIZE,
            P_W * MAX_PATCH_SIZE,
            1,
            11,
        )
        assert preds["worldcover"].requires_grad

    def test_regression_head(self) -> None:
        """Regression head produces [B, H, W, 1, 1] output."""
        cfg = {
            "srtm": SupervisionModalityConfig(
                task_type=SupervisionTaskType.REGRESSION,
                num_output_channels=1,
            ),
        }
        head = SupervisionHead(cfg, embedding_dim=D, max_patch_size=MAX_PATCH_SIZE)
        preds = head(_make_register_grid(), _make_batch_with_srtm())
        assert "srtm" in preds
        assert preds["srtm"].shape == (B, H_PIX, W_PIX, 1, 1)

    def test_non_spatial_forward(self) -> None:
        """Non-spatial modality reads the mean-pooled grid and produces [B, C]."""
        cfg = {
            "latlon": SupervisionModalityConfig(
                task_type=SupervisionTaskType.REGRESSION,
                num_output_channels=NON_SPATIAL_CHANNELS,
            ),
        }
        head = SupervisionHead(cfg, embedding_dim=D, max_patch_size=MAX_PATCH_SIZE)
        timestamps = torch.tensor([[1, 1, 2023]], dtype=torch.long).expand(B, -1, -1)
        batch = MaskedOlmoEarthSample(timestamps=timestamps, latlon=torch.rand(B, 2))
        preds = head(_make_register_grid(), batch)
        assert "latlon" in preds
        assert preds["latlon"].shape == (B, NON_SPATIAL_CHANNELS)
        assert preds["latlon"].requires_grad


class TestBuildValidMask:
    """Test _build_valid_mask helper."""

    def test_all_valid(self) -> None:
        """No MISSING_VALUE means all True."""
        target = torch.ones(B, H_PIX, W_PIX, 1, 1)
        mask = _build_valid_mask(target)
        assert mask.all()

    def test_some_missing(self) -> None:
        """MISSING_VALUE pixels are False."""
        target = torch.ones(B, H_PIX, W_PIX, 1, 1)
        target[0, 0, 0, 0, 0] = MISSING_VALUE
        mask = _build_valid_mask(target)
        assert not mask[0, 0, 0, 0]
        assert mask[0, 0, 1, 0]

    def test_multitemporal(self) -> None:
        """Valid mask works with T > 1."""
        T = 3
        target = torch.ones(B, H_PIX, W_PIX, T, 1)
        target[0, 0, 0, 1, 0] = MISSING_VALUE
        mask = _build_valid_mask(target)
        assert mask.shape == (B, H_PIX, W_PIX, T)
        assert mask[0, 0, 0, 0]
        assert not mask[0, 0, 0, 1]
        assert mask[0, 0, 0, 2]


class TestComputeSupervisionLoss:
    """Test compute_supervision_loss for each task type."""

    def test_classification_loss(self) -> None:
        """Classification loss is positive and finite."""
        cfg = {
            "worldcover": SupervisionModalityConfig(
                task_type=SupervisionTaskType.CLASSIFICATION,
                num_output_channels=11,
                weight=0.1,
                class_values=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0],
            ),
        }
        head = SupervisionHead(cfg, embedding_dim=D, max_patch_size=MAX_PATCH_SIZE)
        pred = torch.randn(B, H_PIX, W_PIX, 1, 11)
        batch = _make_batch_with_worldcover()
        total_loss, per_mod = compute_supervision_loss(
            {"worldcover": pred}, batch, head
        )
        assert total_loss.ndim == 0
        assert total_loss > 0
        assert "worldcover" in per_mod

    def test_regression_loss(self) -> None:
        """Regression loss is positive and finite."""
        cfg = {
            "srtm": SupervisionModalityConfig(
                task_type=SupervisionTaskType.REGRESSION,
                num_output_channels=1,
                weight=1.0,
            ),
        }
        head = SupervisionHead(cfg, embedding_dim=D, max_patch_size=MAX_PATCH_SIZE)
        pred = torch.randn(B, H_PIX, W_PIX, 1, 1)
        batch = _make_batch_with_srtm()
        total_loss, per_mod = compute_supervision_loss({"srtm": pred}, batch, head)
        assert total_loss.ndim == 0
        assert total_loss > 0
        assert "srtm" in per_mod

    def test_multitemporal_regression_loss(self) -> None:
        """Regression loss works across multiple timesteps."""
        T = 3
        cfg = {
            "ndvi": SupervisionModalityConfig(
                task_type=SupervisionTaskType.REGRESSION,
                num_output_channels=1,
                weight=1.0,
            ),
        }
        head = SupervisionHead(cfg, embedding_dim=D, max_patch_size=MAX_PATCH_SIZE)
        pred = torch.randn(B, H_PIX, W_PIX, T, 1)
        timestamps = torch.tensor([[1, 1, 2023]], dtype=torch.long).expand(B, -1, -1)
        ndvi_target = torch.rand(B, H_PIX, W_PIX, T, 1)
        batch = MaskedOlmoEarthSample(timestamps=timestamps, ndvi=ndvi_target)
        total_loss, per_mod = compute_supervision_loss({"ndvi": pred}, batch, head)
        assert total_loss.ndim == 0
        assert total_loss > 0
        assert "ndvi" in per_mod

    def test_all_missing_returns_zero(self) -> None:
        """Entirely missing target yields zero loss."""
        cfg = {
            "srtm": SupervisionModalityConfig(
                task_type=SupervisionTaskType.REGRESSION,
                num_output_channels=1,
            ),
        }
        head = SupervisionHead(cfg, embedding_dim=D, max_patch_size=MAX_PATCH_SIZE)
        timestamps = torch.tensor([[1, 1, 2023]], dtype=torch.long).expand(B, -1, -1)
        srtm = torch.full((B, H_PIX, W_PIX, 1, 1), MISSING_VALUE, dtype=torch.float)
        batch = MaskedOlmoEarthSample(timestamps=timestamps, srtm=srtm)
        pred = torch.randn(B, H_PIX, W_PIX, 1, 1)
        total_loss, per_mod = compute_supervision_loss({"srtm": pred}, batch, head)
        assert total_loss == 0.0

    def test_binary_classification_loss(self) -> None:
        """Binary classification loss is positive and finite."""
        cfg = {
            "openstreetmap_raster": SupervisionModalityConfig(
                task_type=SupervisionTaskType.BINARY_CLASSIFICATION,
                num_output_channels=30,
            ),
        }
        head = SupervisionHead(cfg, embedding_dim=D, max_patch_size=MAX_PATCH_SIZE)
        pred = torch.randn(B, H_PIX, W_PIX, 1, 30)
        timestamps = torch.tensor([[1, 1, 2023]], dtype=torch.long).expand(B, -1, -1)
        osm = torch.randint(0, 2, (B, H_PIX, W_PIX, 1, 30)).float()
        batch = MaskedOlmoEarthSample(
            timestamps=timestamps,
            openstreetmap_raster=osm,
        )
        total_loss, per_mod = compute_supervision_loss(
            {"openstreetmap_raster": pred}, batch, head
        )
        assert total_loss.ndim == 0
        assert "openstreetmap_raster" in per_mod


class TestSupervisionHeadConfig:
    """Test SupervisionHeadConfig building."""

    def test_build(self) -> None:
        """Config builds a SupervisionHead with correct modality heads."""
        config = SupervisionHeadConfig(
            modality_configs={
                "worldcover": SupervisionModalityConfig(
                    task_type=SupervisionTaskType.CLASSIFICATION,
                    num_output_channels=11,
                    class_values=[
                        0.1,
                        0.2,
                        0.3,
                        0.4,
                        0.5,
                        0.6,
                        0.7,
                        0.8,
                        0.9,
                        0.95,
                        1.0,
                    ],
                ),
                "srtm": SupervisionModalityConfig(
                    task_type=SupervisionTaskType.REGRESSION,
                    num_output_channels=1,
                ),
            }
        )
        head = config.build(embedding_dim=D, max_patch_size=MAX_PATCH_SIZE)
        assert isinstance(head, SupervisionHead)
        assert "worldcover" in head.heads
        assert "srtm" in head.heads
        assert head.max_patch_size == MAX_PATCH_SIZE
        assert head.heads["worldcover"].out_features == MAX_PATCH_SIZE**2 * 11
        assert head.heads["srtm"].out_features == MAX_PATCH_SIZE**2 * 1

    def test_non_spatial_modality_head_size(self) -> None:
        """Non-spatial modality (latlon) head output is num_channels, not mps^2 * num_channels."""
        config = SupervisionHeadConfig(
            modality_configs={
                "latlon": SupervisionModalityConfig(
                    task_type=SupervisionTaskType.REGRESSION,
                    num_output_channels=NON_SPATIAL_CHANNELS,
                ),
            }
        )
        head = config.build(embedding_dim=D, max_patch_size=MAX_PATCH_SIZE)
        assert head.heads["latlon"].out_features == NON_SPATIAL_CHANNELS
        assert "latlon" in head._non_spatial_modalities

    def test_classification_requires_class_values(self) -> None:
        """Classification without class_values raises ValueError."""
        with pytest.raises(ValueError, match="class_values"):
            SupervisionModalityConfig(
                task_type=SupervisionTaskType.CLASSIFICATION,
                num_output_channels=11,
            )

    def test_spatial_unfold_overrides_max_patch_size(self) -> None:
        """spatial_unfold=1 gives one prediction per register cell (pixel registers)."""
        config = SupervisionHeadConfig(
            modality_configs={
                "srtm": SupervisionModalityConfig(
                    task_type=SupervisionTaskType.REGRESSION,
                    num_output_channels=1,
                ),
            },
            spatial_unfold=1,
        )
        head = config.build(embedding_dim=D, max_patch_size=MAX_PATCH_SIZE)
        assert head.max_patch_size == 1
        assert head.heads["srtm"].out_features == 1
        # A grid already at target resolution predicts it directly (no resize).
        register_grid = torch.randn(B, H_PIX, W_PIX, D)
        preds = head(register_grid, _make_batch_with_srtm())
        assert preds["srtm"].shape == (B, H_PIX, W_PIX, 1, 1)

    def test_spatial_unfold_must_be_positive(self) -> None:
        """spatial_unfold below 1 is rejected."""
        with pytest.raises(ValueError, match="spatial_unfold"):
            SupervisionHeadConfig(modality_configs={}, spatial_unfold=0)


def _s2_time_conditioned_config(
    num_bands: int = 12,
) -> dict[str, SupervisionModalityConfig]:
    """The pixrecon head: time-conditioned MSE on the raw S2 L2A bands."""
    return {
        "sentinel2_l2a": SupervisionModalityConfig(
            task_type=SupervisionTaskType.REGRESSION,
            num_output_channels=num_bands,
            weight=0.05,
            regression_loss_type="mse",
            time_conditioned=True,
            time_harmonics=4,
            time_mlp_hidden_dim=16,
        ),
    }


class TestTimeConditionedSupervision:
    """Time-conditioned (register grid x day-of-year MLP) supervision heads."""

    T = 3
    NUM_BANDS = 12

    def _make_timestamps(self) -> torch.Tensor:
        # (day, month0, year): Jan 1, Apr 15, Jul 1 of 2023.
        return torch.tensor(
            [[[1, 0, 2023], [15, 3, 2023], [1, 6, 2023]]], dtype=torch.long
        ).expand(B, -1, -1)

    def _make_head(self) -> SupervisionHead:
        return SupervisionHead(
            _s2_time_conditioned_config(self.NUM_BANDS),
            embedding_dim=D,
            max_patch_size=MAX_PATCH_SIZE,
        )

    def test_head_is_a_small_mlp_over_cell_and_time(self) -> None:
        """Linear(D + 2K, hidden) -> GELU -> Linear(hidden, C); no sub-cell unfold."""
        head = self._make_head()
        mlp = head.heads["sentinel2_l2a"]
        assert mlp[0].in_features == D + 2 * 4
        assert mlp[0].out_features == 16
        assert mlp[2].out_features == self.NUM_BANDS
        assert "sentinel2_l2a" in head._time_conditioned_modalities

    def test_forward_shape_time_dependence_and_locality(self) -> None:
        """Per-(cell, timestep) predictions from the time-free register grid.

        With a grid-resolution target (no interpolation): predictions vary across
        timesteps (the time conditioning is live), and cell (i, j)'s prediction
        depends ONLY on register_grid[:, i, j] (the per-cell forcing that makes the
        fitted trajectory readable by a frozen per-cell probe).
        """
        head = self._make_head()
        register_grid = torch.randn(B, P_H, P_W, D)
        target = torch.randn(B, P_H, P_W, self.T, self.NUM_BANDS)
        batch = MaskedOlmoEarthSample(
            timestamps=self._make_timestamps(), sentinel2_l2a=target
        )
        preds = head(register_grid, batch)
        out = preds["sentinel2_l2a"]
        assert out.shape == (B, P_H, P_W, self.T, self.NUM_BANDS)
        # Same cell, different timesteps -> different predictions.
        assert not torch.allclose(out[:, :, :, 0], out[:, :, :, 1])
        # Perturbing one cell leaves every other cell's predictions unchanged.
        perturbed = register_grid.clone()
        perturbed[:, 0, 0] += 1.0
        out_perturbed = head(perturbed, batch)["sentinel2_l2a"]
        assert not torch.allclose(out_perturbed[:, 0, 0], out[:, 0, 0])
        torch.testing.assert_close(out_perturbed[:, 1:], out[:, 1:])

    def test_forward_interpolates_to_pixel_target(self) -> None:
        """A pixel-resolution target triggers bilinear upsampling of the grid preds."""
        head = self._make_head()
        register_grid = torch.randn(B, P_H, P_W, D)
        target = torch.randn(B, H_PIX, W_PIX, self.T, self.NUM_BANDS)
        batch = MaskedOlmoEarthSample(
            timestamps=self._make_timestamps(), sentinel2_l2a=target
        )
        preds = head(register_grid, batch)
        assert preds["sentinel2_l2a"].shape == (
            B,
            H_PIX,
            W_PIX,
            self.T,
            self.NUM_BANDS,
        )

    def test_loss_masks_missing_and_reaches_registers(self) -> None:
        """Masked MSE over (pixel, timestep) holes; the loss reaches the grid."""
        head = self._make_head()
        register_grid = torch.randn(B, P_H, P_W, D, requires_grad=True)
        target = torch.randn(B, H_PIX, W_PIX, self.T, self.NUM_BANDS)
        # A cloud hole (all bands missing at one pixel/timestep block) and a fully
        # missing timestep, both excluded by the valid mask.
        target[:, :4, :4, 0] = MISSING_VALUE
        target[:, :, :, 2] = MISSING_VALUE
        batch = MaskedOlmoEarthSample(
            timestamps=self._make_timestamps(), sentinel2_l2a=target
        )
        preds = head(register_grid, batch)
        total_loss, per_mod = compute_supervision_loss(preds, batch, head)
        assert total_loss.ndim == 0
        assert torch.isfinite(total_loss)
        assert "sentinel2_l2a" in per_mod
        total_loss.backward()
        assert register_grid.grad is not None
        assert torch.isfinite(register_grid.grad).all()
        assert register_grid.grad.abs().sum() > 0
        # A fully missing modality contributes a zero loss, not an error.
        valid = _build_valid_mask(target)
        assert valid.shape == (B, H_PIX, W_PIX, self.T)
        assert not valid[:, :, :, 2].any()

    def test_requires_timestamps(self) -> None:
        """No timestamps -> no day-of-year basis -> a clear error."""
        head = self._make_head()
        batch = MaskedOlmoEarthSample(
            timestamps=None,  # type: ignore[arg-type]
            sentinel2_l2a=torch.randn(B, P_H, P_W, self.T, self.NUM_BANDS),
        )
        with pytest.raises(ValueError, match="timestamps"):
            head(torch.randn(B, P_H, P_W, D), batch)

    def test_day_of_year_encoding(self) -> None:
        """Jan 1 encodes as (sin 0, cos 1) x K, and the encoding is year-invariant."""
        jan1_2023 = torch.tensor([[[1, 0, 2023]]], dtype=torch.long)
        phi = _day_of_year_encoding(jan1_2023, num_harmonics=4)  # [1, 1, 8]
        torch.testing.assert_close(phi[0, 0, :4], torch.zeros(4))
        torch.testing.assert_close(phi[0, 0, 4:], torch.ones(4))
        jul15_2019 = torch.tensor([[[15, 6, 2019]]], dtype=torch.long)
        jul15_2024 = torch.tensor([[[15, 6, 2024]]], dtype=torch.long)
        torch.testing.assert_close(
            _day_of_year_encoding(jul15_2019, num_harmonics=4),
            _day_of_year_encoding(jul15_2024, num_harmonics=4),
        )

    def test_requires_multitemporal_modality(self) -> None:
        """time_conditioned on a static modality (srtm) raises."""
        cfg = {
            "srtm": SupervisionModalityConfig(
                task_type=SupervisionTaskType.REGRESSION,
                num_output_channels=1,
                time_conditioned=True,
            ),
        }
        with pytest.raises(ValueError, match="multitemporal"):
            SupervisionHead(cfg, embedding_dim=D, max_patch_size=MAX_PATCH_SIZE)

    def test_requires_regression(self) -> None:
        """time_conditioned classification is rejected at config time."""
        with pytest.raises(ValueError, match="regression"):
            SupervisionModalityConfig(
                task_type=SupervisionTaskType.CLASSIFICATION,
                num_output_channels=2,
                class_values=[0.0, 1.0],
                time_conditioned=True,
            )

    def test_requires_positive_harmonics(self) -> None:
        """time_harmonics < 1 is rejected at config time."""
        with pytest.raises(ValueError, match="time_harmonics"):
            SupervisionModalityConfig(
                task_type=SupervisionTaskType.REGRESSION,
                num_output_channels=1,
                time_conditioned=True,
                time_harmonics=0,
            )


def _s2_masked_recon_config(
    num_bands: int = 12, masked_only: bool = True
) -> dict[str, SupervisionModalityConfig]:
    """The maskedrecon head: pixrecon scored only on encoder-masked timesteps."""
    return {
        "sentinel2_l2a": SupervisionModalityConfig(
            task_type=SupervisionTaskType.REGRESSION,
            num_output_channels=num_bands,
            weight=0.1,
            regression_loss_type="mse",
            time_conditioned=True,
            time_harmonics=4,
            time_mlp_hidden_dim=16,
            masked_timesteps_only=masked_only,
        ),
    }


class TestMaskedTimestepsOnlySupervision:
    """``masked_timesteps_only``: reconstruction scored on non-ONLINE units only."""

    T = 3
    NUM_BANDS = 12

    def _make_timestamps(self) -> torch.Tensor:
        return torch.tensor(
            [[[1, 0, 2023], [15, 3, 2023], [1, 6, 2023]]], dtype=torch.long
        ).expand(B, -1, -1)

    def _make_head(self, masked_only: bool = True) -> SupervisionHead:
        return SupervisionHead(
            _s2_masked_recon_config(self.NUM_BANDS, masked_only),
            embedding_dim=D,
            max_patch_size=MAX_PATCH_SIZE,
        )

    def _make_batch(self, mask: torch.Tensor) -> MaskedOlmoEarthSample:
        torch.manual_seed(0)
        return MaskedOlmoEarthSample(
            timestamps=self._make_timestamps(),
            sentinel2_l2a=torch.randn(B, P_H, P_W, self.T, self.NUM_BANDS),
            sentinel2_l2a_mask=mask,
        )

    def _mixed_mask(self) -> torch.Tensor:
        """Timestep 0 ONLINE everywhere, timestep 1 DECODER, timestep 2 mixed by row."""
        mask = torch.full(
            (B, P_H, P_W, self.T, self.NUM_BANDS), MaskValue.ONLINE_ENCODER.value
        )
        mask[:, :, :, 1] = MaskValue.DECODER.value
        mask[:, : P_H // 2, :, 2] = MaskValue.TARGET_ENCODER_ONLY.value
        return mask

    def test_loss_ignores_online_timesteps(self) -> None:
        """Perturbing the targets at ONLINE units leaves the loss unchanged.

        Perturbing a DECODER unit changes it, so the mask is doing the selecting and
        not silently dropping everything.
        """
        head = self._make_head()
        register_grid = torch.randn(B, P_H, P_W, D)
        mask = self._mixed_mask()
        batch = self._make_batch(mask)
        preds = head(register_grid, batch)
        loss, _ = compute_supervision_loss(preds, batch, head)
        assert torch.isfinite(loss) and loss > 0

        online = mask[..., 0] == MaskValue.ONLINE_ENCODER.value  # [B, H, W, T]
        assert online.any() and not online.all()
        target: torch.Tensor = batch.sentinel2_l2a
        assert target is not None
        perturbed_online = target.clone()
        perturbed_online[online] += 100.0
        batch_online = MaskedOlmoEarthSample(
            timestamps=batch.timestamps,
            sentinel2_l2a=perturbed_online,
            sentinel2_l2a_mask=mask,
        )
        loss_online, _ = compute_supervision_loss(preds, batch_online, head)
        torch.testing.assert_close(loss_online, loss)

        perturbed_masked = target.clone()
        perturbed_masked[~online] += 100.0
        batch_masked = MaskedOlmoEarthSample(
            timestamps=batch.timestamps,
            sentinel2_l2a=perturbed_masked,
            sentinel2_l2a_mask=mask,
        )
        loss_masked, _ = compute_supervision_loss(preds, batch_masked, head)
        assert not torch.allclose(loss_masked, loss)

    def test_equals_unmasked_loss_when_nothing_is_online(self) -> None:
        """With every unit DECODER the masked and plain losses coincide."""
        torch.manual_seed(1)
        register_grid = torch.randn(B, P_H, P_W, D)
        mask = torch.full(
            (B, P_H, P_W, self.T, self.NUM_BANDS), MaskValue.DECODER.value
        )
        batch = self._make_batch(mask)
        masked_head = self._make_head(masked_only=True)
        plain_head = self._make_head(masked_only=False)
        plain_head.load_state_dict(masked_head.state_dict())
        loss_masked, _ = compute_supervision_loss(
            masked_head(register_grid, batch), batch, masked_head
        )
        loss_plain, _ = compute_supervision_loss(
            plain_head(register_grid, batch), batch, plain_head
        )
        torch.testing.assert_close(loss_masked, loss_plain)

    def test_all_online_gives_zero_loss_with_gradient_path(self) -> None:
        """Nothing masked -> zero loss (the FSDP-friendly ``0 * pred.sum()`` branch)."""
        head = self._make_head()
        register_grid = torch.randn(B, P_H, P_W, D, requires_grad=True)
        mask = torch.full(
            (B, P_H, P_W, self.T, self.NUM_BANDS), MaskValue.ONLINE_ENCODER.value
        )
        batch = self._make_batch(mask)
        loss, per_mod = compute_supervision_loss(
            head(register_grid, batch), batch, head
        )
        assert loss.item() == 0.0
        assert per_mod["sentinel2_l2a"].item() == 0.0
        loss.backward()
        assert register_grid.grad is not None

    def test_missing_units_stay_excluded(self) -> None:
        """MISSING mask values never count as masked-for-reconstruction targets."""
        head = self._make_head()
        register_grid = torch.randn(B, P_H, P_W, D)
        mask = torch.full(
            (B, P_H, P_W, self.T, self.NUM_BANDS), MaskValue.ONLINE_ENCODER.value
        )
        mask[:, :, :, 1] = MaskValue.MISSING.value
        batch = self._make_batch(mask)
        loss, _ = compute_supervision_loss(head(register_grid, batch), batch, head)
        assert loss.item() == 0.0

    def test_requires_mask_on_batch(self) -> None:
        """A batch without the modality mask raises a clear error."""
        head = self._make_head()
        register_grid = torch.randn(B, P_H, P_W, D)
        batch = MaskedOlmoEarthSample(
            timestamps=self._make_timestamps(),
            sentinel2_l2a=torch.randn(B, P_H, P_W, self.T, self.NUM_BANDS),
        )
        preds = head(register_grid, batch)
        with pytest.raises(ValueError, match="sentinel2_l2a_mask"):
            compute_supervision_loss(preds, batch, head)

    def test_requires_time_conditioned(self) -> None:
        """masked_timesteps_only without time_conditioned is rejected at config time."""
        with pytest.raises(ValueError, match="time_conditioned"):
            SupervisionModalityConfig(
                task_type=SupervisionTaskType.REGRESSION,
                num_output_channels=1,
                masked_timesteps_only=True,
            )
