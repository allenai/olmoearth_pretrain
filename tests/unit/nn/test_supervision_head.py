"""Unit tests for the supervision head module."""

import pytest
import torch

from olmoearth_pretrain.data.constants import MISSING_VALUE
from olmoearth_pretrain.datatypes import MaskedOlmoEarthSample, MaskValue
from olmoearth_pretrain.nn.flexi_vit import TokensAndMasks
from olmoearth_pretrain.nn.supervision_head import (
    LATLON_TARGET_DIM,
    SupervisionHead,
    SupervisionHeadConfig,
    SupervisionModalityConfig,
    SupervisionTaskType,
    _build_valid_mask,
    _day_of_year_encoding,
    _flatten_time,
    _latlon_regression_loss,
    _latlon_unit_xyz_target,
    _reduce_time_mean,
    compute_supervision_loss,
)

B, P_H, P_W, D = 2, 4, 4, 8
MAX_PATCH_SIZE = 8
H_PIX, W_PIX = P_H * MAX_PATCH_SIZE, P_W * MAX_PATCH_SIZE  # 32, 32


def _make_decoder_output_with_worldcover(
    mask_value: int = MaskValue.DECODER.value,
) -> TokensAndMasks:
    """Decoder output with worldcover tokens (T=1, BS=1)."""
    return TokensAndMasks(
        sentinel2_l2a=torch.randn(B, P_H, P_W, 3, 2, D),
        sentinel2_l2a_mask=torch.full((B, P_H, P_W, 3, 2), mask_value),
        worldcover=torch.randn(B, P_H, P_W, 1, 1, D),
        worldcover_mask=torch.full((B, P_H, P_W, 1, 1), mask_value),
    )


def _make_decoder_output_with_srtm(
    mask_value: int = MaskValue.DECODER.value,
) -> TokensAndMasks:
    """Decoder output with srtm tokens (T=1, BS=1)."""
    return TokensAndMasks(
        sentinel2_l2a=torch.randn(B, P_H, P_W, 3, 2, D),
        sentinel2_l2a_mask=torch.full((B, P_H, P_W, 3, 2), mask_value),
        srtm=torch.randn(B, P_H, P_W, 1, 1, D),
        srtm_mask=torch.full((B, P_H, P_W, 1, 1), mask_value),
    )


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
    """Test SupervisionHead forward pass."""

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

    def test_forward_shape_t1(
        self, worldcover_config: dict[str, SupervisionModalityConfig]
    ) -> None:
        """Non-multitemporal: output is [B, H, W, T=1, C]."""
        head = SupervisionHead(
            worldcover_config, embedding_dim=D, max_patch_size=MAX_PATCH_SIZE
        )
        decoded = _make_decoder_output_with_worldcover()
        batch = _make_batch_with_worldcover()
        preds = head(decoded, batch)
        assert "worldcover" in preds
        assert preds["worldcover"].shape == (B, H_PIX, W_PIX, 1, 11)

    def test_forward_shape_multitemporal(self) -> None:
        """Multitemporal modality (e.g. NDVI) preserves T > 1."""
        T = 3
        cfg = {
            "ndvi": SupervisionModalityConfig(
                task_type=SupervisionTaskType.REGRESSION,
                num_output_channels=1,
            ),
        }
        head = SupervisionHead(cfg, embedding_dim=D, max_patch_size=MAX_PATCH_SIZE)
        decoded = TokensAndMasks(
            ndvi=torch.randn(B, P_H, P_W, T, 1, D),
            ndvi_mask=torch.full((B, P_H, P_W, T, 1), MaskValue.DECODER.value),
        )
        ndvi_target = torch.rand(B, H_PIX, W_PIX, T, 1)
        timestamps = torch.tensor([[1, 1, 2023]], dtype=torch.long).expand(B, -1, -1)
        batch = MaskedOlmoEarthSample(timestamps=timestamps, ndvi=ndvi_target)
        preds = head(decoded, batch)
        assert preds["ndvi"].shape == (B, H_PIX, W_PIX, T, 1)

    def test_forward_uses_per_modality_tokens(
        self, worldcover_config: dict[str, SupervisionModalityConfig]
    ) -> None:
        """Each head uses its own modality tokens, not a cross-modality pool."""
        head = SupervisionHead(
            worldcover_config, embedding_dim=D, max_patch_size=MAX_PATCH_SIZE
        )
        wc_tokens = torch.randn(B, P_H, P_W, 1, 1, D)
        decoded_a = TokensAndMasks(
            worldcover=wc_tokens,
            worldcover_mask=torch.full((B, P_H, P_W, 1, 1), MaskValue.DECODER.value),
        )
        decoded_b = TokensAndMasks(
            worldcover=wc_tokens,
            worldcover_mask=torch.full((B, P_H, P_W, 1, 1), MaskValue.DECODER.value),
            sentinel2_l2a=torch.randn(B, P_H, P_W, 3, 2, D),
            sentinel2_l2a_mask=torch.full((B, P_H, P_W, 3, 2), MaskValue.DECODER.value),
        )
        batch = _make_batch_with_worldcover()
        preds_a = head(decoded_a, batch)
        preds_b = head(decoded_b, batch)
        torch.testing.assert_close(preds_a["worldcover"], preds_b["worldcover"])

    def test_forward_downsample(
        self, worldcover_config: dict[str, SupervisionModalityConfig]
    ) -> None:
        """When prediction > target, output is downsampled to target size."""
        small_h, small_w = 16, 16
        head = SupervisionHead(
            worldcover_config, embedding_dim=D, max_patch_size=MAX_PATCH_SIZE
        )
        decoded = _make_decoder_output_with_worldcover()
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
        preds = head(decoded, batch)
        assert preds["worldcover"].shape == (B, small_h, small_w, 1, 11)

    def test_missing_modality_tokens_still_produces_output(
        self, worldcover_config: dict[str, SupervisionModalityConfig]
    ) -> None:
        """Heads run even when the modality is absent from decoder output (FSDP)."""
        head = SupervisionHead(
            worldcover_config, embedding_dim=D, max_patch_size=MAX_PATCH_SIZE
        )
        decoded = TokensAndMasks(
            sentinel2_l2a=torch.randn(B, P_H, P_W, 3, 2, D),
            sentinel2_l2a_mask=torch.full((B, P_H, P_W, 3, 2), MaskValue.DECODER.value),
        )
        timestamps = torch.tensor([[1, 1, 2023]], dtype=torch.long).expand(B, -1, -1)
        batch = MaskedOlmoEarthSample(timestamps=timestamps)
        preds = head(decoded, batch)
        assert "worldcover" in preds
        assert preds["worldcover"].requires_grad

    def test_regression_head(self) -> None:
        """Regression head produces [B, H, W, T=1, 1] output."""
        cfg = {
            "srtm": SupervisionModalityConfig(
                task_type=SupervisionTaskType.REGRESSION,
                num_output_channels=1,
            ),
        }
        head = SupervisionHead(cfg, embedding_dim=D, max_patch_size=MAX_PATCH_SIZE)
        decoded = _make_decoder_output_with_srtm()
        batch = _make_batch_with_srtm()
        preds = head(decoded, batch)
        assert "srtm" in preds
        assert preds["srtm"].shape == (B, H_PIX, W_PIX, 1, 1)

    def test_non_spatial_forward(self) -> None:
        """Non-spatial modality (latlon) produces [B, C] output."""
        cfg = {
            "latlon": SupervisionModalityConfig(
                task_type=SupervisionTaskType.REGRESSION,
                num_output_channels=LATLON_TARGET_DIM,
            ),
        }
        head = SupervisionHead(cfg, embedding_dim=D, max_patch_size=MAX_PATCH_SIZE)
        decoded = TokensAndMasks(
            latlon=torch.randn(B, 1, D),
            latlon_mask=torch.full((B, 1), MaskValue.DECODER.value),
        )
        timestamps = torch.tensor([[1, 1, 2023]], dtype=torch.long).expand(B, -1, -1)
        batch = MaskedOlmoEarthSample(timestamps=timestamps, latlon=torch.rand(B, 2))
        preds = head(decoded, batch)
        assert "latlon" in preds
        assert preds["latlon"].shape == (B, LATLON_TARGET_DIM)

    def test_non_spatial_missing_tokens(self) -> None:
        """Non-spatial head runs with dummy zeros when tokens are absent (FSDP)."""
        cfg = {
            "latlon": SupervisionModalityConfig(
                task_type=SupervisionTaskType.REGRESSION,
                num_output_channels=LATLON_TARGET_DIM,
            ),
        }
        head = SupervisionHead(cfg, embedding_dim=D, max_patch_size=MAX_PATCH_SIZE)
        decoded = TokensAndMasks(
            sentinel2_l2a=torch.randn(B, P_H, P_W, 3, 2, D),
            sentinel2_l2a_mask=torch.full((B, P_H, P_W, 3, 2), MaskValue.DECODER.value),
        )
        timestamps = torch.tensor([[1, 1, 2023]], dtype=torch.long).expand(B, -1, -1)
        batch = MaskedOlmoEarthSample(timestamps=timestamps)
        preds = head(decoded, batch)
        assert "latlon" in preds
        assert preds["latlon"].requires_grad


class TestLatlonSupervision:
    """Test the latlon unit-sphere xyz target conversion and loss."""

    def _normalize(self, lat: float, lon: float) -> list[float]:
        """Apply the predefined latlon normalization (degrees -> [0, 1])."""
        return [(lat + 90.0) / 180.0, (lon + 180.0) / 360.0]

    def test_unit_xyz_target_matches_trig(self) -> None:
        """Normalized (lat, lon) converts to the expected unit-sphere point."""
        import math

        cases = [(0.0, 0.0), (90.0, 0.0), (-45.0, 0.0), (47.6, -122.3)]
        raw = torch.tensor([self._normalize(lat, lon) for lat, lon in cases])
        xyz = _latlon_unit_xyz_target(raw)
        for (lat, lon), got in zip(cases, xyz):
            la, lo = math.radians(lat), math.radians(lon)
            expected = torch.tensor(
                [
                    math.cos(la) * math.cos(lo),
                    math.cos(la) * math.sin(lo),
                    math.sin(la),
                ]
            )
            assert torch.allclose(got, expected, atol=1e-5)
        assert torch.allclose(xyz.norm(dim=-1), torch.ones(len(cases)), atol=1e-5)

    def test_unit_xyz_target_no_dateline_discontinuity(self) -> None:
        """Lon = +180 and lon = -180 map to the same point on the sphere."""
        raw = torch.tensor(
            [self._normalize(10.0, 180.0), self._normalize(10.0, -180.0)]
        )
        xyz = _latlon_unit_xyz_target(raw)
        assert torch.allclose(xyz[0], xyz[1], atol=1e-5)

    def test_loss_zero_for_perfect_prediction(self) -> None:
        """Predicting the exact xyz target gives zero loss."""
        raw = torch.rand(B, 2)
        pred = _latlon_unit_xyz_target(raw)
        assert _latlon_regression_loss(pred, raw).item() == pytest.approx(0.0)

    def test_loss_excludes_missing_rows(self) -> None:
        """Missing-valued latlon rows do not contribute to the loss."""
        raw = torch.rand(B, 2)
        pred = _latlon_unit_xyz_target(raw)
        raw[0] = MISSING_VALUE
        pred[0] = 99.0  # garbage on the missing row must not matter
        assert _latlon_regression_loss(pred, raw).item() == pytest.approx(0.0)

    def test_loss_all_missing_keeps_grad_path(self) -> None:
        """All-missing latlon yields zero loss that still touches the prediction."""
        pred = torch.randn(B, LATLON_TARGET_DIM, requires_grad=True)
        raw = torch.full((B, 2), float(MISSING_VALUE))
        loss = _latlon_regression_loss(pred, raw)
        assert loss.item() == 0.0
        assert loss.requires_grad

    def test_head_rejects_wrong_latlon_config(self) -> None:
        """Latlon must be a 3-channel regression head."""
        for bad in (
            SupervisionModalityConfig(
                task_type=SupervisionTaskType.REGRESSION, num_output_channels=2
            ),
            SupervisionModalityConfig(
                task_type=SupervisionTaskType.CLASSIFICATION,
                num_output_channels=LATLON_TARGET_DIM,
                class_values=[0.0, 1.0],
            ),
        ):
            with pytest.raises(ValueError, match="unit-sphere"):
                SupervisionHead(
                    {"latlon": bad}, embedding_dim=D, max_patch_size=MAX_PATCH_SIZE
                )


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

    def test_non_spatial_regression_loss(self) -> None:
        """Non-spatial regression loss (latlon) is positive and finite."""
        cfg = {
            "latlon": SupervisionModalityConfig(
                task_type=SupervisionTaskType.REGRESSION,
                num_output_channels=LATLON_TARGET_DIM,
                weight=0.3,
            ),
        }
        head = SupervisionHead(cfg, embedding_dim=D, max_patch_size=MAX_PATCH_SIZE)
        pred = torch.randn(B, LATLON_TARGET_DIM)
        timestamps = torch.tensor([[1, 1, 2023]], dtype=torch.long).expand(B, -1, -1)
        batch = MaskedOlmoEarthSample(timestamps=timestamps, latlon=torch.rand(B, 2))
        total_loss, per_mod = compute_supervision_loss({"latlon": pred}, batch, head)
        assert total_loss.ndim == 0
        assert total_loss > 0
        assert "latlon" in per_mod

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
                    num_output_channels=LATLON_TARGET_DIM,
                ),
            }
        )
        head = config.build(embedding_dim=D, max_patch_size=MAX_PATCH_SIZE)
        assert head.heads["latlon"].out_features == LATLON_TARGET_DIM
        assert "latlon" in head._non_spatial_modalities

    def test_classification_requires_class_values(self) -> None:
        """Classification without class_values raises ValueError."""
        with pytest.raises(ValueError, match="class_values"):
            SupervisionModalityConfig(
                task_type=SupervisionTaskType.CLASSIFICATION,
                num_output_channels=11,
            )


def _ndvi_time_conditioned_config() -> dict[str, SupervisionModalityConfig]:
    """NDVI time-conditioned register-supervision config."""
    return {
        "ndvi": SupervisionModalityConfig(
            task_type=SupervisionTaskType.REGRESSION,
            num_output_channels=1,
            weight=1.0,
            time_conditioned=True,
            time_harmonics=4,
        )
    }


class TestTimeConditionedSupervision:
    """Time-conditioned (register grid x day-of-year MLP) supervision, e.g. NDVI."""

    T = 3

    def _make_timestamps(self) -> torch.Tensor:
        # (day, month0, year): Jan 1, Apr 15, Jul 1 of 2023.
        return torch.tensor(
            [[[1, 0, 2023], [15, 3, 2023], [1, 6, 2023]]], dtype=torch.long
        ).expand(B, -1, -1)

    def _make_head(self) -> SupervisionHead:
        return SupervisionHead(
            _ndvi_time_conditioned_config(),
            embedding_dim=D,
            max_patch_size=MAX_PATCH_SIZE,
            register_supervision=True,
        )

    def test_forward_shape_time_dependence_and_locality(self) -> None:
        """Per-(cell, timestep) predictions from the time-free register grid.

        With a grid-resolution target (no interpolation): predictions vary across
        timesteps (the time conditioning is live), and cell (i, j)'s prediction
        depends ONLY on register_grid[:, i, j] (the per-cell forcing that makes the
        fitted trajectory readable by a frozen per-cell probe).
        """
        head = self._make_head()
        register_grid = torch.randn(B, P_H, P_W, D)
        ndvi_target = torch.rand(B, P_H, P_W, self.T, 1)
        batch = MaskedOlmoEarthSample(
            timestamps=self._make_timestamps(), ndvi=ndvi_target
        )
        preds = head(TokensAndMasks(), batch, register_grid=register_grid)
        assert preds["ndvi"].shape == (B, P_H, P_W, self.T, 1)
        # Same cell, different timesteps -> different predictions.
        assert not torch.allclose(preds["ndvi"][:, :, :, 0], preds["ndvi"][:, :, :, 1])
        # Perturbing one cell leaves every other cell's predictions unchanged.
        perturbed = register_grid.clone()
        perturbed[:, 0, 0] += 1.0
        preds_perturbed = head(TokensAndMasks(), batch, register_grid=perturbed)
        assert not torch.allclose(
            preds_perturbed["ndvi"][:, 0, 0], preds["ndvi"][:, 0, 0]
        )
        torch.testing.assert_close(preds_perturbed["ndvi"][:, 1:], preds["ndvi"][:, 1:])

    def test_forward_interpolates_to_pixel_target(self) -> None:
        """A pixel-resolution target triggers bilinear upsampling of the grid preds."""
        head = self._make_head()
        register_grid = torch.randn(B, P_H, P_W, D)
        ndvi_target = torch.rand(B, H_PIX, W_PIX, self.T, 1)
        batch = MaskedOlmoEarthSample(
            timestamps=self._make_timestamps(), ndvi=ndvi_target
        )
        preds = head(TokensAndMasks(), batch, register_grid=register_grid)
        assert preds["ndvi"].shape == (B, H_PIX, W_PIX, self.T, 1)

    def test_loss_and_register_gradients(self) -> None:
        """The supervision loss backpropagates into the register grid."""
        head = self._make_head()
        register_grid = torch.randn(B, P_H, P_W, D, requires_grad=True)
        ndvi_target = torch.rand(B, H_PIX, W_PIX, self.T, 1)
        # Punch some MISSING holes (cloud/absent obs); the masked loss skips them.
        ndvi_target[:, :4, :4, 0] = MISSING_VALUE
        batch = MaskedOlmoEarthSample(
            timestamps=self._make_timestamps(), ndvi=ndvi_target
        )
        preds = head(TokensAndMasks(), batch, register_grid=register_grid)
        total_loss, per_mod = compute_supervision_loss(preds, batch, head)
        assert total_loss.ndim == 0
        assert torch.isfinite(total_loss)
        total_loss.backward()
        assert register_grid.grad is not None
        assert torch.isfinite(register_grid.grad).all()
        assert register_grid.grad.abs().sum() > 0

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

    def test_requires_register_supervision(self) -> None:
        """time_conditioned without register_supervision raises."""
        with pytest.raises(ValueError, match="register_supervision"):
            SupervisionHead(
                _ndvi_time_conditioned_config(),
                embedding_dim=D,
                max_patch_size=MAX_PATCH_SIZE,
                register_supervision=False,
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
            SupervisionHead(
                cfg,
                embedding_dim=D,
                max_patch_size=MAX_PATCH_SIZE,
                register_supervision=True,
            )

    def test_requires_regression(self) -> None:
        """time_conditioned classification is rejected at config time."""
        with pytest.raises(ValueError, match="regression"):
            SupervisionModalityConfig(
                task_type=SupervisionTaskType.CLASSIFICATION,
                num_output_channels=2,
                class_values=[0.0, 1.0],
                time_conditioned=True,
            )


# era5_10 is the non-spatial multitemporal target these paths exist for:
# a per-scene [B, T=12, C=6] weather trajectory pooled to a single [B, C'] vector.
ERA5_T, ERA5_C = 12, 6
ERA5_SIGNATURE_DIM = ERA5_T * ERA5_C  # 72 (12 months x 6 vars)


class TestTemporalReduction:
    """temporal_reduction for non-spatial multitemporal regression (era5_10)."""

    def test_flatten_shape_and_time_major_order(self) -> None:
        """Flatten maps [B, T, C] -> [B, T*C], bands contiguous within each month."""
        tgt = torch.arange(B * ERA5_T * ERA5_C, dtype=torch.float32).reshape(
            B, ERA5_T, ERA5_C
        )
        flat = _flatten_time(tgt)
        assert flat.shape == (B, ERA5_SIGNATURE_DIM)
        # First C entries are month 0's bands, next C are month 1's bands (time-major).
        torch.testing.assert_close(flat[0, :ERA5_C], tgt[0, 0])
        torch.testing.assert_close(flat[0, ERA5_C : 2 * ERA5_C], tgt[0, 1])

    def test_flatten_passthrough_non_3d(self) -> None:
        """A target that is not [B, T, C] is returned unchanged."""
        already_flat = torch.randn(B, ERA5_SIGNATURE_DIM)
        assert _flatten_time(already_flat) is already_flat

    def test_flatten_missing_month_drops_sample(self) -> None:
        """A sample with any fully-missing month is dropped by _build_valid_mask."""
        tgt = torch.randn(B, ERA5_T, ERA5_C)
        tgt[1, 3, :] = MISSING_VALUE  # sample 1, month 3 missing
        valid = _build_valid_mask(_flatten_time(tgt))  # mirrors the loss path
        assert valid.shape == (B,)
        assert valid[0] and not valid[1]

    def test_mean_shape_and_missing_awareness(self) -> None:
        """Mean maps [B, T, C] -> [B, C], averaging only over valid timesteps."""
        tgt = torch.ones(B, ERA5_T, ERA5_C)
        tgt[0, :6] = 3.0  # 6 months at 3, 6 months at 1 -> mean 2 for sample 0
        tgt[0, 6:] = 1.0
        tgt[1, 2, :] = MISSING_VALUE  # one missing month for sample 1
        reduced = _reduce_time_mean(tgt)
        assert reduced.shape == (B, ERA5_C)
        torch.testing.assert_close(reduced[0], torch.full((ERA5_C,), 2.0))
        # Sample 1: missing month excluded, remaining 11 months are all 1.0.
        torch.testing.assert_close(reduced[1], torch.ones(ERA5_C))

    def test_mean_all_missing_sample_is_dropped(self) -> None:
        """A sample with no valid timestep collapses to MISSING (dropped downstream)."""
        tgt = torch.ones(B, ERA5_T, ERA5_C)
        tgt[0] = MISSING_VALUE
        reduced = _reduce_time_mean(tgt)
        valid = _build_valid_mask(reduced)
        assert not valid[0] and valid[1]

    def test_config_accepts_mean_and_flatten(self) -> None:
        """Both reduction modes validate for regression."""
        for mode, n_out in (("mean", ERA5_C), ("flatten", ERA5_SIGNATURE_DIM)):
            cfg = SupervisionModalityConfig(
                task_type=SupervisionTaskType.REGRESSION,
                num_output_channels=n_out,
                temporal_reduction=mode,
            )
            assert cfg.temporal_reduction == mode

    def test_config_rejects_bad_reduction(self) -> None:
        """An unknown temporal_reduction value is rejected."""
        with pytest.raises(ValueError, match="temporal_reduction"):
            SupervisionModalityConfig(
                task_type=SupervisionTaskType.REGRESSION,
                num_output_channels=ERA5_C,
                temporal_reduction="sum",
            )

    def test_config_reduction_requires_regression(self) -> None:
        """temporal_reduction on a classification task is rejected."""
        with pytest.raises(ValueError, match="regression"):
            SupervisionModalityConfig(
                task_type=SupervisionTaskType.CLASSIFICATION,
                num_output_channels=2,
                class_values=[0.0, 1.0],
                temporal_reduction="mean",
            )

    def test_config_reduction_mutually_exclusive_with_time_conditioned(self) -> None:
        """Collapsing time and predicting per-timestep cannot both be set."""
        with pytest.raises(ValueError, match="mutually exclusive"):
            SupervisionModalityConfig(
                task_type=SupervisionTaskType.REGRESSION,
                num_output_channels=ERA5_C,
                temporal_reduction="mean",
                time_conditioned=True,
            )

    def test_non_spatial_head_output_width_matches_signature(self) -> None:
        """The pooled era5_10 head emits num_output_channels (72 for flatten)."""
        head = SupervisionHead(
            {
                "era5_10": SupervisionModalityConfig(
                    task_type=SupervisionTaskType.REGRESSION,
                    num_output_channels=ERA5_SIGNATURE_DIM,
                    temporal_reduction="flatten",
                )
            },
            embedding_dim=D,
            max_patch_size=MAX_PATCH_SIZE,
            register_supervision=True,
        )
        assert "era5_10" in head._non_spatial_modalities
        assert head.heads["era5_10"].out_features == ERA5_SIGNATURE_DIM

    def _run_pooled_era5(
        self, temporal_reduction: str, num_output_channels: int
    ) -> None:
        """End-to-end: pooled register grid -> [B, C'] era5 pred, loss backprops."""
        head = SupervisionHead(
            {
                "era5_10": SupervisionModalityConfig(
                    task_type=SupervisionTaskType.REGRESSION,
                    num_output_channels=num_output_channels,
                    weight=0.1,
                    regression_loss_type="l1",
                    temporal_reduction=temporal_reduction,
                )
            },
            embedding_dim=D,
            max_patch_size=MAX_PATCH_SIZE,
            register_supervision=True,
        )
        register_grid = torch.randn(B, P_H, P_W, D, requires_grad=True)
        era5_target = torch.rand(B, ERA5_T, ERA5_C)
        timestamps = torch.tensor([[1, 1, 2023]], dtype=torch.long).expand(B, -1, -1)
        batch = MaskedOlmoEarthSample(timestamps=timestamps, era5_10=era5_target)

        preds = head(TokensAndMasks(), batch, register_grid=register_grid)
        assert preds["era5_10"].shape == (B, num_output_channels)

        total_loss, per_mod = compute_supervision_loss(preds, batch, head)
        assert total_loss.ndim == 0
        assert torch.isfinite(total_loss) and total_loss > 0
        assert "era5_10" in per_mod
        total_loss.backward()
        assert register_grid.grad is not None
        assert torch.isfinite(register_grid.grad).all()
        assert register_grid.grad.abs().sum() > 0

    def test_end_to_end_flatten(self) -> None:
        """The climate arm's path: predict the full 72-dim monthly signature."""
        self._run_pooled_era5("flatten", ERA5_SIGNATURE_DIM)

    def test_end_to_end_mean(self) -> None:
        """The alternative: predict the 6-dim annual level."""
        self._run_pooled_era5("mean", ERA5_C)

    def test_end_to_end_flatten_drops_missing_month_sample(self) -> None:
        """A sample with a missing month is excluded but the batch loss still flows."""
        head = SupervisionHead(
            {
                "era5_10": SupervisionModalityConfig(
                    task_type=SupervisionTaskType.REGRESSION,
                    num_output_channels=ERA5_SIGNATURE_DIM,
                    regression_loss_type="l1",
                    temporal_reduction="flatten",
                )
            },
            embedding_dim=D,
            max_patch_size=MAX_PATCH_SIZE,
            register_supervision=True,
        )
        register_grid = torch.randn(B, P_H, P_W, D, requires_grad=True)
        era5_target = torch.rand(B, ERA5_T, ERA5_C)
        era5_target[0, 5, :] = MISSING_VALUE  # sample 0 has a missing month
        timestamps = torch.tensor([[1, 1, 2023]], dtype=torch.long).expand(B, -1, -1)
        batch = MaskedOlmoEarthSample(timestamps=timestamps, era5_10=era5_target)
        preds = head(TokensAndMasks(), batch, register_grid=register_grid)
        total_loss, _ = compute_supervision_loss(preds, batch, head)
        assert torch.isfinite(total_loss) and total_loss > 0
        total_loss.backward()
        assert register_grid.grad.abs().sum() > 0
