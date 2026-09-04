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
    _latlon_regression_loss,
    _latlon_unit_xyz_target,
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
