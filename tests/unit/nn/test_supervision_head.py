"""Unit tests for the supervision head module."""

import pytest
import torch

from olmoearth_pretrain.data.constants import MISSING_VALUE
from olmoearth_pretrain.datatypes import MaskedOlmoEarthSample, MaskValue
from olmoearth_pretrain.nn.flexi_vit import TokensAndMasks
from olmoearth_pretrain.nn.supervision_head import (
    SupervisionHead,
    SupervisionHeadConfig,
    SupervisionModalityConfig,
    SupervisionTaskType,
    _build_valid_mask,
    _masked_mean_over_t_bs,
    _pool_decoder_features,
    compute_supervision_loss,
)

B, P_H, P_W, T, BS, D = 2, 4, 4, 3, 2, 8
H_PIX, W_PIX = 32, 32


def _make_decoder_output(
    mask_value: int = MaskValue.DECODER.value,
) -> TokensAndMasks:
    """Decoder output with sentinel2_l2a tokens."""
    return TokensAndMasks(
        sentinel2_l2a=torch.randn(B, P_H, P_W, T, BS, D),
        sentinel2_l2a_mask=torch.full((B, P_H, P_W, T, BS), mask_value),
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


class TestMaskedMeanOverTBS:
    """Test _masked_mean_over_t_bs helper."""

    def test_shape(self) -> None:
        """Output has T and BandSets collapsed."""
        tokens = torch.randn(B, P_H, P_W, T, BS, D)
        mask = torch.ones(B, P_H, P_W, T, BS, dtype=torch.bool)
        result = _masked_mean_over_t_bs(tokens, mask)
        assert result.shape == (B, P_H, P_W, D)

    def test_all_masked_returns_zero(self) -> None:
        """All-False mask yields zero output."""
        tokens = torch.randn(B, P_H, P_W, T, BS, D)
        mask = torch.zeros(B, P_H, P_W, T, BS, dtype=torch.bool)
        result = _masked_mean_over_t_bs(tokens, mask)
        assert result.shape == (B, P_H, P_W, D)
        assert (result == 0).all()

    def test_partial_mask(self) -> None:
        """Partial mask averages only over unmasked slots."""
        tokens = torch.ones(B, P_H, P_W, T, BS, D) * 2.0
        mask = torch.zeros(B, P_H, P_W, T, BS, dtype=torch.bool)
        mask[:, :, :, 0, 0] = True
        result = _masked_mean_over_t_bs(tokens, mask)
        assert torch.allclose(result, torch.full((B, P_H, P_W, D), 2.0))


class TestPoolDecoderFeatures:
    """Test _pool_decoder_features."""

    def test_basic(self) -> None:
        """Single modality pooled at patch resolution."""
        decoded = _make_decoder_output()
        result = _pool_decoder_features(decoded)
        assert result is not None
        assert result.shape == (B, P_H, P_W, D)

    def test_no_valid_tokens_returns_none(self) -> None:
        """All MISSING tokens yield None."""
        decoded = _make_decoder_output(mask_value=MaskValue.MISSING.value)
        result = _pool_decoder_features(decoded)
        assert result is None

    def test_encoder_tokens_included(self) -> None:
        """ONLINE_ENCODER tokens in decoder output are also pooled."""
        decoded = _make_decoder_output(mask_value=MaskValue.ONLINE_ENCODER.value)
        result = _pool_decoder_features(decoded)
        assert result is not None
        assert result.shape == (B, P_H, P_W, D)

    def test_multiple_modalities(self) -> None:
        """Features from multiple modalities are averaged."""
        decoded = TokensAndMasks(
            sentinel2_l2a=torch.randn(B, P_H, P_W, T, BS, D),
            sentinel2_l2a_mask=torch.full(
                (B, P_H, P_W, T, BS), MaskValue.DECODER.value
            ),
            sentinel1=torch.randn(B, P_H, P_W, T, 1, D),
            sentinel1_mask=torch.full((B, P_H, P_W, T, 1), MaskValue.DECODER.value),
        )
        result = _pool_decoder_features(decoded)
        assert result is not None
        assert result.shape == (B, P_H, P_W, D)


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

    def test_forward_shape(
        self, worldcover_config: dict[str, SupervisionModalityConfig]
    ) -> None:
        """Predictions have the correct pixel-resolution shape."""
        head = SupervisionHead(worldcover_config, embedding_dim=D)
        decoded = _make_decoder_output()
        batch = _make_batch_with_worldcover()
        preds = head(decoded, batch)
        assert "worldcover" in preds
        assert preds["worldcover"].shape == (B, H_PIX, W_PIX, 11)

    def test_missing_target_still_produces_output(
        self, worldcover_config: dict[str, SupervisionModalityConfig]
    ) -> None:
        """Heads always run (for FSDP), even when target is absent."""
        head = SupervisionHead(worldcover_config, embedding_dim=D)
        decoded = _make_decoder_output()
        timestamps = torch.tensor([[1, 1, 2023]], dtype=torch.long).expand(B, -1, -1)
        batch = MaskedOlmoEarthSample(timestamps=timestamps)
        preds = head(decoded, batch)
        assert "worldcover" in preds
        assert preds["worldcover"].requires_grad

    def test_regression_head(self) -> None:
        """Regression head produces [B, H, W, 1] output."""
        cfg = {
            "srtm": SupervisionModalityConfig(
                task_type=SupervisionTaskType.REGRESSION,
                num_output_channels=1,
            ),
        }
        head = SupervisionHead(cfg, embedding_dim=D)
        decoded = _make_decoder_output()
        batch = _make_batch_with_srtm()
        preds = head(decoded, batch)
        assert "srtm" in preds
        assert preds["srtm"].shape == (B, H_PIX, W_PIX, 1)


class TestBuildValidMask:
    """Test _build_valid_mask helper."""

    def test_all_valid(self) -> None:
        """No MISSING_VALUE means all True."""
        target = torch.ones(B, H_PIX, W_PIX, 1)
        mask = _build_valid_mask(target)
        assert mask.all()

    def test_some_missing(self) -> None:
        """MISSING_VALUE pixels are False."""
        target = torch.ones(B, H_PIX, W_PIX, 1)
        target[0, 0, 0, 0] = MISSING_VALUE
        mask = _build_valid_mask(target)
        assert not mask[0, 0, 0]
        assert mask[0, 0, 1]


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
        pred = torch.randn(B, H_PIX, W_PIX, 11)
        batch = _make_batch_with_worldcover()
        total_loss, per_mod = compute_supervision_loss({"worldcover": pred}, batch, cfg)
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
        pred = torch.randn(B, H_PIX, W_PIX, 1)
        batch = _make_batch_with_srtm()
        total_loss, per_mod = compute_supervision_loss({"srtm": pred}, batch, cfg)
        assert total_loss.ndim == 0
        assert total_loss > 0
        assert "srtm" in per_mod

    def test_all_missing_returns_zero(self) -> None:
        """Entirely missing target yields zero loss."""
        cfg = {
            "srtm": SupervisionModalityConfig(
                task_type=SupervisionTaskType.REGRESSION,
                num_output_channels=1,
            ),
        }
        timestamps = torch.tensor([[1, 1, 2023]], dtype=torch.long).expand(B, -1, -1)
        srtm = torch.full((B, H_PIX, W_PIX, 1, 1), MISSING_VALUE, dtype=torch.float)
        batch = MaskedOlmoEarthSample(timestamps=timestamps, srtm=srtm)
        pred = torch.randn(B, H_PIX, W_PIX, 1)
        total_loss, per_mod = compute_supervision_loss({"srtm": pred}, batch, cfg)
        assert total_loss == 0.0

    def test_binary_classification_loss(self) -> None:
        """Binary classification loss is positive and finite."""
        cfg = {
            "openstreetmap_raster": SupervisionModalityConfig(
                task_type=SupervisionTaskType.BINARY_CLASSIFICATION,
                num_output_channels=30,
            ),
        }
        pred = torch.randn(B, H_PIX, W_PIX, 30)
        timestamps = torch.tensor([[1, 1, 2023]], dtype=torch.long).expand(B, -1, -1)
        osm = torch.randint(0, 2, (B, H_PIX, W_PIX, 1, 30)).float()
        batch = MaskedOlmoEarthSample(
            timestamps=timestamps,
            openstreetmap_raster=osm,
        )
        total_loss, per_mod = compute_supervision_loss(
            {"openstreetmap_raster": pred}, batch, cfg
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
        head = config.build(embedding_dim=D)
        assert isinstance(head, SupervisionHead)
        assert "worldcover" in head.heads
        assert "srtm" in head.heads

    def test_classification_requires_class_values(self) -> None:
        """Classification without class_values raises ValueError."""
        with pytest.raises(ValueError, match="class_values"):
            SupervisionModalityConfig(
                task_type=SupervisionTaskType.CLASSIFICATION,
                num_output_channels=11,
            )
