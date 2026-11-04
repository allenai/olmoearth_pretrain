"""Unit tests for the supervision head module."""

from typing import Any

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


def _make_batch_with_naip(missing_block: bool = False) -> MaskedOlmoEarthSample:
    """Batch with naip_10 [B, 4H, 4W, 1, 4] on the 4x finer grid."""
    naip = torch.randn(B, 4 * H_PIX, 4 * W_PIX, 1, 4)
    if missing_block:
        naip[0, :4, :4, 0, 0] = MISSING_VALUE
    timestamps = torch.tensor([[1, 1, 2023]], dtype=torch.long).expand(B, -1, -1)
    return MaskedOlmoEarthSample(
        timestamps=timestamps,
        naip_10=naip,
        naip_10_mask=torch.full(naip.shape[:-1] + (1,), MaskValue.DECODER.value),
    )


class TestHighResTargets:
    """laplacian_loss and the upsample decoder for finer-grid targets."""

    def _head(self, **kwargs: Any) -> SupervisionHead:
        cfg = SupervisionModalityConfig(
            task_type=SupervisionTaskType.REGRESSION,
            num_output_channels=4,
            regression_loss_type="l1",
            **kwargs,
        )
        return SupervisionHeadConfig(modality_configs={"naip_10": cfg}).build(
            embedding_dim=D, max_patch_size=MAX_PATCH_SIZE
        )

    @pytest.mark.parametrize("decoder", ["linear", "upsample"])
    def test_output_shape_matches_target(self, decoder: str) -> None:
        """Both decoders predict the full 4x-resolution target grid."""
        head = self._head(decoder=decoder)
        batch = _make_batch_with_naip()
        pred = head(_make_register_grid(), batch)["naip_10"]
        assert batch.naip_10 is not None
        assert pred.shape == batch.naip_10.shape

    def test_upsample_gradients_reach_register_grid(self) -> None:
        """The upsample decoder backpropagates into the register grid."""
        head = self._head(decoder="upsample", laplacian_loss=True)
        batch = _make_batch_with_naip()
        grid = _make_register_grid()
        total, _ = compute_supervision_loss(head(grid, batch), batch, head)
        total.backward()
        assert grid.grad is not None and grid.grad.abs().sum() > 0

    def test_laplacian_loss_matches_manual(self) -> None:
        """MSE on 4x4 block means plus L1 on the within-block residual."""
        head = self._head(laplacian_loss=True)
        batch = _make_batch_with_naip()
        pred = torch.randn_like(batch.naip_10)
        _, losses = compute_supervision_loss({"naip_10": pred}, batch, head)

        def split(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            blocks = x.unfold(1, 4, 4).unfold(2, 4, 4)  # [B, H, W, T, C, 4, 4]
            low = blocks.mean(dim=(-2, -1), keepdim=True)
            return low, blocks - low

        low_p, high_p = split(pred)
        low_t, high_t = split(batch.naip_10)
        expected = ((low_p - low_t) ** 2).mean() + (high_p - high_t).abs().mean()
        assert torch.allclose(losses["naip_10"], expected, atol=1e-5)

    def test_laplacian_loss_is_zero_for_exact_prediction_and_skips_missing(
        self,
    ) -> None:
        """A perfect prediction scores 0 even with a missing block in the target."""
        head = self._head(laplacian_loss=True)
        batch = _make_batch_with_naip(missing_block=True)
        pred = torch.where(
            batch.naip_10 == MISSING_VALUE,
            torch.zeros_like(batch.naip_10),
            batch.naip_10,
        )
        # Other bands of the missing block are wrong, but the whole block is excluded.
        pred[0, :4, :4, 0, 1:] = 1e6
        _, losses = compute_supervision_loss({"naip_10": pred}, batch, head)
        assert losses["naip_10"].item() == pytest.approx(0.0, abs=1e-6)

    def test_laplacian_loss_rejects_base_grid_modality(self) -> None:
        """laplacian_loss on a 10 m modality is a config error."""
        cfg = SupervisionModalityConfig(
            task_type=SupervisionTaskType.REGRESSION,
            num_output_channels=1,
            laplacian_loss=True,
        )
        with pytest.raises(ValueError, match="finer-grid"):
            SupervisionHeadConfig(modality_configs={"srtm": cfg}).build(
                embedding_dim=D, max_patch_size=MAX_PATCH_SIZE
            )


def _viirs_head(mode: str = "log1p", loss_type: str = "l1") -> SupervisionHead:
    cfg = SupervisionModalityConfig(
        task_type=SupervisionTaskType.REGRESSION,
        num_output_channels=1,
        regression_loss_type=loss_type,
        footprint_loss=mode,
    )
    return SupervisionHeadConfig(modality_configs={"viirs": cfg}).build(
        embedding_dim=D, max_patch_size=MAX_PATCH_SIZE
    )


def _footprint_batch(stored: torch.Tensor) -> MaskedOlmoEarthSample:
    """A VIIRS-like batch: per-sample 2x2 footprints of 16x16 px, one stored value each.

    ``stored`` is ``[B, 2, 2]`` in normalized units.
    """
    target = stored.repeat_interleave(16, dim=1).repeat_interleave(16, dim=2)
    target = target[..., None, None]  # [B, 32, 32, 1, 1]
    timestamps = torch.tensor([[1, 1, 2023]], dtype=torch.long).expand(B, -1, -1)
    return MaskedOlmoEarthSample(
        timestamps=timestamps,
        viirs=target,
        viirs_mask=torch.full_like(target, MaskValue.DECODER.value),
    )


class TestFootprintLoss:
    """footprint_loss: score footprint means of a coarse, nearest-resampled target."""

    def _affine(self) -> tuple[float, float]:
        scale, offset = _viirs_head().get_footprint_affine("viirs").tolist()
        return scale, offset

    def _to_norm(self, log1p_value: torch.Tensor) -> torch.Tensor:
        scale, offset = self._affine()
        return (log1p_value - offset) / scale

    def test_computed_affine_inverts_the_dataset_normalizer(self) -> None:
        """The stored [scale, offset] undo Normalizer's computed-stats mapping."""
        import numpy as np

        from olmoearth_pretrain.data.constants import Modality
        from olmoearth_pretrain.data.normalize import Normalizer, Strategy

        x = np.array([0.0, 1.0, 3.5])
        normed = Normalizer(Strategy.COMPUTED).normalize(Modality.VIIRS, x)
        scale, offset = self._affine()
        assert np.allclose(normed * scale + offset, x, atol=1e-5)

    def test_zero_when_radiance_means_match_despite_subpixel_structure(self) -> None:
        """Lights placed anywhere inside a footprint cost nothing if the mean is right."""
        radiance = torch.tensor([[[5.0, 20.0], [8.0, 50.0]], [[6.0, 9.0], [30.0, 7.0]]])
        batch = _footprint_batch(self._to_norm(torch.log1p(radiance)))
        # Inside each footprint: left half at 0.5r, right half at 1.5r (mean r).
        fine = radiance.repeat_interleave(16, dim=1).repeat_interleave(16, dim=2)
        stripes = torch.tensor([0.5] * 8 + [1.5] * 8).repeat(2)  # over W
        pred = self._to_norm(torch.log1p(fine * stripes))[..., None, None]

        _, footprint = compute_supervision_loss({"viirs": pred}, batch, _viirs_head())
        plain_head = SupervisionHeadConfig(
            modality_configs={
                "viirs": SupervisionModalityConfig(
                    task_type=SupervisionTaskType.REGRESSION,
                    num_output_channels=1,
                    regression_loss_type="l1",
                )
            }
        ).build(embedding_dim=D, max_patch_size=MAX_PATCH_SIZE)
        _, plain = compute_supervision_loss({"viirs": pred}, batch, plain_head)
        assert footprint["viirs"].item() == pytest.approx(0.0, abs=1e-4)
        assert plain["viirs"].item() > 0.05

    def test_log1p_mode_averages_radiance_not_its_log(self) -> None:
        """Matching the mean of log1p values is not enough: the sensor averages radiance."""
        log_target = torch.full((B, 2, 2), 2.0)
        batch = _footprint_batch(self._to_norm(log_target))
        # Half the pixels at log1p 1, half at 3: the log mean is 2, the radiance mean
        # (e^1 - 1 + e^3 - 1) / 2 is well above e^2 - 1.
        stripes = torch.tensor([1.0] * 8 + [3.0] * 8).repeat(2)
        pred = self._to_norm(stripes.expand(B, 32, 32))[..., None, None]
        _, losses = compute_supervision_loss({"viirs": pred}, batch, _viirs_head())
        assert losses["viirs"].item() > 0.01

    def test_identity_mode_matches_manual(self) -> None:
        """identity: each pixel is scored by |footprint mean of pred - target|."""
        stored = torch.tensor([[[0.1, 0.2], [0.3, 0.4]], [[0.5, 0.6], [0.7, 0.8]]])
        batch = _footprint_batch(stored)
        pred = torch.randn(B, 32, 32, 1, 1)
        _, losses = compute_supervision_loss(
            {"viirs": pred}, batch, _viirs_head(mode="identity")
        )
        block_means = pred[..., 0, 0].unfold(1, 16, 16).unfold(2, 16, 16).mean((-2, -1))
        expected = (block_means - stored).abs().mean()  # equal-size footprints
        assert losses["viirs"].item() == pytest.approx(expected.item(), abs=1e-5)

    def test_same_value_in_different_samples_is_not_one_footprint(self) -> None:
        """Footprints never span samples, even when their stored values coincide."""
        batch = _footprint_batch(torch.full((B, 2, 2), 0.5))
        pred = torch.full((B, 32, 32, 1, 1), 0.5)
        pred[1] += 0.2  # sample 0 exact, sample 1 off by 0.2 everywhere
        _, losses = compute_supervision_loss(
            {"viirs": pred}, batch, _viirs_head(mode="identity", loss_type="mse")
        )
        # Per sample: (0 + 0.2^2) / 2. Pooled across samples it would be 0.1^2.
        assert losses["viirs"].item() == pytest.approx(0.02, abs=1e-5)

    def test_missing_pixels_are_ignored(self) -> None:
        """Missing target pixels neither join a footprint nor get scored."""
        stored = torch.tensor([[[0.1, 0.2], [0.3, 0.4]], [[0.5, 0.6], [0.7, 0.8]]])
        batch = _footprint_batch(stored)
        assert batch.viirs is not None
        batch.viirs[0, :4, :4] = MISSING_VALUE
        pred = batch.viirs.clone()
        pred[0, :4, :4] = 1e6  # garbage under the missing pixels
        _, losses = compute_supervision_loss(
            {"viirs": pred}, batch, _viirs_head(mode="identity")
        )
        assert losses["viirs"].item() == pytest.approx(0.0, abs=1e-6)

    def test_gradients_reach_register_grid_at_init(self) -> None:
        """End to end through a freshly built head, the loss has a usable gradient."""
        head = _viirs_head()
        batch = _footprint_batch(torch.rand(B, 2, 2))
        grid = _make_register_grid()
        total, _ = compute_supervision_loss(head(grid, batch), batch, head)
        total.backward()
        assert grid.grad is not None and torch.isfinite(grid.grad).all()
        assert grid.grad.abs().sum() > 1e-3

    @pytest.mark.parametrize("wild", [1e4, -1e4])
    def test_wild_predictions_stay_finite(self, wild: float) -> None:
        """Far outside the data range, expm1 neither overflows nor yields NaN."""
        batch = _footprint_batch(torch.rand(B, 2, 2))
        pred = torch.full((B, 32, 32, 1, 1), wild, requires_grad=True)
        total, _ = compute_supervision_loss({"viirs": pred}, batch, _viirs_head())
        total.backward()
        assert torch.isfinite(total)
        assert pred.grad is not None and torch.isfinite(pred.grad).all()

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"footprint_loss": "mean"},
            {"footprint_loss": "log1p", "laplacian_loss": True},
            {"footprint_loss": "log1p", "num_output_channels": 3},
            {"footprint_loss": "log1p", "regression_loss_type": "cosine"},
        ],
    )
    def test_invalid_configs_are_rejected(self, kwargs: dict[str, Any]) -> None:
        """Only plain single-channel regression can use footprint_loss."""
        base: dict[str, Any] = {
            "task_type": SupervisionTaskType.REGRESSION,
            "num_output_channels": 1,
        }
        with pytest.raises(ValueError):
            SupervisionModalityConfig(**{**base, **kwargs})


class TestMlpHead:
    """decoder='mlp': the per-cell head as Linear -> LayerNorm -> ReLU -> Linear."""

    def _head(self, decoder: str) -> SupervisionHead:
        wc = SupervisionModalityConfig(
            task_type=SupervisionTaskType.CLASSIFICATION,
            num_output_channels=11,
            class_values=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0],
            decoder=decoder,
        )
        return SupervisionHeadConfig(modality_configs={"worldcover": wc}).build(
            embedding_dim=D, max_patch_size=MAX_PATCH_SIZE
        )

    def test_same_output_shape_as_linear(self) -> None:
        """Swapping the head depth leaves the prediction layout unchanged."""
        batch = _make_batch_with_worldcover()
        grid = _make_register_grid()
        linear = self._head("linear")(grid, batch)["worldcover"]
        mlp = self._head("mlp")(grid, batch)["worldcover"]
        assert mlp.shape == linear.shape

    def test_is_a_two_layer_mlp_and_trains(self) -> None:
        """The head has the hidden layer and backpropagates into the register grid."""
        from olmoearth_pretrain.nn.supervision_head import MLP_HEAD_HIDDEN

        head = self._head("mlp")
        layers = list(head.heads["worldcover"])
        assert [type(m).__name__ for m in layers] == [
            "Linear",
            "LayerNorm",
            "ReLU",
            "Linear",
        ]
        assert layers[0].out_features == MLP_HEAD_HIDDEN
        batch = _make_batch_with_worldcover()
        grid = _make_register_grid()
        total, _ = compute_supervision_loss(head(grid, batch), batch, head)
        total.backward()
        assert grid.grad is not None and grid.grad.abs().sum() > 0

    def test_unknown_decoder_rejected(self) -> None:
        """Only linear / mlp / upsample are decoders."""
        with pytest.raises(ValueError, match="decoder"):
            SupervisionModalityConfig(
                task_type=SupervisionTaskType.REGRESSION,
                num_output_channels=1,
                decoder="transformer",
            )
