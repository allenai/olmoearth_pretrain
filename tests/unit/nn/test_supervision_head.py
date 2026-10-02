"""Unit tests for the supervision head module."""

from typing import Any

import pytest
import torch

from olmoearth_pretrain.data.constants import MISSING_VALUE
from olmoearth_pretrain.datatypes import MaskedOlmoEarthSample, MaskValue
from olmoearth_pretrain.nn.encodings import get_2d_sincos_pos_encoding
from olmoearth_pretrain.nn.supervision_head import (
    SupervisionHead,
    SupervisionHeadConfig,
    SupervisionModalityConfig,
    SupervisionTaskType,
    _build_valid_mask,
    _day_of_year_encoding,
    compute_supervision_loss,
    highpass_patch_target,
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


# --- time-conditioned (per-pixel, per-timestep) reconstruction ------------------------

S2_T = 3
S2_RECON_BANDS = [0, 1, 2, 3]


def _s2_recon_config(**kwargs: Any) -> SupervisionModalityConfig:
    return SupervisionModalityConfig(
        task_type=SupervisionTaskType.REGRESSION,
        num_output_channels=len(S2_RECON_BANDS),
        time_conditioned=True,
        band_indices=S2_RECON_BANDS,
        **kwargs,
    )


def _make_s2_batch() -> MaskedOlmoEarthSample:
    s2 = torch.randn(B, H_PIX, W_PIX, S2_T, 12)
    s2[0, :, :, 2] = MISSING_VALUE  # a missing timestep
    timestamps = torch.tensor(
        [[1, 0, 2023], [15, 5, 2023], [1, 10, 2023]], dtype=torch.long
    ).expand(B, -1, -1)
    return MaskedOlmoEarthSample(
        timestamps=timestamps,
        sentinel2_l2a=s2,
        sentinel2_l2a_mask=torch.zeros(B, H_PIX, W_PIX, S2_T, 3, dtype=torch.long),
    )


def _s2_target(batch: MaskedOlmoEarthSample) -> torch.Tensor:
    """The reconstructed S2 bands of ``batch``."""
    assert batch.sentinel2_l2a is not None
    return batch.sentinel2_l2a[..., S2_RECON_BANDS]


class TestTimeConditionedHead:
    """The per-(pixel, timestep) reconstruction head and its losses."""

    @pytest.mark.parametrize("grid", [(P_H, P_W), (H_PIX, W_PIX), (16, 16)])
    def test_prediction_shape(self, grid: tuple[int, int]) -> None:
        """Pixel-resolution predictions from any grid the target is a multiple of."""
        head = SupervisionHeadConfig(
            modality_configs={"sentinel2_l2a": _s2_recon_config()}
        ).build(embedding_dim=D, max_patch_size=MAX_PATCH_SIZE)
        registers = torch.randn(B, *grid, D)
        preds = head(registers, _make_s2_batch())
        assert preds["sentinel2_l2a"].shape == (B, H_PIX, W_PIX, S2_T, 4)

    def test_factorized_head_equals_concat_mlp(self) -> None:
        """W_z z + W_t phi + W_o enc(offset) is the MLP on the concatenation."""
        torch.manual_seed(0)
        head = SupervisionHeadConfig(
            modality_configs={"sentinel2_l2a": _s2_recon_config()}
        ).build(embedding_dim=D, max_patch_size=MAX_PATCH_SIZE)
        tc = head.heads["sentinel2_l2a"]
        batch = _make_s2_batch()
        registers = torch.randn(B, P_H, P_W, D)
        pred = head(registers, batch)["sentinel2_l2a"]
        s = MAX_PATCH_SIZE
        phi = _day_of_year_encoding(batch.timestamps, 4)
        off = torch.arange(s, dtype=torch.float32) - (s - 1) / 2
        enc = get_2d_sincos_pos_encoding(
            torch.stack(torch.meshgrid(off, off, indexing="ij")), 16
        ).view(s, s, -1)
        weight = torch.cat([tc.z_proj.weight, tc.t_proj.weight, tc.o_proj.weight], 1)
        for b, r, c, t in [(0, 0, 0, 0), (1, 13, 30, 2), (0, 31, 5, 1)]:
            x = torch.cat([registers[b, r // s, c // s], phi[b, t], enc[r % s, c % s]])
            hidden = torch.nn.functional.gelu(weight @ x + tc.z_proj.bias)
            torch.testing.assert_close(pred[b, r, c, t], tc.out(hidden))

    def test_full_target_loss_masks_missing(self) -> None:
        """Perfect predictions on valid units give zero loss; MISSING units are ignored."""
        cfg = _s2_recon_config()
        head = SupervisionHeadConfig(modality_configs={"sentinel2_l2a": cfg}).build(
            embedding_dim=D, max_patch_size=MAX_PATCH_SIZE
        )
        batch = _make_s2_batch()
        target = _s2_target(batch)
        pred = torch.where(target == MISSING_VALUE, torch.zeros_like(target), target)
        loss, _ = compute_supervision_loss({"sentinel2_l2a": pred}, batch, head)
        assert loss.item() == pytest.approx(0.0, abs=1e-6)

    def test_highpass_target_is_zero_on_constant_patches(self) -> None:
        """A patch-constant image has no high-pass content; P=1 gives zero loss."""
        cfg = _s2_recon_config(highpass_patch_mean=True)
        head = SupervisionHeadConfig(modality_configs={"sentinel2_l2a": cfg}).build(
            embedding_dim=D, max_patch_size=MAX_PATCH_SIZE
        )
        batch = _make_s2_batch()
        p = 4
        coarse = torch.randn(B, H_PIX // p, W_PIX // p, S2_T, 12)
        flat = coarse.repeat_interleave(p, 1).repeat_interleave(p, 2)
        batch = batch._replace(sentinel2_l2a=flat)
        zero = torch.zeros(B, H_PIX, W_PIX, S2_T, 4)
        loss, _ = compute_supervision_loss(
            {"sentinel2_l2a": zero}, batch, head, patch_size=p
        )
        assert loss.item() == pytest.approx(0.0, abs=1e-6)
        noisy = torch.randn(B, H_PIX, W_PIX, S2_T, 4, requires_grad=True)
        loss_p1, _ = compute_supervision_loss(
            {"sentinel2_l2a": noisy}, _make_s2_batch(), head, patch_size=1
        )
        assert loss_p1.item() == 0.0
        loss_p1.backward()
        assert noisy.grad is not None and not noisy.grad.any()

    def test_highpass_target_values(self) -> None:
        """The high-pass target is pixel minus its patch's valid-pixel mean."""
        torch.manual_seed(0)
        target = torch.randn(1, 4, 4, 1, 2)
        valid = torch.ones(1, 4, 4, 1, dtype=torch.bool)
        valid[0, 0, 0, 0] = False
        hp = highpass_patch_target(target * valid[..., None], valid, 2)
        patch = target[0, 0:2, 0:2, 0].reshape(4, 2)[1:]
        torch.testing.assert_close(hp[0, 1, 1, 0], target[0, 1, 1, 0] - patch.mean(0))
        torch.testing.assert_close(
            hp[0, 3, 3, 0],
            target[0, 3, 3, 0] - target[0, 2:4, 2:4, 0].reshape(4, 2).mean(0),
        )

    def test_variance_normalization(self) -> None:
        """Predicting the per-band mean gives a normalized loss of exactly 1."""
        cfg = _s2_recon_config(normalize_by_target_variance=True)
        head = SupervisionHeadConfig(modality_configs={"sentinel2_l2a": cfg}).build(
            embedding_dim=D, max_patch_size=MAX_PATCH_SIZE
        )
        batch = _make_s2_batch()
        target = _s2_target(batch)
        valid = (target != MISSING_VALUE).all(-1, keepdim=True)
        mean = (target * valid).sum((0, 1, 2, 3)) / valid.sum()
        pred = mean.expand_as(target).clone()
        loss, _ = compute_supervision_loss({"sentinel2_l2a": pred}, batch, head)
        assert loss.item() == pytest.approx(1.0, rel=1e-4)

    def test_masked_timesteps_only(self) -> None:
        """Only units the online encoder did not see are scored."""
        cfg = _s2_recon_config(masked_timesteps_only=True)
        head = SupervisionHeadConfig(modality_configs={"sentinel2_l2a": cfg}).build(
            embedding_dim=D, max_patch_size=MAX_PATCH_SIZE
        )
        batch = _make_s2_batch()
        assert batch.sentinel2_l2a_mask is not None
        mask = batch.sentinel2_l2a_mask.clone()
        mask[:, :, :, 1] = MaskValue.DECODER.value
        batch = batch._replace(sentinel2_l2a_mask=mask)
        target = _s2_target(batch)
        pred = target.clone()
        pred[:, :, :, 0] += 5.0  # wrong only on the ONLINE timestep 0
        loss, _ = compute_supervision_loss({"sentinel2_l2a": pred}, batch, head)
        assert loss.item() == pytest.approx(0.0, abs=1e-6)

    def test_options_require_time_conditioned(self) -> None:
        """The pixel-head options are rejected on a register-resolution head."""
        with pytest.raises(ValueError, match="time_conditioned"):
            SupervisionModalityConfig(
                task_type=SupervisionTaskType.REGRESSION,
                num_output_channels=1,
                highpass_patch_mean=True,
            )
