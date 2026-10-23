"""Tests for the v1.3 NAIP arms (``scripts/official/v1_3/naip``).

* ``gather_pixels`` keeps the ``f x f`` block of a finer-grid modality (naip_10, f = 4)
  under the drawn base pixel;
* the ``subpixel_stats`` supervision target and the cosine regression loss;
* the train module's pixel-target forward with naip_10 encoded (N2) and with naip_10 /
  naip_dino as supervision-only targets (N3 / N4).
"""

from unittest.mock import patch

import pytest
import torch
from olmo_core.optim.adamw import AdamWConfig

from olmoearth_pretrain.data.constants import MISSING_VALUE, Modality
from olmoearth_pretrain.data.transform import TransformConfig
from olmoearth_pretrain.nn.flexi_vit import (
    EncoderConfig,
    PerceiverConfig,
    PredictorConfig,
)
from olmoearth_pretrain.nn.latent_mim import LatentMIM, LatentMIMConfig
from olmoearth_pretrain.nn.pixel_targets import (
    gather_pixels,
    sample_pixel_offsets,
    spatial_token_grid,
)
from olmoearth_pretrain.nn.supervision_head import (
    SupervisionHeadConfig,
    SupervisionModalityConfig,
    SupervisionTaskType,
    _subpixel_stats,
    compute_supervision_loss,
)
from olmoearth_pretrain.train.loss import LossConfig
from olmoearth_pretrain.train.masking import (
    MaskedOlmoEarthSample,
    MaskingConfig,
    MaskValue,
)
from olmoearth_pretrain.train.train_module.latent_mim import (
    LatentMIMTrainModule,
    LatentMIMTrainModuleConfig,
)

B, H, W, T = 2, 8, 8, 2
F_NAIP = Modality.NAIP_10.image_tile_size_factor
S2 = Modality.SENTINEL2_L2A.name
NAIP = Modality.NAIP_10.name
DINO = Modality.NAIP_DINO.name


def _make_sample(with_dino: bool = False) -> MaskedOlmoEarthSample:
    """S2 + naip_10 (+ naip_dino); top-left 4x4 S2 block decoded at t=0.

    Sample 1 has no NAIP (all MISSING), as ~82% of the training samples.
    """
    torch.manual_seed(1234)
    s2_bands = Modality.SENTINEL2_L2A.num_bands
    s2_mask = torch.zeros(B, H, W, T, s2_bands, dtype=torch.long)
    s2_mask[:, 0:4, 0:4, 0, :] = MaskValue.DECODER.value
    naip = torch.randn(B, H * F_NAIP, W * F_NAIP, 1, Modality.NAIP_10.num_bands)
    naip[1] = MISSING_VALUE
    naip_mask = torch.zeros_like(naip, dtype=torch.long)
    naip_mask[:, 0 : 4 * F_NAIP, 4 * F_NAIP :] = MaskValue.DECODER.value
    naip_mask[1] = MaskValue.MISSING.value
    fields = dict(
        sentinel2_l2a=torch.randn(B, H, W, T, s2_bands),
        sentinel2_l2a_mask=s2_mask,
        naip_10=naip,
        naip_10_mask=naip_mask,
        timestamps=torch.tensor(
            [[[1, 0, 2020], [2, 1, 2020]]], dtype=torch.long
        ).expand(B, -1, -1),
    )
    if with_dino:
        dino = torch.randn(B, H, W, 1, Modality.NAIP_DINO.num_bands)
        dino[1] = MISSING_VALUE
        dino_mask = torch.full_like(dino, MaskValue.DECODER.value, dtype=torch.long)
        dino_mask[1] = MaskValue.MISSING.value
        fields.update(naip_dino=dino, naip_dino_mask=dino_mask)
    return MaskedOlmoEarthSample(**fields)


def _model_config(
    encoded: list[str], supervision: dict[str, SupervisionModalityConfig] | None
) -> LatentMIMConfig:
    """Small rc_pix512-shaped model (per-pixel random-stride latents, projection target)."""
    encoder_config = EncoderConfig(
        supported_modality_names=encoded,
        embedding_size=32,
        num_heads=2,
        depth=2,
        mlp_ratio=2.0,
        max_patch_size=4,
        min_patch_size=1,
        max_sequence_length=12,
        drop_path=0.0,
        position_encoding="rope",
        perceiver_config=PerceiverConfig(
            register_dim=16,
            latent_depth=1,
            pixel_latents=True,
            random_latent_stride=True,
            max_latents=32,
            eval_latent_stride=1,
        ),
    )
    decoder_config = PredictorConfig(
        supported_modality_names=encoded,
        encoder_embedding_size=32,
        decoder_embedding_size=16,
        num_heads=2,
        depth=1,
        mlp_ratio=2.0,
        max_sequence_length=12,
        drop_path=0.0,
        position_encoding="rope",
        use_perceiver=True,
        register_dim=16,
    )
    return LatentMIMConfig(
        encoder_config=encoder_config,
        decoder_config=decoder_config,
        projection_only_target=True,
        supervision_head_config=(
            SupervisionHeadConfig(modality_configs=supervision) if supervision else None
        ),
    )


def _train_module(model: LatentMIM) -> LatentMIMTrainModule:
    config = LatentMIMTrainModuleConfig(
        optim_config=AdamWConfig(lr=1e-4),
        rank_microbatch_size=B,
        loss_config=LossConfig(
            loss_config={
                "type": "modality_patch_discrimination_masked_negatives_vec",
                "tau": 0.1,
                "same_target_threshold": 0.999,
            }
        ),
        masking_config=MaskingConfig(strategy_config={"type": "random"}),
        token_exit_cfg={modality: 0 for modality in Modality.names()},
        ema_decay=(1.0, 1.0),
        transform_config=TransformConfig(transform_type="no_transform"),
        pixel_targets=True,
    )
    with patch("olmoearth_pretrain.train.train_module.train_module.build_world_mesh"):
        return config.build(model, device=torch.device("cpu"))


@pytest.mark.parametrize("patch_size", [1, 2, 4])
def test_gather_keeps_the_naip_block_under_the_drawn_pixel(patch_size: int) -> None:
    """naip_10 cell (i, j) is the f x f block under S2 pixel (i*p + o_r, j*p + o_c)."""
    sample = _make_sample()
    grid = spatial_token_grid(sample, patch_size)
    assert grid == (H // patch_size, W // patch_size)
    offsets = sample_pixel_offsets(B, grid, patch_size, torch.device("cpu"))
    gathered = gather_pixels(sample, offsets, patch_size)
    assert gathered.naip_10 is not None and gathered.naip_10_mask is not None
    assert gathered.naip_10.shape == (B, grid[0] * F_NAIP, grid[1] * F_NAIP, 1, 4)
    naip, naip_mask = sample.naip_10, sample.naip_10_mask
    s2, s2_gathered = sample.sentinel2_l2a, gathered.sentinel2_l2a
    assert naip is not None and naip_mask is not None
    assert s2 is not None and s2_gathered is not None
    for b in range(B):
        for i in range(grid[0]):
            for j in range(grid[1]):
                r = (i * patch_size + int(offsets[b, i, j, 0])) * F_NAIP
                c = (j * patch_size + int(offsets[b, i, j, 1])) * F_NAIP
                got = gathered.naip_10[
                    b, i * F_NAIP : (i + 1) * F_NAIP, j * F_NAIP : (j + 1) * F_NAIP
                ]
                assert torch.equal(got, naip[b, r : r + F_NAIP, c : c + F_NAIP])
                got_mask = gathered.naip_10_mask[
                    b, i * F_NAIP : (i + 1) * F_NAIP, j * F_NAIP : (j + 1) * F_NAIP
                ]
                assert torch.equal(
                    got_mask, naip_mask[b, r : r + F_NAIP, c : c + F_NAIP]
                )
                # The S2 pixel is still the base-grid gather.
                assert torch.equal(
                    s2_gathered[b, i, j],
                    s2[b, r // F_NAIP, c // F_NAIP],
                )


def test_token_grid_rejects_a_misaligned_fine_grid() -> None:
    """A naip_10 field that is not f x the S2 grid is refused."""
    sample = _make_sample()
    assert sample.naip_10 is not None
    bad = sample._replace(naip_10=sample.naip_10[:, :-4], naip_10_mask=None)
    with pytest.raises(ValueError):
        spatial_token_grid(bad, 2)


def test_subpixel_stats() -> None:
    """5 m means of 2x2 blocks + 10 m stds of 4x4 blocks; missing blocks stay missing."""
    x = torch.randn(1, 8, 8, 1, 4)
    x[0, 0, 5] = MISSING_VALUE  # in S2 pixel (0, 1)
    out = _subpixel_stats(x, 4)
    assert out.shape == (1, 4, 4, 1, 8)
    torch.testing.assert_close(out[0, 3, 2, 0, :4], x[0, 6:8, 4:6, 0].mean(dim=(0, 1)))
    std = x[0, 4:8, 4:8, 0].reshape(16, 4).std(dim=0, unbiased=False)
    for r in (2, 3):
        for c in (2, 3):
            torch.testing.assert_close(out[0, r, c, 0, 4:], std)
    # S2 pixel (0, 1) holds a missing NAIP pixel: all four 5 m cells are missing.
    assert (out[0, 0:2, 2:4] == MISSING_VALUE).all()
    assert (out[0, 0:2, 0:2] != MISSING_VALUE).all()


def test_cosine_supervision_loss() -> None:
    """Cosine regression: 0 for a perfect prediction, missing pixels ignored."""
    sample = _make_sample(with_dino=True)
    cfg = SupervisionModalityConfig(
        task_type=SupervisionTaskType.REGRESSION,
        num_output_channels=Modality.NAIP_DINO.num_bands,
        regression_loss_type="cosine",
    )
    head = SupervisionHeadConfig(modality_configs={DINO: cfg}).build(
        embedding_dim=8, max_patch_size=1
    )
    dino = sample.naip_dino
    assert dino is not None
    perfect = {DINO: dino.clone() * 3.0}
    loss, _ = compute_supervision_loss(perfect, sample, head)
    assert float(loss) == pytest.approx(0.0, abs=1e-5)
    flipped = {DINO: -dino.clone()}
    loss, _ = compute_supervision_loss(flipped, sample, head)
    assert float(loss) == pytest.approx(2.0, abs=1e-4)


@pytest.mark.parametrize("patch_size", [1, 2, 4])
def test_n2_forward_with_naip_encoded(patch_size: int) -> None:
    """N2: naip_10 encoded + pixel targets: finite loss, naip_10 decoded on the S2 grid."""
    torch.manual_seed(0)
    model: LatentMIM = _model_config([S2, NAIP], None).build()
    train_module = _train_module(model)
    loss, _latent, decoded, target_output, _metrics = train_module.model_forward(
        _make_sample(), patch_size, train_module.token_exit_cfg
    )
    assert torch.isfinite(loss)
    assert decoded.naip_10 is not None and target_output.naip_10 is not None
    assert decoded.naip_10.shape[:3] == (B, H // patch_size, W // patch_size)
    assert target_output.naip_10.shape[:-1] == decoded.naip_10.shape[:-1]
    loss.backward()
    grads = [
        p.grad
        for name, p in model.encoder.named_parameters()
        if "naip_10" in name and p.grad is not None
    ]
    assert grads and all(torch.isfinite(g).all() for g in grads)


@pytest.mark.parametrize("patch_size", [1, 2, 4])
def test_n3_n4_forward_with_target_only_naip(patch_size: int) -> None:
    """N3/N4: naip_10 / naip_dino only feed supervision heads; both get gradients."""
    torch.manual_seed(0)
    supervision = {
        NAIP: SupervisionModalityConfig(
            task_type=SupervisionTaskType.REGRESSION,
            num_output_channels=2 * Modality.NAIP_10.num_bands,
            regression_loss_type="l1",
            target_transform="subpixel_stats",
        ),
        DINO: SupervisionModalityConfig(
            task_type=SupervisionTaskType.REGRESSION,
            num_output_channels=Modality.NAIP_DINO.num_bands,
            regression_loss_type="cosine",
        ),
    }
    model: LatentMIM = _model_config([S2], supervision).build()
    train_module = _train_module(model)
    loss, _latent, decoded, _target, metrics = train_module.model_forward(
        _make_sample(with_dino=True), patch_size, train_module.token_exit_cfg
    )
    assert torch.isfinite(loss)
    assert decoded.naip_10 is None and decoded.naip_dino is None
    assert metrics is not None
    assert float(metrics[f"supervision/{NAIP}"]) > 0
    assert float(metrics[f"supervision/{DINO}"]) > 0
    loss.backward()
    assert model.supervision_head is not None
    for name in (NAIP, DINO):
        grad = model.supervision_head.heads[name].weight.grad
        assert grad is not None and torch.isfinite(grad).all() and grad.abs().sum() > 0
