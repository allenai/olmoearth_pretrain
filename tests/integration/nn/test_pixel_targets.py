"""Tests for subsampled pixel-resolution MIM targets (``nn/pixel_targets.py``).

Covers the three pieces the ``rc_pixtgt_pix512`` arms rely on:

* ``gather_pixels`` keeps exactly the drawn pixel of every token cell;
* a shifted decoder query lands on the coordinate of the per-pixel latent it targets;
* the train module's pixel-target forward runs end to end on a small per-pixel
  latent model, with the same decode-query count as the patch-target forward.
"""

from unittest.mock import patch

import pytest
import torch
from olmo_core.optim.adamw import AdamWConfig

from olmoearth_pretrain.data.constants import Modality
from olmoearth_pretrain.data.transform import TransformConfig
from olmoearth_pretrain.nn.flexi_vit import (
    CompositeEncodings,
    EncoderConfig,
    Perceiver,
    PerceiverConfig,
    PredictorConfig,
)
from olmoearth_pretrain.nn.latent_mim import LatentMIM, LatentMIMConfig
from olmoearth_pretrain.nn.pixel_targets import (
    gather_pixels,
    offsets_to_query_shift,
    sample_pixel_offsets,
    spatial_token_grid,
)
from olmoearth_pretrain.train.loss import LossConfig
from olmoearth_pretrain.train.masking import (
    MaskedOlmoEarthSample,
    MaskingConfig,
    MaskValue,
)
from olmoearth_pretrain.train.train_module.latent_mim import LatentMIMTrainModuleConfig

B, H, W, T = 2, 8, 8, 2
MODALITIES = [Modality.SENTINEL2_L2A.name, Modality.LATLON.name]


def _make_sample() -> MaskedOlmoEarthSample:
    """S2 + latlon sample; the top-left 4x4 block is decoded at t=0."""
    torch.manual_seed(1234)
    num_bands = Modality.SENTINEL2_L2A.num_bands
    mask = torch.zeros(B, H, W, T, num_bands, dtype=torch.long)
    mask[:, 0:4, 0:4, 0, :] = MaskValue.DECODER.value
    mask[:, 4:8, 0:4, 1, :] = MaskValue.TARGET_ENCODER_ONLY.value
    return MaskedOlmoEarthSample(
        sentinel2_l2a=torch.randn(B, H, W, T, num_bands),
        sentinel2_l2a_mask=mask,
        latlon=torch.randn(B, Modality.LATLON.num_bands),
        latlon_mask=torch.zeros(B, Modality.LATLON.num_bands, dtype=torch.long),
        timestamps=torch.tensor(
            [[[1, 0, 2020], [2, 1, 2020]]], dtype=torch.long
        ).expand(B, -1, -1),
    )


def _model_config() -> LatentMIMConfig:
    """Small rc_pix512-shaped model with a projection-only target.

    A Perceiver (run at ``latent_patch_size=1``: per-pixel latents) and a 2D-RoPE
    decoder over it. Default tokenization: Sentinel-2 in three band sets.
    """
    encoder_config = EncoderConfig(
        supported_modality_names=MODALITIES,
        embedding_size=32,
        num_heads=2,
        depth=2,
        mlp_ratio=2.0,
        max_patch_size=4,
        min_patch_size=1,
        max_sequence_length=12,
        drop_path=0.0,
        position_encoding="rope",
        perceiver_config=PerceiverConfig(register_dim=16, latent_depth=2),
    )
    decoder_config = PredictorConfig(
        supported_modality_names=MODALITIES,
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
    )


def test_gather_pixels_keeps_the_drawn_pixel() -> None:
    """Cell (i, j) of the gathered field is pixel (i*p + o_r, j*p + o_c)."""
    sample = _make_sample()
    patch_size = 4
    grid = spatial_token_grid(sample, patch_size)
    assert grid == (H // patch_size, W // patch_size)
    offsets = sample_pixel_offsets(B, grid, patch_size, torch.device("cpu"))
    gathered = gather_pixels(sample, offsets, patch_size)
    assert gathered.sentinel2_l2a is not None and sample.sentinel2_l2a is not None
    assert gathered.sentinel2_l2a.shape == (
        B,
        *grid,
        T,
        Modality.SENTINEL2_L2A.num_bands,
    )
    assert (
        gathered.sentinel2_l2a_mask is not None
        and sample.sentinel2_l2a_mask is not None
    )
    for b in range(B):
        for i in range(grid[0]):
            for j in range(grid[1]):
                r = i * patch_size + int(offsets[b, i, j, 0])
                c = j * patch_size + int(offsets[b, i, j, 1])
                assert torch.equal(
                    gathered.sentinel2_l2a[b, i, j], sample.sentinel2_l2a[b, r, c]
                )
                assert torch.equal(
                    gathered.sentinel2_l2a_mask[b, i, j],
                    sample.sentinel2_l2a_mask[b, r, c],
                )
    # Non-spatial modalities pass through untouched.
    assert gathered.latlon is sample.latlon


@pytest.mark.parametrize("patch_size", [2, 4])
def test_shifted_query_lands_on_its_pixel_latent(patch_size: int) -> None:
    """A shifted query's coordinate is its pixel's latent (latent patch size 1)."""
    torch.manual_seed(0)
    model = _model_config().build()
    decoder = model.decoder
    h_p, w_p = H // patch_size, W // patch_size
    offsets = sample_pixel_offsets(B, (h_p, w_p), patch_size, torch.device("cpu"))
    shift = offsets_to_query_shift(offsets, patch_size)
    gsd_ratio = (
        CompositeEncodings.calculate_gsd_ratio(10, patch_size)
        * decoder.rope_coordinate_scale
    )
    tokens = torch.zeros(B, h_p, w_p, T, 1, 16)
    query_positions = decoder._build_2d_rope_positions_for_modality(
        modality_name="sentinel2_l2a",
        modality=Modality.SENTINEL2_L2A,
        tokens=tokens,
        gsd_ratio=gsd_ratio,
        query_pixel_shift=shift,
    )
    register_positions = Perceiver.build_pixel_latent_positions(
        B, (H, W), patch_size, gsd_ratio, torch.device("cpu"), latent_patch_size=1
    ).view(B, H, W, 2)
    for b in range(B):
        for i in range(h_p):
            for j in range(w_p):
                r = i * patch_size + int(offsets[b, i, j, 0])
                c = j * patch_size + int(offsets[b, i, j, 1])
                for t in range(T):
                    torch.testing.assert_close(
                        query_positions[b, i, j, t, 0], register_positions[b, r, c]
                    )


def test_zero_shift_is_the_patch_query_and_a_shift_moves_it() -> None:
    """A zero shift reproduces the unshifted decoder; a real shift changes it."""
    torch.manual_seed(0)
    model = _model_config().build()
    model.eval()
    sample = _make_sample()
    patch_size = 2
    grid = (H // patch_size, W // patch_size)
    with torch.no_grad():
        base = model.forward(sample, patch_size, latent_patch_size=1)[1].sentinel2_l2a
        zero = model.forward(
            sample,
            patch_size,
            query_pixel_shift=torch.zeros(B, *grid, 2),
            latent_patch_size=1,
        )[1].sentinel2_l2a
        shifted = model.forward(
            sample,
            patch_size,
            query_pixel_shift=torch.full((B, *grid, 2), 0.25),
            latent_patch_size=1,
        )[1].sentinel2_l2a
    assert base is not None and zero is not None and shifted is not None
    torch.testing.assert_close(zero, base)
    assert not torch.allclose(shifted, base)


@pytest.mark.parametrize("patch_size", [1, 2, 4])
def test_train_module_pixel_target_forward(patch_size: int) -> None:
    """model_forward with pixel targets: finite loss, gradients, same query count."""
    torch.manual_seed(0)
    model: LatentMIM = _model_config().build()
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
        train_module = config.build(model, device=torch.device("cpu"))
    sample = _make_sample()
    loss, _latent, decoded, target_output, _metrics = train_module.model_forward(
        sample, patch_size, train_module.token_exit_cfg, latent_patch_size=1
    )
    assert torch.isfinite(loss)
    # One query and one target per token, exactly as with patch targets.
    assert decoded.sentinel2_l2a is not None and target_output.sentinel2_l2a is not None
    assert decoded.sentinel2_l2a.shape[:3] == (B, H // patch_size, W // patch_size)
    assert target_output.sentinel2_l2a.shape[:-1] == decoded.sentinel2_l2a.shape[:-1]
    loss.backward()
    grads = [p.grad for p in model.decoder.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)


def test_pixel_targets_require_projection_target() -> None:
    """A full target encoder cannot be projected per pixel: refuse it at build time."""
    model_config = _model_config()
    model_config.projection_only_target = False
    model = model_config.build()
    config = LatentMIMTrainModuleConfig(
        optim_config=AdamWConfig(lr=1e-4),
        rank_microbatch_size=B,
        token_exit_cfg={modality: 0 for modality in Modality.names()},
        transform_config=TransformConfig(transform_type="no_transform"),
        pixel_targets=True,
    )
    with patch("olmoearth_pretrain.train.train_module.train_module.build_world_mesh"):
        with pytest.raises(ValueError, match="projection_only_target"):
            config.build(model, device=torch.device("cpu"))
