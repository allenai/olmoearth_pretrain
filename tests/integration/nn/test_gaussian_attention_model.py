"""A small RC-shaped model with Gaussian attention windows, end to end."""

from unittest.mock import patch

import pytest
import torch
from olmo_core.optim import AdamWConfig

from olmoearth_pretrain.data.constants import Modality
from olmoearth_pretrain.data.transform import TransformConfig
from olmoearth_pretrain.nn.encodings import PositionEncoding
from olmoearth_pretrain.nn.flexi_vit import (
    EncoderConfig,
    PerceiverConfig,
    PredictorConfig,
)
from olmoearth_pretrain.nn.latent_mim import LatentMIM, LatentMIMConfig
from olmoearth_pretrain.train.loss import LossConfig
from olmoearth_pretrain.train.masking import (
    MaskedOlmoEarthSample,
    MaskingConfig,
    MaskValue,
)
from olmoearth_pretrain.train.train_module.latent_mim import LatentMIMTrainModuleConfig

B, H, W, T = 2, 8, 8, 3
MODALITIES = [Modality.SENTINEL2_L2A.name, Modality.SENTINEL1.name]


def _model_config(modalities: list[str] = MODALITIES) -> LatentMIMConfig:
    """rc_ld1_pixtgt_pix512's shape (tiny): Gaussian windows everywhere."""
    encoder_config = EncoderConfig(
        supported_modality_names=modalities,
        embedding_size=32,
        num_heads=2,
        depth=2,
        mlp_ratio=2.0,
        max_patch_size=4,
        min_patch_size=1,
        max_sequence_length=12,
        drop_path=0.0,
        position_encoding=PositionEncoding.GAUSSIAN_3D,
        rope_temporal_coordinate_scale=1 / 30,
        perceiver_config=PerceiverConfig(
            register_dim=32,
            latent_depth=1,
            attn_dim=32,
            read_time_rope=True,
            pixel_latents=True,
            random_latent_stride=True,
            max_latents=32,
            eval_latent_stride=1,
        ),
    )
    decoder_config = PredictorConfig(
        supported_modality_names=modalities,
        encoder_embedding_size=32,
        decoder_embedding_size=16,
        num_heads=2,
        depth=1,
        mlp_ratio=2.0,
        max_sequence_length=12,
        drop_path=0.0,
        position_encoding=PositionEncoding.GAUSSIAN_2D,
        use_perceiver=True,
        register_dim=32,
    )
    return LatentMIMConfig(
        encoder_config=encoder_config,
        decoder_config=decoder_config,
        projection_only_target=True,
    )


def _make_sample() -> MaskedOlmoEarthSample:
    """S2 + S1, 3 dates; the last date of both is decoded."""
    torch.manual_seed(1234)
    nb2, nb1 = Modality.SENTINEL2_L2A.num_bands, Modality.SENTINEL1.num_bands
    s2_mask = torch.full((B, H, W, T, nb2), MaskValue.ONLINE_ENCODER.value)
    s2_mask[:, :, :, -1] = MaskValue.DECODER.value
    s1_mask = torch.full((B, H, W, T, nb1), MaskValue.ONLINE_ENCODER.value)
    s1_mask[:, :, :, -1] = MaskValue.DECODER.value
    return MaskedOlmoEarthSample(
        sentinel2_l2a=torch.randn(B, H, W, T, nb2),
        sentinel2_l2a_mask=s2_mask,
        sentinel1=torch.randn(B, H, W, T, nb1),
        sentinel1_mask=s1_mask,
        timestamps=torch.tensor(
            [[[1, 2, 2020], [1, 5, 2020], [1, 8, 2020]]], dtype=torch.long
        ).expand(B, -1, -1),
    )


def test_every_attention_is_gaussian_and_rope_free() -> None:
    """Encoder, Perceiver read (3D), latent self-attention and decoder (2D)."""
    model: LatentMIM = _model_config().build()
    attns = {name: m for name, m in model.named_modules() if name.endswith(".attn")}
    assert attns
    for name, attn in attns.items():
        assert attn.gaussian is not None, name
        assert attn.rope_mixed_freqs is None, name
    perceiver = model.encoder.perceiver
    assert perceiver.read_blocks[0].attn.gaussian.ndim == 3
    assert perceiver.latent_blocks[0].attn.gaussian.ndim == 2
    assert model.encoder.blocks[0].attn.gaussian.ndim == 3
    assert model.decoder.blocks[0].attn.gaussian.ndim == 2


def test_dates_change_the_encoding() -> None:
    """Calendar time reaches the encoder through the windows."""
    torch.manual_seed(0)
    model: LatentMIM = _model_config().build().eval()
    # Give the windows something to do: random (non-prior) predictors.
    for m in model.modules():
        if hasattr(m, "gaussian") and m.gaussian is not None:
            torch.nn.init.normal_(m.gaussian.to_params.weight, std=0.05)
    sample = _make_sample()
    # The first (encoded) capture moves back one year: same month embedding, and no
    # slot-index time encoding in a 3D encoder, so only the windows see the change.
    later = sample._replace(timestamps=sample.timestamps.clone())
    later.timestamps[:, 0, 2] = 2019
    with torch.no_grad():
        a = model.encoder(sample, patch_size=2, input_res=10)["registers"]
        b = model.encoder(later, patch_size=2, input_res=10)["registers"]
    assert torch.isfinite(a).all()
    assert not torch.allclose(a, b)


def test_every_window_gets_gradient() -> None:
    """Gradients reach the window predictor at every attention site."""
    torch.manual_seed(0)
    model: LatentMIM = _model_config().build()
    latent, decoded, *_ = model(_make_sample(), patch_size=2)
    loss = decoded.sentinel2_l2a.square().mean() + decoded.sentinel1.square().mean()
    loss.backward()
    for name, attn in model.named_modules():
        if name.endswith(".attn"):
            grad = attn.gaussian.to_params.weight.grad
            assert grad is not None and grad.abs().sum() > 0, name


@pytest.mark.parametrize("patch_size", [1, 2])
def test_train_step_with_pixel_targets(patch_size: int) -> None:
    """The RC's train module (pixel targets) runs: non-zero finite loss, finite grads."""
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
    loss, *_ = train_module.model_forward(
        _make_sample(), patch_size, train_module.token_exit_cfg
    )
    assert torch.isfinite(loss) and loss > 0
    loss.backward()
    window_grads = {
        name: p.grad
        for name, p in model.named_parameters()
        if "gaussian.to_params" in name
    }
    assert window_grads
    for name, grad in window_grads.items():
        assert grad is not None and torch.isfinite(grad).all(), name


def test_non_spatial_modalities_are_refused() -> None:
    """A latlon token has no place in space for a window to cover."""
    with pytest.raises(ValueError, match="spatial modalities only"):
        _model_config([Modality.SENTINEL2_L2A.name, Modality.LATLON.name]).build()
