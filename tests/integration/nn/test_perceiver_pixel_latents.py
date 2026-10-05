"""Pixel (sub-patch) latents on the Perceiver: ``PerceiverConfig.pixel_latents``."""

from typing import Any

import pytest
import torch

from olmoearth_pretrain.data.constants import Modality
from olmoearth_pretrain.datatypes import MaskedOlmoEarthSample, MaskValue
from olmoearth_pretrain.nn.flexi_vit import (
    Encoder,
    EncoderConfig,
    Perceiver,
    PerceiverConfig,
)

B, H, W, T = 2, 8, 8, 2
MODALITIES = [Modality.SENTINEL2_L2A.name]


def _sample() -> MaskedOlmoEarthSample:
    torch.manual_seed(1234)
    num_bands = Modality.SENTINEL2_L2A.num_bands
    return MaskedOlmoEarthSample(
        sentinel2_l2a=torch.randn(B, H, W, T, num_bands),
        sentinel2_l2a_mask=torch.full(
            (B, H, W, T, num_bands), MaskValue.ONLINE_ENCODER.value, dtype=torch.long
        ),
        timestamps=torch.tensor(
            [[[1, 0, 2020], [2, 1, 2020]]], dtype=torch.long
        ).expand(B, -1, -1),
    )


def _encoder(**perceiver_kwargs: Any) -> Encoder:
    torch.manual_seed(0)
    return EncoderConfig(
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
        perceiver_config=PerceiverConfig(
            register_dim=16, latent_depth=2, **perceiver_kwargs
        ),
    ).build()


def test_latent_patch_size_equal_to_patch_size_is_the_patch_grid() -> None:
    """A latent patch size equal to the patch size is the patch-latent Perceiver."""
    patch_model = _encoder().eval()
    coarse = _encoder(pixel_latents=True, eval_latent_patch_size=2).eval()
    coarse.load_state_dict(patch_model.state_dict())
    sample = _sample()
    with torch.no_grad():
        a = patch_model(sample, patch_size=2, input_res=10)
        b = coarse(sample, patch_size=2, input_res=10)
    torch.testing.assert_close(a["registers"], b["registers"])
    torch.testing.assert_close(a["register_positions"], b["register_positions"])


def test_pixel_latents_train_and_eval_grids() -> None:
    """Eval at latent patch size 1 gives a latent per pixel; training stays in budget."""
    encoder = _encoder(
        pixel_latents=True, random_latent_patch_size=True, max_latents=64
    )
    sample = _sample()
    encoder.eval()
    with torch.no_grad():
        out = encoder(sample, patch_size=2, input_res=10)
    assert out["registers"].shape == (B, 8, 8, 16)  # 4x4 patches x 2x2 pixels
    assert out["register_positions"].shape == (B, 64, 2)
    encoder.train()
    shapes = set()
    for _ in range(20):
        out = encoder(sample, patch_size=2, input_res=10)
        shapes.add(tuple(out["registers"].shape[1:3]))
    # 4x4 patches at ps2: latent patch size 1 = 64 latents (== budget), 2 = 16.
    assert shapes <= {(8, 8), (4, 4)} and (8, 8) in shapes
    out["registers"].sum().backward()
    assert encoder.perceiver is not None
    for blk in encoder.perceiver.read_blocks:
        assert any(
            p.grad is not None and p.grad.abs().sum() > 0 for p in blk.parameters()
        )


def _perceiver(**kwargs: Any) -> Perceiver:
    return PerceiverConfig(register_dim=16, pixel_latents=True, **kwargs).build(
        encoder_embedding_size=32,
        encoder_num_heads=2,
        mlp_ratio=2.0,
        position_encoding="rope",
        rope_base=10000.0,
        qk_norm=False,
    )


def test_choose_latent_patch_size_respects_the_budget() -> None:
    """Latent patch sizes whose latent count exceeds the budget are never drawn."""
    torch.manual_seed(0)
    perceiver = _perceiver(random_latent_patch_size=True, max_latents=16).train()
    sizes = {perceiver.choose_latent_patch_size((2, 2), 4) for _ in range(100)}
    # latent patch size 1 = 64 latents (over budget), 2 = 16, 4 = 4.
    assert sizes == {2, 4}
    perceiver.eval()
    assert perceiver.choose_latent_patch_size((2, 2), 4) == 1
    with pytest.raises(ValueError, match="does not divide"):
        _perceiver(eval_latent_patch_size=3).eval().choose_latent_patch_size((2, 2), 4)


def test_pixel_latent_settings_need_pixel_latents() -> None:
    """Latent patch size settings without pixel latents fail validation."""
    config = PerceiverConfig(register_dim=32, eval_latent_patch_size=2)
    with pytest.raises(ValueError, match="pixel_latents"):
        config.validate(encoder_num_heads=4, position_encoding="rope")
    config = PerceiverConfig(
        register_dim=32, pixel_latents=True, random_latent_patch_size=True
    )
    with pytest.raises(ValueError, match="max_latents"):
        config.validate(encoder_num_heads=4, position_encoding="rope")


def test_patch_latent_config_round_trips_without_pixel_fields() -> None:
    """The unset pixel-latent fields are None, so existing configs serialize unchanged."""
    serialized = PerceiverConfig(register_dim=32).as_config_dict()
    for name in (
        "pixel_latents",
        "random_latent_patch_size",
        "max_latents",
        "eval_latent_patch_size",
    ):
        assert name not in serialized
