"""Sub-patch Perceiver latents: the ``latent_patch_size`` forward argument."""

import numpy as np
import pytest
import torch

from olmoearth_pretrain.data.constants import Modality
from olmoearth_pretrain.data.dataloader import choose_latent_patch_size
from olmoearth_pretrain.datatypes import MaskedOlmoEarthSample, MaskValue
from olmoearth_pretrain.nn.flexi_vit import Encoder, EncoderConfig, PerceiverConfig

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


def _encoder(perceiver: bool = True) -> Encoder:
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
        perceiver_config=(
            PerceiverConfig(register_dim=16, latent_depth=2) if perceiver else None
        ),
    ).build()


def test_latent_patch_size_equal_to_patch_size_is_the_patch_grid() -> None:
    """``latent_patch_size == patch_size`` reproduces the default (one per token)."""
    encoder = _encoder().eval()
    sample = _sample()
    with torch.no_grad():
        a = encoder(sample, patch_size=2, input_res=10)
        b = encoder(sample, patch_size=2, input_res=10, latent_patch_size=2)
    torch.testing.assert_close(a["registers"], b["registers"])
    torch.testing.assert_close(a["register_positions"], b["register_positions"])


@pytest.mark.parametrize("training", [True, False])
@pytest.mark.parametrize(
    ("patch_size", "latent_patch_size", "grid"),
    [(2, 1, 8), (4, 1, 8), (4, 2, 4), (4, 4, 2)],
)
def test_latent_patch_size_sets_the_grid(
    training: bool, patch_size: int, latent_patch_size: int, grid: int
) -> None:
    """The grid is ``H / latent_patch_size`` per side, in training and at eval alike."""
    encoder = _encoder().train(training)
    out = encoder(
        _sample(),
        patch_size=patch_size,
        input_res=10,
        latent_patch_size=latent_patch_size,
    )
    assert out["registers"].shape == (B, grid, grid, 16)
    assert out["register_positions"].shape == (B, grid * grid, 2)
    if training:
        out["registers"].sum().backward()
        assert encoder.perceiver is not None
        for blk in encoder.perceiver.read_blocks:
            assert any(
                p.grad is not None and p.grad.abs().sum() > 0 for p in blk.parameters()
            )


def test_latent_patch_size_must_divide_patch_size_and_needs_a_perceiver() -> None:
    """A latent patch size that splits a token unevenly, or no Perceiver, is refused."""
    with pytest.raises(ValueError, match="does not divide"):
        _encoder().eval()(_sample(), patch_size=4, input_res=10, latent_patch_size=3)
    with pytest.raises(ValueError, match="Perceiver"):
        _encoder(perceiver=False).eval()(
            _sample(), patch_size=4, input_res=10, latent_patch_size=1
        )


def test_dataloader_latent_patch_size_draw_respects_the_budget() -> None:
    """Latent patch sizes whose latent count exceeds the budget are never drawn."""
    rng = np.random.default_rng(0)
    # 2x2 tokens at patch size 4: latent patch size 1 = 64 latents (over budget),
    # 2 = 16, 4 = 4.
    sizes = {choose_latent_patch_size(rng, 4, 2, max_latents=16) for _ in range(100)}
    assert sizes == {2, 4}
    # The token's own patch size is always allowed, even over budget.
    over_budget = {
        choose_latent_patch_size(rng, 4, 8, max_latents=1) for _ in range(20)
    }
    assert over_budget == {4}
