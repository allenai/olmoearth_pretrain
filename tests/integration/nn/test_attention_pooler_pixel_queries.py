"""Sub-patch location queries: the ``location_query_patch_size`` forward argument."""

import pytest
import torch

from olmoearth_pretrain.data.constants import Modality
from olmoearth_pretrain.datatypes import MaskedOlmoEarthSample, MaskValue
from olmoearth_pretrain.nn.flexi_vit import (
    AttentionPoolerConfig,
    Encoder,
    EncoderConfig,
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


def _encoder(attention_pooler: bool = True) -> Encoder:
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
        attention_pooler_config=(
            AttentionPoolerConfig(pooler_dim=16, pooler_depth=2)
            if attention_pooler
            else None
        ),
    ).build()


def test_location_query_patch_size_equal_to_patch_size_is_the_patch_grid() -> None:
    """``location_query_patch_size == patch_size`` reproduces the default (one per token)."""
    encoder = _encoder().eval()
    sample = _sample()
    with torch.no_grad():
        a = encoder(sample, patch_size=2, input_res=10)
        b = encoder(sample, patch_size=2, input_res=10, location_query_patch_size=2)
    torch.testing.assert_close(a["location_tokens"], b["location_tokens"])
    torch.testing.assert_close(
        a["location_query_positions"], b["location_query_positions"]
    )


@pytest.mark.parametrize("training", [True, False])
@pytest.mark.parametrize(
    ("patch_size", "location_query_patch_size", "grid"),
    [(2, 1, 8), (4, 1, 8), (4, 2, 4), (4, 4, 2)],
)
def test_location_query_patch_size_sets_the_grid(
    training: bool, patch_size: int, location_query_patch_size: int, grid: int
) -> None:
    """The grid is ``H / location_query_patch_size`` per side, in training and at eval alike."""
    encoder = _encoder().train(training)
    out = encoder(
        _sample(),
        patch_size=patch_size,
        input_res=10,
        location_query_patch_size=location_query_patch_size,
    )
    assert out["location_tokens"].shape == (B, grid, grid, 16)
    assert out["location_query_positions"].shape == (B, grid * grid, 2)
    if training:
        out["location_tokens"].sum().backward()
        assert encoder.attention_pooler is not None
        for blk in encoder.attention_pooler.read_blocks:
            assert any(
                p.grad is not None and p.grad.abs().sum() > 0 for p in blk.parameters()
            )


def test_location_query_patch_size_must_divide_patch_size_and_needs_an_attention_pooler() -> (
    None
):
    """A location-location-query patch size that splits a token unevenly, or no AttentionPooler, is refused."""
    with pytest.raises(ValueError, match="does not divide"):
        _encoder().eval()(
            _sample(), patch_size=4, input_res=10, location_query_patch_size=3
        )
    with pytest.raises(ValueError, match="AttentionPooler"):
        _encoder(attention_pooler=False).eval()(
            _sample(), patch_size=4, input_res=10, location_query_patch_size=1
        )
