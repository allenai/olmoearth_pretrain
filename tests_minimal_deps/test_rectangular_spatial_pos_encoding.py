"""Regression test for rectangular spatial grids in CompositeEncodings.

This test lives in tests_minimal_deps/ so it can run without heavy geo
dependencies (e.g., rasterio) required by the main tests/ suite.
"""

import torch

from olmoearth_pretrain.data.constants import Modality
from olmoearth_pretrain.nn.flexi_vit import CompositeEncodings


def test_composite_encodings_worldcover_rectangular_grid() -> None:
    """Rectangular (H!=W) spatial grids should not crash."""
    composite = CompositeEncodings(
        embedding_size=16,
        supported_modalities=[Modality.WORLDCOVER],
        max_sequence_length=12,
        random_channel_embeddings=True,
    ).eval()

    B, H, W, C, D = 1, 18, 20, 1, 16
    patch_size = 4
    input_res = 10
    tokens = torch.randn(B, H, W, C, D)

    out = composite._apply_encodings_per_modality(
        "worldcover", tokens, timestamps=None, patch_size=patch_size, input_res=input_res
    )
    assert out.shape == tokens.shape

