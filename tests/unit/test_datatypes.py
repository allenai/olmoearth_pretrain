"""MaskedOlmoEarthSample helpers."""

import torch

from olmoearth_pretrain.datatypes import MaskedOlmoEarthSample


def test_crop_slices_spatial_modalities_and_keeps_the_rest() -> None:
    """Spatial modalities and their masks are cropped; timestamps and latlon are not."""
    s2 = torch.randn(1, 8, 6, 3, 12)
    sample = MaskedOlmoEarthSample(
        sentinel2_l2a=s2,
        sentinel2_l2a_mask=torch.zeros(1, 8, 6, 3, 12),
        latlon=torch.randn(1, 2),
        latlon_mask=torch.zeros(1, 2),
        timestamps=torch.zeros(1, 3, 3, dtype=torch.long),
    )
    crop = sample.crop(slice(2, 5), slice(1, 4))
    assert crop.sentinel2_l2a is not None and crop.sentinel2_l2a_mask is not None
    torch.testing.assert_close(crop.sentinel2_l2a, s2[:, 2:5, 1:4])
    assert crop.sentinel2_l2a_mask.shape == (1, 3, 3, 3, 12)
    assert crop.latlon is sample.latlon
    assert crop.timestamps is sample.timestamps
