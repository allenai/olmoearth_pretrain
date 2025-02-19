"""Test masking."""

import logging

import torch

from helios.data.constants import Modality
from helios.data.dataset import HeliosSample
from helios.train.masking import MaskValue, RandomMaskingStrategy

logger = logging.getLogger(__name__)


def test_random_masking_and_unmask() -> None:
    """Test random masking ratios."""
    b, h, w, t = 100, 16, 16, 8

    days = torch.randint(1, 31, (b, 1, t), dtype=torch.long)
    months = torch.randint(1, 13, (b, 1, t), dtype=torch.long)
    years = torch.randint(2018, 2020, (b, 1, t), dtype=torch.long)
    timestamps = torch.cat([days, months, years], dim=1)  # Shape: (B, 3, T)
    sentinel2_num_bands = Modality.SENTINEL2.num_bands
    latlon_num_bands = Modality.LATLON.num_bands
    batch = HeliosSample(
        sentinel2=torch.ones((b, h, w, t, sentinel2_num_bands)),
        latlon=torch.ones((b, latlon_num_bands)),
        timestamps=timestamps,
    )
    encode_ratio, decode_ratio = 0.25, 0.5
    masked_sample = RandomMaskingStrategy(
        encode_ratio=encode_ratio,
        decode_ratio=decode_ratio,
    ).apply_mask(
        batch,
    )
    # check that each modality has the right masking ratio
    for modality_name in masked_sample._fields:
        if modality_name.endswith("mask"):
            mask = getattr(masked_sample, modality_name)
            logger.info(f"Mask name: {modality_name}")
            if mask is None:
                continue
            total_elements = mask.numel()
            num_encoder = len(mask[mask == MaskValue.ONLINE_ENCODER.value])
            num_decoder = len(mask[mask == MaskValue.DECODER.value])
            assert (
                num_encoder / total_elements
            ) == encode_ratio, f"{modality_name} has incorrect encode mask ratio"
            assert (
                num_decoder / total_elements
            ) == decode_ratio, f"{modality_name} has incorrect decode mask ratio"
            assert (
                mask.shape
                == getattr(
                    batch,
                    masked_sample.get_unmasked_modality_name(modality_name),
                ).shape
            ), f"{modality_name} has incorrect shape"

    unmasked_sample = masked_sample.unmask()
    for modality_name in unmasked_sample._fields:
        if modality_name.endswith("mask"):
            mask = getattr(unmasked_sample, modality_name)
            if mask is not None:
                assert (mask == 0).all()
