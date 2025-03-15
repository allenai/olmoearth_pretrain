"""Test masking."""

import logging

import torch

from helios.data.constants import Modality
from helios.data.dataset import HeliosSample
from helios.train.masking import (
    MaskValue,
    ModalityMaskingStrategy,
    RandomMaskingStrategy,
    SpaceMaskingStrategy,
    TimeMaskingStrategy,
)

logger = logging.getLogger(__name__)


def test_random_masking_and_unmask() -> None:
    """Test random masking ratios."""
    b, h, w, t = 100, 16, 16, 8

    patch_size = 4

    days = torch.randint(1, 31, (b, 1, t), dtype=torch.long)
    months = torch.randint(1, 13, (b, 1, t), dtype=torch.long)
    years = torch.randint(2018, 2020, (b, 1, t), dtype=torch.long)
    timestamps = torch.cat([days, months, years], dim=1)  # Shape: (B, 3, T)
    sentinel2_l2a_num_bands = Modality.SENTINEL2_L2A.num_bands
    worldcover_num_bands = Modality.WORLDCOVER.num_bands
    latlon_num_bands = Modality.LATLON.num_bands
    batch = HeliosSample(
        sentinel2_l2a=torch.ones((b, h, w, t, sentinel2_l2a_num_bands)),
        latlon=torch.ones((b, latlon_num_bands)),
        timestamps=timestamps,
        worldcover=torch.ones((b, h, w, 1, worldcover_num_bands)),
    )
    encode_ratio, decode_ratio = 0.25, 0.5
    masked_sample = RandomMaskingStrategy(
        encode_ratio=encode_ratio,
        decode_ratio=decode_ratio,
    ).apply_mask(
        batch,
        patch_size=patch_size,
    )
    # Check that all values in the first patch are the same (consistent masking)
    first_patch: torch.Tensor = masked_sample.sentinel2_l2a_mask[0, :4, :4, 0, 0]
    first_value: int = first_patch[0, 0]
    assert (first_patch == first_value).all()
    second_patch: torch.Tensor = masked_sample.sentinel2_l2a_mask[0, :4, :4, 1, 0]
    second_value: int = second_patch[0, 0]
    assert (second_patch == second_value).all()
    worldcover_patch: torch.Tensor = masked_sample.worldcover_mask[0, :4, :4, 0]  # type: ignore
    worldcover_value: int = worldcover_patch[0, 0]
    assert (worldcover_patch == worldcover_value).all()
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


def test_space_structure_masking_and_unmask() -> None:
    """Test space structure masking ratios."""
    b, h, w, t = 100, 16, 16, 8

    days = torch.randint(1, 31, (b, 1, t), dtype=torch.long)
    months = torch.randint(1, 13, (b, 1, t), dtype=torch.long)
    years = torch.randint(2018, 2020, (b, 1, t), dtype=torch.long)
    timestamps = torch.cat([days, months, years], dim=1)  # Shape: (B, 3, T)
    sentinel2_l2a_num_bands = Modality.SENTINEL2_L2A.num_bands
    latlon_num_bands = Modality.LATLON.num_bands
    worldcover_num_bands = Modality.WORLDCOVER.num_bands
    batch = HeliosSample(
        sentinel2_l2a=torch.ones((b, h, w, t, sentinel2_l2a_num_bands)),
        latlon=torch.ones((b, latlon_num_bands)),
        timestamps=timestamps,
        worldcover=torch.ones((b, h, w, worldcover_num_bands)),
    )
    encode_ratio, decode_ratio = 0.25, 0.5
    masked_sample = SpaceMaskingStrategy(
        encode_ratio=encode_ratio,
        decode_ratio=decode_ratio,
    ).apply_mask(
        batch,
        patch_size=4,
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


def test_time_structure_masking_and_unmask() -> None:
    """Test time structure masking ratios."""
    b, h, w, t = 100, 16, 16, 8

    patch_size = 4

    days = torch.randint(1, 31, (b, 1, t), dtype=torch.long)
    months = torch.randint(1, 13, (b, 1, t), dtype=torch.long)
    years = torch.randint(2018, 2020, (b, 1, t), dtype=torch.long)
    timestamps = torch.cat([days, months, years], dim=1)  # Shape: (B, 3, T)
    sentinel2_l2a_num_bands = Modality.SENTINEL2_L2A.num_bands
    latlon_num_bands = Modality.LATLON.num_bands
    worldcover_num_bands = Modality.WORLDCOVER.num_bands
    batch = HeliosSample(
        sentinel2_l2a=torch.ones((b, h, w, t, sentinel2_l2a_num_bands)),
        latlon=torch.ones((b, latlon_num_bands)),
        timestamps=timestamps,
        worldcover=torch.ones((b, h, w, worldcover_num_bands)),
    )
    encode_ratio, decode_ratio = 0.25, 0.5
    masked_sample = TimeMaskingStrategy(
        encode_ratio=encode_ratio,
        decode_ratio=decode_ratio,
    ).apply_mask(
        batch,
        patch_size=patch_size,
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


def test_modality_mask_and_unmask() -> None:
    """Test modality structure masking ratios."""
    b, h, w, t = 100, 16, 16, 8

    days = torch.randint(1, 31, (b, 1, t), dtype=torch.long)
    months = torch.randint(1, 13, (b, 1, t), dtype=torch.long)
    years = torch.randint(2018, 2020, (b, 1, t), dtype=torch.long)
    timestamps = torch.cat([days, months, years], dim=1)  # Shape: (B, 3, T)
    sentinel2_l2a_num_bands = Modality.SENTINEL2_L2A.num_bands
    latlon_num_bands = Modality.LATLON.num_bands
    worldcover_num_bands = Modality.WORLDCOVER.num_bands
    batch = HeliosSample(
        sentinel2_l2a=torch.ones((b, h, w, t, sentinel2_l2a_num_bands)),
        latlon=torch.ones((b, latlon_num_bands)),
        timestamps=timestamps,
        worldcover=torch.ones((b, h, w, worldcover_num_bands)),
    )
    total_modalities = len(batch.as_dict()) - 1
    encode_ratio, decode_ratio = 0.25, 0.5
    masked_sample = ModalityMaskingStrategy(
        encode_ratio=encode_ratio,
        decode_ratio=decode_ratio,
    ).apply_mask(
        batch,
    )

    mask_per_modality: list[torch.Tensor] = []  # each tensor will have shape [b, 1]
    for modality_name in masked_sample._fields:
        if modality_name.endswith("mask"):
            mask = getattr(masked_sample, modality_name)
            logger.info(f"Mask name: {modality_name}")
            if mask is None:
                continue

            # check all elements in the mask are the same, per instance in
            # the batch
            flat_mask = torch.flatten(mask, start_dim=1)
            unique_per_instance = torch.unique(flat_mask, dim=1)
            assert unique_per_instance.size(1) == 1
            mask_per_modality.append(unique_per_instance)

            assert (
                mask.shape
                == getattr(
                    batch,
                    masked_sample.get_unmasked_modality_name(modality_name),
                ).shape
            ), f"{modality_name} has incorrect shape"

    # shape [b, num_modalities]
    total_mask = torch.concat(mask_per_modality, dim=-1)
    total_elements = total_mask.numel()
    num_encoder = len(total_mask[total_mask == MaskValue.ONLINE_ENCODER.value])
    num_decoder = len(total_mask[total_mask == MaskValue.DECODER.value])

    expected_encoded_modalities = max(1, int(total_modalities * encode_ratio))
    expected_decoded_modalities = max(1, int(total_modalities * decode_ratio))

    expected_encode_ratio = expected_encoded_modalities / total_modalities
    expected_decode_ratio = expected_decoded_modalities / total_modalities
    assert (
        num_encoder / total_elements
    ) == expected_encode_ratio, "Incorrect encode mask ratio"
    assert (
        num_decoder / total_elements
    ) == expected_decode_ratio, "Incorrect decode mask ratio"
