"""Test masking."""

import logging
import random

import torch

from olmoearth_pretrain.data.constants import MISSING_VALUE, Modality
from olmoearth_pretrain.data.dataset import OlmoEarthSample
from olmoearth_pretrain.train.masking import (
    MaskedOlmoEarthSample,
    MaskValue,
    ModalityCrossRandomMaskingStrategy,
    ModalityCrossSpaceMaskingStrategy,
    ModalityMaskingStrategy,
    ModalitySpaceTimeMaskingStrategy,
    RandomMaskingStrategy,
    RandomRangeMaskingStrategy,
    SpaceMaskingStrategy,
    TimeMaskingStrategy,
)

logger = logging.getLogger(__name__)


def test_random_masking_and_unmask() -> None:
    """Test random masking ratios."""
    b, h, w, t = 4, 16, 16, 8

    patch_size = 4

    days = torch.randint(1, 31, (b, 1, t), dtype=torch.long)
    months = torch.randint(1, 13, (b, 1, t), dtype=torch.long)
    years = torch.randint(2018, 2020, (b, 1, t), dtype=torch.long)
    timestamps = torch.cat([days, months, years], dim=1)  # Shape: (B, 3, T)
    sentinel2_l2a_num_bands = Modality.SENTINEL2_L2A.num_bands
    worldcover_num_bands = Modality.WORLDCOVER.num_bands
    latlon_num_bands = Modality.LATLON.num_bands
    batch = OlmoEarthSample(
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
    assert masked_sample.sentinel2_l2a_mask is not None
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
            unmasked_modality_name = masked_sample.get_unmasked_modality_name(
                modality_name
            )
            modality = Modality.get(unmasked_modality_name)
            mask = getattr(masked_sample, modality_name)
            data = getattr(masked_sample, unmasked_modality_name)
            logger.info(f"Mask name: {modality_name}")
            if mask is None:
                continue
            total_elements = mask.numel()
            num_encoder = len(mask[mask == MaskValue.ONLINE_ENCODER.value])
            num_decoder = len(mask[mask == MaskValue.DECODER.value])
            assert (num_encoder / total_elements) == encode_ratio, (
                f"{modality_name} has incorrect encode mask ratio"
            )
            assert (num_decoder / total_elements) == decode_ratio, (
                f"{modality_name} has incorrect decode mask ratio"
            )
            assert mask.shape[:-1] == data.shape[:-1], (
                f"{modality_name} has incorrect shape"
            )
            assert mask.shape[-1] == modality.num_band_sets, (
                f"{modality_name} has incorrect num band sets"
            )

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
    batch = OlmoEarthSample(
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
            unmasked_modality_name = masked_sample.get_unmasked_modality_name(
                modality_name
            )
            modality = Modality.get(unmasked_modality_name)
            mask = getattr(masked_sample, modality_name)
            data = getattr(masked_sample, unmasked_modality_name)
            logger.info(f"Mask name: {modality_name}")
            if mask is None:
                continue
            total_elements = mask.numel()
            num_encoder = len(mask[mask == MaskValue.ONLINE_ENCODER.value])
            num_decoder = len(mask[mask == MaskValue.DECODER.value])
            assert (num_encoder / total_elements) == encode_ratio, (
                f"{modality_name} has incorrect encode mask ratio"
            )
            assert (num_decoder / total_elements) == decode_ratio, (
                f"{modality_name} has incorrect decode mask ratio"
            )
            assert mask.shape[:-1] == data.shape[:-1], (
                f"{modality_name} has incorrect shape"
            )
            assert mask.shape[-1] == modality.num_band_sets, (
                f"{modality_name} has incorrect num band sets"
            )

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
    batch = OlmoEarthSample(
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
            unmasked_modality_name = masked_sample.get_unmasked_modality_name(
                modality_name
            )
            modality = Modality.get(unmasked_modality_name)
            mask = getattr(masked_sample, modality_name)
            data = getattr(masked_sample, unmasked_modality_name)
            logger.info(f"Mask name: {modality_name}")
            if mask is None:
                continue
            total_elements = mask.numel()
            num_encoder = len(mask[mask == MaskValue.ONLINE_ENCODER.value])
            num_decoder = len(mask[mask == MaskValue.DECODER.value])
            assert (num_encoder / total_elements) == encode_ratio, (
                f"{modality_name} has incorrect encode mask ratio"
            )
            assert (num_decoder / total_elements) == decode_ratio, (
                f"{modality_name} has incorrect decode mask ratio"
            )
            assert mask.shape[:-1] == data.shape[:-1], (
                f"{modality_name} has incorrect shape"
            )
            assert mask.shape[-1] == modality.num_band_sets, (
                f"{modality_name} has incorrect num band sets"
            )

    unmasked_sample = masked_sample.unmask()
    for modality_name in unmasked_sample._fields:
        if modality_name.endswith("mask"):
            mask = getattr(unmasked_sample, modality_name)
            if mask is not None:
                assert (mask == 0).all()


def test_time_with_missing_timesteps_structure_masking_and_unmask() -> None:
    """Test time structure masking ratios."""
    b, h, w, t = 8, 8, 8, 8

    patch_size = 4

    days = torch.randint(1, 31, (b, 1, t), dtype=torch.long)
    months = torch.randint(1, 13, (b, 1, t), dtype=torch.long)
    years = torch.randint(2018, 2020, (b, 1, t), dtype=torch.long)
    timestamps = torch.cat([days, months, years], dim=1)  # Shape: (B, 3, T)
    sentinel2_l2a_num_bands = Modality.SENTINEL2_L2A.num_bands
    sentinel1_num_bands = Modality.SENTINEL1.num_bands
    latlon_num_bands = Modality.LATLON.num_bands
    worldcover_num_bands = Modality.WORLDCOVER.num_bands

    # Create data with random missing timesteps for each modality and sample
    sentinel2_l2a = torch.ones((b, h, w, t, sentinel2_l2a_num_bands))
    sentinel1 = torch.ones((b, h, w, t, sentinel1_num_bands))

    # Randomly set some timesteps to missing for each sample and modality
    for sample_idx in range(b):
        # For sentinel2_l2a: randomly mask 2-4 timesteps per sample
        num_missing_timesteps = torch.randint(2, 5, (1,)).item()
        missing_timesteps = torch.randint(2, 5, (num_missing_timesteps,))
        sentinel2_l2a[sample_idx, :, :, missing_timesteps, :] = MISSING_VALUE

        # For sentinel1: randomly mask 2-4 timesteps per sample (different from sentinel2)
        num_missing_timesteps = torch.randint(2, 5, (1,)).item()
        missing_timesteps = torch.randint(2, 5, (num_missing_timesteps,))
        sentinel1[sample_idx, :, :, missing_timesteps, :] = MISSING_VALUE

    batch = OlmoEarthSample(
        sentinel2_l2a=sentinel2_l2a,
        sentinel1=sentinel1,
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
    for i in range(b):
        # for every sample check that at least 1 modality is present at each timestep
        timestamp_present_mask = torch.zeros((t), dtype=torch.bool)
        for modality_name in masked_sample._fields:
            if modality_name.endswith("mask"):
                mask = getattr(masked_sample, modality_name)
                unmasked_modality_name = masked_sample.get_unmasked_modality_name(
                    modality_name
                )
                modality_spec = Modality.get(unmasked_modality_name)
                if not modality_spec.is_multitemporal:
                    continue
                logger.info(f"Mask name: {modality_name}")
                if mask is None:
                    continue
                present_timesteps = (mask[i] != MaskValue.MISSING.value).all(
                    dim=(0, 1, 3)
                )
                timestamp_present_mask = timestamp_present_mask | present_timesteps
        assert timestamp_present_mask.any(), f"Sample {i} has no present modalities"

    unmasked_sample = masked_sample.unmask()
    for modality_name in unmasked_sample._fields:
        if modality_name.endswith("mask"):
            mask = getattr(unmasked_sample, modality_name)
            if mask is not None:
                assert (mask == 0).all()


def test_modality_space_time_masking_and_unmask() -> None:
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
    batch = OlmoEarthSample(
        sentinel2_l2a=torch.ones((b, h, w, t, sentinel2_l2a_num_bands)),
        latlon=torch.ones((b, latlon_num_bands)),
        timestamps=timestamps,
        worldcover=torch.ones((b, h, w, worldcover_num_bands)),
    )
    encode_ratio, decode_ratio = 0.5, 0.5
    strategy = ModalitySpaceTimeMaskingStrategy(
        encode_ratio=encode_ratio,
        decode_ratio=decode_ratio,
    )
    # Run random masking a few times just to make sure it works
    for i in range(10):
        masked_sample = strategy.apply_mask(
            batch,
            patch_size=patch_size,
        )
        for modality_name in masked_sample._fields:
            if modality_name.endswith("mask"):
                unmasked_modality_name = masked_sample.get_unmasked_modality_name(
                    modality_name
                )
                modality = Modality.get(unmasked_modality_name)
                mask = getattr(masked_sample, modality_name)
                data = getattr(masked_sample, unmasked_modality_name)
                logger.info(f"Mask name: {modality_name}")
                if mask is None:
                    continue
                # TODO check ratios depending on masking strategy?
                assert mask.shape[:-1] == data.shape[:-1], (
                    f"{modality_name} has incorrect shape"
                )
                assert mask.shape[-1] == modality.num_band_sets, (
                    f"{modality_name} has incorrect num band sets {mask.shape} {modality.num_band_sets}"
                )

        unmasked_sample = masked_sample.unmask()
        for modality_name in unmasked_sample._fields:
            if modality_name.endswith("mask"):
                mask = getattr(unmasked_sample, modality_name)
                if mask is not None:
                    assert (mask == 0).all()


def test_create_random_mask_with_missing_mask() -> None:
    """Test that missing_mask in OlmoEarthSample is respected during masking."""
    b, h, w, t = 5, 8, 8, 4

    # Create a sample with sentinel1 data where some samples are missing
    sentinel1 = torch.ones((b, h, w, t, 2))  # 2 bands for simplicity

    # Create a missing mask for sentinel1 where half the batch is missing
    sentinel1[b // 2 :] = MISSING_VALUE

    # Create the OlmoEarthSample
    days = torch.randint(1, 31, (b, 1, t), dtype=torch.long)
    months = torch.randint(1, 13, (b, 1, t), dtype=torch.long)
    years = torch.randint(2018, 2020, (b, 1, t), dtype=torch.long)
    timestamps = torch.cat([days, months, years], dim=1)

    batch = OlmoEarthSample(
        sentinel2_l2a=torch.ones((b, h, w, t, 12)),  # 12 bands for sentinel2
        sentinel1=sentinel1,
        timestamps=timestamps,
    )

    # Apply random masking
    encode_ratio, decode_ratio = 0.25, 0.5
    masked_sample = RandomMaskingStrategy(
        encode_ratio=encode_ratio, decode_ratio=decode_ratio
    ).apply_mask(batch, patch_size=1)

    # Check the sentinel1 mask
    sentinel1_mask = masked_sample.sentinel1_mask
    assert sentinel1_mask is not None

    # For non-missing samples, check the ratios
    non_missing_indices = torch.where(sentinel1 != MISSING_VALUE)[0]
    for idx in non_missing_indices:
        mask_slice = sentinel1_mask[idx]
        total_elements = mask_slice.numel()

        num_encoder = (mask_slice == MaskValue.ONLINE_ENCODER.value).sum().item()
        num_decoder = (mask_slice == MaskValue.DECODER.value).sum().item()
        num_target = (mask_slice == MaskValue.TARGET_ENCODER_ONLY.value).sum().item()

        # Check with tolerance for rounding
        assert abs(num_encoder / total_elements - encode_ratio) < 0.05, (
            "Encoder ratio incorrect for non-missing samples"
        )
        assert abs(num_decoder / total_elements - decode_ratio) < 0.05, (
            "Decoder ratio incorrect for non-missing samples"
        )
        assert (
            abs(num_target / total_elements - (1 - encode_ratio - decode_ratio)) < 0.05
        ), "Target ratio incorrect for non-missing samples"

    # Check that missing samples have the missing value
    missing_indices = torch.where(sentinel1 == MISSING_VALUE)[0]
    for idx in missing_indices:
        mask_slice = sentinel1_mask[idx]
        # All values for missing samples should be set to MaskValue.MISSING.value
        assert (mask_slice == MaskValue.MISSING.value).all(), (
            f"Missing sample {idx} should have all mask values set to MISSING"
        )


def test_create_spatial_mask_with_patch_size() -> None:
    """Test the _create_patch_spatial_mask function with different patch sizes."""
    b = 4
    h, w = 16, 16
    shape = (b, h, w)
    patch_size = 4

    encode_ratio, decode_ratio = 0.25, 0.5
    strategy = SpaceMaskingStrategy(
        encode_ratio=encode_ratio, decode_ratio=decode_ratio
    )

    # Call the _create_patch_spatial_mask function directly
    patch_mask = strategy._create_patch_spatial_mask(
        modality=Modality.SENTINEL2_L2A, shape=shape, patch_size_at_16=patch_size
    )
    mask = strategy._resize_spatial_mask_for_modality(
        patch_mask,
        modality=Modality.SENTINEL2_L2A,
        patch_size_at_16=patch_size,
    )

    # Check that the mask has the right shape
    assert mask.shape == shape, "Mask shape should match the input shape"

    # Check that patches have consistent values
    for i in range(0, h, patch_size):
        for j in range(0, w, patch_size):
            for b_idx in range(b):
                patch = mask[b_idx, i : i + patch_size, j : j + patch_size]
                # All values within a patch should be the same
                assert (patch == patch[0, 0]).all(), (
                    f"Patch at ({b_idx},{i},{j}) has inconsistent values"
                )

    # Check the ratios across all values
    total_elements = mask.numel()
    num_encoder = len(mask[mask == MaskValue.ONLINE_ENCODER.value])
    num_decoder = len(mask[mask == MaskValue.DECODER.value])
    num_target = len(mask[mask == MaskValue.TARGET_ENCODER_ONLY.value])

    assert num_encoder / total_elements == encode_ratio, "Incorrect encode mask ratio"
    assert num_decoder / total_elements == decode_ratio, "Incorrect decode mask ratio"
    assert num_target / total_elements == 1 - encode_ratio - decode_ratio, (
        "Incorrect target mask ratio"
    )


def test_create_temporal_mask() -> None:
    """Test the _create_temporal_mask function."""
    b = 10
    t = 8
    shape = (b, t)

    encode_ratio, decode_ratio = 0.25, 0.5
    strategy = TimeMaskingStrategy(encode_ratio=encode_ratio, decode_ratio=decode_ratio)

    # Call the _create_temporal_mask function directly
    timesteps_with_at_least_one_modality = torch.tensor(list(range(t)))
    mask = strategy._create_temporal_mask(
        shape=shape,
        timesteps_with_at_least_one_modality=timesteps_with_at_least_one_modality,
    )

    # Check the masking ratios for non-missing timesteps

    total_non_missing = mask.numel()
    num_encoder = len(mask[mask == MaskValue.ONLINE_ENCODER.value])
    num_decoder = len(mask[mask == MaskValue.DECODER.value])
    num_target = len(mask[mask == MaskValue.TARGET_ENCODER_ONLY.value])

    # Check that the ratios are close to expected for non-missing values
    # Note: With small values of t, the ratios might not be exactly as expected
    assert abs(num_encoder / total_non_missing - encode_ratio) < 0.2, (
        "Encode mask ratio too far from expected"
    )
    assert abs(num_decoder / total_non_missing - decode_ratio) < 0.2, (
        "Decode mask ratio too far from expected"
    )
    assert (
        abs(num_target / total_non_missing - (1 - encode_ratio - decode_ratio)) < 0.2
    ), "Target mask ratio too far from expected"


def test_space_masking_with_missing_modality_mask() -> None:
    """Test SpaceMaskingStrategy with missing_modalities_masks."""
    b, h, w, t = 4, 8, 8, 8

    days = torch.randint(1, 31, (b, 1, t), dtype=torch.long)
    months = torch.randint(1, 13, (b, 1, t), dtype=torch.long)
    years = torch.randint(2018, 2020, (b, 1, t), dtype=torch.long)
    timestamps = torch.cat([days, months, years], dim=1)  # Shape: (B, 3, T)

    # Include all modalities but mark some sentinel1 samples as missing
    sentinel2_l2a_num_bands = Modality.SENTINEL2_L2A.num_bands
    sentinel1_num_bands = Modality.SENTINEL1.num_bands
    worldcover_num_bands = Modality.WORLDCOVER.num_bands
    latlon_num_bands = Modality.LATLON.num_bands

    # Create a missing mask for sentinel1 where half the batch is missing
    sentinel1 = torch.ones((b, h, w, t, sentinel1_num_bands))
    sentinel1[b // 2 :] = MISSING_VALUE

    # Create the OlmoEarthSample with missing_modalities_masks
    batch = OlmoEarthSample(
        sentinel2_l2a=torch.ones((b, h, w, t, sentinel2_l2a_num_bands)),
        sentinel1=sentinel1,
        latlon=torch.ones((b, latlon_num_bands)),
        timestamps=timestamps,
        worldcover=torch.ones((b, h, w, worldcover_num_bands)),
    )

    # Test the SpaceMaskingStrategy
    encode_ratio, decode_ratio = 0.25, 0.5
    strategy = SpaceMaskingStrategy(
        encode_ratio=encode_ratio, decode_ratio=decode_ratio
    )

    # Apply masking
    masked_sample = strategy.apply_mask(batch, patch_size=4)

    # Check that sentinel1_mask has been created
    sentinel1_mask = masked_sample.sentinel1_mask
    assert sentinel1_mask is not None, "sentinel1_mask should not be None"

    # Verify masking was applied correctly to non-missing samples:
    present_indices = torch.where(sentinel1 != MISSING_VALUE)[0]
    for idx in present_indices:
        mask_slice = sentinel1_mask[idx]
        total_elements = mask_slice.numel()

        # Count occurrences of each mask value
        num_encoder = (mask_slice == MaskValue.ONLINE_ENCODER.value).sum().item()
        num_decoder = (mask_slice == MaskValue.DECODER.value).sum().item()
        num_target = (mask_slice == MaskValue.TARGET_ENCODER_ONLY.value).sum().item()

        # Check that the mask values are distributed according to the ratios
        # with some tolerance for rounding
        assert abs(num_encoder / total_elements - encode_ratio) < 0.05, (
            "Incorrect encode mask ratio for present samples"
        )
        assert abs(num_decoder / total_elements - decode_ratio) < 0.05, (
            "Incorrect decode mask ratio for present samples"
        )
        assert (
            abs(num_target / total_elements - (1 - encode_ratio - decode_ratio)) < 0.05
        ), "Incorrect target mask ratio for present samples"

    # Check that masks for spatial data have consistent values within patches
    for idx in present_indices:
        for i in range(0, h, 4):
            for j in range(0, w, 4):
                for t_idx in range(t):
                    patch = sentinel1_mask[idx, i : i + 4, j : j + 4, t_idx]
                    # All values within a patch should be the same
                    assert (patch == patch[0, 0]).all(), (
                        f"Patch at ({idx},{i},{j},{t_idx}) has inconsistent values"
                    )

    # Test unmasking
    unmasked_sample = masked_sample.unmask()

    # Check that sentinel1_mask was created
    unmasked_sentinel1_mask = unmasked_sample.sentinel1_mask
    assert unmasked_sentinel1_mask is not None

    # Check that non-missing samples have been set to ONLINE_ENCODER
    for idx in present_indices:
        # All non-missing values should be set to ONLINE_ENCODER (0)
        assert (unmasked_sentinel1_mask[idx] == MaskValue.ONLINE_ENCODER.value).all(), (
            "Unmasked should be ONLINE_ENCODER for present samples"
        )


def test_time_masking_with_missing_modality_mask() -> None:
    """Test TimeMaskingStrategy with missing_modalities_masks."""
    b, h, w, t = 4, 8, 8, 8

    days = torch.randint(1, 31, (b, 1, t), dtype=torch.long)
    months = torch.randint(1, 13, (b, 1, t), dtype=torch.long)
    years = torch.randint(2018, 2020, (b, 1, t), dtype=torch.long)
    timestamps = torch.cat([days, months, years], dim=1)  # Shape: (B, 3, T)

    # Include all modalities but mark some sentinel1 samples as missing
    sentinel2_l2a_num_bands = Modality.SENTINEL2_L2A.num_bands
    sentinel1_num_bands = Modality.SENTINEL1.num_bands
    worldcover_num_bands = Modality.WORLDCOVER.num_bands
    latlon_num_bands = Modality.LATLON.num_bands

    # Create a missing mask for sentinel1 where half the batch is missing
    sentinel1 = torch.ones((b, h, w, t, sentinel1_num_bands))
    sentinel1[b // 2 :] = MISSING_VALUE

    # Create the OlmoEarthSample with missing_modalities_masks
    batch = OlmoEarthSample(
        sentinel2_l2a=torch.ones((b, h, w, t, sentinel2_l2a_num_bands)),
        sentinel1=sentinel1,
        latlon=torch.ones((b, latlon_num_bands)),
        timestamps=timestamps,
        worldcover=torch.ones((b, h, w, worldcover_num_bands)),
    )

    # Test the TimeMaskingStrategy
    encode_ratio, decode_ratio = 0.25, 0.5
    strategy = TimeMaskingStrategy(encode_ratio=encode_ratio, decode_ratio=decode_ratio)

    # Apply masking
    masked_sample = strategy.apply_mask(batch, patch_size=4)

    # Check that sentinel1_mask has been created
    sentinel1_mask = masked_sample.sentinel1_mask
    assert sentinel1_mask is not None, "sentinel1_mask should not be None"

    # Verify masking was applied correctly to non-missing samples:
    present_indices = torch.where(sentinel1 != MISSING_VALUE)[0]
    for idx in present_indices:
        mask_slice = sentinel1_mask[idx]
        total_elements = mask_slice.numel()

        # Count occurrences of each mask value
        num_encoder = (mask_slice == MaskValue.ONLINE_ENCODER.value).sum().item()
        num_decoder = (mask_slice == MaskValue.DECODER.value).sum().item()
        num_target = (mask_slice == MaskValue.TARGET_ENCODER_ONLY.value).sum().item()

        # Check that the mask values are distributed according to the ratios
        # with some tolerance for rounding
        assert abs(num_encoder / total_elements - encode_ratio) < 0.05, (
            "Incorrect encode mask ratio for present samples"
        )
        assert abs(num_decoder / total_elements - decode_ratio) < 0.05, (
            "Incorrect decode mask ratio for present samples"
        )
        assert (
            abs(num_target / total_elements - (1 - encode_ratio - decode_ratio)) < 0.05
        ), "Incorrect target mask ratio for present samples"

    # Check that missing samples are set to MISSING
    missing_indices = torch.where(sentinel1 == MISSING_VALUE)[0]
    for idx in missing_indices:
        assert (sentinel1_mask[idx] == MaskValue.MISSING.value).all(), (
            f"Sample {idx} should be set to MISSING"
        )

    # Test unmasking
    unmasked_sample = masked_sample.unmask()

    # Check that sentinel1_mask was created
    unmasked_sentinel1_mask = unmasked_sample.sentinel1_mask
    assert unmasked_sentinel1_mask is not None

    # Check that non-missing samples have been set to ONLINE_ENCODER
    for idx in present_indices:
        # All non-missing values should be set to ONLINE_ENCODER (0)
        assert (unmasked_sentinel1_mask[idx] == MaskValue.ONLINE_ENCODER.value).all(), (
            "Unmasked should be ONLINE_ENCODER for present samples"
        )


def test_random_masking_with_missing_modality_mask() -> None:
    """Test RandomMaskingStrategy with missing_modalities_masks."""
    b, h, w, t = 10, 16, 16, 8

    days = torch.randint(1, 31, (b, 1, t), dtype=torch.long)
    months = torch.randint(1, 13, (b, 1, t), dtype=torch.long)
    years = torch.randint(2018, 2020, (b, 1, t), dtype=torch.long)
    timestamps = torch.cat([days, months, years], dim=1)  # Shape: (B, 3, T)

    # Include all modalities but mark some sentinel1 samples as missing
    sentinel2_l2a_num_bands = Modality.SENTINEL2_L2A.num_bands
    sentinel1_num_bands = Modality.SENTINEL1.num_bands
    worldcover_num_bands = Modality.WORLDCOVER.num_bands
    latlon_num_bands = Modality.LATLON.num_bands

    # Create a missing mask for sentinel1 where half the batch is missing
    sentinel1 = torch.ones((b, h, w, t, sentinel1_num_bands))
    sentinel1[b // 2 :] = MISSING_VALUE

    # Create the OlmoEarthSample with missing_modalities_masks
    batch = OlmoEarthSample(
        sentinel2_l2a=torch.ones((b, h, w, t, sentinel2_l2a_num_bands)),
        sentinel1=sentinel1,
        latlon=torch.ones((b, latlon_num_bands)),
        timestamps=timestamps,
        worldcover=torch.ones((b, h, w, worldcover_num_bands)),
    )

    # Test the RandomMaskingStrategy
    encode_ratio, decode_ratio = 0.25, 0.5
    strategy = RandomMaskingStrategy(
        encode_ratio=encode_ratio, decode_ratio=decode_ratio
    )

    # Apply masking
    masked_sample = strategy.apply_mask(batch, patch_size=1)

    # Check that sentinel1_mask has been created
    sentinel1_mask = masked_sample.sentinel1_mask
    assert sentinel1_mask is not None, "sentinel1_mask should not be None"

    # Note: The current implementation might not set missing samples to
    # MaskValue.MISSING.value, as it depends on how missing_mask is used
    # in the _create_random_mask function.

    # Verify masking was applied correctly to non-missing samples:
    present_indices = torch.where(sentinel1 != MISSING_VALUE)[0]
    for idx in present_indices:
        mask_slice = sentinel1_mask[idx]
        total_elements = mask_slice.numel()

        # Count occurrences of each mask value
        num_encoder = (mask_slice == MaskValue.ONLINE_ENCODER.value).sum().item()
        num_decoder = (mask_slice == MaskValue.DECODER.value).sum().item()
        num_target = (mask_slice == MaskValue.TARGET_ENCODER_ONLY.value).sum().item()

        # Check that the mask values are distributed according to the ratios
        # with some tolerance for rounding
        assert abs(num_encoder / total_elements - encode_ratio) < 0.05, (
            "Incorrect encode mask ratio for present samples"
        )
        assert abs(num_decoder / total_elements - decode_ratio) < 0.05, (
            "Incorrect decode mask ratio for present samples"
        )
        assert (
            abs(num_target / total_elements - (1 - encode_ratio - decode_ratio)) < 0.05
        ), "Incorrect target mask ratio for present samples"

    # Test unmasking
    unmasked_sample = masked_sample.unmask()

    # Check that sentinel1_mask was created
    unmasked_sentinel1_mask = unmasked_sample.sentinel1_mask
    assert unmasked_sentinel1_mask is not None

    # Check that non-missing samples have been set to ONLINE_ENCODER
    for idx in present_indices:
        # All non-missing values should be set to ONLINE_ENCODER (0)
        assert (unmasked_sentinel1_mask[idx] == MaskValue.ONLINE_ENCODER.value).all(), (
            "Unmasked should be ONLINE_ENCODER for present samples"
        )


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
    batch = OlmoEarthSample(
        sentinel2_l2a=torch.ones((b, h, w, t, sentinel2_l2a_num_bands)),
        latlon=torch.ones((b, latlon_num_bands)),
        timestamps=timestamps,
        worldcover=torch.ones((b, h, w, worldcover_num_bands)),
    )
    # count all modalities except timestamps
    total_modalities = len(batch.modalities) - 1
    encode_ratio, decode_ratio = 0.25, 0.5
    masked_sample = ModalityMaskingStrategy(
        encode_ratio=encode_ratio,
        decode_ratio=decode_ratio,
    ).apply_mask(
        batch,
    )

    mask_per_modality: list[torch.Tensor] = []  # each tensor will have shape [b, 1]
    for modality_name in masked_sample._fields:
        logger.info(f"Modality name: {modality_name}")
        if modality_name.endswith("mask"):
            unmasked_modality_name = masked_sample.get_unmasked_modality_name(
                modality_name
            )
            modality = Modality.get(unmasked_modality_name)
            mask = getattr(masked_sample, modality_name)
            data = getattr(masked_sample, unmasked_modality_name)
            logger.info(f"Mask name: {modality_name}")
            if mask is None:
                continue

            # check all elements in the mask are the same, per instance in
            # the batch
            flat_mask = torch.flatten(mask, start_dim=1)
            unique_per_instance = torch.unique(flat_mask, dim=1)
            assert unique_per_instance.size(1) == 1
            mask_per_modality.append(unique_per_instance)

            assert mask.shape[:-1] == data.shape[:-1], (
                f"{modality_name} has incorrect shape"
            )
            assert mask.shape[-1] == modality.num_band_sets, (
                f"{modality_name} has incorrect num band sets"
            )

    # shape [b, num_modalities]
    total_mask = torch.concat(mask_per_modality, dim=-1)
    total_elements = total_mask.numel()
    num_encoder = len(total_mask[total_mask == MaskValue.ONLINE_ENCODER.value])
    num_decoder = len(total_mask[total_mask == MaskValue.DECODER.value])

    expected_encoded_modalities = max(1, int(total_modalities * encode_ratio))
    expected_decoded_modalities = max(1, int(total_modalities * decode_ratio))

    expected_encode_ratio = expected_encoded_modalities / total_modalities
    expected_decode_ratio = expected_decoded_modalities / total_modalities
    assert (num_encoder / total_elements) == expected_encode_ratio, (
        "Incorrect encode mask ratio"
    )
    assert (num_decoder / total_elements) == expected_decode_ratio, (
        "Incorrect decode mask ratio"
    )


def test_random_range_masking() -> None:
    """Test random range masking."""
    b, h, w, t = 100, 16, 16, 8
    patch_size = 4

    days = torch.randint(1, 31, (b, 1, t), dtype=torch.long)
    months = torch.randint(1, 13, (b, 1, t), dtype=torch.long)
    years = torch.randint(2018, 2020, (b, 1, t), dtype=torch.long)
    timestamps = torch.cat([days, months, years], dim=1)  # Shape: (B, 3, T)
    sentinel2_l2a_num_bands = Modality.SENTINEL2_L2A.num_bands
    worldcover_num_bands = Modality.WORLDCOVER.num_bands
    latlon_num_bands = Modality.LATLON.num_bands
    batch = OlmoEarthSample(
        sentinel2_l2a=torch.ones((b, h, w, t, sentinel2_l2a_num_bands)),
        latlon=torch.ones((b, latlon_num_bands)),
        timestamps=timestamps,
        worldcover=torch.ones((b, h, w, 1, worldcover_num_bands)),
    )
    min_encode_ratio = 0.4
    max_encode_ratio = 0.9
    masked_sample = RandomRangeMaskingStrategy(
        min_encode_ratio=min_encode_ratio,
        max_encode_ratio=max_encode_ratio,
    ).apply_mask(
        batch,
        patch_size=patch_size,
    )
    # Check that all values in the first patch are the same (consistent masking)
    assert masked_sample.sentinel2_l2a_mask is not None
    first_patch: torch.Tensor = masked_sample.sentinel2_l2a_mask[0, :4, :4, 0, 0]
    first_value: int = first_patch[0, 0]
    assert (first_patch == first_value).all()
    second_patch: torch.Tensor = masked_sample.sentinel2_l2a_mask[0, :4, :4, 1, 0]
    second_value: int = second_patch[0, 0]
    assert (second_patch == second_value).all()
    worldcover_patch: torch.Tensor = masked_sample.worldcover_mask[0, :4, :4, 0]  # type: ignore
    worldcover_value: int = worldcover_patch[0, 0]
    assert (worldcover_patch == worldcover_value).all()
    # Check that the distribution of masking ratios is roughly correct.
    encode_ratios = []
    decode_ratios = []
    for example_idx in range(b):
        mask = masked_sample.sentinel2_l2a_mask[example_idx]
        total_elements = mask.numel()
        num_encoder = len(mask[mask == MaskValue.ONLINE_ENCODER.value])
        num_decoder = len(mask[mask == MaskValue.DECODER.value])
        encode_ratios.append(num_encoder / total_elements)
        decode_ratios.append(num_decoder / total_elements)
    eps = 0.02
    assert min_encode_ratio - eps <= min(encode_ratios) < min_encode_ratio + 0.1
    assert max_encode_ratio + eps >= max(encode_ratios) > max_encode_ratio - 0.1
    min_decode_ratio = 1 - max_encode_ratio
    max_decode_ratio = 1 - min_encode_ratio
    assert min_decode_ratio - eps <= min(decode_ratios) < min_decode_ratio + 0.1
    assert max_decode_ratio + eps >= max(decode_ratios) > max_decode_ratio - 0.1


def test_space_cross_modality_masking(set_random_seeds: None) -> None:
    """Test space cross modality masking."""
    b, h, w, t = 100, 4, 4, 3

    patch_size = 1

    days = torch.randint(1, 31, (b, 1, t), dtype=torch.long)
    months = torch.randint(1, 13, (b, 1, t), dtype=torch.long)
    years = torch.randint(2018, 2020, (b, 1, t), dtype=torch.long)
    timestamps = torch.cat([days, months, years], dim=1)  # Shape: (B, 3, T)
    sentinel2_l2a_num_bands = Modality.SENTINEL2_L2A.num_bands
    worldcover_num_bands = Modality.WORLDCOVER.num_bands
    latlon_num_bands = Modality.LATLON.num_bands
    batch = OlmoEarthSample(
        sentinel2_l2a=torch.ones((b, h, w, t, sentinel2_l2a_num_bands)),
        sentinel1=torch.ones((b, h, w, t, Modality.SENTINEL1.num_bands)),
        latlon=torch.ones((b, latlon_num_bands)),
        timestamps=timestamps,
        worldcover=torch.ones((b, h, w, 1, worldcover_num_bands)),
    )

    strategy = ModalityCrossSpaceMaskingStrategy(
        encode_ratio=0.5,
        decode_ratio=0.5,
        allow_encoding_decoding_same_bandset=False,
    )
    masked_sample = strategy.apply_mask(batch, patch_size=patch_size)
    logger.info(f"masked_sample: {masked_sample}")

    # 50% of the latlons will be masked via space masking.
    # For the remaining 50%, cross modality will pick 2, 3, 4, or 5 band sets to encode.
    # We provided six band sets total.
    # This means 0.5 + 0.5 * 1/4 (4/6 + 3/6 + 2/6 + 1/6) = 0.7083 of latlons should be masked.
    latlon_masked_fraction = torch.count_nonzero(
        masked_sample.latlon_mask == MaskValue.DECODER.value
    ) / torch.numel(masked_sample.latlon_mask)
    assert 0.65 <= latlon_masked_fraction <= 0.75

    # We also verify that, for the first band of SENTINEL2_L2A, there should both be
    # samples where ~50% are encoded and 50% target encoder only, and ~50% decoded and
    # 50% target encoder only.
    # And nothing else.
    saw_encode_sample = False
    saw_decode_sample = False
    for sample_idx in range(b):
        assert masked_sample.sentinel2_l2a_mask is not None
        cur_mask = masked_sample.sentinel2_l2a_mask[sample_idx, :, :, :, 0]
        encode_fraction = torch.count_nonzero(
            cur_mask == MaskValue.ONLINE_ENCODER.value
        ) / torch.numel(cur_mask)
        decode_fraction = torch.count_nonzero(
            cur_mask == MaskValue.DECODER.value
        ) / torch.numel(cur_mask)
        target_encoder_fraction = torch.count_nonzero(
            cur_mask == MaskValue.TARGET_ENCODER_ONLY.value
        ) / torch.numel(cur_mask)
        if 0.4 <= encode_fraction <= 0.6 and 0.4 <= target_encoder_fraction <= 0.6:
            saw_encode_sample = True
            continue
        elif 0.4 <= decode_fraction <= 0.6 and 0.4 <= target_encoder_fraction <= 0.6:
            saw_decode_sample = True
            continue
        else:
            raise AssertionError(
                f"got unexpected sample with encode_fraction={encode_fraction}, decode_fraction={decode_fraction}, target_encoder_fraction={target_encoder_fraction}"
            )
    assert saw_encode_sample
    assert saw_decode_sample


def test_space_cross_modality_masking_with_missing_data(set_random_seeds: None) -> None:
    """Test space cross modality masking."""
    b, h, w, t = 4, 4, 4, 3

    patch_size = 1

    days = torch.randint(1, 31, (b, 1, t), dtype=torch.long)
    months = torch.randint(1, 13, (b, 1, t), dtype=torch.long)
    years = torch.randint(2018, 2020, (b, 1, t), dtype=torch.long)
    timestamps = torch.cat([days, months, years], dim=1)  # Shape: (B, 3, T)
    sentinel2_l2a_num_bands = Modality.SENTINEL2_L2A.num_bands
    worldcover_num_bands = Modality.WORLDCOVER.num_bands
    latlon_num_bands = Modality.LATLON.num_bands
    batch = OlmoEarthSample(
        sentinel2_l2a=torch.ones((b, h, w, t, sentinel2_l2a_num_bands)),
        sentinel1=torch.ones((b, h, w, t, Modality.SENTINEL1.num_bands)),
        latlon=torch.ones((b, latlon_num_bands)),
        timestamps=timestamps,
        worldcover=torch.full((b, h, w, 1, worldcover_num_bands), MISSING_VALUE),
    )

    strategy_allow_false = ModalityCrossSpaceMaskingStrategy(
        encode_ratio=0.1,
        decode_ratio=0.75,
        allow_encoding_decoding_same_bandset=False,
    )
    strategy_allow_true = ModalityCrossSpaceMaskingStrategy(
        encode_ratio=0.1,
        decode_ratio=0.75,
        allow_encoding_decoding_same_bandset=True,
    )
    masked_sample_allow_false = strategy_allow_false.apply_mask(
        batch, patch_size=patch_size
    )
    masked_sample_allow_true = strategy_allow_true.apply_mask(
        batch, patch_size=patch_size
    )
    # Check that the worldcover mask has the expected values
    # Check that latlon mask has the expected values
    expected_latlon_mask = torch.tensor([[2], [2], [1], [2]])
    # Assert that the masks match the expected values
    assert (masked_sample_allow_false.worldcover_mask == MaskValue.MISSING.value).all()  # type: ignore
    assert torch.equal(masked_sample_allow_false.latlon_mask, expected_latlon_mask)

    # Compare the masks between two strategies
    assert not torch.equal(
        masked_sample_allow_false.sentinel2_l2a_mask,
        masked_sample_allow_true.sentinel2_l2a_mask,
    )
    assert not torch.equal(
        masked_sample_allow_false.sentinel1_mask,
        masked_sample_allow_true.sentinel1_mask,
    )


def test_modality_cross_random_masking() -> None:
    """Test modality cross random masking."""
    b, h, w, t = 4, 4, 4, 3

    patch_size = 1

    days = torch.randint(1, 31, (b, 1, t), dtype=torch.long)
    months = torch.randint(1, 13, (b, 1, t), dtype=torch.long)
    years = torch.randint(2018, 2020, (b, 1, t), dtype=torch.long)
    timestamps = torch.cat([days, months, years], dim=1)  # Shape: (B, 3, T)
    sentinel2_l2a_num_bands = Modality.SENTINEL2_L2A.num_bands
    worldcover_num_bands = Modality.WORLDCOVER.num_bands
    latlon_num_bands = Modality.LATLON.num_bands
    batch = OlmoEarthSample(
        sentinel2_l2a=torch.ones((b, h, w, t, sentinel2_l2a_num_bands)),
        sentinel1=torch.ones((b, h, w, t, Modality.SENTINEL1.num_bands)),
        latlon=torch.ones((b, latlon_num_bands)),
        timestamps=timestamps,
        worldcover=torch.full((b, h, w, 1, worldcover_num_bands), MISSING_VALUE),
    )

    masking_strategy = ModalityCrossRandomMaskingStrategy(
        encode_ratio=0.5,
        decode_ratio=0.5,
        allow_encoding_decoding_same_bandset=True,
        only_decode_modalities=[Modality.WORLDCOVER.name],
    )
    masked_sample = masking_strategy.apply_mask(batch, patch_size=patch_size)
    logger.info(f"masked_sample: {masked_sample.sentinel2_l2a_mask}")

    expected_sentinel2_l2a_mask = torch.tensor(
        [
            [
                [
                    [[0, 2, 0], [2, 2, 0], [2, 2, 2]],
                    [[2, 2, 2], [0, 2, 0], [0, 0, 0]],
                    [[2, 2, 2], [0, 0, 2], [0, 0, 2]],
                    [[0, 0, 2], [0, 0, 0], [2, 2, 0]],
                ],
                [
                    [[0, 0, 0], [0, 0, 0], [2, 0, 0]],
                    [[0, 2, 0], [2, 2, 0], [2, 0, 2]],
                    [[2, 2, 2], [2, 2, 2], [2, 2, 0]],
                    [[0, 0, 2], [0, 2, 2], [2, 0, 0]],
                ],
                [
                    [[2, 2, 0], [2, 0, 2], [2, 0, 2]],
                    [[0, 2, 0], [0, 0, 0], [0, 2, 0]],
                    [[2, 2, 2], [0, 2, 2], [2, 2, 2]],
                    [[2, 0, 2], [2, 0, 0], [2, 0, 2]],
                ],
                [
                    [[0, 0, 0], [2, 2, 0], [0, 0, 0]],
                    [[2, 0, 0], [2, 2, 0], [2, 0, 0]],
                    [[0, 2, 0], [2, 0, 0], [2, 0, 2]],
                    [[0, 0, 2], [2, 2, 2], [0, 0, 2]],
                ],
            ],
            [
                [
                    [[2, 1, 2], [2, 2, 2], [2, 1, 0]],
                    [[0, 2, 2], [0, 2, 2], [2, 2, 0]],
                    [[2, 1, 2], [2, 2, 2], [0, 1, 0]],
                    [[2, 1, 2], [2, 1, 2], [0, 1, 2]],
                ],
                [
                    [[2, 1, 2], [0, 2, 2], [0, 1, 2]],
                    [[2, 1, 0], [2, 2, 0], [0, 1, 0]],
                    [[2, 2, 2], [0, 2, 2], [0, 1, 0]],
                    [[0, 1, 0], [2, 2, 2], [2, 1, 0]],
                ],
                [
                    [[0, 1, 0], [0, 1, 2], [0, 2, 0]],
                    [[2, 2, 0], [0, 2, 2], [2, 1, 0]],
                    [[0, 1, 0], [0, 2, 2], [0, 2, 0]],
                    [[0, 2, 2], [2, 2, 0], [2, 2, 0]],
                ],
                [
                    [[0, 1, 2], [2, 2, 0], [0, 2, 2]],
                    [[0, 1, 0], [0, 1, 0], [2, 1, 0]],
                    [[0, 2, 2], [2, 1, 0], [0, 2, 2]],
                    [[2, 1, 0], [2, 2, 0], [2, 2, 2]],
                ],
            ],
            [
                [
                    [[1, 2, 0], [0, 2, 2], [0, 1, 2]],
                    [[0, 1, 2], [0, 2, 2], [1, 1, 2]],
                    [[0, 2, 2], [1, 1, 2], [1, 1, 2]],
                    [[1, 1, 2], [0, 1, 0], [1, 1, 0]],
                ],
                [
                    [[1, 1, 0], [0, 1, 0], [0, 2, 2]],
                    [[0, 2, 2], [1, 2, 0], [1, 1, 2]],
                    [[1, 1, 0], [1, 1, 0], [1, 2, 0]],
                    [[1, 1, 2], [0, 2, 2], [1, 1, 2]],
                ],
                [
                    [[0, 1, 0], [1, 2, 2], [0, 2, 0]],
                    [[0, 1, 2], [1, 1, 0], [1, 2, 0]],
                    [[1, 1, 2], [0, 1, 2], [1, 1, 0]],
                    [[0, 1, 0], [0, 2, 0], [1, 2, 0]],
                ],
                [
                    [[0, 1, 0], [1, 1, 2], [1, 2, 0]],
                    [[0, 2, 2], [1, 1, 0], [0, 2, 2]],
                    [[1, 1, 2], [0, 2, 0], [0, 2, 0]],
                    [[0, 2, 2], [0, 2, 2], [1, 2, 0]],
                ],
            ],
            [
                [
                    [[0, 0, 1], [1, 0, 1], [0, 0, 1]],
                    [[1, 0, 1], [1, 0, 1], [0, 0, 0]],
                    [[1, 0, 1], [1, 2, 1], [0, 0, 0]],
                    [[0, 2, 0], [1, 0, 1], [0, 2, 0]],
                ],
                [
                    [[0, 0, 0], [1, 2, 1], [0, 2, 0]],
                    [[0, 2, 0], [1, 2, 1], [0, 0, 0]],
                    [[0, 0, 1], [1, 0, 0], [1, 2, 1]],
                    [[0, 0, 0], [1, 0, 0], [0, 2, 1]],
                ],
                [
                    [[1, 0, 1], [0, 0, 1], [1, 2, 0]],
                    [[1, 2, 1], [1, 0, 1], [0, 2, 1]],
                    [[0, 2, 0], [0, 2, 1], [0, 0, 1]],
                    [[0, 2, 0], [1, 0, 1], [0, 0, 0]],
                ],
                [
                    [[1, 2, 1], [0, 2, 1], [0, 2, 0]],
                    [[0, 0, 1], [0, 2, 1], [0, 2, 1]],
                    [[0, 0, 1], [0, 2, 1], [0, 0, 1]],
                    [[1, 2, 1], [0, 0, 0], [1, 2, 0]],
                ],
            ],
        ]
    )
    # ensure none of the worldcover mask is encoded
    assert (masked_sample.worldcover_mask == MaskValue.ONLINE_ENCODER.value).sum() == 0  # type: ignore
    assert (masked_sample.sentinel2_l2a_mask == expected_sentinel2_l2a_mask).all()


def test_modality_cross_random_masking_has_online_encoder_and_decoder_tokens_many_missing() -> (
    None
):
    """Test modality cross random masking."""
    masking_strategy = ModalityCrossRandomMaskingStrategy(
        encode_ratio=0.5,
        decode_ratio=0.5,
        allow_encoding_decoding_same_bandset=True,
    )

    for _ in range(10):
        h_w = random.choice([1, 2, 3, 4, 5, 6])
        t = random.choice([1, 2, 3])
        b = 100
        patch_size = h_w

        days = torch.randint(1, 31, (b, 1, t), dtype=torch.long)
        months = torch.randint(1, 13, (b, 1, t), dtype=torch.long)
        years = torch.randint(2018, 2020, (b, 1, t), dtype=torch.long)
        timestamps = torch.cat([days, months, years], dim=1)  # Shape: (B, 3, T)
        batch = OlmoEarthSample(
            sentinel2_l2a=torch.ones(
                (b, h_w, h_w, t, Modality.SENTINEL2_L2A.num_bands)
            ),
            timestamps=timestamps,
        )

        masked_sample = masking_strategy.apply_mask(batch, patch_size=patch_size)
        logger.info(f"masked_sample: {masked_sample.sentinel2_l2a_mask}")
        num_encoded = torch.sum(
            masked_sample.sentinel2_l2a_mask == MaskValue.ONLINE_ENCODER.value,
            dim=(1, 2, 3, 4),
        )
        num_decoded = torch.sum(
            masked_sample.sentinel2_l2a_mask == MaskValue.DECODER.value,
            dim=(1, 2, 3, 4),
        )
        assert (num_encoded > 0).all()
        assert (num_decoded > 0).all()


def test_mask_when_most_samples_are_missing() -> None:
    """Test the following failure case no longer occurs.

    https://beaker.allen.ai/orgs/ai2/workspaces/earth-systems/work/01K796J483408TEV6S5THV7J6M
    """
    masking_strategy = ModalityCrossRandomMaskingStrategy(
        encode_ratio=0.5,
        decode_ratio=0.5,
        allow_encoding_decoding_same_bandset=True,
    )

    # this is a real example which triggered the following failure in the
    # beaker job linked above
    mask = torch.tensor(
        [
            [
                [
                    [
                        [2, 2, 2],
                        [2, 2, 2],
                        [3, 3, 3],
                        [3, 3, 3],
                        [3, 3, 3],
                        [3, 3, 3],
                        [3, 3, 3],
                        [3, 3, 3],
                        [3, 3, 3],
                        [3, 3, 3],
                        [3, 3, 3],
                        [3, 3, 3],
                    ]
                ]
            ]
        ]
    )
    filled_mask = masking_strategy._random_fill_unmasked(
        mask, modality=Modality.SENTINEL2_L2A, patch_size_at_16=1
    )
    num_encoded = torch.sum(
        filled_mask == MaskValue.ONLINE_ENCODER.value,
        dim=(1, 2, 3, 4),
    )
    num_decoded = torch.sum(
        filled_mask == MaskValue.DECODER.value,
        dim=(1, 2, 3, 4),
    )
    assert (num_encoded > 0).all()
    assert (num_decoded > 0).all()
    # also, check the original missing values are still missing
    assert (
        filled_mask[mask == MaskValue.MISSING.value] == MaskValue.MISSING.value
    ).all()


def test_modality_cross_random_masking_has_online_encoder_and_decoder_tokens() -> None:
    """Test modality cross random masking."""
    masking_strategy = ModalityCrossRandomMaskingStrategy(
        encode_ratio=0.5,
        decode_ratio=0.5,
        allow_encoding_decoding_same_bandset=True,
    )

    for _ in range(10):
        h_w = random.choice([1, 2, 3, 4, 5, 6])
        t = random.choice([1, 2, 3])
        b = 100
        patch_size = h_w

        days = torch.randint(1, 31, (b, 1, t), dtype=torch.long)
        months = torch.randint(1, 13, (b, 1, t), dtype=torch.long)
        years = torch.randint(2018, 2020, (b, 1, t), dtype=torch.long)
        timestamps = torch.cat([days, months, years], dim=1)  # Shape: (B, 3, T)
        batch = OlmoEarthSample(
            sentinel2_l2a=torch.ones(
                (b, h_w, h_w, t, Modality.SENTINEL2_L2A.num_bands)
            ),
            timestamps=timestamps,
        )

        masked_sample = masking_strategy.apply_mask(batch, patch_size=patch_size)
        logger.info(f"masked_sample: {masked_sample.sentinel2_l2a_mask}")
        num_encoded = torch.sum(
            masked_sample.sentinel2_l2a_mask == MaskValue.ONLINE_ENCODER.value,
            dim=(1, 2, 3, 4),
        )
        num_decoded = torch.sum(
            masked_sample.sentinel2_l2a_mask == MaskValue.DECODER.value,
            dim=(1, 2, 3, 4),
        )
        assert (num_encoded > 0).all()
        assert (num_decoded > 0).all()


class TestModalityCrossMaskingStrategy:
    """Test class for ModalityCrossMaskingStrategy."""

    def test_get_sample_present_modalities_bandsets(self) -> None:
        """Test get sample present modalities bandsets."""
        # 2 samples, 2 modalities, 2 bandsets each
        b, h, w, t = 2, 2, 2, 2
        s2_bands = Modality.SENTINEL2_L2A.num_bands
        wc_bands = Modality.WORLDCOVER.num_bands
        # All tokens encoded for s2, none for worldcover in sample 0, both encoded in sample 1
        s2_mask = torch.full((b, h, w, t, 2), MaskValue.ONLINE_ENCODER.value)
        wc_mask = torch.full((b, h, w, 1, 1), MaskValue.ONLINE_ENCODER.value)
        wc_mask[0] = MaskValue.MISSING.value  # worldcover missing for sample 0
        batch = MaskedOlmoEarthSample(
            sentinel2_l2a=torch.ones((b, h, w, t, s2_bands)),
            sentinel2_l2a_mask=s2_mask,
            worldcover=torch.ones((b, h, w, 1, wc_bands)),
            worldcover_mask=wc_mask,
            latlon=torch.ones((b, Modality.LATLON.num_bands)),
            latlon_mask=torch.full(
                (b, Modality.LATLON.num_bands), MaskValue.ONLINE_ENCODER.value
            ),
            timestamps=torch.ones((b, 3, t), dtype=torch.long),
        )
        strat = ModalityCrossRandomMaskingStrategy()
        expected_present = [
            [
                (Modality.SENTINEL2_L2A.name, 0),
                (Modality.SENTINEL2_L2A.name, 1),
                (Modality.LATLON.name, 0),
                (Modality.LATLON.name, 1),
            ],
            [
                (Modality.SENTINEL2_L2A.name, 0),
                (Modality.SENTINEL2_L2A.name, 1),
                (Modality.WORLDCOVER.name, 0),
                (Modality.LATLON.name, 0),
                (Modality.LATLON.name, 1),
            ],
        ]
        present = strat.get_sample_present_modalities_bandsets(batch)
        assert present == expected_present

    def test_get_sample_present_modalities_bandsets_no_encoded(self) -> None:
        """Test get sample present modalities bandsets with no encoded tokens."""
        # If a modality has no encoded tokens for a sample, it should not be present
        # Test with batch size 2: sample 0 has no encoded tokens, sample 1 does

        b, h, w, t = 2, 2, 2, 2
        s2_bands = Modality.SENTINEL2_L2A.num_bands
        # sample 0: all decoder, sample 1: all encoder
        s2_mask = torch.full((b, h, w, t, 2), MaskValue.DECODER.value)
        s2_mask[1] = MaskValue.ONLINE_ENCODER.value
        batch = MaskedOlmoEarthSample(
            sentinel2_l2a=torch.ones((b, h, w, t, s2_bands)),
            sentinel2_l2a_mask=s2_mask,
            worldcover=torch.ones((b, h, w, 1, Modality.WORLDCOVER.num_bands)),
            worldcover_mask=torch.full((b, h, w, 1, 1), MaskValue.ONLINE_ENCODER.value),
            latlon=torch.ones((b, Modality.LATLON.num_bands)),
            latlon_mask=torch.full(
                (b, Modality.LATLON.num_bands), MaskValue.ONLINE_ENCODER.value
            ),
            timestamps=torch.ones((b, 3, t), dtype=torch.long),
        )
        strat = ModalityCrossRandomMaskingStrategy()
        present = strat.get_sample_present_modalities_bandsets(batch)
        logger.info(f"present: {present}")
        expected_present = [
            [
                (Modality.WORLDCOVER.name, 0),
                (Modality.LATLON.name, 0),
                (Modality.LATLON.name, 1),
            ],
            [
                (Modality.SENTINEL2_L2A.name, 0),
                (Modality.SENTINEL2_L2A.name, 1),
                (Modality.WORLDCOVER.name, 0),
                (Modality.LATLON.name, 0),
                (Modality.LATLON.name, 1),
            ],
        ]
        assert expected_present == present

    def test_sample_present_with_missing_for_member(self) -> None:
        """Test sample present with missing for member."""
        b, h, w, t = 2, 2, 2, 2
        s2_bands = Modality.SENTINEL2_L2A.num_bands
        wc_bands = Modality.WORLDCOVER.num_bands
        s2_mask = torch.full((b, h, w, t, 2), MaskValue.ONLINE_ENCODER.value)
        wc_mask = torch.full((b, h, w, 1, 1), MaskValue.ONLINE_ENCODER.value)
        wc_mask[0] = MaskValue.MISSING.value  # missing for sample 0
        batch = MaskedOlmoEarthSample(
            sentinel2_l2a=torch.ones((b, h, w, t, s2_bands)),
            sentinel2_l2a_mask=s2_mask,
            worldcover=torch.ones((b, h, w, 1, wc_bands)),
            worldcover_mask=wc_mask,
            latlon=torch.ones((b, Modality.LATLON.num_bands)),
            latlon_mask=torch.full(
                (b, Modality.LATLON.num_bands), MaskValue.ONLINE_ENCODER.value
            ),
            timestamps=torch.ones((b, 3, t), dtype=torch.long),
        )
        strat = ModalityCrossRandomMaskingStrategy()
        present = strat.get_sample_present_modalities_bandsets(batch)
        expected_present = [
            [
                (Modality.SENTINEL2_L2A.name, 0),
                (Modality.SENTINEL2_L2A.name, 1),
                (Modality.LATLON.name, 0),
                (Modality.LATLON.name, 1),
            ],
            [
                (Modality.SENTINEL2_L2A.name, 0),
                (Modality.SENTINEL2_L2A.name, 1),
                (Modality.WORLDCOVER.name, 0),
                (Modality.LATLON.name, 0),
                (Modality.LATLON.name, 1),
            ],
        ]
        assert expected_present == present

    def test_select_encoded_decoded_bandsets(self) -> None:
        """Test select encoded decoded bandsets."""
        present_modalities_bandsets = [
            [
                (Modality.SENTINEL2_L2A.name, 0),
                (Modality.SENTINEL2_L2A.name, 1),
                (Modality.SENTINEL2_L2A.name, 2),
                (Modality.WORLDCOVER.name, 0),
                (Modality.LATLON.name, 0),
            ],
        ] * 64
        strat = ModalityCrossRandomMaskingStrategy(
            allow_encoding_decoding_same_bandset=False
        )
        encoded_decoded_bandsets = strat.select_encoded_decoded_bandsets(
            present_modalities_bandsets
        )
        # Make sure all the different numbers of encode band sets are captured.
        # Could be 2, 3, or 4.
        # 1 encoded band set should not occur, since ModalityCrossMaskingStrategy enforces
        # it to be at least 2 so that we don't try to encode only latlon (which could get
        # masked entirely for some samples since it is masked on batch dimension).
        # 5 encoded band sets should not occur, since we need at least one decode band set
        # with allow_encoding_decoding_same_bandset is False.
        counts = {
            len(encoded_decoded_tuple[0])
            for encoded_decoded_tuple in encoded_decoded_bandsets
        }
        assert counts == {2, 3, 4}

    def test_select_encoded_decoded_bandsets_only_decode_modalities(self) -> None:
        """Test select encoded decoded bandsets with only decode modalities."""
        present_modalities_bandsets = [
            [
                (Modality.SENTINEL2_L2A.name, 0),
                (Modality.SENTINEL2_L2A.name, 1),
                (Modality.SENTINEL2_L2A.name, 2),
                (Modality.SENTINEL1.name, 0),
                (Modality.LATLON.name, 0),
                (Modality.OPENSTREETMAP_RASTER.name, 0),
            ],
            [
                (Modality.SENTINEL2_L2A.name, 0),
                (Modality.SENTINEL2_L2A.name, 1),
                (Modality.SENTINEL2_L2A.name, 2),
                (Modality.WORLDCOVER.name, 0),
                (Modality.SENTINEL1.name, 0),
                (Modality.LATLON.name, 0),
            ],
        ]
        strat = ModalityCrossRandomMaskingStrategy(
            only_decode_modalities=[Modality.WORLDCOVER.name]
        )
        encoded_decoded_bandsets = strat.select_encoded_decoded_bandsets(
            present_modalities_bandsets
        )
        logger.info(f"encoded_decoded_bandsets: {encoded_decoded_bandsets}")
        # WorldCover should not be encoded for either sample since it is only decode.
        assert (Modality.WORLDCOVER.name, 0) not in encoded_decoded_bandsets[0][0]
        assert (Modality.WORLDCOVER.name, 0) not in encoded_decoded_bandsets[1][0]
        # WorldCover should not be decoded for first sample since it isn't present.
        assert (Modality.WORLDCOVER.name, 0) not in encoded_decoded_bandsets[0][1]
        # WorldCover should be decoded for second sample.
        assert (Modality.WORLDCOVER.name, 0) in encoded_decoded_bandsets[1][1]

    def test_select_encoded_decoded_bandsets_no_overlap(self) -> None:
        """Test select encoded decoded bandsets with no overlap."""
        present_modalities_bandsets = [
            [
                (Modality.SENTINEL2_L2A.name, 0),
                (Modality.SENTINEL2_L2A.name, 1),
                (Modality.SENTINEL2_L2A.name, 2),
                (Modality.WORLDCOVER.name, 0),
                (Modality.SENTINEL1.name, 0),
                (Modality.LATLON.name, 0),
            ],
        ] * 64
        strat = ModalityCrossRandomMaskingStrategy(
            allow_encoding_decoding_same_bandset=False
        )
        encoded_decoded_bandsets = strat.select_encoded_decoded_bandsets(
            present_modalities_bandsets
        )

        # Now we should see 2, 3, 4, and 5 band sets being encoded in different samples.
        # 5 band sets encoded is possible since we can decode portions of each one.
        counts = {
            len(encoded_decoded_tuple[0])
            for encoded_decoded_tuple in encoded_decoded_bandsets
        }
        assert counts == {2, 3, 4, 5}
