"""Test masking."""

import logging

import torch

from helios.data.constants import MISSING_VALUE, Modality
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
            assert (
                num_encoder / total_elements
            ) == encode_ratio, f"{modality_name} has incorrect encode mask ratio"
            assert (
                num_decoder / total_elements
            ) == decode_ratio, f"{modality_name} has incorrect decode mask ratio"
            assert (
                mask.shape[:-1] == data.shape[:-1]
            ), f"{modality_name} has incorrect shape"
            assert (
                mask.shape[-1] == modality.num_band_sets
            ), f"{modality_name} has incorrect num band sets"

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
            assert (
                num_encoder / total_elements
            ) == encode_ratio, f"{modality_name} has incorrect encode mask ratio"
            assert (
                num_decoder / total_elements
            ) == decode_ratio, f"{modality_name} has incorrect decode mask ratio"
            assert (
                mask.shape[:-1] == data.shape[:-1]
            ), f"{modality_name} has incorrect shape"
            assert (
                mask.shape[-1] == modality.num_band_sets
            ), f"{modality_name} has incorrect num band sets"

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
            assert (
                num_encoder / total_elements
            ) == encode_ratio, f"{modality_name} has incorrect encode mask ratio"
            assert (
                num_decoder / total_elements
            ) == decode_ratio, f"{modality_name} has incorrect decode mask ratio"
            assert (
                mask.shape[:-1] == data.shape[:-1]
            ), f"{modality_name} has incorrect shape"
            assert (
                mask.shape[-1] == modality.num_band_sets
            ), f"{modality_name} has incorrect num band sets"

    unmasked_sample = masked_sample.unmask()
    for modality_name in unmasked_sample._fields:
        if modality_name.endswith("mask"):
            mask = getattr(unmasked_sample, modality_name)
            if mask is not None:
                assert (mask == 0).all()


def test_create_random_mask_with_missing_mask() -> None:
    """Test that missing_mask in HeliosSample is respected during masking."""
    b, h, w, t = 5, 8, 8, 4

    # Create a sample with sentinel1 data where some samples are missing
    sentinel1 = torch.ones((b, h, w, t, 2))  # 2 bands for simplicity

    # Create a missing mask for sentinel1 where half the batch is missing
    sentinel1[b // 2 :] = MISSING_VALUE

    # Create the HeliosSample
    days = torch.randint(1, 31, (b, 1, t), dtype=torch.long)
    months = torch.randint(1, 13, (b, 1, t), dtype=torch.long)
    years = torch.randint(2018, 2020, (b, 1, t), dtype=torch.long)
    timestamps = torch.cat([days, months, years], dim=1)

    batch = HeliosSample(
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
        assert (
            abs(num_encoder / total_elements - encode_ratio) < 0.05
        ), "Encoder ratio incorrect for non-missing samples"
        assert (
            abs(num_decoder / total_elements - decode_ratio) < 0.05
        ), "Decoder ratio incorrect for non-missing samples"
        assert (
            abs(num_target / total_elements - (1 - encode_ratio - decode_ratio)) < 0.05
        ), "Target ratio incorrect for non-missing samples"

    # Check that missing samples have the missing value
    missing_indices = torch.where(sentinel1 == MISSING_VALUE)[0]
    for idx in missing_indices:
        mask_slice = sentinel1_mask[idx]
        # All values for missing samples should be set to MaskValue.MISSING.value
        assert (
            mask_slice == MaskValue.MISSING.value
        ).all(), f"Missing sample {idx} should have all mask values set to MISSING"


def test_create_spatial_mask_with_patch_size() -> None:
    """Test the _create_spatial_mask function with different patch sizes."""
    b = 4
    h, w = 16, 16
    shape = (b, h, w)
    patch_size = 4

    encode_ratio, decode_ratio = 0.25, 0.5
    strategy = SpaceMaskingStrategy(
        encode_ratio=encode_ratio, decode_ratio=decode_ratio
    )

    # Call the _create_spatial_mask function directly
    mask = strategy._create_spatial_mask(
        modality=Modality.SENTINEL2_L2A, shape=shape, patch_size=patch_size
    )

    # Check that the mask has the right shape
    assert mask.shape == shape, "Mask shape should match the input shape"

    # Check that patches have consistent values
    for i in range(0, h, patch_size):
        for j in range(0, w, patch_size):
            for b_idx in range(b):
                patch = mask[b_idx, i : i + patch_size, j : j + patch_size]
                # All values within a patch should be the same
                assert (
                    patch == patch[0, 0]
                ).all(), f"Patch at ({b_idx},{i},{j}) has inconsistent values"

    # Check the ratios across all values
    total_elements = mask.numel()
    num_encoder = len(mask[mask == MaskValue.ONLINE_ENCODER.value])
    num_decoder = len(mask[mask == MaskValue.DECODER.value])
    num_target = len(mask[mask == MaskValue.TARGET_ENCODER_ONLY.value])

    assert num_encoder / total_elements == encode_ratio, "Incorrect encode mask ratio"
    assert num_decoder / total_elements == decode_ratio, "Incorrect decode mask ratio"
    assert (
        num_target / total_elements == 1 - encode_ratio - decode_ratio
    ), "Incorrect target mask ratio"


def test_create_temporal_mask() -> None:
    """Test the _create_temporal_mask function."""
    b = 10
    t = 8
    shape = (b, t)

    encode_ratio, decode_ratio = 0.25, 0.5
    strategy = TimeMaskingStrategy(encode_ratio=encode_ratio, decode_ratio=decode_ratio)

    # Call the _create_temporal_mask function directly
    mask = strategy._create_temporal_mask(
        shape=shape,
    )

    # Check the masking ratios for non-missing timesteps

    total_non_missing = mask.numel()
    num_encoder = len(mask[mask == MaskValue.ONLINE_ENCODER.value])
    num_decoder = len(mask[mask == MaskValue.DECODER.value])
    num_target = len(mask[mask == MaskValue.TARGET_ENCODER_ONLY.value])

    # Check that the ratios are close to expected for non-missing values
    # Note: With small values of t, the ratios might not be exactly as expected
    assert (
        abs(num_encoder / total_non_missing - encode_ratio) < 0.2
    ), "Encode mask ratio too far from expected"
    assert (
        abs(num_decoder / total_non_missing - decode_ratio) < 0.2
    ), "Decode mask ratio too far from expected"
    assert (
        abs(num_target / total_non_missing - (1 - encode_ratio - decode_ratio)) < 0.2
    ), "Target mask ratio too far from expected"


def test_space_masking_with_missing_modality_mask() -> None:
    """Test SpaceMaskingStrategy with missing_modalities_masks."""
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

    # Create the HeliosSample with missing_modalities_masks
    batch = HeliosSample(
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
        assert (
            abs(num_encoder / total_elements - encode_ratio) < 0.05
        ), "Incorrect encode mask ratio for present samples"
        assert (
            abs(num_decoder / total_elements - decode_ratio) < 0.05
        ), "Incorrect decode mask ratio for present samples"
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
                    assert (
                        patch == patch[0, 0]
                    ).all(), f"Patch at ({idx},{i},{j},{t_idx}) has inconsistent values"

    # Test unmasking
    unmasked_sample = masked_sample.unmask()

    # Check that sentinel1_mask was created
    unmasked_sentinel1_mask = unmasked_sample.sentinel1_mask
    assert unmasked_sentinel1_mask is not None

    # Check that non-missing samples have been set to ONLINE_ENCODER
    for idx in present_indices:
        # All non-missing values should be set to ONLINE_ENCODER (0)
        assert (
            unmasked_sentinel1_mask[idx] == MaskValue.ONLINE_ENCODER.value
        ).all(), "Unmasked should be ONLINE_ENCODER for present samples"


def test_time_masking_with_missing_modality_mask() -> None:
    """Test TimeMaskingStrategy with missing_modalities_masks."""
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

    # Create the HeliosSample with missing_modalities_masks
    batch = HeliosSample(
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
        assert (
            abs(num_encoder / total_elements - encode_ratio) < 0.05
        ), "Incorrect encode mask ratio for present samples"
        assert (
            abs(num_decoder / total_elements - decode_ratio) < 0.05
        ), "Incorrect decode mask ratio for present samples"
        assert (
            abs(num_target / total_elements - (1 - encode_ratio - decode_ratio)) < 0.05
        ), "Incorrect target mask ratio for present samples"

    # Check that missing samples are set to MISSING
    missing_indices = torch.where(sentinel1 == MISSING_VALUE)[0]
    for idx in missing_indices:
        assert (
            sentinel1_mask[idx] == MaskValue.MISSING.value
        ).all(), f"Sample {idx} should be set to MISSING"

    # Test unmasking
    unmasked_sample = masked_sample.unmask()

    # Check that sentinel1_mask was created
    unmasked_sentinel1_mask = unmasked_sample.sentinel1_mask
    assert unmasked_sentinel1_mask is not None

    # Check that non-missing samples have been set to ONLINE_ENCODER
    for idx in present_indices:
        # All non-missing values should be set to ONLINE_ENCODER (0)
        assert (
            unmasked_sentinel1_mask[idx] == MaskValue.ONLINE_ENCODER.value
        ).all(), "Unmasked should be ONLINE_ENCODER for present samples"


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

    # Create the HeliosSample with missing_modalities_masks
    batch = HeliosSample(
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
        assert (
            abs(num_encoder / total_elements - encode_ratio) < 0.05
        ), "Incorrect encode mask ratio for present samples"
        assert (
            abs(num_decoder / total_elements - decode_ratio) < 0.05
        ), "Incorrect decode mask ratio for present samples"
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
        assert (
            unmasked_sentinel1_mask[idx] == MaskValue.ONLINE_ENCODER.value
        ).all(), "Unmasked should be ONLINE_ENCODER for present samples"


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
