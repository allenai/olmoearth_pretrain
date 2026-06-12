"""Tests for shared OlmoEarth data structures."""

import pytest
import torch

from olmoearth_pretrain.datatypes import (
    MASKED_SAMPLE_MODALITY_FIELDS,
    SAMPLE_MODALITY_FIELDS,
    TOKEN_MODALITY_FIELDS,
    MaskedOlmoEarthSample,
    MaskValue,
    OlmoEarthSample,
    TokensAndMasks,
    make_modality_mask_like,
)
from olmoearth_pretrain.modalities import Modality


def test_from_olmoearthsample_uses_bandset_mask_shape_for_batched_sample() -> None:
    """Masked samples should use one mask channel per band set, not per band."""
    batch_size, height, width, timesteps = 2, 8, 8, 3
    sample = OlmoEarthSample(
        sentinel2_l2a=torch.ones(
            batch_size,
            height,
            width,
            timesteps,
            Modality.SENTINEL2_L2A.num_bands,
        ),
        timestamps=torch.ones(batch_size, timesteps, 3),
    )

    masked = MaskedOlmoEarthSample.from_olmoearthsample(sample)

    assert masked.sentinel2_l2a_mask is not None
    assert masked.sentinel2_l2a_mask.shape == (
        batch_size,
        height,
        width,
        timesteps,
        Modality.SENTINEL2_L2A.num_band_sets,
    )
    assert (masked.sentinel2_l2a_mask == MaskValue.ONLINE_ENCODER.value).all()


def test_make_modality_mask_like_uses_bandset_shape_and_requested_dtype() -> None:
    """Manual samples should build masks over band sets, not raw bands."""
    tensor = torch.ones(2, 8, 8, 3, Modality.LANDSAT.num_bands)

    mask = make_modality_mask_like(
        tensor,
        Modality.LANDSAT,
        fill_value=MaskValue.DECODER.value,
        dtype=tensor.dtype,
    )

    assert mask.shape == (2, 8, 8, 3, Modality.LANDSAT.num_band_sets)
    assert mask.dtype == tensor.dtype
    assert (mask == MaskValue.DECODER.value).all()


def test_sample_containers_have_same_modality_fields() -> None:
    """Raw, masked, and token containers should stay aligned on modality names."""
    assert set(SAMPLE_MODALITY_FIELDS) == set(MASKED_SAMPLE_MODALITY_FIELDS)
    assert set(SAMPLE_MODALITY_FIELDS) == set(TOKEN_MODALITY_FIELDS)


def test_masked_and_token_containers_have_matching_mask_fields() -> None:
    """Every data modality in masked containers should have one matching mask field."""
    for field in SAMPLE_MODALITY_FIELDS:
        mask_field = MaskedOlmoEarthSample.get_masked_modality_name(field)
        assert mask_field in MaskedOlmoEarthSample._fields
        assert mask_field in TokensAndMasks._fields


def test_unmasked_modality_name_removes_only_suffix() -> None:
    """Unmasking helper should only remove a trailing mask suffix."""
    assert (
        MaskedOlmoEarthSample.get_unmasked_modality_name("sentinel2_l2a_mask")
        == "sentinel2_l2a"
    )
    assert (
        MaskedOlmoEarthSample.get_unmasked_modality_name("mask_quality_mask")
        == "mask_quality"
    )
    assert (
        MaskedOlmoEarthSample.get_unmasked_modality_name("mask_quality")
        == "mask_quality"
    )


def test_container_to_device_preserves_present_fields() -> None:
    """Container device moves should keep only populated tensor-like fields."""
    device = torch.device("cpu")
    sample = OlmoEarthSample(
        sentinel1=torch.ones(1, 4, 4, 2, Modality.SENTINEL1.num_bands),
        timestamps=torch.ones(1, 2, 3),
    )
    masked = MaskedOlmoEarthSample.from_olmoearthsample(sample)
    tokens = TokensAndMasks(
        sentinel1=torch.ones(1, 2, 2, 2, Modality.SENTINEL1.num_band_sets, 8),
        sentinel1_mask=torch.ones(1, 2, 2, 2, Modality.SENTINEL1.num_band_sets),
    )

    moved_sample = sample.to_device(device)
    moved_masked = masked.to_device(device)
    moved_tokens = tokens.to_device(device)

    assert moved_sample.sentinel1 is not None
    assert moved_sample.sentinel1.device == device
    assert moved_sample.sentinel2_l2a is None
    assert moved_masked.sentinel1_mask is not None
    assert moved_masked.sentinel1_mask.device == device
    assert moved_tokens.sentinel1 is not None
    assert moved_tokens.sentinel1.device == device


def test_tokens_and_masks_first_field_errors_are_clear() -> None:
    """Empty token containers should fail clearly for first-field properties."""
    tokens = TokensAndMasks()

    for attr_name in ("batch_size", "device"):
        with pytest.raises(ValueError, match="No data is present"):
            getattr(tokens, attr_name)


def test_legacy_sample_type_import_paths_still_resolve() -> None:
    """Legacy import paths should keep resolving to the canonical datatypes."""
    from olmoearth_pretrain.data.dataset import OlmoEarthSample as DatasetSample
    from olmoearth_pretrain.nn.flexi_vit import TokensAndMasks as FlexiTokensAndMasks
    from olmoearth_pretrain.train.masking import (
        MaskedOlmoEarthSample as MaskingSample,
    )
    from olmoearth_pretrain.train.masking import MaskValue as MaskingMaskValue

    assert DatasetSample is OlmoEarthSample
    assert MaskingSample is MaskedOlmoEarthSample
    assert MaskingMaskValue is MaskValue
    assert FlexiTokensAndMasks is TokensAndMasks
