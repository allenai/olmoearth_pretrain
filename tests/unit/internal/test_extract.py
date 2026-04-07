"""Tests for the embedding extraction pipeline."""

from typing import Any

import numpy as np
import torch

from olmoearth_pretrain.data.collate import collate_olmoearth_pretrain
from olmoearth_pretrain.data.constants import Modality
from olmoearth_pretrain.data.dataset import OlmoEarthSample, subset_sample_default
from olmoearth_pretrain.datatypes import MaskedOlmoEarthSample, MaskValue
from olmoearth_pretrain.nn.flexi_vit import Encoder
from olmoearth_pretrain.nn.pooling import PoolingType, pool_unmasked_tokens


def _make_sample() -> OlmoEarthSample:
    """Create a minimal OlmoEarthSample with sentinel2 + latlon."""
    H, W, T = 8, 8, 2
    s2_C = Modality.SENTINEL2_L2A.num_bands
    return OlmoEarthSample(
        sentinel2_l2a=np.random.randn(H, W, T, s2_C).astype(np.float32),
        latlon=np.array([43.65, -79.38], dtype=np.float32),
        timestamps=np.array([[15, 6, 2023], [15, 7, 2023]], dtype=np.int32),
    )


def _make_encoder(embedding_size: int = 16) -> Encoder:
    return Encoder(
        supported_modalities=[Modality.SENTINEL2_L2A],
        embedding_size=embedding_size,
        max_patch_size=8,
        min_patch_size=1,
        num_heads=2,
        mlp_ratio=4.0,
        max_sequence_length=12,
        depth=2,
        drop_path=0.0,
    )


def _batched_to_masked(batched: OlmoEarthSample) -> MaskedOlmoEarthSample:
    """Convert a batched OlmoEarthSample to MaskedOlmoEarthSample (all ONLINE_ENCODER)."""
    masked_dict: dict[str, Any] = {}
    for key, val in batched.as_dict().items():
        if key == "timestamps":
            masked_dict[key] = val
        elif val is None:
            masked_dict[key] = None
            masked_dict[MaskedOlmoEarthSample.get_masked_modality_name(key)] = None
        else:
            masked_dict[key] = val
            masked_dict[MaskedOlmoEarthSample.get_masked_modality_name(key)] = (
                torch.full(val.shape, MaskValue.ONLINE_ENCODER.value)
            )
    return MaskedOlmoEarthSample(**masked_dict)


def test_extraction_pipeline_shapes_and_dtype() -> None:
    """Verify encoder(fast_pass=True) -> pool(MEAN) -> float16 produces correct output."""
    B = 2
    EMBEDDING_SIZE = 16
    PATCH_SIZE = 4

    encoder = _make_encoder(EMBEDDING_SIZE)
    encoder.eval()

    raw_samples = [(PATCH_SIZE, _make_sample()) for _ in range(B)]
    patch_size, batched_sample = collate_olmoearth_pretrain(raw_samples)
    masked_sample = _batched_to_masked(batched_sample)

    with torch.no_grad():
        output = encoder(masked_sample, patch_size=patch_size, fast_pass=True)

    tokens = output["tokens_and_masks"]
    embeddings = pool_unmasked_tokens(tokens, PoolingType.MEAN, spatial_pooling=False)

    assert embeddings.shape == (B, EMBEDDING_SIZE)
    assert embeddings.dtype == torch.float32

    embeddings_fp16 = embeddings.to(torch.float16).numpy()
    assert embeddings_fp16.dtype == np.float16
    assert embeddings_fp16.shape == (B, EMBEDDING_SIZE)
    assert not np.any(np.isnan(embeddings_fp16))


def test_batched_mask_sets_online_encoder() -> None:
    """Verify batched mask creation sets all present masks to ONLINE_ENCODER."""
    raw_samples = [(4, _make_sample())]
    _, batched_sample = collate_olmoearth_pretrain(raw_samples)
    masked = _batched_to_masked(batched_sample)

    assert masked.sentinel2_l2a_mask is not None
    assert masked.sentinel2_l2a is not None
    assert (masked.sentinel2_l2a_mask == MaskValue.ONLINE_ENCODER.value).all()
    assert masked.sentinel2_l2a_mask.shape == masked.sentinel2_l2a.shape

    assert masked.sentinel1 is None
    assert masked.sentinel1_mask is None


def test_latlon_survives_collation() -> None:
    """Verify latlon is present after collation and has correct shape."""
    B = 3
    raw_samples = [(4, _make_sample()) for _ in range(B)]
    _, batched_sample = collate_olmoearth_pretrain(raw_samples)

    assert batched_sample.latlon is not None
    latlon_t = torch.as_tensor(batched_sample.latlon)
    assert latlon_t.shape == (B, 2)


def test_center_crop_produces_correct_window() -> None:
    """Verify center_crop=True extracts the centered spatial region."""
    H, W, T = 128, 128, 4
    PATCH_SIZE = 8
    SAMPLED_HW_P = 12
    CROP_PX = SAMPLED_HW_P * PATCH_SIZE  # 96

    s2_C = Modality.SENTINEL2_L2A.num_bands
    s2_data = np.arange(H * W * T * s2_C, dtype=np.float32).reshape(H, W, T, s2_C)
    sample = OlmoEarthSample(
        sentinel2_l2a=s2_data,
        latlon=np.array([10.0, 20.0], dtype=np.float32),
        timestamps=np.array(
            [[15, 1, 2023], [15, 2, 2023], [15, 3, 2023], [15, 4, 2023]],
            dtype=np.int32,
        ),
    )

    cropped = subset_sample_default(
        sample,
        patch_size=PATCH_SIZE,
        max_tokens_per_instance=None,
        sampled_hw_p=SAMPLED_HW_P,
        current_length=T,
        center_crop=True,
    )

    offset = (128 - 96) // 2  # 16
    assert cropped.sentinel2_l2a is not None
    assert cropped.sentinel2_l2a.shape == (CROP_PX, CROP_PX, T, s2_C)
    expected = s2_data[offset : offset + CROP_PX, offset : offset + CROP_PX, :, :]
    np.testing.assert_array_equal(cropped.sentinel2_l2a, expected)
    assert cropped.timestamps is not None
    assert cropped.timestamps.shape == (T, 3)
    np.testing.assert_array_equal(cropped.latlon, sample.latlon)


def test_center_crop_is_deterministic() -> None:
    """Verify center_crop produces identical results across calls."""
    H, W, T = 128, 128, 2
    s2_C = Modality.SENTINEL2_L2A.num_bands
    s2_data = np.random.randn(H, W, T, s2_C).astype(np.float32)
    sample = OlmoEarthSample(
        sentinel2_l2a=s2_data,
        latlon=np.array([0.0, 0.0], dtype=np.float32),
        timestamps=np.array([[1, 1, 2023], [1, 2, 2023]], dtype=np.int32),
    )

    results = [
        subset_sample_default(
            sample,
            patch_size=8,
            max_tokens_per_instance=None,
            sampled_hw_p=12,
            current_length=T,
            center_crop=True,
        )
        for _ in range(3)
    ]
    for r in results[1:]:
        assert r.sentinel2_l2a is not None
        assert results[0].sentinel2_l2a is not None
        np.testing.assert_array_equal(r.sentinel2_l2a, results[0].sentinel2_l2a)
