"""Test the train utils."""

import pytest
import torch

from olmoearth_pretrain.data.dataset import OlmoEarthSample
from olmoearth_pretrain.datatypes import MaskedOlmoEarthSample
from olmoearth_pretrain.train.masking import MaskValue
from olmoearth_pretrain.train.utils import split_batch, split_masked_batch


@pytest.mark.parametrize("microbatch_size", [1, 2, 5])
def test_split_batch(microbatch_size: int) -> None:
    """Test the split_batch function."""
    B, H, W, T, D = 10, 2, 2, 2, 4
    sentinel2_tokens = torch.zeros(B, H, W, T, D)
    latlon_tokens = torch.randn(B, 1, D)
    x = {"sentinel2_l2a": sentinel2_tokens, "latlon": latlon_tokens}
    batch = OlmoEarthSample(**x)
    micro_batches = split_batch(batch, microbatch_size)
    assert len(micro_batches) == (B + microbatch_size - 1) // microbatch_size
    for i, micro_batch in enumerate(micro_batches):
        if i == len(micro_batches) - 1:
            microbatch_size = B - i * microbatch_size
        assert micro_batch.batch_size == microbatch_size
        assert micro_batch.sentinel2_l2a is not None
        assert micro_batch.sentinel2_l2a.shape == (microbatch_size, H, W, T, D)
        assert micro_batch.latlon is not None
        assert micro_batch.latlon.shape == (microbatch_size, 1, D)


@pytest.mark.parametrize("microbatch_size", [1, 2, 5, 10, 15])
def test_split_masked_batch(microbatch_size: int) -> None:
    """Test the split_masked_batch function."""
    B, H, W, T, D = 10, 4, 4, 3, 12
    timestamps = torch.ones(B, T, 3).long()
    sentinel2_l2a = torch.randn(B, H, W, T, D)
    sentinel2_l2a_mask = (
        torch.ones(B, H, W, T, D).long() * MaskValue.ONLINE_ENCODER.value
    )

    batch = MaskedOlmoEarthSample(
        timestamps=timestamps,
        sentinel2_l2a=sentinel2_l2a,
        sentinel2_l2a_mask=sentinel2_l2a_mask,
    )

    micro_batches = split_masked_batch(batch, microbatch_size)

    # Check number of microbatches
    expected_num = (B + microbatch_size - 1) // microbatch_size
    assert len(micro_batches) == expected_num

    # Track total samples to verify we got all of them
    total_samples = 0
    expected_mb_size = microbatch_size

    for i, micro_batch in enumerate(micro_batches):
        # Last batch may be smaller
        if i == len(micro_batches) - 1:
            expected_mb_size = B - i * microbatch_size

        mb_size = micro_batch.timestamps.shape[0]
        assert mb_size == expected_mb_size
        total_samples += mb_size

        # Check shapes
        assert micro_batch.timestamps.shape == (expected_mb_size, T, 3)
        assert micro_batch.sentinel2_l2a is not None
        assert micro_batch.sentinel2_l2a.shape == (expected_mb_size, H, W, T, D)
        assert micro_batch.sentinel2_l2a_mask is not None
        assert micro_batch.sentinel2_l2a_mask.shape == (expected_mb_size, H, W, T, D)

        # Verify data integrity - values should match original slices
        start = i * microbatch_size
        end = start + expected_mb_size
        assert torch.equal(micro_batch.timestamps, timestamps[start:end])
        assert torch.equal(micro_batch.sentinel2_l2a, sentinel2_l2a[start:end])
        assert torch.equal(
            micro_batch.sentinel2_l2a_mask, sentinel2_l2a_mask[start:end]
        )
        # None modalities should stay None
        assert micro_batch.worldcover is None
        assert micro_batch.worldcover_mask is None

    assert total_samples == B


def test_split_masked_batch_no_split_needed() -> None:
    """Test split_masked_batch when batch is already small enough."""
    B, T = 4, 2
    batch = MaskedOlmoEarthSample(
        timestamps=torch.ones(B, T, 3).long(),
    )

    micro_batches = split_masked_batch(batch, microbatch_size=10)
    assert len(micro_batches) == 1
    assert micro_batches[0] is batch  # Should return same object
