"""Test the train utils."""

import pytest
import torch

from olmoearth_pretrain.data.constants import MISSING_VALUE
from olmoearth_pretrain.datatypes import MaskedOlmoEarthSample, MaskValue
from olmoearth_pretrain.train.utils import (
    HistogramConfig,
    compute_histogram,
    compute_histograms_for_batch,
    split_masked_batch,
)


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


class TestComputeHistogram:
    """Tests for the compute_histogram function."""

    def test_basic_categorical_histogram(self) -> None:
        """Test basic categorical histogram computation."""
        # Create a batch with known class distribution
        # Sample 0: all class 0, Sample 1: all class 1
        B, H, W = 2, 4, 4
        values = torch.zeros(B, H, W, 1, 1)
        values[0] = 0  # All class 0
        values[1] = 1  # All class 1

        hist = compute_histogram(values, num_bins=3, categorical=True)

        assert hist.shape == (2, 3)
        # Sample 0 should have all mass in bin 0
        assert hist[0, 0] == 1.0
        assert hist[0, 1] == 0.0
        assert hist[0, 2] == 0.0
        # Sample 1 should have all mass in bin 1
        assert hist[1, 0] == 0.0
        assert hist[1, 1] == 1.0
        assert hist[1, 2] == 0.0

    def test_histogram_with_missing_values(self) -> None:
        """Test histogram computation excludes missing values."""
        B, H, W = 1, 4, 4
        values = torch.zeros(B, H, W, 1, 1)
        # Set half the values to class 0, half to MISSING_VALUE
        values[0, :2, :, :, :] = 0
        values[0, 2:, :, :, :] = MISSING_VALUE

        hist = compute_histogram(values, num_bins=3, categorical=True)

        # Should only count non-missing values
        assert hist.shape == (1, 3)
        assert hist[0, 0] == 1.0  # All valid values are class 0
        assert hist[0, 1] == 0.0
        assert hist[0, 2] == 0.0

    def test_histogram_with_mask(self) -> None:
        """Test histogram computation respects mask tensor."""
        B, H, W = 1, 4, 4
        values = torch.zeros(B, H, W, 1, 1)
        values[0, :2, :, :, :] = 0
        values[0, 2:, :, :, :] = 1

        # Mask out the class 1 values
        mask = torch.ones(B, H, W, 1, 1) * MaskValue.ONLINE_ENCODER.value
        mask[0, 2:, :, :, :] = MaskValue.MISSING.value

        hist = compute_histogram(values, num_bins=3, mask=mask, categorical=True)

        # Should only count unmasked values (class 0)
        assert hist[0, 0] == 1.0
        assert hist[0, 1] == 0.0

    def test_histogram_all_missing_returns_uniform(self) -> None:
        """Test that all-missing samples return uniform distribution."""
        B, H, W = 1, 4, 4
        values = torch.full((B, H, W, 1, 1), MISSING_VALUE)

        hist = compute_histogram(values, num_bins=4, categorical=True)

        # Should return uniform distribution
        expected = torch.tensor([[0.25, 0.25, 0.25, 0.25]])
        assert torch.allclose(hist, expected)

    def test_histogram_mixed_classes(self) -> None:
        """Test histogram with mixed class distribution."""
        # Shape [1, 2, 2, 1, 1] = 4 pixels: 2x class 0, 1x class 1, 1x class 2
        values = torch.tensor([[[[[0]], [[1]]], [[[2]], [[0]]]]]).float()

        hist = compute_histogram(values, num_bins=3, categorical=True)

        assert hist.shape == (1, 3)
        assert torch.allclose(hist[0], torch.tensor([0.5, 0.25, 0.25]))

    def test_histogram_normalization(self) -> None:
        """Test that histograms sum to 1."""
        B, H, W = 3, 8, 8
        values = torch.randint(0, 10, (B, H, W, 1, 1)).float()

        hist = compute_histogram(values, num_bins=10, categorical=True)

        # Each histogram should sum to 1
        assert torch.allclose(hist.sum(dim=1), torch.ones(B))

    def test_histogram_with_class_values(self) -> None:
        """Test histogram with explicit class values (e.g., normalized WorldCover)."""
        # Simulate normalized WorldCover values: 0.1, 0.2, 0.3, ... 1.0
        # Create a sample with values close to class 0.1 and 0.3
        # Shape: [1, 2, 2, 1, 1] -> 4 spatial locations
        values = torch.tensor([[[[0.1], [0.11]], [[0.29], [0.31]]]]).float()

        class_values = [0.1, 0.2, 0.3, 0.4, 0.5]

        hist = compute_histogram(
            values, num_bins=5, categorical=True, class_values=class_values
        )

        # 0.1, 0.11 should map to bin 0 (class 0.1)
        # 0.29, 0.31 should map to bin 2 (class 0.3)
        assert hist.shape == (1, 5)
        assert hist[0, 0] == 0.5  # 2 out of 4 values
        assert hist[0, 1] == 0.0  # 0 values
        assert hist[0, 2] == 0.5  # 2 out of 4 values
        assert hist[0, 3] == 0.0
        assert hist[0, 4] == 0.0

    def test_histogram_one_hot(self) -> None:
        """Test histogram with one-hot encoded data."""
        # Shape [B, H, W, T, C] = [1, 2, 2, 1, 3] -> 4 spatial locations, 3 channels
        # Channels represent categories, values are binary (0 or 1)
        values = torch.tensor(
            [
                [
                    [[[1.0, 0.0, 0.0]], [[0.0, 1.0, 0.0]]],
                    [[[1.0, 0.0, 1.0]], [[0.0, 0.0, 0.0]]],
                ]
            ]
        )  # [1, 2, 2, 1, 3]

        hist = compute_histogram(
            values, num_bins=3, one_hot=True, one_hot_threshold=0.5
        )

        # 4 spatial locations total
        # Channel 0: 2 positives (locations [0,0] and [1,0])
        # Channel 1: 1 positive (location [0,1])
        # Channel 2: 1 positive (location [1,0])
        assert hist.shape == (1, 3)
        assert torch.allclose(hist[0], torch.tensor([0.5, 0.25, 0.25]))

    def test_histogram_one_hot_with_missing(self) -> None:
        """Test one-hot histogram excludes missing values."""
        # Create data with some missing values
        values = torch.tensor(
            [
                [
                    [[[1.0, 0.0]], [[0.0, 1.0]]],
                    [[[MISSING_VALUE, MISSING_VALUE]], [[1.0, 1.0]]],
                ]
            ]
        )  # [1, 2, 2, 1, 2]

        hist = compute_histogram(
            values, num_bins=2, one_hot=True, one_hot_threshold=0.5
        )

        # 3 valid spatial locations (one is missing)
        # Channel 0: 2 positives out of 3 valid
        # Channel 1: 2 positives out of 3 valid
        assert hist.shape == (1, 2)
        assert torch.allclose(hist[0], torch.tensor([2 / 3, 2 / 3]))


class TestComputeHistogramsForBatch:
    """Tests for compute_histograms_for_batch function."""

    def test_compute_histograms_for_batch(self) -> None:
        """Test computing histograms for multiple modalities in a batch."""
        B, H, W, T = 2, 4, 4, 1
        timestamps = torch.ones(B, T, 3).long()
        worldcover = torch.randint(0, 11, (B, H, W, T, 1)).float()
        worldcover_mask = torch.ones(B, H, W, T, 1).long() * MaskValue.DECODER.value

        batch = MaskedOlmoEarthSample(
            timestamps=timestamps,
            worldcover=worldcover,
            worldcover_mask=worldcover_mask,
        )

        histogram_configs = {
            "worldcover": HistogramConfig(num_bins=11, categorical=True),
        }

        histograms = compute_histograms_for_batch(batch, histogram_configs)

        assert "worldcover" in histograms
        assert histograms["worldcover"].shape == (B, 11)
        # Each histogram should sum to 1
        assert torch.allclose(histograms["worldcover"].sum(dim=1), torch.ones(B))

    def test_compute_histograms_missing_modality(self) -> None:
        """Test that missing modalities are skipped."""
        B, T = 2, 1
        timestamps = torch.ones(B, T, 3).long()

        batch = MaskedOlmoEarthSample(
            timestamps=timestamps,
            # No worldcover data
        )

        histogram_configs = {
            "worldcover": HistogramConfig(num_bins=11, categorical=True),
        }

        histograms = compute_histograms_for_batch(batch, histogram_configs)

        # worldcover should not be in output since it's not in batch
        assert "worldcover" not in histograms
