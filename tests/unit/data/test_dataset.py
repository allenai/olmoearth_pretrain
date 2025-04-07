"""Unit tests for the dataset module."""

import torch

from helios.data.constants import MISSING_VALUE
from helios.data.dataset import HeliosSample, collate_helios


def test_collate_helios(
    samples_with_missing_modalities: list[tuple[int, HeliosSample]],
) -> None:
    """Test the collate_helios function."""
    collated_sample = collate_helios(
        samples_with_missing_modalities,
    )

    # Check that all required fields are present
    assert collated_sample[1].sentinel2_l2a is not None
    assert collated_sample[1].sentinel1 is not None
    assert collated_sample[1].worldcover is not None
    assert collated_sample[1].latlon is not None
    assert collated_sample[1].timestamps is not None

    # Check the shapes
    assert collated_sample[1].sentinel2_l2a.shape[0] == 3
    assert collated_sample[1].sentinel1.shape[0] == 3
    assert collated_sample[1].worldcover.shape[0] == 3
    assert collated_sample[1].latlon.shape[0] == 3
    assert collated_sample[1].timestamps.shape[0] == 3

    # Check the missing modality mask values
    assert torch.all(collated_sample[1].sentinel1[1] == MISSING_VALUE)
    assert torch.all(collated_sample[1].worldcover[2] == MISSING_VALUE)


class TestHeliosSample:
    """Test the HeliosSample class."""

    # Test subsetting collate function with missing modalities
    def test_subset_with_missing_modalities(
        self,
        samples_with_missing_modalities: list[tuple[int, HeliosSample]],
    ) -> None:
        """Test subsetting a collated sample with missing modalities."""
        sampled_hw_p = 4
        patch_size = 2
        max_tokens_per_instance = 100
        sample: HeliosSample = samples_with_missing_modalities[1][1]
        subset_sample = sample.subset(
            patch_size=patch_size,
            max_tokens_per_instance=max_tokens_per_instance,
            sampled_hw_p=sampled_hw_p,
        )

        # Check that the shapes are correct
        assert subset_sample.sentinel2_l2a is not None
        assert subset_sample.sentinel1 is not None
        assert subset_sample.worldcover is not None

        assert subset_sample.sentinel2_l2a.shape[0] == 8
        assert subset_sample.sentinel1.shape[0] == 8
        assert subset_sample.worldcover.shape[0] == 8

        # Check that the missing modality masks are preserved
        # Check the missing modality mask values
        assert (subset_sample.sentinel1[1] != MISSING_VALUE).sum() == 0
