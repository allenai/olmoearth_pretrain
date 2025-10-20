"""Unit tests for the dataset module."""

import numpy as np
from torch.utils.data import Dataset

from olmoearth_pretrain.data.concat import OlmoEarthConcatDataset
from olmoearth_pretrain.data.constants import Modality
from olmoearth_pretrain.data.dataset import GetItemArgs

MOCK_FINGERPRINT_VERSION = "mockversion"


class MockDataset(Dataset):
    """Mock OlmoEarthDataset for testing."""

    def __init__(self, items: list[int]):
        """Create a new MockDataset."""
        self.items = items
        self.training_modalities = [Modality.SENTINEL2_L2A]
        self.fingerprint_version = MOCK_FINGERPRINT_VERSION
        self.fingerprint = "0"
        # Create fake latlon distribution by using each integer item as a lat/lon
        # point.
        self.latlon_distribution = np.array([[item, item] for item in items])

    def prepare(self) -> None:
        """Mock prepare function."""
        pass

    def __len__(self) -> int:
        """Mock len function."""
        return len(self.items)

    def __getitem__(self, args: GetItemArgs) -> int:
        """Mock getitem function."""
        return self.items[args.idx]


def test_basic_concat() -> None:
    """Basic test where we concatenate two datasets."""
    dataset1 = MockDataset([0, 1])
    dataset2 = MockDataset([2])
    concat_dataset = OlmoEarthConcatDataset([dataset1, dataset2])
    concat_dataset.prepare()
    concat_items = [
        concat_dataset[GetItemArgs(idx=idx, patch_size=4, sampled_hw_p=4)]
        for idx in range(len(concat_dataset))
    ]
    # Should be concatenated in the order we provided the datasets.
    assert concat_items == [0, 1, 2]


def test_fingerprint() -> None:
    """Make sure the fingerprint and version don't give errors."""
    dataset = MockDataset([0])
    concat_dataset = OlmoEarthConcatDataset([dataset])
    assert concat_dataset.fingerprint_version == MOCK_FINGERPRINT_VERSION
    assert len(concat_dataset.fingerprint) > 0
