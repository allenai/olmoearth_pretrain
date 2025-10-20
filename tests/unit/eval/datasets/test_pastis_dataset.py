"""Test Pastis dataset."""

import os
from pathlib import Path

import pytest
import torch

from olmoearth_pretrain.data.constants import Modality
from olmoearth_pretrain.evals.datasets.pastis_dataset import PASTISRDataset


@pytest.fixture
def mock_pastis_data(tmp_path: Path) -> Path:
    """Create mock PASTIS-R data for testing."""
    # Create mock data with small dimensions
    s2_images = torch.randn(12, 13, 64, 64)
    s1_images = torch.randn(12, 2, 64, 64)
    targets = torch.randint(0, 2, (1, 64, 64))
    months = torch.tensor(
        [
            201809,
            201810,
            201811,
            201812,
            201901,
            201902,
            201903,
            201904,
            201905,
            201906,
            201907,
            201908,
        ],
        dtype=torch.long,
    ).unsqueeze(0)

    # Save mock data
    s2_path = tmp_path / "pastis_r_train" / "s2_images" / "0.pt"
    os.makedirs(s2_path.parent, exist_ok=True)
    s1_path = tmp_path / "pastis_r_train" / "s1_images" / "0.pt"
    os.makedirs(s1_path.parent, exist_ok=True)
    targets_path = tmp_path / "pastis_r_train" / "targets.pt"
    os.makedirs(targets_path.parent, exist_ok=True)
    months_path = tmp_path / "pastis_r_train" / "months.pt"
    os.makedirs(months_path.parent, exist_ok=True)

    # Save mock data
    torch.save(s2_images, s2_path)
    torch.save(s1_images, s1_path)
    torch.save(targets, targets_path)
    torch.save(months, months_path)

    return tmp_path


def test_pastis_dataset_initialization(mock_pastis_data: Path) -> None:
    """Test basic initialization and functionality of PASTISRDataset."""
    # Test multimodal initialization
    dataset = PASTISRDataset(
        path_to_splits=mock_pastis_data,
        split="train",
        input_modalities=[Modality.SENTINEL1.name, Modality.SENTINEL2_L2A.name],
    )

    assert len(dataset) == 1  # Should have 1 sample

    # Test single sample access
    sample, label = dataset[0]

    # Check basic properties
    assert isinstance(sample.sentinel2_l2a, torch.Tensor)
    assert isinstance(sample.sentinel1, torch.Tensor)
    assert isinstance(label, torch.Tensor)

    # Check shapes
    assert sample.sentinel2_l2a.shape[2] == 12  # 12 timestamps
    assert sample.sentinel1.shape[2] == 12  # 12 timestamps
    assert sample.timestamps[0].equal(torch.tensor([1, 8, 2018], dtype=torch.long))
    assert label.shape == (64, 64)  # Label should be 64x64

    # Test non-multimodal initialization
    dataset_s2_only = PASTISRDataset(
        path_to_splits=mock_pastis_data,
        split="train",
        input_modalities=[Modality.SENTINEL2_L2A.name],
    )

    sample_s2, label_s2 = dataset_s2_only[0]
    assert sample_s2.sentinel1 is None  # Should not have S1 data
    assert sample_s2.sentinel2_l2a is not None  # Should have S2 data
