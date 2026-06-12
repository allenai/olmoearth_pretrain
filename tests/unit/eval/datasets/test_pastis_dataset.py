"""Test Pastis dataset."""

import os
from pathlib import Path

import pytest
import torch

from olmoearth_pretrain.evals.datasets.pastis_dataset import PASTISRDataset
from olmoearth_pretrain.evals.datasets.pastis_processor import PASTISRProcessor
from olmoearth_pretrain.modalities import Modality


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
    assert sample.timestamps is not None
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


def test_pastis_processor_imputes_s2_bands(tmp_path: Path) -> None:
    """PASTIS S2 band imputation expands the source 10 bands to eval S2 order."""
    processor = PASTISRProcessor(str(tmp_path), str(tmp_path / "out"))
    image = torch.arange(10 * 2 * 3, dtype=torch.float32).reshape(10, 2, 3)

    imputed = processor.impute(image)

    assert imputed.shape == (13, 2, 3)
    assert imputed[0].equal(image[0])
    assert imputed[1].equal(image[0])
    assert imputed[9].equal(image[7])
    assert imputed[10].equal(image[8])
    assert imputed[12].equal(image[9])


def test_pastis_processor_aggregates_months(tmp_path: Path) -> None:
    """Monthly aggregation averages repeated months and preserves chronological order."""
    processor = PASTISRProcessor(str(tmp_path), str(tmp_path / "out"))
    images = torch.stack(
        [
            torch.ones((2, 2, 2)),
            3 * torch.ones((2, 2, 2)),
            5 * torch.ones((2, 2, 2)),
        ]
    )
    dates = {"0": 20180901, "1": 20180920, "2": 20181001}

    aggregated, months = processor.aggregate_months(
        Modality.SENTINEL1.name, images, dates
    )

    assert months.equal(torch.tensor([201809, 201810], dtype=torch.long))
    assert torch.equal(aggregated[0], 2 * torch.ones((2, 2, 2)))
    assert torch.equal(aggregated[1], 5 * torch.ones((2, 2, 2)))
