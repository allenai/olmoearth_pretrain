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


@pytest.fixture
def mock_pastis_data_with_gse(mock_pastis_data: Path) -> tuple[Path, torch.Tensor]:
    """Add a mock precomputed GSE embedding to the mock PASTIS data."""
    gse = torch.randn(len(Modality.GSE.band_order), 64, 64)
    gse_path = mock_pastis_data / "pastis_r_train" / "gse_images" / "0.pt"
    os.makedirs(gse_path.parent, exist_ok=True)
    torch.save(gse, gse_path)
    return mock_pastis_data, gse


def test_pastis_dataset_gse_embeddings(
    mock_pastis_data_with_gse: tuple[Path, torch.Tensor],
) -> None:
    """Precomputed embeddings load as (H, W, 1, C) without normalization."""
    path_to_splits, gse = mock_pastis_data_with_gse
    dataset = PASTISRDataset(
        path_to_splits=path_to_splits,
        split="train",
        input_modalities=[Modality.GSE.name],
    )
    sample, label = dataset[0]

    assert sample.gse is not None
    assert sample.gse.shape == (64, 64, 1, len(Modality.GSE.band_order))
    # Consumed exactly as stored: no normalization applied.
    torch.testing.assert_close(sample.gse[:, :, 0, :], gse.permute(1, 2, 0))
    # Imagery is not loaded when not requested.
    assert sample.sentinel2_l2a is None
    assert sample.sentinel1 is None
    assert label.shape == (64, 64)


def test_pastis_dataset_window_size_tiles_samples(
    mock_pastis_data_with_gse: tuple[Path, torch.Tensor],
) -> None:
    """window_size tiles imagery, embeddings, and labels consistently."""
    path_to_splits, gse = mock_pastis_data_with_gse
    full = PASTISRDataset(
        path_to_splits=path_to_splits,
        split="train",
        input_modalities=[Modality.SENTINEL2_L2A.name, Modality.GSE.name],
    )
    tiled = PASTISRDataset(
        path_to_splits=path_to_splits,
        split="train",
        input_modalities=[Modality.SENTINEL2_L2A.name, Modality.GSE.name],
        window_size=32,
    )
    assert len(full) == 1
    assert len(tiled) == 4

    full_sample, full_labels = full[0]
    # Tile 3 is (row 1, col 1) -> the bottom-right 32x32 window.
    sample, labels = tiled[3]
    assert sample.sentinel2_l2a is not None and sample.gse is not None
    assert full_sample.sentinel2_l2a is not None and full_sample.gse is not None
    assert sample.sentinel2_l2a.shape[:2] == (32, 32)
    assert sample.gse.shape == (32, 32, 1, len(Modality.GSE.band_order))
    assert labels.shape == (32, 32)
    torch.testing.assert_close(
        sample.sentinel2_l2a, full_sample.sentinel2_l2a[32:, 32:]
    )
    torch.testing.assert_close(sample.gse, full_sample.gse[32:, 32:])
    torch.testing.assert_close(labels, full_labels[32:, 32:])


def test_pastis_dataset_window_size_must_divide(mock_pastis_data: Path) -> None:
    """A window_size that doesn't divide the sample size raises."""
    with pytest.raises(ValueError, match="must divide"):
        PASTISRDataset(
            path_to_splits=mock_pastis_data,
            split="train",
            input_modalities=[Modality.SENTINEL2_L2A.name],
            window_size=48,
        )


def test_pastis_dataset_missing_embeddings_raise(mock_pastis_data: Path) -> None:
    """Requesting embeddings from splits without them gives a clear error."""
    with pytest.raises(FileNotFoundError, match="--embedding_products"):
        PASTISRDataset(
            path_to_splits=mock_pastis_data,
            split="train",
            input_modalities=[Modality.TESSERA.name],
        )


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
