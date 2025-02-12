"""Test GeoBench dataset."""

from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader

from helios.evals.datasets import GeobenchDataset


@pytest.fixture
def geobench_dir() -> Path:
    """Fixture providing path to test dataset index."""
    return Path("tests/fixtures/sample_geobench")


def test_geobench_dataset(geobench_dir: Path) -> None:
    """Test the dataset works."""
    d = GeobenchDataset(
        dataset="m-eurosat",
        geobench_dir=geobench_dir,
        split="train",
        partition="0.01x_train",
    )
    sample, _ = d[0]
    assert isinstance(sample.s2, torch.Tensor)
    assert sample.s2.shape == (64, 64, 1, 13)


def test_geobench_dataset_and_dataloader(geobench_dir: Path) -> None:
    """Test the dataloader (and specifically the collate fn) works."""
    d = DataLoader(
        GeobenchDataset(
            dataset="m-eurosat",
            geobench_dir=geobench_dir,
            split="train",
            partition="0.01x_train",
        ),
        collate_fn=GeobenchDataset.collate_fn,
        batch_size=1,
        shuffle=False,
    )
    sample, _ = next(iter(d))
    assert isinstance(sample.s2, torch.Tensor)
    assert sample.s2.shape == (1, 64, 64, 1, 13)
