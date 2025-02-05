from pathlib import Path

import pytest
from torch.utils.data import DataLoader

from helios.evals.datasets import GeobenchDataset


@pytest.fixture
def geobench_dir() -> Path:
    """Fixture providing path to test dataset index."""
    return Path("tests/fixtures/sample_geobench")


def test_geobench_dataset(geobench_dir):
    d = GeobenchDataset(
        dataset="m-eurosat",
        geobench_dir=geobench_dir,
        split="train",
        partition="0.01x_train",
    )
    sample, _ = d[0]
    assert sample.s2.shape == (13, 1, 64, 64)


def test_geobench_dataset(geobench_dir):
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
    assert sample.s2.shape == (1, 13, 1, 64, 64)
