"""Test GeoBench dataset."""

from pathlib import Path

import pytest
from torch.utils.data import DataLoader

from helios.evals.datasets import GeobenchDataset
from helios.train.encoder import PatchEncoder


@pytest.fixture
def geobench_dir() -> Path:
    """Fixture providing path to test dataset index."""
    return Path("tests/fixtures/sample_geobench")


def test_geobench_dataset(geobench_dir: Path) -> None:
    """Test forward pass from GeoBench data."""
    d = DataLoader(
        GeobenchDataset(
            dataset="m-eurosat",
            geobench_dir=geobench_dir,
            split="train",
            partition="0.01x_train",
        ),
        collate_fn=GeobenchDataset.collate_fn,
        shuffle=False,
        batch_size=1,
    )
    model = PatchEncoder(in_channels=13, time_patch_size=1)

    # add a batch dimension
    batch, _ = next(iter(d))
    _ = model(x=batch.s2.float())
