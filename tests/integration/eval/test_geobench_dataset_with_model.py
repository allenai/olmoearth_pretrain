"""Test GeoBench dataset."""

from pathlib import Path

import pytest
from torch.utils.data import DataLoader

from helios.data.constants import Modality
from helios.evals.datasets import GeobenchDataset
from helios.nn.flexihelios import Encoder


@pytest.fixture
def geobench_dir() -> Path:
    """Fixture providing path to test dataset index."""
    return Path("tests/fixtures/sample_geobench")


def test_geobench_dataset(geobench_dir: Path) -> None:
    """Test forward pass from GeoBench data."""
    supported_modalities = [Modality.SENTINEL2]
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
    encoder = Encoder(
        embedding_size=16,
        max_patch_size=4,
        num_heads=2,
        depth=2,
        mlp_ratio=1.0,
        drop_path=0.1,
        max_sequence_length=12,
        use_channel_embs=True,
        supported_modalities=supported_modalities,
    )

    batch, _ = next(iter(d))
    _ = encoder(batch, patch_size=4)
