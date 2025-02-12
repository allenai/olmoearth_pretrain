"""Test GeoBench dataset."""

from pathlib import Path

import pytest
from torch.utils.data import DataLoader

from helios.constants import S2_BANDS
from helios.evals.datasets import GeobenchDataset
from helios.nn.flexihelios import Encoder
from helios.train.masking import MaskedHeliosSample


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
    modalities_to_channel_groups_dict = {
        "s2": {
            "S2_RGB": [S2_BANDS.index(b) for b in ["B02", "B03", "B04"]],
            "S2_Red_Edge": [S2_BANDS.index(b) for b in ["B05", "B06", "B07"]],
            "S2_NIR_10m": [S2_BANDS.index(b) for b in ["B08"]],
            "S2_NIR_20m": [S2_BANDS.index(b) for b in ["B8A"]],
            "S2_SWIR": [S2_BANDS.index(b) for b in ["B11", "B12"]],
        },
        "latlon": {
            "latlon": [0, 1],
        },
    }
    encoder = Encoder(
        embedding_size=16,
        max_patch_size=8,
        num_heads=2,
        depth=2,
        mlp_ratio=1.0,
        drop_path=0.1,
        max_sequence_length=12,
        base_patch_size=8,
        use_channel_embs=True,
        modalities_to_channel_groups_dict=modalities_to_channel_groups_dict,
    )

    batch, _ = next(iter(d))
    masked_batch = MaskedHeliosSample.from_heliossample(
        batch, encoder.modalities_to_channel_groups_dict
    )
    _ = encoder(masked_batch, patch_size=4)
