"""Test GeoBench dataset."""

from pathlib import Path

from torch.utils.data import DataLoader

from olmoearth_pretrain.data.constants import Modality
from olmoearth_pretrain.evals.datasets import CropHarvestDataset
from olmoearth_pretrain.evals.datasets.utils import eval_collate_fn
from olmoearth_pretrain.nn.flexi_vit import Encoder

CROPHARVEST_TEST_DIR = Path(__file__).parents[3] / "cropharvest"


def test_cropharvest_dataset_maybe() -> None:
    """Test forward pass from CropHarvest data."""
    if not CROPHARVEST_TEST_DIR.exists():
        return None
    for norm_stats_from_pretrained in [True, False]:
        supported_modalities = [Modality.SENTINEL2_L2A]
        ds = CropHarvestDataset(
            cropharvest_dir=CROPHARVEST_TEST_DIR,
            country="Togo",
            split="train",
            partition="default",
            norm_stats_from_pretrained=norm_stats_from_pretrained,
            timesteps=12,
            input_modalities=[x.name for x in supported_modalities],
        )
        d = DataLoader(
            dataset=ds,
            collate_fn=eval_collate_fn,
            shuffle=False,
            batch_size=1,
        )
        encoder = Encoder(
            embedding_size=16,
            max_patch_size=4,
            min_patch_size=1,
            num_heads=2,
            depth=2,
            mlp_ratio=1.0,
            drop_path=0.1,
            max_sequence_length=12,
            supported_modalities=supported_modalities,
        )

        batch, _ = next(iter(d))
        _ = encoder(batch, patch_size=1)
