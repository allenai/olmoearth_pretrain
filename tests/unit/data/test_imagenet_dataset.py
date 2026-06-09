"""Tests for ImageNet natural-image dataset support."""

from pathlib import Path

import numpy as np
from PIL import Image

from olmoearth_pretrain.data.collate import collate_single_masked_batched
from olmoearth_pretrain.data.constants import Modality
from olmoearth_pretrain.data.dataset import GetItemArgs, ImageNetDatasetConfig
from olmoearth_pretrain.datatypes import MaskedOlmoEarthSample, OlmoEarthSample
from olmoearth_pretrain.train.masking import MaskingConfig


def _write_rgb_image(path: Path, value: int) -> None:
    """Write a small RGB image."""
    path.parent.mkdir(parents=True, exist_ok=True)
    image = np.full((12, 10, 3), value, dtype=np.uint8)
    Image.fromarray(image, mode="RGB").save(path)


def test_imagenet_modality_spec() -> None:
    """ImageNet is a separate RGB, non-temporal modality."""
    assert Modality.IMAGENET.name == "imagenet"
    assert Modality.IMAGENET.band_order == ["R", "G", "B"]
    assert not Modality.IMAGENET.is_multitemporal
    assert OlmoEarthSample.compute_expected_shape("imagenet", 256, 256, 12) == (
        256,
        256,
        1,
        3,
    )


def test_imagenet_dataset_loads_imagefolder_sample(tmp_path: Path) -> None:
    """ImageNetDatasetConfig builds a dataset that returns OlmoEarth samples."""
    _write_rgb_image(tmp_path / "n00000001" / "image.JPEG", 128)

    dataset = ImageNetDatasetConfig(
        root_dir=str(tmp_path),
        image_size=16,
        normalize=False,
    ).build()
    dataset.prepare()

    patch_size, sample = dataset[
        GetItemArgs(idx=0, patch_size=4, sampled_hw_p=2, token_budget=100000)
    ]

    assert patch_size == 4
    assert sample.modalities == ["imagenet"]
    assert sample.imagenet is not None
    assert sample.imagenet.shape == (8, 8, 1, 3)
    assert sample.timestamps is not None
    assert sample.timestamps.shape == (1, 3)
    assert sample.imagenet.dtype == np.float32
    assert np.all((sample.imagenet >= 0) & (sample.imagenet <= 1))


def test_imagenet_dataset_collates_and_masks(tmp_path: Path) -> None:
    """ImageNet samples are compatible with the existing collate/masking path."""
    _write_rgb_image(tmp_path / "n00000001" / "image_0.JPEG", 64)
    _write_rgb_image(tmp_path / "n00000002" / "image_1.JPEG", 192)

    dataset = ImageNetDatasetConfig(
        root_dir=str(tmp_path),
        image_size=16,
        normalize=False,
    ).build()
    dataset.prepare()
    batch = [
        dataset[GetItemArgs(idx=idx, patch_size=4, sampled_hw_p=2, token_budget=100000)]
        for idx in range(2)
    ]
    masking_strategy = MaskingConfig(strategy_config={"type": "random"}).build()

    patch_size, masked = collate_single_masked_batched(
        batch,
        transform=None,
        masking_strategy=masking_strategy,
    )

    assert patch_size == 4
    assert isinstance(masked, MaskedOlmoEarthSample)
    assert masked.imagenet is not None
    assert masked.imagenet_mask is not None
    assert masked.imagenet.shape == (2, 8, 8, 1, 3)
    assert masked.imagenet_mask.shape == (2, 8, 8, 1, 1)
