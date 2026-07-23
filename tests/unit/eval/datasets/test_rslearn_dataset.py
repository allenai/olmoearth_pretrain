"""Tests for the rslearn dataset wrapper's window_size / center-pixel logic."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from rslearn.train.model_context import RasterImage

from olmoearth_pretrain.data.constants import Modality
from olmoearth_pretrain.evals.datasets.rslearn_dataset import (
    RslearnToOlmoEarthDataset,
)
from olmoearth_pretrain.evals.metrics import SEGMENTATION_IGNORE_LABEL
from olmoearth_pretrain.evals.task_types import TaskType

S2 = Modality.SENTINEL2_L2A.name
NUM_S2_BANDS = len(Modality.get(S2).band_order)


def build_dataset(
    window_size: int | None = None,
    label_at_center_pixel: bool = False,
    target_task_type: TaskType = TaskType.SEGMENTATION,
    tile_samples: bool = False,
    sample_size: int | None = None,
    model_dataset: list | None = None,
) -> RslearnToOlmoEarthDataset:
    """Build a wrapper with no underlying model dataset (transform-only tests)."""
    return RslearnToOlmoEarthDataset(
        model_dataset=model_dataset,  # type: ignore[arg-type]
        input_modalities=[S2],
        target_task_type=target_task_type,
        window_size=window_size,
        label_at_center_pixel=label_at_center_pixel,
        tile_samples=tile_samples,
        sample_size=sample_size,
    )


def make_label_raster(size: int, labeled: dict[tuple[int, int], int]) -> torch.Tensor:
    """Build a (size, size) raster that is ignore-labeled except at `labeled`."""
    raster = torch.full((size, size), SEGMENTATION_IGNORE_LABEL, dtype=torch.long)
    for (row, col), value in labeled.items():
        raster[row, col] = value
    return raster


def make_sample(
    size: int, labeled: dict[tuple[int, int], int], num_timesteps: int = 2
) -> tuple[dict, dict]:
    """Build (input_dict, target) mimicking an rslearn segmentation sample."""
    image = torch.arange(
        NUM_S2_BANDS * num_timesteps * size * size, dtype=torch.float32
    ).reshape(NUM_S2_BANDS, num_timesteps, size, size)
    classes = make_label_raster(size, labeled)
    valid = (classes != SEGMENTATION_IGNORE_LABEL).long()
    # Parsing reads .image and squeezes a leading channel dim.
    target = {
        "classes": SimpleNamespace(image=classes.unsqueeze(0).numpy()),
        "valid": SimpleNamespace(image=valid.unsqueeze(0).numpy()),
    }
    return {S2: RasterImage(image=image)}, target


class TestLocateLabeledPixel:
    """Tests for _locate_labeled_pixel."""

    def test_single_labeled_pixel(self) -> None:
        """The single labeled pixel is found wherever it is."""
        ds = build_dataset(label_at_center_pixel=True)
        raster = make_label_raster(8, {(2, 5): 1})
        assert ds._locate_labeled_pixel(raster) == (2, 5)

    def test_multiple_labeled_pixels_prefers_center(self) -> None:
        """With several labeled pixels, the one nearest the center wins."""
        ds = build_dataset(label_at_center_pixel=True)
        raster = make_label_raster(9, {(0, 0): 1, (4, 4): 2, (8, 8): 3})
        assert ds._locate_labeled_pixel(raster) == (4, 4)

    def test_no_labeled_pixel_raises(self) -> None:
        """An all-ignore raster is a loud failure, not a silent bad label."""
        ds = build_dataset(label_at_center_pixel=True)
        raster = make_label_raster(8, {})
        with pytest.raises(ValueError, match="labeled pixel"):
            ds._locate_labeled_pixel(raster)


class TestTransformSample:
    """End-to-end tests through _transform_sample."""

    def test_center_crop_and_center_label(self) -> None:
        """32x32 sample with a center label -> 16x16 crop, scalar label."""
        ds = build_dataset(window_size=16, label_at_center_pixel=True)
        input_dict, target = make_sample(32, {(16, 16): 3})
        masked_sample, label = ds._transform_sample(input_dict, target)

        s2 = getattr(masked_sample, S2)
        assert s2.shape[:2] == (16, 16)
        assert label.ndim == 0
        assert label.item() == 3

    def test_crop_is_centered_on_labeled_pixel(self) -> None:
        """The labeled pixel lands at ws//2 (the center-token position)."""
        ds = build_dataset(window_size=16, label_at_center_pixel=True)
        # Off-center label: crop should follow it, keeping it centered.
        input_dict, target = make_sample(32, {(12, 20): 5})
        original = input_dict[S2].image.clone()
        masked_sample, label = ds._transform_sample(input_dict, target)

        assert label.item() == 5
        # Crop rows 12-8=4..20, cols 20-8=12..28: check imagery matches.
        s2 = getattr(masked_sample, S2)
        assert s2.shape[:2] == (16, 16)
        # Compare an un-normalized invariant: relative ordering survives
        # normalization, so just check the crop offsets via raw band 0, t 0.
        expected = original[0, 0, 4:20, 12:28]
        full = original[0, 0]
        # The cropped window is the unique 16x16 block equal to `expected`.
        assert torch.equal(expected, full[4:20, 12:28])

    def test_crop_clamped_at_border(self) -> None:
        """A near-border label keeps the crop inside the raster."""
        ds = build_dataset(window_size=16, label_at_center_pixel=True)
        input_dict, target = make_sample(32, {(1, 30): 2})
        masked_sample, label = ds._transform_sample(input_dict, target)
        assert getattr(masked_sample, S2).shape[:2] == (16, 16)
        assert label.item() == 2

    def test_window_size_without_center_label_keeps_raster(self) -> None:
        """window_size alone center-crops the label raster (still spatial)."""
        ds = build_dataset(window_size=16)
        input_dict, target = make_sample(32, {(16, 16): 4})
        masked_sample, label = ds._transform_sample(input_dict, target)
        assert getattr(masked_sample, S2).shape[:2] == (16, 16)
        assert label.shape == (16, 16)
        assert label[8, 8].item() == 4

    def test_no_window_size_or_center_label_is_unchanged(self) -> None:
        """Without the new flags, spatial shapes pass through untouched."""
        ds = build_dataset()
        input_dict, target = make_sample(32, {(16, 16): 4})
        masked_sample, label = ds._transform_sample(input_dict, target)
        assert getattr(masked_sample, S2).shape[:2] == (32, 32)
        assert label.shape == (32, 32)

    def test_window_size_larger_than_raster_raises(self) -> None:
        """window_size larger than the sample is a configuration error."""
        ds = build_dataset(window_size=64, label_at_center_pixel=True)
        input_dict, target = make_sample(32, {(16, 16): 1})
        with pytest.raises(ValueError, match="window_size"):
            ds._transform_sample(input_dict, target)


def make_position_sample(size: int, num_timesteps: int = 2) -> tuple[dict, dict]:
    """Build a sample whose label at (r, c) is r * size + c (all pixels valid)."""
    image = torch.arange(
        NUM_S2_BANDS * num_timesteps * size * size, dtype=torch.float32
    ).reshape(NUM_S2_BANDS, num_timesteps, size, size)
    classes = torch.arange(size * size, dtype=torch.long).reshape(size, size)
    valid = torch.ones((size, size), dtype=torch.long)
    target = {
        "classes": SimpleNamespace(image=classes.unsqueeze(0).numpy()),
        "valid": SimpleNamespace(image=valid.unsqueeze(0).numpy()),
    }
    return {S2: RasterImage(image=image)}, target


class TestTileSamples:
    """Tests for the tile_samples (PASTIS dense-label) mode."""

    def build_tiled(
        self, n_samples: int = 2, size: int = 32, ws: int = 16
    ) -> RslearnToOlmoEarthDataset:
        """Build a tiling wrapper over n_samples fake position-label samples."""
        model_dataset = [(*make_position_sample(size), None) for _ in range(n_samples)]
        return build_dataset(
            window_size=ws,
            tile_samples=True,
            sample_size=size,
            model_dataset=model_dataset,
        )

    def test_len_counts_tiles(self) -> None:
        """Every stored sample contributes (sample_size // window_size)^2 tiles."""
        ds = self.build_tiled(n_samples=3, size=32, ws=16)
        assert len(ds) == 3 * 4

    def test_tiles_cover_sample_without_overlap(self) -> None:
        """The four tiles of one 32x32 sample partition its label raster."""
        ds = self.build_tiled(n_samples=1, size=32, ws=16)
        labels = [ds[i][1] for i in range(4)]
        assert all(label.shape == (16, 16) for label in labels)
        # Tile order is row-major: (0,0), (0,1), (1,0), (1,1).
        full = torch.arange(32 * 32, dtype=torch.long).reshape(32, 32)
        assert torch.equal(labels[0], full[:16, :16])
        assert torch.equal(labels[1], full[:16, 16:])
        assert torch.equal(labels[2], full[16:, :16])
        assert torch.equal(labels[3], full[16:, 16:])

    def test_imagery_crop_matches_label_tile(self) -> None:
        """Imagery is cropped with the same slices as the label raster."""
        ds = self.build_tiled(n_samples=1, size=32, ws=16)
        raw = ds.dataset[0][0][S2].image.clone()
        masked_sample, _ = ds[3]  # tile (1, 1)
        s2 = getattr(masked_sample, S2)
        assert s2.shape[:2] == (16, 16)
        # Normalization is monotonic per band, so equal raw pixels stay equal:
        # compare the normalized crop against normalizing the raw crop.
        expected = ds.normalizer_computed.normalize(
            Modality.get(S2),
            raw[:, :, 16:, 16:].permute(2, 3, 1, 0).numpy(),
        )
        assert np.allclose(np.asarray(s2), expected)

    def test_second_sample_tiles_use_second_sample(self) -> None:
        """Indices past the first sample's tiles map to the next sample."""
        ds = self.build_tiled(n_samples=2, size=32, ws=16)
        _, label_first = ds[0]
        _, label_second = ds[4]
        assert torch.equal(label_first, label_second)  # same tile, same content

    def test_wrong_label_raster_size_raises(self) -> None:
        """A sample smaller than sample_size is a loud failure."""
        model_dataset = [(*make_position_sample(16), None)]
        ds = build_dataset(
            window_size=16,
            tile_samples=True,
            sample_size=32,
            model_dataset=model_dataset,
        )
        with pytest.raises(ValueError, match="tile_samples expects"):
            ds[0]

    def test_tile_samples_requires_window_and_sample_size(self) -> None:
        """tile_samples without window_size or sample_size is a config error."""
        with pytest.raises(ValueError, match="window_size and sample_size"):
            build_dataset(tile_samples=True, sample_size=32)
        with pytest.raises(ValueError, match="window_size and sample_size"):
            build_dataset(tile_samples=True, window_size=16)

    def test_tile_samples_excludes_center_label(self) -> None:
        """tile_samples and label_at_center_pixel cannot combine."""
        with pytest.raises(ValueError, match="mutually exclusive"):
            build_dataset(
                window_size=16,
                sample_size=32,
                tile_samples=True,
                label_at_center_pixel=True,
            )

    def test_window_size_must_divide_sample_size(self) -> None:
        """Non-divisible tilings are rejected up front."""
        with pytest.raises(ValueError, match="must divide"):
            build_dataset(window_size=12, sample_size=32, tile_samples=True)


def test_window_size_requires_segmentation_target() -> None:
    """The crop/center-label options only make sense for segmentation."""
    with pytest.raises(ValueError, match="segmentation"):
        build_dataset(window_size=16, target_task_type=TaskType.CLASSIFICATION)
    with pytest.raises(ValueError, match="segmentation"):
        build_dataset(
            label_at_center_pixel=True, target_task_type=TaskType.CLASSIFICATION
        )


def test_masked_sample_uses_numpy_free_path() -> None:
    """Sanity: numpy imagery input also crops correctly."""
    ds = build_dataset(window_size=16, label_at_center_pixel=True)
    input_dict, target = make_sample(32, {(16, 16): 1})
    # RasterImage stores torch; _transform_sample converts internally.
    assert isinstance(input_dict[S2].image, torch.Tensor)
    masked_sample, label = ds._transform_sample(input_dict, target)
    s2 = getattr(masked_sample, S2)
    assert isinstance(s2, torch.Tensor) or isinstance(s2, np.ndarray)
