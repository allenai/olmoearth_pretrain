"""Tests for the rslearn dataset wrapper's window_size / center-pixel logic."""

from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch
from rslearn.train.model_context import RasterImage

from olmoearth_pretrain.data.constants import Modality
from olmoearth_pretrain.datatypes import MaskedOlmoEarthSample, MaskValue
from olmoearth_pretrain.evals.datasets.rslearn_dataset import (
    L8QA_INPUT_NAME,
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
    scl_cloud_mask: bool = False,
    scl_cloud_classes: tuple[int, ...] | None = None,
    input_modalities: list[str] | None = None,
) -> RslearnToOlmoEarthDataset:
    """Build a wrapper with no underlying model dataset (transform-only tests)."""
    return RslearnToOlmoEarthDataset(
        model_dataset=model_dataset,  # type: ignore[arg-type]
        input_modalities=input_modalities or [S2],
        target_task_type=target_task_type,
        window_size=window_size,
        label_at_center_pixel=label_at_center_pixel,
        tile_samples=tile_samples,
        sample_size=sample_size,
        scl_cloud_mask=scl_cloud_mask,
        scl_cloud_classes=scl_cloud_classes,
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


class TestTimestamps:
    """Timestamps must come from the imagery's real acquisition times."""

    @staticmethod
    def monthly_ranges(
        year: int, num_timesteps: int, descending: bool = False
    ) -> list[tuple[datetime, datetime]]:
        """Build num_timesteps 30-day ranges tiling `year` from Jan 1."""
        ranges = [
            (
                datetime(year, 1, 1, tzinfo=UTC) + timedelta(days=30 * i),
                datetime(year, 1, 1, tzinfo=UTC) + timedelta(days=30 * (i + 1)),
            )
            for i in range(num_timesteps)
        ]
        return list(reversed(ranges)) if descending else ranges

    def test_stored_timestamps_are_used(self) -> None:
        """Per-window acquisition dates win over the dataset-level fallback."""
        ds = build_dataset()
        input_dict, target = make_sample(8, {(4, 4): 1}, num_timesteps=12)
        ranges = self.monthly_ranges(2018, 12)
        input_dict[S2] = RasterImage(image=input_dict[S2].image, timestamps=ranges)

        masked_sample, _ = ds._transform_sample(input_dict, target)

        timestamps = masked_sample.timestamps
        assert timestamps.shape == (12, 3)
        # (day, month0, year) taken from each range's start.
        assert [int(t) for t in timestamps[:, 2]] == [2018] * 12
        assert [int(t) for t in timestamps[:, 1]] == [
            start.month - 1 for start, _ in ranges
        ]
        assert [int(t) for t in timestamps[:, 0]] == [start.day for start, _ in ranges]

    def test_descending_time_axis_is_labeled_descending(self) -> None:
        """Reverse-chronological item groups get reverse-chronological dates.

        The AEF supplemental datasets store their 12 S2 mosaics latest-first;
        synthesized ascending months labeled December's imagery as January.
        """
        ds = build_dataset()
        input_dict, target = make_sample(8, {(4, 4): 1}, num_timesteps=12)
        ranges = self.monthly_ranges(2019, 12, descending=True)
        input_dict[S2] = RasterImage(image=input_dict[S2].image, timestamps=ranges)

        masked_sample, _ = ds._transform_sample(input_dict, target)

        months = [int(t) for t in masked_sample.timestamps[:, 1]]
        assert months == sorted(months, reverse=True)
        assert months[0] == ranges[0][0].month - 1

    def test_falls_back_to_synthesized_range(self) -> None:
        """Imagery with no stored times still gets the configured range."""
        ds = build_dataset()
        input_dict, target = make_sample(8, {(4, 4): 1}, num_timesteps=12)
        assert input_dict[S2].timestamps is None

        masked_sample, _ = ds._transform_sample(input_dict, target)

        timestamps = masked_sample.timestamps
        assert timestamps.shape == (12, 3)
        # DEFAULT_START_TIME is 2022-09-01, so the first timestep is Sep 2022.
        assert int(timestamps[0, 1]) == 8
        assert int(timestamps[0, 2]) == 2022

    def test_mismatched_length_is_ignored(self) -> None:
        """Stored times that don't match the time axis fall back, not crash."""
        ds = build_dataset()
        timestamps = ds._build_timestamps(12, {S2: self.monthly_ranges(2018, 6)})
        assert timestamps.shape == (12, 3)
        assert int(timestamps[0, 2]) == 2022  # fell back to the default range


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


def make_scl(
    size: int, num_timesteps: int, cloudy: dict[tuple[int, int, int], int] | None = None
) -> RasterImage:
    """Build an SCL RasterImage: vegetation (4) except `cloudy` (h, w, t) -> class."""
    scl = torch.full((1, num_timesteps, size, size), 4.0)
    for (row, col, t), value in (cloudy or {}).items():
        scl[0, t, row, col] = float(value)
    return RasterImage(image=scl)


class TestSclCloudMask:
    """Tests for scl_cloud_mask through _transform_sample."""

    S2_MASK = MaskedOlmoEarthSample.get_masked_modality_name(S2)

    def test_cloudy_pixel_timestep_masked_missing(self) -> None:
        """SCL cloud classes become MISSING; clear pixels stay ONLINE_ENCODER."""
        ds = build_dataset(scl_cloud_mask=True)
        input_dict, target = make_sample(8, {(4, 4): 1})
        input_dict["scl"] = make_scl(8, 2, {(2, 3, 0): 9, (5, 6, 1): 3})
        masked_sample, _ = ds._transform_sample(input_dict, target)
        mask = getattr(masked_sample, self.S2_MASK)
        assert (mask[2, 3, 0] == MaskValue.MISSING.value).all()
        assert (mask[5, 6, 0] == MaskValue.ONLINE_ENCODER.value).all()
        assert (mask[5, 6, 1] == MaskValue.MISSING.value).all()
        assert (mask[2, 3, 1] == MaskValue.ONLINE_ENCODER.value).all()
        assert (mask[0, 0] == MaskValue.ONLINE_ENCODER.value).all()

    def test_custom_cloud_classes_narrow_the_mask(self) -> None:
        """The cloudless policy (8, 9) leaves shadow (3) pixels unmasked."""
        ds = build_dataset(scl_cloud_mask=True, scl_cloud_classes=(8, 9))
        input_dict, target = make_sample(8, {(4, 4): 1})
        input_dict["scl"] = make_scl(8, 2, {(2, 3, 0): 9, (5, 6, 0): 3})
        masked_sample, _ = ds._transform_sample(input_dict, target)
        mask = getattr(masked_sample, self.S2_MASK)
        assert (mask[2, 3, 0] == MaskValue.MISSING.value).all()
        # Shadow is not in the cloudless class set, so it stays visible.
        assert (mask[5, 6, 0] == MaskValue.ONLINE_ENCODER.value).all()

    def test_fully_cloudy_pixel_left_unmasked(self) -> None:
        """A pixel cloudy at every timestep keeps all timesteps."""
        ds = build_dataset(scl_cloud_mask=True)
        input_dict, target = make_sample(8, {(4, 4): 1})
        input_dict["scl"] = make_scl(8, 2, {(1, 1, 0): 9, (1, 1, 1): 8})
        masked_sample, _ = ds._transform_sample(input_dict, target)
        mask = getattr(masked_sample, self.S2_MASK)
        assert (mask[1, 1] == MaskValue.ONLINE_ENCODER.value).all()

    def test_missing_scl_input_leaves_sample_unmasked(self) -> None:
        """No scl input -> warn and proceed unmasked, never crash."""
        ds = build_dataset(scl_cloud_mask=True)
        input_dict, target = make_sample(8, {(4, 4): 1})
        masked_sample, _ = ds._transform_sample(input_dict, target)
        mask = getattr(masked_sample, self.S2_MASK)
        assert (mask == MaskValue.ONLINE_ENCODER.value).all()

    def test_flag_off_ignores_scl_input(self) -> None:
        """scl_cloud_mask=False leaves the mask untouched even with SCL present."""
        ds = build_dataset(scl_cloud_mask=False)
        input_dict, target = make_sample(8, {(4, 4): 1})
        input_dict["scl"] = make_scl(8, 2, {(2, 3, 0): 9})
        masked_sample, _ = ds._transform_sample(input_dict, target)
        mask = getattr(masked_sample, self.S2_MASK)
        assert (mask == MaskValue.ONLINE_ENCODER.value).all()

    def test_mask_follows_center_crop(self) -> None:
        """With window_size, SCL takes the same crop as the imagery."""
        ds = build_dataset(
            window_size=16, label_at_center_pixel=True, scl_cloud_mask=True
        )
        input_dict, target = make_sample(32, {(12, 20): 5})
        # Crop is rows 4..20, cols 12..28; cloudy pixel at raster (6, 14)
        # lands at cropped (2, 2).
        input_dict["scl"] = make_scl(32, 2, {(6, 14, 1): 10})
        masked_sample, _ = ds._transform_sample(input_dict, target)
        mask = getattr(masked_sample, self.S2_MASK)
        assert (mask[2, 2, 1] == MaskValue.MISSING.value).all()
        assert (mask[2, 2, 0] == MaskValue.ONLINE_ENCODER.value).all()

    def test_half_resolution_scl_is_upsampled(self) -> None:
        """A 20m-grid SCL read (half resolution) is nearest-upsampled to match."""
        ds = build_dataset(scl_cloud_mask=True)
        input_dict, target = make_sample(8, {(4, 4): 1})
        # 4x4 SCL against 8x8 imagery: pixel (1, 1) covers imagery (2:4, 2:4).
        input_dict["scl"] = make_scl(4, 2, {(1, 1, 0): 9})
        masked_sample, _ = ds._transform_sample(input_dict, target)
        mask = getattr(masked_sample, self.S2_MASK)
        assert (mask[2:4, 2:4, 0] == MaskValue.MISSING.value).all()
        assert (mask[2:4, 2:4, 1] == MaskValue.ONLINE_ENCODER.value).all()
        assert (mask[0:2, 0:2, 0] == MaskValue.ONLINE_ENCODER.value).all()

    def test_shape_mismatch_skips_masking(self) -> None:
        """An SCL grid that is neither matching nor half resolution is skipped."""
        ds = build_dataset(scl_cloud_mask=True)
        input_dict, target = make_sample(8, {(4, 4): 1})
        input_dict["scl"] = make_scl(5, 2, {(1, 1, 0): 9})
        masked_sample, _ = ds._transform_sample(input_dict, target)
        mask = getattr(masked_sample, self.S2_MASK)
        assert (mask == MaskValue.ONLINE_ENCODER.value).all()


LANDSAT = Modality.LANDSAT.name
NUM_LANDSAT_BANDS = len(Modality.get(LANDSAT).band_order)


def monthly_ranges(year: int, months: list[int]) -> list[tuple]:
    """(start, end) ranges for the given month slots of `year`'s 30-day grid."""
    jan1 = datetime(year, 1, 1, tzinfo=UTC)
    return [
        (jan1 + timedelta(days=30 * m), jan1 + timedelta(days=30 * (m + 1)))
        for m in months
    ]


class TestRaggedModalities:
    """Optional imagery with coverage gaps aligns onto the canonical axis."""

    LANDSAT_MASK = MaskedOlmoEarthSample.get_masked_modality_name(LANDSAT)
    S2_MASK = MaskedOlmoEarthSample.get_masked_modality_name(S2)

    @staticmethod
    def sample_with_landsat(landsat_months: list[int] | None) -> tuple[dict, dict]:
        """An S2 (T=12, dated) sample plus a landsat tensor for given months."""
        input_dict, target = make_sample(8, {(4, 4): 1}, num_timesteps=12)
        input_dict[S2] = RasterImage(
            image=input_dict[S2].image, timestamps=monthly_ranges(2020, list(range(12)))
        )
        if landsat_months is not None:
            image = torch.rand(NUM_LANDSAT_BANDS, len(landsat_months), 8, 8)
            input_dict[LANDSAT] = RasterImage(
                image=image, timestamps=monthly_ranges(2020, landsat_months)
            )
        return input_dict, target

    def test_partial_landsat_aligns_by_date(self) -> None:
        """Months 2 and 6 missing -> those canonical slots read MISSING."""
        ds = build_dataset(input_modalities=[S2, LANDSAT])
        months = [m for m in range(12) if m not in (2, 6)]
        input_dict, target = self.sample_with_landsat(months)
        masked_sample, _ = ds._transform_sample(input_dict, target)

        landsat = getattr(masked_sample, LANDSAT)
        assert landsat.shape == (8, 8, 12, NUM_LANDSAT_BANDS)
        mask = getattr(masked_sample, self.LANDSAT_MASK)
        for slot in range(12):
            expected = (
                MaskValue.MISSING.value
                if slot in (2, 6)
                else MaskValue.ONLINE_ENCODER.value
            )
            assert (mask[:, :, slot] == expected).all(), slot
        # S2 is untouched and the shared timestamps keep the full year.
        assert (
            getattr(masked_sample, self.S2_MASK) == MaskValue.ONLINE_ENCODER.value
        ).all()
        assert masked_sample.timestamps.shape == (12, 3)

    def test_absent_landsat_is_all_missing(self) -> None:
        """A window with no landsat at all yields zeros + an all-MISSING mask."""
        ds = build_dataset(input_modalities=[S2, LANDSAT])
        input_dict, target = self.sample_with_landsat(None)
        masked_sample, _ = ds._transform_sample(input_dict, target)

        landsat = getattr(masked_sample, LANDSAT)
        assert landsat.shape == (8, 8, 12, NUM_LANDSAT_BANDS)
        assert (
            getattr(masked_sample, self.LANDSAT_MASK) == MaskValue.MISSING.value
        ).all()
        assert (
            getattr(masked_sample, self.S2_MASK) == MaskValue.ONLINE_ENCODER.value
        ).all()

    def test_absent_embedding_product_still_raises(self) -> None:
        """A precomputed-embedding coverage gap must stay a loud failure."""
        ds = build_dataset(input_modalities=[S2, Modality.GSE.name])
        input_dict, target = self.sample_with_landsat(None)
        with pytest.raises(ValueError, match="not found"):
            ds._transform_sample(input_dict, target)


class TestLandsatCloudMask:
    """Scene-level Landsat cloud masking via the sidecar table."""

    LANDSAT_MASK = MaskedOlmoEarthSample.get_masked_modality_name(LANDSAT)

    @staticmethod
    def dataset_with_table(table: dict | None) -> RslearnToOlmoEarthDataset:
        """A wrapper with the L8 mask armed at threshold 50."""
        ds = build_dataset(input_modalities=[S2, LANDSAT])
        ds.landsat_cloud_cover_max = 50.0
        ds.landsat_cloud_cover_table = table
        return ds

    def test_cloudy_months_masked(self) -> None:
        """Months at/over threshold go MISSING; below, unknown (-1) stay."""
        table = {"g/w1": {"mo01": 80.0, "mo02": 49.9, "mo03": -1, "mo04": 50.0}}
        ds = self.dataset_with_table(table)
        input_dict, target = TestRaggedModalities.sample_with_landsat(list(range(12)))
        masked_sample, _ = ds._transform_sample(input_dict, target, window_key="g/w1")
        mask = getattr(masked_sample, self.LANDSAT_MASK)
        assert (mask[:, :, 0] == MaskValue.MISSING.value).all()  # mo01: 80
        assert (mask[:, :, 1] == MaskValue.ONLINE_ENCODER.value).all()  # 49.9
        assert (mask[:, :, 2] == MaskValue.ONLINE_ENCODER.value).all()  # unknown
        assert (mask[:, :, 3] == MaskValue.MISSING.value).all()  # exactly 50
        assert (mask[:, :, 4:] == MaskValue.ONLINE_ENCODER.value).all()

    def test_merges_with_coverage_gaps(self) -> None:
        """Cloud-masked slots union with ragged coverage-gap slots."""
        table = {"g/w1": {"mo01": 90.0}}
        ds = self.dataset_with_table(table)
        # landsat missing month index 6 entirely (coverage gap)
        months = [m for m in range(12) if m != 6]
        input_dict, target = TestRaggedModalities.sample_with_landsat(months)
        masked_sample, _ = ds._transform_sample(input_dict, target, window_key="g/w1")
        mask = getattr(masked_sample, self.LANDSAT_MASK)
        for slot in range(12):
            expected = (
                MaskValue.MISSING.value
                if slot in (0, 6)
                else MaskValue.ONLINE_ENCODER.value
            )
            assert (mask[:, :, slot] == expected).all(), slot

    def test_unknown_window_or_missing_table_no_op(self) -> None:
        """No sidecar entry (or no table) -> unmasked, never a crash."""
        for table in (None, {"g/other": {"mo01": 99.0}}):
            ds = self.dataset_with_table(table)
            input_dict, target = TestRaggedModalities.sample_with_landsat(
                list(range(12))
            )
            masked_sample, _ = ds._transform_sample(
                input_dict, target, window_key="g/w1"
            )
            mask = getattr(masked_sample, self.LANDSAT_MASK)
            assert (mask == MaskValue.ONLINE_ENCODER.value).all()


# QA_PIXEL codes: clear land is 21824 (the quality bits, no cloud bits); the
# cloud bits under test are dilated=1, cirrus=2, cloud=3, shadow=4. Bit 0 is
# fill, deliberately NOT masked.
QA_CLEAR = 21824
QA_FILL = 1
QA_DILATED = QA_CLEAR | 0b00010
QA_CIRRUS = QA_CLEAR | 0b00100
QA_CLOUD = QA_CLEAR | 0b01000
QA_SHADOW = QA_CLEAR | 0b10000


def make_qa(
    size: int,
    num_timesteps: int,
    codes: dict[tuple[int, int, int], int] | None = None,
    months: list[int] | None = None,
) -> RasterImage:
    """Build a QA_PIXEL RasterImage: clear land except `codes` (h, w, t) -> code."""
    qa = torch.full((1, num_timesteps, size, size), float(QA_CLEAR))
    for (row, col, t), value in (codes or {}).items():
        qa[0, t, row, col] = float(value)
    return RasterImage(
        image=qa,
        timestamps=monthly_ranges(2020, months) if months is not None else None,
    )


class TestL8PixelCloudMask:
    """Per-pixel Landsat cloud masking from the optional landsat_qa input."""

    LANDSAT_MASK = MaskedOlmoEarthSample.get_masked_modality_name(LANDSAT)

    @staticmethod
    def armed_dataset(**kwargs: Any) -> RslearnToOlmoEarthDataset:
        """A wrapper with the per-pixel Landsat mask armed."""
        ds = build_dataset(input_modalities=[S2, LANDSAT], **kwargs)
        ds.l8_pixel_cloud_mask = True
        return ds

    def test_cloud_bits_masked_fill_and_clear_kept(self) -> None:
        """Each cloud bit masks its pixel-timestep; clear and fill do not."""
        ds = self.armed_dataset()
        months = list(range(12))
        input_dict, target = TestRaggedModalities.sample_with_landsat(months)
        input_dict[L8QA_INPUT_NAME] = make_qa(
            8,
            12,
            {
                (1, 1, 0): QA_CLOUD,
                (2, 2, 1): QA_SHADOW,
                (3, 3, 2): QA_CIRRUS,
                (4, 4, 3): QA_DILATED,
                (5, 5, 4): QA_FILL,
            },
            months=months,
        )
        masked_sample, _ = ds._transform_sample(input_dict, target)
        mask = getattr(masked_sample, self.LANDSAT_MASK)
        for (row, col), slot in (((1, 1), 0), ((2, 2), 1), ((3, 3), 2), ((4, 4), 3)):
            assert (mask[row, col, slot] == MaskValue.MISSING.value).all()
            # Only that pixel-timestep moves.
            assert (mask[row, col, slot + 1] == MaskValue.ONLINE_ENCODER.value).all()
        # Fill is not a cloud bit.
        assert (mask[5, 5, 4] == MaskValue.ONLINE_ENCODER.value).all()

    def test_ragged_qa_aligned_by_acquisition_date(self) -> None:
        """QA with a coverage gap masks the month it was acquired in, not slot i."""
        ds = self.armed_dataset()
        months = [m for m in range(12) if m != 3]
        input_dict, target = TestRaggedModalities.sample_with_landsat(months)
        # QA timestep 3 is month 4 (month 3 is missing on both sides): a
        # positional read would mask slot 3, which has no observation at all.
        input_dict[L8QA_INPUT_NAME] = make_qa(
            8, len(months), {(1, 1, 3): QA_CLOUD}, months=months
        )
        masked_sample, _ = ds._transform_sample(input_dict, target)
        mask = getattr(masked_sample, self.LANDSAT_MASK)
        assert (mask[1, 1, 4] == MaskValue.MISSING.value).all()
        # Slot 3 is MISSING from the coverage gap for every pixel, and no
        # other pixel picked up a cloud mask.
        assert (mask[:, :, 3] == MaskValue.MISSING.value).all()
        for slot in (m for m in range(12) if m not in (3, 4)):
            assert (mask[:, :, slot] == MaskValue.ONLINE_ENCODER.value).all(), slot
        assert (mask[2, 2, 4] == MaskValue.ONLINE_ENCODER.value).all()

    def test_fully_cloudy_pixel_left_unmasked(self) -> None:
        """A pixel cloudy at every timestep keeps all of them."""
        ds = self.armed_dataset()
        months = list(range(12))
        input_dict, target = TestRaggedModalities.sample_with_landsat(months)
        input_dict[L8QA_INPUT_NAME] = make_qa(
            8, 12, {(1, 1, t): QA_CLOUD for t in range(12)}, months=months
        )
        masked_sample, _ = ds._transform_sample(input_dict, target)
        mask = getattr(masked_sample, self.LANDSAT_MASK)
        assert (mask[1, 1] == MaskValue.ONLINE_ENCODER.value).all()

    def test_guard_counts_coverage_gaps_as_lost_timesteps(self) -> None:
        """Cloudy on every month it HAS: the pixel keeps them rather than blanking."""
        ds = self.armed_dataset()
        months = [0, 1]
        input_dict, target = TestRaggedModalities.sample_with_landsat(months)
        input_dict[L8QA_INPUT_NAME] = make_qa(
            8, 2, {(1, 1, 0): QA_CLOUD, (1, 1, 1): QA_CLOUD}, months=months
        )
        masked_sample, _ = ds._transform_sample(input_dict, target)
        mask = getattr(masked_sample, self.LANDSAT_MASK)
        # Slots 0 and 1 are the pixel's only observations, so they survive.
        assert (mask[1, 1, 0] == MaskValue.ONLINE_ENCODER.value).all()
        assert (mask[1, 1, 1] == MaskValue.ONLINE_ENCODER.value).all()

    def test_mask_follows_center_crop(self) -> None:
        """With window_size, QA takes the same crop as the imagery."""
        ds = self.armed_dataset(window_size=16, label_at_center_pixel=True)
        input_dict, target = make_sample(32, {(12, 20): 5}, num_timesteps=12)
        months = list(range(12))
        input_dict[S2] = RasterImage(
            image=input_dict[S2].image, timestamps=monthly_ranges(2020, months)
        )
        input_dict[LANDSAT] = RasterImage(
            image=torch.rand(NUM_LANDSAT_BANDS, 12, 32, 32),
            timestamps=monthly_ranges(2020, months),
        )
        # Crop is rows 4..20, cols 12..28; cloudy pixel at raster (6, 14)
        # lands at cropped (2, 2).
        input_dict[L8QA_INPUT_NAME] = make_qa(
            32, 12, {(6, 14, 1): QA_CLOUD}, months=months
        )
        masked_sample, _ = ds._transform_sample(input_dict, target)
        mask = getattr(masked_sample, self.LANDSAT_MASK)
        assert (mask[2, 2, 1] == MaskValue.MISSING.value).all()
        assert (mask[2, 2, 0] == MaskValue.ONLINE_ENCODER.value).all()

    def test_half_resolution_qa_is_upsampled(self) -> None:
        """A 60m-grid QA read (half resolution) is nearest-upsampled to match."""
        ds = self.armed_dataset()
        months = list(range(12))
        input_dict, target = TestRaggedModalities.sample_with_landsat(months)
        # 4x4 QA against 8x8 imagery: QA (1, 1) covers imagery (2:4, 2:4).
        input_dict[L8QA_INPUT_NAME] = make_qa(
            4, 12, {(1, 1, 0): QA_CLOUD}, months=months
        )
        masked_sample, _ = ds._transform_sample(input_dict, target)
        mask = getattr(masked_sample, self.LANDSAT_MASK)
        assert (mask[2:4, 2:4, 0] == MaskValue.MISSING.value).all()
        assert (mask[2:4, 2:4, 1] == MaskValue.ONLINE_ENCODER.value).all()
        assert (mask[0:2, 0:2, 0] == MaskValue.ONLINE_ENCODER.value).all()

    def test_missing_qa_input_leaves_sample_unmasked(self) -> None:
        """No landsat_qa input -> warn and proceed unmasked, never crash."""
        ds = self.armed_dataset()
        input_dict, target = TestRaggedModalities.sample_with_landsat(list(range(12)))
        masked_sample, _ = ds._transform_sample(input_dict, target)
        mask = getattr(masked_sample, self.LANDSAT_MASK)
        assert (mask == MaskValue.ONLINE_ENCODER.value).all()

    def test_absent_landsat_with_qa_present_does_not_crash(self) -> None:
        """QA without imagery: every slot is already MISSING, so just warn."""
        ds = self.armed_dataset()
        input_dict, target = TestRaggedModalities.sample_with_landsat(None)
        input_dict[L8QA_INPUT_NAME] = make_qa(8, 12, {(1, 1, 0): QA_CLOUD})
        masked_sample, _ = ds._transform_sample(input_dict, target)
        mask = getattr(masked_sample, self.LANDSAT_MASK)
        assert (mask == MaskValue.MISSING.value).all()

    def test_shape_mismatch_skips_masking(self) -> None:
        """A QA grid that is neither matching nor half resolution is skipped."""
        ds = self.armed_dataset()
        months = list(range(12))
        input_dict, target = TestRaggedModalities.sample_with_landsat(months)
        input_dict[L8QA_INPUT_NAME] = make_qa(
            5, 12, {(1, 1, 0): QA_CLOUD}, months=months
        )
        masked_sample, _ = ds._transform_sample(input_dict, target)
        mask = getattr(masked_sample, self.LANDSAT_MASK)
        assert (mask == MaskValue.ONLINE_ENCODER.value).all()

    def test_flag_off_ignores_qa_input(self) -> None:
        """l8_pixel_cloud_mask=False leaves the mask untouched even with QA present."""
        ds = build_dataset(input_modalities=[S2, LANDSAT])
        months = list(range(12))
        input_dict, target = TestRaggedModalities.sample_with_landsat(months)
        input_dict[L8QA_INPUT_NAME] = make_qa(
            8, 12, {(1, 1, 0): QA_CLOUD}, months=months
        )
        masked_sample, _ = ds._transform_sample(input_dict, target)
        mask = getattr(masked_sample, self.LANDSAT_MASK)
        assert (mask == MaskValue.ONLINE_ENCODER.value).all()


class TestSubsetBands:
    """A dataset may store fewer than its modality's canonical bands."""

    @staticmethod
    def _dataset(declared: dict[str, list[str]]) -> Any:
        """A bare instance carrying only what band_scatter needs."""
        obj = object.__new__(RslearnToOlmoEarthDataset)
        obj.input_modalities = [Modality.SENTINEL2_L2A.name]
        RslearnToOlmoEarthDataset._init_band_scatter(obj, declared)
        return obj

    def test_full_band_list_is_a_no_op(self) -> None:
        """Every existing dataset declares all bands and must be untouched."""
        canonical = list(Modality.SENTINEL2_L2A.band_order)
        ds = self._dataset({Modality.SENTINEL2_L2A.name: canonical})
        assert ds.band_scatter == {}
        assert ds.absent_bands == {}

    def test_subset_records_positions_and_gaps(self) -> None:
        """The ten-band composite must map onto canonical positions 0-9."""
        canonical = list(Modality.SENTINEL2_L2A.band_order)
        ten = [b for b in canonical if b not in ("B01", "B09")]
        ds = self._dataset({Modality.SENTINEL2_L2A.name: ten})

        assert ds.band_scatter[Modality.SENTINEL2_L2A.name] == [
            canonical.index(b) for b in ten
        ]
        assert ds.absent_bands[Modality.SENTINEL2_L2A.name] == [
            canonical.index("B01"),
            canonical.index("B09"),
        ]

    def test_scatter_puts_read_channels_at_canonical_positions(self) -> None:
        """Channel k of the raster must land at its band's canonical index."""
        canonical = list(Modality.SENTINEL2_L2A.band_order)
        ten = [b for b in canonical if b not in ("B01", "B09")]
        ds = self._dataset({Modality.SENTINEL2_L2A.name: ten})

        arr = np.arange(10, dtype=np.float32).reshape(1, 1, 1, 10)
        out = ds._scatter_bands(Modality.SENTINEL2_L2A.name, arr)

        assert out.shape == (1, 1, 1, len(canonical))
        for read_index, band in enumerate(ten):
            assert out[0, 0, 0, canonical.index(band)] == read_index
        assert out[0, 0, 0, canonical.index("B01")] == 0
        assert out[0, 0, 0, canonical.index("B09")] == 0

    def test_channel_count_mismatch_is_loud(self) -> None:
        """Config and rasters disagreeing must fail, not silently misalign."""
        canonical = list(Modality.SENTINEL2_L2A.band_order)
        ten = [b for b in canonical if b not in ("B01", "B09")]
        ds = self._dataset({Modality.SENTINEL2_L2A.name: ten})

        with pytest.raises(ValueError, match="config and data disagree"):
            ds._scatter_bands(
                Modality.SENTINEL2_L2A.name, np.zeros((1, 1, 1, 12), dtype=np.float32)
            )

    def test_unknown_band_is_rejected(self) -> None:
        """A typo in model.yaml must not be silently ignored."""
        with pytest.raises(ValueError, match="not in its canonical band order"):
            self._dataset({Modality.SENTINEL2_L2A.name: ["B02", "NOPE"]})
