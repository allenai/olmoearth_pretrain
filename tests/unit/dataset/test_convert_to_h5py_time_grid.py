"""Tests for ConvertToH5py on a regular time grid (16 quarterly period mosaics)."""

from datetime import UTC, datetime, timedelta
from pathlib import Path

import h5py
import numpy as np
import pytest
from upath import UPath

from olmoearth_pretrain.data.constants import BandSet, Modality
from olmoearth_pretrain.dataset.convert_to_h5py import (
    ConvertToH5py,
    ConvertToH5pyConfig,
)
from olmoearth_pretrain.dataset.parse import (
    GridTile,
    ModalityImage,
    ModalityTile,
    TimeSpan,
)
from olmoearth_pretrain.dataset.sample import SampleInformation
from olmoearth_pretrain.dataset_creation.constants import WINDOW_DURATION
from tests.conftest import create_geotiff

CRS = "EPSG:32610"
GRID = GridTile(crs=CRS, resolution_factor=16, col=165, row=-1968)
WINDOW_START = datetime(2018, 6, 18, tzinfo=UTC)
PERIOD = timedelta(days=91)
NUM_PERIODS = 16


def period_images(periods: list[int]) -> list[ModalityImage]:
    """ModalityImages for the given grid periods (as convert_period_mosaic writes them)."""
    return [
        ModalityImage(WINDOW_START + k * PERIOD, WINDOW_START + (k + 1) * PERIOD)
        for k in periods
    ]


def make_sample(
    tmp_path: Path, s2_periods: list[int], s1_periods: list[int]
) -> SampleInformation:
    """A sample with S2 / S1 period stacks plus static WorldCover."""
    s2 = {
        BandSet(["B02", "B03", "B04", "B08"], 16): (256, 4),
        BandSet(["B05", "B06", "B07", "B8A", "B11", "B12"], 32): (128, 6),
        BandSet(["B01", "B09"], 64): (64, 2),
    }
    s2_files = {}
    for i, (band_set, (size, nb)) in enumerate(s2.items()):
        fname = tmp_path / f"s2_{i}.tif"
        create_geotiff(fname, size, size, 10 * 256 / size, CRS, nb * len(s2_periods))
        s2_files[band_set] = UPath(fname)
    s1_file = tmp_path / "s1.tif"
    create_geotiff(s1_file, 256, 256, 10, CRS, 2 * len(s1_periods))
    wc_file = tmp_path / "wc.tif"
    create_geotiff(wc_file, 256, 256, 10, CRS, 1)
    center_time = WINDOW_START + WINDOW_DURATION // 2

    def tile(modality, images, band_sets):  # type: ignore[no-untyped-def]
        return ModalityTile(
            grid_tile=GRID,
            images=images,
            center_time=center_time,
            band_sets=band_sets,
            modality=modality,
        )

    return SampleInformation(
        grid_tile=GRID,
        time_span=TimeSpan.YEAR,
        modalities={
            Modality.SENTINEL2_L2A: tile(
                Modality.SENTINEL2_L2A, period_images(s2_periods), s2_files
            ),
            Modality.SENTINEL1: tile(
                Modality.SENTINEL1,
                period_images(s1_periods),
                {BandSet(["vv", "vh"], 16): UPath(s1_file)},
            ),
            Modality.WORLDCOVER: tile(
                Modality.WORLDCOVER, [], {BandSet(["B1"], 16): UPath(wc_file)}
            ),
        },
    )


def make_converter(tmp_path: Path, **kwargs) -> ConvertToH5py:  # type: ignore[no-untyped-def]
    """ConvertToH5py on the 91 d x 16 grid, single process."""
    return ConvertToH5pyConfig(
        tile_path=str(tmp_path),
        supported_modality_names=["sentinel2_l2a", "sentinel1", "worldcover"],
        multiprocessed_h5_creation=False,
        compression="zstd",
        compression_opts=3,
        tile_size=128,
        min_num_timesteps=1,
        time_grid_period_days=91,
        time_grid_num_periods=NUM_PERIODS,
        **kwargs,
    ).build()


def test_time_grid_aligns_modalities_with_different_gaps(tmp_path: Path) -> None:
    """S2 and S1 with different missing periods land on the same 16-step grid."""
    s2_periods = [k for k in range(16) if k != 5]
    s1_periods = [k for k in range(16) if k not in (0, 9)]
    sample = make_sample(tmp_path, s2_periods, s1_periods)
    converter = make_converter(tmp_path)
    converter.multiprocessed_sample_processing = False
    samples = converter._filter_samples([sample])
    assert len(samples) == 1
    converter.prepare_h5_dataset(samples)
    assert converter.h5py_dir is not None
    assert converter.h5py_dir.name == "4"  # 4 sub-tiles of 128

    expected_dates = [WINDOW_START + k * PERIOD for k in range(16)]
    with h5py.File(converter.h5py_dir / "sample_0.h5", "r") as f:
        ts = f["timestamps"][()]
        assert ts.shape == (16, 3)
        assert [tuple(r) for r in ts] == [
            (d.day, d.month - 1, d.year) for d in expected_dates
        ]
        s2_mask = f["missing_timesteps_masks/sentinel2_l2a"][()]
        s1_mask = f["missing_timesteps_masks/sentinel1"][()]
        np.testing.assert_array_equal(np.flatnonzero(s2_mask), s2_periods)
        np.testing.assert_array_equal(np.flatnonzero(s1_mask), s1_periods)
        assert f["sentinel2_l2a"].shape == (128, 128, 15, 12)
        assert f["sentinel1"].shape == (128, 128, 14, 2)
        assert f["worldcover"].shape == (128, 128, 1, 1)


def test_time_grid_empty_first_period_is_kept(tmp_path: Path) -> None:
    """A period with no data in any modality stays in the sequence (all masks False)."""
    periods = list(range(1, 16))
    converter = make_converter(tmp_path)
    timestamps, masks = converter.get_timestamps_and_masks(
        make_sample(tmp_path, periods, periods)
    )
    assert len(timestamps) == 16
    assert not masks["sentinel2_l2a"][0] and masks["sentinel2_l2a"][1:].all()


def test_time_grid_rejects_off_grid_timestamps(tmp_path: Path) -> None:
    """A modality whose timestamps are off the grid is skipped by the filter."""
    sample = make_sample(tmp_path, list(range(16)), list(range(16)))
    sample.modalities[Modality.SENTINEL1].images[3] = ModalityImage(
        WINDOW_START + timedelta(days=100), WINDOW_START + timedelta(days=130)
    )
    converter = make_converter(tmp_path)
    converter.multiprocessed_sample_processing = False
    with pytest.raises(ValueError, match="time grid"):
        converter.get_timestamps_and_masks(sample)
    assert converter._filter_samples([sample]) == []


def test_default_mode_and_min_num_timesteps(tmp_path: Path) -> None:
    """Without a grid the longest-series timestamps are used; min_num_timesteps filters."""
    sample = make_sample(tmp_path, list(range(16)), list(range(1, 16)))
    converter = ConvertToH5py(
        tile_path=UPath(tmp_path),
        supported_modalities=[
            Modality.SENTINEL2_L2A,
            Modality.SENTINEL1,
            Modality.WORLDCOVER,
        ],
        multiprocessed_sample_processing=False,
        multiprocessed_h5_creation=False,
    )
    timestamps, masks = converter.get_timestamps_and_masks(sample)
    assert len(timestamps) == 16 and masks["sentinel1"].sum() == 15
    assert len(converter._filter_samples([sample])) == 1
    converter.min_num_timesteps = 17
    assert converter._filter_samples([sample]) == []


def test_config_requires_both_grid_fields() -> None:
    """time_grid_period_days and time_grid_num_periods go together."""
    with pytest.raises(ValueError):
        ConvertToH5pyConfig(
            tile_path="x",
            supported_modality_names=["sentinel2_l2a"],
            time_grid_period_days=91,
        ).build()
