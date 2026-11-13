"""Tests for the PANGAEA rslearn export's window tiling."""

from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pytest

from olmoearth_pretrain.evals.datasets.pangaea_rslearn_export import (
    ai4sf_tile_dates,
    ai4sf_windows,
    ctm_selected_timesteps,
    window_offsets,
)


@pytest.mark.parametrize("size", [128, 256, 300, 497, 807])
def test_window_offsets_cover_every_pixel_once(size: int) -> None:
    """Scored pixels across windows tile [0, size) exactly once."""
    hits = np.zeros(size, dtype=int)
    for start, first_new in window_offsets(size, 128):
        assert 0 <= start <= size - 128
        hits[start + first_new : start + 128] += 1
    assert (hits == 1).all()


def test_window_offsets_rejects_small_tile() -> None:
    """A tile smaller than one window cannot be cut."""
    with pytest.raises(ValueError):
        window_offsets(100, 128)


def test_ai4sf_tile_dates_accepts_separate_metadata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Acquisition metadata can be read from outside the downloaded source."""
    metadata_path = tmp_path / "tiles_asia.gpkg"
    opened_paths = []

    class Tiles:
        def __enter__(self) -> list[dict[str, dict[str, int | str]]]:
            return [{"properties": {"year": 2020, "month": 2, "id": 1, "country": "x"}}]

        def __exit__(self, *_: object) -> None:
            return None

    def open_metadata(path: Path) -> Tiles:
        opened_paths.append(path)
        return Tiles()

    monkeypatch.setattr(
        "olmoearth_pretrain.evals.datasets.pangaea_rslearn_export.fiona.open",
        open_metadata,
    )

    dates = ai4sf_tile_dates(tmp_path, metadata_path)

    assert opened_paths == [metadata_path]
    assert dates["1_x"] == (
        datetime(2020, 2, 1, tzinfo=UTC),
        datetime(2020, 2, 29, 23, 59, 59, tzinfo=UTC),
    )


def test_ai4sf_windows_falls_back_to_downloaded_val_split(tmp_path: Path) -> None:
    """The downloader's ``val`` directory is exported as the validate split."""
    image = tmp_path / "sentinel-2-asia/val/images/tile.tif"
    image.parent.mkdir(parents=True)
    image.touch()

    assert list(ai4sf_windows(tmp_path)) == [("val", "tile", image)]


def test_ctm_selected_timesteps_keeps_last_six_in_order() -> None:
    """The last six acquisitions are kept with imagery and dates aligned."""
    arr = np.arange(8).reshape(1, 1, 1, 8)
    dates = [datetime(2020, 1, day, tzinfo=UTC) for day in range(1, 9)]

    selected_arr, selected_dates = ctm_selected_timesteps(arr, dates)

    np.testing.assert_array_equal(selected_arr.ravel(), [2, 3, 4, 5, 6, 7])
    assert selected_dates == dates[2:]
