"""Unit tests for create_windows_from_corpus.

These don't touch Sentinel-2 / NAIP, so they run fully offline: `verify_s2=False`.
"""

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest
from upath import UPath

from olmoearth_pretrain.dataset_creation.constants import WINDOW_RESOLUTIONS
from olmoearth_pretrain.dataset_creation.create_windows.util import (
    STATUS_CREATED,
    FALLBACK_RESOLUTION,
    create_windows_from_corpus,
)


def _bootstrap_minimal_dataset(ds_path: Path) -> None:
    """Write the minimum config.json an rslearn Dataset needs to load."""
    ds_path.mkdir(parents=True, exist_ok=True)
    (ds_path / "config.json").write_text(json.dumps({"layers": {}}))


@pytest.fixture
def ds_path(tmp_path: Path) -> UPath:
    ds = tmp_path / "rslearn_ds"
    _bootstrap_minimal_dataset(ds)
    return UPath(ds)


def test_create_windows_from_corpus_basic(ds_path: UPath) -> None:
    """5 synthetic points -> expected windows at every resolution >= 10m."""
    entries = [
        (-122.40, 37.77, datetime(2022, 6, 1, tzinfo=UTC)),  # SF
        (2.35, 48.86, datetime(2023, 4, 15, tzinfo=UTC)),    # Paris
        (139.69, 35.69, datetime(2021, 9, 10, tzinfo=UTC)),  # Tokyo
        (-46.63, -23.55, datetime(2022, 12, 1, tzinfo=UTC)), # Sao Paulo
        (151.21, -33.87, datetime(2020, 3, 20, tzinfo=UTC)), # Sydney
    ]

    results = create_windows_from_corpus(
        ds_path=ds_path,
        corpus_entries=entries,
        verify_s2=False,
        workers=2,
    )

    # One result per input row, all created, center times preserved.
    assert len(results) == len(entries)
    for (_, _, entry_time), (tile, selected_time, status) in zip(entries, results):
        assert status == STATUS_CREATED
        assert selected_time == entry_time
        assert tile.resolution == FALLBACK_RESOLUTION

    # Each unique fallback tile should have produced a window at every resolution
    # >= FALLBACK_RESOLUTION.
    expected_resolutions = [r for r in WINDOW_RESOLUTIONS if r >= FALLBACK_RESOLUTION]
    windows_root = ds_path / "windows"
    for r in expected_resolutions:
        group_dir = windows_root / f"res_{r}"
        assert group_dir.exists(), f"missing group dir for resolution {r}"
        window_dirs = [p for p in group_dir.iterdir() if p.is_dir()]
        # 5 scattered points -> 5 distinct fallback cells at 10m resolution.
        # At coarser resolutions they may collapse if two points are near each other,
        # but these 5 cities are far apart so we always expect 5.
        assert len(window_dirs) == 5, (
            f"resolution {r}: expected 5 windows, got {len(window_dirs)}"
        )


def test_create_windows_from_corpus_deduplicates_same_tile(ds_path: UPath) -> None:
    """Two corpus points in the same fallback cell -> one window, both marked created."""
    entries = [
        (-122.4000, 37.7700, datetime(2022, 6, 1, tzinfo=UTC)),
        (-122.40001, 37.77001, datetime(2099, 1, 1, tzinfo=UTC)),  # ignored time
    ]

    results = create_windows_from_corpus(
        ds_path=ds_path,
        corpus_entries=entries,
        verify_s2=False,
        workers=1,
    )

    # Both rows land in the same fallback tile; both report status created.
    assert len(results) == 2
    assert all(r[2] == STATUS_CREATED for r in results)
    assert results[0][0] == results[1][0]

    # Only one window directory at the fallback resolution.
    fallback_group = ds_path / "windows" / f"res_{FALLBACK_RESOLUTION}"
    window_dirs = [p for p in fallback_group.iterdir() if p.is_dir()]
    assert len(window_dirs) == 1
