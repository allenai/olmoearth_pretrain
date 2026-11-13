"""Tests for the ERA5-Land hourly corpus helpers."""

import numpy as np
import pandas as pd
import pytest

from olmoearth_pretrain.dataset_creation.era5_hourly import plan
from olmoearth_pretrain.dataset_creation.era5_hourly.common import (
    BANDS,
    N_LAT,
    WINDOW_HOURS,
    deaccumulate,
    grid_cell,
    hour_index,
)
from olmoearth_pretrain.dataset_creation.era5_hourly.shards import (
    create_shard_file,
    write_rows,
)


def running_totals(amounts: np.ndarray, first_hour: int) -> np.ndarray:
    """ERA5-Land totals: accumulated from 00 UTC, the 00 UTC value closing the day."""
    totals = np.empty_like(amounts)
    total = 0.0
    for i, amount in enumerate(amounts):
        if (first_hour + i) % 24 == 1:
            total = 0.0
        total += amount
        totals[i] = total
    return totals


@pytest.mark.parametrize("first_hour", [0, 1, 22, 23])
def test_deaccumulate_recovers_hourly_amounts(first_hour: int) -> None:
    amounts = np.random.default_rng(0).random(80).astype(np.float64)
    out = deaccumulate(running_totals(amounts, first_hour), first_hour)
    np.testing.assert_allclose(out[1:], amounts[1:])
    if first_hour % 24 == 1:
        assert out[0] == amounts[0]
    else:
        assert np.isnan(out[0])


def test_grid_cell() -> None:
    ilat, ilon = grid_cell(
        np.array([90.0, -57.1, -57.16, 0.04]), np.array([-0.04, -0.06, 359.96, 180.0])
    )
    assert ilat.tolist() == [0, N_LAT - 1, N_LAT, 900]
    assert ilon.tolist() == [0, 3599, 0, 1800]


def test_window_hours_end_at_the_window_end() -> None:
    windows = pd.DataFrame(dict(
        window=["a"], crs=["EPSG:4326"], x=[10.0], y=[45.0],
        window_start=["2018-10-10T00:00:00+00:00"], window_end=["2020-01-01T00:00:00+00:00"],
    ))  # fmt: skip
    df = plan.add_cells_and_hours(windows, last_hour=10**7)
    row = df.iloc[0]
    assert row["end_hour"] == hour_index(np.datetime64("2020-01-01T00"))
    assert row["end_hour"] - row["first_hour"] == WINDOW_HOURS - 1
    assert row["first_time"] == pd.Timestamp("2019-01-01T01:00")
    assert row["status"] == "ok" and (row["ilat"], row["ilon"]) == (450, 100)


def test_assign_shards_packs_whole_tiles(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(plan, "SHARD_WINDOWS", 5)
    # Tiles (lat chunk, lon chunk) with 3, 2, 4 and 7 windows; one window is skipped.
    cells = [(0, 0)] * 3 + [(60, 0)] * 2 + [(0, 150)] * 4 + [(0, 350)] * 7 + [(0, 0)]
    df = pd.DataFrame(dict(
        window=[f"w{k:02d}" for k in range(len(cells))],
        ilat=[c[0] for c in cells], ilon=[c[1] for c in cells],
        first_hour=1000, end_hour=1000 + WINDOW_HOURS - 1,
        status=["ok"] * (len(cells) - 1) + ["south_of_store"],
    ))  # fmt: skip
    out, shards = plan.assign_shards(df)
    assert [s["n_windows"] for s in shards] == [5, 4, 5, 2]
    assert [len(s["tiles"]) for s in shards] == [2, 1, 1, 1]
    assert out.loc[out["status"] != "ok", "shard"].tolist() == [-1]
    for s in shards:
        rows = out.loc[out["shard"] == s["shard"], "row"].sort_values().tolist()
        assert rows == list(range(s["n_windows"]))
    assert shards[0]["tiles"][0]["time_chunks"] == list(
        range(0, (1000 + WINDOW_HOURS) // 1440 + 1)
    )


def test_write_rows_round_trip(tmp_path) -> None:
    path = tmp_path / "shard.npy"
    create_shard_file(path, 3)
    offset = np.load(path, mmap_mode="r").offset
    block = (
        np.random.default_rng(1)
        .random((2, WINDOW_HOURS, len(BANDS)))
        .astype(np.float32)
    )
    write_rows(path, offset + WINDOW_HOURS * len(BANDS) * 4, block)
    data = np.load(path)
    assert data.shape == (3, WINDOW_HOURS, len(BANDS))
    np.testing.assert_array_equal(data[1:], block)
    assert not data[0].any()
