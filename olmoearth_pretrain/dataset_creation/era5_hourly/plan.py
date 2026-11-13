"""Stage 1: the window index and the shard layout.

Reads every window of the daily corpus (``WINDOWS_DIR``, read-only) and gives
it an ERA5-Land cell and an hourly span:

- cell: the grid point the window centre falls in, with no neighbour fallback;
- span: the ``WINDOW_HOURS`` (365 days) ending at the window's end time. Row k
  is store hour ``first_hour + k``; instantaneous bands are the value at that
  hour, accumulated bands the amount over the hour ending there.

A window gets no data (``status``) when its cell is south of the store
(57.1 S: every Antarctic window) or is not land in ERA5-Land (coasts and small
islands: the corpus took land from Natural Earth). Land is the weka daily
mirror's ``valid_mask``, on the same grid as the hourly store.

Usable windows are ordered by store tile (50 x 100 cells, longitude first) and
packed into shards of at most ``SHARD_WINDOWS`` windows made of whole tiles
(a larger tile is split), so a tile is downloaded by one job and a shard covers
one longitude band.

Outputs under ``--root``: ``index.parquet`` (one row per corpus window, with
its ``shard`` and ``row``, -1 when it gets no data), ``shards.json``,
``bands.json`` and ``README.md``. Existing outputs are kept: the shards are
built against them.

Usage::

    python -m olmoearth_pretrain.dataset_creation.era5_hourly.plan --workers 64
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pyproj import Transformer

from .common import (
    BANDS,
    DEFAULT_ROOT,
    EPOCH,
    ERA5_MIRROR,
    GRID_DEG,
    HOURLY_URL,
    LAT_CHUNK,
    LON_CHUNK,
    N_LAT,
    N_LON,
    NODATA,
    SHARD_WINDOWS,
    TIME_CHUNK,
    WINDOW_HOURS,
    WINDOWS_DIR,
    grid_cell,
    open_store,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def read_windows(windows_dir: Path, workers: int) -> pd.DataFrame:
    """One row per window: name, centre (projected), time range and options.

    Args:
        windows_dir: The corpus window group directory.
        workers: Reader threads.

    Returns:
        Windows sorted by name.
    """
    names = sorted(e.name for e in os.scandir(windows_dir))
    logger.info("%d windows in %s", len(names), windows_dir)

    def load(name: str) -> dict[str, Any]:
        with open(windows_dir / name / "metadata.json") as f:
            meta = json.load(f)
        proj, b = meta["projection"], meta["bounds"]
        return dict(
            window=name, crs=proj["crs"],
            x=(b[0] + b[2]) / 2 * proj["x_resolution"],
            y=(b[1] + b[3]) / 2 * proj["y_resolution"],
            window_start=meta["time_range"][0], window_end=meta["time_range"][1],
            **meta.get("options", {}),
        )  # fmt: skip

    rows = []
    with ThreadPoolExecutor(workers) as ex:
        for k, row in enumerate(ex.map(load, names)):
            rows.append(row)
            if (k + 1) % 25_000 == 0:
                logger.info("read %d / %d window metadata", k + 1, len(names))
    return pd.DataFrame(rows)


def add_cells_and_hours(windows: pd.DataFrame, last_hour: int) -> pd.DataFrame:
    """Centre lat/lon, store cell, hourly span and a first ``status``.

    Args:
        windows: :func:`read_windows` output.
        last_hour: Last hour index of the store.

    Returns:
        The table with ``center_lat/lon, ilat, ilon, cell_lat/lon, first_hour,
        end_hour, first_time`` and ``status`` (``ok``, ``south_of_store`` or
        ``outside_store_time``; land is checked by :func:`land_mask`).
    """
    df = windows.copy()
    lat = np.full(len(df), np.nan)
    lon = np.full(len(df), np.nan)
    for crs, idx in df.groupby("crs").indices.items():
        transformer = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
        lon[idx], lat[idx] = transformer.transform(
            df["x"].to_numpy()[idx], df["y"].to_numpy()[idx]
        )
    df["center_lat"], df["center_lon"] = lat, lon
    ilat, ilon = grid_cell(lat, lon)
    df["ilat"], df["ilon"] = ilat, ilon
    df["cell_lat"] = 90.0 - ilat * GRID_DEG
    df["cell_lon"] = (ilon * GRID_DEG + 180.0) % 360.0 - 180.0
    end = pd.to_datetime(df["window_end"], utc=True).dt.floor("h").dt.tz_localize(None)
    end_hour = (
        (end.to_numpy().astype("datetime64[h]") - EPOCH) // np.timedelta64(1, "h")
    ).astype(np.int64)
    df["end_hour"] = end_hour
    df["first_hour"] = end_hour - WINDOW_HOURS + 1
    df["first_time"] = EPOCH + df["first_hour"].to_numpy().astype("timedelta64[h]")
    status = np.full(len(df), "ok", dtype=object)
    status[(df["first_hour"] < 1) | (df["end_hour"] > last_hour)] = "outside_store_time"
    status[ilat >= N_LAT] = "south_of_store"
    df["status"] = status
    return df.drop(columns=["x", "y"])


def land_mask(ilat: np.ndarray, ilon: np.ndarray, workers: int) -> np.ndarray:
    """Whether each cell is land in ERA5-Land (has data), from the daily mirror.

    Args:
        ilat: Store latitude indices (all < ``N_LAT``).
        ilon: Store longitude indices.
        workers: Reader threads.

    Returns:
        Boolean per cell.
    """
    import zarr  # not in the lock; jobs install it

    group = zarr.open_group(str(ERA5_MIRROR), mode="r")
    lat, lon = np.asarray(group["latitude"][:]), np.asarray(group["longitude"][:])
    assert np.allclose(lat, 90.0 - GRID_DEG * np.arange(N_LAT), atol=1e-6), (
        "mirror grid"
    )
    assert np.allclose(lon, GRID_DEG * np.arange(N_LON), atol=1e-6), "mirror grid"
    valid = group["valid_mask"]
    ci, cj = valid.chunks
    keys = sorted(set(zip((ilat // ci).tolist(), (ilon // cj).tolist())))

    def read(key: tuple[int, int]) -> tuple[tuple[int, int], np.ndarray]:
        a, b = key
        return key, np.asarray(
            valid[a * ci : (a + 1) * ci, b * cj : (b + 1) * cj], dtype=bool
        )

    with ThreadPoolExecutor(workers) as ex:
        blocks = dict(ex.map(read, keys))
    return np.array(
        [
            blocks[(i // ci, j // cj)][i % ci, j % cj]
            for i, j in zip(ilat.tolist(), ilon.tolist())
        ],
        dtype=bool,
    )


def time_chunks(first_hour: np.ndarray, end_hour: np.ndarray) -> list[int]:
    """Store time chunks a set of windows reads (one hour before each span too).

    Args:
        first_hour: First row's hour index per window.
        end_hour: Last row's hour index per window.

    Returns:
        Sorted chunk indices.
    """
    chunks: set[int] = set()
    for f, e in zip(first_hour.tolist(), end_hour.tolist()):
        chunks.update(range((f - 1) // TIME_CHUNK, e // TIME_CHUNK + 1))
    return sorted(chunks)


def assign_shards(df: pd.DataFrame) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """Pack the usable windows into shards of whole store tiles.

    Args:
        df: Index with ``status``, ``ilat``, ``ilon``, hours.

    Returns:
        The index with ``shard`` and ``row`` (-1 when not stored), and the
        shard list for ``shards.json``.
    """
    df = df.copy()
    df["tile_lat"] = df["ilat"] // LAT_CHUNK
    df["tile_lon"] = df["ilon"] // LON_CHUNK
    df["shard"], df["row"] = -1, -1
    ok = df[df["status"] == "ok"].sort_values(["tile_lon", "tile_lat", "window"])
    pieces = []
    for (tile_lon, tile_lat), g in ok.groupby(["tile_lon", "tile_lat"], sort=True):
        labels = g.index.to_numpy()
        for s in range(0, len(labels), SHARD_WINDOWS):
            pieces.append((int(tile_lat), int(tile_lon), labels[s : s + SHARD_WINDOWS]))
    groups: list[list[tuple[int, int, np.ndarray]]] = []
    current: list[tuple[int, int, np.ndarray]] = []
    count = 0
    for piece in pieces:
        if count and count + len(piece[2]) > SHARD_WINDOWS:
            groups.append(current)
            current, count = [], 0
        current.append(piece)
        count += len(piece[2])
    if current:
        groups.append(current)

    shards = []
    for shard, group in enumerate(groups):
        row, tiles = 0, []
        for tile_lat, tile_lon, labels in group:
            df.loc[labels, "shard"] = shard
            df.loc[labels, "row"] = np.arange(row, row + len(labels))
            chunks = time_chunks(
                df.loc[labels, "first_hour"].to_numpy(),
                df.loc[labels, "end_hour"].to_numpy(),
            )
            tiles.append(dict(
                tile_lat=tile_lat, tile_lon=tile_lon, row_start=row,
                row_end=row + len(labels), time_chunks=chunks,
            ))  # fmt: skip
            row += len(labels)
        lons = sorted({t["tile_lon"] for t in tiles})
        shards.append(dict(
            shard=shard, n_windows=row, n_tiles=len(tiles),
            lon_range=[lons[0] * LON_CHUNK * GRID_DEG, (lons[-1] + 1) * LON_CHUNK * GRID_DEG],
            tiles=tiles,
        ))  # fmt: skip
    return df, shards


def bands_spec(last_hour: int) -> dict[str, Any]:
    """Contents of ``bands.json``.

    Args:
        last_hour: Last hour index of the store when planned.

    Returns:
        Layout, conventions and the band table.
    """
    return dict(
        layout=f"shards/shard_XXX.npy: (n_windows, {WINDOW_HOURS}, {len(BANDS)}) float32, "
        "row order given by index.parquet (shard, row)",
        nodata=NODATA,
        time=f"row k of a window is store hour first_hour + k (UTC, hours since {EPOCH}); "
        "instantaneous bands are the value at that hour, accumulated bands the amount "
        "over the hour ending there",
        source=HOURLY_URL,
        source_note="values are bit-rounded to 13 mantissa bits by EarthDataHub; ERA5-Land "
        "sign convention (downward fluxes positive; evaporation negative)",
        store_last_time=str(EPOCH + np.timedelta64(last_hour, "h")),
        bands=[dict(index=k, **asdict(b)) for k, b in enumerate(BANDS)],
    )


def write_parquet(df: pd.DataFrame, path: Path) -> None:
    """Write a parquet file atomically.

    Args:
        df: Table.
        path: Destination.
    """
    tmp = path.with_suffix(".tmp.parquet")
    df.to_parquet(tmp, index=False)
    os.replace(tmp, path)


def write_json(obj: Any, path: Path) -> None:
    """Write a JSON file atomically.

    Args:
        obj: Content.
        path: Destination.
    """
    tmp = path.with_suffix(".tmp.json")
    tmp.write_text(json.dumps(obj, indent=2))
    os.replace(tmp, path)


def run(root: Path, windows_dir: Path, workers: int) -> None:
    """Build the index, shard layout and band spec (kept if already there).

    Args:
        root: Output root.
        windows_dir: Corpus window group directory.
        workers: Reader threads.
    """
    root.mkdir(parents=True, exist_ok=True)
    if (root / "index.parquet").exists():
        logger.info(
            "%s exists; keeping it (the shards are built against it)",
            root / "index.parquet",
        )
        return
    last_hour = open_store()["t2m"].shape[0] - 1
    df = add_cells_and_hours(read_windows(windows_dir, workers), last_hour)
    candidates = (df["status"] == "ok").to_numpy()
    land = land_mask(
        df["ilat"].to_numpy()[candidates], df["ilon"].to_numpy()[candidates], workers
    )
    status = df["status"].to_numpy().copy()
    status[np.flatnonzero(candidates)[~land]] = "not_land_in_era5_land"
    df["status"] = status
    df, shards = assign_shards(df)
    for name, n in df["status"].value_counts().items():
        logger.info("status %s: %d windows", name, n)
    logger.info(
        "%d shards, %d to %d windows; %d tile pieces, %d time chunks per band",
        len(shards), min(s["n_windows"] for s in shards), max(s["n_windows"] for s in shards),
        sum(s["n_tiles"] for s in shards),
        sum(len(t["time_chunks"]) for s in shards for t in s["tiles"]),
    )  # fmt: skip
    write_json(bands_spec(last_hour), root / "bands.json")
    write_json(dict(windows_dir=str(windows_dir), shard_windows=SHARD_WINDOWS, shards=shards),
               root / "shards.json")  # fmt: skip
    shutil.copyfile(Path(__file__).with_name("README.md"), root / "README.md")
    write_parquet(df, root / "index.parquet")
    logger.info("wrote %s", root / "index.parquet")


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--root", default=str(DEFAULT_ROOT))
    ap.add_argument("--windows-dir", default=str(WINDOWS_DIR))
    ap.add_argument("--workers", type=int, default=64)
    args = ap.parse_args()
    run(Path(args.root), Path(args.windows_dir), args.workers)


if __name__ == "__main__":
    main()
