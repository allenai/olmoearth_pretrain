"""Utilities for creating windows."""

import functools
import multiprocessing
import random
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from multiprocessing.pool import IMapIterator
from typing import Any

import shapely
import tqdm
from rasterio.crs import CRS
from rslearn.config import QueryConfig, SpaceMode
from rslearn.const import WGS84_PROJECTION
from rslearn.data_sources import DataSource
from rslearn.dataset import Dataset, Window
from rslearn.utils.geometry import Projection, STGeometry
from rslearn.utils.get_utm_ups_crs import get_utm_ups_projection
from rslearn.utils.mp import StarImapUnorderedWrapper
from upath import UPath

from olmoearth_pretrain.dataset.utils import WindowMetadata

from ..constants import WINDOW_DURATION, WINDOW_SIZE

# All windows are created at 10 m/pixel (the "res_10" group).
RESOLUTION = 10

# Probability of using a random timestamp even when NAIP imagery is available. NAIP
# acquisitions are summer-biased, so this preserves seasonal diversity at CONUS tiles.
DEFAULT_RANDOM_TIME_PROB = 0.25

# Time range from which window timestamps are drawn.
START_TIME = datetime(2016, 6, 1, tzinfo=UTC)
END_TIME = datetime(2024, 6, 1, tzinfo=UTC)


@dataclass(frozen=True)
class Tile:
    """A tile (grid cell) with a specific CRS, resolution, column, and row.

    The CRS and resolution specify the grid, while the column and row specify the tile
    within that grid.
    """

    crs: CRS
    resolution: float
    col: int
    row: int


def star_imap(
    p: multiprocessing.pool.Pool,
    fn: Callable[..., Any],
    kwargs_list: list[dict[str, Any]],
) -> IMapIterator:
    """Wrapper for Pool.imap that exposes kwargs to the function.

    Args:
        p: the multiprocessing.pool.Pool to use.
        fn: the function to call, which accepts keyword arguments.
        kwargs_list: list of kwargs dicts to pass to the function.

    Returns:
        generator for outputs from the function in arbitrary order.
    """
    return p.imap(StarImapUnorderedWrapper(fn), kwargs_list)


@functools.cache
def get_dataset(ds_path: UPath) -> Dataset:
    """Get a (cached) rslearn Dataset for the given path."""
    return Dataset(ds_path)


def create_window(ds_path: UPath, tile: Tile, center_time: datetime) -> None:
    """Create one res_10 rslearn window for ingesting data for OlmoEarth Pretrain.

    Args:
        ds_path: the rslearn dataset path.
        tile: the res_10 grid tile to create the window at.
        center_time: the center time of the window.
    """
    dataset = get_dataset(ds_path)
    metadata = WindowMetadata(
        str(tile.crs), tile.resolution, tile.col, tile.row, center_time
    )
    group = f"res_{RESOLUTION}"
    window_name = metadata.get_window_name()
    bounds = (
        tile.col * WINDOW_SIZE,
        tile.row * WINDOW_SIZE,
        (tile.col + 1) * WINDOW_SIZE,
        (tile.row + 1) * WINDOW_SIZE,
    )
    time_range = (
        center_time - WINDOW_DURATION // 2,
        center_time + WINDOW_DURATION // 2,
    )
    projection = Projection(
        CRS.from_string(metadata.crs), tile.resolution, -tile.resolution
    )
    window = Window(
        storage=dataset.storage,
        group=group,
        name=window_name,
        projection=projection,
        bounds=bounds,
        time_range=time_range,
        data_factory=dataset.window_data_storage_factory,
    )
    window.save()


@functools.cache
def get_naip_source(ds_path: UPath) -> DataSource:
    """Get a NAIP data source for looking up available images.

    Args:
        ds_path: the dataset path, with config_init.json.

    Returns:
        the data source.
    """
    dataset = Dataset(ds_path)
    return dataset.layers["naip"].instantiate_data_source(dataset.path)


@functools.cache
def get_sentinel2_source(ds_path: UPath) -> DataSource:
    """Get a Sentinel-2 data source for looking up available images.

    Args:
        ds_path: the dataset path, with config_init.json.

    Returns:
        the data source.
    """
    dataset = Dataset(ds_path)
    return dataset.layers["sentinel2_freq"].instantiate_data_source(dataset.path)


def get_naip_times(ds_path: UPath, tile: Tile) -> list[datetime]:
    """Get the timestamps when NAIP imagery intersects a tile.

    Args:
        ds_path: path to the rslearn dataset.
        tile: the res_10 Tile to check.

    Returns:
        a list of timestamps when NAIP imagery is available.
    """
    naip_source = get_naip_source(ds_path)
    projection = Projection(tile.crs, tile.resolution, -tile.resolution)
    bounds = (
        tile.col * WINDOW_SIZE,
        tile.row * WINDOW_SIZE,
        (tile.col + 1) * WINDOW_SIZE,
        (tile.row + 1) * WINDOW_SIZE,
    )
    window_geom = STGeometry(projection, shapely.box(*bounds), (START_TIME, END_TIME))
    query_config = QueryConfig(max_matches=9999, space_mode=SpaceMode.CONTAINS)
    groups = naip_source.get_items([window_geom], query_config)[0]

    timestamps = []
    for group in groups:
        for item in group.items:
            timestamps.append(item.geometry.time_range[0])
    return timestamps


def get_sentinel2_times(
    ds_path: UPath, tile: Tile, time_range: tuple[datetime, datetime]
) -> list[datetime]:
    """Get the timestamps when Sentinel-2 is available of a tile.

    Args:
        ds_path: path to the rslearn dataset to add the window to.
        tile: the res_10 Tile to check.
        time_range: the time range to search for Sentinel-2 images.

    Returns:
        a list of timestamps when Sentinel-2 imagery is available.
    """
    sentinel2_source = get_sentinel2_source(ds_path)
    projection = Projection(tile.crs, tile.resolution, -tile.resolution)
    bounds = (
        tile.col * WINDOW_SIZE,
        tile.row * WINDOW_SIZE,
        (tile.col + 1) * WINDOW_SIZE,
        (tile.row + 1) * WINDOW_SIZE,
    )
    window_geom = STGeometry(projection, shapely.box(*bounds), time_range)
    query_config = QueryConfig(max_matches=9999, space_mode=SpaceMode.CONTAINS)
    groups = sentinel2_source.get_items([window_geom], query_config)[0]

    timestamps = []
    for group in groups:
        for item in group.items:
            timestamps.append(item.geometry.time_range[0])
    return timestamps


def get_res10_tile(lonlat: tuple[float, float]) -> Tile:
    """Get the res_10 (10 m/pixel) tile containing the specified longitude/latitude.

    Args:
        lonlat: the (longitude, latitude) tuple.

    Returns:
        the Tile (CRS, column, and row) at RESOLUTION.
    """
    lon, lat = lonlat
    projection = get_utm_ups_projection(lon, lat, RESOLUTION, -RESOLUTION)
    src_geom = STGeometry(WGS84_PROJECTION, shapely.Point(lon, lat), None)
    dst_geom = src_geom.to_projection(projection)
    col = int(dst_geom.shp.x) // WINDOW_SIZE
    row = int(dst_geom.shp.y) // WINDOW_SIZE
    return Tile(projection.crs, RESOLUTION, col, row)


def sample_timestamp(start_time: datetime, end_time: datetime) -> datetime:
    """Sample a date in the given time range.

    Args:
        start_time: start of the time range.
        end_time: end of the time range.

    Returns:
        a date (with hour/minute/second=0) between start_time and end_time.
    """
    total_seconds = (END_TIME - START_TIME).total_seconds()
    selected_seconds = random.randint(0, int(total_seconds))
    selected_ts = START_TIME + timedelta(seconds=selected_seconds)
    selected_date = datetime(
        selected_ts.year, selected_ts.month, selected_ts.day, tzinfo=UTC
    )
    return selected_date


def create_windows(
    ds_path: UPath,
    lonlats: list[tuple[float, float]],
    random_time_prob: float = DEFAULT_RANDOM_TIME_PROB,
    workers: int = 32,
) -> None:
    """Create res_10 windows at the given locations.

    For each location we find the containing 10 m/pixel tile and pick a window
    timestamp: if NAIP imagery intersects the tile we (usually) use the timestamp of a
    random NAIP acquisition; otherwise -- or with ``random_time_prob`` probability even
    when NAIP is available -- we sample a random timestamp. Tiles without any Sentinel-2
    coverage in the window's time range are dropped.

    Args:
        ds_path: path to the rslearn dataset to add windows to.
        lonlats: list of (longitude, latitude) points to create windows at.
        random_time_prob: probability of using a random timestamp even when NAIP
            imagery is available (NAIP is summer-biased, so this adds seasonal
            diversity).
        workers: number of worker processes.
    """
    p = multiprocessing.Pool(workers)

    # (1) Convert lonlats to res_10 tiles and de-duplicate.
    tiles: list[Tile] = list(
        tqdm.tqdm(
            p.imap(get_res10_tile, lonlats),
            desc="Getting tiles",
            total=len(lonlats),
        )
    )
    tiles = list(set(tiles))
    print(f"have {len(tiles)} tiles after de-duplication")

    # (2) Look up NAIP acquisition timestamps intersecting each tile.
    naip_times: list[list[datetime]] = list(
        tqdm.tqdm(
            star_imap(
                p,
                get_naip_times,
                [dict(ds_path=ds_path, tile=tile) for tile in tiles],
            ),
            desc="Get NAIP timestamps",
            total=len(tiles),
        )
    )

    # (3) Choose a center time for each tile: usually a NAIP timestamp if available,
    # otherwise (or with random_time_prob) a random timestamp.
    tiles_and_times: list[tuple[Tile, datetime]] = []
    for tile, timestamps in zip(tiles, naip_times):
        if timestamps and random.random() >= random_time_prob:
            center_time = random.choice(timestamps)
        else:
            center_time = sample_timestamp(START_TIME, END_TIME)
        tiles_and_times.append((tile, center_time))

    # (4) Drop tiles without Sentinel-2 coverage in the window's time range.
    get_sentinel2_times_jobs = [
        dict(
            ds_path=ds_path,
            tile=tile,
            time_range=(
                center_time - WINDOW_DURATION // 2,
                center_time + WINDOW_DURATION // 2,
            ),
        )
        for tile, center_time in tiles_and_times
    ]
    sentinel2_times: list[list[datetime]] = list(
        tqdm.tqdm(
            star_imap(p, get_sentinel2_times, get_sentinel2_times_jobs),
            desc="Get Sentinel-2 times",
            total=len(get_sentinel2_times_jobs),
        )
    )
    good_tiles_and_times = [
        (tile, center_time)
        for (tile, center_time), s2_times in zip(tiles_and_times, sentinel2_times)
        if len(s2_times) > 0
    ]
    print(
        f"kept {len(good_tiles_and_times)} of {len(tiles_and_times)} tiles with "
        "Sentinel-2 coverage"
    )

    # (5) Create the windows.
    create_window_jobs = [
        dict(ds_path=ds_path, tile=tile, center_time=center_time)
        for tile, center_time in good_tiles_and_times
    ]
    outputs = star_imap(p, create_window, create_window_jobs)
    for _ in tqdm.tqdm(outputs, desc="Create windows", total=len(create_window_jobs)):
        pass

    p.close()
