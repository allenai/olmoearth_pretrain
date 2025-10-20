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
from rslearn.data_sources import DataSource, data_source_from_config
from rslearn.dataset import Dataset, Window
from rslearn.utils.geometry import Projection, STGeometry
from rslearn.utils.get_utm_ups_crs import get_utm_ups_projection
from rslearn.utils.mp import StarImapUnorderedWrapper
from upath import UPath

from olmoearth_pretrain.dataset.utils import WindowMetadata

from ..constants import WINDOW_DURATION, WINDOW_RESOLUTIONS, WINDOW_SIZE

# Resolution to use if high-resolution imagery is available.
HIGH_RESOLUTION = 0.625

# Resolution to use otherwise.
FALLBACK_RESOLUTION = 10

# Coarse-grained resolution at which to pick tile timestamps.
COARSE_RESOLUTION = 160

# Time range to use in case no high-resolution imagery is available.
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

    def to_resolution(self, resolution: float) -> "Tile":
        """Get the corresponding tile at a coarser resolution.

        Args:
            resolution: the target resolution.

        Returns:
            a Tile at the target resolution that contains this Tile.
        """
        if resolution < self.resolution:
            raise ValueError(
                f"target resolution {resolution} is not coarser than {resolution}"
            )
        factor = round(resolution / self.resolution)
        return Tile(
            crs=self.crs,
            resolution=resolution,
            col=self.col // factor,
            row=self.row // factor,
        )


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


def create_window(ds_path: UPath, metadata: WindowMetadata) -> list[Window]:
    """Create one or more rslearn windows for ingesting data for OlmoEarth Pretrain.

    A window is created at each predefined resolution that is equal to or coarser than
    the provided resolution. This way, lower resolution data is included at all
    locations where higher resolution data is ingested.

    This function assumes the highest resolution grid cell has been decided, along with
    the time range. Use create_windows_with_highres_time for higher-level API.

    Args:
        ds_path: the rslearn dataset path.
        metadata: the metadata that defines the window.

    Returns:
        the new windows.
    """
    windows = []
    for resolution in WINDOW_RESOLUTIONS:
        # Only create windows at resolutions equal to or coarser than the provided one.
        if resolution < metadata.resolution:
            continue

        # Adjust the metadata for this resolution (i.e., compute the window that is
        # aligned with the grid in case the resolution is coarser).
        factor = round(resolution / metadata.resolution)
        cur_metadata = WindowMetadata(
            metadata.crs,
            resolution,
            metadata.col // factor,
            metadata.row // factor,
            metadata.time,
        )

        # Compute the window attributes based on the WindowMetadata.
        group = f"res_{resolution}"
        window_name = cur_metadata.get_window_name()
        bounds = (
            cur_metadata.col * WINDOW_SIZE,
            cur_metadata.row * WINDOW_SIZE,
            (cur_metadata.col + 1) * WINDOW_SIZE,
            (cur_metadata.row + 1) * WINDOW_SIZE,
        )
        time_range = (
            cur_metadata.time - WINDOW_DURATION // 2,
            cur_metadata.time + WINDOW_DURATION // 2,
        )
        projection = Projection(
            CRS.from_string(cur_metadata.crs), resolution, -resolution
        )

        # Create the window.
        window = Window(
            path=Window.get_window_root(ds_path, group, window_name),
            group=group,
            name=window_name,
            projection=projection,
            bounds=bounds,
            time_range=time_range,
        )
        window.save()
        windows.append(window)

    return windows


@functools.cache
def get_naip_source(ds_path: UPath) -> DataSource:
    """Get a NAIP data source for looking up available images.

    Args:
        ds_path: the dataset path, with config_init.json.

    Returns:
        the data source.
    """
    dataset = Dataset(ds_path)
    return data_source_from_config(dataset.layers["naip"], dataset.path)


@functools.cache
def get_sentinel2_source(ds_path: UPath) -> DataSource:
    """Get a Sentinel-2 data source for looking up available images.

    Args:
        ds_path: the dataset path, with config_init.json.

    Returns:
        the data source.
    """
    dataset = Dataset(ds_path)
    return data_source_from_config(dataset.layers["sentinel2_freq"], dataset.path)


def get_highres_times(ds_path: UPath, tile: Tile) -> list[datetime]:
    """Get the timestamps when high-resolution imagery is available of a tile.

    Args:
        ds_path: path to the rslearn dataset to add the window to.
        tile: the Tile (at HIGH_RESOLUTION) to check.

    Returns:
        a list of timestamps when high-resolution imagery is available.
    """
    # Determine what timestamp to use based on NAIP data source.
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
        assert len(group) == 1
        timestamps.append(group[0].geometry.time_range[0])
    return timestamps


def get_sentinel2_times(
    ds_path: UPath, tile: Tile, time_range: tuple[datetime, datetime]
) -> list[datetime]:
    """Get the timestamps when Sentinel-2 is available of a tile.

    Args:
        ds_path: path to the rslearn dataset to add the window to.
        tile: the Tile (at FALLBACK_RESOLUTION) to check.
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
        assert len(group) == 1
        timestamps.append(group[0].geometry.time_range[0])
    return timestamps


def get_highres_tile(lonlat: tuple[float, float]) -> Tile:
    """Get the high-resolution tile containing the specified longitude/latitude.

    Args:
        lonlat: the (longitude, latitude) tuple.

    Returns:
        the Tile (CRS, column, and row) at HIGH_RESOLUTION.
    """
    # Find the 0.625 m/pixel grid cell that contains the specified longitude/latitude.
    lon, lat = lonlat
    projection = get_utm_ups_projection(lon, lat, HIGH_RESOLUTION, -HIGH_RESOLUTION)
    src_geom = STGeometry(WGS84_PROJECTION, shapely.Point(lon, lat), None)
    dst_geom = src_geom.to_projection(projection)
    col = int(dst_geom.shp.x) // WINDOW_SIZE
    row = int(dst_geom.shp.y) // WINDOW_SIZE
    return Tile(projection.crs, HIGH_RESOLUTION, col, row)


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


def create_windows_with_highres_time(
    ds_path: UPath,
    lonlats: list[tuple[float, float]],
    force_lowres_prob: float = 0.0,
    workers: int = 32,
) -> None:
    """Create windows using the timestamp of high-resolution (0.625 m/pixel) imagery.

    If high-resolution imagery covers a location, then the timestamp of the
    high-resolution image is used for the window's center time. If there are multiple
    high-resolution images, we uniformly sample one to get the timestamp from.

    Otherwise, we create a 10 m/pixel window at the location with a random timestamp
    between START_TIME and END_TIME. We also do this with force_lowres_prob probability
    even if high-resolution image covers the location.

    Args:
        ds_path: path to the rslearn dataset to add the window to.
        lonlats: list of points at which to create windows. We create one set of
            windows for each point (starting from either 0.625 m/pixel or 10 m/pixel
            and going down to the coarsest resolution). We ensure that, across windows,
            each grid cell at the coarsest resolution uses the same timestamp.
        force_lowres_prob: probability to use random timestamp and 10 m/pixel
            resolution even if high-resolution imagery is available.
        workers: number of worker processes for looking up high-resolution image
            availability and for creating windows.
    """
    # A key constraint is that we want every coarse-grained tile to have one timestamp,
    # which means all the finer-grained tiles contained within that big tile need to
    # share the same timestamp.
    # So we will:
    # (1) In parallel, convert the lonlats to tiles.
    # (2) In parallel, list the timestamps when high-res imagery is available.
    # (3) Sequentially, decide which timestamps to use for the coarse grained tiles.
    # (4) In parallel, create the resulting windows.
    p = multiprocessing.Pool(workers)
    highres_tiles: list[Tile] = list(
        tqdm.tqdm(
            p.imap(get_highres_tile, lonlats),
            desc="Getting high-res tiles",
            total=len(lonlats),
        )
    )
    print(f"got {len(highres_tiles)} initial high-res tiles")
    # De-duplicate in case user gave some lonlats that fall in the same tile.
    highres_tiles = list(set(highres_tiles))
    print(f"have {len(highres_tiles)} after de-duplication")

    # List timestamps.
    get_highres_times_jobs = []
    for tile in highres_tiles:
        get_highres_times_jobs.append(
            dict(
                ds_path=ds_path,
                tile=tile,
            )
        )
    highres_timestamps: list[list[datetime]] = list(
        tqdm.tqdm(
            star_imap(p, get_highres_times, get_highres_times_jobs),
            desc="Get high-res timestamps",
            total=len(get_highres_times_jobs),
        )
    )

    # Decide which timestamps to use for coarse-grained tiles.
    # We shuffle the high-res tiles/timestamps since we will be using the first
    # high-res tile for a given coarse-grained tile to decide the timestamp to use.
    highres_tiles_and_timestamps = list(zip(highres_tiles, highres_timestamps))
    random.shuffle(highres_tiles_and_timestamps)
    coarse_times: dict[Tile, datetime] = {}
    for highres_tile, timestamps in highres_tiles_and_timestamps:
        coarse_tile = highres_tile.to_resolution(COARSE_RESOLUTION)
        if coarse_tile in coarse_times:
            continue

        # Only attempt to use high-resolution imagery if we roll high enough number.
        # So for some coarse-grained tiles, even if they are spatially covered by
        # high-res imagery, we still want to uniformly sample a timestamp.
        if len(timestamps) == 0 or random.random() < force_lowres_prob:
            selected_time = sample_timestamp(START_TIME, END_TIME)
        else:
            selected_time = random.choice(timestamps)

        coarse_times[coarse_tile] = selected_time

    print(
        f"got {len(highres_tiles)} high-resolution tiles and {len(coarse_times)} coarse-grained tiles"
    )

    # For each high-res tile:
    # - If it has high-resolution imagery available matching the coarse-grained
    #   timestamp, then we can add it as a high-res tile.
    # - Otherwise, we try to add it at the fallback resolution (in case it has 10
    #   m/pixel data but no 0.625 m/pixel data).
    good_highres_tiles: set[Tile] = set()
    fallback_tiles: set[Tile] = set()
    for highres_tile, timestamps in highres_tiles_and_timestamps:
        coarse_tile = highres_tile.to_resolution(COARSE_RESOLUTION)
        fallback_tile = highres_tile.to_resolution(FALLBACK_RESOLUTION)

        # See if there is any high-res timestamp within WINDOW_DURATION//2 of the
        # coarse time (which will be the center time of the window).
        coarse_time = coarse_times[coarse_tile]
        chosen_timestamp = None
        for timestamp in timestamps:
            if timestamp < coarse_time - WINDOW_DURATION // 2:
                continue
            if timestamp > coarse_time + WINDOW_DURATION // 2:
                continue
            chosen_timestamp = timestamp
            break

        if chosen_timestamp is None:
            fallback_tiles.add(fallback_tile)
        else:
            good_highres_tiles.add(highres_tile)
    print(
        f"found {len(good_highres_tiles)} good high-res tiles and {len(fallback_tiles)} initial fallback tiles"
    )

    # For now, filter the fallback tiles for ones where Sentinel-2 imagery is
    # available.
    get_sentinel2_times_jobs = []
    for fallback_tile in fallback_tiles:
        coarse_tile = fallback_tile.to_resolution(COARSE_RESOLUTION)
        coarse_time = coarse_times[coarse_tile]
        time_range = (
            coarse_time - WINDOW_DURATION // 2,
            coarse_time + WINDOW_DURATION // 2,
        )
        get_sentinel2_times_jobs.append(
            dict(
                ds_path=ds_path,
                tile=fallback_tile,
                time_range=time_range,
            )
        )
    sentinel2_times = list(
        tqdm.tqdm(
            star_imap(p, get_sentinel2_times, get_sentinel2_times_jobs),
            desc="Get Sentinel-2 times",
            total=len(get_sentinel2_times_jobs),
        )
    )
    good_fallback_tiles: set[Tile] = set()
    for fallback_tile, timestamps in zip(fallback_tiles, sentinel2_times):
        if len(timestamps) == 0:
            continue
        good_fallback_tiles.add(fallback_tile)
    print(f"filtered down to {len(good_fallback_tiles)} good fallback tiles")

    # Finally now we can create the windows.
    create_window_jobs = []
    good_tiles = good_highres_tiles.union(good_fallback_tiles)
    for tile in good_tiles:
        coarse_tile = tile.to_resolution(COARSE_RESOLUTION)
        coarse_time = coarse_times[coarse_tile]
        window_metadata = WindowMetadata(
            str(tile.crs), tile.resolution, tile.col, tile.row, coarse_time
        )
        create_window_jobs.append(
            dict(
                ds_path=ds_path,
                metadata=window_metadata,
            )
        )

    outputs = star_imap(p, create_window, create_window_jobs)
    for _ in tqdm.tqdm(outputs, desc="Create windows", total=len(create_window_jobs)):
        pass

    p.close()
