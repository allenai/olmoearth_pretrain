"""Create windows based on the timestamp of NAIP images.

All of the resulting windows are 1 m/pixel windows.
"""

import argparse
import logging
import random
from datetime import datetime, timezone

import rslearn.data_sources
import shapely
from rslearn.config import QueryConfig
from rslearn.const import WGS84_PROJECTION
from rslearn.dataset import Dataset, Window
from rslearn.utils.geometry import STGeometry
from rslearn.utils.get_utm_ups_crs import get_utm_ups_projection
from upath import UPath

from ..util import WindowMetadata
from .util import WINDOW_SIZE, create_window

logger = logging.getLogger(__name__)


# Some arbitrarily chosen locations for now.
# These should all be in the continental US, otherwise no NAIP data will be available.
LOCATIONS = [
    (-117.42, 47.66),
    (-121.75, 46.86),
    (-123.95, 45.52),
    (-97.53, 35.48),
    (-97.30, 29.93),
    (-90.69, 35.84),
    (-115.16, 36.09),
    (-89.59, 40.70),
    (-77.04, 38.90),
    (-81.74, 28.03),
    (-112.92, 33.34),
]

RESOLUTION = 0.625

START_TIME = datetime(2016, 6, 1, tzinfo=timezone.utc)
END_TIME = datetime(2024, 6, 1, tzinfo=timezone.utc)


def create_window_naip_time(
    ds_path: UPath, lon: float, lat: float
) -> list[Window] | None:
    """Create windows corresponding to the lon/lat using the timestamp of a NAIP image.

    The timestamp of the window will be chosen based on the timestamp of a NAIP image
    that covers that location. If there are multiple NAIP images, we uniformly sample
    one to get the timestamp from.

    Args:
        ds_path: path to the rslearn dataset to add the window to.
        lon: the longitude that the window should contain.
        lat: the latitude that the window should contain.

    Returns:
        the new windows, or None if no matching NAIP image was found.
    """
    # Find the 0.625 m/pixel grid cell that contains the specified longitude/latitude.
    projection = get_utm_ups_projection(lon, lat, RESOLUTION, -RESOLUTION)
    src_geom = STGeometry(WGS84_PROJECTION, shapely.Point(lon, lat), None)
    dst_geom = src_geom.to_projection(projection)
    col = int(dst_geom.shp.x) // WINDOW_SIZE
    row = int(dst_geom.shp.y) // WINDOW_SIZE

    # Determine what timestamp to use based on NAIP data source.
    bounds = (
        col * WINDOW_SIZE,
        row * WINDOW_SIZE,
        (col + 1) * WINDOW_SIZE,
        (row + 1) * WINDOW_SIZE,
    )
    window_geom = STGeometry(projection, shapely.box(*bounds), (START_TIME, END_TIME))
    groups = naip_source.get_items([window_geom], query_config)[0]
    if len(groups) == 0:
        return None
    item = random.choice(groups)[0]
    item_time = item.geometry.time_range[0]

    return create_window(
        ds_path, WindowMetadata(str(projection.crs), RESOLUTION, col, row, item_time)
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create 1 m/pixel windows based on NAIP timestamp",
    )
    parser.add_argument(
        "--ds_path",
        type=str,
        help="Dataset path",
        required=True,
    )
    args = parser.parse_args()

    ds_path = UPath(args.ds_path)
    dataset = Dataset(ds_path)
    naip_source = rslearn.data_sources.data_source_from_config(
        dataset.layers["naip"], dataset.path
    )
    query_config = QueryConfig()

    for lon_base, lat_base in LOCATIONS:
        for lon_offset in [-0.03, 0, 0.03]:
            for lat_offset in [-0.03, 0, 0.03]:
                lon = lon_base + lon_offset
                lat = lat_base + lat_offset
                window = create_window_naip_time(ds_path, lon, lat)
                if window is None:
                    logger.warning(
                        "error creating window centered at %f, %f: no matching NAIP image found",
                        lon,
                        lat,
                    )
