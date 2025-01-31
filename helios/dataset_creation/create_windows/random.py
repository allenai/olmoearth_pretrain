"""Create windows with random timestamps."""

import argparse
import random
from datetime import datetime, timedelta, timezone

import shapely
from rslearn.const import WGS84_PROJECTION
from rslearn.dataset.window import Window
from rslearn.utils.geometry import STGeometry
from rslearn.utils.get_utm_ups_crs import get_utm_ups_projection
from upath import UPath

from ..util import WindowMetadata
from .util import WINDOW_SIZE, create_window

# Some arbitrarily chosen locations for now.
LOCATIONS = [
    (-122.32, 47.62),
    (-122.68, 45.52),
    (-121.49, 38.58),
    (-122.42, 37.78),
    (-96.80, 32.78),
    (-84.39, 33.76),
    (-80.84, 35.23),
    (-71.06, 42.36),
    (2.35, 48.85),
    (51.53, 25.29),
    (103.99, 1.36),
    (104.92, 11.57),
    (135.49, 34.71),
    (111.90, 43.72),
    (106.55, 29.56),
    (111.74, 27.26),
    (78.88, 17.51),
]

RESOLUTION = 10

START_TIME = datetime(2016, 6, 1, tzinfo=timezone.utc)
END_TIME = datetime(2024, 6, 1, tzinfo=timezone.utc)


def create_window_random_time(ds_path: UPath, lon: float, lat: float) -> list[Window]:
    """Create windows corresponding to the specified longitude and latitude.

    It will have a random timestamp between START_TIME and END_TIME.

    Args:
        ds_path: path to the rslearn dataset to add the window to.
        lon: the longitude that the window should contain.
        lat: the latitude that the window should contain.

    Returns:
        the new windows.
    """
    # Find the 10 m/pixel grid cell that contains the specified longitude/latitude.
    projection = get_utm_ups_projection(lon, lat, RESOLUTION, -RESOLUTION)
    src_geom = STGeometry(WGS84_PROJECTION, shapely.Point(lon, lat), None)
    dst_geom = src_geom.to_projection(projection)
    col = int(dst_geom.shp.x) // WINDOW_SIZE
    row = int(dst_geom.shp.y) // WINDOW_SIZE

    # Uniformly sample a timestamp.
    total_seconds = (END_TIME - START_TIME).total_seconds()
    selected_seconds = random.randint(0, int(total_seconds))
    selected_ts = START_TIME + timedelta(seconds=selected_seconds)
    selected_date = datetime(
        selected_ts.year, selected_ts.month, selected_ts.day, tzinfo=timezone.utc
    )

    return create_window(
        ds_path,
        WindowMetadata(str(projection.crs), RESOLUTION, col, row, selected_date),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create windows with random timestamp for data ingestion",
    )
    parser.add_argument(
        "--ds_path",
        type=str,
        help="Dataset path",
        required=True,
    )
    args = parser.parse_args()

    ds_path = UPath(args.ds_path)

    for lon_base, lat_base in LOCATIONS:
        for lon_offset in [-0.03, 0, 0.03]:
            for lat_offset in [-0.03, 0, 0.03]:
                lon = lon_base + lon_offset
                lat = lat_base + lat_offset
                create_window_random_time(ds_path, lon, lat)
