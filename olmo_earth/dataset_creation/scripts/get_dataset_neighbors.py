"""Create new dataset by adding windows neighboring windows in an existing dataset."""

import argparse
import json
import multiprocessing

import shapely
import tqdm
from rasterio.crs import CRS
from rslearn.const import WGS84_PROJECTION
from rslearn.utils.geometry import Projection, STGeometry
from upath import UPath

WINDOW_SIZE = 256


def tile_to_lon_lat(tile: tuple[Projection, int, int]) -> tuple[float, float]:
    """Convert from tile to longitude latitude.

    Args:
        tile: a (projection, col, row) tile at the WINDOW_SIZE.

    Returns:
        the (lon, lat) tuple.
    """
    projection, col, row = tile
    src_geom = STGeometry(
        Projection(CRS.from_string(projection), 10, -10),
        shapely.Point(
            col * WINDOW_SIZE + WINDOW_SIZE // 2, row * WINDOW_SIZE + WINDOW_SIZE // 2
        ),
        None,
    )
    dst_geom = src_geom.to_projection(WGS84_PROJECTION)
    return (dst_geom.shp.x, dst_geom.shp.y)


def get_neighbor_lon_lat(
    existing_ds_path: UPath,
) -> list[tuple[float, float]]:
    """Get neighboring (lon, lat) list given an existing dataset.

    For each window in the dataset, we include the center longitude/latitude of each
    adjacent tile (unless that tile also appears as another window in the dataset).

    Args:
        existing_ds_path: the path to the existing rslearn dataset generated for
            Helios dataset creation. The windows must be named proj_res_col_row.

    Returns:
        a list of (lon, lat) tuples.
    """
    print("enumerate existing tiles")
    existing_tiles = set()
    for window_dir in (existing_ds_path / "windows" / "res_10").iterdir():
        projection, _, col, row = window_dir.name.split("_")
        existing_tiles.add((projection, int(col), int(row)))

    print("identify new tiles")
    new_tiles = set()
    for projection, col, row in existing_tiles:
        for col_offset in [-1, 0, 1]:
            for row_offset in [-1, 0, 1]:
                new_tile = (projection, col + col_offset, row + row_offset)
                if new_tile in existing_tiles:
                    continue
                new_tiles.add(new_tile)

    # Convert the tiles to lon, lat.
    print("convert new tiles to lon lat")
    p = multiprocessing.Pool(64)
    lon_lats = list(
        tqdm.tqdm(p.imap_unordered(tile_to_lon_lat, new_tiles), total=len(new_tiles))
    )
    p.close()

    return lon_lats


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Get dataset neighbors",
    )
    parser.add_argument(
        "--ds_path",
        type=str,
        help="Existing dataset path",
        required=True,
    )
    parser.add_argument(
        "--fname",
        type=str,
        help="Output JSON filename containing list of [lot, lat]",
        required=True,
    )
    args = parser.parse_args()
    lon_lats = get_neighbor_lon_lat(UPath(args.ds_path))
    with open(args.fname, "w") as f:
        json.dump(lon_lats, f)
