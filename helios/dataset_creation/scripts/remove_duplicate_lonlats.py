"""Create new dataset by adding windows neighboring windows in an existing dataset."""

import argparse
import json
import multiprocessing

import shapely
import tqdm
from rslearn.const import WGS84_PROJECTION
from rslearn.utils.geometry import STGeometry
from rslearn.utils.get_utm_ups_crs import get_utm_ups_projection
from upath import UPath

WINDOW_SIZE = 256


def lon_lat_to_tile(lonlat: tuple[float, float]) -> tuple[str, int, int]:
    """Convert longitude, latitude to tile.

    Args:
        lonlat: the (longitude, latitude) tuple.

    Returns:
        the corresponding 10 m/pixel tile (projection_str, col, row).
    """
    lon, lat = lonlat
    src_geom = STGeometry(WGS84_PROJECTION, shapely.Point(lon, lat), None)
    dst_projection = get_utm_ups_projection(lon, lat, 10, -10)
    dst_geom = src_geom.to_projection(dst_projection)
    col = int(dst_geom.shp.x) // 256
    row = int(dst_geom.shp.y) // 256
    return (str(dst_projection.crs), col, row)


def get_existing_tiles(ds_path: UPath) -> list[tuple[str, int, int]]:
    """Get the existing (projection, col, row) tiles in the specified dataset.

    Args:
        ds_path: the path to the existing dataset.

    Returns:
        a list of (projection_str, col, row) tuples.
    """
    existing_tiles = []
    for window_dir in (ds_path / "windows" / "res_10").iterdir():
        projection, _, col, row = window_dir.name.split("_")
        existing_tiles.append((projection, int(col), int(row)))
    return existing_tiles


def remove_duplicate_lonlats(
    lonlats: list[tuple[float, float]], ds_paths: list[UPath]
) -> list[tuple[float, float]]:
    """Prune (lon, lat) pairs that appear in existing datasets.

    Args:
        lonlats: the locations to prune.
        ds_paths: the existing rslearn dataset paths for Helios dataset creation.

    Returns:
        a list of (lon, lat) tuples.
    """
    print(f"convert {len(lonlats)} lon lats to tiles")
    p = multiprocessing.Pool(64)
    lonlat_tiles = list(tqdm.tqdm(p.imap(lon_lat_to_tile, lonlats), total=len(lonlats)))
    p.close()

    print("get existing tiles")
    existing_tiles = set()
    for ds_path in ds_paths:
        print("...", ds_path)
        for tile in get_existing_tiles(ds_path):
            existing_tiles.add(tile)
    print(f"got {len(existing_tiles)} existing tiles")

    print("pruning")
    pruned_lonlats = []
    for lonlat, tile in zip(lonlats, lonlat_tiles):
        if tile in existing_tiles:
            continue
        existing_tiles.add(tile)
        pruned_lonlats.append(lonlat)

    print(f"got {len(pruned_lonlats)} after pruning")
    return pruned_lonlats


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Get dataset neighbors",
    )
    parser.add_argument(
        "--ds_path",
        type=str,
        nargs="+",
        help="Existing dataset paths",
        required=True,
    )
    parser.add_argument(
        "--in_fname",
        type=str,
        help="Input JSON filename containing list of [lot, lat]",
        required=True,
    )
    parser.add_argument(
        "--out_fname",
        type=str,
        help="Output JSON filename containing pruned list",
        required=True,
    )
    args = parser.parse_args()
    with open(args.in_fname) as f:
        lonlats = json.load(f)
    pruned_lonlats = remove_duplicate_lonlats(
        lonlats, [UPath(ds_path) for ds_path in args.ds_path]
    )
    with open(args.out_fname, "w") as f:
        json.dump(pruned_lonlats, f)
