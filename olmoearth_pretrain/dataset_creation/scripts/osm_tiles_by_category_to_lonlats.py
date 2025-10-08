"""Go from tiles_by_category.json from osm_sampling.go to longitude/latitude list."""

import argparse
import json
import random

# Size of tiles to split world into in degrees.
TILE_SIZE = 0.001


def sample_tiles(in_fname: str, out_fname: str, tiles_per_category: int) -> None:
    """Sample tiles from tiles_by_category.json to longitude/latitude list.

    Args:
        in_fname: the JSON file containing a map from an OpenStreetMap-based category
            to the corresponding list of tiles containing a feature with that category.
        out_fname: the output to write a JSON list of longitude/latitude positions.
        tiles_per_category: the maximum number of tiles to sample based on each
            category.
    """
    with open(in_fname) as f:
        tiles_by_category = json.load(f)

    # Collect {tiles_per_category} tiles for each category.
    # For now these tiles are (lon, lat) divided by TILE_SIZE.
    selected_tiles: set[tuple[int, int]] = set()
    for category, tile_list in tiles_by_category.items():
        if len(tile_list) > tiles_per_category:
            tile_list = random.sample(tile_list, tiles_per_category)
        print(f"got {len(tile_list)} tiles for category {category}")
        for tile in tile_list:
            selected_tiles.add((tile[0], tile[1]))
        print(f"selected tiles is now {len(selected_tiles)}")

    # Now get the longitudes/latitudes and write them.
    lonlats: list[tuple[float, float]] = []
    for col, row in selected_tiles:
        lonlats.append((col * TILE_SIZE, row * TILE_SIZE))

    with open(out_fname, "w") as f:
        json.dump(lonlats, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert tiles_by_category.json to a longitude/latitude list",
    )
    parser.add_argument(
        "--in_fname",
        type=str,
        help="Input filename (tiles by category JSON)",
        required=True,
    )
    parser.add_argument(
        "--out_fname",
        type=str,
        help="Output filename (longitude/latitude list)",
        required=True,
    )
    parser.add_argument(
        "--tiles_per_category",
        type=int,
        help="Number of tiles to sample for each category",
        required=True,
    )
    args = parser.parse_args()

    sample_tiles(args.in_fname, args.out_fname, args.tiles_per_category)
