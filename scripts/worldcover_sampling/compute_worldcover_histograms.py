"""Generates a csv file with the class pixel amounts for tiles in ESA WorldCover grids.

The WorldCover data is split into GeoTIFFs on a relatively coarse-grained grid. This
script processes a subset of those GeoTIFFs, and is meant to be run in parallel across
many Beaker jobs (see beaker_launcher.py).

It generates the class pixel amount for each smaller tile within those GeoTIFF.
"""

import argparse
import os
from typing import Any

import numpy as np
import pandas as pd
import rioxarray
from tqdm import tqdm

DEFAULT_TILE_SIZE = 100

s3_url_prefix = "https://esa-worldcover.s3.eu-central-1.amazonaws.com"

legend = {
    10: "Trees",
    20: "Shrubland",
    30: "Grassland",
    40: "Cropland",
    50: "Built-up",
    60: "Barren / sparse vegetation",
    70: "Snow and ice",
    80: "Open water",
    90: "Herbaceous wetland",
    95: "Mangroves",
    100: "Moss and lichen",
}


def process_tile(
    tile_name: str, out_dir: str, tile_size: int = DEFAULT_TILE_SIZE
) -> None:
    """Compute histogram for the given WorldCover tile.

    Args:
        tile_name: the name of the WorldCover tile (GeoTIFF).
        out_dir: the directory to write output CSVs.
        tile_size: the size of the sub-tiles for which we want to compute histograms.
    """
    # Skip if the output already exists.
    out_fname = os.path.join(out_dir, f"{tile_name}.csv")
    if os.path.exists(out_fname):
        print("Skipping {tile_name} because {out_fname} already exists")
        return

    output_dict: dict[str, list[Any]] = {"tile_id": [], "lat": [], "lon": []}
    for k in legend.keys():
        output_dict[f"class_{k}"] = []

    url = f"{s3_url_prefix}/v100/2020/map/ESA_WorldCover_10m_2020_v100_{tile_name}_Map.tif"
    tif_file = rioxarray.open_rasterio(url, cache=False)

    for x_i in tqdm(
        range(0, len(tif_file.x), tile_size),  # type: ignore
        leave=False,
        desc=f"Sweeping x for {tile_name}",
    ):  # type: ignore
        for y_i in tqdm(range(0, len(tif_file.y), tile_size), leave=False):  # type: ignore
            sub_tile = tif_file.isel(
                x=slice(x_i, x_i + tile_size), y=slice(y_i, y_i + tile_size)
            )  # type: ignore
            keys, amounts = np.unique(sub_tile, return_counts=True)

            if (len(keys) == 1) and (keys[0] == 0):
                continue

            output_dict["tile_id"].append(tile_name)
            output_dict["lat"].append(sub_tile.y.mean().item())  # type: ignore
            output_dict["lon"].append(sub_tile.x.mean().item())  # type: ignore
            for k in legend.keys():
                if k in keys:
                    output_dict[f"class_{k}"].append(amounts[keys == k][0])
                else:
                    output_dict[f"class_{k}"].append(0)

    # Write to tmp file in case of error during writing.
    # Then we atomically rename it.
    pd.DataFrame(output_dict).to_csv(out_fname + ".tmp", index=False)
    os.rename(out_fname + ".tmp", out_fname)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract class pixel amounts from an ESA WorldCover tile",
    )
    parser.add_argument(
        "--tile_names",
        type=str,
        help="Comma-separated list of tile names to process",
        required=True,
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        help="The output directory, the output CSV will be put here and named based on the tile name",
        required=True,
    )
    parser.add_argument(
        "--tile_size",
        type=int,
        help="The tile size",
        default=DEFAULT_TILE_SIZE,
    )
    args = parser.parse_args()

    for tile_name in args.tile_names.split(","):
        process_tile(tile_name, args.out_dir, tile_size=args.tile_size)
