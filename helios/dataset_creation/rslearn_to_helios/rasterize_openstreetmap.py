"""Create openstreetmap_raster from openstreetmap in the Helios dataset."""

import argparse
import json
import multiprocessing
from collections.abc import Callable

import numpy as np
import numpy.typing as npt
import skimage.draw
import tqdm
from rasterio.crs import CRS
from rslearn.utils.geometry import Projection
from upath import UPath

from ..constants import GEOTIFF_RASTER_FORMAT

WINDOW_SIZE = 256
# Factor by which coordinates in openstreetmap are zoomed in.
IN_FACTOR = 16
# Factor to zoom in for output. So output will be 1024x1024.
FACTOR = 4
OUTPUT_SIZE = WINDOW_SIZE * FACTOR
OUTPUT_RESOLUTION = 10 / FACTOR
OUTPUT_MODALITY = "10_openstreetmap_raster"

CATEGORIES = [
    "aerialway_pylon",
    "aerodrome",
    "airstrip",
    "amenity_fuel",
    "building",
    "chimney",
    "communications_tower",
    "crane",
    "flagpole",
    "fountain",
    "generator_wind",
    "helipad",
    "highway",
    "leisure",
    "lighthouse",
    "obelisk",
    "observatory",
    "parking",
    "petroleum_well",
    "power_plant",
    "power_substation",
    "power_tower",
    "river",
    "runway",
    "satellite_dish",
    "silo",
    "storage_tank",
    "taxiway",
    "water_tower",
    "works",
]


def draw_polygon(
    array: npt.NDArray,
    coords: list[list[list[float]]],
    category_id: int,
    transform: Callable[[npt.NDArray], npt.NDArray],
) -> None:
    """Draw a polygon on the array.

    Args:
        array: the array to write to.
        coords: the pixel coordinates of the polygon. coords[0] should correspond to
            the interior, while the remaining perimeters (if any) should be interior
            holes.
        category_id: the category of this polygon.
        transform: transform to apply on the coordinates.
    """
    exterior = transform(np.array(coords[0]))
    rows, cols = skimage.draw.polygon(
        exterior[:, 1], exterior[:, 0], shape=(OUTPUT_SIZE, OUTPUT_SIZE)
    )

    # If this polygon has no holes, we can draw it directly.
    # Otherwise, we create a mask from the exterior, but then negate the holes.
    if len(coords) == 1:
        array[category_id, rows, cols] = 1
        return

    mask = np.zeros((OUTPUT_SIZE, OUTPUT_SIZE), dtype=bool)
    mask[rows, cols] = True

    for ring in coords[1:]:
        interior = transform(np.array(ring))
        rows, cols = skimage.draw.polygon(
            interior[:, 1], interior[:, 0], shape=(OUTPUT_SIZE, OUTPUT_SIZE)
        )
        mask[rows, cols] = False

    array[category_id, mask] = 1


def draw_line_string(
    array: npt.NDArray,
    coords: list[list[float]],
    category_id: int,
    transform: Callable[[npt.NDArray], npt.NDArray],
) -> None:
    """Draw a line string on the array.

    Args:
        array: the array to write to.
        coords: the pixel coordinates of the line string.
        category_id: the category of this line string.
        transform: transform to apply on the coordinates.
    """
    coords = transform(np.array(coords))

    for i in range(len(coords) - 1):
        rows, cols = skimage.draw.line(
            coords[i][1], coords[i][0], coords[i + 1][1], coords[i + 1][0]
        )
        valid = (rows >= 0) & (rows < OUTPUT_SIZE) & (cols >= 0) & (cols < OUTPUT_SIZE)
        array[category_id, rows[valid], cols[valid]] = 1


def rasterize_openstreetmap(in_fname: UPath) -> None:
    """Rasterize OpenStreetMap data.

    Args:
        in_fname: the input filename containing the GeoJSON data. Outputs will be
            written to a corresponding name in the openstreetmap_raster folder.
    """
    # Parse the column and row from the filename.
    fname_parts = in_fname.name.split(".")[0].split("_")
    crs = CRS.from_string(fname_parts[0])
    col = int(fname_parts[1])
    row = int(fname_parts[2])

    # Construct the transform from the input coordinates to coordinates within the
    # image. The input coordinates are in CRS units while we want the output to be in
    # pixel coordinates within the output 1024x1024 image.
    def transform(coords: npt.NDArray) -> npt.NDArray:
        """Transform the GeoJSON coordinates to pixel coordinates within the image.

        Args:
            coords: the GeoJSON coordinates.

        Returns:
            the pixel coordinates within the image.
        """
        flat_coords = coords.reshape(-1, 2)
        # Convert to global pixel coordinates at OUTPUT_RESOLUTION.
        flat_coords[:, 0] /= OUTPUT_RESOLUTION
        flat_coords[:, 1] /= -OUTPUT_RESOLUTION
        # Subtract the column and row offsets.
        flat_coords[:, 0] -= col * OUTPUT_SIZE
        flat_coords[:, 1] -= row * OUTPUT_SIZE
        coords = flat_coords.reshape(coords.shape)
        return coords.astype(np.int32)

    with in_fname.open() as f:
        fc = json.load(f)

    array = np.zeros((len(CATEGORIES), OUTPUT_SIZE, OUTPUT_SIZE), dtype=np.uint8)

    for feat in fc["features"]:
        category = feat["properties"]["category"]
        if category not in CATEGORIES:
            continue
        category_id = CATEGORIES.index(category)
        geometry = feat["geometry"]
        if geometry["type"] == "Polygon":
            draw_polygon(array, geometry["coordinates"], category_id, transform)
        elif geometry["type"] == "LineString":
            draw_line_string(array, geometry["coordinates"], category_id, transform)
        elif geometry["type"] == "Point":
            coords = transform(np.array(geometry["coordinates"]))
            if coords[0] < 0 or coords[0] >= OUTPUT_SIZE:
                continue
            if coords[1] < 0 or coords[1] >= OUTPUT_SIZE:
                continue
            array[category_id, coords[1], coords[0]] = 1
        else:
            raise ValueError(f"cannot handle geometry type {geometry['type']}")

    out_fname = (
        in_fname.parent.parent
        / OUTPUT_MODALITY
        / f"{crs}_{col}_{row}_{OUTPUT_RESOLUTION}.tif"
    )
    bounds = (
        col * OUTPUT_SIZE,
        row * OUTPUT_SIZE,
        (col + 1) * OUTPUT_SIZE,
        (row + 1) * OUTPUT_SIZE,
    )
    GEOTIFF_RASTER_FORMAT.encode_raster(
        path=out_fname.parent,
        projection=Projection(crs, OUTPUT_RESOLUTION, -OUTPUT_RESOLUTION),
        bounds=bounds,
        array=array,
        fname=out_fname.name,
    )


if __name__ == "__main__":
    multiprocessing.set_start_method("forkserver")

    parser = argparse.ArgumentParser(
        description="Rasterize OpenStreetMap",
    )
    parser.add_argument(
        "--helios_path",
        type=str,
        help="Destination Helios dataset path",
        required=True,
    )
    parser.add_argument(
        "--workers",
        type=int,
        help="Number of workers to use",
        default=32,
    )
    args = parser.parse_args()

    helios_path = UPath(args.helios_path)

    geojson_fnames = list((helios_path / "10_openstreetmap").iterdir())
    p = multiprocessing.Pool(args.workers)
    outputs = p.imap_unordered(rasterize_openstreetmap, geojson_fnames)
    for _ in tqdm.tqdm(outputs, total=len(geojson_fnames)):
        pass
    p.close()
