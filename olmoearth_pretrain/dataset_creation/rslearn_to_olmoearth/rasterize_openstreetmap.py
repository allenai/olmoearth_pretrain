"""Create openstreetmap_raster from openstreetmap in the OlmoEarth Pretrain dataset."""

import argparse
import csv
import json
import multiprocessing
from collections.abc import Callable
from datetime import datetime

import numpy as np
import numpy.typing as npt
import skimage.draw
import tqdm
from rasterio.crs import CRS
from rslearn.utils.geometry import Projection
from rslearn.utils.mp import star_imap_unordered
from rslearn.utils.raster_array import RasterArray
from upath import UPath

from olmoearth_pretrain.data.constants import Modality, TimeSpan
from olmoearth_pretrain.dataset.utils import (
    WindowMetadata,
    get_modality_dir,
    get_modality_fname,
)

from ..constants import GEOTIFF_RASTER_FORMAT

DEFAULT_WINDOW_SIZE = 256
# Factor to zoom in for output. So the output resolution is 10 m / FACTOR = 2.5 m and a
# window of N pixels at 10 m becomes an N * FACTOR pixel raster.
FACTOR = 4
OUTPUT_RESOLUTION = 10 / FACTOR
MODALITY_NAME = "openstreetmap_raster"

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
    output_size: int,
) -> None:
    """Draw a polygon on the array.

    Args:
        array: the array to write to.
        coords: the pixel coordinates of the polygon. coords[0] should correspond to
            the interior, while the remaining perimeters (if any) should be interior
            holes.
        category_id: the category of this polygon.
        transform: transform to apply on the coordinates.
        output_size: the height/width of the output raster in pixels.
    """
    exterior = transform(np.array(coords[0]))
    rows, cols = skimage.draw.polygon(
        exterior[:, 1], exterior[:, 0], shape=(output_size, output_size)
    )

    # If this polygon has no holes, we can draw it directly.
    # Otherwise, we create a mask from the exterior, but then negate the holes.
    if len(coords) == 1:
        array[category_id, rows, cols] = 1
        return

    mask = np.zeros((output_size, output_size), dtype=bool)
    mask[rows, cols] = True

    for ring in coords[1:]:
        interior = transform(np.array(ring))
        rows, cols = skimage.draw.polygon(
            interior[:, 1], interior[:, 0], shape=(output_size, output_size)
        )
        mask[rows, cols] = False

    array[category_id, mask] = 1


def draw_line_string(
    array: npt.NDArray,
    coords: list[list[float]],
    category_id: int,
    transform: Callable[[npt.NDArray], npt.NDArray],
    output_size: int,
) -> None:
    """Draw a line string on the array.

    Args:
        array: the array to write to.
        coords: the pixel coordinates of the line string.
        category_id: the category of this line string.
        transform: transform to apply on the coordinates.
        output_size: the height/width of the output raster in pixels.
    """
    coords = transform(np.array(coords))

    for i in range(len(coords) - 1):
        rows, cols = skimage.draw.line(
            coords[i][1], coords[i][0], coords[i + 1][1], coords[i + 1][0]
        )
        valid = (rows >= 0) & (rows < output_size) & (cols >= 0) & (cols < output_size)
        array[category_id, rows[valid], cols[valid]] = 1


def rasterize_openstreetmap(
    olmoearth_path: UPath,
    in_fname: UPath,
    window_size: int = DEFAULT_WINDOW_SIZE,
    pixel_coord_windows: bool = False,
    window_metadata: WindowMetadata | None = None,
) -> None:
    """Rasterize OpenStreetMap data.

    Args:
        olmoearth_path: path to OlmoEarth Pretrain dataset where OpenStreetMap vector data has been
            written.
        in_fname: the input filename containing the GeoJSON data. Outputs will be
            written to a corresponding name in the openstreetmap_raster folder.
        window_size: the window size in pixels at the 10 m base resolution. Defaults to
            256; the open-set dataset uses its 128 px window size.
        pixel_coord_windows: if True, metadata col/row are absolute pixel coordinates of
            the window center rather than grid-tile indices.
        window_metadata: identity and position read from the modality metadata CSV. If
            omitted, legacy grid metadata is parsed from ``in_fname``.
    """
    if window_metadata is None:
        fname_parts = in_fname.name.split(".")[0].split("_")
        window_metadata = WindowMetadata(
            crs=fname_parts[0],
            resolution=10,
            col=int(fname_parts[1]),
            row=int(fname_parts[2]),
            time=datetime.min,
        )
    crs = CRS.from_string(window_metadata.crs)
    col = window_metadata.col
    row = window_metadata.row

    output_size = window_size * FACTOR

    # Compute the origin (top-left) of the window in 10 m pixels.
    if pixel_coord_windows:
        # col/row are the window center in absolute 10 m pixel coordinates.
        origin_col = col - window_size // 2
        origin_row = row - window_size // 2
    else:
        # col/row are grid-tile indices; each tile is window_size pixels at 10 m.
        origin_col = col * window_size
        origin_row = row * window_size
    # Offsets in output-resolution (2.5 m) pixels.
    off_x = origin_col * FACTOR
    off_y = origin_row * FACTOR

    # Construct the transform from the input coordinates to coordinates within the
    # image. The input coordinates are in CRS units while we want the output to be in
    # pixel coordinates within the output image.
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
        # Subtract the window origin offsets.
        flat_coords[:, 0] -= off_x
        flat_coords[:, 1] -= off_y
        coords = flat_coords.reshape(coords.shape)
        return coords.astype(np.int32)

    with in_fname.open() as f:
        fc = json.load(f)

    array = np.zeros((len(CATEGORIES), output_size, output_size), dtype=np.uint8)

    for feat in fc["features"]:
        # Get the category ID, which indicates the channel to rasterize on.
        category = feat["properties"]["category"]
        if category not in CATEGORIES:
            continue
        category_id = CATEGORIES.index(category)

        # Now rasterize based on the geometry type.
        geometry = feat["geometry"]
        if geometry["type"] == "Polygon":
            draw_polygon(
                array, geometry["coordinates"], category_id, transform, output_size
            )
        elif geometry["type"] == "LineString":
            draw_line_string(
                array, geometry["coordinates"], category_id, transform, output_size
            )
        elif geometry["type"] == "Point":
            coords = transform(np.array(geometry["coordinates"]))
            if coords[0] < 0 or coords[0] >= output_size:
                continue
            if coords[1] < 0 or coords[1] >= output_size:
                continue
            array[category_id, coords[1], coords[0]] = 1
        elif geometry["type"] == "MultiLineString":
            for line_string_coords in geometry["coordinates"]:
                draw_line_string(
                    array, line_string_coords, category_id, transform, output_size
                )
        elif geometry["type"] == "MultiPolygon":
            for polygon_coords in geometry["coordinates"]:
                draw_polygon(array, polygon_coords, category_id, transform, output_size)
        else:
            raise ValueError(f"cannot handle geometry type {geometry['type']}")

    # Upload the rasterized data as GeoTIFF.
    out_fname = get_modality_fname(
        olmoearth_path,
        Modality.OPENSTREETMAP_RASTER,
        TimeSpan.STATIC,
        window_metadata,
        OUTPUT_RESOLUTION,
        "tif",
    )
    bounds = (
        off_x,
        off_y,
        off_x + output_size,
        off_y + output_size,
    )
    GEOTIFF_RASTER_FORMAT.encode_raster(
        path=out_fname.parent,
        projection=Projection(crs, OUTPUT_RESOLUTION, -OUTPUT_RESOLUTION),
        bounds=bounds,
        raster=RasterArray(chw_array=array),
        fname=out_fname.name,
    )


if __name__ == "__main__":
    multiprocessing.set_start_method("forkserver")

    parser = argparse.ArgumentParser(
        description="Rasterize OpenStreetMap",
    )
    parser.add_argument(
        "--olmoearth_path",
        type=str,
        help="Destination OlmoEarth Pretrain dataset path",
        required=True,
    )
    parser.add_argument(
        "--workers",
        type=int,
        help="Number of workers to use",
        default=32,
    )
    parser.add_argument(
        "--window_size",
        type=int,
        help="Window size in pixels at the 10 m base resolution (open-set uses 128)",
        default=DEFAULT_WINDOW_SIZE,
    )
    parser.add_argument(
        "--pixel_coord_windows",
        action="store_true",
        help=(
            "Set if metadata col/row are absolute pixel coordinates of the window "
            "center (e.g. the open-set dataset) rather than grid-tile indices"
        ),
    )
    args = parser.parse_args()

    olmoearth_path = UPath(args.olmoearth_path)

    src_modality_dir = get_modality_dir(
        olmoearth_path, Modality.OPENSTREETMAP, TimeSpan.STATIC
    )
    src_metadata_fname = olmoearth_path / f"{src_modality_dir.name}.csv"
    with src_metadata_fname.open() as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        if fieldnames is None:
            raise ValueError(f"got None for field names in {src_metadata_fname}")
        csv_rows = list(reader)

    rasterize_jobs = []
    for csv_row in csv_rows:
        window_metadata = WindowMetadata(
            crs=csv_row["crs"],
            resolution=10,
            col=int(csv_row["col"]),
            row=int(csv_row["row"]),
            time=datetime.fromisoformat(csv_row["tile_time"]),
            example_id=csv_row.get("example_id") or None,
        )
        geojson_fname = get_modality_fname(
            olmoearth_path,
            Modality.OPENSTREETMAP,
            TimeSpan.STATIC,
            window_metadata,
            10,
            "geojson",
        )
        rasterize_jobs.append(
            dict(
                olmoearth_path=olmoearth_path,
                in_fname=geojson_fname,
                window_size=args.window_size,
                pixel_coord_windows=args.pixel_coord_windows,
                window_metadata=window_metadata,
            )
        )
    p = multiprocessing.Pool(args.workers)
    outputs = star_imap_unordered(p, rasterize_openstreetmap, rasterize_jobs)
    for _ in tqdm.tqdm(outputs, total=len(rasterize_jobs)):
        pass
    p.close()

    # Also copy the metadata CSV but with image_idx replaced from "N/A" to "0".
    dst_modality_dir = get_modality_dir(
        olmoearth_path, Modality.OPENSTREETMAP_RASTER, TimeSpan.STATIC
    )
    dst_metadata_fname = olmoearth_path / f"{dst_modality_dir.name}.csv"
    for csv_row in csv_rows:
        if csv_row["image_idx"] != "N/A":
            raise ValueError("expected image_idx = N/A")
        csv_row["image_idx"] = "0"
    with dst_metadata_fname.open("w") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_rows)
