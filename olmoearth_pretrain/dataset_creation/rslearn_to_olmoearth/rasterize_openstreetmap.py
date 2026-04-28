"""Create openstreetmap_raster from openstreetmap in the OlmoEarth Pretrain dataset."""

import argparse
import csv
import json
import multiprocessing
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime

import numpy as np
import numpy.typing as npt
import skimage.draw
import tqdm
from rasterio.crs import CRS
from rslearn.utils.geometry import Projection
from rslearn.utils.mp import star_imap_unordered
from upath import UPath

from olmoearth_pretrain.data.constants import Modality, TimeSpan
from olmoearth_pretrain.dataset.utils import (
    WindowMetadata,
    get_modality_dir,
    get_modality_fname,
)

from ..constants import GEOTIFF_RASTER_FORMAT
from ..util import parse_bool
from .raster_api import encode_chw_raster

WINDOW_SIZE = 256
# Factor to zoom in for output. So output will be 1024x1024.
FACTOR = 4
OUTPUT_SIZE = WINDOW_SIZE * FACTOR
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


@dataclass(frozen=True)
class OpenStreetMapRasterJob:
    """A rasterization job for one OpenStreetMap sample."""

    in_fname: UPath
    out_fname: UPath
    projection: Projection
    bounds: tuple[int, int, int, int]


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


def rasterize_openstreetmap(job: OpenStreetMapRasterJob) -> None:
    """Rasterize OpenStreetMap data.

    Args:
        job: the rasterization job for one OpenStreetMap sample.
    """

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
        flat_coords[:, 0] /= OUTPUT_RESOLUTION
        flat_coords[:, 1] /= -OUTPUT_RESOLUTION
        flat_coords[:, 0] -= job.bounds[0]
        flat_coords[:, 1] -= job.bounds[1]
        coords = flat_coords.reshape(coords.shape)
        return coords.astype(np.int32)

    with job.in_fname.open() as f:
        fc = json.load(f)

    array = np.zeros((len(CATEGORIES), OUTPUT_SIZE, OUTPUT_SIZE), dtype=np.uint8)

    for feat in fc["features"]:
        # Get the category ID, which indicates the channel to rasterize on.
        category = feat["properties"]["category"]
        if category not in CATEGORIES:
            continue
        category_id = CATEGORIES.index(category)

        # Now rasterize based on the geometry type.
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
        elif geometry["type"] == "MultiLineString":
            for line_string_coords in geometry["coordinates"]:
                draw_line_string(array, line_string_coords, category_id, transform)
        elif geometry["type"] == "MultiPolygon":
            for polygon_coords in geometry["coordinates"]:
                draw_polygon(array, polygon_coords, category_id, transform)
        else:
            raise ValueError(f"cannot handle geometry type {geometry['type']}")

    # Upload the rasterized data as GeoTIFF.
    encode_chw_raster(
        GEOTIFF_RASTER_FORMAT,
        path=job.out_fname.parent,
        projection=job.projection,
        bounds=job.bounds,
        array=array,
        fname=job.out_fname.name,
    )


def _parse_optional_int(value: str | None) -> int | None:
    if value is None or value == "":
        return None
    return int(value)


def _parse_optional_float(value: str | None) -> float | None:
    if value is None or value == "":
        return None
    return float(value)


def _parse_optional_str(value: str | None) -> str | None:
    if value is None or value == "":
        return None
    return value


def _parse_bounds(csv_row: dict[str, str]) -> tuple[int, int, int, int]:
    if csv_row.get("bounds_left") in (None, ""):
        col = _parse_optional_int(csv_row.get("col"))
        row = _parse_optional_int(csv_row.get("row"))
        if col is None or row is None:
            raise ValueError("missing both bounds and legacy col/row metadata")
        return (
            col * WINDOW_SIZE,
            row * WINDOW_SIZE,
            (col + 1) * WINDOW_SIZE,
            (row + 1) * WINDOW_SIZE,
        )
    return (
        int(csv_row["bounds_left"]),
        int(csv_row["bounds_bottom"]),
        int(csv_row["bounds_right"]),
        int(csv_row["bounds_top"]),
    )


def _window_metadata_from_csv_row(csv_row: dict[str, str]) -> WindowMetadata:
    use_grid_reference = parse_bool(csv_row.get("use_grid_reference"), True)
    x_resolution = _parse_optional_float(csv_row.get("x_resolution")) or 10.0
    y_resolution = _parse_optional_float(csv_row.get("y_resolution")) or -10.0
    return WindowMetadata(
        crs=csv_row["crs"],
        resolution=abs(x_resolution),
        time=datetime.fromisoformat(csv_row["tile_time"]),
        col=_parse_optional_int(csv_row.get("col")),
        row=_parse_optional_int(csv_row.get("row")),
        sample_id=_parse_optional_str(csv_row.get("sample_id")),
        use_grid_reference=use_grid_reference,
        x_resolution=x_resolution,
        y_resolution=y_resolution,
        bounds=_parse_bounds(csv_row),
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
        window_metadata = _window_metadata_from_csv_row(csv_row)
        bounds = window_metadata.bounds
        if bounds is None:
            raise ValueError("window metadata must include bounds")
        rasterize_jobs.append(
            dict(
                job=OpenStreetMapRasterJob(
                    in_fname=get_modality_fname(
                        olmoearth_path,
                        Modality.OPENSTREETMAP,
                        TimeSpan.STATIC,
                        window_metadata,
                        10,
                        "geojson",
                    ),
                    out_fname=get_modality_fname(
                        olmoearth_path,
                        Modality.OPENSTREETMAP_RASTER,
                        TimeSpan.STATIC,
                        window_metadata,
                        OUTPUT_RESOLUTION,
                        "tif",
                    ),
                    projection=Projection(
                        CRS.from_string(window_metadata.crs),
                        OUTPUT_RESOLUTION,
                        -OUTPUT_RESOLUTION,
                    ),
                    bounds=(
                        bounds[0] * FACTOR,
                        bounds[1] * FACTOR,
                        bounds[2] * FACTOR,
                        bounds[3] * FACTOR,
                    ),
                )
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
