"""Post-process EuroCrops data from rslearn to OlmoEarth format.

EuroCrops is vector data with HCAT crop type codes. We rasterize it during the
conversion process. The HCAT codes are hierarchical 10-digit codes. We convert them to
flat class IDs using the mapping in data/eurocrops_hcat3_mapping.json, which was
generated from the HCAT3.csv file from https://zenodo.org/records/14094196. The IDs are
assigned in file order so that hierarchically similar crops have close IDs.
"""

import argparse
import csv
import json
import multiprocessing
from collections.abc import Callable

import numpy as np
import numpy.typing as npt
import shapely
import skimage.draw
import tqdm
from rasterio.crs import CRS
from rslearn.data_sources import Item
from rslearn.dataset import Dataset, Window
from rslearn.utils.geometry import flatten_shape
from rslearn.utils.mp import star_imap_unordered
from rslearn.utils.vector_format import GeojsonVectorFormat
from upath import UPath

from olmoearth_pretrain.data.constants import Modality, TimeSpan
from olmoearth_pretrain.dataset.utils import get_modality_dir

from ..constants import GEOTIFF_RASTER_FORMAT, METADATA_COLUMNS

# Use the EUROCROPS modality from constants.
MODALITY = Modality.EUROCROPS

# Path to the HCAT3 mapping JSON file
HCAT3_MAPPING_PATH = "data/eurocrops_hcat3_mapping.json"

# Layer name in the input rslearn dataset.
LAYER_NAME = "eurocrops"

# Property name in the EuroCrops features that contains the HCAT code.
HCAT_CODE_PROPERTY = "EC_hcat_c"


def draw_polygon(
    array: npt.NDArray,
    coords: list[list[list[float]]],
    class_id: int,
    transform: Callable[[npt.NDArray], npt.NDArray],
    output_size: tuple[int, int],
) -> None:
    """Draw a polygon on the array.

    Args:
        array: the array to write to (H, W).
        coords: list of coordinate rings. coords[0] is the exterior ring (list of
            [x, y] points), and coords[1:] are interior holes.
        class_id: the class ID to fill the polygon with.
        transform: transform to apply on the coordinates.
        output_size: (height, width) of the output array.
    """
    exterior = transform(np.array(coords[0]))
    rows, cols = skimage.draw.polygon(exterior[:, 1], exterior[:, 0], shape=output_size)

    # If this polygon has no holes, we can draw it directly.
    if len(coords) == 1:
        array[rows, cols] = class_id
        return

    # Otherwise, create a mask from the exterior and negate the holes.
    mask = np.zeros(output_size, dtype=bool)
    mask[rows, cols] = True

    for ring in coords[1:]:
        interior = transform(np.array(ring))
        hole_rows, hole_cols = skimage.draw.polygon(
            interior[:, 1], interior[:, 0], shape=output_size
        )
        mask[hole_rows, hole_cols] = False

    array[mask] = class_id


def convert_eurocrops(
    window: Window,
    olmoearth_path: UPath,
    hcat_to_id: dict[int, int],
) -> None:
    """Convert EuroCrops data for this window to OlmoEarth format.

    Args:
        window: the rslearn window to read data from.
        olmoearth_path: OlmoEarth Pretrain dataset path to write to.
        hcat_to_id: mapping from HCAT code to flat class ID.
    """
    layer_datas = window.load_layer_datas()
    if LAYER_NAME not in layer_datas:
        return
    layer_data = layer_datas[LAYER_NAME]

    # Get start and end time from the EuroCrops item.
    item_groups = layer_data.serialized_item_groups
    if len(item_groups) == 0:
        return
    item = Item.deserialize(item_groups[0][0])
    start_time = item.geometry.time_range[0]
    end_time = item.geometry.time_range[1]

    # Load vector data from all groups.
    # We skip if some GeoJSONs have bad polygons or if we got no polygons.
    try:
        features = []
        for group_idx in range(len(item_groups)):
            layer_dir = window.get_layer_dir(LAYER_NAME, group_idx=group_idx)
            cur_features = GeojsonVectorFormat().decode_vector(
                layer_dir, window.projection, window.bounds
            )
            features.extend(cur_features)

        if not features:
            return
    except Exception as e:
        print(
            f"warning: skipping window {window.name} since we got error reading eurocrops polygons: {e}"
        )
        return

    # Parse window metadata from name.
    fname_parts = window.name.split("_")
    crs = CRS.from_string(fname_parts[0])
    col = int(fname_parts[2])
    row = int(fname_parts[3])

    # Get output size from window bounds.
    output_width = window.bounds[2] - window.bounds[0]
    output_height = window.bounds[3] - window.bounds[1]
    output_size = (output_height, output_width)

    def transform(coords: npt.NDArray) -> npt.NDArray:
        """Subtract window offset from pixel coordinates to get relative coordinates."""
        # Input coords must be in absolute pixel coordinates.
        # This should be the case since we read the features in window.projection.
        flat_coords = coords.reshape(-1, 2)
        # Subtract the window bounds offset.
        flat_coords[:, 0] -= window.bounds[0]
        flat_coords[:, 1] -= window.bounds[1]
        coords = flat_coords.reshape(coords.shape)
        return coords.astype(np.int32)

    # Create raster array. Use uint16 to support many classes.
    # Shape is (1, H, W) for single band.
    array = np.zeros((1, output_height, output_width), dtype=np.uint16)

    for feat in features:
        # Get the HCAT code from properties.
        hcat_code = feat.properties.get(HCAT_CODE_PROPERTY)
        if hcat_code is None:
            continue

        # Convert to int if it's a string.
        if isinstance(hcat_code, str):
            hcat_code = int(hcat_code)

        # Get the flat class ID.
        class_id = hcat_to_id.get(hcat_code)
        if class_id is None:
            continue

        # Iterate over polygons.
        for shp in flatten_shape(feat.geometry.shp):
            if not isinstance(shp, shapely.Polygon):
                # Sometimes there are LineStrings and stuff, we should skip those bad ones.
                print(f"got invalid shape {shp} in window {window.name}")
                return

            coords = [shp.exterior.coords[:]]
            for interior in shp.interiors:
                coords.append(interior.coords[:])
            draw_polygon(array[0], coords, class_id, transform, output_size)

    # Skip if no crops were rasterized (all background).
    if array.max() == 0:
        return

    # Write the rasterized data as GeoTIFF.
    out_modality_dir = get_modality_dir(olmoearth_path, MODALITY, TimeSpan.STATIC)
    out_fname = (
        out_modality_dir / f"{crs}_{col}_{row}_{window.projection.x_resolution}.tif"
    )
    GEOTIFF_RASTER_FORMAT.encode_raster(
        path=out_fname.parent,
        projection=window.projection,
        bounds=window.bounds,
        array=array,
        fname=out_fname.name,
    )

    # Write metadata.
    metadata_dir = olmoearth_path / f"{out_modality_dir.name}_meta"
    metadata_dir.mkdir(parents=True, exist_ok=True)
    metadata_fname = metadata_dir / f"{window.name}.csv"
    with metadata_fname.open("w") as f:
        writer = csv.DictWriter(f, fieldnames=METADATA_COLUMNS)
        writer.writeheader()
        writer.writerow(
            dict(
                crs=str(crs),
                col=col,
                row=row,
                tile_time=window.time_range[0].isoformat() if window.time_range else "",
                image_idx="0",
                start_time=start_time.isoformat(),
                end_time=end_time.isoformat(),
            )
        )


if __name__ == "__main__":
    multiprocessing.set_start_method("forkserver")

    parser = argparse.ArgumentParser(
        description="Convert EuroCrops from rslearn to OlmoEarth format with rasterization",
    )
    parser.add_argument(
        "--ds_path",
        type=str,
        help="Source rslearn dataset path",
        required=True,
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

    # Load HCAT3 mapping from local JSON file.
    # ID 0 is reserved for background/nodata.
    print(f"Loading HCAT mapping from {HCAT3_MAPPING_PATH}...")
    with open(HCAT3_MAPPING_PATH) as f:
        mapping_list = json.load(f)
    hcat_to_id = {entry["hcat_code"]: entry["flat_id"] for entry in mapping_list}
    print(f"Loaded {len(hcat_to_id)} HCAT codes")

    dataset = Dataset(UPath(args.ds_path))
    olmoearth_path = UPath(args.olmoearth_path)

    # Ensure output directory exists.
    out_modality_dir = get_modality_dir(olmoearth_path, MODALITY, TimeSpan.STATIC)
    out_modality_dir.mkdir(parents=True, exist_ok=True)

    # Process all windows.
    jobs = []
    for window in dataset.load_windows(workers=args.workers, show_progress=True):
        jobs.append(
            dict(
                window=window,
                olmoearth_path=olmoearth_path,
                hcat_to_id=hcat_to_id,
            )
        )

    print(f"Processing {len(jobs)} windows...")
    p = multiprocessing.Pool(args.workers)
    outputs = star_imap_unordered(p, convert_eurocrops, jobs)
    for _ in tqdm.tqdm(outputs, total=len(jobs)):
        pass
    p.close()

    print("Done!")
