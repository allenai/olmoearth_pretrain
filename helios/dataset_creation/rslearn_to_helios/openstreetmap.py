"""Post-process ingested OpenStreetMap data into the Helios dataset."""

import argparse
import csv
import multiprocessing
from datetime import datetime, timezone

import tqdm
from rslearn.dataset import Window
from rslearn.utils.geometry import Projection
from rslearn.utils.mp import star_imap_unordered
from rslearn.utils.vector_format import GeojsonCoordinateMode, GeojsonVectorFormat
from upath import UPath

from ..constants import METADATA_COLUMNS
from ..util import get_modality_fname, get_modality_temp_meta_fname, parse_window_name

# Placeholder time range for OpenStreetMap.
START_TIME = datetime(2020, 1, 1, tzinfo=timezone.utc)
END_TIME = datetime(2025, 1, 1, tzinfo=timezone.utc)

# Layer name in the input rslearn dataset.
LAYER_NAME = "openstreetmap"

# Modality in the output Helios dataset.
MODALITY = "10_openstreetmap"

RESOLUTION = 0.625
# Coordinates of OSM features are 16x zoomed in from the 10 m/pixel tiles.
FACTOR = 16


def convert_openstreetmap(window_path: UPath, helios_path: UPath) -> None:
    """Add OpenStreetMap data for this window to the Helios dataset.

    Args:
        window_path: the rslearn window directory to read data from.
        helios_path: Helios dataset path to write to.
    """
    window = Window.load(window_path)
    window_metadata = parse_window_name(window.name)
    vector_format = GeojsonVectorFormat(coordinate_mode=GeojsonCoordinateMode.CRS)

    if not window.is_layer_completed(LAYER_NAME):
        return

    layer_dir = window.get_layer_dir(LAYER_NAME)
    adjusted_bounds = (
        window.bounds[0] * FACTOR,
        window.bounds[1] * FACTOR,
        window.bounds[2] * FACTOR,
        window.bounds[3] * FACTOR,
    )
    features = vector_format.decode_vector(layer_dir, adjusted_bounds)
    dst_fname = get_modality_fname(
        helios_path, MODALITY, window_metadata, RESOLUTION, "geojson"
    )
    dst_fname.parent.mkdir(parents=True, exist_ok=True)
    vector_format.encode_to_file(
        fname=dst_fname,
        projection=Projection(window.projection.crs, RESOLUTION, -RESOLUTION),
        features=features,
    )
    metadata_fname = get_modality_temp_meta_fname(helios_path, MODALITY, window.name)
    metadata_fname.parent.mkdir(parents=True, exist_ok=True)
    with metadata_fname.open("w") as f:
        writer = csv.DictWriter(f, fieldnames=METADATA_COLUMNS)
        writer.writeheader()
        writer.writerow(
            dict(
                crs=window_metadata.crs,
                col=window_metadata.col,
                row=window_metadata.row,
                tile_time=window_metadata.time.isoformat(),
                image_idx="N/A",
                start_time=START_TIME.isoformat(),
                end_time=END_TIME.isoformat(),
            )
        )


if __name__ == "__main__":
    multiprocessing.set_start_method("forkserver")

    parser = argparse.ArgumentParser(
        description="Post-process Helios data",
    )
    parser.add_argument(
        "--ds_path",
        type=str,
        help="Source rslearn dataset path",
        required=True,
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

    ds_path = UPath(args.ds_path)
    helios_path = UPath(args.helios_path)

    metadata_fnames = ds_path.glob("windows/*/*/metadata.json")
    jobs = []
    for metadata_fname in metadata_fnames:
        jobs.append(
            dict(
                window_path=metadata_fname.parent,
                helios_path=helios_path,
            )
        )

    p = multiprocessing.Pool(args.workers)
    outputs = star_imap_unordered(p, convert_openstreetmap, jobs)
    for _ in tqdm.tqdm(outputs, total=len(jobs)):
        pass
    p.close()
