"""Post-process ingested OpenStreetMap data into the Helios dataset.

OpenStreetMap is vector data, so we want to keep the precision of the data as high as
possible, but the data size (i.e. bytes) is also small enough that we can store it
under the 10 m/pixel tiles without needing too much storage space.

So, we use the 10 m/pixel grid, but store it with 16x zoomed in coordinates (meaning
the coordinates actually match those of the 0.625 m/pixel tiles). This way we can use
the data for training even at coarser resolution.
"""

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

from helios.data.constants import Modality, TimeSpan
from helios.dataset.util import get_modality_fname

from ..constants import METADATA_COLUMNS
from ..util import get_modality_temp_meta_fname, get_window_metadata

# Placeholder time range for OpenStreetMap.
START_TIME = datetime(2020, 1, 1, tzinfo=timezone.utc)
END_TIME = datetime(2025, 1, 1, tzinfo=timezone.utc)

# Layer name in the input rslearn dataset.
LAYER_NAME = "openstreetmap"

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
    window_metadata = get_window_metadata(window)
    vector_format = GeojsonVectorFormat(coordinate_mode=GeojsonCoordinateMode.CRS)

    if not window.is_layer_completed(LAYER_NAME):
        return

    # Load the vector data.
    # decode_vector requires bounds to be passed, but the window bounds need to be
    # adjusted by the zoom offset to match that of the stored data.
    layer_dir = window.get_layer_dir(LAYER_NAME)
    adjusted_bounds = (
        window.bounds[0] * FACTOR,
        window.bounds[1] * FACTOR,
        window.bounds[2] * FACTOR,
        window.bounds[3] * FACTOR,
    )
    features = vector_format.decode_vector(layer_dir, adjusted_bounds)

    # Upload the data.
    dst_fname = get_modality_fname(
        helios_path,
        Modality.OPENSTREETMAP,
        TimeSpan.STATIC,
        window_metadata,
        RESOLUTION,
        "geojson",
    )
    dst_fname.parent.mkdir(parents=True, exist_ok=True)
    vector_format.encode_to_file(
        fname=dst_fname,
        projection=Projection(window.projection.crs, RESOLUTION, -RESOLUTION),
        features=features,
    )

    # Create the metadata file for this data.
    metadata_fname = get_modality_temp_meta_fname(
        helios_path, Modality.OPENSTREETMAP, TimeSpan.STATIC, window.name
    )
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
