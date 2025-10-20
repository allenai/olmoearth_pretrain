"""Post-process ingested OpenStreetMap data into the OlmoEarth Pretrain dataset.

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
from datetime import UTC, datetime

import tqdm
from rslearn.dataset import Window
from rslearn.utils.mp import star_imap_unordered
from rslearn.utils.vector_format import GeojsonCoordinateMode, GeojsonVectorFormat
from upath import UPath

from olmoearth_pretrain.data.constants import Modality, TimeSpan
from olmoearth_pretrain.dataset.utils import get_modality_fname

from ..constants import METADATA_COLUMNS
from ..util import get_modality_temp_meta_fname, get_window_metadata

# Placeholder time range for OpenStreetMap.
START_TIME = datetime(2020, 1, 1, tzinfo=UTC)
END_TIME = datetime(2025, 1, 1, tzinfo=UTC)

# Layer name in the input rslearn dataset.
LAYER_NAME = "openstreetmap"

RESOLUTION = 10


def convert_openstreetmap(window_path: UPath, helios_path: UPath) -> None:
    """Add OpenStreetMap data for this window to the OlmoEarth Pretrain dataset.

    Args:
        window_path: the rslearn window directory to read data from.
        helios_path: OlmoEarth Pretrain dataset path to write to.
    """
    window = Window.load(window_path)
    layer_datas = window.load_layer_datas()

    if LAYER_NAME not in layer_datas:
        return

    layer_data = layer_datas[LAYER_NAME]
    window_metadata = get_window_metadata(window)
    vector_format = GeojsonVectorFormat(coordinate_mode=GeojsonCoordinateMode.CRS)

    # Load the vector data.
    # It may end up in multiple layers if there are different OpenStreetMap GeoJSONs
    # that match to the window due to just using their bounding box instead of actual
    # extent. So we need to concatenate the features across all of the layers.
    features = []
    for group_idx in range(len(layer_data.serialized_item_groups)):
        layer_dir = window.get_layer_dir(LAYER_NAME, group_idx=group_idx)
        cur_features = vector_format.decode_vector(
            layer_dir, window.projection, window.bounds
        )
        features.extend(cur_features)

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
        description="Post-process OlmoEarth Pretrain data",
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
