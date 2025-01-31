"""Post-process ingested WorldCover data into the Helios dataset."""

import argparse
import csv
import multiprocessing
from datetime import datetime, timezone

import tqdm
from rslearn.dataset import Window
from rslearn.utils.mp import star_imap_unordered
from rslearn.utils.raster_format import GeotiffRasterFormat
from upath import UPath

from helios.constants import WORLDCOVER_BANDS

from ..const import METADATA_COLUMNS
from ..util import get_modality_fname, get_modality_temp_meta_fname

START_TIME = datetime(2021, 1, 1, tzinfo=timezone.utc)
END_TIME = datetime(2022, 1, 1, tzinfo=timezone.utc)

# Layer name in the input rslearn dataset.
LAYER_NAME = "worldcover"

# Modality in the output Helios dataset.
MODALITY = "worldcover"


def convert_worldcover(window_path: UPath, helios_path: UPath) -> None:
    """Add WorldCover data for this window to the Helios dataset.

    Args:
        window_path: the rslearn window directory to read data from.
        helios_path: Helios dataset path to write to.
    """
    window = Window.load(window_path)
    raster_format = GeotiffRasterFormat()

    if not window.is_layer_completed(LAYER_NAME):
        return

    raster_dir = window.get_raster_dir(LAYER_NAME, WORLDCOVER_BANDS)
    image = raster_format.decode_raster(raster_dir, window.bounds)
    dst_fname = get_modality_fname(helios_path, MODALITY, window.name, "tif")
    raster_format.encode_raster(
        path=dst_fname.parent,
        projection=window.projection,
        bounds=window.bounds,
        array=image,
        fname=dst_fname.name,
    )
    metadata_fname = get_modality_temp_meta_fname(helios_path, MODALITY, window.name)
    with metadata_fname.open("w") as f:
        writer = csv.DictWriter(f, fieldnames=METADATA_COLUMNS)
        writer.writeheader()
        writer.writerow(
            dict(
                example_id=window.name,
                image_idx="0",
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
    outputs = star_imap_unordered(p, convert_worldcover, jobs)
    for _ in tqdm.tqdm(outputs, total=len(jobs)):
        pass
    p.close()
