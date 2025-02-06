"""Post-process ingested NAIP data into the Helios dataset."""

import argparse
import csv
import multiprocessing

import tqdm
from rslearn.data_sources import Item
from rslearn.dataset import Window
from rslearn.utils.mp import star_imap_unordered
from rslearn.utils.raster_format import GeotiffRasterFormat
from upath import UPath

from ..constants import METADATA_COLUMNS
from ..util import get_modality_fname, get_modality_temp_meta_fname, parse_window_name

BANDS = [
    "R",
    "G",
    "B",
    "IR",
]
RESOLUTION = 0.625

# Layer name in the input rslearn dataset.
LAYER_NAME = "naip"

# Modality in the output Helios dataset.
MODALITY = "0.625_naip"


def convert_naip(window_path: UPath, helios_path: UPath) -> None:
    """Add NAIP data for this window to the Helios dataset.

    Args:
        window_path: the rslearn window directory to read data from.
        helios_path: Helios dataset path to write to.
    """
    window = Window.load(window_path)
    window_metadata = parse_window_name(window.name)
    layer_datas = window.load_layer_datas()
    raster_format = GeotiffRasterFormat()

    # NAIP is just one mosaic.
    item_groups = layer_datas[LAYER_NAME].serialized_item_groups
    if len(item_groups) == 0:
        return
    item_group = item_groups[0]

    # Get start and end of mosaic.
    start_time = None
    end_time = None
    for item_data in item_group:
        item = Item.deserialize(item_data)
        if start_time is None or item.geometry.time_range[0] < start_time:
            start_time = item.geometry.time_range[0]
        if end_time is None or item.geometry.time_range[1] > end_time:
            end_time = item.geometry.time_range[1]

    # Assert for type checking: we already checked that len(item_groups) > 0 so the
    # times should never be None.
    assert start_time is not None and end_time is not None  # nosec

    raster_dir = window.get_raster_dir(LAYER_NAME, BANDS)
    image = raster_format.decode_raster(raster_dir, window.bounds)
    dst_fname = get_modality_fname(
        helios_path, MODALITY, window_metadata, RESOLUTION, "tif"
    )
    raster_format.encode_raster(
        path=dst_fname.parent,
        projection=window.projection,
        bounds=window.bounds,
        array=image,
        fname=dst_fname.name,
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
                image_idx="0",
                start_time=start_time.isoformat(),
                end_time=end_time.isoformat(),
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

    metadata_fnames = ds_path.glob("windows/res_0.625/*/metadata.json")
    jobs = []
    for metadata_fname in metadata_fnames:
        jobs.append(
            dict(
                window_path=metadata_fname.parent,
                helios_path=helios_path,
            )
        )

    p = multiprocessing.Pool(args.workers)
    outputs = star_imap_unordered(p, convert_naip, jobs)
    for _ in tqdm.tqdm(outputs, total=len(jobs)):
        pass
    p.close()
