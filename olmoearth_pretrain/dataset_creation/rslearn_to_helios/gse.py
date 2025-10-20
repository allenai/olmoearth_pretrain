"""Post-process ingested Google Satellite Embedding data into the OlmoEarth Pretrain dataset."""

import argparse
import csv
import multiprocessing
from datetime import datetime, timedelta

import tqdm
from rslearn.data_sources import Item
from rslearn.dataset import Window
from rslearn.utils.mp import star_imap_unordered
from upath import UPath

from olmoearth_pretrain.data.constants import Modality, TimeSpan
from olmoearth_pretrain.dataset.utils import get_modality_fname

from ..constants import GEOTIFF_RASTER_FORMAT, METADATA_COLUMNS
from ..util import get_modality_temp_meta_fname, get_window_metadata

# Layer name in the input rslearn dataset.
LAYER_NAME = "gse"


def convert_gse(window_path: UPath, helios_path: UPath) -> None:
    """Add Google Satellite Embedding data for this window to the OlmoEarth Pretrain dataset.

    Args:
        window_path: the rslearn window directory to read data from.
        helios_path: OlmoEarth Pretrain dataset path to write to.
    """
    window = Window.load(window_path)
    window_metadata = get_window_metadata(window)

    if not window.is_layer_completed(LAYER_NAME):
        return

    # Get start time of the mosaic.
    # We compute end time by adding a year. This is because the GEE data source sets
    # the end time to match the start time and we don't want that behavior.
    layer_datas = window.load_layer_datas()
    item_groups = layer_datas[LAYER_NAME].serialized_item_groups
    if len(item_groups) == 0:
        return
    item_group = item_groups[0]

    start_time: datetime | None = None
    for item_data in item_group:
        item = Item.deserialize(item_data)
        if start_time is None or item.geometry.time_range[0] < start_time:
            start_time = item.geometry.time_range[0]
    assert start_time is not None
    end_time = start_time + timedelta(days=365)

    assert len(Modality.GSE.band_sets) == 1
    band_set = Modality.GSE.band_sets[0]
    raster_dir = window.get_raster_dir(LAYER_NAME, band_set.bands)
    image = GEOTIFF_RASTER_FORMAT.decode_raster(
        raster_dir, window.projection, window.bounds
    )
    dst_fname = get_modality_fname(
        helios_path,
        Modality.GSE,
        TimeSpan.STATIC,
        window_metadata,
        band_set.get_resolution(),
        "tif",
    )
    GEOTIFF_RASTER_FORMAT.encode_raster(
        path=dst_fname.parent,
        projection=window.projection,
        bounds=window.bounds,
        array=image,
        fname=dst_fname.name,
    )
    metadata_fname = get_modality_temp_meta_fname(
        helios_path, Modality.GSE, TimeSpan.STATIC, window.name
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
                image_idx="0",
                start_time=start_time.isoformat(),
                end_time=end_time.isoformat(),
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

    metadata_fnames = ds_path.glob("windows/res_10/*/metadata.json")
    jobs = []
    for metadata_fname in metadata_fnames:
        jobs.append(
            dict(
                window_path=metadata_fname.parent,
                helios_path=helios_path,
            )
        )

    p = multiprocessing.Pool(args.workers)
    outputs = star_imap_unordered(p, convert_gse, jobs)
    for _ in tqdm.tqdm(outputs, total=len(jobs)):
        pass
    p.close()
