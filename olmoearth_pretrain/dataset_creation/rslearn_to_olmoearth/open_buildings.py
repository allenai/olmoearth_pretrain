"""Post-process materialized Open Buildings Temporal into the OlmoEarth Pretrain dataset."""

import argparse
import csv
import multiprocessing
from datetime import datetime, timedelta

import numpy as np
import tqdm
from rslearn.data_sources import Item
from rslearn.dataset import Dataset, Window
from rslearn.utils.mp import star_imap_unordered
from rslearn.utils.raster_array import RasterArray
from upath import UPath

from olmoearth_pretrain.data.constants import MISSING_VALUE, Modality, TimeSpan
from olmoearth_pretrain.dataset.utils import get_modality_fname

from ..constants import GEOTIFF_RASTER_FORMAT, METADATA_COLUMNS
from ..gee_data_sources import NODATA_VALUE
from ..util import get_modality_temp_meta_fname, get_window_metadata
from .cli import add_common_arguments
from .multitemporal_raster import get_adjusted_projection_and_bounds

# Default layer name in the input rslearn dataset.
LAYER_NAME = "open_buildings"

# Each annual raster is stamped at June 30 and represents the surrounding year.
HALF_YEAR = timedelta(days=182)


def convert_open_buildings(
    window: Window, olmoearth_path: UPath, layer_name: str = LAYER_NAME
) -> None:
    """Add Open Buildings Temporal data for this window to the OlmoEarth Pretrain dataset.

    Args:
        window: the rslearn window to read data from.
        olmoearth_path: OlmoEarth Pretrain dataset path to write to.
        layer_name: the layer to read in the rslearn dataset.
    """
    window_metadata = get_window_metadata(window)

    if not window.is_layer_completed(layer_name):
        return

    layer_datas = window.load_layer_datas()
    item_groups = layer_datas[layer_name].serialized_item_groups
    if len(item_groups) == 0:
        return
    item_time_range = Item.deserialize(item_groups[0][0]).geometry.time_range
    assert item_time_range is not None
    inference_time: datetime = item_time_range[0]

    assert len(Modality.OPEN_BUILDINGS.band_sets) == 1
    band_set = Modality.OPEN_BUILDINGS.band_sets[0]
    # The window is on the 10 m grid but the layer is stored at 2.5 m.
    projection, bounds = get_adjusted_projection_and_bounds(
        Modality.OPEN_BUILDINGS, band_set, window.projection, window.bounds
    )
    image = window.data.read_raster(
        layer_name,
        band_set.bands,
        GEOTIFF_RASTER_FORMAT,
        projection=projection,
        bounds=bounds,
    ).get_chw_array()

    # A pixel is missing if any band is (outside the dataset's coverage).
    missing = (image == NODATA_VALUE).any(axis=0, keepdims=True)
    if missing.all():
        return
    image = np.where(missing, MISSING_VALUE, image).astype(np.float32)

    dst_fname = get_modality_fname(
        olmoearth_path,
        Modality.OPEN_BUILDINGS,
        TimeSpan.STATIC,
        window_metadata,
        band_set.get_resolution(),
        "tif",
    )
    GEOTIFF_RASTER_FORMAT.encode_raster(
        path=dst_fname.parent,
        projection=projection,
        bounds=bounds,
        raster=RasterArray(chw_array=image),
        fname=dst_fname.name,
    )
    metadata_fname = get_modality_temp_meta_fname(
        olmoearth_path, Modality.OPEN_BUILDINGS, TimeSpan.STATIC, window.name
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
                start_time=(inference_time - HALF_YEAR).isoformat(),
                end_time=(inference_time + HALF_YEAR).isoformat(),
            )
        )


if __name__ == "__main__":
    multiprocessing.set_start_method("forkserver")

    parser = argparse.ArgumentParser(
        description="Post-process OlmoEarth Pretrain data",
    )
    add_common_arguments(parser)
    parser.add_argument(
        "--layer_name",
        type=str,
        default=LAYER_NAME,
        help="Layer to read in the rslearn dataset",
    )
    args = parser.parse_args()

    dataset = Dataset(UPath(args.ds_path))
    olmoearth_path = UPath(args.olmoearth_path)

    jobs = []
    for window in dataset.load_windows(
        workers=args.workers, show_progress=True, groups=["res_10"]
    ):
        jobs.append(
            dict(
                window=window,
                olmoearth_path=olmoearth_path,
                layer_name=args.layer_name,
            )
        )

    p = multiprocessing.Pool(args.workers)
    outputs = star_imap_unordered(p, convert_open_buildings, jobs)
    for _ in tqdm.tqdm(outputs, total=len(jobs)):
        pass
    p.close()
