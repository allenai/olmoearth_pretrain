"""Post-process ingested OlmoEarth-v1-Base embedding data into the OlmoEarth Pretrain dataset."""

import argparse
import csv
import multiprocessing
from datetime import datetime

import numpy as np
import tqdm
from rslearn.data_sources import Item
from rslearn.dataset import Dataset, Window
from rslearn.utils.mp import star_imap_unordered
from rslearn.utils.raster_array import RasterArray
from upath import UPath

from olmoearth_pretrain.data.constants import Modality, TimeSpan
from olmoearth_pretrain.dataset.utils import get_modality_fname

from ..constants import GEOTIFF_RASTER_FORMAT, METADATA_COLUMNS
from ..util import get_modality_temp_meta_fname, get_window_metadata
from .multitemporal_raster import get_adjusted_projection_and_bounds

LAYER_NAME = "olmoearth_v1_base_embedding"


def convert_olmoearth_v1_base_embedding(window: Window, olmoearth_path: UPath) -> None:
    """Add OlmoEarth-v1-Base embedding data for this window to the OlmoEarth Pretrain dataset.

    Args:
        window: the rslearn window to read data from.
        olmoearth_path: OlmoEarth Pretrain dataset path to write to.
    """
    window_metadata = get_window_metadata(window)

    if not window.is_layer_completed(LAYER_NAME):
        return

    # Get start time of the source Sentinel-2 data.
    # We look at sentinel2_l2a_mo01 through sentinel2_l2a_mo12 to determine time range.
    layer_datas = window.load_layer_datas()

    # Find the time range from any available sentinel2 layer.
    start_time: datetime | None = None
    end_time: datetime | None = None
    for layer_name, layer_data in layer_datas.items():
        if not layer_name.startswith("sentinel2_l2a_mo"):
            continue
        for item_group in layer_data.serialized_item_groups:
            for item_data in item_group:
                item = Item.deserialize(item_data)
                t = item.geometry.time_range[0]
                if start_time is None or t < start_time:
                    start_time = t
                t = item.geometry.time_range[1]
                if end_time is None or t > end_time:
                    end_time = t

    if start_time is None or end_time is None:
        raise ValueError(
            f"Window {window.name} has embeddings but no sentinel2_l2a layers to determine time range"
        )

    assert len(Modality.OLMOEARTH_V1_BASE_EMBEDDING.band_sets) == 1
    band_set = Modality.OLMOEARTH_V1_BASE_EMBEDDING.band_sets[0]
    adjusted_projection, adjusted_bounds = get_adjusted_projection_and_bounds(
        Modality.OLMOEARTH_V1_BASE_EMBEDDING,
        band_set,
        window.projection,
        window.bounds,
    )
    raster_dir = window.get_raster_dir(LAYER_NAME, band_set.bands)
    raster = GEOTIFF_RASTER_FORMAT.decode_raster(
        raster_dir, adjusted_projection, adjusted_bounds
    )
    # Quantize float32 embeddings ([-1, 1]) to uint8 ([0, 255]).
    uint8_array = np.clip(raster.array * 128 + 128, 0, 255).astype(np.uint8)
    raster = RasterArray(
        array=uint8_array, timestamps=raster.timestamps, metadata=raster.metadata
    )
    dst_fname = get_modality_fname(
        olmoearth_path,
        Modality.OLMOEARTH_V1_BASE_EMBEDDING,
        TimeSpan.STATIC,
        window_metadata,
        band_set.get_resolution(),
        "tif",
    )
    GEOTIFF_RASTER_FORMAT.encode_raster(
        path=dst_fname.parent,
        projection=adjusted_projection,
        bounds=adjusted_bounds,
        raster=raster,
        fname=dst_fname.name,
    )
    metadata_fname = get_modality_temp_meta_fname(
        olmoearth_path,
        Modality.OLMOEARTH_V1_BASE_EMBEDDING,
        TimeSpan.STATIC,
        window.name,
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
            )
        )

    p = multiprocessing.Pool(args.workers)
    outputs = star_imap_unordered(p, convert_olmoearth_v1_base_embedding, jobs)
    for _ in tqdm.tqdm(outputs, total=len(jobs)):
        pass
    p.close()
