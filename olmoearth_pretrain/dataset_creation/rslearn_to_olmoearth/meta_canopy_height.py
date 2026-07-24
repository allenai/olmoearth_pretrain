"""Post-process ingested Meta Canopy Height V2 data into the OlmoEarth Pretrain dataset."""

import argparse
import csv
import multiprocessing
from datetime import UTC, datetime

import numpy as np
import tqdm
from rslearn.dataset import Dataset, Window
from rslearn.utils.mp import star_imap_unordered
from rslearn.utils.raster_array import RasterArray
from upath import UPath

from olmoearth_pretrain.data.constants import Modality, TimeSpan
from olmoearth_pretrain.dataset.utils import get_modality_fname

from ..constants import GEOTIFF_RASTER_FORMAT, METADATA_COLUMNS
from ..util import get_modality_temp_meta_fname, get_window_metadata

# Fake time range; the Meta Canopy Height V2 source imagery is primarily from 2018-2020.
START_TIME = datetime(2018, 1, 1, tzinfo=UTC)
END_TIME = datetime(2021, 1, 1, tzinfo=UTC)

# Layer name in the input rslearn dataset.
LAYER_NAME = "meta_canopy_height"

# nodata value for the uint8 canopy height band.
NODATA_VALUE = 255
# Skip windows where fewer than this fraction of pixels have positive canopy height.
# Mirrors the wri_canopy_height_map converter; set to 0.0 to keep every window.
MIN_POSITIVE_FRACTION = 0.2


def convert_meta_canopy_height(window: Window, olmoearth_path: UPath) -> None:
    """Add Meta Canopy Height V2 data for this window to the OlmoEarth Pretrain dataset.

    Args:
        window: the rslearn window to read data from.
        olmoearth_path: OlmoEarth Pretrain dataset path to write to.
    """
    window_metadata = get_window_metadata(window)

    if not window.is_layer_completed(LAYER_NAME):
        return

    assert len(Modality.META_CANOPY_HEIGHT.band_sets) == 1
    band_set = Modality.META_CANOPY_HEIGHT.band_sets[0]
    raster_dir = window.get_raster_dir(LAYER_NAME, band_set.bands)
    image = GEOTIFF_RASTER_FORMAT.decode_raster(
        raster_dir, window.projection, window.bounds
    ).get_chw_array()

    # Skip areas with any nodata.
    if image.max() == NODATA_VALUE:
        return
    # Also skip if there are not enough positive pixels.
    if np.count_nonzero(image) / image.size < MIN_POSITIVE_FRACTION:
        return

    dst_fname = get_modality_fname(
        olmoearth_path,
        Modality.META_CANOPY_HEIGHT,
        TimeSpan.STATIC,
        window_metadata,
        band_set.get_resolution(),
        "tif",
    )
    GEOTIFF_RASTER_FORMAT.encode_raster(
        path=dst_fname.parent,
        projection=window.projection,
        bounds=window.bounds,
        raster=RasterArray(chw_array=image),
        fname=dst_fname.name,
    )
    metadata_fname = get_modality_temp_meta_fname(
        olmoearth_path, Modality.META_CANOPY_HEIGHT, TimeSpan.STATIC, window.name
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
    outputs = star_imap_unordered(p, convert_meta_canopy_height, jobs)
    for _ in tqdm.tqdm(outputs, total=len(jobs)):
        pass
    p.close()
