"""Post-process predicted WorldCover data into the OlmoEarth Pretrain dataset.

Unlike worldcover.py (which ingests the static ESA WorldCover 2021 map), this converts
the land-cover map predicted by the worldcover segmentation model from each window's own
Sentinel-2 stack. The prediction is written to the rslearn "output" layer by
`rslearn model predict`; here we copy it into the worldcover_pred modality and record the
window's real time range so the modality is time-aligned with the sample.
"""

import argparse
import csv
import multiprocessing

import tqdm
from rslearn.dataset import Dataset, Window
from rslearn.utils.mp import star_imap_unordered
from upath import UPath

from olmoearth_pretrain.data.constants import Modality, TimeSpan
from olmoearth_pretrain.dataset.utils import get_modality_fname

from ..constants import GEOTIFF_RASTER_FORMAT, METADATA_COLUMNS
from ..util import get_modality_temp_meta_fname, get_window_metadata

# Layer name in the input rslearn dataset (written by RslearnWriter during predict).
LAYER_NAME = "output"

# Band name that the prediction writer stores in the output layer (single-band argmax
# class map). Note this differs from the destination modality band name ("B1").
OUTPUT_BANDS = ["output"]


def convert_worldcover_pred(window: Window, olmoearth_path: UPath) -> None:
    """Add predicted WorldCover data for this window to the OlmoEarth Pretrain dataset.

    Args:
        window: the rslearn window to read data from.
        olmoearth_path: OlmoEarth Pretrain dataset path to write to.
    """
    window_metadata = get_window_metadata(window)

    if not window.is_layer_completed(LAYER_NAME):
        return

    assert len(Modality.WORLDCOVER_PRED.band_sets) == 1
    band_set = Modality.WORLDCOVER_PRED.band_sets[0]
    raster_dir = window.get_raster_dir(LAYER_NAME, OUTPUT_BANDS)
    image = GEOTIFF_RASTER_FORMAT.decode_raster(
        raster_dir, window.projection, window.bounds
    )
    dst_fname = get_modality_fname(
        olmoearth_path,
        Modality.WORLDCOVER_PRED,
        TimeSpan.STATIC,
        window_metadata,
        band_set.get_resolution(),
        "tif",
    )
    GEOTIFF_RASTER_FORMAT.encode_raster(
        path=dst_fname.parent,
        projection=window.projection,
        bounds=window.bounds,
        raster=image,
        fname=dst_fname.name,
    )
    # Record the window's actual time range so the predicted land cover is aligned with
    # the sample year (rather than a hardcoded map vintage).
    start_time, end_time = window.time_range
    metadata_fname = get_modality_temp_meta_fname(
        olmoearth_path, Modality.WORLDCOVER_PRED, TimeSpan.STATIC, window.name
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
    for window in dataset.load_windows(workers=args.workers, show_progress=True):
        jobs.append(
            dict(
                window=window,
                olmoearth_path=olmoearth_path,
            )
        )

    p = multiprocessing.Pool(args.workers)
    outputs = star_imap_unordered(p, convert_worldcover_pred, jobs)
    for _ in tqdm.tqdm(outputs, total=len(jobs)):
        pass
    p.close()
