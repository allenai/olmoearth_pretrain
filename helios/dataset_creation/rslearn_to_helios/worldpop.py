"""Post-process ingested WorldPop data into the Helios dataset."""

import argparse
import csv
import multiprocessing
from datetime import UTC, datetime

import tqdm
from rslearn.dataset import Window
from rslearn.utils.mp import star_imap_unordered
from upath import UPath

from helios.data.constants import Modality, TimeSpan
from helios.dataset.utils import get_modality_fname

from ..constants import GEOTIFF_RASTER_FORMAT, METADATA_COLUMNS
from ..util import get_modality_temp_meta_fname, get_window_metadata

START_TIME = datetime(2020, 1, 1, tzinfo=UTC)
END_TIME = datetime(2021, 1, 1, tzinfo=UTC)

# Layer name in the input rslearn dataset.
LAYER_NAME = "worldpop"


def convert_worldpop(window_path: UPath, helios_path: UPath) -> None:
    """Add WorldPop data for this window to the Helios dataset.

    Args:
        window_path: the rslearn window directory to read data from.
        helios_path: Helios dataset path to write to.
    """
    window = Window.load(window_path)
    window_metadata = get_window_metadata(window)

    if not window.is_layer_completed(LAYER_NAME):
        return

    assert len(Modality.WORLDPOP.band_sets) == 1
    band_set = Modality.WORLDPOP.band_sets[0]
    raster_dir = window.get_raster_dir(LAYER_NAME, band_set.bands)
    image = GEOTIFF_RASTER_FORMAT.decode_raster(
        raster_dir, window.projection, window.bounds
    )

    # Clip population count to 0. NODATA is -99999 and includes locations that are
    # mapped as "unsettled" but really that is 0 population.
    image[image < 0] = 0
    # Skip areas that are fully nodata since it is more likely to be wrong and also is
    # a less useful target.
    if image.max() == 0:
        return

    dst_fname = get_modality_fname(
        helios_path,
        Modality.WORLDPOP,
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
        helios_path, Modality.WORLDPOP, TimeSpan.STATIC, window.name
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
    outputs = star_imap_unordered(p, convert_worldpop, jobs)
    for _ in tqdm.tqdm(outputs, total=len(jobs)):
        pass
    p.close()
