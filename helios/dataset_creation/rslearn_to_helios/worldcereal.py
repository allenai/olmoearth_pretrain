"""Post-process ingested WorldCereal data into the Helios dataset."""

import argparse
import csv
import multiprocessing
from datetime import UTC, datetime

import numpy as np
import tqdm
from rslearn.dataset import Window
from rslearn.utils.mp import star_imap_unordered
from upath import UPath

from helios.data.constants import Modality, TimeSpan
from helios.dataset.utils import get_modality_fname

from ..constants import GEOTIFF_RASTER_FORMAT, METADATA_COLUMNS
from ..util import get_modality_temp_meta_fname, get_window_metadata

START_TIME = datetime(2021, 1, 1, tzinfo=UTC)
END_TIME = datetime(2022, 1, 1, tzinfo=UTC)


def _fill_nones_with_zeros(ndarrays: list[np.ndarray | None]) -> np.ndarray | None:
    filler = None
    for x in ndarrays:
        if x is not None:
            filler = np.zeros_like(x)
            break
    if filler is None:
        return None

    return_list = []
    for x in ndarrays:
        if x is not None:
            return_list.append(x)
        else:
            return_list.append(filler.copy())
    return np.concatenate(return_list, axis=0)


def convert_worldcereal(window_path: UPath, helios_path: UPath) -> None:
    """Add WorldCereal data for this window to the Helios dataset.

    Args:
        window_path: the rslearn window directory to read data from.
        helios_path: Helios dataset path to write to.
    """
    ndarrays: list[np.ndarray | None] = []
    assert len(Modality.WORLDCEREAL.band_sets) == 1
    band_set = Modality.WORLDCEREAL.band_sets[0]
    window = Window.load(window_path)
    window_metadata = get_window_metadata(window)
    for band in band_set.bands:
        if not window.is_layer_completed(band):
            ndarrays.append(None)
            continue
        window_dir = window.get_raster_dir(band, [band])

        ndarrays.append(
            GEOTIFF_RASTER_FORMAT.decode_raster(
                path=window_dir, projection=window.projection, bounds=window.bounds
            )
        )

    assert len(ndarrays) == len(band_set.bands), (
        f"Expected {len(band_set.bands)} arrays, got {len(ndarrays)}"
    )
    concatenated_arrays = _fill_nones_with_zeros(ndarrays)

    if concatenated_arrays is None:
        return None

    # 255 = missing data, which we will treat as 0s
    # 254 = not cropland. This only occurs in crop type products
    # in addition, because of our resampling we rarely get
    # other values (e.g. 252). Lets set them all to 0
    concatenated_arrays[concatenated_arrays > 100] = 0
    assert concatenated_arrays.min() >= 0
    assert concatenated_arrays.max() <= 100

    dst_fname = get_modality_fname(
        helios_path,
        Modality.WORLDCEREAL,
        TimeSpan.STATIC,
        window_metadata,
        band_set.get_resolution(),
        "tif",
    )
    GEOTIFF_RASTER_FORMAT.encode_raster(
        path=dst_fname.parent,
        projection=window.projection,
        bounds=window.bounds,
        array=concatenated_arrays,
        fname=dst_fname.name,
    )
    metadata_fname = get_modality_temp_meta_fname(
        helios_path, Modality.WORLDCEREAL, TimeSpan.STATIC, window.name
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
    outputs = star_imap_unordered(p, convert_worldcereal, jobs)
    for _ in tqdm.tqdm(outputs, total=len(jobs)):
        pass
    p.close()
