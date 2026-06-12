"""Post-process ingested WorldCereal data into the OlmoEarth Pretrain dataset."""

import csv
from datetime import UTC, datetime

import numpy as np
from rslearn.dataset import Window
from rslearn.utils.raster_array import RasterArray
from upath import UPath

from olmoearth_pretrain.dataset.utils import get_modality_fname
from olmoearth_pretrain.modalities import Modality, TimeSpan

from ..constants import GEOTIFF_RASTER_FORMAT, METADATA_COLUMNS
from ..util import get_modality_temp_meta_fname, get_window_metadata
from .runner import run_window_converter

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


def convert_worldcereal(window: Window, olmoearth_path: UPath) -> None:
    """Add WorldCereal data for this window to the OlmoEarth Pretrain dataset.

    Args:
        window: the rslearn window to read data from.
        olmoearth_path: OlmoEarth Pretrain dataset path to write to.
    """
    ndarrays: list[np.ndarray | None] = []
    assert len(Modality.WORLDCEREAL.band_sets) == 1
    band_set = Modality.WORLDCEREAL.band_sets[0]
    window_metadata = get_window_metadata(window)
    for band in band_set.bands:
        if not window.is_layer_completed(band):
            ndarrays.append(None)
            continue
        window_dir = window.get_raster_dir(band, [band])

        ndarrays.append(
            GEOTIFF_RASTER_FORMAT.decode_raster(
                path=window_dir, projection=window.projection, bounds=window.bounds
            ).get_chw_array()
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
        olmoearth_path,
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
        raster=RasterArray(chw_array=concatenated_arrays),
        fname=dst_fname.name,
    )
    metadata_fname = get_modality_temp_meta_fname(
        olmoearth_path, Modality.WORLDCEREAL, TimeSpan.STATIC, window.name
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
    run_window_converter(convert_worldcereal)
