"""Post-process ingested ERA5-Land daily data into the OlmoEarth Pretrain dataset.

Reads ~720-day daily ERA5-Land data from rslearn's NumpyRasterFormat (C=14, T, H=1, W=1),
converts to an intermediate GeoTIFF in T-major band ordering (matching what the H5
pipeline expects), and writes a companion metadata CSV with per-timestep time ranges.
"""

import argparse
import csv
import logging
import multiprocessing
import random

import tqdm
from rslearn.dataset import Dataset, Window
from rslearn.utils.mp import star_imap_unordered
from rslearn.utils.raster_array import RasterArray
from rslearn.utils.raster_format import GeotiffRasterFormat, NumpyRasterFormat
from upath import UPath

from olmoearth_pretrain.data.constants import Modality, TimeSpan
from olmoearth_pretrain.dataset.utils import get_modality_fname

from ..constants import METADATA_COLUMNS
from ..util import get_modality_temp_meta_fname, get_window_metadata
from .cli import add_common_arguments
from .multitemporal_raster import get_adjusted_projection_and_bounds

LAYER_NAME = "era5_365dhistory"

logger = logging.getLogger(__name__)


def convert_era5l_day(window: Window, olmoearth_path: UPath) -> None:
    """Convert ERA5-Land daily data for one window to OlmoEarth Pretrain format.

    Args:
        window: the rslearn window to read data from.
        olmoearth_path: OlmoEarth Pretrain dataset path to write to.
    """
    modality = Modality.ERA5L_DAY_10
    assert len(modality.band_sets) == 1
    band_set = modality.band_sets[0]

    window_metadata = get_window_metadata(window)

    # Skip windows whose output already exists (enables resume and multi-machine runs).
    dst_fname = get_modality_fname(
        olmoearth_path,
        modality,
        TimeSpan.YEAR,
        window_metadata,
        band_set.get_resolution(),
        "tif",
    )
    if dst_fname.exists():
        logger.debug(
            f"skipping window {window.name}: output already exists at {dst_fname}"
        )
        return

    layer_datas = window.load_layer_datas()

    logger.debug(f"processing window {window.name}")

    if LAYER_NAME not in layer_datas:
        logger.warning(
            f"skipping window {window.name} because it is not prepared for {LAYER_NAME}"
        )
        return

    # ERA5-Land daily uses a single group (max_matches=1, SPATIAL_MOSAIC_TEMPORAL_STACK)
    # containing all ~720 daily timesteps.
    group_idx = 0
    if not window.is_layer_completed(LAYER_NAME, group_idx):
        logger.warning(
            f"skipping window {window.name} because {LAYER_NAME} group {group_idx} is not completed"
        )
        return

    adjusted_projection, adjusted_bounds = get_adjusted_projection_and_bounds(
        modality, band_set, window.projection, window.bounds
    )

    numpy_format = NumpyRasterFormat()
    raster_array = window.data.read_raster(
        LAYER_NAME,
        band_set.bands,
        numpy_format,
        projection=adjusted_projection,
        bounds=adjusted_bounds,
        group_idx=group_idx,
    )
    # raster_array.array shape: (C=14, T=~720, H=1, W=1)

    num_channels = raster_array.array.shape[0]
    num_timesteps = raster_array.array.shape[1]
    if num_channels != len(band_set.bands):
        logger.warning(
            f"skipping window {window.name}: expected {len(band_set.bands)} channels, "
            f"got {num_channels}"
        )
        return

    if num_timesteps == 0:
        logger.warning(f"skipping window {window.name}: zero timesteps")
        return

    timestamps = raster_array.timestamps
    if timestamps is None or len(timestamps) != num_timesteps:
        logger.warning(
            f"skipping window {window.name}: timestamps missing or mismatched "
            f"(expected {num_timesteps}, got {len(timestamps) if timestamps else 'None'})"
        )
        return

    logger.info(
        f"window {window.name}: {num_channels} channels, {num_timesteps} timesteps"
    )

    # Reorder from (C, T, H, W) to T-major (T*C, H, W) for the GeoTIFF.
    # sample.py reads (num_bands, H, W) and does .reshape(-1, len(band_set.bands))
    # to get (T, C), so bands must be laid out as [t0_c0, t0_c1, ..., t1_c0, ...].
    cthw = raster_array.array
    tchw = cthw.transpose(1, 0, 2, 3)
    flat_chw = tchw.reshape(-1, *cthw.shape[2:])
    geotiff_raster = RasterArray(chw_array=flat_chw)

    geotiff_format = GeotiffRasterFormat()
    geotiff_format.encode_raster(
        path=dst_fname.parent,
        projection=adjusted_projection,
        bounds=adjusted_bounds,
        raster=geotiff_raster,
        fname=dst_fname.name,
    )

    # Write the companion metadata CSV (one row per timestep).
    metadata_fname = get_modality_temp_meta_fname(
        olmoearth_path, modality, TimeSpan.YEAR, window.name
    )
    metadata_fname.parent.mkdir(parents=True, exist_ok=True)
    with metadata_fname.open("w") as f:
        writer = csv.DictWriter(f, fieldnames=METADATA_COLUMNS)
        writer.writeheader()
        for idx, ts in enumerate(timestamps):
            writer.writerow(
                dict(
                    crs=window_metadata.crs,
                    col=window_metadata.col,
                    row=window_metadata.row,
                    tile_time=window_metadata.time.isoformat(),
                    image_idx=idx,
                    start_time=ts[0].isoformat(),
                    end_time=ts[1].isoformat(),
                )
            )


if __name__ == "__main__":
    multiprocessing.set_start_method("forkserver")

    parser = argparse.ArgumentParser(
        description="Convert ERA5-Land daily data to OlmoEarth Pretrain format",
    )
    add_common_arguments(parser, default_groups=["res_10"])
    args = parser.parse_args()

    dataset = Dataset(UPath(args.ds_path))
    olmoearth_path = UPath(args.olmoearth_path)

    jobs = []
    for window in dataset.load_windows(
        workers=args.workers, show_progress=True, groups=args.groups
    ):
        jobs.append(
            dict(
                window=window,
                olmoearth_path=olmoearth_path,
            )
        )

    random.shuffle(jobs)

    p = multiprocessing.Pool(args.workers)
    outputs = star_imap_unordered(p, convert_era5l_day, jobs)
    for _ in tqdm.tqdm(outputs, total=len(jobs)):
        pass
    p.close()
