"""Post-process ingested ERA5-Land daily data into the OlmoEarth Pretrain dataset.

Reads ~720-day daily ERA5-Land data from rslearn's NumpyRasterFormat (C=14, T, H=1, W=1),
converts to an intermediate GeoTIFF in T-major band ordering (matching what the H5
pipeline expects), and writes a companion metadata CSV with per-timestep time ranges.
"""

import argparse
import logging
import multiprocessing
import random

import tqdm
from rslearn.dataset import Dataset, Window
from rslearn.utils.fsspec import open_rasterio_upath_reader
from rslearn.utils.geometry import PixelBounds, Projection
from rslearn.utils.mp import star_imap_unordered
from rslearn.utils.raster_array import RasterArray
from rslearn.utils.raster_format import (
    GeotiffRasterFormat,
    NumpyRasterFormat,
    get_raster_projection_and_bounds,
)
from upath import UPath

from olmoearth_pretrain.data.constants import Modality, TimeSpan
from olmoearth_pretrain.dataset.utils import get_modality_fname

from ..util import (
    get_metadata_row,
    get_modality_temp_meta_fname,
    get_window_metadata,
    write_metadata_rows,
)

LAYER_NAME = "era5_365dhistory"

logger = logging.getLogger(__name__)


def _decode_era5_raster(
    raster_dir: UPath,
    projection: Projection,
    bounds: PixelBounds,
) -> RasterArray:
    """Read an ERA5 raster materialized as either NumPy or GeoTIFF."""
    numpy_format = NumpyRasterFormat()
    if (raster_dir / numpy_format.data_fname).exists():
        return numpy_format.decode_raster(
            raster_dir,
            projection,
            bounds,
            expect_bounds_mismatch=True,
        )

    geotiff_format = GeotiffRasterFormat()
    if (raster_dir / geotiff_format.fname).exists():
        with open_rasterio_upath_reader(raster_dir / geotiff_format.fname) as src:
            source_projection, source_bounds = get_raster_projection_and_bounds(src)
        return geotiff_format.decode_raster(
            raster_dir,
            source_projection,
            source_bounds,
        )

    raise FileNotFoundError(f"no NumPy or GeoTIFF raster found in {raster_dir}")


def _as_era5_cthw(raster_array: RasterArray, num_bands: int) -> RasterArray:
    """Normalize ERA5 arrays to (C, T, H, W)."""
    if raster_array.array.shape[0] == num_bands:
        return raster_array

    if (
        raster_array.array.shape[1] == 1
        and raster_array.array.shape[0] % num_bands == 0
    ):
        num_timesteps = raster_array.array.shape[0] // num_bands
        array = raster_array.array.reshape(
            num_bands,
            num_timesteps,
            *raster_array.array.shape[2:],
        )
        return RasterArray(
            array=array,
            timestamps=raster_array.timestamps,
            metadata=raster_array.metadata,
        )

    return raster_array


def convert_era5l_day(
    window: Window,
    olmoearth_path: UPath,
    time_span: TimeSpan = TimeSpan.HIGH_FREQ,
) -> None:
    """Convert ERA5-Land daily data for one window to OlmoEarth Pretrain format.

    Args:
        window: the rslearn window to read data from.
        olmoearth_path: OlmoEarth Pretrain dataset path to write to.
        time_span: OlmoEarth time span to write to.
    """
    modality = Modality.ERA5L_DAY_10
    assert len(modality.band_sets) == 1
    band_set = modality.band_sets[0]

    window_metadata = get_window_metadata(window)

    # Skip windows whose output already exists (enables resume and multi-machine runs).
    dst_fname = get_modality_fname(
        olmoearth_path,
        modality,
        time_span,
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

    raster_dir = window.get_raster_dir(LAYER_NAME, band_set.bands, group_idx)
    raster_array = _as_era5_cthw(
        _decode_era5_raster(raster_dir, window.projection, window.bounds),
        len(band_set.bands),
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
    output_bounds = (
        window.bounds[0],
        window.bounds[1],
        window.bounds[0] + flat_chw.shape[2],
        window.bounds[1] + flat_chw.shape[1],
    )
    geotiff_format.encode_raster(
        path=dst_fname.parent,
        projection=window.projection,
        bounds=output_bounds,
        raster=geotiff_raster,
        fname=dst_fname.name,
    )

    # Write the companion metadata CSV (one row per timestep).
    metadata_fname = get_modality_temp_meta_fname(
        olmoearth_path, modality, time_span, window.name
    )
    write_metadata_rows(
        metadata_fname,
        [
            get_metadata_row(window_metadata, idx, start_time, end_time)
            for idx, (start_time, end_time) in enumerate(timestamps)
        ],
    )


if __name__ == "__main__":
    multiprocessing.set_start_method("forkserver")

    parser = argparse.ArgumentParser(
        description="Convert ERA5-Land daily data to OlmoEarth Pretrain format",
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
    parser.add_argument(
        "--group",
        type=str,
        nargs="+",
        help="rslearn window group(s) to convert",
        default=["res_10"],
    )
    parser.add_argument(
        "--time_span",
        type=str,
        help="OlmoEarth time span to write",
        default=TimeSpan.HIGH_FREQ.value,
    )
    args = parser.parse_args()

    dataset = Dataset(UPath(args.ds_path))
    olmoearth_path = UPath(args.olmoearth_path)

    jobs = []
    for window in dataset.load_windows(
        workers=args.workers, show_progress=True, groups=args.group
    ):
        jobs.append(
            dict(
                window=window,
                olmoearth_path=olmoearth_path,
                time_span=TimeSpan(args.time_span),
            )
        )

    random.shuffle(jobs)

    p = multiprocessing.Pool(args.workers)
    outputs = star_imap_unordered(p, convert_era5l_day, jobs)
    for _ in tqdm.tqdm(outputs, total=len(jobs)):
        pass
    p.close()
