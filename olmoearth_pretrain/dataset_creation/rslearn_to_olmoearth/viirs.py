"""Add annual VIIRS night lights to the OlmoEarth Pretrain dataset.

Unlike the other converters this does not read an rslearn layer: it reads the global
annual composites written by olmoearth_pretrain.dataset_creation.viirs.export_viirs
directly, reprojecting onto each window. rslearn's LocalFiles items carry no
timestamp, so it could not pick the right year per window.
"""

import argparse
import csv
import multiprocessing
from datetime import UTC, datetime

import numpy as np
import rasterio
import rasterio.vrt
import tqdm
from rasterio.enums import Resampling
from rslearn.dataset import Dataset, Window
from rslearn.utils.mp import star_imap_unordered
from rslearn.utils.raster_array import RasterArray
from rslearn.utils.raster_format import get_transform_from_projection_and_bounds
from upath import UPath

from olmoearth_pretrain.data.constants import MISSING_VALUE, Modality, TimeSpan
from olmoearth_pretrain.dataset.utils import get_modality_fname

from ..constants import GEOTIFF_RASTER_FORMAT, METADATA_COLUMNS
from ..util import get_modality_temp_meta_fname, get_window_metadata
from .cli import add_common_arguments

RADIANCE_BAND = "average_masked"
COVERAGE_BAND = "cf_cvg"


def read_band(fname: UPath, window: Window) -> np.ndarray:
    """Read one global band reprojected onto the window, NaN outside the source."""
    bounds = window.bounds
    with rasterio.open(fname.path) as src:
        with rasterio.vrt.WarpedVRT(
            src,
            crs=window.projection.crs,
            transform=get_transform_from_projection_and_bounds(
                window.projection, bounds
            ),
            width=bounds[2] - bounds[0],
            height=bounds[3] - bounds[1],
            resampling=Resampling.nearest,
            dtype="float32",
            nodata=np.nan,
        ) as vrt:
            return vrt.read(1)


def convert_viirs(window: Window, olmoearth_path: UPath, viirs_dir: UPath) -> None:
    """Add VIIRS night lights for this window to the OlmoEarth Pretrain dataset.

    The composite is the calendar year containing the window start, which is also
    the year whose midpoint is closest to the centre of the window's Sentinel-2
    year (months are offset -180 to +180 days from the window start).

    The stored band is log1p of the annual masked average radiance (radiance is
    heavy-tailed and mostly zero). Pixels with no cloud-free observations that year
    are written as MISSING_VALUE.

    Args:
        window: the rslearn window to convert.
        olmoearth_path: OlmoEarth Pretrain dataset path to write to.
        viirs_dir: directory written by export_viirs.
    """
    window_metadata = get_window_metadata(window)

    assert window.time_range is not None
    year = window.time_range[0].year
    year_dir = viirs_dir / str(year)
    if not (year_dir / f"{COVERAGE_BAND}.tif").exists():
        return

    radiance = read_band(year_dir / f"{RADIANCE_BAND}.tif", window)
    coverage = read_band(year_dir / f"{COVERAGE_BAND}.tif", window)
    missing = np.isnan(radiance) | np.isnan(coverage) | (coverage <= 0)
    if missing.all():
        return
    image = np.where(missing, MISSING_VALUE, np.log1p(np.clip(radiance, 0, None)))

    assert len(Modality.VIIRS.band_sets) == 1
    band_set = Modality.VIIRS.band_sets[0]
    dst_fname = get_modality_fname(
        olmoearth_path,
        Modality.VIIRS,
        TimeSpan.STATIC,
        window_metadata,
        band_set.get_resolution(),
        "tif",
    )
    GEOTIFF_RASTER_FORMAT.encode_raster(
        path=dst_fname.parent,
        projection=window.projection,
        bounds=window.bounds,
        raster=RasterArray(chw_array=image[None].astype(np.float32)),
        fname=dst_fname.name,
    )
    metadata_fname = get_modality_temp_meta_fname(
        olmoearth_path, Modality.VIIRS, TimeSpan.STATIC, window.name
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
                start_time=datetime(year, 1, 1, tzinfo=UTC).isoformat(),
                end_time=datetime(year + 1, 1, 1, tzinfo=UTC).isoformat(),
            )
        )


if __name__ == "__main__":
    multiprocessing.set_start_method("forkserver")

    parser = argparse.ArgumentParser(
        description="Post-process OlmoEarth Pretrain data",
    )
    add_common_arguments(parser)
    parser.add_argument(
        "--viirs_dir",
        type=str,
        help="Directory of annual composites written by export_viirs",
        required=True,
    )
    args = parser.parse_args()

    dataset = Dataset(UPath(args.ds_path))
    olmoearth_path = UPath(args.olmoearth_path)
    viirs_dir = UPath(args.viirs_dir)

    jobs = []
    for window in dataset.load_windows(
        workers=args.workers, show_progress=True, groups=["res_10"]
    ):
        jobs.append(
            dict(
                window=window,
                olmoearth_path=olmoearth_path,
                viirs_dir=viirs_dir,
            )
        )

    p = multiprocessing.Pool(args.workers)
    outputs = star_imap_unordered(p, convert_viirs, jobs)
    for _ in tqdm.tqdm(outputs, total=len(jobs)):
        pass
    p.close()
