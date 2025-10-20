"""Post-process ingested ERA5_10 data into the OlmoEarth Pretrain dataset."""

import argparse
import csv
import logging
import multiprocessing
from datetime import datetime

import numpy as np
import numpy.typing as npt
import tqdm
from rslearn.data_sources import Item
from rslearn.dataset import Window
from rslearn.utils.mp import star_imap_unordered
from rslearn.utils.raster_format import GeotiffRasterFormat
from upath import UPath

from olmoearth_pretrain.data.constants import Modality, TimeSpan
from olmoearth_pretrain.dataset.utils import get_modality_fname

from ..constants import METADATA_COLUMNS
from ..util import get_modality_temp_meta_fname, get_window_metadata
from .multitemporal_raster import get_adjusted_projection_and_bounds

# Layer name in the input rslearn dataset.
LAYER_NAME = "era5_10"

logger = logging.getLogger(__name__)


def convert_era5(window_path: UPath, helios_path: UPath) -> None:
    """Add ERA5 data for this window to the OlmoEarth Pretrain dataset.

    Args:
        window_path: the rslearn window directory to read data from.
        helios_path: OlmoEarth Pretrain dataset path to write to.
    """
    modality = Modality.ERA5_10
    assert len(modality.band_sets) == 1
    band_set = modality.band_sets[0]

    window = Window.load(window_path)
    window_metadata = get_window_metadata(window)
    layer_datas = window.load_layer_datas()
    raster_format = GeotiffRasterFormat()

    logger.debug(f"processing window {window.name}")

    # Skip windows that are not prepared for ERA5_10.
    if LAYER_NAME not in layer_datas:
        logger.warning(
            f"skipping window {window.name} because it is not prepared for {LAYER_NAME}"
        )
        return

    # Read the images over time.
    # The items in this data source are based on the calendar month, so we use all of
    # the groups for the one-year monthly data but then also see which monthly group
    # contains the window's time range for the frequent data.
    year_images: list[npt.NDArray] = []
    year_time_ranges = []
    two_week_image: npt.NDArray | None = None
    two_week_time_range: tuple[datetime, datetime] | None = None
    for group_idx, group in enumerate(layer_datas[LAYER_NAME].serialized_item_groups):
        # Can be uncompleted due to errors since for some reason the API occasionally
        # just returns two bands instead of all the requested variables.
        is_completed = window.is_layer_completed(LAYER_NAME, group_idx)
        if not is_completed:
            continue

        # Use first item in the group to get the start time for this image.
        time_range = Item.deserialize(group[0]).geometry.time_range

        # Compute bounds of this raster adjusted for the resolution.
        adjusted_projection, adjusted_bounds = get_adjusted_projection_and_bounds(
            Modality.ERA5_10, band_set, window.projection, window.bounds
        )

        raster_dir = window.get_raster_dir(LAYER_NAME, band_set.bands, group_idx)
        image = raster_format.decode_raster(
            raster_dir, adjusted_projection, adjusted_bounds
        )

        year_images.append(image)
        year_time_ranges.append(time_range)

        # Should we use this image for the frequent data for this window?
        assert window.time_range is not None, "Window time range should not be None"
        assert time_range is not None, "Item time range should not be None"
        if (
            window.time_range[0] < time_range[1]
            and time_range[0] < window.time_range[1]
        ):
            two_week_image = image
            two_week_time_range = time_range

    if len(year_images) < 12:
        logger.warning(
            f"skipping window {window.name} because it only has {len(year_images)} images in {LAYER_NAME}"
        )
        return
    else:
        # In case there are more than 12 images, only use the first 12
        year_images = year_images[:12]
        year_time_ranges = year_time_ranges[:12]

    if two_week_image is None or two_week_time_range is None:
        logger.warning(
            f"skipping window {window.name} because it did not have an image intersecting the window time range"
        )
        return

    logger.warning(f"window {window.name} is good")

    # Save the one-year image and metadata.
    year_stacked_image = np.concatenate(year_images, axis=0)
    year_dst_fname = get_modality_fname(
        helios_path,
        Modality.ERA5_10,
        TimeSpan.YEAR,
        window_metadata,
        band_set.get_resolution(),
        "tif",
    )
    raster_format.encode_raster(
        path=year_dst_fname.parent,
        projection=adjusted_projection,
        bounds=adjusted_bounds,
        array=year_stacked_image,
        fname=year_dst_fname.name,
    )
    year_metadata_fname = get_modality_temp_meta_fname(
        helios_path, Modality.ERA5_10, TimeSpan.YEAR, window.name
    )
    year_metadata_fname.parent.mkdir(parents=True, exist_ok=True)
    with year_metadata_fname.open("w") as f:
        writer = csv.DictWriter(f, fieldnames=METADATA_COLUMNS)
        writer.writeheader()
        for group_idx, time_range in enumerate(year_time_ranges):
            writer.writerow(
                dict(
                    crs=window_metadata.crs,
                    col=window_metadata.col,
                    row=window_metadata.row,
                    tile_time=window_metadata.time.isoformat(),
                    image_idx=group_idx,
                    start_time=time_range[0].isoformat(),
                    end_time=time_range[1].isoformat(),
                )
            )

    # Save the two-week image and metadata.
    two_week_dst_fname = get_modality_fname(
        helios_path,
        Modality.ERA5_10,
        TimeSpan.TWO_WEEK,
        window_metadata,
        band_set.get_resolution(),
        "tif",
    )
    raster_format.encode_raster(
        path=two_week_dst_fname.parent,
        projection=adjusted_projection,
        bounds=adjusted_bounds,
        array=two_week_image,
        fname=two_week_dst_fname.name,
    )
    two_week_metadata_fname = get_modality_temp_meta_fname(
        helios_path, Modality.ERA5_10, TimeSpan.TWO_WEEK, window.name
    )
    two_week_metadata_fname.parent.mkdir(parents=True, exist_ok=True)
    with two_week_metadata_fname.open("w") as f:
        writer = csv.DictWriter(f, fieldnames=METADATA_COLUMNS)
        writer.writeheader()
        writer.writerow(
            dict(
                crs=window_metadata.crs,
                col=window_metadata.col,
                row=window_metadata.row,
                tile_time=window_metadata.time.isoformat(),
                image_idx="0",
                start_time=two_week_time_range[0].isoformat(),
                end_time=two_week_time_range[1].isoformat(),
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
        "--helios_path",
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
    outputs = star_imap_unordered(p, convert_era5, jobs)
    for _ in tqdm.tqdm(outputs, total=len(jobs)):
        pass
    p.close()
