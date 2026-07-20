"""Post-process ingested ERA5 data into the OlmoEarth Pretrain dataset."""

import argparse
import csv
import logging
import multiprocessing

import numpy as np
import numpy.typing as npt
import tqdm
from rslearn.data_sources import Item
from rslearn.dataset import Dataset, Window
from rslearn.utils.mp import star_imap_unordered
from rslearn.utils.raster_array import RasterArray
from rslearn.utils.raster_format import GeotiffRasterFormat
from upath import UPath

from olmoearth_pretrain.data.constants import Modality, TimeSpan
from olmoearth_pretrain.dataset.utils import get_modality_fname

from ..constants import METADATA_COLUMNS
from ..util import get_modality_temp_meta_fname, get_window_metadata
from .cli import add_common_arguments

# Layer name in the input rslearn dataset.
LAYER_NAME = "era5"

logger = logging.getLogger(__name__)


def convert_era5(window: Window, olmoearth_path: UPath) -> None:
    """Add ERA5 data for this window to the OlmoEarth Pretrain dataset.

    Args:
        window: the rslearn window to read data from.
        olmoearth_path: OlmoEarth Pretrain dataset path to write to.
    """
    modality = Modality.ERA5
    assert len(modality.band_sets) == 1
    band_set = modality.band_sets[0]

    window_metadata = get_window_metadata(window)
    layer_datas = window.load_layer_datas()
    raster_format = GeotiffRasterFormat()

    logger.debug(f"processing window {window.name}")

    # Skip windows that are not prepared for ERA5.
    if LAYER_NAME not in layer_datas:
        logger.warning(
            f"skipping window {window.name} because it is not prepared for {LAYER_NAME}"
        )
        return

    # Read the images over time.
    # The items in this data source are based on the calendar month, so we use all of
    # the groups for the one-year monthly data.
    year_images: list[npt.NDArray] = []
    year_time_ranges = []
    for group_idx, group in enumerate(layer_datas[LAYER_NAME].serialized_item_groups):
        # Can be uncompleted due to errors since for some reason the API occasionally
        # just returns two bands instead of all the requested variables.
        is_completed = window.is_layer_completed(LAYER_NAME, group_idx)
        if not is_completed:
            continue

        # Use first item in the group to get the start time for this image.
        time_range = Item.deserialize(group[0]).geometry.time_range

        image = window.data.read_raster(
            LAYER_NAME, band_set.bands, raster_format, group_idx=group_idx
        ).get_chw_array()

        year_images.append(image)
        year_time_ranges.append(time_range)

    if len(year_images) < 12:
        logger.warning(
            f"skipping window {window.name} because it only has {len(year_images)} images in {LAYER_NAME}"
        )
        return
    else:
        # In case there are more than 12 images, only use the first 12
        year_images = year_images[:12]
        year_time_ranges = year_time_ranges[:12]

    logger.warning(f"window {window.name} is good")

    # Save the one-year image and metadata.
    year_stacked_image = np.concatenate(year_images, axis=0)
    year_dst_fname = get_modality_fname(
        olmoearth_path,
        Modality.ERA5,
        TimeSpan.YEAR,
        window_metadata,
        band_set.get_resolution(),
        "tif",
    )
    raster_format.encode_raster(
        path=year_dst_fname.parent,
        projection=window.projection,
        bounds=window.bounds,
        raster=RasterArray(chw_array=year_stacked_image),
        fname=year_dst_fname.name,
    )
    year_metadata_fname = get_modality_temp_meta_fname(
        olmoearth_path, Modality.ERA5, TimeSpan.YEAR, window.name
    )
    year_metadata_fname.parent.mkdir(parents=True, exist_ok=True)
    with year_metadata_fname.open("w") as f:
        writer = csv.DictWriter(f, fieldnames=METADATA_COLUMNS)
        writer.writeheader()
        for group_idx, time_range in enumerate(year_time_ranges):
            writer.writerow(
                dict(
                    example_id=window_metadata.example_id or "",
                    crs=window_metadata.crs,
                    col=window_metadata.col,
                    row=window_metadata.row,
                    tile_time=window_metadata.time.isoformat(),
                    image_idx=group_idx,
                    start_time=time_range[0].isoformat(),
                    end_time=time_range[1].isoformat(),
                )
            )


if __name__ == "__main__":
    multiprocessing.set_start_method("forkserver")

    parser = argparse.ArgumentParser(
        description="Post-process OlmoEarth Pretrain data",
    )
    add_common_arguments(parser, default_groups=["res_160"])
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

    p = multiprocessing.Pool(args.workers)
    outputs = star_imap_unordered(p, convert_era5, jobs)
    for _ in tqdm.tqdm(outputs, total=len(jobs)):
        pass
    p.close()
