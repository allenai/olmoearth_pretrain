"""Helper functions to convert multitemporal rasters into Helios dataset."""

import csv
from datetime import timedelta

import numpy as np
from rslearn.data_sources import Item
from rslearn.dataset import Window
from rslearn.utils.raster_format import GeotiffRasterFormat
from upath import UPath

from ..const import METADATA_COLUMNS
from ..util import get_modality_fname, get_modality_temp_meta_fname


def convert_freq(
    window_path: UPath,
    helios_path: UPath,
    layer_name: str,
    bands: list[str],
    modality_name: str,
) -> None:
    """Add frequent (two-week) data from this window to the Helios dataset.

    Args:
        window_path: the rslearn window directory to read data from.
        helios_path: Helios dataset path to write to.
        layer_name: the name of the layer containing frequent data in the rslearn
            dataset. It should be configured to individually store each item from the
            two-week period that spatially intersects with the window, i.e.
            space_mode=intersects, max_matches=9999.
        bands: the band names.
        modality_name: the name of the modality in the output Helios dataset.
    """
    window = Window.load(window_path)
    layer_datas = window.load_layer_datas()
    raster_format = GeotiffRasterFormat()

    # We read the individual images and their timestamps, then write the stacked
    # images and CSV.
    images = []
    timestamps = []
    for group_idx, group in enumerate(layer_datas[layer_name].serialized_item_groups):
        if len(group) != 1:
            raise ValueError(
                f"expected Landsat groups to have length 1 but got {len(group)}"
            )
        item = Item.deserialize(group[0])
        timestamp = item.geometry.time_range[0]
        raster_dir = window.get_raster_dir(layer_name, bands, group_idx)
        image = raster_format.decode_raster(raster_dir, window.bounds)

        # Sometimes the image is blank because the window actually does not intersect
        # the raster. This is due to raster geometry information being too coarse in
        # some data sources. Here we skip those rasters so they don't get included with
        # this example in the Helios dataset.
        if image.max() == 0:
            continue

        images.append(image)
        timestamps.append(timestamp.isoformat())

    if len(images) > 0:
        stacked_image = np.concatenate(images, axis=0)
        dst_fname = get_modality_fname(helios_path, modality_name, window.name, "tif")
        raster_format.encode_raster(
            path=dst_fname.parent,
            projection=window.projection,
            bounds=window.bounds,
            array=stacked_image,
            fname=dst_fname.name,
        )
        metadata_fname = get_modality_temp_meta_fname(
            helios_path, modality_name, window.name
        )
        with metadata_fname.open("w") as f:
            writer = csv.DictWriter(f, fieldnames=METADATA_COLUMNS)
            writer.writeheader()
            for group_idx, timestamp in enumerate(timestamps):
                writer.writerow(
                    dict(
                        example_id=window.name,
                        image_idx=group_idx,
                        start_time=timestamp,
                        end_time=timestamp,
                    )
                )


def convert_monthly(
    window_path: UPath,
    helios_path: UPath,
    layer_prefix: str,
    bands: list[str],
    modality_name: str,
) -> None:
    """Add monthly (one-year) data from this window to the Helios dataset.

    Args:
        window_path: the rslearn window directory to read data from.
        helios_path: Helios dataset path to write to.
        layer_prefix: the prefix for the layer names containing monthly data in the
            rslearn dataset. The layers should be named with suffixes "_mo01", "_mo02",
            ..., "_mo12", where each layer contains a single mosaic for that month.
        bands: the band names.
        modality_name: the name of the modality in the output Helios dataset.
    """
    window = Window.load(window_path)
    raster_format = GeotiffRasterFormat()

    # The monthly images are stored in different layers, so we read one image per
    # layer. Then we reconstruct the time range to match the dataset configuration. And
    # finally stack the images and write them along with CSV.
    images = []
    time_ranges = []
    for month_idx in range(1, 13):
        layer_name = f"{layer_prefix}_mo{month_idx:02d}"
        start_time = window.time_range[0] + timedelta(days=(month_idx - 7) * 30)
        end_time = start_time + timedelta(days=30)
        raster_dir = window.get_raster_dir(layer_name, bands)
        if not raster_dir.exists():
            continue
        image = raster_format.decode_raster(raster_dir, window.bounds)
        images.append(image)
        time_ranges.append((start_time.isoformat(), end_time.isoformat()))

    if len(images) > 0:
        stacked_image = np.concatenate(images, axis=0)
        dst_fname = get_modality_fname(helios_path, modality_name, window.name, "tif")
        raster_format.encode_raster(
            path=dst_fname.parent,
            projection=window.projection,
            bounds=window.bounds,
            array=stacked_image,
            fname=dst_fname.name,
        )
        metadata_fname = get_modality_temp_meta_fname(
            helios_path, modality_name, window.name
        )
        with metadata_fname.open("w") as f:
            writer = csv.DictWriter(f, fieldnames=METADATA_COLUMNS)
            writer.writeheader()
            for image_idx, (start_time, end_time) in enumerate(time_ranges):
                writer.writerow(
                    dict(
                        example_id=window.name,
                        image_idx=image_idx,
                        start_time=start_time,
                        end_time=end_time,
                    )
                )
