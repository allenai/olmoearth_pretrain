"""Helper functions to convert multitemporal rasters into Helios dataset."""

import csv
from datetime import timedelta
from typing import Any

import numpy as np
import numpy.typing as npt
from rslearn.data_sources import Item
from rslearn.dataset import Window
from upath import UPath

from ..constants import GEOTIFF_RASTER_FORMAT, METADATA_COLUMNS
from ..util import get_modality_fname, get_modality_temp_meta_fname, parse_window_name

PIXELS_PER_TILE = 256
EPSILON = 1e-6


class BandSet:
    """A band set to store together in the Helios dataset.

    This should correspond to a set of bands stored together in the rslearn dataset, as
    defined by the per-modality rslearn dataset configuration file.

    It just specifies the band names, along with the resolution at which they should be
    stored.
    """

    def __init__(self, bands: list[str], resolution: float):
        """Create a new BandSet.

        Args:
            bands: the list of band names for this set.
            resolution: the resolution at which these bands should be stored.
        """
        self.bands = bands
        self.resolution = resolution

    def get_factor(self, window_resolution: float) -> int:
        """Get how much lower in resolution these bands are than the window.

        The window's resolution should be based on the highest resolution band(s) for
        each modality.

        Args:
            window_resolution: the resolution of the window.

        Returns:
            the factor by which these bands are lower, which should be a power of 2.
        """
        return round(self.resolution / window_resolution)

    def get_expected_image_size(self, window_resolution: float) -> int:
        """Get the expected size of images containing these bands.

        Args:
            window_resolution: the resolution of the window.

        Returns:
            the expected image size.
        """
        return PIXELS_PER_TILE // self.get_factor(window_resolution)

    def __hash__(self) -> int:
        """Compute hash."""
        return hash(("_".join(self.bands), self.resolution))

    def __eq__(self, other: Any) -> bool:
        """Returns whether this BandSet is the same as other."""
        if not isinstance(other, BandSet):
            return False
        return (
            self.bands == other.bands
            and abs(self.resolution - other.resolution) < EPSILON
        )


def convert_freq(
    window_path: UPath,
    helios_path: UPath,
    layer_name: str,
    modality_name: str,
    band_sets: list[BandSet],
    missing_okay: bool = False,
) -> None:
    """Add frequent (two-week) data from this window to the Helios dataset.

    Args:
        window_path: the rslearn window directory to read data from.
        helios_path: Helios dataset path to write to.
        layer_name: the name of the layer containing frequent data in the rslearn
            dataset. It should be configured to individually store each item from the
            two-week period that spatially intersects with the window, i.e.
            space_mode=intersects, max_matches=9999.
        modality_name: the name of the modality in the output Helios dataset.
        band_sets: the band sets.
        missing_okay: whether it is okay if some images that appear in items.json are
            missing. This should only be enabled if there are unresolvable errors
            during ingestion.
    """
    window = Window.load(window_path)
    window_metadata = parse_window_name(window.name)
    layer_datas = window.load_layer_datas()

    # We read the individual images and their timestamps, then write the stacked
    # images and CSV.
    # Map from band set to the images for that band set.
    images: dict[BandSet, list[npt.NDArray]] = {band_set: [] for band_set in band_sets}
    timestamps = []
    for group_idx, group in enumerate(layer_datas[layer_name].serialized_item_groups):
        if len(group) != 1:
            raise ValueError(
                f"expected Landsat groups to have length 1 but got {len(group)}"
            )

        item = Item.deserialize(group[0])
        timestamp = item.geometry.time_range[0]
        cur_images: dict[BandSet, npt.NDArray] = {}

        for band_set in band_sets:
            # Compute bounds of this raster adjusted for the resolution.
            factor = band_set.get_factor(window_metadata.resolution)
            adjusted_bounds = (
                window.bounds[0] // factor,
                window.bounds[1] // factor,
                window.bounds[2] // factor,
                window.bounds[3] // factor,
            )

            is_completed = window.is_layer_completed(layer_name, group_idx)
            # If missing images are okay, we ignore the uncompleted layer here.
            # Otherwise we will get an error when we try to read the GeoTIFF.
            if not is_completed and missing_okay:
                continue

            raster_dir = window.get_raster_dir(layer_name, band_set.bands, group_idx)
            image = GEOTIFF_RASTER_FORMAT.decode_raster(raster_dir, adjusted_bounds)
            print(
                f"bounds={window.bounds} adjusted={adjusted_bounds} source={raster_dir} max={image.max()}"
            )
            expected_image_size = band_set.get_expected_image_size(
                window_metadata.resolution
            )
            if (
                image.shape[1] != expected_image_size
                or image.shape[2] != expected_image_size
            ):
                raise ValueError(
                    f"expected image size {expected_image_size} but got {image.shape}"
                )

            cur_images[band_set] = image

        if len(cur_images) < len(band_sets):
            continue

        # Sometimes the images are blank because the window actually does not intersect
        # the raster. This is due to raster geometry information being too coarse in
        # some data sources. Here we skip those rasters so they don't get included with
        # this example in the Helios dataset.
        all_images_blank = all(image.max() == 0 for image in cur_images.values())
        if all_images_blank:
            continue

        timestamps.append(timestamp.isoformat())
        for band_set, image in cur_images.items():
            images[band_set].append(image)

    if len(timestamps) > 0:
        for band_set, band_set_images in images.items():
            stacked_image = np.concatenate(band_set_images, axis=0)
            dst_fname = get_modality_fname(
                helios_path, modality_name, window_metadata, band_set.resolution, "tif"
            )
            GEOTIFF_RASTER_FORMAT.encode_raster(
                path=dst_fname.parent,
                projection=window.projection,
                bounds=window.bounds,
                array=stacked_image,
                fname=dst_fname.name,
            )

        metadata_fname = get_modality_temp_meta_fname(
            helios_path, modality_name, window.name
        )
        metadata_fname.parent.mkdir(parents=True, exist_ok=True)
        with metadata_fname.open("w") as f:
            writer = csv.DictWriter(f, fieldnames=METADATA_COLUMNS)
            writer.writeheader()
            for group_idx, timestamp in enumerate(timestamps):
                writer.writerow(
                    dict(
                        crs=window_metadata.crs,
                        col=window_metadata.col,
                        row=window_metadata.row,
                        tile_time=window_metadata.time.isoformat(),
                        image_idx=group_idx,
                        start_time=timestamp,
                        end_time=timestamp,
                    )
                )


def convert_monthly(
    window_path: UPath,
    helios_path: UPath,
    layer_prefix: str,
    modality_name: str,
    band_sets: list[BandSet],
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
        band_sets: the band sets.
    """
    window = Window.load(window_path)
    window_metadata = parse_window_name(window.name)

    # The monthly images are stored in different layers, so we read one image per
    # layer. Then we reconstruct the time range to match the dataset configuration. And
    # finally stack the images and write them along with CSV.
    # Map from band set to list of images for that band set.
    images: dict[BandSet, list[npt.NDArray]] = {band_set: [] for band_set in band_sets}
    time_ranges = []
    for month_idx in range(1, 13):
        layer_name = f"{layer_prefix}_mo{month_idx:02d}"
        start_time = window.time_range[0] + timedelta(days=(month_idx - 7) * 30)
        end_time = start_time + timedelta(days=30)

        cur_images: dict[BandSet, npt.NDArray] = {}

        for band_set in band_sets:
            # Compute bounds of this raster adjusted for the resolution.
            factor = band_set.get_factor(window_metadata.resolution)
            adjusted_bounds = (
                window.bounds[0] // factor,
                window.bounds[1] // factor,
                window.bounds[2] // factor,
                window.bounds[3] // factor,
            )

            raster_dir = window.get_raster_dir(layer_name, band_set.bands)

            # Rasters may be missing for some months if there is no suitable data
            # during that month. So if any band is missing we exit and don't use that
            # month at this window.
            if not raster_dir.exists():
                break

            image = GEOTIFF_RASTER_FORMAT.decode_raster(raster_dir, adjusted_bounds)
            expected_image_size = band_set.get_expected_image_size(
                window_metadata.resolution
            )
            if (
                image.shape[1] != expected_image_size
                or image.shape[2] != expected_image_size
            ):
                raise ValueError(
                    f"expected image size {expected_image_size} but got {image.shape}"
                )

            cur_images[band_set] = image

        if len(cur_images) < len(band_sets):
            continue

        for band_set, image in cur_images.items():
            images[band_set].append(image)
        time_ranges.append((start_time.isoformat(), end_time.isoformat()))

    if len(images) > 0:
        for band_set, band_set_images in images.items():
            stacked_image = np.concatenate(band_set_images, axis=0)
            dst_fname = get_modality_fname(
                helios_path, modality_name, window_metadata, band_set.resolution, "tif"
            )
            GEOTIFF_RASTER_FORMAT.encode_raster(
                path=dst_fname.parent,
                projection=window.projection,
                bounds=window.bounds,
                array=stacked_image,
                fname=dst_fname.name,
            )

        metadata_fname = get_modality_temp_meta_fname(
            helios_path, modality_name, window.name
        )
        metadata_fname.parent.mkdir(parents=True, exist_ok=True)
        with metadata_fname.open("w") as f:
            writer = csv.DictWriter(f, fieldnames=METADATA_COLUMNS)
            writer.writeheader()
            for image_idx, (start_time, end_time) in enumerate(time_ranges):
                writer.writerow(
                    dict(
                        crs=window_metadata.crs,
                        col=window_metadata.col,
                        row=window_metadata.row,
                        tile_time=window_metadata.time.isoformat(),
                        image_idx=image_idx,
                        start_time=start_time,
                        end_time=end_time,
                    )
                )
