"""Helper functions to convert multitemporal rasters into Helios dataset."""

import csv
import logging
from datetime import timedelta

import numpy as np
import numpy.typing as npt
from rslearn.data_sources import Item
from rslearn.dataset import Window
from rslearn.utils.geometry import PixelBounds, Projection
from upath import UPath

from helios.data.constants import BandSet, ModalitySpec, TimeSpan
from helios.dataset.utils import get_modality_fname

from ..constants import GEOTIFF_RASTER_FORMAT, METADATA_COLUMNS
from ..util import get_modality_temp_meta_fname, get_window_metadata

PIXELS_PER_TILE = 256
EPSILON = 1e-6

logger = logging.getLogger(__name__)


def get_adjusted_projection_and_bounds(
    modality: ModalitySpec,
    band_set: BandSet,
    projection: Projection,
    window_bounds: PixelBounds,
) -> tuple[Projection, PixelBounds]:
    """Compute projection and bounds adjusted for this band set's resolution.

    Some bands may be stored at lower resolutions than the window bounds. So given the
    window projection and bounds, we compute the coarser projection corresponding to
    the band set, as well as the appropriate bounds in pixel coordinates under that
    projection.

    Args:
        modality: the ModalitySpec. It specifies a grid resolution.
        band_set: the BandSet. It specifies an resolution for the images that may be
            lower than the one used for the grid.
        projection: the projection of the window.
        window_bounds: the bounds of the window (which matches the modality's grid
            resolution).

    Returns:
        updated bounds at the resolution of the BandSet.
    """
    if band_set.resolution_factor >= modality.tile_resolution_factor:
        factor = band_set.resolution_factor // modality.tile_resolution_factor
        adjusted_projection = Projection(
            projection.crs,
            projection.x_resolution * factor,
            projection.y_resolution * factor,
        )
        adjusted_bounds = (
            window_bounds[0] // factor,
            window_bounds[1] // factor,
            window_bounds[2] // factor,
            window_bounds[3] // factor,
        )
    else:
        factor = modality.tile_resolution_factor // band_set.resolution_factor
        adjusted_projection = Projection(
            projection.crs,
            projection.x_resolution / factor,
            projection.y_resolution / factor,
        )
        adjusted_bounds = (
            window_bounds[0] * factor,
            window_bounds[1] * factor,
            window_bounds[2] * factor,
            window_bounds[3] * factor,
        )
    return adjusted_projection, adjusted_bounds


def convert_freq(
    window_path: UPath,
    helios_path: UPath,
    layer_name: str,
    modality: ModalitySpec,
    missing_okay: bool = False,
    unprepared_okay: bool = False,
) -> None:
    """Add frequent (two-week) data from this window to the Helios dataset.

    Args:
        window_path: the rslearn window directory to read data from.
        helios_path: Helios dataset path to write to.
        layer_name: the name of the layer containing frequent data in the rslearn
            dataset. It should be configured to individually store each item from the
            two-week period that spatially intersects with the window, i.e.
            space_mode=intersects, max_matches=9999.
        modality: the modality.
        missing_okay: whether it is okay if some images that appear in items.json are
            missing. This should only be enabled if there are unresolvable errors
            during ingestion.
        unprepared_okay: whether we should ignore the case where the window hasn't been
            prepared.
    """
    window = Window.load(window_path)
    window_metadata = get_window_metadata(window)
    layer_datas = window.load_layer_datas()

    if abs(window_metadata.resolution - modality.get_tile_resolution()) > EPSILON:
        raise ValueError(
            f"window ({window_metadata.resolution}) must have same "
            + f"resolution as modality ({modality.get_tile_resolution()})"
        )

    # Check if the layer is missing from the window's layer datas.
    # If unprepared_okay is set, then we return immediately since there is no work to
    # do for this window.
    if layer_name not in layer_datas:
        if unprepared_okay:
            return
        raise ValueError(
            f"layer {layer_name} is missing from layer datas for window {window.name}"
        )

    # We read the individual images and their timestamps, then write the stacked
    # images and CSV.
    # Map from band set to the images for that band set.
    images: dict[BandSet, list[npt.NDArray]] = {
        band_set: [] for band_set in modality.band_sets
    }
    timestamps = []
    for group_idx, group in enumerate(layer_datas[layer_name].serialized_item_groups):
        if len(group) != 1:
            raise ValueError(
                f"expected Landsat groups to have length 1 but got {len(group)}"
            )

        item = Item.deserialize(group[0])
        timestamp = item.geometry.time_range[0]
        cur_images: dict[BandSet, npt.NDArray] = {}

        for band_set in modality.band_sets:
            # Compute bounds of this raster adjusted for the resolution.
            adjusted_projection, adjusted_bounds = get_adjusted_projection_and_bounds(
                modality, band_set, window.projection, window.bounds
            )

            is_completed = window.is_layer_completed(layer_name, group_idx)
            # If missing images are okay, we ignore the uncompleted layer here.
            # Otherwise we will get an error when we try to read the GeoTIFF.
            if not is_completed and missing_okay:
                continue

            raster_dir = window.get_raster_dir(layer_name, band_set.bands, group_idx)
            logger.debug(
                "reading raster from %s with orig_bounds=%s adjusted_bounds=%s",
                raster_dir,
                window.bounds,
                adjusted_bounds,
            )
            image = GEOTIFF_RASTER_FORMAT.decode_raster(
                raster_dir, adjusted_projection, adjusted_bounds
            )
            expected_image_size = band_set.get_expected_image_size(
                window_metadata.get_resolution_factor()
            )
            if (
                image.shape[1] != expected_image_size
                or image.shape[2] != expected_image_size
            ):
                raise ValueError(
                    f"expected image size {expected_image_size} but got {image.shape}"
                )

            cur_images[band_set] = image

        if len(cur_images) < len(modality.band_sets):
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
            # Compute bounds of this raster adjusted for the resolution.
            adjusted_projection, adjusted_bounds = get_adjusted_projection_and_bounds(
                modality, band_set, window.projection, window.bounds
            )

            stacked_image = np.concatenate(band_set_images, axis=0)
            dst_fname = get_modality_fname(
                helios_path,
                modality,
                TimeSpan.TWO_WEEK,
                window_metadata,
                band_set.get_resolution(),
                "tif",
            )
            GEOTIFF_RASTER_FORMAT.encode_raster(
                path=dst_fname.parent,
                projection=adjusted_projection,
                bounds=adjusted_bounds,
                array=stacked_image,
                fname=dst_fname.name,
            )

        metadata_fname = get_modality_temp_meta_fname(
            helios_path, modality, TimeSpan.TWO_WEEK, window.name
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
    modality: ModalitySpec,
) -> None:
    """Add monthly (one-year) data from this window to the Helios dataset.

    Args:
        window_path: the rslearn window directory to read data from.
        helios_path: Helios dataset path to write to.
        layer_prefix: the prefix for the layer names containing monthly data in the
            rslearn dataset. The layers should be named with suffixes "_mo01", "_mo02",
            ..., "_mo12", where each layer contains a single mosaic for that month.
        bands: the band names.
        modality: the modality.
        band_sets: the band sets.
    """
    window = Window.load(window_path)
    window_metadata = get_window_metadata(window)

    if abs(window_metadata.resolution - modality.get_tile_resolution()) > EPSILON:
        raise ValueError(
            f"window ({window_metadata.resolution}) must have same "
            + f"resolution as modality ({modality.get_tile_resolution()})"
        )

    # The monthly images are stored in different layers, so we read one image per
    # layer. Then we reconstruct the time range to match the dataset configuration. And
    # finally stack the images and write them along with CSV.
    # Map from band set to list of images for that band set.
    images: dict[BandSet, list[npt.NDArray]] = {
        band_set: [] for band_set in modality.band_sets
    }
    time_ranges = []
    for month_idx in range(1, 13):
        layer_name = f"{layer_prefix}_mo{month_idx:02d}"
        start_time = window.time_range[0] + timedelta(days=(month_idx - 7) * 30)
        end_time = start_time + timedelta(days=30)

        cur_images: dict[BandSet, npt.NDArray] = {}

        for band_set in modality.band_sets:
            # Compute bounds of this raster adjusted for the resolution.
            adjusted_projection, adjusted_bounds = get_adjusted_projection_and_bounds(
                modality, band_set, window.projection, window.bounds
            )

            raster_dir = window.get_raster_dir(layer_name, band_set.bands)

            # Rasters may be missing for some months if there is no suitable data
            # during that month. So if any band is missing we exit and don't use that
            # month at this window.
            if not raster_dir.exists():
                break

            image = GEOTIFF_RASTER_FORMAT.decode_raster(
                raster_dir, adjusted_projection, adjusted_bounds
            )
            expected_image_size = band_set.get_expected_image_size(
                modality.tile_resolution_factor
            )
            if (
                image.shape[1] != expected_image_size
                or image.shape[2] != expected_image_size
            ):
                raise ValueError(
                    f"expected image size {expected_image_size} but got {image.shape}"
                )

            cur_images[band_set] = image

        if len(cur_images) < len(modality.band_sets):
            continue

        # Sometimes the images are blank because the window actually does not intersect
        # the raster. This is due to raster geometry information being too coarse in
        # some data sources. Here we skip those rasters so they don't get included with
        # this example in the Helios dataset.
        all_images_blank = all(image.max() == 0 for image in cur_images.values())
        if all_images_blank:
            continue

        time_ranges.append((start_time.isoformat(), end_time.isoformat()))
        for band_set, image in cur_images.items():
            images[band_set].append(image)

    if len(images[modality.band_sets[0]]) > 0:
        for band_set, band_set_images in images.items():
            # Compute bounds of this raster adjusted for the resolution.
            adjusted_projection, adjusted_bounds = get_adjusted_projection_and_bounds(
                modality, band_set, window.projection, window.bounds
            )

            stacked_image = np.concatenate(band_set_images, axis=0)
            dst_fname = get_modality_fname(
                helios_path,
                modality,
                TimeSpan.YEAR,
                window_metadata,
                band_set.get_resolution(),
                "tif",
            )
            GEOTIFF_RASTER_FORMAT.encode_raster(
                path=dst_fname.parent,
                projection=adjusted_projection,
                bounds=adjusted_bounds,
                array=stacked_image,
                fname=dst_fname.name,
            )

        metadata_fname = get_modality_temp_meta_fname(
            helios_path, modality, TimeSpan.YEAR, window.name
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
