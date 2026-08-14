"""Helper functions to convert multitemporal rasters into OlmoEarth Pretrain dataset."""

import csv
import logging
from datetime import datetime, timedelta

import numpy as np
import numpy.typing as npt
from rslearn.dataset import Window
from rslearn.utils.geometry import PixelBounds, Projection
from rslearn.utils.raster_array import RasterArray
from upath import UPath

from olmoearth_pretrain.data.constants import (
    SENTINEL1_NODATA,
    BandSet,
    Modality,
    ModalitySpec,
    TimeSpan,
)
from olmoearth_pretrain.dataset.utils import get_modality_fname

from ..constants import GEOTIFF_RASTER_FORMAT, METADATA_COLUMNS
from ..util import get_modality_temp_meta_fname, get_window_metadata

PIXELS_PER_TILE = 256
EPSILON = 1e-6

logger = logging.getLogger(__name__)


def _is_blank_mosaic(modality: ModalitySpec, image: npt.NDArray) -> bool:
    """Return whether an image contains only a configured blank/nodata fill."""
    if np.all(image == 0):
        return True
    return modality == Modality.SENTINEL1 and np.all(image == SENTINEL1_NODATA)


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


def convert_period_mosaic(
    window: Window,
    olmoearth_path: UPath,
    layer_name: str,
    modality: ModalitySpec,
    time_span: TimeSpan = TimeSpan.YEAR,
    missing_okay: bool = True,
    unprepared_okay: bool = True,
) -> None:
    """Add period-mosaic multitemporal data (one mosaic per period) to the dataset.

    Reads a single MOSAIC layer whose item groups are per-period mosaics
    (``space_mode=MOSAIC`` with ``period_duration``). Each item group becomes one
    timestep, stacked in the order the groups were materialized (chronological when
    ``per_period_mosaic_reverse_time_order=false``), using each group's period time
    range for the metadata. Item groups may contain multiple items (a mosaic), and
    per-group timestamps come from the layer's ``group_time_ranges``.

    Args:
        window: the rslearn window to read data from.
        olmoearth_path: OlmoEarth Pretrain dataset path to write to.
        layer_name: the name of the single MOSAIC layer to read.
        modality: the modality.
        time_span: the OlmoEarth Pretrain time span to write under (default YEAR).
        missing_okay: whether it is okay if some item groups are not completed.
        unprepared_okay: whether to skip windows where the layer is not prepared.
    """
    window_metadata = get_window_metadata(window)
    layer_datas = window.load_layer_datas()

    if abs(window_metadata.resolution - modality.get_tile_resolution()) > EPSILON:
        raise ValueError(
            f"window ({window_metadata.resolution}) must have same "
            + f"resolution as modality ({modality.get_tile_resolution()})"
        )

    if layer_name not in layer_datas:
        if unprepared_okay:
            return
        raise ValueError(
            f"layer {layer_name} is missing from layer datas for window {window.name}"
        )

    layer_data = layer_datas[layer_name]
    group_time_ranges = layer_data.group_time_ranges

    images: dict[BandSet, list[npt.NDArray]] = {
        band_set: [] for band_set in modality.band_sets
    }
    time_ranges: list[tuple[datetime, datetime] | None] = []
    for group_idx in range(len(layer_data.serialized_item_groups)):
        if not window.is_layer_completed(layer_name, group_idx):
            if missing_okay:
                continue
            raise ValueError(
                f"item group {group_idx} of layer {layer_name} is not completed "
                f"for window {window.name}"
            )

        cur_images: dict[BandSet, npt.NDArray] = {}
        for band_set in modality.band_sets:
            adjusted_projection, adjusted_bounds = get_adjusted_projection_and_bounds(
                modality, band_set, window.projection, window.bounds
            )
            image = window.data.read_raster(
                layer_name,
                band_set.bands,
                GEOTIFF_RASTER_FORMAT,
                projection=adjusted_projection,
                bounds=adjusted_bounds,
                group_idx=group_idx,
            ).get_chw_array()
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

        # Skip blank mosaics (window did not actually intersect the raster).
        if all(_is_blank_mosaic(modality, image) for image in cur_images.values()):
            continue

        cur_time_range = (
            group_time_ranges[group_idx]
            if group_time_ranges is not None and group_idx < len(group_time_ranges)
            else None
        )
        if cur_time_range is None:
            logger.warning(
                "skipping item group %d of layer %s for window %s because it has "
                "no period time range",
                group_idx,
                layer_name,
                window.name,
            )
            continue
        time_ranges.append(cur_time_range)
        for band_set, image in cur_images.items():
            images[band_set].append(image)

    if len(time_ranges) == 0:
        return

    for band_set, band_set_images in images.items():
        adjusted_projection, adjusted_bounds = get_adjusted_projection_and_bounds(
            modality, band_set, window.projection, window.bounds
        )
        stacked_image = np.concatenate(band_set_images, axis=0)
        dst_fname = get_modality_fname(
            olmoearth_path,
            modality,
            time_span,
            window_metadata,
            band_set.get_resolution(),
            "tif",
        )
        GEOTIFF_RASTER_FORMAT.encode_raster(
            path=dst_fname.parent,
            projection=adjusted_projection,
            bounds=adjusted_bounds,
            raster=RasterArray(chw_array=stacked_image),
            fname=dst_fname.name,
        )

    metadata_fname = get_modality_temp_meta_fname(
        olmoearth_path, modality, time_span, window.name
    )
    metadata_fname.parent.mkdir(parents=True, exist_ok=True)
    with metadata_fname.open("w") as f:
        writer = csv.DictWriter(f, fieldnames=METADATA_COLUMNS)
        writer.writeheader()
        for image_idx, cur_time_range in enumerate(time_ranges):
            start_time = cur_time_range[0].isoformat() if cur_time_range else ""
            end_time = cur_time_range[1].isoformat() if cur_time_range else ""
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


def convert_monthly(
    window: Window,
    olmoearth_path: UPath,
    layer_prefix: str,
    modality: ModalitySpec,
) -> None:
    """Add monthly (one-year) data from this window to the OlmoEarth Pretrain dataset.

    Args:
        window: the rslearn window to read data from.
        olmoearth_path: OlmoEarth Pretrain dataset path to write to.
        layer_prefix: the prefix for the layer names containing monthly data in the
            rslearn dataset. The layers should be named with suffixes "_mo01", "_mo02",
            ..., "_mo12", where each layer contains a single mosaic for that month.
        bands: the band names.
        modality: the modality.
        band_sets: the band sets.
    """
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

            # Rasters may be missing for some months if there is no suitable data
            # during that month. So if any band is missing we exit and don't use that
            # month at this window.
            if not window.is_layer_completed(layer_name):
                break

            image = window.data.read_raster(
                layer_name,
                band_set.bands,
                GEOTIFF_RASTER_FORMAT,
                projection=adjusted_projection,
                bounds=adjusted_bounds,
            ).get_chw_array()
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
        # this example in the OlmoEarth Pretrain dataset.
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
                olmoearth_path,
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
                raster=RasterArray(chw_array=stacked_image),
                fname=dst_fname.name,
            )

        metadata_fname = get_modality_temp_meta_fname(
            olmoearth_path, modality, TimeSpan.YEAR, window.name
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
