"""Construct training samples from parsed OlmoEarth Pretrain CSVs."""

import logging
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import pandas as pd
import rasterio
import rasterio.windows
from pyproj import Transformer

from olmoearth_pretrain.data.constants import (
    PROJECTION_CRS,
    Modality,
    ModalitySpec,
    TimeSpan,
)

from .parse import GridTile, ModalityTile

logger = logging.getLogger(__name__)


@dataclass
class SampleInformation:
    """Specification of a training example.

    The example corresponds to one GridTile that appears in the dataset.

    It includes all of the information to load modalities at this tile, along with
    crops from coarser grained tiles that contain this tile.
    """

    grid_tile: GridTile

    # Whether this training example covers a one-year (TimeSpan.YEAR), two-week
    # (TimeSpan.TWO_WEEK), or high-frequency (TimeSpan.HIGH_FREQ) period.
    # Note that time_span should never be TimeSpan.STATIC since a training sample is
    # always tied to a specific time range.
    time_span: TimeSpan

    # The modalities available at this grid tile or coarser ones containing this tile.
    # The time spans from which the ModalityTiles are sourced should either match the
    # time span of the sample, or should be TimeSpan.STATIC.
    modalities: dict[ModalitySpec, ModalityTile]

    def get_latlon(self) -> np.ndarray:
        """Get the latlon of the sample."""
        x0, y0, x1, y1 = self.grid_tile.get_projected_extent()
        x = (x0 + x1) / 2
        y = (y0 + y1) / 2
        transformer = Transformer.from_crs(
            self.grid_tile.crs, PROJECTION_CRS, always_xy=True
        )
        lon, lat = transformer.transform(x, y)
        return np.array([lat, lon])

    def get_timestamps(self) -> dict[ModalitySpec, np.ndarray]:
        """Get the timestamps of the sample."""
        timestamps_dict: dict[ModalitySpec, np.ndarray] = {}

        for modality in self.modalities:
            if modality.is_multitemporal:
                sample_modality = self.modalities[modality]
                timestamps = [i.start_time for i in sample_modality.images]
                dt = pd.to_datetime(timestamps)
                timestamps_dict[modality] = np.array([dt.day, dt.month - 1, dt.year]).T

        return timestamps_dict


def image_tiles_to_samples(
    image_tiles: dict[ModalitySpec, dict[TimeSpan, list[ModalityTile]]],
    supported_modalities: list[ModalitySpec] = Modality.values(),
) -> list[SampleInformation]:
    """Compute samples from the parsed per-modality image tiles.

    Args:
        image_tiles: the parsed dataset from parse_dataset.
        supported_modalities: the modalities to include in the samples. Default is all
            modalities.

    Returns:
        a list of training examples (SampleInformation objects).
    """
    # TODO: make into separate function
    # Convert from (modality -> time_span -> tile list) to
    # (modality, grid_tile, time_span) -> tile).
    image_tile_index: dict[tuple[ModalitySpec, GridTile, TimeSpan], ModalityTile] = {}
    sample_id_index: dict[tuple[ModalitySpec, str, TimeSpan], list[ModalityTile]] = {}
    geometry_index: dict[tuple[ModalitySpec, TimeSpan, str], list[ModalityTile]] = {}
    for modality, modality_tiles in image_tiles.items():
        for time_span, time_span_tiles in modality_tiles.items():
            for tile in time_span_tiles:
                index_key = (modality, tile.grid_tile, time_span)
                image_tile_index[index_key] = tile
                if tile.grid_tile.sample_id:
                    sample_id_key = (modality, tile.grid_tile.sample_id, time_span)
                    sample_id_index.setdefault(sample_id_key, []).append(tile)
                geometry_key = (modality, time_span, tile.grid_tile.crs)
                geometry_index.setdefault(geometry_key, []).append(tile)

    # Enumerate all the (grid_tile, time_span) tuples present in the dataset.
    # Each of these identifies a training example.
    # We ignore static time span here, unless it is at the base resolution, in which
    # case we add it as both year and two-week, since currently all data at the base
    # resolution is static. (The intention here is to avoid adding a two-week tile
    # based on WorldCover being available if Sentinel-2 and others are only available
    # for one-year, but to still add NAIP or Maxar tiles.)
    unique_image_tiles: set[tuple[GridTile, TimeSpan]] = set()
    for modality, grid_tile, time_span in image_tile_index.keys():
        if time_span == TimeSpan.STATIC:
            if grid_tile.resolution_factor > 1:
                logger.debug(
                    f"ignoring static tile {grid_tile.resolution_factor} "
                    f"because it is coarser than the base resolution for modality {modality.name}"
                )
                continue
            else:
                unique_image_tiles.add((grid_tile, TimeSpan.TWO_WEEK))  # type: ignore
                unique_image_tiles.add((grid_tile, TimeSpan.YEAR))  # type: ignore
        else:
            unique_image_tiles.add((grid_tile, time_span))  # type: ignore

    # Now for each (grid_tile, time_span), construct the Sample object.
    # We also skip if not all modalities are available.
    samples: list[SampleInformation] = []
    for grid_tile, time_span in unique_image_tiles:
        sample = SampleInformation(
            grid_tile=grid_tile,
            time_span=time_span,
            modalities={},
        )

        # Add modalities one by one.
        for modality in image_tiles.keys():
            if modality not in supported_modalities:
                logger.warning(
                    f"ignoring modality {modality.name} not in supported_modalities"
                )
                continue
            # We only use modalities that are at an equal or coarser resolution.
            if modality.tile_resolution_factor < sample.grid_tile.resolution_factor:
                logger.debug(
                    f"ignoring modality {modality.name} with resolution factor "
                    f"{modality.tile_resolution_factor} because it is coarser than "
                    f"the sample grid tile resolution factor {sample.grid_tile.resolution_factor}"
                )
                continue

            downscale_factor = (
                modality.tile_resolution_factor // sample.grid_tile.resolution_factor
            )

            # Check to see if there is an available image tile for this modality.
            # If modality is static, then we just use TimeSpan.STATIC for the lookup.
            # If the modality is multitemporal, then we use the time span of the sample
            # for the lookup.
            lookup_time_span: TimeSpan
            if modality.is_multitemporal:
                lookup_time_span = sample.time_span  # type: ignore
            else:
                lookup_time_span = TimeSpan.STATIC  # type: ignore

            image_tile: ModalityTile | None = None
            if (
                grid_tile.use_grid_reference
                and grid_tile.col is not None
                and grid_tile.row is not None
            ):
                modality_grid_tile = GridTile(
                    crs=grid_tile.crs,
                    resolution_factor=modality.tile_resolution_factor,
                    col=grid_tile.col // downscale_factor,
                    row=grid_tile.row // downscale_factor,
                )
                index_key = (modality, modality_grid_tile, lookup_time_span)
                image_tile = image_tile_index.get(index_key)
                if image_tile is None:
                    logger.debug(
                        "ignoring modality %s because no tile found for index_key=%s",
                        modality.name,
                        index_key,
                    )
            else:
                if grid_tile.sample_id:
                    sample_id_key = (modality, grid_tile.sample_id, lookup_time_span)
                    candidates = sample_id_index.get(sample_id_key, [])
                    containing_tiles = [
                        candidate
                        for candidate in candidates
                        if candidate.grid_tile.contains(grid_tile)
                    ]
                    if containing_tiles:
                        image_tile = min(
                            containing_tiles,
                            key=lambda candidate: candidate.grid_tile.area(),
                        )

                if image_tile is None:
                    geometry_key = (modality, lookup_time_span, grid_tile.crs)
                    containing_tiles = [
                        candidate
                        for candidate in geometry_index.get(geometry_key, [])
                        if candidate.grid_tile.contains(grid_tile)
                    ]
                    if containing_tiles:
                        image_tile = min(
                            containing_tiles,
                            key=lambda candidate: candidate.grid_tile.area(),
                        )

                if image_tile is None:
                    logger.debug(
                        "ignoring modality %s because no containing tile found for sample_id=%s",
                        modality.name,
                        grid_tile.sample_id,
                    )
                    continue

            assert image_tile is not None
            sample.modalities[modality] = image_tile

        samples.append(sample)
    return samples


def load_image_for_sample(
    image_tile: ModalityTile, sample: SampleInformation
) -> npt.NDArray:
    """Loads the portion of the image that corresponds with the sample.

    If image_tile and sample share the same resolution, then we load the entire image.
    Otherwise, if the image tile is at a coarser resolution, then we load just the crop
    that is aligned with the sample.

    The sample must not have a coarser resolution -- that would require reading many
    image tiles and downsampling, but we do not want to do that.

    Args:
        image_tile: the image to load.
        sample: the SampleInformation. This is used to determine if the entire image
            should be loaded or just a portion of it.

    Returns:
        the image as a numpy array TCHW (time is on the first dimension).
        In the future, this may include vector data too, or that may go in a separate
        function.
    """
    # Compute the factor by which image_tile is bigger (coarser) than the sample.
    factor = max(
        1, image_tile.grid_tile.resolution_factor // sample.grid_tile.resolution_factor
    )
    sample_tile_size = sample.grid_tile.get_tile_size()
    # Read the modality image one band set at a time.
    # For now we resample all bands to the grid resolution of the modality.
    band_set_images = []
    for band_set, fname in image_tile.band_sets.items():
        logger.debug(f"band_set={band_set}, fname={fname}")
        with fname.open("rb") as f:
            with rasterio.open(f) as raster:
                # Identify the portion of the tile that we need to read.
                # We refer to this as a subtile.
                if raster.width != raster.height:
                    raise ValueError(
                        f"expected tile to be square but width={raster.width} != height={raster.height}"
                    )
                # If the modality does not vary in space (e.g., ERA5), we read the entire tile.
                if not image_tile.modality.is_spatial:
                    logger.debug(
                        f"reading entire tile {fname} for modality {image_tile.modality.name}"
                    )
                    image: npt.NDArray = raster.read()
                    # Remove spatial dimension since they're not needed.
                    image = image.reshape(-1, len(band_set.bands))
                    band_set_images.append(image)
                    continue

                # Assuming all tiles cover the same area as the resolution factor 16 tile
                if (
                    image_tile.grid_tile.contains(sample.grid_tile)
                    and image_tile.grid_tile.get_projected_extent()
                    != sample.grid_tile.get_projected_extent()
                ):
                    col_offset, row_offset, crop_width, crop_height = (
                        image_tile.grid_tile.get_crop_window(
                            sample.grid_tile, raster.width, raster.height
                        )
                    )
                    rasterio_window = rasterio.windows.Window(
                        col_off=col_offset,
                        row_off=row_offset,
                        width=crop_width,
                        height=crop_height,
                    )
                    logger.debug(f"reading window={rasterio_window} from {fname}")
                    image = raster.read(window=rasterio_window)  # type: ignore
                else:
                    image = raster.read()
                logger.debug(f"image.shape={image.shape}")
                subtile_size = image.shape[1]

                # And then for now resample it to the grid resolution.
                # The difference in resolution should always be a power of 2.
                # If the factor is less than 1 we want the desired size to be multiplied by the thing
                # If the tile size is greater we want to keep that extent
                desired_subtile_size = int(
                    sample_tile_size
                    * image_tile.modality.image_tile_size_factor
                    // factor
                )
                if desired_subtile_size < subtile_size:
                    # In this case we need to downscale.
                    # This should not be common, since usually bands would be stored at
                    # the image tile resolution or lower. But it could happen for
                    # OpenStreetMap. We just subsample the numpy array since averaging
                    # the pixels would not be correct for OpenStreetMap.
                    downscale_factor = subtile_size // desired_subtile_size
                    image = image[:, ::downscale_factor, ::downscale_factor]
                elif desired_subtile_size > subtile_size:
                    logger.debug(
                        f"desired_subtile_size={desired_subtile_size}, subtile_size={subtile_size}"
                    )
                    # This is the more common case, where we need to upscale because we
                    # stored some bands at a lower resolution, e.g. for Sentinel-2 or
                    # Landsat.
                    upscale_factor = desired_subtile_size // subtile_size
                    image = image.repeat(repeats=upscale_factor, axis=1).repeat(
                        repeats=upscale_factor, axis=2
                    )

                # Uncouple time / channel dimensions.
                shape = (
                    -1,
                    len(band_set.bands),
                    desired_subtile_size,
                    desired_subtile_size,
                )
                image = image.reshape(shape)
                logger.debug(f"shape after scaling image.shape={image.shape}")
                band_set_images.append(image)

    return np.concatenate(band_set_images, axis=1)


def load_nodata_mask_for_sample(
    image_tile: ModalityTile, sample: SampleInformation
) -> npt.NDArray:
    """Load a boolean valid-pixel mask for a spatial modality.

    Mirrors the spatial read logic in ``load_image_for_sample`` but returns a
    per-pixel, per-timestep boolean mask instead of the image data.

    A pixel is marked **False** (nodata) when *any* band at that (t, h, w)
    position equals the GeoTIFF nodata value.  Masks from different band sets
    are combined with AND (pixel must be valid in every band set).

    Convention matches ``missing_timesteps_masks``: **True = valid**.

    Args:
        image_tile: the modality tile to read.
        sample: the parent sample (used for windowing / resolution logic).

    Returns:
        boolean array of shape ``(T, H, W)`` where *H* and *W* are at the
        full raw-tile resolution (before subtile splitting).
    """
    if not image_tile.modality.is_spatial:
        raise ValueError(
            "load_nodata_mask_for_sample is only supported for spatial modalities"
        )

    factor = max(
        1, image_tile.grid_tile.resolution_factor // sample.grid_tile.resolution_factor
    )
    sample_tile_size = sample.grid_tile.get_tile_size()

    band_set_masks: list[npt.NDArray] = []
    for band_set, fname in image_tile.band_sets.items():
        with fname.open("rb") as f:
            with rasterio.open(f) as raster:
                if raster.width != raster.height:
                    raise ValueError(
                        f"expected tile to be square but width={raster.width} != height={raster.height}"
                    )

                nodata_val = raster.nodata

                if (
                    image_tile.grid_tile.contains(sample.grid_tile)
                    and image_tile.grid_tile.get_projected_extent()
                    != sample.grid_tile.get_projected_extent()
                ):
                    col_offset, row_offset, crop_width, crop_height = (
                        image_tile.grid_tile.get_crop_window(
                            sample.grid_tile, raster.width, raster.height
                        )
                    )
                    rasterio_window = rasterio.windows.Window(
                        col_off=col_offset,
                        row_off=row_offset,
                        width=crop_width,
                        height=crop_height,
                    )
                    image: npt.NDArray = raster.read(window=rasterio_window)
                else:
                    image = raster.read()

                subtile_size = image.shape[1]
                desired_subtile_size = int(
                    sample_tile_size
                    * image_tile.modality.image_tile_size_factor
                    // factor
                )

                if nodata_val is not None:
                    is_nodata = image == nodata_val
                else:
                    is_nodata = np.zeros_like(image, dtype=bool)

                if desired_subtile_size < subtile_size:
                    downscale_factor = subtile_size // desired_subtile_size
                    is_nodata = is_nodata[:, ::downscale_factor, ::downscale_factor]
                elif desired_subtile_size > subtile_size:
                    upscale_factor = desired_subtile_size // subtile_size
                    is_nodata = is_nodata.repeat(repeats=upscale_factor, axis=1).repeat(
                        repeats=upscale_factor, axis=2
                    )

                num_bands = len(band_set.bands)
                is_nodata = is_nodata.reshape(
                    -1, num_bands, desired_subtile_size, desired_subtile_size
                )

                # Any band nodata -> pixel nodata; then invert to valid mask
                valid_mask = ~np.any(is_nodata, axis=1)  # (T, H, W)
                band_set_masks.append(valid_mask)

    if not band_set_masks:
        raise ValueError("No spatial band sets found for nodata mask computation")

    combined = band_set_masks[0]
    for mask in band_set_masks[1:]:
        combined = combined & mask

    return combined
