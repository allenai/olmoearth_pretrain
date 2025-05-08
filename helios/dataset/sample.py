"""Construct training samples from parsed Helios CSVs."""

import logging
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import pandas as pd
import rasterio
import rasterio.windows
from pyproj import Transformer

from helios.data.constants import (
    BASE_RESOLUTION,
    IMAGE_TILE_SIZE,
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

    # Whether this training example covers a one-year (TimeSpan.YEAR) or two-week
    # (TimeSpan.TWO_WEEK) period.
    # Note that time_span should never be TimeSpan.STATIC since a training sample is
    # always tied to a specific time range.
    time_span: TimeSpan

    # The modalities available at this grid tile or coarser ones containing this tile.
    # The time spans from which the ModalityTiles are sourced should either match the
    # time span of the sample, or should be TimeSpan.STATIC.
    modalities: dict[ModalitySpec, ModalityTile]

    def get_latlon(self) -> np.ndarray:
        """Get the latlon of the sample."""
        # Get coordinates at projection units, and then transform to latlon
        grid_resolution = self.grid_tile.resolution_factor * BASE_RESOLUTION
        x, y = (
            (self.grid_tile.col + 0.5) * grid_resolution * IMAGE_TILE_SIZE,
            (self.grid_tile.row + 0.5) * -grid_resolution * IMAGE_TILE_SIZE,
        )
        transformer = Transformer.from_crs(
            self.grid_tile.crs, PROJECTION_CRS, always_xy=True
        )
        lon, lat = transformer.transform(x, y)
        return np.array([lat, lon])

    def get_timestamps(self) -> np.ndarray:
        """Get the timestamps of the sample."""
        # Assume that all multitemporal modalities have the same timestamps
        for modality in self.modalities:
            if modality.is_multitemporal:
                sample_modality = self.modalities[modality]
                timestamps = [i.start_time for i in sample_modality.images]
                dt = pd.to_datetime(timestamps)
                if len(timestamps) != 12:
                    raise ValueError(
                        "Expected 12 timestamps for multitemporal modality, must adapt if this is to change"
                    )
                return np.array([dt.day, dt.month - 1, dt.year]).T

        # Now try non-multitemporal modalities as backups
        for modality in self.modalities:
            if not modality.is_multitemporal:
                sample_modality = self.modalities[modality]
                timestamps = [i.start_time for i in sample_modality.images]
                dt = pd.to_datetime(timestamps)
                return np.array([dt.day, dt.month - 1, dt.year]).T

        raise ValueError("No multitemporal or non-multitemporal modalities found")


def image_tiles_to_samples(
    image_tiles: dict[ModalitySpec, dict[TimeSpan, list[ModalityTile]]],
    supported_modalities: list[ModalitySpec] = Modality.values(),
) -> list[SampleInformation]:
    """Compute samples from the parsed per-modality image tiles.

    Args:
        image_tiles: the parsed dataset from parse_helios_dataset.
        supported_modalities: the modalities to include in the samples. Default is all
            modalities.

    Returns:
        a list of training examples (SampleInformation objects).
    """
    # TODO: make into separate function
    # Convert from (modality -> time_span -> tile list) to
    # (modality, grid_tile, time_span) -> tile).
    image_tile_index: dict[tuple[ModalitySpec, GridTile, TimeSpan], ModalityTile] = {}
    for modality, modality_tiles in image_tiles.items():
        for time_span, time_span_tiles in modality_tiles.items():
            for tile in time_span_tiles:
                index_key = (modality, tile.grid_tile, time_span)
                image_tile_index[index_key] = tile

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

            # We need to downscale the grid tile for the lookup.
            modality_grid_tile = GridTile(
                crs=grid_tile.crs,
                resolution_factor=modality.tile_resolution_factor,
                col=grid_tile.col // downscale_factor,
                row=grid_tile.row // downscale_factor,
            )

            index_key = (modality, modality_grid_tile, lookup_time_span)
            if index_key not in image_tile_index:
                logger.debug(
                    f"ignoring modality {modality.name} because no tile found for index_key={index_key}"
                )
                continue
            image_tile = image_tile_index[index_key]

            # We found a tile, so we just add it in the modality map for this sample.
            # The ImageTile object includes all the information needed to load the
            # image (potentially requiring cropping).
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
    factor = (
        image_tile.grid_tile.resolution_factor // sample.grid_tile.resolution_factor
    )
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
                # Assuming all tiles cover the same area as the resolution factor 16 tile
                subtile_size = raster.width // factor
                col_offset = subtile_size * (sample.grid_tile.col % factor)
                row_offset = subtile_size * (sample.grid_tile.row % factor)

                # Now we can perform a windowed read.
                rasterio_window = rasterio.windows.Window(
                    col_off=col_offset,
                    row_off=row_offset,
                    width=subtile_size,
                    height=subtile_size,
                )
                logger.debug(f"reading window={rasterio_window} from {fname}")
                image: npt.NDArray = raster.read(window=rasterio_window)
                logger.debug(f"image.shape={image.shape}")

                # And then for now resample it to the grid resolution.
                # The difference in resolution should always be a power of 2.
                # If the factor is less than 1 we want the desired size to be multiplied by the thing
                # If the tile size is greater we want to keep that extent
                desired_subtile_size = int(
                    IMAGE_TILE_SIZE
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
