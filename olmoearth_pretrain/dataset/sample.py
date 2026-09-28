"""Construct training samples from parsed OlmoEarth Pretrain CSVs."""

import logging
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import pandas as pd
import rasterio
from pyproj import Transformer

from olmoearth_pretrain.data.constants import (
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

    The example corresponds to one GridTile that appears in the dataset. Each modality is
    materialized per-window at this tile, so it includes all of the information to load
    every modality at this tile.
    """

    grid_tile: GridTile

    # Whether this training example covers a one-year (TimeSpan.YEAR) or two-week
    # (TimeSpan.TWO_WEEK) period.
    # Note that time_span should never be TimeSpan.STATIC since a training sample is
    # always tied to a specific time range.
    time_span: TimeSpan

    # The modalities available at this grid tile.
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
    for modality, modality_tiles in image_tiles.items():
        for time_span, time_span_tiles in modality_tiles.items():
            for tile in time_span_tiles:
                index_key = (modality, tile.grid_tile, time_span)
                image_tile_index[index_key] = tile

    # Enumerate all the (grid_tile, time_span) tuples present in the dataset.
    # Each of these identifies a training example. Only multitemporal (YEAR) tiles
    # define training examples; static tiles are attached to those samples below (in
    # the per-modality loop) rather than creating their own.
    unique_image_tiles: set[tuple[GridTile, TimeSpan]] = set()
    for modality, grid_tile, time_span in image_tile_index.keys():
        if time_span == TimeSpan.STATIC:
            continue
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

            # Every modality is materialized per-window at the sample's own grid tile
            # (the 10 m/pixel base grid), so we look it up directly. Per-band-set
            # resolution differences (e.g. 20 m Sentinel-2 bands, or naip_10 stored
            # finer than 10 m) are handled by resampling in load_image_for_sample.
            # If modality is static, then we just use TimeSpan.STATIC for the lookup.
            # If the modality is multitemporal, then we use the time span of the sample
            # for the lookup.
            lookup_time_span: TimeSpan
            if modality.is_multitemporal:
                lookup_time_span = sample.time_span  # type: ignore
            else:
                lookup_time_span = TimeSpan.STATIC  # type: ignore

            index_key = (modality, grid_tile, lookup_time_span)
            if index_key not in image_tile_index:
                logger.debug(
                    f"ignoring modality {modality.name} because no tile found for index_key={index_key}"
                )
                continue

            # We found a tile, so we just add it in the modality map for this sample.
            sample.modalities[modality] = image_tile_index[index_key]

        samples.append(sample)
    return samples


def load_image_for_sample(
    image_tile: ModalityTile,
    sample: SampleInformation,
) -> npt.NDArray:
    """Loads the per-window image for a modality.

    Every modality is materialized per-window at the sample's own extent, so the entire
    raster is read. Band sets stored at a coarser or finer resolution than the modality
    grid (e.g. 20 m Sentinel-2 bands, or naip_10 stored at 2.5 m/pixel) are resampled to
    the modality's grid resolution.

    Args:
        image_tile: the image to load.
        sample: the SampleInformation.

    Returns:
        the image as a numpy array TCHW (time is on the first dimension).
        In the future, this may include vector data too, or that may go in a separate
        function.
    """
    # Read the modality image one band set at a time.
    # For now we resample all bands to the grid resolution of the modality.
    band_set_images = []
    for band_set, fname in image_tile.band_sets.items():
        logger.debug(f"band_set={band_set}, fname={fname}")
        with fname.open("rb") as f:
            with rasterio.open(f) as raster:
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

                image: npt.NDArray = raster.read()  # type: ignore
                subtile_size = raster.width
                logger.debug(f"image.shape={image.shape}")

                # Resample the band set to the modality's grid resolution.
                # The difference in resolution should always be a power of 2.
                desired_subtile_size = int(
                    IMAGE_TILE_SIZE * image_tile.modality.image_tile_size_factor
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
