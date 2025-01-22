"""Input/Output handlers for different data source types."""

from abc import ABC, abstractmethod

import numpy as np
import rasterio
from einops import rearrange
from upath import UPath

from helios.constants import S2_BANDS


class DataSourceReader(ABC):
    """Base class for data source readers.

    All readers should return data in (H, W, T, C) format where:
    - H, W are spatial dimensions
    - T is number of timesteps (1 for static data)
    - C is number of channels/bands
    """

    @classmethod
    @abstractmethod
    def load(cls, file_path: UPath | str) -> tuple[np.ndarray, int]:
        """Load data from file path.

        Args:
            file_path: Path to the data file

        Returns:
            Tuple of:
                - data_array: Numpy array of shape (H, W, T, C)
                - num_timesteps: Number of timesteps in the data
        """
        pass


class TiffReader(DataSourceReader):
    """Base reader for Tiff files."""

    @classmethod
    def load(cls, file_path: UPath | str) -> tuple[np.ndarray, int]:
        """Load data from a Tiff file."""
        with rasterio.open(file_path) as data:
            values = data.read()
        return values, 1  # Default to single timestep


class GeoJSONReader(DataSourceReader):
    """Base reader for GeoJSON files."""

    @classmethod
    def load(cls, file_path: UPath | str) -> tuple[np.ndarray, int]:
        """Load data from a GeoJSON file."""
        # Convert GeoJSON to appropriate array format
        # Implementation depends on how we want to represent vector data
        return np.array([]), 1


class Sentinel2Reader(TiffReader):
    """Reader for Sentinel-2 data."""

    @classmethod
    def load(cls, file_path: UPath | str) -> tuple[np.ndarray, int]:
        """Load Sentinel-2 data with specific band handling.

        Returns:
            Tuple of:
                - array of shape (H, W, T, C) where C is len(S2_BANDS)
                - number of timesteps T
        """
        values, _ = super().load(file_path)

        num_timesteps = values.shape[0] / len(S2_BANDS)
        assert num_timesteps % 1 == 0, (
            f"{file_path} has incorrect number of channels {S2_BANDS} "
            f"{values.shape[0]=} {len(S2_BANDS)=}"
        )
        num_timesteps = int(num_timesteps)

        space_time_x = rearrange(
            values, "(t c) h w -> h w t c", c=len(S2_BANDS), t=num_timesteps
        )

        return space_time_x, num_timesteps


class WorldCoverReader(TiffReader):
    """Reader for WorldCover data."""

    @classmethod
    def load(cls, file_path: UPath | str) -> tuple[np.ndarray, int]:
        """Load WorldCover data.

        Returns:
            Tuple of:
                - array of shape (H, W, 1, 1) containing land cover classes
                - always returns 1 timestep
        """
        values, _ = super().load(file_path)
        # Ensure single channel and add time dimension for consistency
        values = values[0][None, :, :, None]  # Shape: (H, W, 1, 1)
        return values, 1


class OpenStreetMapReader(GeoJSONReader):
    """Reader for OpenStreetMap data."""

    @classmethod
    def load(cls, file_path: UPath | str) -> tuple[np.ndarray, int]:
        """Load OpenStreetMap data.

        Returns:
            Tuple of:
                - array of shape (H, W, 1, 3) with binary channels for [roads, buildings, water]
                - always returns 1 timestep
        """
        # Create placeholder raster with binary channels for different feature types
        placeholder = np.zeros(
            (256, 256, 1, 3), dtype=np.float32
        )  # H, W, T, C(roads, buildings, water)
        return placeholder, 1


class NAIPReader(TiffReader):
    """Reader for NAIP imagery."""

    @classmethod
    def load(cls, file_path: UPath | str) -> tuple[np.ndarray, int]:
        """Load NAIP data.

        Returns:
            Tuple of:
                - array of shape (H, W, 1, 4) containing [R, G, B, NIR] bands
                - always returns 1 timestep
        """
        values, _ = super().load(file_path)
        # NAIP typically has 4 bands (RGB + NIR) in single timestep
        # Reshape to expected format (H, W, 1, C)
        values = values.transpose(1, 2, 0)[..., None, :]  # (H, W, C) -> (H, W, 1, C)
        return values, 1


class DataSourceLoaderRegistry:
    """Registry for data source readers."""

    _readers: dict[str, type[DataSourceReader]] = {}

    @classmethod
    def register(cls, source_name: str, reader: type[DataSourceReader]) -> None:
        """Register a reader for a data source."""
        cls._readers[source_name] = reader

    @classmethod
    def get_reader(cls, source_name: str) -> type[DataSourceReader]:
        """Get the reader for a data source."""
        if source_name not in cls._readers:
            raise ValueError(f"No reader registered for source: {source_name}")
        return cls._readers[source_name]


# Register all readers
DataSourceLoaderRegistry.register("sentinel2", Sentinel2Reader)
DataSourceLoaderRegistry.register("worldcover", WorldCoverReader)
DataSourceLoaderRegistry.register("openstreetmap", OpenStreetMapReader)
DataSourceLoaderRegistry.register("naip", NAIPReader)
