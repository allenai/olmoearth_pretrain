"""This is the data package for OlmoEarth training."""

from .concat import (
    HeliosConcatDataset,
    HeliosConcatDatasetConfig,
    OlmoEarthConcatDataset,
    OlmoEarthConcatDatasetConfig,
)
from .dataloader import (
    HeliosDataLoader,
    HeliosDataLoaderConfig,
    OlmoEarthDataLoader,
    OlmoEarthDataLoaderConfig,
)
from .dataset import (
    HeliosDataset,
    HeliosDatasetConfig,
    HeliosSample,
    OlmoEarthDataset,
    OlmoEarthDatasetConfig,
    OlmoEarthSample,
)

__all__ = [
    # New names
    "OlmoEarthSample",
    "OlmoEarthDataset",
    "OlmoEarthDatasetConfig",
    "OlmoEarthDataLoader",
    "OlmoEarthDataLoaderConfig",
    "OlmoEarthConcatDataset",
    "OlmoEarthConcatDatasetConfig",
    # Deprecated aliases
    "HeliosSample",
    "HeliosDataset",
    "HeliosDatasetConfig",
    "HeliosDataLoader",
    "HeliosDataLoaderConfig",
    "HeliosConcatDataset",
    "HeliosConcatDatasetConfig",
]
