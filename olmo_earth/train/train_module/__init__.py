"""OlmoEarth train modules."""

from .train_module import (
    HeliosTrainModule,
    HeliosTrainModuleConfig,
    OlmoEarthTrainModule,
    OlmoEarthTrainModuleConfig,
)

__all__ = [
    # New names
    "OlmoEarthTrainModule",
    "OlmoEarthTrainModuleConfig",
    # Deprecated aliases
    "HeliosTrainModule",
    "HeliosTrainModuleConfig",
]
