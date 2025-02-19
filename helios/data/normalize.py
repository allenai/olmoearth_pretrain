"""Normalizer for the Helios dataset."""

import json
from enum import Enum

import numpy as np

from helios.data.constants import Modality, ModalitySpec


class Strategy(Enum):
    """The strategy to use for normalization."""

    # Whether to use predefined or computed values for normalization
    PREDEFINED = "predefined"
    COMPUTED = "computed"


# Normalization strategy for each modality
NORMALIZE_STRATEGY = {
    Modality.SENTINEL1: Strategy.PREDEFINED,
    Modality.SENTINEL2: Strategy.PREDEFINED,
    Modality.WORLDCOVER: Strategy.PREDEFINED,
}


class Normalizer:
    """Normalize the data."""

    def __init__(
        self,
        strategy: Strategy,
        std_multiplier: float | None = 2,
    ) -> None:
        """Initialize the normalizer.

        Args:
            strategy: The strategy to use for normalization (predefined or computed).
            std_multiplier: Optional, only for strategy COMPUTED.
                            The multiplier for the standard deviation when using computed values.

        Returns:
            None
        """
        self.strategy = strategy
        self.std_multiplier = std_multiplier
        self.norm_config = self._load_config()

    def _load_config(self) -> dict:
        """Load the appropriate config based on the modality strategy."""
        if self.strategy == Strategy.PREDEFINED:
            return self._load_predefined_config()
        elif self.strategy == Strategy.COMPUTED:
            return self._load_computed_config()
        else:
            raise ValueError(f"Invalid strategy: {self.strategy}")

    def _load_predefined_config(self) -> dict:
        """Load the predefined config."""
        with open("data/norm_configs/predefined.json") as f:
            return json.load(f)

    def _load_computed_config(self) -> dict:
        """Load the computed config."""
        with open("data/norm_configs/computed.json") as f:
            return json.load(f)

    def _normalize_predefined(
        self, modality: ModalitySpec, data: np.ndarray
    ) -> np.ndarray:
        """Normalize the data using predefined values."""
        # When using predefined values, we have the min and max values for each band
        modality_bands = modality.band_order
        modality_norm_values = self.norm_config[modality.name]
        min_vals = []
        max_vals = []
        for band in modality_bands:
            if band not in modality_norm_values:
                raise ValueError(f"Band {band} not found in config")
            min_val = modality_norm_values[band]["min"]
            max_val = modality_norm_values[band]["max"]
            min_vals.append(min_val)
            max_vals.append(max_val)
        # The last dimension of data is always the number of bands (channels)
        return (data - np.array(min_vals)) / (np.array(max_vals) - np.array(min_vals))

    def _normalize_computed(
        self, modality: ModalitySpec, data: np.ndarray
    ) -> np.ndarray:
        """Normalize the data using computed values."""
        # When using computed values, we compute the mean and std of each band in advance
        # Then convert the values to min and max values that cover ~90% of the data
        modality_bands = modality.band_order
        modality_norm_values = self.norm_config[modality.name]
        mean_vals = []
        std_vals = []
        for band in modality_bands:
            if band not in modality_norm_values:
                raise ValueError(f"Band {band} not found in config")
            mean_val = modality_norm_values[band]["mean"]
            std_val = modality_norm_values[band]["std"]
            mean_vals.append(mean_val)
            std_vals.append(std_val)
        min_vals = np.array(mean_vals) - self.std_multiplier * np.array(std_vals)
        max_vals = np.array(mean_vals) + self.std_multiplier * np.array(std_vals)
        return (data - min_vals) / (max_vals - min_vals)  # type: ignore

    def normalize(self, modality: ModalitySpec, data: np.ndarray) -> np.ndarray:
        """Normalize the data.

        Args:
            modality: The modality to normalize.
            data: The data to normalize.

        Returns:
            The normalized data.
        """
        if self.strategy == Strategy.PREDEFINED:
            return self._normalize_predefined(modality, data)
        elif self.strategy == Strategy.COMPUTED:
            return self._normalize_computed(modality, data)
        else:
            raise ValueError(f"Invalid strategy: {self.strategy}")
