"""Normalizer for the OlmoEarth Pretrain dataset."""

import json
import logging
from enum import Enum
from importlib.resources import files

import numpy as np

from olmoearth_pretrain.data.constants import ModalitySpec

logger = logging.getLogger(__name__)


def load_predefined_config() -> dict[str, dict[str, dict[str, float]]]:
    """Load the predefined config.

    The normalization config maps from modality -> band name to a dictionary with min
    and max keys.
    """
    with (
        files("olmoearth_pretrain.data.norm_configs") / "predefined.json"
    ).open() as f:
        return json.load(f)


def load_computed_config() -> dict[str, dict]:
    """Load the computed config.

    The normalization config maps from modality -> band name to a dictionary with mean
    and std keys.
    """
    with (files("olmoearth_pretrain.data.norm_configs") / "computed.json").open() as f:
        return json.load(f)


def load_arcsinh_tanh_config() -> dict[str, dict]:
    """Load the arcsinh_tanh config.

    The normalization config maps from modality -> band name to a dictionary with keys:
    ``transform`` (``"arcsinh"`` or ``"identity"``), optional ``c`` (the arcsinh scale,
    required when ``transform == "arcsinh"``), ``mean`` and ``std``. The ``mean`` and
    ``std`` are computed in the transformed space (i.e. after applying the
    variance-stabilizing transform), on tail-clipped data so that cloud/outlier pixels
    do not inflate the standard deviation.
    """
    with (
        files("olmoearth_pretrain.data.norm_configs") / "computed_arcsinh_tanh.json"
    ).open() as f:
        return json.load(f)


class Strategy(Enum):
    """The strategy to use for normalization."""

    # Whether to use predefined or computed values for normalization
    PREDEFINED = "predefined"
    COMPUTED = "computed"
    # Per-band variance-stabilizing transform (arcsinh for skewed bands, identity for
    # already-Gaussian bands) -> z-score in the transformed space -> tanh, mapping the
    # bulk of the distribution across most of (-1, 1) while saturating outliers.
    ARCSINH_TANH = "arcsinh_tanh"


class Normalizer:
    """Normalize the data."""

    def __init__(
        self,
        strategy: Strategy,
        std_multiplier: float | None = 2,
        tanh_gain: float = 1.0,
    ) -> None:
        """Initialize the normalizer.

        Args:
            strategy: The strategy to use for normalization (predefined, computed, or
                arcsinh_tanh).
            std_multiplier: Optional, only for strategy COMPUTED.
                            The multiplier for the standard deviation when using computed values.
            tanh_gain: Only for strategy ARCSINH_TANH. Multiplier applied to the z-scored
                            value before the tanh squash. Larger values saturate the tails
                            more aggressively.

        Returns:
            None
        """
        self.strategy = strategy
        self.std_multiplier = std_multiplier
        self.tanh_gain = tanh_gain
        self.norm_config = self._load_config()

    def _load_config(self) -> dict:
        """Load the appropriate config based on the modality strategy."""
        if self.strategy == Strategy.PREDEFINED:
            return load_predefined_config()
        elif self.strategy == Strategy.COMPUTED:
            return load_computed_config()
        elif self.strategy == Strategy.ARCSINH_TANH:
            return load_arcsinh_tanh_config()
        else:
            raise ValueError(f"Invalid strategy: {self.strategy}")

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

    def _normalize_arcsinh_tanh(
        self, modality: ModalitySpec, data: np.ndarray
    ) -> np.ndarray:
        """Normalize via per-band VST -> z-score -> tanh, output in (-1, 1).

        Each band is first passed through a variance-stabilizing transform (``arcsinh``
        for right-skewed bands, ``identity`` for already-Gaussian bands), then z-scored
        using the transformed-space mean/std from the config, then squashed with
        ``tanh(tanh_gain * z)``. The bulk of the distribution spreads across most of
        (-1, 1) while outliers (e.g. clouds) saturate near +/-1.
        """
        modality_bands = modality.band_order
        modality_norm_values = self.norm_config[modality.name]
        means, stds, cs, is_arcsinh = [], [], [], []
        for band in modality_bands:
            if band not in modality_norm_values:
                raise ValueError(f"Band {band} not found in config")
            band_cfg = modality_norm_values[band]
            transform = band_cfg["transform"]
            means.append(band_cfg["mean"])
            stds.append(band_cfg["std"])
            if transform == "arcsinh":
                is_arcsinh.append(True)
                cs.append(band_cfg["c"])
            elif transform == "identity":
                is_arcsinh.append(False)
                # Placeholder scale; unused because is_arcsinh is False for this band.
                cs.append(1.0)
            else:
                raise ValueError(
                    f"Unknown transform '{transform}' for band {band} of "
                    f"modality {modality.name}"
                )
        means_arr = np.array(means)
        stds_arr = np.array(stds)
        cs_arr = np.array(cs)
        is_arcsinh_arr = np.array(is_arcsinh)
        # arcsinh is nan-safe for any finite input (including MISSING_VALUE), and
        # identity bands use c=1 so the discarded arcsinh branch never divides by zero.
        transformed = np.where(is_arcsinh_arr, np.arcsinh(data / cs_arr), data)
        z = (transformed - means_arr) / stds_arr
        return np.tanh(self.tanh_gain * z)

    def normalize(self, modality: ModalitySpec, data: np.ndarray) -> np.ndarray:
        """Normalize the data.

        Args:
            modality: The modality to normalize.
            data: The data to normalize.

        Returns:
            The normalized data.
        """
        # Categorical/one-hot modalities (e.g. worldcover_onehot) are already in the
        # right range and have no min/max or mean/std entries in the config.
        if modality.skip_normalization:
            return data
        if self.strategy == Strategy.PREDEFINED:
            return self._normalize_predefined(modality, data)
        elif self.strategy == Strategy.COMPUTED:
            return self._normalize_computed(modality, data)
        elif self.strategy == Strategy.ARCSINH_TANH:
            return self._normalize_arcsinh_tanh(modality, data)
        else:
            raise ValueError(f"Invalid strategy: {self.strategy}")
