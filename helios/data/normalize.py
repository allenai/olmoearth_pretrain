"""Normalizer for the Helios dataset."""

import json
import logging
from dataclasses import dataclass
from enum import Enum
from importlib.resources import files

import numpy as np
from olmo_core.config import Config

from helios.data.constants import ModalitySpec

logger = logging.getLogger(__name__)


@dataclass
class BandStats(Config):
    """Statistics for a single band."""

    count: int
    mean: float
    std: float
    var: float


@dataclass
class BandMinMax(Config):
    """Min/max values for a single band."""

    min: float
    max: float


@dataclass
class ModalityComputedConfig(Config):
    """Computed statistics configuration for a modality."""

    bands: dict[str, BandStats]


@dataclass
class ModalityPredefinedConfig(Config):
    """Predefined min/max configuration for a modality."""

    bands: dict[str, BandMinMax]


@dataclass
class ComputedNormConfig(Config):
    """Complete computed normalization configuration."""

    modalities: dict[str, ModalityComputedConfig]
    sampled_n: int
    total_n: int
    tile_path: str

    @classmethod
    def load_from_json_files(cls) -> "ComputedNormConfig":
        """Load computed configuration from JSON file."""
        with (files("helios.data.norm_configs") / "computed.json").open() as f:
            data = json.load(f)

        modalities = {}
        for modality_name, modality_data in data.items():
            if modality_name not in {"sampled_n", "total_n", "tile_path"}:
                bands = {}
                for band_name, stats in modality_data.items():
                    bands[band_name] = BandStats(**stats)
                modalities[modality_name] = ModalityComputedConfig(bands=bands)

        return cls(
            modalities=modalities,
            sampled_n=data["sampled_n"],
            total_n=data["total_n"],
            tile_path=data["tile_path"],
        )


@dataclass
class PredefinedNormConfig(Config):
    """Complete predefined normalization configuration."""

    modalities: dict[str, ModalityPredefinedConfig]

    @classmethod
    def load_from_json_files(cls) -> "PredefinedNormConfig":
        """Load predefined configuration from JSON file."""
        with (files("helios.data.norm_configs") / "predefined.json").open() as f:
            data = json.load(f)

        modalities = {}
        for modality_name, modality_data in data.items():
            bands = {}
            for band_name, minmax in modality_data.items():
                bands[band_name] = BandMinMax(**minmax)
            modalities[modality_name] = ModalityPredefinedConfig(bands=bands)

        return cls(modalities=modalities)


class Strategy(Enum):
    """The strategy to use for normalization."""

    PREDEFINED = "predefined"
    COMPUTED = "computed"


@dataclass
class NormalizerConfig(Config):
    """The normalization config - fully serializable."""

    strategy: Strategy = Strategy.COMPUTED
    std_multiplier: float | None = 2.0

    # Embed the full norm configs for complete serializability
    computed_config: ComputedNormConfig | None = None
    predefined_config: PredefinedNormConfig | None = None

    def __post_init__(self) -> None:
        """Load configs from JSON files if not provided."""
        # allows us to
        if self.computed_config is None:
            self.computed_config = ComputedNormConfig.load_from_json_files()
        if self.predefined_config is None:
            self.predefined_config = PredefinedNormConfig.load_from_json_files()

    def build(self) -> "Normalizer":
        """Build the normalizer."""
        return Normalizer(
            strategy=self.strategy,
            std_multiplier=self.std_multiplier,
            computed_config=self.computed_config,
            predefined_config=self.predefined_config,
        )


class Normalizer:
    """Normalize the data."""

    def __init__(
        self,
        strategy: Strategy,
        std_multiplier: float | None,
        computed_config: ComputedNormConfig | None = None,
        predefined_config: PredefinedNormConfig | None = None,
    ) -> None:
        """Initialize the normalizer."""
        self.strategy = strategy
        self.std_multiplier = std_multiplier
        self.computed_config = (
            computed_config or ComputedNormConfig.load_from_json_files()
        )
        self.predefined_config = (
            predefined_config or PredefinedNormConfig.load_from_json_files()
        )

    def _normalize_predefined(
        self, modality: ModalitySpec, data: np.ndarray
    ) -> np.ndarray:
        """Normalize the data using predefined values."""
        modality_config = self.predefined_config.modalities[modality.name]
        min_vals = []
        max_vals = []

        for band in modality.band_order:
            if band not in modality_config.bands:
                raise ValueError(f"Band {band} not found in config")
            band_config = modality_config.bands[band]
            min_vals.append(band_config.min)
            max_vals.append(band_config.max)

        return (data - np.array(min_vals)) / (np.array(max_vals) - np.array(min_vals))

    def _normalize_computed(
        self, modality: ModalitySpec, data: np.ndarray
    ) -> np.ndarray:
        """Normalize the data using computed values."""
        modality_config = self.computed_config.modalities[modality.name]
        mean_vals = []
        std_vals = []

        for band in modality.band_order:
            if band not in modality_config.bands:
                raise ValueError(f"Band {band} not found in config")
            band_stats = modality_config.bands[band]
            mean_vals.append(band_stats.mean)
            std_vals.append(band_stats.std)

        min_vals = np.array(mean_vals) - self.std_multiplier * np.array(std_vals)
        max_vals = np.array(mean_vals) + self.std_multiplier * np.array(std_vals)
        return (data - min_vals) / (max_vals - min_vals)

    def normalize(
        self, modality: ModalitySpec, data: np.ndarray, strategy: Strategy | None = None
    ) -> np.ndarray:
        """Normalize the data."""
        if strategy is None:
            strategy = self.strategy
        if strategy == Strategy.PREDEFINED:
            return self._normalize_predefined(modality, data)
        elif strategy == Strategy.COMPUTED:
            return self._normalize_computed(modality, data)
        else:
            raise ValueError(f"Invalid strategy: {self.strategy}")
