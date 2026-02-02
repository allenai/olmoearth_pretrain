"""Schema definitions for the eval dataset registry.

This module defines the dataclasses that represent:
1. Per-band normalization statistics (BandStats, ModalityStats)
2. Dataset registry entries (EvalDatasetEntry)

These are serialized to JSON and stored on Weka alongside the dataset.

Design Decisions:
-----------------
- We use dataclasses for simplicity and JSON serialization
- All fields are explicitly typed for clarity
- Optional fields use None as default
- Timestamps are ISO 8601 strings for human readability
- Paths are stored as strings (not UPath) for JSON compatibility

Future Considerations:
---------------------
- May want to add versioning to schema for backwards compatibility
- May want to add validation methods to dataclasses
- May want to support additional task types (detection, etc.)

Todo:
-----
- [ ] Consider migrating to pydantic models for better validation,
      serialization, and schema generation. Would provide:
      - Automatic validation on construction
      - JSON Schema export for documentation
      - Better error messages
      - Field aliasing and serialization customization
"""

from __future__ import annotations

import importlib
import json
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from olmoearth_pretrain.evals.datasets.configs import EvalDatasetConfig

from rslearn.train.tasks.classification import (
    ClassificationTask as RsClassificationTask,
)
from rslearn.train.tasks.segmentation import SegmentationTask as RsSegmentationTask

from olmoearth_pretrain.data.constants import Modality, ModalitySpec
from olmoearth_pretrain.evals.task_types import TaskType

DEFAULT_TARGET_PROPERTY = "category"


# =============================================================================
# Config Instantiation
# =============================================================================


def rslearn_task_type_to_olmoearth_task_type(rslearn_task):
    """Map rslearn Task class to olmoearth TaskType enum."""
    # Note: Adjust as needed to match all possible rslearn task types
    rslearn_name = type(rslearn_task).__name__.lower()
    if "classification" in rslearn_name:
        return TaskType.CLASSIFICATION
    elif "segmentation" in rslearn_name:
        return TaskType.SEGMENTATION
    else:
        # Default/fallback; update if regression is to be supported etc.
        raise ValueError(f"Unknown rslearn task type: {type(rslearn_task)}")


def instantiate_from_config(config: dict) -> Any:
    """Instantiate a class from a class_path + init_args config dict.

    This handles the standard rslearn/LightningCLI config format:
        {
            "class_path": "module.path.ClassName",
            "init_args": {"arg1": value1, ...}
        }

    Args:
        config: Dict with "class_path" and optional "init_args"

    Returns:
        Instantiated object

    Example:
        config = {
            "class_path": "rslearn.train.tasks.segmentation.SegmentationTask",
            "init_args": {"num_classes": 7, "zero_is_invalid": True}
        }
        task = instantiate_from_config(config)
        # Returns SegmentationTask(num_classes=7, zero_is_invalid=True)
    """
    class_path = config["class_path"]
    init_args = config.get("init_args", {})

    # Handle nested configs in init_args (recursive instantiation)
    resolved_args = {}
    for key, value in init_args.items():
        if isinstance(value, dict) and "class_path" in value:
            resolved_args[key] = instantiate_from_config(value)
        else:
            resolved_args[key] = value

    module_path, class_name = class_path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    cls = getattr(module, class_name)
    return cls(**resolved_args)


# =============================================================================
# Normalization Statistics
# =============================================================================

# rslearn layer name -> (olmoearth modality name, all bands)
RSLEARN_TO_OLMOEARTH: dict[str, tuple[str, Modality]] = {
    "sentinel2": Modality.SENTINEL2_L2A,
    "sentinel1": Modality.SENTINEL1,
    "sentinel1_ascending": Modality.SENTINEL1,
    "sentinel1_descending": Modality.SENTINEL1,
    "landsat": Modality.LANDSAT,
}

TASK_TYPE_MAP = {"tolbi_crops": TaskType.SEGMENTATION}

# THis may not be needed if we load stuff from rslearn correctly
TASK_TYPE_TO_RSLEARN_TASK_CLASS = {
    TaskType.SEGMENTATION: RsSegmentationTask,
    TaskType.CLASSIFICATION: RsClassificationTask,
}


@dataclass
class BandStats:
    """Per-band normalization statistics.

    These statistics are computed during dataset ingestion by sampling
    a subset of the data. They're used to normalize inputs during
    evaluation if the user opts not to use pretrain normalization stats.

    Attributes:
        band_name: Name of the band (e.g., "B02", "VV")
        mean: Mean pixel value across sampled data
        std: Standard deviation of pixel values
        min: Minimum observed value
        max: Maximum observed value
        p1: 1st percentile value (for robust min)
        p99: 99th percentile value (for robust max)

    Usage:
        # Normalize using mean/std
        normalized = (pixel - stats.mean) / stats.std

        # Normalize using robust percentiles
        normalized = (pixel - stats.p1) / (stats.p99 - stats.p1)
    """

    band_name: str
    mean: float
    std: float
    min: float
    max: float
    p1: float
    p99: float

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "band_name": self.band_name,
            "mean": self.mean,
            "std": self.std,
            "min": self.min,
            "max": self.max,
            "p1": self.p1,
            "p99": self.p99,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BandStats:
        """Create from dictionary."""
        return cls(**data)


@dataclass
class ModalityStats:
    """Per-modality normalization statistics.

    Contains statistics for all bands within a single modality.
    Also tracks metadata about how the stats were computed.

    Attributes:
        modality: Modality name (e.g., "sentinel2_l2a", "sentinel1")
        bands: Dict mapping band name -> BandStats
        num_samples: Number of samples used to compute stats
        sample_fraction: Fraction of dataset that was sampled
        computed_at: ISO 8601 timestamp of when stats were computed

    Example structure in JSON:
        {
            "modality": "sentinel2_l2a",
            "bands": {
                "B02": {"mean": 1234.5, "std": 456.7, ...},
                "B03": {"mean": 2345.6, "std": 567.8, ...},
                ...
            },
            "num_samples": 5000,
            "sample_fraction": 0.1,
            "computed_at": "2024-01-15T10:30:00Z"
        }
    """

    modality: str
    bands: dict[str, BandStats]
    num_samples: int
    sample_fraction: float
    computed_at: str

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "modality": self.modality,
            "bands": {name: stats.to_dict() for name, stats in self.bands.items()},
            "num_samples": self.num_samples,
            "sample_fraction": self.sample_fraction,
            "computed_at": self.computed_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ModalityStats:
        """Create from dictionary."""
        return cls(
            modality=data["modality"],
            bands={
                name: BandStats.from_dict(stats)
                for name, stats in data["bands"].items()
            },
            num_samples=data["num_samples"],
            sample_fraction=data["sample_fraction"],
            computed_at=data["computed_at"],
        )


# =============================================================================
# Dataset Registry Entry
# =============================================================================


@dataclass
class EvalDatasetEntry:
    """A single entry in the eval dataset registry.

    This represents metadata needed to load and use a dataset for OlmoEarth
    evaluation. Uses a hybrid approach where:
    - Essential task info (num_classes, task_type) is stored here
    - Runtime config (groups, transforms, etc.) is loaded from model.yaml

    Attributes:
        # === Identity ===
        name: Unique identifier (e.g., "lfmc", "tolbi_crops")

        # === Paths (source of truth) ===
        source_path: Path to rslearn dataset (has config.json)
        model_config_path: Path to model.yaml (rslearn training config)

        # === Task Configuration (needed for EvalDatasetConfig) ===
        task_type: One of "classification", "regression", "segmentation"
        num_classes: Number of output classes
        is_multilabel: Whether task is multi-label classification

        # === Modality Configuration ===
        modalities: List of OlmoEarth modality names (e.g., ["sentinel2_l2a"])
        imputes: List of (src_band, tgt_band) tuples for band imputation

        # === Sizing ===
        window_size: Window/patch size (used as height_width for segmentation)
        timeseries: Whether dataset has multiple timesteps

        # === Normalization ===
        norm_stats_path: Path to cached norm stats JSON (relative or absolute)
        use_pretrain_norm: If True, use pretrain normalization stats

        # === Metadata ===
        created_at: ISO 8601 timestamp
        notes: Optional notes

    Design Notes:
    -------------
    - Fields that can be loaded from model.yaml at runtime are NOT stored here
      (e.g., groups, split_tag_key, crop_size, target_layer_name)
    - Use load_runtime_config() from rslearn_builder to get those values
    - This reduces registry bloat and keeps model.yaml as source of truth
    """

    # Identity
    name: str

    # Paths (source of truth for runtime loading)
    source_path: str = ""
    model_config_path: str = ""  # Path to model.yaml

    custom_groups: list[str] = field(default_factory=list)

    # Task configuration (needed for EvalDatasetConfig)
    task_type: str = "classification"  # "classification", "regression", "segmentation"
    num_classes: int | None = None
    is_multilabel: bool = False
    classes: list[str] | None = None  # Optional class names

    # Modality configuration
    modalities: list[str] = field(default_factory=list)
    imputes: list[tuple[str, str]] = field(default_factory=list)

    # Sizing
    window_size: int = 64
    timeseries: bool = False

    # Normalization
    norm_stats_path: str = ""
    use_pretrain_norm: bool = True

    num_timesteps: int = 1


    def __post_init__(self) -> None:
        """Validate and normalize fields after initialization."""
        # Normalize task_type: accept both TaskType enum and string
        if isinstance(self.task_type, TaskType):
            self.task_type = self.task_type.value

        valid_task_types = {t.value for t in TaskType}
        if self.task_type not in valid_task_types:
            raise ValueError(
                f"Invalid task_type '{self.task_type}'. "
                f"Must be one of: {valid_task_types}"
            )

        # Normalize modalities: convert ModalitySpec to lowercase name strings
        #TODO: Can simplify this
        self.modalities = [m.name if isinstance(m, ModalitySpec) else m.lower() if isinstance(m, str) else m for m in self.modalities]

        # Set num_classes from classes if not provided
        if self.classes is not None and self.num_classes is None:
            self.num_classes = len(self.classes)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "name": self.name,
            "source_path": self.source_path,
            "model_config_path": self.model_config_path,
            "custom_groups": self.custom_groups,
            "task_type": self.task_type,
            "num_classes": self.num_classes,
            "is_multilabel": self.is_multilabel,
            "classes": self.classes,
            "modalities": self.modalities,
            "imputes": [list(t) for t in self.imputes],
            "window_size": self.window_size,
            "timeseries": self.timeseries,
            "norm_stats_path": self.norm_stats_path,
            "use_pretrain_norm": self.use_pretrain_norm,
            "num_timesteps": self.num_timesteps,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EvalDatasetEntry":
        """Create from dictionary."""
        # Handle imputes: list of lists -> list of tuples
        imputes = [tuple(x) for x in data.get("imputes", [])]

        # Handle backward compatibility: multilabel -> is_multilabel
        is_multilabel = data.get("is_multilabel", data.get("multilabel", False))

        return cls(
            name=data["name"],
            source_path=data.get("source_path", ""),
            model_config_path=data.get("model_config_path", ""),
            custom_groups=data.get("custom_groups", []),
            task_type=data.get("task_type", "classification"),
            num_classes=data.get("num_classes"),
            is_multilabel=is_multilabel,
            classes=data.get("classes"),
            modalities=data.get("modalities", []),
            imputes=imputes,
            window_size=data.get("window_size", 64),
            timeseries=data.get("timeseries", False),
            norm_stats_path=data.get("norm_stats_path", ""),
            use_pretrain_norm=data.get("use_pretrain_norm", True),
            num_timesteps=data.get("num_timesteps", 1),
        )

    def to_json(self, indent: int | None = 2) -> str:
        """Serialize to JSON string."""
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_json(cls, json_str: str) -> "EvalDatasetEntry":
        """Deserialize from JSON string."""
        return cls.from_dict(json.loads(json_str))

    def to_eval_config(self) -> EvalDatasetConfig:
        """Convert to EvalDatasetConfig for use with eval functions.

        Raises:
            ValueError: If num_classes is not set (required for eval).
        """
        from olmoearth_pretrain.evals.datasets.configs import (
            EvalDatasetConfig,
        )

        if self.num_classes is None:
            raise ValueError(
                f"Cannot convert '{self.name}' to EvalDatasetConfig: num_classes is required"
            )

        # For segmentation, use window_size as height_width
        height_width = self.window_size if self.task_type == "segmentation" else None

        return EvalDatasetConfig(
            task_type=TaskType(self.task_type),
            imputes=self.imputes,
            num_classes=self.num_classes,
            is_multilabel=self.is_multilabel,
            supported_modalities=self.modalities,
            height_width=height_width,
            timeseries=self.timeseries,
        )
