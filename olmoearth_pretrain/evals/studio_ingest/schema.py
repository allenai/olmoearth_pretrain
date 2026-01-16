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
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from rslearn.train.tasks.classification import (
    ClassificationTask as RsClassificationTask,
)
from rslearn.train.tasks.segmentation import SegmentationTask as RsSegmentationTask

from olmoearth_pretrain.data.constants import Modality
from olmoearth_pretrain.evals.datasets.configs import TaskType

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

    This represents all metadata needed to load and use a dataset
    for OlmoEarth evaluation. Entries are stored in registry.json
    and also copied to each dataset's metadata.json.

    Attributes:
        # === Identity ===
        name: Unique identifier (e.g., "lfmc", "forest_loss_driver")
              Used as the directory name on Weka and for loading.

        # === Task Configuration ===
        task_type: One of "classification", "regression", "segmentation"
        classes: List of class names for classification tasks, None otherwise
        num_classes: Number of classes, derived from len(classes) if applicable
        target_property: The rslearn property name that contains the labels
                        (e.g., "category", "lfmc_value")

        # === Data Configuration ===
        modalities: List of OlmoEarth modality names (e.g., ["sentinel2_l2a"])
        temporal_range: Tuple of (start_date, end_date) as ISO strings
        patch_size: Size of image patches in pixels (e.g., 64, 128)

        # === Paths ===
        source_path: Original rslearn dataset path (GCS) - for provenance
        weka_path: Where the data lives on Weka after ingestion
                   Note: For external users, OLMOEARTH_EVAL_DATASETS env var
                   can override the base path to point to local downloads.

        # === Split Information ===
        splits: Dict mapping split name -> sample count
               e.g., {"train": 5000, "val": 500, "test": 1000}
        supports_cv: Whether k-fold cross-validation is supported
        cv_folds: Number of pre-computed CV folds, if any

        # === Normalization ===
        norm_stats_path: Path to norm_stats.json relative to weka_path
        use_pretrain_norm: If True, use pretrain normalization stats
                          instead of dataset-specific stats

        # === Metadata ===
        created_at: ISO 8601 timestamp of when dataset was ingested
        created_by: Username/identifier of who ran the ingestion
        studio_task_id: Optional link back to Studio task ID
        notes: Optional free-form notes about the dataset

    Design Notes:
    -------------
    - The `name` field is the primary key - must be unique across registry
    - Paths are stored as strings for JSON compatibility
    - We store both source_path (provenance) and weka_path (actual data)
    - Normalization stats are separate from this entry (in norm_stats.json)
      but we reference the path here

    Example JSON:
    -------------
    {
        "name": "lfmc",
        "display_name": "Live Fuel Moisture Content",
        "task_type": "regression",
        "classes": null,
        "num_classes": null,
        "target_property": "lfmc_value",
        "modalities": ["sentinel2_l2a", "sentinel1"],
        "temporal_range": ["2022-09-01", "2023-09-01"],
        "patch_size": 64,
        "source_path": "gs://studio-bucket/datasets/lfmc",
        "weka_path": "weka://dfive-default/olmoearth/eval_datasets/lfmc",
        "splits": {"train": 5000, "val": 500, "test": 1000},
        "supports_cv": true,
        "cv_folds": 5,
        "norm_stats_path": "norm_stats.json",
        "use_pretrain_norm": false,
        "created_at": "2024-01-15T10:30:00Z",
        "created_by": "henryh",
        "studio_task_id": "task_abc123",
        "notes": "Initial ingestion from Studio"
    }
    """

    # Identity
    name: str

    # Task configuration
    task_type: str  # "classification", "regression", "segmentation"
    target_property: str
    classes: list[str] | None = None
    num_classes: int | None = None

    # Data configuration
    modalities: list[str] = field(default_factory=list)
    temporal_range: tuple[str, str] = ("", "")
    window_size: int = 64
    multilabel: bool = False
    classes: list[str] | None = None  # num classes can be derived from this
    label_target_property: str | None = None
    band_order: list[str] = field(default_factory=list)
    imputes: list[tuple[str, str]] = field(default_factory=list)
    timeseries: bool = False

    # Paths
    source_path: str = ""
    weka_path: str = ""

    # Split information
    splits: dict[str, int] = field(default_factory=dict)
    supports_cv: bool = False
    cv_folds: int | None = None

    # Normalization
    norm_stats_path: str = "norm_stats.json"
    use_pretrain_norm: bool = False

    # Metadata
    created_at: str = ""
    created_by: str = ""
    studio_task_id: str | None = None
    notes: str | None = None

    def __post_init__(self) -> None:
        """Validate and set derived fields after initialization."""

        # Validate task type against enum values
        valid_task_types = {t for t in TaskType}
        if self.task_type not in valid_task_types:
            raise ValueError(
                f"Invalid task_type '{self.task_type}'. "
                f"Must be one of: {valid_task_types} got {self.task_type}"
            )

        # Set num_classes from classes if not provided
        if self.classes is not None and self.num_classes is None:
            self.num_classes = len(self.classes)

        # Set created_at if not provided
        if not self.created_at:
            self.created_at = datetime.now().isoformat()

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "name": self.name,
            "task_type": self.task_type,
            "target_property": self.target_property,
            "classes": self.classes,
            "num_classes": self.num_classes,
            "modalities": self.modalities,
            "temporal_range": list(self.temporal_range),
            "source_path": self.source_path,
            "weka_path": self.weka_path,
            "splits": self.splits,
            "supports_cv": self.supports_cv,
            "cv_folds": self.cv_folds,
            "norm_stats_path": self.norm_stats_path,
            "use_pretrain_norm": self.use_pretrain_norm,
            "created_at": self.created_at,
            "created_by": self.created_by,
            "studio_task_id": self.studio_task_id,
            "notes": self.notes,
            "imputes": self.imputes,
            "timeseries": self.timeseries,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EvalDatasetEntry:
        """Create from dictionary."""
        # Handle temporal_range which may be list in JSON
        temporal_range = data.get("temporal_range", ("", ""))
        if isinstance(temporal_range, list):
            temporal_range = tuple(temporal_range)

        return cls(
            name=data["name"],
            task_type=data["task_type"],
            target_property=data["target_property"],
            classes=data.get("classes"),
            num_classes=data.get("num_classes"),
            modalities=data.get("modalities", []),
            temporal_range=temporal_range,
            patch_size=data.get("patch_size", 64),
            source_path=data.get("source_path", ""),
            weka_path=data.get("weka_path", ""),
            splits=data.get("splits", {}),
            supports_cv=data.get("supports_cv", False),
            cv_folds=data.get("cv_folds"),
            norm_stats_path=data.get("norm_stats_path", "norm_stats.json"),
            use_pretrain_norm=data.get("use_pretrain_norm", False),
            created_at=data.get("created_at", ""),
            created_by=data.get("created_by", ""),
            studio_task_id=data.get("studio_task_id"),
            notes=data.get("notes"),
            imputes=[tuple(x) for x in data.get("imputes", [])],
            timeseries=data.get("timeseries", False),
        )

    def get_weka_data_path(self, split: str) -> str:
        """Get the full path to a specific split's data on Weka.

        Args:
            split: One of "train", "val", "test"

        Returns:
            Full Weka path to the split directory
        """
        if split not in self.splits:
            raise ValueError(
                f"Unknown split '{split}'. Available: {list(self.splits.keys())}"
            )
        return f"{self.weka_path}/{split}"

    def get_norm_stats_full_path(self) -> str:
        """Get the full path to the normalization stats JSON."""
        return f"{self.weka_path}/{self.norm_stats_path}"

    def to_eval_config(self) -> "EvalDatasetConfig":
        """Convert to EvalDatasetConfig for use with eval functions.

        Raises:
            ValueError: If num_classes is not set (required for eval).
        """
        from olmoearth_pretrain.evals.datasets.configs import EvalDatasetConfig, TaskType

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
            is_multilabel=self.multilabel,
            supported_modalities=self.modalities,
            height_width=height_width,
            timeseries=self.timeseries,
        )
