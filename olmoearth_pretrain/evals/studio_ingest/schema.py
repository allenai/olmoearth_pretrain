"""Schema definitions for the eval dataset registry.

This module defines the dataclasses that represent dataset registry entries
(EvalDatasetEntry), serialized to JSON and stored on Weka alongside the dataset.

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

from olmoearth_pretrain.data.constants import ModalitySpec
from olmoearth_pretrain.evals.constants import RSLEARN_TO_OLMOEARTH
from olmoearth_pretrain.evals.task_types import SplitName, SplitType, TaskType


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


def rslearn_to_olmoearth(layer_name: str) -> ModalitySpec:
    """Map an rslearn layer name to an OlmoEarth ModalitySpec.

    Uses RSLEARN_TO_OLMOEARTH from rslearn_dataset as the single source of truth.
    Also handles layer names prefixed with "pre_" or "post_" (e.g.
    "pre_sentinel2" -> Modality.SENTINEL2_L2A).
    """
    if layer_name in RSLEARN_TO_OLMOEARTH:
        return RSLEARN_TO_OLMOEARTH[layer_name]

    for prefix in ("pre_", "post_"):
        if layer_name.startswith(prefix):
            stripped = layer_name[len(prefix):]
            if stripped in RSLEARN_TO_OLMOEARTH:
                return RSLEARN_TO_OLMOEARTH[stripped]

    raise KeyError(f"Unknown rslearn layer name: {layer_name!r}")


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
        norm_stats: Per-band normalization statistics dict
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
    source_path: str  # Original source path (e.g., GCS)
    weka_path: str  # Copied dataset path on Weka

    # Task configuration (needed for EvalDatasetConfig)
    task_type: str
    num_classes: int | None = None
    is_multilabel: bool = False
    classes: list[str] | None = None  # Optional class names
    # Split configuration
    train_split: str = "train"
    val_split: str = "val"
    test_split: str = "test"
    split_type: str = "tags"  # "groups" or "tags"
    split_tag_key: str = "eval_split"  # Tag key used for split filtering
    split_stats: dict[str, dict[str, Any]] = field(default_factory=dict)  # Per-split sample counts


    # Modality configuration
    modalities: list[str] = field(default_factory=list)
    imputes: list[tuple[str, str]] = field(default_factory=list)

    # Sizing
    window_size: int = 64 # TOD: we should be reading this in
    timeseries: bool = False

    # Normalization
    norm_stats: dict[str, Any] = field(default_factory=dict)
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

        # Normalize split_type
        if isinstance(self.split_type, SplitType):
            self.split_type = self.split_type.value

        # Normalize split names
        if isinstance(self.train_split, SplitName):
            self.train_split = self.train_split.value
        if isinstance(self.val_split, SplitName):
            self.val_split = self.val_split.value
        if isinstance(self.test_split, SplitName):
            self.test_split = self.test_split.value

        # Normalize modalities: convert ModalitySpec to lowercase name strings
        self.modalities = [
            m.name if isinstance(m, ModalitySpec) else m.lower() if isinstance(m, str) else m
            for m in self.modalities
        ]

        # Set num_classes from classes if not provided
        if self.classes is not None and self.num_classes is None:
            self.num_classes = len(self.classes)

    @property
    def model_yaml_path(self) -> str:
        """Get the path to the model.yaml file."""
        return f"{self.weka_path}/model.yaml"

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "name": self.name,
            "source_path": self.source_path,
            "weka_path": self.weka_path,
            "task_type": self.task_type,
            "num_classes": self.num_classes,
            "is_multilabel": self.is_multilabel,
            "classes": self.classes,
            "modalities": self.modalities,
            "imputes": [list(t) for t in self.imputes],
            "window_size": self.window_size,
            "timeseries": self.timeseries,
            "num_timesteps": self.num_timesteps,
            "train_split": self.train_split,
            "val_split": self.val_split,
            "test_split": self.test_split,
            "split_type": self.split_type,
            "split_tag_key": self.split_tag_key,
            "split_stats": self.split_stats,
            "norm_stats": self.norm_stats,
            "use_pretrain_norm": self.use_pretrain_norm,
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
            weka_path=data.get("weka_path", ""),
            task_type=data.get("task_type", TaskType.CLASSIFICATION),
            num_classes=data.get("num_classes"),
            is_multilabel=is_multilabel,
            classes=data.get("classes"),
            modalities=data.get("modalities", []),
            imputes=imputes,
            window_size=data.get("window_size", 64),
            timeseries=data.get("timeseries", False),
            num_timesteps=data.get("num_timesteps", 1),
            train_split=data.get("train_split", SplitName.TRAIN),
            val_split=data.get("val_split", SplitName.VAL),
            test_split=data.get("test_split", SplitName.TEST),
            split_type=data.get("split_type", SplitType.GROUPS),
            split_tag_key=data.get("split_tag_key", "split"),
            split_stats=data.get("split_stats", {}),
            norm_stats=data.get("norm_stats", {}),
            use_pretrain_norm=data.get("use_pretrain_norm", True),
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
