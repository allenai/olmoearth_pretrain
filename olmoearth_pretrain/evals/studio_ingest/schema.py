"""Schema definitions for the eval dataset registry.

This module defines schema models that represent dataset registry entries
(EvalDatasetEntry), serialized to JSON and stored on Weka alongside the dataset.

Design Decisions:
-----------------
- We use pydantic models for validation and serialization
- All fields are explicitly typed for clarity
- Optional fields use None as default
- Timestamps are ISO 8601 strings for human readability
- Paths are stored as strings (not UPath) for JSON compatibility

Future Considerations:
---------------------
- May want to add versioning to schema for backwards compatibility
- May want to support additional task types (detection, etc.)
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

from pydantic import (
    AliasChoices,
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

if TYPE_CHECKING:
    from olmoearth_pretrain.evals.datasets.configs import EvalDatasetConfig

from olmoearth_pretrain.data.constants import ModalitySpec
from olmoearth_pretrain.evals.constants import RSLEARN_TO_OLMOEARTH
from olmoearth_pretrain.evals.task_types import SplitName, SplitType, TaskType

# =============================================================================
# Config Instantiation
# =============================================================================


def rslearn_task_type_to_olmoearth_task_type(rslearn_task: Any) -> TaskType:
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
            stripped = layer_name[len(prefix) :]
            if stripped in RSLEARN_TO_OLMOEARTH:
                return RSLEARN_TO_OLMOEARTH[stripped]

    raise KeyError(f"Unknown rslearn layer name: {layer_name!r}")


class EvalDatasetEntry(BaseModel):
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
    - Use parse_model_config() from rslearn_builder to get those values
    - This reduces registry bloat and keeps model.yaml as source of truth
    """

    # Identity
    model_config = ConfigDict(extra="ignore", use_enum_values=True)

    name: str

    # Paths (source of truth for runtime loading)
    source_path: str  # Original source path (e.g., GCS)
    weka_path: str  # Copied dataset path on Weka

    # Task configuration (needed for EvalDatasetConfig)
    task_type: str
    num_classes: int | None = None
    is_multilabel: bool = Field(
        default=False,
        validation_alias=AliasChoices("is_multilabel", "multilabel"),
    )
    classes: list[str] | None = None  # Optional class names
    # Split configuration
    train_split: str = "train"
    val_split: str = "val"
    test_split: str = "test"
    split_type: str = "tags"  # "groups" or "tags"
    split_tag_key: str = "eval_split"  # Tag key used for split filtering
    split_stats: dict[str, dict[str, Any]] = Field(default_factory=dict)

    # Modality configuration
    modalities: list[str] = Field(default_factory=list)
    imputes: list[tuple[str, str]] = Field(default_factory=list)

    # Sizing
    window_size: int | None = None
    timeseries: bool = False

    # Normalization
    norm_stats: dict[str, Any] = Field(default_factory=dict)
    use_pretrain_norm: bool = True

    num_timesteps: int = 1

    @field_validator("task_type", mode="before")
    @classmethod
    def _normalize_task_type(cls, value: str | TaskType) -> str:
        if isinstance(value, TaskType):
            return value.value
        return value

    @field_validator("task_type")
    @classmethod
    def _validate_task_type(cls, value: str) -> str:
        valid_task_types = {t.value for t in TaskType}
        if value not in valid_task_types:
            raise ValueError(
                f"Invalid task_type '{value}'. Must be one of: {valid_task_types}"
            )
        return value

    @field_validator("split_type", mode="before")
    @classmethod
    def _normalize_split_type(cls, value: str | SplitType) -> str:
        if isinstance(value, SplitType):
            return value.value
        return value

    @field_validator("train_split", "val_split", "test_split", mode="before")
    @classmethod
    def _normalize_split_name(cls, value: str | SplitName) -> str:
        if isinstance(value, SplitName):
            return value.value
        return value

    @field_validator("modalities", mode="before")
    @classmethod
    def _normalize_modalities(cls, value: list[Any]) -> list[str]:
        return [
            modality.name
            if isinstance(modality, ModalitySpec)
            else modality.lower()
            if isinstance(modality, str)
            else modality
            for modality in value
        ]

    @model_validator(mode="after")
    def _set_num_classes_from_classes(self) -> EvalDatasetEntry:
        if self.classes is not None and self.num_classes is None:
            self.num_classes = len(self.classes)
        return self

    @property
    def model_yaml_path(self) -> str:
        """Get the path to the model.yaml file."""
        return f"{self.weka_path}/model.yaml"

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
