"""Runtime configuration loading from rslearn model.yaml using jsonargparse.

This module uses rslearn's native jsonargparse infrastructure to parse and
instantiate objects directly from model.yaml, rather than manually extracting
fields. This ensures we stay in sync with rslearn's actual parsing logic.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from olmoearth_pretrain.evals.studio_ingest.schema import EvalDatasetEntry

import yaml
from upath import UPath

logger = logging.getLogger(__name__)


def _init_rslearn_jsonargparse() -> None:
    """Initialize rslearn's custom jsonargparse serializers."""
    from rslearn.utils.jsonargparse import init_jsonargparse

    init_jsonargparse()


# TODO: we should validate this in some way
def parse_model_config(model_config_path: str) -> dict[str, Any]:
    """Load and parse model.yaml, substituting environment variables.

    Args:
        model_config_path: Path to model.yaml file.

    Returns:
        Parsed config dict with env vars substituted.
    """
    from rslearn.template_params import substitute_env_vars_in_string

    model_config_upath = UPath(model_config_path)
    if not model_config_upath.exists():
        raise FileNotFoundError(f"model.yaml not found at {model_config_path}")

    with model_config_upath.open() as f:
        raw_content = f.read()

    # Substitute environment variables like ${DATASET_PATH}
    substituted_content = substitute_env_vars_in_string(raw_content)
    return yaml.safe_load(substituted_content)


def instantiate_data_inputs(model_config: dict[str, Any]) -> dict[str, Any]:
    """Instantiate DataInput objects from model config using jsonargparse.

    This uses rslearn's native parsing to build DataInput objects,
    ensuring we match their exact behavior.

    Args:
        model_config: Parsed model.yaml dict.

    Returns:
        Dict mapping input name -> instantiated DataInput object.
    """
    import inspect

    import jsonargparse
    from rslearn.train.dataset import DataInput

    _init_rslearn_jsonargparse()

    data_init_args = model_config.get("data", {}).get("init_args", {})
    inputs_config = data_init_args.get("inputs", {})

    if not inputs_config:
        return {}

    # Get valid DataInput field names to filter unknown fields
    valid_fields = set(inspect.signature(DataInput.__init__).parameters.keys()) - {
        "self"
    }

    # Use jsonargparse to instantiate each DataInput
    parser = jsonargparse.ArgumentParser()
    parser.add_argument("--data_input", type=DataInput)

    instantiated_inputs = {}
    for name, input_config in inputs_config.items():
        try:
            # Filter to only known fields
            filtered_config = {
                k: v for k, v in input_config.items() if k in valid_fields
            }
            cfg = parser.parse_object({"data_input": filtered_config})
            instantiated_inputs[name] = parser.instantiate_classes(cfg).data_input
            logger.debug(f"Instantiated DataInput '{name}' via jsonargparse")
        except Exception as e:
            logger.warning(f"Failed to instantiate DataInput '{name}': {e}")
            # Fall back to direct instantiation with filtered fields
            try:
                filtered_config = {
                    k: v for k, v in input_config.items() if k in valid_fields
                }
                instantiated_inputs[name] = DataInput(**filtered_config)
                logger.debug(f"Instantiated DataInput '{name}' directly")
            except Exception as e2:
                logger.warning(
                    f"Failed direct instantiation of DataInput '{name}': {e2}"
                )
                instantiated_inputs[name] = input_config

    return instantiated_inputs


def instantiate_task(model_config: dict[str, Any]) -> Any:
    """Instantiate Task object from model config using jsonargparse.

    Args:
        model_config: Parsed model.yaml dict.

    Returns:
        Instantiated Task object.
    """
    import jsonargparse
    from rslearn.train.tasks import Task

    _init_rslearn_jsonargparse()

    data_init_args = model_config.get("data", {}).get("init_args", {})
    task_config = data_init_args.get("task", {})

    if not task_config:
        return None

    parser = jsonargparse.ArgumentParser()
    parser.add_argument("--task", type=Task)

    try:
        cfg = parser.parse_object({"task": task_config})
        return parser.instantiate_classes(cfg).task
    except Exception as e:
        logger.warning(f"Failed to instantiate Task: {e}")
        return None


def instantiate_split_config(
    model_config: dict[str, Any],
    split: str = "val",
) -> Any:
    """Instantiate SplitConfig for a given split.

    Due to transform instantiation issues (some transforms reference packages
    that may not be fully loadable), we instantiate SplitConfig directly
    without transforms, then add transforms separately if needed.

    Args:
        model_config: Parsed model.yaml dict.
        split: Split name ("train", "val", "test", "predict").

    Returns:
        Instantiated SplitConfig object.
    """
    from rslearn.train.dataset import SplitConfig

    data_init_args = model_config.get("data", {}).get("init_args", {})
    default_config = data_init_args.get("default_config", {})
    split_specific = data_init_args.get(f"{split}_config", {})

    # Merge configs manually (split-specific overrides default)
    merged = {**default_config, **split_specific}

    # Extract fields that SplitConfig accepts (without transforms for now)
    # Transforms are complex and may reference packages that aren't loadable
    valid_fields = {
        "groups",
        "names",
        "tags",
        "num_samples",
        "patch_size",
        "overlap_ratio",
        "load_all_patches",
        "skip_targets",
        "output_layer_name_skip_inference_if_exists",
        "sampler",
    }

    filtered = {k: v for k, v in merged.items() if k in valid_fields}

    try:
        split_config = SplitConfig(**filtered)
        logger.debug(
            f"Instantiated SplitConfig for '{split}' with {list(filtered.keys())}"
        )
        return split_config
    except Exception as e:
        logger.warning(f"Failed to instantiate SplitConfig for '{split}': {e}")
        # Return a minimal SplitConfig
        return SplitConfig(
            groups=merged.get("groups"),
            tags=merged.get("tags"),
            num_samples=merged.get("num_samples"),
        )


@dataclass
class RuntimeConfig:
    """Configuration loaded from model.yaml at runtime.

    Contains both raw config values and optionally instantiated objects.
    """

    # Raw model config dict
    model_config: dict[str, Any] = field(default_factory=dict)

    # Instantiated objects (lazily populated)
    _inputs: dict[str, Any] | None = None
    _task: Any | None = None
    _split_configs: dict[str, Any] = field(default_factory=dict)

    # Extracted values for convenience
    source_path: str = ""

    @property
    def inputs(self) -> dict[str, Any]:
        """Get instantiated DataInput objects."""
        if self._inputs is None:
            self._inputs = instantiate_data_inputs(self.model_config)
        return self._inputs

    @property
    def task(self) -> Any:
        """Get instantiated Task object."""
        if self._task is None:
            self._task = instantiate_task(self.model_config)
        return self._task

    def get_split_config(self, split: str) -> Any:
        """Get instantiated SplitConfig for a split."""
        if split not in self._split_configs:
            self._split_configs[split] = instantiate_split_config(
                self.model_config, split
            )
        return self._split_configs[split]

    # Convenience accessors that extract from instantiated objects or raw config

    def get_groups(self) -> list[str]:
        """Get groups from train config."""
        data_init_args = self.model_config.get("data", {}).get("init_args", {})
        train_config = data_init_args.get("train_config", {})
        return train_config.get("groups", [])

    def get_split_tag_key(self) -> str:
        """Get split tag key from train config."""
        data_init_args = self.model_config.get("data", {}).get("init_args", {})
        train_config = data_init_args.get("train_config", {})
        tags = train_config.get("tags", {})
        return next(iter(tags.keys()), "split") if tags else "split"

    def get_target_input_config(self) -> dict[str, Any]:
        """Get the config for the target/label input."""
        data_init_args = self.model_config.get("data", {}).get("init_args", {})
        inputs_config = data_init_args.get("inputs", {})

        for name, cfg in inputs_config.items():
            if cfg.get("is_target"):
                return {
                    "name": name,
                    "layers": cfg.get("layers", []),
                    "data_type": cfg.get("data_type", "vector"),
                    "bands": cfg.get("bands"),
                }
        return {
            "name": "label",
            "layers": ["label"],
            "data_type": "vector",
            "bands": None,
        }

    def get_task_info(self) -> dict[str, Any]:
        """Get task structure info for parsing targets.

        Returns:
            Dict with:
            - task_name: For MultiTask, the first sub-task name (e.g., "segment").
                         None for single tasks.
            - task_type: The task type ("segmentation", "classification", etc.)
        """
        task = self.task
        if task is None:
            return {"task_name": None, "task_type": "segmentation"}

        task_class = type(task).__name__

        # Check if it's a MultiTask
        if task_class == "MultiTask":
            # Get the first sub-task name and type
            if hasattr(task, "tasks") and task.tasks:
                first_name = next(iter(task.tasks.keys()))
                first_task = task.tasks[first_name]
                first_task_class = type(first_task).__name__.lower()

                # Map class name to task type
                if "segmentation" in first_task_class:
                    task_type = "segmentation"
                elif "classification" in first_task_class:
                    task_type = "classification"
                elif "regression" in first_task_class:
                    task_type = "regression"
                else:
                    task_type = "segmentation"  # default

                return {"task_name": first_name, "task_type": task_type}

        # Single task - no task_name needed
        task_class_lower = task_class.lower()
        if "segmentation" in task_class_lower:
            task_type = "segmentation"
        elif "classification" in task_class_lower:
            task_type = "classification"
        elif "regression" in task_class_lower:
            task_type = "regression"
        else:
            task_type = "segmentation"  # default

        return {"task_name": None, "task_type": task_type}

    def get_modality_layers(self) -> list[str]:
        """Get list of modality layer names (non-target inputs)."""
        data_init_args = self.model_config.get("data", {}).get("init_args", {})
        inputs_config = data_init_args.get("inputs", {})

        layers = []
        for name, cfg in inputs_config.items():
            if not cfg.get("is_target"):
                input_layers = cfg.get("layers", [])
                if input_layers:
                    layers.append(input_layers[0])
        return layers

    def get_crop_size(self, split: str = "val") -> int | None:
        """Extract crop size from transforms for a split."""
        data_init_args = self.model_config.get("data", {}).get("init_args", {})
        config = data_init_args.get(f"{split}_config", {})
        if not config:
            config = data_init_args.get("default_config", {})

        transforms = config.get("transforms", [])
        for transform in transforms:
            class_path = transform.get("class_path", "")
            if "crop.Crop" in class_path:
                crop_size = transform.get("init_args", {}).get("crop_size")
                if isinstance(crop_size, list | tuple):
                    return crop_size[0]
                return crop_size
        return None


def load_runtime_config(
    model_config_path: str,
    source_path: str = "",
) -> RuntimeConfig:
    """Load RuntimeConfig from model.yaml using rslearn's jsonargparse.

    Args:
        model_config_path: Path to model.yaml file.
        source_path: Path to rslearn dataset (stored for reference).

    Returns:
        RuntimeConfig with parsed model config.

    Raises:
        FileNotFoundError: If model.yaml does not exist.
        Exception: If model.yaml cannot be parsed.
    """
    model_config = parse_model_config(model_config_path)

    return RuntimeConfig(
        model_config=model_config,
        source_path=source_path,
    )


def build_model_dataset_from_config(
    runtime_config: RuntimeConfig,
    source_path: str,
    split: str = "val",
    max_samples: int | None = None,
    groups_override: list[str] | None = None,
    tags_override: dict[str, str] | None = None,
) -> Any:
    """Build rslearn ModelDataset directly from RuntimeConfig.

    This uses the jsonargparse-instantiated objects (inputs, task, split_config)
    to build the ModelDataset, avoiding manual construction.

    Args:
        runtime_config: RuntimeConfig with parsed model.yaml.
        source_path: Path to rslearn dataset.
        split: Dataset split ("train", "val", "test").
        max_samples: Optional limit on number of samples.
        groups_override: Optional list of groups to use instead of model.yaml groups.
        tags_override: Optional dict of tags to filter windows (e.g., {"eval_split": "val"}).

    Returns:
        Instantiated ModelDataset.
    """
    from rslearn.dataset.dataset import Dataset as RslearnDataset
    from rslearn.train.dataset import IndexMode, ModelDataset
    from upath import UPath

    # Get instantiated objects from RuntimeConfig
    inputs = runtime_config.inputs
    task = runtime_config.task
    split_config = runtime_config.get_split_config(split)

    if not inputs:
        raise ValueError("No inputs found in runtime config")
    if task is None:
        raise ValueError("No task found in runtime config")
    if split_config is None:
        raise ValueError(f"No split config found for split '{split}'")

    # Apply groups override if provided
    if groups_override:
        split_config.groups = groups_override
        logger.info(f"Using custom groups override: {groups_override}")

    # Apply tags override if provided
    # When filtering by tags, clear groups so rslearn scans all directories
    # first, then filters by tag. Otherwise rslearn only scans the group
    # directories (e.g. windows/train/) which may not exist when all windows
    # live under a single group (e.g. windows/val/) with tag-based splits.
    if tags_override:
        split_config.tags = tags_override
        if not groups_override:
            split_config.groups = None
        logger.info(
            f"Using custom tags override: {tags_override} (groups={split_config.groups})"
        )

    # Apply max_samples override if provided
    if max_samples is not None and hasattr(split_config, "num_samples"):
        split_config.num_samples = max_samples

    # Build the rslearn dataset
    rslearn_dataset = RslearnDataset(UPath(source_path))

    # Build ModelDataset with instantiated objects
    return ModelDataset(
        dataset=rslearn_dataset,
        split_config=split_config,
        inputs=inputs,
        task=task,
        index_mode=IndexMode.USE,
        workers=32,  # TODO: this should be configurable somewhere
    )


def build_dataset_from_registry_entry(
    entry: EvalDatasetEntry,
    split: str = "val",
    max_samples: int | None = None,
) -> Any:
    """Build rslearn ModelDataset from registry entry using jsonargparse.

    This is the main entry point for building datasets from registry entries
    using the hybrid approach. Uses the split tags written during ingestion
    to filter windows.

    Args:
        entry: EvalDatasetEntry from registry.
        split: Dataset split ("train", "val", "test").
        max_samples: Optional sample limit.

    Returns:
        ModelDataset instance.
    """
    runtime_config = load_runtime_config(entry.model_yaml_path, entry.weka_path)

    # Resolve splits based on split_type from ingestion.
    split_value_map = {
        "train": entry.train_split,
        "val": entry.val_split,
        "test": entry.test_split,
    }
    split_value = split_value_map.get(split, split)

    tags_override = None
    groups_override = None

    if entry.split_type == "tags" and entry.split_tag_key:
        # Tag-based: scan all group dirs, filter by tag
        tags_override = {entry.split_tag_key: split_value}
        logger.info(f"Using tag-based splits: {entry.split_tag_key}={split_value}")
    elif entry.split_type == "groups":
        # Group-based: use split name as the group directory
        groups_override = [split_value]
        logger.info(f"Using group-based splits: groups={groups_override}")

    return build_model_dataset_from_config(
        runtime_config=runtime_config,
        source_path=entry.weka_path,
        split=split,
        max_samples=max_samples,
        groups_override=groups_override,
        tags_override=tags_override,
    )
