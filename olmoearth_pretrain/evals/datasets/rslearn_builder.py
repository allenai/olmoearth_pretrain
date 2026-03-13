"""Build rslearn datasets for eval using RslearnDataModule via jsonargparse.

Instead of manually instantiating DataInput, Task, SplitConfig, and ModelDataset
individually, we parse the `data` section of model.yaml into a full
RslearnDataModule. This keeps us in sync with rslearn's construction logic
(config merging, dataset setup, etc.) while allowing eval-specific overrides.
"""

from __future__ import annotations

import copy
import logging
import re
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from olmoearth_pretrain.evals.studio_ingest.schema import EvalDatasetEntry

import jsonargparse
import yaml
from rslearn.template_params import substitute_env_vars_in_string
from rslearn.train.data_module import RslearnDataModule
from rslearn.train.dataset import ModelDataset
from rslearn.utils.jsonargparse import init_jsonargparse
from upath import UPath

logger = logging.getLogger(__name__)

_JSONARGPARSE_INITIALIZED = False


def _ensure_jsonargparse() -> None:
    global _JSONARGPARSE_INITIALIZED
    if not _JSONARGPARSE_INITIALIZED:
        init_jsonargparse()
        _JSONARGPARSE_INITIALIZED = True


def parse_model_config(model_config_path: str) -> dict[str, Any]:
    """Load and parse model.yaml, substituting environment variables.

    In this eval builder path, dataset location is passed separately via
    source_path / registry weka_path, so ${DATASET_PATH} in model.yaml
    does not control where the dataset is loaded from.
    """
    model_config_upath = UPath(model_config_path)
    if not model_config_upath.exists():
        raise FileNotFoundError(f"model.yaml not found at {model_config_path}")

    with model_config_upath.open() as f:
        raw_content = f.read()

    if "${DATASET_PATH}" in raw_content:
        logger.warning(
            "model.yaml contains ${DATASET_PATH}, but dataset loading here uses "
            "explicit source_path/weka_path instead."
        )

    substituted_content = substitute_env_vars_in_string(raw_content)

    unresolved_vars = sorted(
        set(re.findall(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}", substituted_content))
    )
    if unresolved_vars:
        logger.warning(
            "Unresolved env vars in model.yaml after substitution: %s",
            unresolved_vars,
        )

    return yaml.safe_load(substituted_content)


def _strip_normalize_transforms(data_config: dict[str, Any]) -> None:
    """Remove Normalize transforms from all split configs in-place.

    Normalization is handled by OlmoEarth's eval pipeline (either pretrained
    stats or per-dataset stats), not by rslearn's transforms. If we left
    rslearn's Normalize in, the data would be double-normalized.
    """
    init_args = data_config.get("init_args", {})
    for config_key in ["default_config", "train_config", "val_config", "test_config"]:
        cfg = init_args.get(config_key, {})
        if "transforms" not in cfg:
            continue
        original_count = len(cfg["transforms"])
        cfg["transforms"] = [
            t for t in cfg["transforms"] if "Normalize" not in t.get("class_path", "")
        ]
        removed = original_count - len(cfg["transforms"])
        if removed:
            logger.info(
                "Stripped %d Normalize transform(s) from %s", removed, config_key
            )


def _apply_split_overrides(
    data_config: dict[str, Any],
    split: str,
    groups_override: list[str] | None,
    tags_override: dict[str, str] | None,
    max_samples: int | None,
) -> None:
    """Apply eval-time split overrides to the data config dict in-place."""
    init_args = data_config.get("init_args", {})
    split_key = f"{split}_config"
    split_cfg = init_args.setdefault(split_key, {})

    if tags_override:
        split_cfg["tags"] = tags_override
        # When filtering by tags, clear groups so rslearn scans all directories
        # first, then filters by tag. Otherwise rslearn only scans the group
        # directories (e.g. windows/train/) which may not exist when all windows
        # live under a single group with tag-based splits.
        if not groups_override:
            split_cfg["groups"] = None
        logger.info(
            "Split override: tags=%s, groups=%s", tags_override, groups_override
        )

    if groups_override:
        split_cfg["groups"] = groups_override
        logger.info("Split override: groups=%s", groups_override)

    if max_samples is not None:
        split_cfg["num_samples"] = max_samples


def _instantiate_data_module(
    model_config: dict[str, Any],
    source_path: str,
    split: str = "val",
    init_workers: int = 32,
    groups_override: list[str] | None = None,
    tags_override: dict[str, str] | None = None,
    max_samples: int | None = None,
) -> RslearnDataModule:
    """Instantiate RslearnDataModule from model.yaml's data section.

    Parses only the ``data`` block via jsonargparse — no model or trainer
    instantiation needed.
    """
    _ensure_jsonargparse()

    data_config = copy.deepcopy(model_config["data"])
    init_args = data_config.setdefault("init_args", {})

    init_args["path"] = source_path
    init_args["index_mode"] = "USE"
    init_args["init_workers"] = init_workers

    _strip_normalize_transforms(data_config)
    _apply_split_overrides(
        data_config, split, groups_override, tags_override, max_samples
    )

    parser = jsonargparse.ArgumentParser()
    parser.add_argument("--data", type=RslearnDataModule)
    parsed = parser.parse_object({"data": data_config})
    return parser.instantiate_classes(parsed).data


def build_model_dataset(
    model_config: dict[str, Any],
    source_path: str,
    split: str = "val",
    init_workers: int = 32,
    max_samples: int | None = None,
    groups_override: list[str] | None = None,
    tags_override: dict[str, str] | None = None,
):
    """Build an rslearn dataset for eval.

    Uses RslearnDataModule's setup() to construct the dataset, which keeps
    us in sync with rslearn's config merging and ModelDataset construction.

    Returns whatever dataset type rslearn produces (map-style or iterable).
    """
    stage_map = {"train": "fit", "val": "validate", "test": "test"}
    stage = stage_map.get(split, "validate")

    data_module = _instantiate_data_module(
        model_config=model_config,
        source_path=source_path,
        split=split,
        init_workers=init_workers,
        groups_override=groups_override,
        tags_override=tags_override,
        max_samples=max_samples,
    )
    data_module.setup(stage)

    dataset = data_module.datasets.get(split)
    if dataset is None:
        available = list(data_module.datasets.keys())
        raise ValueError(
            f"Split '{split}' not found after setup('{stage}'). Available: {available}"
        )

    if hasattr(dataset, "__len__"):
        logger.info("Built dataset for split '%s': %d samples", split, len(dataset))
    else:
        logger.info(
            "Built dataset for split '%s' (iterable: %s)",
            split, type(dataset).__name__,
        )
    return dataset


# ---------------------------------------------------------------------------
# Helpers that read from raw model_config (no instantiation needed)
# ---------------------------------------------------------------------------


def get_task_info(model_config: dict[str, Any]) -> dict[str, Any]:
    """Get task type from model config.

    Returns dict with:
        task_name: For MultiTask, the first sub-task name. None for single tasks.
        task_type: "segmentation", "classification", etc.
    """
    _ensure_jsonargparse()

    data_init_args = model_config.get("data", {}).get("init_args", {})
    task_config = data_init_args.get("task", {})
    if not task_config:
        raise ValueError("No task config found in model.yaml data.init_args.task")

    from rslearn.train.tasks import Task

    parser = jsonargparse.ArgumentParser()
    parser.add_argument("--task", type=Task)

    cfg = parser.parse_object({"task": task_config})
    task = parser.instantiate_classes(cfg).task

    return _classify_task(task)


def _classify_task(task: Any) -> dict[str, Any]:
    """Map a Task instance to task_name and task_type."""
    task_class = type(task).__name__

    if task_class == "MultiTask" and hasattr(task, "tasks") and task.tasks:
        first_name = next(iter(task.tasks.keys()))
        first_task = task.tasks[first_name]
        return {
            "task_name": first_name,
            "task_type": _task_type_from_class(type(first_task).__name__),
        }

    return {"task_name": None, "task_type": _task_type_from_class(task_class)}


def _task_type_from_class(class_name: str) -> str:
    name = class_name.lower()
    for kind in ("segmentation", "classification", "regression"):
        if kind in name:
            return kind
    raise ValueError(f"Cannot determine task type from class name: {class_name}")


def get_modality_layers(model_config: dict[str, Any]) -> list[str]:
    """Get list of modality layer names (non-target inputs) from raw config."""
    data_init_args = model_config.get("data", {}).get("init_args", {})
    inputs_config = data_init_args.get("inputs", {})

    layers = []
    for _name, cfg in inputs_config.items():
        if not cfg.get("is_target"):
            input_layers = cfg.get("layers", [])
            if input_layers:
                layers.append(input_layers[0])
    return layers


# ---------------------------------------------------------------------------
# High-level entry points
# ---------------------------------------------------------------------------


def load_and_build_dataset(
    model_config_path: str,
    source_path: str,
    split: str = "val",
    init_workers: int = 32,
    max_samples: int | None = None,
    groups_override: list[str] | None = None,
    tags_override: dict[str, str] | None = None,
) -> tuple[ModelDataset, dict[str, Any]]:
    """Parse model.yaml and build a ModelDataset for eval.

    Returns:
        Tuple of (ModelDataset, parsed model_config dict).
    """
    model_config = parse_model_config(model_config_path)
    dataset = build_model_dataset(
        model_config=model_config,
        source_path=source_path,
        split=split,
        init_workers=init_workers,
        max_samples=max_samples,
        groups_override=groups_override,
        tags_override=tags_override,
    )
    return dataset, model_config


def build_dataset_from_registry_entry(
    entry: EvalDatasetEntry,
    split: str = "val",
    max_samples: int | None = None,
) -> ModelDataset:
    """Build rslearn ModelDataset from a registry entry.

    Splits are always tag-based: the ingest CLI writes an ``eval_split`` tag
    (or whatever ``entry.split_tag_key`` is) with values ``train``/``val``/``test``.
    """
    tags_override = {entry.split_tag_key: split}
    logger.info("Using tag-based splits: %s=%s", entry.split_tag_key, split)

    model_config = parse_model_config(entry.model_yaml_path)
    return build_model_dataset(
        model_config=model_config,
        source_path=entry.weka_path,
        split=split,
        max_samples=max_samples,
        tags_override=tags_override,
    )
