"""Main ingestion logic for Studio datasets.

This module provides the core functionality to ingest an rslearn dataset
from Studio/GCS into the OlmoEarth eval system on Weka.

Ingestion Flow:
--------------
1. Create the destination directory on Weka
2. Copy data from source to destination
3. Compute normalization statistics
4. Create metadata.json in the dataset directory
5. Register the dataset in the central registry

Design Decisions:
----------------
- We copy data rather than reference it, for:
  - Faster access (Weka is faster than GCS for our workloads)
  - Immutability (source can change, our copy won't)
  - Provenance (we record where it came from)

- We preserve rslearn structure in the copy, so existing loaders work

- We compute normalization stats during ingestion, not on-demand, because:
  - Stats are computed once and reused many times
  - Ingestion is a good time to catch data issues
  - Avoids recomputation overhead during evaluation

Rollback Handling:
-----------------
If ingestion fails partway through:
- We don't register incomplete datasets
- Partial data on Weka should be cleaned up manually
- TODO: Add automatic cleanup on failure

Todo:
-----
- [ ] Add progress bar for copy operation
- [ ] Add resumable copying for large datasets
- [ ] Add automatic cleanup on failure
- [ ] Add dry-run mode
- [ ] Support incremental updates (add new samples to existing dataset)
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from rslearn.config import DatasetConfig
from upath import UPath

from olmoearth_pretrain.evals.datasets.rslearn_builder import parse_model_config
from olmoearth_pretrain.evals.studio_ingest import paths
from olmoearth_pretrain.evals.studio_ingest.band_stats import (
    compute_band_stats_from_model_config,
)
from olmoearth_pretrain.evals.studio_ingest.copying import (
    copy_dataset,
    prepare_copied_dataset_config,
)
from olmoearth_pretrain.evals.studio_ingest.schema import (
    EvalDatasetEntry,
    instantiate_from_config,
    rslearn_task_type_to_olmoearth_task_type,
    rslearn_to_olmoearth,
)
from olmoearth_pretrain.evals.studio_ingest.splits import (
    EVAL_SPLIT_TAG_KEY,
    count_split_stats,
    create_missing_splits,
    scan_windows_and_splits,
    write_split_tags,
)
from olmoearth_pretrain.evals.task_types import SplitName, TaskType

logger = logging.getLogger(__name__)


def _infer_window_size(dataset_path: str) -> int | None:
    """Read a single window's metadata.json to get its spatial size.

    Walks the windows/ directory and reads the first metadata.json found,
    avoiding loading the full dataset or all windows.
    """
    windows_dir = UPath(dataset_path) / "windows"
    if not windows_dir.exists():
        return None
    for group_dir in windows_dir.iterdir():
        if not group_dir.is_dir():
            continue
        for window_dir in group_dir.iterdir():
            metadata_path = window_dir / "metadata.json"
            if not metadata_path.exists():
                continue
            with metadata_path.open() as f:
                metadata = json.load(f)
            bounds = metadata.get("bounds")
            if bounds and len(bounds) >= 4:
                return bounds[3] - bounds[1]
    return None


# =============================================================================
# Configuration
# =============================================================================

# Environment variable for dataset root path
# This allows external users to download datasets from HF and point to their local copy
EVAL_DATASETS_ENV_VAR = paths.EVAL_DATASETS_ENV_VAR

# Default base path on Weka where all eval datasets are stored (internal)
DEFAULT_WEKA_BASE_PATH = paths.DEFAULT_WEKA_BASE_PATH


def get_eval_datasets_base_path() -> str:
    """Get the base path for eval datasets.

    Checks OLMOEARTH_EVAL_DATASETS env var first, falls back to Weka path.
    This allows external users to download datasets from HF and use them locally.

    Returns:
        Base path string (either from env var or default Weka path)
    """
    return paths.get_eval_datasets_base_path()


# =============================================================================
# Ingestion Config
# =============================================================================


@dataclass
class IngestConfig:
    """Configuration for dataset ingestion.

    This captures all the parameters needed to ingest a dataset.
    It's used by the CLI and can be serialized for reproducibility.

    Attributes:
        # Required
        name: Unique identifier for the dataset
        source_path: Path to the source rslearn dataset
        olmoearth_run_config_path: Path to model.yaml config

        # Source filtering
        source_groups: Groups to pull from source dataset
        source_tags: Tags to filter source windows

        # Split configuration
        val_test_split_ratio: Ratio when splitting val into val+test (default 0.5)
        train_val_split_ratio: Ratio when splitting train into train+val (default 0.8)
        split_seed: Random seed for reproducible splits

        # Normalization
        num_samples: Number of samples for stats computation (default 50k)

        # Archive handling
        untar_source: If True, source_path points to a .tar.gz archive on GCS
            that will be streamed and extracted directly to the destination.
    """

    # Required
    name: str
    source_path: str
    olmoearth_run_config_path: str

    # Source filtering
    source_groups: list[str] | None = None
    source_tags: dict[str, str] | None = None

    # Split configuration
    val_test_split_ratio: float = 0.5
    train_val_split_ratio: float = 0.8
    split_seed: int = 42

    # Normalization
    num_samples: int | None = 50_000

    # Archive handling
    untar_source: bool = False


@dataclass(frozen=True)
class _ModalityMetadata:
    modalities: list[str]
    timeseries: bool
    num_timesteps: int


@dataclass(frozen=True)
class _TaskMetadata:
    task_type: TaskType
    num_classes: int
    label_values: list[str]


def _load_dataset_config(dataset_path: str) -> DatasetConfig:
    """Load and patch rslearn dataset config from the copied dataset path."""
    config_json_path = UPath(dataset_path) / "config.json"
    with config_json_path.open() as f:
        dataset_dict = json.load(f)

    # Strip the "output" layer before pydantic parsing — its deprecated format
    # field ({'name': 'geojson'}) triggers a pydantic ValidationError.
    # The output layer has no data_source and is unused during ingest.
    if "layers" in dataset_dict and "output" in dataset_dict["layers"]:
        del dataset_dict["layers"]["output"]
        logger.info("[Step 0a] Stripped 'output' layer (not needed for ingest)")
        with config_json_path.open("w") as f:
            json.dump(dataset_dict, f, indent=2)
        logger.info("[Step 0a] Wrote patched config.json back to disk")

    return DatasetConfig.model_validate(dataset_dict)


def _load_model_config(dataset_path: str) -> tuple[dict[str, Any], Path]:
    """Load model.yaml from the copied dataset path and validate rslearn parsing."""
    model_yaml_path = Path(dataset_path) / "model.yaml"
    with open(model_yaml_path) as f:
        model_config = yaml.safe_load(f)
    parse_model_config(str(model_yaml_path))
    return model_config, model_yaml_path


def _extract_modality_metadata(dataset_config: DatasetConfig) -> _ModalityMetadata:
    """Extract OlmoEarth modality names and time-series metadata."""
    modalities = []
    max_timesteps_modalities = []

    for layer_name, layer_config in dataset_config.layers.items():
        if layer_config.data_source is None:
            continue
        try:
            olmoearth_modality = rslearn_to_olmoearth(layer_name)
        except KeyError:
            logger.warning(
                f"  Skipping layer {layer_name!r}: no OlmoEarth modality mapping"
            )
            continue
        modalities.append(olmoearth_modality.name)
        query_config = layer_config.data_source.query_config
        max_timesteps_modalities.append(query_config.max_matches)

    num_timesteps = max(max_timesteps_modalities) if max_timesteps_modalities else 1
    return _ModalityMetadata(
        modalities=modalities,
        timeseries=num_timesteps > 1,
        num_timesteps=num_timesteps,
    )


def _select_task_config(model_config: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    """Select the single rslearn task config supported by Studio ingest."""
    task_wrapper_config = model_config["data"]["init_args"]["task"]
    task_init_args = task_wrapper_config.get("init_args", {})

    if "tasks" not in task_init_args:
        return "task", task_wrapper_config

    tasks_dict = task_init_args["tasks"]
    if len(tasks_dict) != 1:
        raise NotImplementedError(
            "Multiple tasks not supported in this workflow; found: "
            + ", ".join(tasks_dict)
        )
    return next(iter(tasks_dict.items()))


def _num_classes_from_task_config(
    task_name: str,
    task_config: dict[str, Any],
) -> int:
    """Extract class count from supported rslearn task config shapes."""
    task_init_args = task_config.get("init_args", {})
    task_class_path = task_config.get("class_path", "")

    if "num_classes" in task_init_args:
        return task_init_args["num_classes"]
    if "classes" in task_init_args:
        return len(task_init_args["classes"])

    raise ValueError(
        f"Could not determine num_classes from task config '{task_name}' "
        f"(class_path: {task_class_path}). "
        "Expected 'num_classes' or 'classes' in init_args."
    )


def _extract_task_metadata(model_config: dict[str, Any]) -> _TaskMetadata:
    """Instantiate the configured rslearn task and extract class metadata."""
    task_name, task_config = _select_task_config(model_config)

    logger.info(
        f"[Step 0d] Instantiating task '{task_name}': {task_config['class_path']}"
    )
    rslearn_task = instantiate_from_config(task_config)

    num_classes = _num_classes_from_task_config(task_name, task_config)
    return _TaskMetadata(
        task_type=rslearn_task_type_to_olmoearth_task_type(rslearn_task),
        num_classes=num_classes,
        label_values=[str(i) for i in range(num_classes)],
    )


def _extract_window_size(model_config: dict[str, Any], dataset_path: str) -> int | None:
    """Extract crop size from model config, falling back to actual window metadata."""
    data_init_args = model_config["data"]["init_args"]
    default_config = data_init_args.get("default_config", {})
    window_size = default_config.get("crop_size")

    if window_size is None:
        window_size = _infer_window_size(dataset_path)
        if window_size is not None:
            logger.info(
                "No crop_size in config, inferred window_size=%d from data",
                window_size,
            )
        else:
            logger.warning("No windows found to infer window_size, leaving as None")

    return window_size


# =============================================================================
# Main Ingestion Function
# =============================================================================


def ingest_dataset(config: IngestConfig) -> EvalDatasetEntry:
    """Ingest a dataset from Studio/GCS into the OlmoEarth eval system.

    This is the main entry point for dataset ingestion. It runs all steps
    in order and returns the created registry entry.

    Args:
        config: Ingestion configuration

    Returns:
        The EvalDatasetEntry for the ingested dataset

    Raises:
        ValueError: If validation fails
        FileExistsError: If destination exists and overwrite=False
        Exception: If any step fails

    Example:
        config = IngestConfig(
            name="lfmc",
            display_name="Live Fuel Moisture Content",
            source_path="gs://bucket/lfmc",
            task_type="regression",
            modalities=["sentinel2_l2a", "sentinel1"],
            target_property="lfmc_value",
        )

        entry = ingest_dataset(config)
        print(f"Ingested {entry.name} with {sum(entry.splits.values())} samples")
    """
    logger.info(f"{'=' * 60}")
    logger.info(f"INGEST START: {config.name}")
    logger.info(f"{'=' * 60}")
    logger.info(f"Source: {config.source_path}")
    logger.info(f"Model config: {config.olmoearth_run_config_path}")

    # Step 1: Copy dataset to Weka
    logger.info("[Step 1/6] Copying dataset to Weka...")
    weka_path = copy_dataset(
        config.source_path,
        config.name,
        config.source_groups,
        config.source_tags,
        config.untar_source,
    )
    logger.info(f"[Step 1/6] Copy complete: {weka_path}")

    # Ensure config.json and model.yaml are canonically available from the
    # copied dataset, independent of the original source location.
    prepare_copied_dataset_config(weka_path, config.olmoearth_run_config_path)

    # Step 0a: Load dataset config from the dataset folder
    logger.info("[Step 0a] Loading dataset config...")
    dataset_config = _load_dataset_config(weka_path)
    logger.info("[Step 0a] Dataset config loaded successfully")

    # Step 0b: Load and validate model config from the canonical weka location
    logger.info("[Step 0b] Loading and validating model.yaml with rslearn...")
    model_config, model_yaml_path = _load_model_config(weka_path)
    logger.info("[Step 0b] Model config loaded and validated successfully")

    # Step 0c: Extract modalities from dataset config
    logger.info("[Step 0c] Extracting modalities from dataset config...")
    modality_metadata = _extract_modality_metadata(dataset_config)
    logger.info(
        f"[Step 0c] Modalities: {modality_metadata.modalities}, "
        f"timeseries: {modality_metadata.timeseries}, "
        f"num_timesteps: {modality_metadata.num_timesteps}"
    )

    # Step 0d: Extract and instantiate the rslearn task from model config
    logger.info("[Step 0d] Extracting task from model config...")
    task_metadata = _extract_task_metadata(model_config)
    logger.info(
        f"[Step 0d] Task: {task_metadata.task_type}, num_classes: {task_metadata.num_classes}"
    )

    # Step 2: Scan windows and determine existing splits
    logger.info("[Step 2/6] Scanning windows and determining splits...")
    splits = scan_windows_and_splits(
        weka_path,
        source_groups=config.source_groups,
        source_tags=config.source_tags,
    )
    logger.info(
        f"[Step 2/6] Scan complete: train={len(splits['train'])}, "
        f"val={len(splits['val'])}, test={len(splits['test'])}"
    )

    # Step 3: Create missing splits if needed
    logger.info("[Step 3/6] Creating missing splits if needed...")
    splits = create_missing_splits(
        splits,
        val_test_ratio=config.val_test_split_ratio,
        train_val_ratio=config.train_val_split_ratio,
        seed=config.split_seed,
    )
    logger.info(
        f"[Step 3/6] Split creation complete: train={len(splits['train'])}, "
        f"val={len(splits['val'])}, test={len(splits['test'])}"
    )

    # Step 4: Write split tags to metadata
    logger.info("[Step 4/6] Writing split tags to window metadata...")
    write_split_tags(weka_path, splits)
    logger.info("[Step 4/6] Split tags written")

    # Step 5: Count split statistics
    logger.info("[Step 5/6] Counting split statistics...")
    split_stats = count_split_stats(splits)
    logger.info(f"[Step 5/6] Stats: {split_stats}")

    # Step 6: Compute normalization stats (from copied dataset, use train for stats)
    logger.info("[Step 6/6] Computing normalization stats from train split...")
    norm_stats = compute_band_stats_from_model_config(
        model_config_path=str(model_yaml_path),
        source_path=weka_path,
        groups=config.source_groups,
        tags={EVAL_SPLIT_TAG_KEY: SplitName.TRAIN},
        num_samples=config.num_samples,
    )
    logger.info(
        f"[Step 6/6] Normalization stats computed for {len(norm_stats)} modalities"
    )

    # Extract window_size: prefer crop_size from config, fall back to actual
    # window dimensions from the dataset.
    window_size = _extract_window_size(model_config, weka_path)

    logger.info("Creating EvalDatasetEntry...")
    entry = EvalDatasetEntry(
        name=config.name,
        source_path=config.source_path,
        weka_path=weka_path,
        task_type=task_metadata.task_type,
        num_classes=task_metadata.num_classes,
        classes=task_metadata.label_values,
        modalities=modality_metadata.modalities,
        window_size=window_size,
        timeseries=modality_metadata.timeseries,
        num_timesteps=modality_metadata.num_timesteps,
        split_tag_key=EVAL_SPLIT_TAG_KEY,
        split_stats=split_stats,
        norm_stats=norm_stats,
    )

    logger.info(f"{'=' * 60}")
    logger.info(f"INGEST COMPLETE: {config.name}")
    logger.info(f"{'=' * 60}")
    logger.info(f"  Weka path: {weka_path}")
    logger.info(
        f"  Task: {task_metadata.task_type}, classes: {task_metadata.num_classes}"
    )
    logger.info(
        f"  Splits: train={split_stats.get('train', {}).get('count', 0)}, "
        f"val={split_stats.get('val', {}).get('count', 0)}, "
        f"test={split_stats.get('test', {}).get('count', 0)}"
    )

    return entry
