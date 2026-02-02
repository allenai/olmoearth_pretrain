"""Main ingestion logic for Studio datasets.

This module provides the core functionality to ingest an rslearn dataset
from Studio/GCS into the OlmoEarth eval system on Weka.

Ingestion Flow:
--------------
1. Validate the source dataset
2. Create the destination directory on Weka
3. Copy data from source to destination
4. Compute normalization statistics
5. Create metadata.json in the dataset directory
6. Register the dataset in the central registry

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
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime

import numpy as np
import yaml
from rslearn.config import DatasetConfig
from rslearn.dataset.dataset import Dataset as RslearnDataset
from rslearn.utils.raster_format import GeotiffRasterFormat
from tqdm import tqdm
from upath import UPath

from olmoearth_pretrain.evals.studio_ingest.band_stats import (
    compute_band_stats_from_rslearn_dataset,
)
from olmoearth_pretrain.evals.studio_ingest.schema import (
    RSLEARN_TO_OLMOEARTH,
    EvalDatasetEntry,
    instantiate_from_config,
    rslearn_task_type_to_olmoearth_task_type,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================

# Environment variable for dataset root path
# This allows external users to download datasets from HF and point to their local copy
EVAL_DATASETS_ENV_VAR = "OLMOEARTH_EVAL_DATASETS"

# Default base path on Weka where all eval datasets are stored (internal)
# TODO: Confirm this path with the team
DEFAULT_WEKA_BASE_PATH = "weka://dfive-default/olmoearth/eval_datasets"

# Environment variable for number of workers
# Defaults to cpu_count - 1, capped at 32 for thread pools
import os as _os

_default_workers = (_os.cpu_count() or 1) - 1
NUM_WORKERS = int(_os.environ.get("OLMOEARTH_INGEST_WORKERS", _default_workers))
MAX_THREAD_WORKERS = int(_os.environ.get("OLMOEARTH_INGEST_MAX_THREADS", 32))


def get_eval_datasets_base_path() -> str:
    """Get the base path for eval datasets.

    Checks OLMOEARTH_EVAL_DATASETS env var first, falls back to Weka path.
    This allows external users to download datasets from HF and use them locally.

    Returns:
        Base path string (either from env var or default Weka path)
    """
    import os

    return os.environ.get(EVAL_DATASETS_ENV_VAR, DEFAULT_WEKA_BASE_PATH)


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
        display_name: Human-readable name
        source_path: Path to the source rslearn dataset
        task_type: Type of task (classification, regression, segmentation)
        modalities: List of modality names
        target_property: Name of the property containing labels

        # Optional - Task specific
        classes: List of class names (for classification)

        # Optional - Temporal
        temporal_range: Tuple of (start_date, end_date) as ISO strings

        # Optional - Normalization
        compute_norm_stats: Whether to compute normalization statistics
        sample_fraction: Fraction of data to sample for stats
        max_samples: Maximum samples for stats computation

        # Optional - Metadata
        studio_task_id: Optional link back to Studio
        notes: Optional notes about the dataset

        # Optional - Behavior
        overwrite: Whether to overwrite existing dataset
        skip_validation: Whether to skip validation (not recommended)
    """

    # Required # THis can be simplified and likely will in the end need to just deal with unified config but for now this should have all the info
    name: str
    source_path: str
    olmoearth_run_config_path: str

    # Optional - Sampling for stats computation
    max_samples: int | None = None
    sample_fraction: float | None = None
    groups: list[str] | None = None


# =============================================================================
# Label Scanning Utilities
# =============================================================================

GEOTIFF_RASTER_FORMAT = GeotiffRasterFormat()


def get_label_layer_info(dataset_config: DatasetConfig) -> tuple[str, list[str]] | None:
    """Find the label layer in the dataset config.

    The label layer is identified by having no data_source (meaning it's derived/local).

    Args:
        dataset_config: The rslearn DatasetConfig

    Returns:
        Tuple of (layer_name, bands) if found, None otherwise
    """
    for layer_name, layer_config in dataset_config.layers.items():
        if layer_config.data_source is not None:
            # This is a modality layer with an external data source
            continue
        # Found a layer without data_source - this should be the label layer
        if layer_config.band_sets:
            bands = layer_config.band_sets[0].bands
            return (layer_name, bands)
    return None


# =============================================================================
# Transform Extraction Utilities
# =============================================================================


def _extract_sizing(
    transforms: list[dict],
    default_config: dict | None = None,
) -> tuple[int | None, int | None]:
    """Extract crop and pad sizes from transforms or default_config.

    Scans the transforms for Crop and Pad transforms and extracts
    their sizing parameters. Falls back to default_config.patch_size
    if no Crop transform is found.

    NOTE: The rslearn API for patch_size in default_config is expected to change.
    This extraction logic may need updates when rslearn updates their config schema.

    Args:
        transforms: List of transform config dicts with class_path and init_args.
        default_config: Optional default_config dict from model.yaml data.init_args.

    Returns:
        Tuple of (crop_size, pad_size), each None if not found.
    """
    crop_size: int | None = None
    pad_size: int | None = None

    for transform in transforms:
        class_path = transform.get("class_path", "")
        init_args = transform.get("init_args", {})

        if "crop.Crop" in class_path:
            crop_size = init_args.get("crop_size")
            # Handle tuple case (min, max) - take the min for deterministic eval
            if isinstance(crop_size, (list, tuple)):
                crop_size = crop_size[0]
        elif "pad.Pad" in class_path:
            pad_size = init_args.get("size")
            # Handle tuple case (min, max) - take the min for deterministic eval
            if isinstance(pad_size, (list, tuple)):
                pad_size = pad_size[0]

    # Fallback: use default_config.patch_size if no Crop transform found
    # NOTE: This is rslearn's window/tile size for sampling data
    if crop_size is None and default_config is not None:
        crop_size = default_config.get("patch_size")

    return crop_size, pad_size


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
    logger.info(f"Starting ingestion of dataset: {config.name}")
    logger.info(f"Source: {config.source_path}")


    config_upath = UPath(config.source_path) / "config.json"
    with config_upath.open() as f:
        dataset_dict = json.load(f)

    dataset_config = DatasetConfig.model_validate(dataset_dict)


    model_config_path = UPath(config.olmoearth_run_config_path) / "model.yaml"
    with model_config_path.open() as f:
        model_config = yaml.safe_load(f)

    # Extract modalities from dataset config
    modalities = []
    modality_layer_names = []
    max_timesteps_modalities = []
    for layer_name, layer_config in dataset_config.layers.items():
        if layer_config.data_source is None:
            continue
        olmoearth_modality = RSLEARN_TO_OLMOEARTH[layer_name]
        modalities.append(olmoearth_modality.name)
        modality_layer_names.append(layer_name)
        query_config = layer_config.data_source.query_config
        max_timesteps_modalities.append(query_config.max_matches)

    num_timesteps = max(max_timesteps_modalities) if max_timesteps_modalities else 1
    timeseries = num_timesteps > 1
    logger.info(f"Modalities: {modalities}, timeseries: {timeseries}")

    # Extract and instantiate the rslearn task from model config
    # model.yaml structure: data.init_args.task.init_args.tasks.{task_name}
    task_wrapper_config = model_config["data"]["init_args"]["task"]
    tasks_dict = task_wrapper_config["init_args"]["tasks"]
    # For now, grab the first (and typically only) task
    if len(tasks_dict) != 1:
        raise NotImplementedError(
            "Multiple tasks not supported in this workflow; found: "
            + ", ".join(tasks_dict)
        )
    task_name, task_config = next(iter(tasks_dict.items()))
    logger.info(
        f"Instantiating task '{task_name}' from model config: {task_config['class_path']}"
    )
    rslearn_task = instantiate_from_config(task_config)

    # Get num_classes from the task config based on task type
    task_init_args = task_config.get("init_args", {})
    task_class_path = task_config.get("class_path", "")

    num_classes: int | None = None
    if "num_classes" in task_init_args:
        # SegmentationTask, PerPixelRegressionTask with classes
        num_classes = task_init_args["num_classes"]
    elif "classes" in task_init_args:
        # ClassificationTask, DetectionTask use a 'classes' list
        num_classes = len(task_init_args["classes"])

    if num_classes is None:
        raise ValueError(
            f"Could not determine num_classes from task config '{task_name}' "
            f"(class_path: {task_class_path}). "
            "Expected 'num_classes' or 'classes' in init_args."
        )

    # Assume 0-indexed consecutive labels (0 to num_classes-1)
    label_values = [str(i) for i in range(num_classes)]
    logger.info(f"Got {num_classes} classes from model config")

    # Compute and cache normalization stats
    import hashlib
    import os
    import pickle

    def _norm_stats_cache_path(source_path, modalities, task_config):
        src_hash = hashlib.sha256(
            (
                str(source_path)
                + str(sorted(modalities))
                + str(task_config["class_path"])
                + str(task_config.get("init_args", {}))
            ).encode()
        ).hexdigest()[:16]
        cache_dir = "/tmp/olmoearth_norm_stats_cache"
        os.makedirs(cache_dir, exist_ok=True)
        return os.path.join(cache_dir, f"{src_hash}_norm_stats.pkl")

    norm_stats_cache_path = _norm_stats_cache_path(
        config.source_path, modality_layer_names, task_config
    )
    if os.path.exists(norm_stats_cache_path):
        with open(norm_stats_cache_path, "rb") as f:
            norm_stats = pickle.load(f)
        logger.info(f"Loaded cached norm stats from {norm_stats_cache_path}")
    else:
        logger.info(f"Computing norm stats for {config.source_path}")
        norm_stats = compute_band_stats_from_rslearn_dataset(
            dataset_path=config.source_path,
            modalities=modality_layer_names,
            task=rslearn_task,
            groups=config.groups,
            max_samples=config.max_samples,
            sample_fraction=config.sample_fraction,
        )
        with open(norm_stats_cache_path, "wb") as f:
            pickle.dump(norm_stats, f)
        logger.info(f"Computed and cached norm stats at {norm_stats_cache_path}")

    task_type = rslearn_task_type_to_olmoearth_task_type(rslearn_task)

    # Extract window_size from default_config
    data_init_args = model_config["data"]["init_args"]
    default_config = data_init_args.get("default_config", {})
    window_size = default_config.get("patch_size", 64)

    entry = EvalDatasetEntry(
        name=config.name,
        source_path=config.source_path,
        model_config_path=config.olmoearth_run_config_path,
        task_type=task_type,
        num_classes=num_classes,
        classes=label_values,
        modalities=modalities,
        window_size=window_size,
        timeseries=timeseries,
        norm_stats_path=norm_stats_cache_path,
        custom_groups=config.groups or [],
        num_timesteps=num_timesteps,
    )
    return entry
