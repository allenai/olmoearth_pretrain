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
    DEFAULT_TARGET_PROPERTY,
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


def get_unique_label_values(
    source_path: str,
    dataset_config: DatasetConfig,
    max_windows: int | None = None,
) -> tuple[set[int], int]:
    """Scan the rslearn dataset to get unique label values.

    Iterates through all windows in the dataset, reads the label raster
    from each, and collects all unique integer values.

    Args:
        source_path: Path to the rslearn dataset
        dataset_config: The parsed DatasetConfig
        max_windows: Optional limit on number of windows to scan (for debugging)

    Returns:
        Tuple of (set of unique label values, number of classes)

    Raises:
        ValueError: If no label layer is found in the dataset
    """
    # Find the label layer
    label_info = get_label_layer_info(dataset_config)
    if label_info is None:
        raise ValueError(
            "No label layer found in dataset. Label layer should have no data_source."
        )
    label_layer_name, label_bands = label_info
    logger.info(f"Found label layer: {label_layer_name} with bands: {label_bands}")

    # Load the rslearn dataset
    rslearn_dataset = RslearnDataset(UPath(source_path))

    # Get all windows
    windows = rslearn_dataset.load_windows(workers=NUM_WORKERS, show_progress=True)
    if max_windows is not None:
        windows = windows[:max_windows]

    logger.info(f"Scanning {len(windows)} windows for unique label values...")

    def scan_window_labels(window):
        """Scan a single window for unique label values."""
        try:
            raster_dir = window.get_raster_dir(label_layer_name, label_bands)
            if not raster_dir.exists():
                return set()
            label_raster = GEOTIFF_RASTER_FORMAT.decode_raster(
                raster_dir, window.projection, window.bounds
            )
            return set(np.unique(label_raster).astype(int).tolist())
        except Exception as e:
            logger.warning(f"Error reading label for window {window.name}: {e}")
            return set()

    unique_values: set[int] = set()
    num_workers = min(MAX_THREAD_WORKERS, len(windows))
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(scan_window_labels, w): w for w in windows}
        for future in tqdm(
            as_completed(futures), total=len(futures), desc="Scanning labels"
        ):
            unique_values.update(future.result())

    # Filter out common NODATA values if needed (typically -1 or 255 for uint8)
    # For now, we return all values and let the caller handle NODATA filtering
    num_classes = len(unique_values)

    return unique_values, num_classes


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

    # What information will need to run the task?
    # What modalities are supported? - config.json
    # is it time series? - config.json
    # is it multilabel? - config.json
    # Any imputes or missing data - config.json
    # label target property - config.json

    # Derived elsewhere
    # How many classes are there? - I think this is only possible by reading and summarizing all the labels in the dataset
    # hw if it is segmentation?
    # what is the task type? - I think we would probably need some sort of finetuning config associated with this task
    # what data belongs to which split? (in the individual metadata.json files)

    config_upath = UPath(config.source_path) / "config.json"
    with config_upath.open() as f:
        dataset_dict = json.load(f)
        # TODO: Potentiually validate witht the rsleanr class
    # would be nice to not have to check windows here at all
    dataset_config = DatasetConfig.model_validate(dataset_dict)

    # Load the olmoearth configs starting with the model config
    # I can either parse the config as needed or fully create the whole thing and then parse it
    # probably best to just parse the config as needed
    # AND THEN I CAN VALIDATE CERTAIN INFORMATION IS PRESENT AGAINST THAT

    model_config_path = UPath(config.olmoearth_run_config_path) / "model.yaml"
    with model_config_path.open() as f:
        model_config = yaml.safe_load(f)
    # TODO: Validate the model config
    print(model_config)
    # Get the modalities
    modalities = []
    modality_layer_names = []
    num_timesteps_modalities = []
    for layer_name, layer_config in dataset_config.layers.items():
        if layer_config.data_source is None:
            # This means it is some other layer like a label layer and not one of the modalities we are interested in
            continue
        # Get the modality
        print(layer_name)
        print(layer_config)
        olmoearth_modality = RSLEARN_TO_OLMOEARTH[layer_name]
        modalities.append(olmoearth_modality)
        modality_layer_names.append(layer_name)
        # get the temporal range
        query_config = layer_config.data_source.query_config
        # For now we raise an error if min_matches and max_matches are not the same
        if query_config.min_matches != query_config.max_matches:
            raise ValueError(
                f"Min matches and max matches are not the same for layer {layer_name}"
            )
        num_timesteps_modalities.append(query_config.min_matches)
    print(modalities)
    # assert all the num_timesteps_modalities are the same
    if len(set(num_timesteps_modalities)) != 1:
        raise ValueError("Num timesteps are not the same for all modalities")
    num_timesteps = num_timesteps_modalities[0]

    # get the target property
    # I am pretty sure this is just the band name of the label layer
    target_property = DEFAULT_TARGET_PROPERTY
    # for layer_name, layer_config in dataset_config.layers.items():
    #     if layer_config.data_source is not None:
    #         # This means it is a modality layer
    #         continue
    #     # Assuming that no data source means it is a label layer, this could be potentially wrong
    #     band_sets = layer_config.band_sets
    #     if len(band_sets) != 1:
    #         raise ValueError(f"Expected only one band set for label layer {layer_name}, multiple label bandsets are not supported yet")
    #     band_set = band_sets[0]
    #     # May need to revise target properyty var name later
    #     label_layer_name = band_set.bands[0]
    #     # the target property is used in the metadata.json inside each windows but is not easily available
    #     # check the metadata.json of the first window to see if the target property is the default one
    #     # TODO: Support a target property specified by the config.json

    #     # Assume that there is only one label layer
    #     break

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
    label_values = set(range(num_classes))
    logger.info(f"Got {num_classes} classes from model config: {sorted(label_values)}")

    # get the NODATA value -> This would be in the olmoearth_run.yaml config
    nodata_value = 0
    zero_is_invalid = 0 == nodata_value

    # get the multilabel -> assume false for now
    is_multilabel = False
    # get the imputes -> assume no imputes for now
    imputes: list[tuple[str, str]] = []
    # determine if timeseries from num_timesteps
    timeseries = num_timesteps > 1

    # norm stats: cache so we don't recompute every time for the same data + task
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
    print(norm_stats)

    # get the created at
    created_at = datetime.now().isoformat()
    # TODO: get the studio task id
    # TODO: class names but that may not be easy to get as of right now
    # Map rslearn task type to olmoearth task type
    # Apply the mapping here
    task_type = rslearn_task_type_to_olmoearth_task_type(rslearn_task)

    entry = EvalDatasetEntry(
        name=config.name,
        source_path=config.source_path,
        task_type=task_type,
        target_property=target_property,
        classes=label_values,
        num_classes=num_classes,
        modalities=modalities,
    )
    return entry
