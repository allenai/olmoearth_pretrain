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

import getpass
import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime

import numpy as np
from rslearn.config import DatasetConfig
from rslearn.dataset.dataset import Dataset as RslearnDataset
from rslearn.utils.raster_format import GeotiffRasterFormat
from tqdm import tqdm
from upath import UPath

# TODO: Integrate band_stats.py into the ingestion workflow
# from olmoearth_pretrain.evals.studio_ingest.band_stats import compute_band_stats
from olmoearth_pretrain.evals.studio_ingest.registry import REGISTRY_PATH, Registry
from olmoearth_pretrain.evals.studio_ingest.schema import (
    DEFAULT_TARGET_PROPERTY,
    RSLEARN_TO_OLMOEARTH,
    EvalDatasetEntry,
)
from olmoearth_pretrain.evals.studio_ingest.validate import validate_dataset

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

    # Required
    name: str
    source_path: str
    # We want to be able to infer all the relevant information and metadata and if not we should serialize and write it in studio to make this possible.


# =============================================================================
# Ingestion Steps
# =============================================================================


def step_validate(config: IngestConfig) -> None:
    """Step 1: Validate the source dataset.

    Args:
        config: Ingestion configuration

    Raises:
        ValueError: If validation fails
    """
    if config.skip_validation:
        logger.warning("Skipping validation as requested")
        return

    logger.info(f"Validating source dataset: {config.source_path}")

    result = validate_dataset(
        source_path=config.source_path,
        modalities=config.modalities,
        task_type=config.task_type,
        target_property=config.target_property,
    )

    if not result.is_valid:
        raise ValueError(f"Dataset validation failed:\n{result}")

    logger.info("Validation passed")

    # Log any warnings
    for warning in result.warnings:
        logger.warning(warning)


def step_create_destination(config: IngestConfig) -> UPath:
    """Step 2: Create the destination directory on Weka.

    Args:
        config: Ingestion configuration

    Returns:
        Path to the destination directory

    Raises:
        FileExistsError: If destination exists and overwrite=False
    """
    base_path = get_eval_datasets_base_path()
    dest_path = UPath(base_path) / config.name

    if dest_path.exists():
        if config.overwrite:
            logger.warning(f"Destination exists, will overwrite: {dest_path}")
            # TODO: Should we delete the existing directory?
            # For safety, we don't delete automatically
        else:
            raise FileExistsError(
                f"Destination already exists: {dest_path}. Use --overwrite to replace."
            )
    else:
        logger.info(f"Creating destination directory: {dest_path}")
        dest_path.mkdir(parents=True, exist_ok=True)

    return dest_path


def step_copy_data(
    config: IngestConfig,
    dest_path: UPath,
) -> dict[str, int]:
    """Step 3: Copy data from source to destination.

    This copies the rslearn dataset structure to Weka.

    Args:
        config: Ingestion configuration
        dest_path: Destination path on Weka

    Returns:
        Dict mapping split name -> sample count

    Todo:
        - Add progress bar
        - Add resumable copying
        - Handle large datasets efficiently
        - Preserve rslearn structure properly
    """
    source_path = UPath(config.source_path)
    logger.info(f"Copying data from {source_path} to {dest_path}")

    # TODO: Implement actual copying
    # This is a placeholder - actual implementation depends on:
    # 1. rslearn dataset structure
    # 2. How splits are organized
    # 3. Whether we copy everything or filter

    # Placeholder: count splits
    splits = {}

    # TODO: Use rslearn to determine splits and copy
    # For now, just list expected splits
    for split in ["train", "val", "test"]:
        split_source = source_path / split
        split_dest = dest_path / split

        if split_source.exists():
            logger.info(f"Copying split: {split}")
            # TODO: Implement actual copy
            # shutil.copytree won't work across cloud storage
            # Need to use cloud-specific copy or streaming

            # Placeholder count
            splits[split] = 0  # TODO: Count actual samples
        else:
            logger.warning(f"Split '{split}' not found at {split_source}")

    return splits


def step_compute_stats(
    config: IngestConfig,
    dest_path: UPath,
) -> None:
    """Step 4: Compute normalization statistics.

    Uses the existing band_stats module which was moved from
    scripts/tools/compute_rslearn_dataset_band_stats.py.

    Args:
        config: Ingestion configuration
        dest_path: Destination path on Weka (where data was copied)

    Todo:
        - Integrate band_stats.compute_band_stats() here
        - Need to build the model_ds from the copied data
        - Save output to dest_path / "norm_stats.json"
    """
    if not config.compute_norm_stats:
        logger.info("Skipping normalization stats computation")
        return

    logger.info("Computing normalization statistics")

    # TODO: Integrate band_stats.py here
    # For now, this is a placeholder - stats should be computed manually:
    #
    # uv run --group ingest python -m olmoearth_pretrain.evals.studio_ingest.band_stats \
    #     --ds_path {dest_path} \
    #     --input_layers {config.modalities} \
    #     --output_json {dest_path}/norm_stats.json
    #
    logger.warning(
        "Band stats computation not yet integrated. "
        "Run band_stats.py manually after ingestion."
    )


def step_create_metadata(
    config: IngestConfig,
    dest_path: UPath,
    splits: dict[str, int],
) -> EvalDatasetEntry:
    """Step 5: Create metadata.json in the dataset directory.

    Args:
        config: Ingestion configuration
        dest_path: Destination path on Weka
        splits: Split counts from copy step

    Returns:
        The EvalDatasetEntry that was created
    """
    logger.info("Creating metadata")

    entry = EvalDatasetEntry(
        name=config.name,
        display_name=config.display_name,
        task_type=config.task_type,
        target_property=config.target_property,
        classes=config.classes,
        modalities=config.modalities,
        temporal_range=config.temporal_range,
        patch_size=config.patch_size,
        source_path=config.source_path,
        weka_path=str(dest_path),
        splits=splits,
        supports_cv=False,  # TODO: Support CV
        cv_folds=None,
        norm_stats_path="norm_stats.json",
        use_pretrain_norm=not config.compute_norm_stats,
        created_at=datetime.now().isoformat(),
        created_by=getpass.getuser(),
        studio_task_id=config.studio_task_id,
        notes=config.notes,
    )

    # Save to metadata.json
    metadata_path = dest_path / "metadata.json"
    with metadata_path.open("w") as f:
        json.dump(entry.to_dict(), f, indent=2)

    logger.info(f"Saved metadata to {metadata_path}")
    return entry


def step_register(
    entry: EvalDatasetEntry,
    overwrite: bool = False,
) -> None:
    """Step 6: Register the dataset in the central registry.

    Args:
        entry: The dataset entry to register
        overwrite: Whether to overwrite existing entry
    """
    logger.info("Registering dataset in central registry")

    registry = Registry.load(REGISTRY_PATH)
    registry.add(entry, overwrite=overwrite)
    registry.save(REGISTRY_PATH)

    logger.info(f"Dataset '{entry.name}' registered successfully")


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
    windows = rslearn_dataset.load_windows(workers=8, show_progress=True)
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
    num_workers = min(32, len(windows))  # cap at 32 threads
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
    dataset_config = DatasetConfig.model_validate(dataset_dict)
    # Get the modalities
    modalities = []
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
    # get the label values and the number of classes by scanning the data
    label_values, num_classes = get_unique_label_values(
        config.source_path, dataset_config
    )
    logger.info(f"Found {num_classes} unique label values: {sorted(label_values)}")

    # get the NODATA values

    # get the multilabel
    # get the imputes
    # get the missing data
    # get the splits
    # get the cv folds
    # get the norm stats
    # get the pretrain norm
    # get the created at
    # get the created by
    # get the studio task id
    # get the notes
    # Optional it would be nice to have the class names but that may not be easy to get as of right now

    raise ValueError("Not implemented")
    entry = EvalDatasetEntry(
        name=config.name,
        source_path=config.source_path,
    )

    # step_validate(config)
    # dest_path = step_create_destination(config)

    # try:
    #     splits = step_copy_data(config, dest_path)
    #     step_compute_stats(config, dest_path)
    #     entry = step_create_metadata(config, dest_path, splits)
    #     step_register(entry, overwrite=config.overwrite)
    # except Exception:
    #     # TODO: Add cleanup on failure
    #     logger.error(
    #         f"Ingestion failed. Partial data may exist at {dest_path}. "
    #         "Please clean up manually."
    #     )
    #     raise

    # logger.info(f"Successfully ingested dataset: {config.name}")
    return entry
