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
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING

from upath import UPath

# TODO: Integrate band_stats.py into the ingestion workflow
# from olmoearth_pretrain.evals.studio_ingest.band_stats import compute_band_stats
from olmoearth_pretrain.evals.studio_ingest.registry import REGISTRY_PATH, Registry
from olmoearth_pretrain.evals.studio_ingest.schema import EvalDatasetEntry
from olmoearth_pretrain.evals.studio_ingest.validate import validate_dataset

if TYPE_CHECKING:
    pass

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
    display_name: str
    source_path: str
    task_type: str
    modalities: list[str]
    target_property: str

    # Optional - Task specific
    classes: list[str] | None = None

    # Optional - Temporal
    temporal_range: tuple[str, str] = ("", "")

    # Optional - Patch config
    patch_size: int = 64

    # Optional - Normalization
    compute_norm_stats: bool = True
    sample_fraction: float = 0.1
    max_samples: int = 10000

    # Optional - Metadata
    studio_task_id: str | None = None
    notes: str | None = None

    # Optional - Behavior
    overwrite: bool = False
    skip_validation: bool = False


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
    logger.info(f"Task type: {config.task_type}")
    logger.info(f"Modalities: {config.modalities}")

    # Run all steps
    step_validate(config)
    dest_path = step_create_destination(config)

    try:
        splits = step_copy_data(config, dest_path)
        step_compute_stats(config, dest_path)
        entry = step_create_metadata(config, dest_path, splits)
        step_register(entry, overwrite=config.overwrite)
    except Exception:
        # TODO: Add cleanup on failure
        logger.error(
            f"Ingestion failed. Partial data may exist at {dest_path}. "
            "Please clean up manually."
        )
        raise

    logger.info(f"Successfully ingested dataset: {config.name}")
    return entry
