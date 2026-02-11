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
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from rslearn.config import DatasetConfig
from rslearn.dataset.dataset import Dataset as RslearnDataset
from upath import UPath

from olmoearth_pretrain.evals.studio_ingest.band_stats import (
    compute_band_stats_from_model_config,
)
from olmoearth_pretrain.evals.studio_ingest.schema import (
    EvalDatasetEntry,
    instantiate_from_config,
    rslearn_task_type_to_olmoearth_task_type,
    rslearn_to_olmoearth,
)
from olmoearth_pretrain.evals.task_types import SplitName, SplitType

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
_default_workers = (os.cpu_count() or 1) - 1
NUM_WORKERS = int(os.environ.get("OLMOEARTH_INGEST_WORKERS", _default_workers))
MAX_THREAD_WORKERS = int(os.environ.get("OLMOEARTH_INGEST_MAX_THREADS", 32))


def get_eval_datasets_base_path() -> str:
    """Get the base path for eval datasets.

    Checks OLMOEARTH_EVAL_DATASETS env var first, falls back to Weka path.
    This allows external users to download datasets from HF and use them locally.

    Returns:
        Base path string (either from env var or default Weka path)
    """
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
        num_samples: Number of samples for stats computation (None = all)

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
    num_samples: int | None = None

    # Archive handling
    untar_source: bool = False


# =============================================================================
# Dataset Copy Utilities
# =============================================================================

# Base path for eval datasets on Weka
EVAL_DATASETS_BASE_PATH = "/weka/dfive-default/olmoearth/eval_datasets"

# Tag key for eval splits (we use our own key to avoid overwriting original split info)
EVAL_SPLIT_TAG_KEY = "eval_split"


def _check_weka_exists() -> bool:
    """Check if Weka filesystem path exists.

    Returns:
        True if /weka exists.
    """
    return Path("/weka").exists()


def _copy_from_gcs(
    source_path: str,
    dest_path: str,
    source_groups: list[str] | None = None,
) -> str:
    """Copy dataset from GCS using gsutil with parallel transfers.

    Uses gsutil -m for multi-threaded/multi-processing transfers.
    Streams output directly to console for progress visibility.

    Args:
        source_path: GCS path (gs://bucket/path)
        dest_path: Local destination path
        source_groups: If specified, only copy these groups (subdirs under windows/)

    Returns:
        Destination path
    """
    logger.info(f"  Copy method: gsutil (parallel GCS transfer)")

    # Create destination directory
    Path(dest_path).mkdir(parents=True, exist_ok=True)

    # Always copy config.json first
    config_src = f"{source_path}/config.json"
    config_dst = f"{dest_path}/config.json"
    logger.info(f"  Copying config.json...")
    subprocess.run(["gsutil", "cp", config_src, config_dst], check=True)

    if source_groups:
        # Copy only specified groups
        logger.info(f"  Copying only groups: {source_groups}")
        for group in source_groups:
            group_src = f"{source_path}/windows/{group}"
            group_dst_parent = f"{dest_path}/windows"
            Path(group_dst_parent).mkdir(parents=True, exist_ok=True)
            logger.info(f"  Running: gsutil -m cp -r {group_src} {group_dst_parent}")
            subprocess.run(["gsutil", "-m", "cp", "-r", group_src, group_dst_parent], check=True)
    else:
        # Copy entire windows directory
        windows_src = f"{source_path}/windows"
        logger.info(f"  Running: gsutil -m cp -r {windows_src} {dest_path}")
        subprocess.run(["gsutil", "-m", "cp", "-r", windows_src, dest_path], check=True)

    logger.info(f"  gsutil copy complete")
    return dest_path


def _copy_from_gcs_tar(
    source_path: str,
    dest_path: str,
) -> str:
    """Download a .tar.gz archive from GCS and extract it to dest_path.

    Downloads the archive to dest_path first, then extracts in place and
    removes the archive file. If the archive contains a single top-level
    directory, returns the path to that directory (e.g. dest_path/dataset/).

    Args:
        source_path: GCS path to a .tar.gz archive (gs://bucket/dataset.tar.gz)
        dest_path: Local destination path to extract into

    Returns:
        Path to the extracted dataset directory
    """
    logger.info(f"  Copy method: gsutil download + tar extract")

    Path(dest_path).mkdir(parents=True, exist_ok=True)

    archive_name = Path(source_path).name
    local_archive = Path(dest_path) / archive_name

    logger.info(f"  Downloading {source_path} -> {local_archive}")
    subprocess.run(["gsutil", "cp", source_path, str(local_archive)], check=True)

    logger.info(f"  Extracting {local_archive} -> {dest_path}")
    subprocess.run(["tar", "xzf", str(local_archive), "-C", dest_path], check=True)

    logger.info(f"  Removing archive {local_archive}")
    local_archive.unlink()

    # If the archive extracted into a single subdirectory, use that as the
    # dataset path (e.g. dataset.tar.gz -> dest_path/dataset/)
    entries = [p for p in Path(dest_path).iterdir() if p.name != archive_name]
    if len(entries) == 1 and entries[0].is_dir():
        dataset_path = str(entries[0])
        logger.info(f"  Extracted dataset directory: {dataset_path}")
        return dataset_path

    logger.info(f"  tar extract complete")
    return dest_path


def _tar_copy_cmd(src: str, dst: str, use_pv: bool) -> str:
    """Build a streaming tar copy command, optionally with pv progress.

    TODO: remove this helper once pv progress bar is no longer needed.
    """
    if use_pv:
        return f"tar cf - -C {src} . | pv | tar xf - -C {dst}"
    return f"tar cf - -C {src} . | tar xf - -C {dst}"


def _copy_local(
    source_path: str,
    dest_path: str,
    source_groups: list[str] | None = None,
) -> str:
    """Copy dataset locally using streaming tar pipe.

    Uses ``tar cf - | tar xf -`` to stream data directly from source to
    destination without intermediate files. This avoids per-file overhead
    and is faster than rsync for bulk local copies.  Directory structure
    is preserved because tar archives relative paths from the source and
    recreates them at the destination.

    Args:
        source_path: Local source path
        dest_path: Local destination path
        source_groups: If specified, only copy these groups (subdirs under windows/)

    Returns:
        Destination path

    Raises:
        RuntimeError: If destination is on Weka but Weka path doesn't exist.
    """
    # Verify Weka path exists if destination is on Weka
    if dest_path.startswith("/weka") and not _check_weka_exists():
        raise RuntimeError(
            "Weka filesystem path /weka does not exist. "
            "Cannot copy dataset. Ensure Weka is available before running ingestion."
        )

    # Create destination directory
    Path(dest_path).mkdir(parents=True, exist_ok=True)

    # TODO: remove pv progress bar once copy performance is validated
    has_pv = shutil.which("pv") is not None

    logger.info("  Copy method: streaming tar pipe%s", " (with pv progress)" if has_pv else "")

    # Always copy config.json first
    config_src = Path(source_path) / "config.json"
    if config_src.exists():
        shutil.copy2(config_src, Path(dest_path) / "config.json")
        logger.info("  Copied config.json")

    if source_groups:
        # Copy only specified groups under windows/
        logger.info(f"  Copying only groups: {source_groups}")
        for group in source_groups:
            group_src = f"{source_path}/windows/{group}"
            group_dst = f"{dest_path}/windows/{group}"
            Path(group_dst).mkdir(parents=True, exist_ok=True)

            cmd = _tar_copy_cmd(group_src, group_dst, has_pv)
            logger.info(f"  Running: {cmd}")
            subprocess.run(cmd, shell=True, check=True)
            logger.info(f"  Copied group '{group}'")
    else:
        # Copy entire directory using streaming tar
        cmd = _tar_copy_cmd(source_path, dest_path, has_pv)
        logger.info(f"  Running: {cmd}")
        subprocess.run(cmd, shell=True, check=True)
        logger.info("  Copy complete")

    return dest_path


def _copy_generic(
    source_path: str,
    dest_path: str,
    source_groups: list[str] | None = None,
) -> str:
    """Fallback copy using UPath for unknown storage backends.

    Sequential file-by-file copy. Slower but works for any storage.

    Args:
        source_path: Source path (any UPath-compatible)
        dest_path: Destination path
        source_groups: If specified, only copy these groups (subdirs under windows/)

    Returns:
        Destination path
    """
    logger.info(f"  Copy method: UPath (generic, sequential)")

    source = UPath(source_path)
    dest = UPath(dest_path)

    dest.mkdir(parents=True, exist_ok=True)

    # Copy config.json
    config_src = source / "config.json"
    if config_src.exists():
        with config_src.open() as f:
            config_data = f.read()
        with (dest / "config.json").open("w") as f:
            f.write(config_data)
        logger.info(f"    Copied config.json")

    # Copy windows directory (filtered by groups if specified)
    windows_src = source / "windows"
    windows_dst = dest / "windows"
    if windows_src.exists():
        if source_groups:
            logger.info(f"    Copying only groups: {source_groups}")
            for group in source_groups:
                group_src = windows_src / group
                group_dst = windows_dst / group
                if group_src.exists():
                    _copy_directory_recursive(group_src, group_dst)
                    logger.info(f"    Copied group '{group}'")
        else:
            _copy_directory_recursive(windows_src, windows_dst)
            logger.info(f"    Copied windows directory")

    return dest_path


def _copy_directory_recursive(src: UPath, dst: UPath) -> None:
    """Recursively copy a directory using UPath (fallback method)."""
    dst.mkdir(parents=True, exist_ok=True)

    for item in src.iterdir():
        if item.is_dir():
            _copy_directory_recursive(item, dst / item.name)
        else:
            with item.open("rb") as f:
                data = f.read()
            with (dst / item.name).open("wb") as f:
                f.write(data)


def copy_dataset(
    source_path: str,
    name: str,
    source_groups: list[str] | None = None,
    untar_source: bool = False,
) -> str:
    """Copy an rslearn dataset to our Weka location.

    Dispatches to the fastest available copy method based on source path:
    - GCS tar.gz (gs://, untar_source=True) -> gsutil stream + tar extract
    - GCS (gs://) -> gsutil -m cp -r (parallel transfers)
    - Local/Weka (/weka, /) -> find + xargs -P (parallel local copy)
    - Other -> UPath generic copy (fallback)

    Args:
        source_path: Path to source rslearn dataset
        name: Name for the copied dataset
        source_groups: If specified, only copy these groups (subdirs under windows/).
            If None, copies everything.
        untar_source: If True, source_path is a .tar.gz archive on GCS that
            will be streamed and extracted directly to the destination.

    Returns:
        Path to the copied dataset on Weka
    """
    dest_path = f"{EVAL_DATASETS_BASE_PATH}/{name}"

    logger.info(f"=== Dataset Copy ===")
    logger.info(f"  Source: {source_path}")
    logger.info(f"  Destination: {dest_path}")
    if source_groups:
        logger.info(f"  Filtering to groups: {source_groups}")
    else:
        logger.info(f"  Copying all groups")

    # Check if destination already exists
    if Path(dest_path).exists():
        logger.warning(f"  Destination already exists, skipping copy...")
        return dest_path

    # Dispatch to appropriate copy method based on source
    if untar_source and source_path.startswith("gs://"):
        _copy_from_gcs_tar(source_path, dest_path)
    elif source_path.startswith("gs://"):
        _copy_from_gcs(source_path, dest_path, source_groups)
    elif source_path.startswith("/weka") or source_path.startswith("/"):
        _copy_local(source_path, dest_path, source_groups)
    else:
        _copy_generic(source_path, dest_path, source_groups)

    logger.info(f"  Dataset copy complete: {dest_path}")
    return dest_path


# =============================================================================
# Split Management Utilities
# =============================================================================


def scan_windows_and_splits(
    dataset_path: str,
    source_groups: list[str] | None = None,
    source_tags: dict[str, str] | None = None,
) -> dict[str, list[tuple[str, str]]]:
    """Scan windows using rslearn's native load_windows and determine splits.

    Args:
        dataset_path: Path to rslearn dataset
        source_groups: Filter to these groups only
        source_tags: Filter windows by tags. Dict of {key: value}.
            Empty string value means "key exists" (any value).

    Returns:
        Dict mapping split name -> list of (group, window_name) tuples
        e.g., {"train": [("train", "w1"), ("train", "w2")], "val": [...], "test": [...]}
    """
    logger.info(f"  Opening dataset at {dataset_path}...")
    dataset = RslearnDataset(UPath(dataset_path))

    # Use rslearn's native load_windows with parallel loading
    logger.info(f"  Loading windows (groups={source_groups}, workers={NUM_WORKERS})...")
    windows = dataset.load_windows(groups=source_groups, workers=NUM_WORKERS)
    logger.info(f"  Loaded {len(windows)} windows from dataset")

    # Filter by tags if specified
    if source_tags:
        filtered_windows = []
        for window in windows:
            if not window.options:
                continue
            match = True
            for key, value in source_tags.items():
                if key not in window.options:
                    match = False
                    break
                # Empty value means "key exists" (any value is ok)
                if value and window.options[key] != value:
                    match = False
                    break
            if match:
                filtered_windows.append(window)
        logger.info(f"  Filtered to {len(filtered_windows)} windows matching tags {source_tags}")
        windows = filtered_windows

    # Use string keys for consistency
    splits: dict[str, list[tuple[str, str]]] = {
        str(SplitName.TRAIN): [],
        str(SplitName.VAL): [],
        str(SplitName.TEST): [],
    }

    for window in windows:
        # Tags are in window.options (rslearn native)
        split_val = window.options.get("split") if window.options else None

        if split_val in splits:
            splits[split_val].append((window.group, window.name))
        elif window.group in ("train",):
            splits[str(SplitName.TRAIN)].append((window.group, window.name))
        elif window.group in ("val", "valid", "validation"):
            splits[str(SplitName.VAL)].append((window.group, window.name))
        elif window.group in ("test", "test_hard"):
            splits[str(SplitName.TEST)].append((window.group, window.name))
        else:
            # Unknown group, add to train by default
            splits[str(SplitName.TRAIN)].append((window.group, window.name))

    return splits


def create_missing_splits(
    splits: dict[str, list[tuple[str, str]]],
    val_test_ratio: float = 0.5,
    train_val_ratio: float = 0.8,
    seed: int = 42,
) -> dict[str, list[tuple[str, str]]]:
    """Create missing splits using one of three strategies.

    Paths:
    1. All splits present (train, val, test) - no action needed
    2. Train and val exist, no test - split val into val+test
    3. Any other case - pool all windows and resplit randomly into train/val/test

    Args:
        splits: Current split assignments (split_name -> list of (group, window_name) tuples)
        val_test_ratio: Ratio of val to keep when splitting val into val+test
        train_val_ratio: Ratio of train to keep (rest goes to val, then val splits for test)
        seed: Random seed

    Returns:
        Updated splits dict with string keys
    """
    import random
    random.seed(seed)

    # Make a copy to avoid mutating input, ensure string keys
    splits = {str(k): list(v) for k, v in splits.items()}

    has_train = bool(splits["train"])
    has_val = bool(splits["val"])
    has_test = bool(splits["test"])

    logger.info(f"Split detection: train={has_train} ({len(splits['train'])}), "
                f"val={has_val} ({len(splits['val'])}), "
                f"test={has_test} ({len(splits['test'])})")

    # PATH 1: All splits present - no action needed
    if has_train and has_val and has_test:
        logger.info("PATH 1: All splits present - no splitting needed")
        return splits

    # PATH 2: Train and val exist, no test - split val into val+test
    if has_train and has_val and not has_test:
        logger.info("PATH 2: Have train+val, missing test - splitting val into val+test")
        val_windows = splits["val"]
        random.shuffle(val_windows)
        split_idx = int(len(val_windows) * val_test_ratio)
        splits["val"] = val_windows[:split_idx]
        splits["test"] = val_windows[split_idx:]
        logger.info(f"  Split val: {len(splits['val'])} val, {len(splits['test'])} test")
        return splits

    # PATH 3: Any other case - pool all windows and resplit randomly
    logger.info("PATH 3: Resplitting all windows randomly into train/val/test")

    # Pool all windows from all splits
    all_windows = []
    for split_name in ["train", "val", "test"]:
        all_windows.extend(splits[split_name])
        splits[split_name] = []

    if not all_windows:
        logger.warning("No windows found to split!")
        return splits

    random.shuffle(all_windows)
    total = len(all_windows)

    # Split: train_ratio for train, then split remainder into val/test
    train_end = int(total * train_val_ratio)
    remaining = all_windows[train_end:]
    val_end = int(len(remaining) * val_test_ratio)

    splits["train"] = all_windows[:train_end]
    splits["val"] = remaining[:val_end]
    splits["test"] = remaining[val_end:]

    logger.info(f"  Resplit {total} windows: "
                f"train={len(splits['train'])}, "
                f"val={len(splits['val'])}, "
                f"test={len(splits['test'])}")

    return splits


def write_split_tags(
    dataset_path: str,
    splits: dict[str, list[tuple[str, str]]],
) -> None:
    """Write split tags to window metadata using rslearn's native Window.save().

    Args:
        dataset_path: Path to rslearn dataset
        splits: Dict mapping split name -> list of (group, window_name) tuples
    """
    logger.info(f"  Opening dataset at {dataset_path}...")
    dataset = RslearnDataset(UPath(dataset_path))

    # Load all windows and build a lookup map
    total_windows = sum(len(v) for v in splits.values())
    logger.info(f"  Loading windows for tag writing (workers={NUM_WORKERS})...")
    all_windows = dataset.load_windows(workers=NUM_WORKERS)
    window_map = {(w.group, w.name): w for w in all_windows}
    logger.info(f"  Loaded {len(all_windows)} windows, will update {total_windows} with split tags")

    updated_count = 0
    for split_name, window_ids in splits.items():
        logger.info(f"  Writing '{split_name}' tag to {len(window_ids)} windows...")
        for group_name, window_name in window_ids:
            window = window_map.get((group_name, window_name))
            if window is None:
                logger.warning(f"Window not found: {group_name}/{window_name}")
                continue

            # Update options with our eval split tag (don't overwrite original "split")
            if window.options is None:
                window.options = {}
            window.options[EVAL_SPLIT_TAG_KEY] = str(split_name)

            # Use rslearn's native save method
            window.save()

            # Verify the tag persisted by re-reading metadata.json from disk
            metadata_path = Path(dataset_path) / "windows" / group_name / window_name / "metadata.json"
            with open(metadata_path) as f:
                saved_meta = json.load(f)
            saved_tag = saved_meta.get("options", {}).get(EVAL_SPLIT_TAG_KEY)
            if saved_tag != str(split_name):
                raise RuntimeError(
                    f"write_split_tags: window.save() did not persist "
                    f"{EVAL_SPLIT_TAG_KEY}={split_name} for "
                    f"{group_name}/{window_name}. "
                    f"Got: {saved_tag}"
                )
            updated_count += 1

    logger.info(f"Wrote split tags for {updated_count} windows")


def count_split_stats(
    splits: dict[str, list[tuple[str, str]]],
) -> dict[str, dict[str, Any]]:
    """Count samples per split.

    Args:
        splits: Dict mapping split name -> list of (group, window_name) tuples

    Returns:
        Dict like {"train": {"count": 100}, "val": {"count": 50}, ...}
    """
    stats = {}

    for split_name, window_ids in splits.items():
        stats[split_name] = {
            "count": len(window_ids),
        }

    return stats


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
    logger.info(f"{'='*60}")
    logger.info(f"INGEST START: {config.name}")
    logger.info(f"{'='*60}")
    logger.info(f"Source: {config.source_path}")
    logger.info(f"Model config: {config.olmoearth_run_config_path}")

    # If source is a tar.gz archive, we must extract it before reading config.json
    if config.untar_source:
        logger.info(f"[Step 1/6] Extracting tar archive to Weka (before config load)...")
        weka_path = copy_dataset(config.source_path, config.name, config.source_groups, config.untar_source)
        logger.info(f"[Step 1/6] Extract complete")
        dataset_config_path = UPath(weka_path) / "config.json"
    else:
        weka_path = None
        dataset_config_path = UPath(config.source_path) / "config.json"

    # Step 0a: Load dataset config
    logger.info(f"[Step 0a] Loading dataset config.json...")
    with dataset_config_path.open() as f:
        dataset_dict = json.load(f)
    logger.info(f"[Step 0a] Parsing dataset config...")
    dataset_config = DatasetConfig.model_validate(dataset_dict)
    logger.info(f"[Step 0a] Dataset config loaded successfully")

    # Step 0b: Load model config
    logger.info(f"[Step 0b] Loading model.yaml...")
    model_config_path = UPath(config.olmoearth_run_config_path) / "model.yaml"
    with model_config_path.open() as f:
        model_config = yaml.safe_load(f)
    logger.info(f"[Step 0b] Model config loaded successfully")

    # Step 0c: Extract modalities from dataset config
    logger.info(f"[Step 0c] Extracting modalities from dataset config...")
    modalities = []
    modality_layer_names = []
    max_timesteps_modalities = []
    for layer_name, layer_config in dataset_config.layers.items():
        if layer_config.data_source is None:
            continue
        olmoearth_modality = rslearn_to_olmoearth(layer_name)
        modalities.append(olmoearth_modality.name)
        modality_layer_names.append(layer_name)
        query_config = layer_config.data_source.query_config
        max_timesteps_modalities.append(query_config.max_matches)

    num_timesteps = max(max_timesteps_modalities) if max_timesteps_modalities else 1
    timeseries = num_timesteps > 1
    logger.info(f"[Step 0c] Modalities: {modalities}, timeseries: {timeseries}, num_timesteps: {num_timesteps}")

    # Step 0d: Extract and instantiate the rslearn task from model config
    logger.info(f"[Step 0d] Extracting task from model config...")
    task_wrapper_config = model_config["data"]["init_args"]["task"]
    task_init_args = task_wrapper_config.get("init_args", {})

    if "tasks" in task_init_args:
        # Multi-task: data.init_args.task.init_args.tasks.{task_name}
        tasks_dict = task_init_args["tasks"]
        if len(tasks_dict) != 1:
            raise NotImplementedError(
                "Multiple tasks not supported in this workflow; found: "
                + ", ".join(tasks_dict)
            )
        task_name, task_config = next(iter(tasks_dict.items()))
    else:
        # Single task: data.init_args.task is the task directly
        task_name = "task"
        task_config = task_wrapper_config

    logger.info(f"[Step 0d] Instantiating task '{task_name}': {task_config['class_path']}")
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
    logger.info(f"[Step 0d] Task: {task_name}, num_classes: {num_classes}")



    # Step 1: Copy dataset to Weka (skip if already extracted via untar_source)
    if weka_path is None:
        logger.info(f"[Step 1/6] Copying dataset to Weka...")
        weka_path = copy_dataset(config.source_path, config.name, config.source_groups)
        logger.info(f"[Step 1/6] Copy complete")
    else:
        logger.info(f"[Step 1/6] Already extracted to {weka_path}, skipping copy")

    # Step 2: Scan windows and determine existing splits
    logger.info(f"[Step 2/6] Scanning windows and determining splits...")
    splits = scan_windows_and_splits(
        weka_path,
        source_groups=config.source_groups,
        source_tags=config.source_tags,
    )
    logger.info(f"[Step 2/6] Scan complete: train={len(splits['train'])}, "
                f"val={len(splits['val'])}, test={len(splits['test'])}")

    # Step 3: Create missing splits if needed
    logger.info(f"[Step 3/6] Creating missing splits if needed...")
    splits = create_missing_splits(
        splits,
        val_test_ratio=config.val_test_split_ratio,
        train_val_ratio=config.train_val_split_ratio,
        seed=config.split_seed,
    )
    logger.info(f"[Step 3/6] Split creation complete: train={len(splits['train'])}, "
                f"val={len(splits['val'])}, test={len(splits['test'])}")

    # Step 4: Write split tags to metadata
    logger.info(f"[Step 4/6] Writing split tags to window metadata...")
    write_split_tags(weka_path, splits)
    logger.info(f"[Step 4/6] Split tags written")

    # Step 5: Count split statistics
    logger.info(f"[Step 5/6] Counting split statistics...")
    split_stats = count_split_stats(splits)
    logger.info(f"[Step 5/6] Stats: {split_stats}")

    # Step 6: Compute normalization stats (from copied dataset, use train for stats)
    logger.info(f"[Step 6/6] Computing normalization stats from train split...")
    norm_stats = compute_band_stats_from_model_config(
        model_config_path=str(model_config_path),
        source_path=weka_path,
        groups=config.source_groups,
        tags={EVAL_SPLIT_TAG_KEY: str(SplitName.TRAIN)},
        num_samples=config.num_samples,
    )
    logger.info(f"[Step 6/6] Normalization stats computed for {len(norm_stats)} modalities")

    task_type = rslearn_task_type_to_olmoearth_task_type(rslearn_task)

    # Extract window_size from default_config
    data_init_args = model_config["data"]["init_args"]
    default_config = data_init_args.get("default_config", {})
    window_size = default_config.get("patch_size", 64)

    logger.info(f"Creating EvalDatasetEntry...")
    entry = EvalDatasetEntry(
        name=config.name,
        source_path=config.source_path,
        weka_path=weka_path,
        model_config_path=config.olmoearth_run_config_path,
        task_type=task_type,
        num_classes=num_classes,
        classes=label_values,
        modalities=modalities,
        window_size=window_size,
        timeseries=timeseries,
        num_timesteps=num_timesteps,
        train_split=SplitName.TRAIN,
        val_split=SplitName.VAL,
        test_split=SplitName.TEST,
        split_type=SplitType.TAGS,
        split_tag_key=EVAL_SPLIT_TAG_KEY,
        split_stats=split_stats,
        norm_stats=norm_stats,
    )

    logger.info(f"{'='*60}")
    logger.info(f"INGEST COMPLETE: {config.name}")
    logger.info(f"{'='*60}")
    logger.info(f"  Weka path: {weka_path}")
    logger.info(f"  Task: {task_type}, classes: {num_classes}")
    logger.info(f"  Splits: train={split_stats.get('train', {}).get('count', 0)}, "
                f"val={split_stats.get('val', {}).get('count', 0)}, "
                f"test={split_stats.get('test', {}).get('count', 0)}")

    return entry
