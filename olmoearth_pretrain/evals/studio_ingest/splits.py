"""Split detection and tagging helpers for Studio ingest."""

from __future__ import annotations

import json
import logging
import os
import random
from pathlib import Path
from typing import Any

from rslearn.dataset.dataset import Dataset as RslearnDataset
from upath import UPath

from olmoearth_pretrain.evals.studio_ingest.tags import tags_match_options
from olmoearth_pretrain.evals.task_types import SplitName

logger = logging.getLogger(__name__)

# Tag key for eval splits. Use our own key to avoid overwriting original split info.
EVAL_SPLIT_TAG_KEY = "eval_split"

_default_workers = (os.cpu_count() or 1) - 1
NUM_WORKERS = int(os.environ.get("OLMOEARTH_INGEST_WORKERS", _default_workers))


def scan_windows_and_splits(
    dataset_path: str,
    source_groups: list[str] | None = None,
    source_tags: dict[str, str] | None = None,
) -> dict[str, list[tuple[str, str]]]:
    """Scan windows using rslearn's native load_windows and determine splits."""
    logger.info(f"  Opening dataset at {dataset_path}...")
    dataset = RslearnDataset(UPath(dataset_path))

    logger.info(f"  Loading windows (groups={source_groups}, workers={NUM_WORKERS})...")
    windows = dataset.load_windows(groups=source_groups, workers=NUM_WORKERS)
    logger.info(f"  Loaded {len(windows)} windows from dataset")

    if source_tags:
        filtered_windows = []
        for window in windows:
            if tags_match_options(window.options, source_tags):
                filtered_windows.append(window)
        logger.info(
            f"  Filtered to {len(filtered_windows)} windows matching tags {source_tags}"
        )
        windows = filtered_windows

    splits: dict[str, list[tuple[str, str]]] = {
        SplitName.TRAIN: [],
        SplitName.VAL: [],
        SplitName.TEST: [],
    }

    for window in windows:
        split_val = window.options.get("split") if window.options else None

        if split_val in splits:
            splits[split_val].append((window.group, window.name))
        elif window.group == "train":
            splits[SplitName.TRAIN].append((window.group, window.name))
        elif window.group in ("val", "valid", "validation"):
            splits[SplitName.VAL].append((window.group, window.name))
        elif window.group in ("test", "test_hard"):
            splits[SplitName.TEST].append((window.group, window.name))
        else:
            splits[SplitName.TRAIN].append((window.group, window.name))

    return splits


def create_missing_splits(
    splits: dict[str, list[tuple[str, str]]],
    val_test_ratio: float = 0.5,
    train_val_ratio: float = 0.8,
    seed: int = 42,
) -> dict[str, list[tuple[str, str]]]:
    """Create train, val, and test splits when one or more are missing."""
    rng = random.Random(seed)

    split_map = {str(k): list(v) for k, v in splits.items()}

    has_train = bool(split_map["train"])
    has_val = bool(split_map["val"])
    has_test = bool(split_map["test"])

    logger.info(
        f"Split detection: train={has_train} ({len(split_map['train'])}), "
        f"val={has_val} ({len(split_map['val'])}), "
        f"test={has_test} ({len(split_map['test'])})"
    )

    if has_train and has_val and has_test:
        logger.info("PATH 1: All splits present - no splitting needed")
        return split_map

    if has_train and has_val and not has_test:
        logger.info(
            "PATH 2: Have train+val, missing test - splitting val into val+test"
        )
        val_windows = split_map["val"]
        rng.shuffle(val_windows)
        split_idx = int(len(val_windows) * val_test_ratio)
        split_map["val"] = val_windows[:split_idx]
        split_map["test"] = val_windows[split_idx:]
        logger.info(
            f"  Split val: {len(split_map['val'])} val, {len(split_map['test'])} test"
        )
        return split_map

    if has_train and has_test and not has_val:
        logger.info(
            "PATH 3: Have train+test, missing val - splitting test into val+test"
        )
        test_windows = split_map["test"]
        rng.shuffle(test_windows)
        split_idx = int(len(test_windows) * val_test_ratio)
        split_map["val"] = test_windows[:split_idx]
        split_map["test"] = test_windows[split_idx:]
        logger.info(
            f"  Split test: {len(split_map['val'])} val, {len(split_map['test'])} test"
        )
        return split_map

    logger.info("PATH 4: Resplitting all windows randomly into train/val/test")

    all_windows = []
    for split_name in ["train", "val", "test"]:
        all_windows.extend(split_map[split_name])
        split_map[split_name] = []

    if not all_windows:
        logger.warning("No windows found to split!")
        return split_map

    rng.shuffle(all_windows)
    total = len(all_windows)

    train_end = int(total * train_val_ratio)
    remaining = all_windows[train_end:]
    val_end = int(len(remaining) * val_test_ratio)

    split_map["train"] = all_windows[:train_end]
    split_map["val"] = remaining[:val_end]
    split_map["test"] = remaining[val_end:]

    logger.info(
        f"  Resplit {total} windows: "
        f"train={len(split_map['train'])}, "
        f"val={len(split_map['val'])}, "
        f"test={len(split_map['test'])}"
    )

    return split_map


def write_split_tags(
    dataset_path: str,
    splits: dict[str, list[tuple[str, str]]],
) -> None:
    """Write split tags to window metadata using rslearn's native Window.save()."""
    logger.info(f"  Opening dataset at {dataset_path}...")
    dataset = RslearnDataset(UPath(dataset_path))

    total_windows = sum(len(v) for v in splits.values())
    logger.info(f"  Loading windows for tag writing (workers={NUM_WORKERS})...")
    all_windows = dataset.load_windows(workers=NUM_WORKERS)
    window_map = {(w.group, w.name): w for w in all_windows}
    logger.info(
        f"  Loaded {len(all_windows)} windows, will update {total_windows} with split tags"
    )

    updated_count = 0
    for split_name, window_ids in splits.items():
        logger.info(f"  Writing '{split_name}' tag to {len(window_ids)} windows...")
        for group_name, window_name in window_ids:
            window = window_map.get((group_name, window_name))
            if window is None:
                logger.warning(f"Window not found: {group_name}/{window_name}")
                continue

            if window.options is None:
                window.options = {}
            window.options[EVAL_SPLIT_TAG_KEY] = str(split_name)
            window.save()

            metadata_path = (
                Path(dataset_path)
                / "windows"
                / group_name
                / window_name
                / "metadata.json"
            )
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
    """Count samples per split."""
    return {
        split_name: {
            "count": len(window_ids),
        }
        for split_name, window_ids in splits.items()
    }
