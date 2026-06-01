"""Precomputed labeled-class metadata for downstream eval splits."""

from __future__ import annotations

import json
from enum import StrEnum
from pathlib import Path

import pandas as pd

CLASS_SUPPORT_FILENAME = "class_support.json"


class EvalLabeledClassMode(StrEnum):
    """Which precomputed class list to use when averaging macro metrics."""

    TILE_PRESENCE = "tile_presence"
    PIXELS = "pixels"


def labeled_classes_from_summary(summary: pd.DataFrame) -> dict[str, list[int]]:
    """Derive labeled-class ids from a split class summary table."""
    tile_presence = summary.loc[
        summary["tile_presence_count"] > 0, "class_id"
    ].astype(int)
    pixels = summary.loc[summary["pixel_count"] > 0, "class_id"].astype(int)
    return {
        EvalLabeledClassMode.TILE_PRESENCE.value: tile_presence.tolist(),
        EvalLabeledClassMode.PIXELS.value: pixels.tolist(),
    }


def build_class_support_from_split_dir(split_dir: str | Path) -> dict:
    """Build class-support metadata from existing *_class_summary.csv files."""
    split_dir = Path(split_dir)
    splits: dict[str, dict[str, list[int]]] = {}
    num_classes: int | None = None
    for split in ("train", "valid", "test"):
        summary_path = split_dir / f"{split}_class_summary.csv"
        if not summary_path.exists():
            continue
        summary = pd.read_csv(summary_path)
        if num_classes is None:
            num_classes = int(summary["class_id"].max()) + 1
        splits[split] = labeled_classes_from_summary(summary)
    if num_classes is None:
        raise ValueError(f"No class summary CSVs found under {split_dir}")
    return {"num_classes": num_classes, "splits": splits}


def write_class_support(split_dir: str | Path) -> Path:
    """Write ``class_support.json`` next to a pretrain split directory."""
    split_dir = Path(split_dir)
    payload = build_class_support_from_split_dir(split_dir)
    output_path = split_dir / CLASS_SUPPORT_FILENAME
    output_path.write_text(json.dumps(payload, indent=2) + "\n")
    return output_path


def load_class_support(split_dir: str | Path) -> dict | None:
    """Load precomputed class support for a split directory, if present."""
    path = Path(split_dir) / CLASS_SUPPORT_FILENAME
    if not path.exists():
        return None
    return json.loads(path.read_text())


def labeled_classes_for_split(
    class_support: dict | None,
    split: str,
    mode: EvalLabeledClassMode | str,
    override: list[int] | None = None,
) -> list[int] | None:
    """Return the configured labeled-class ids for one eval split."""
    if override is not None:
        return list(override)
    if class_support is None:
        return None
    normalized_split = "valid" if split == "val" else split
    split_entry = class_support.get("splits", {}).get(normalized_split)
    if split_entry is None:
        return None
    mode_value = EvalLabeledClassMode(mode).value
    return list(split_entry[mode_value])
