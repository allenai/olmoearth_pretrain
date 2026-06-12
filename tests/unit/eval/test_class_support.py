"""Tests for precomputed eval class support metadata."""

from pathlib import Path

import pandas as pd

from olmoearth_pretrain.evals.class_support import (
    EvalLabeledClassMode,
    build_class_support_from_split_dir,
    labeled_classes_for_split,
    write_class_support,
)


def test_labeled_classes_from_summary(tmp_path: Path) -> None:
    """Class-summary CSVs should produce per-mode labeled-class lists."""
    summary = pd.DataFrame(
        {
            "class_id": [0, 1, 2],
            "class_name": ["a", "b", "c"],
            "anchor_class_count": [0, 2, 0],
            "tile_presence_count": [0, 3, 1],
            "pixel_count": [0, 10, 0],
        }
    )
    split_dir = tmp_path / "split"
    split_dir.mkdir()
    summary.to_csv(split_dir / "valid_class_summary.csv", index=False)

    payload = build_class_support_from_split_dir(split_dir)
    assert payload["num_classes"] == 3
    assert payload["splits"]["valid"]["anchor_class"] == [1]
    assert payload["splits"]["valid"]["tile_presence"] == [1, 2]
    assert payload["splits"]["valid"]["pixels"] == [1]


def test_write_and_load_class_support(tmp_path: Path) -> None:
    """write_class_support should round-trip through build_class_support_from_split_dir."""
    split_dir = tmp_path / "split"
    split_dir.mkdir()
    summary = pd.DataFrame(
        {
            "class_id": [0, 1],
            "class_name": ["a", "b"],
            "tile_presence_count": [1, 0],
            "pixel_count": [5, 0],
        }
    )
    summary.to_csv(split_dir / "test_class_summary.csv", index=False)
    write_class_support(split_dir)

    payload = build_class_support_from_split_dir(split_dir)
    labeled = labeled_classes_for_split(
        payload,
        split="test",
        mode=EvalLabeledClassMode.TILE_PRESENCE,
    )
    assert labeled == [0]
