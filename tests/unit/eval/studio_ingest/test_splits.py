"""Tests for Studio ingest split helpers."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from olmoearth_pretrain.evals.studio_ingest import splits

WindowId = tuple[str, str]


@dataclass
class FakeWindow:
    """Minimal rslearn Window stand-in for split helper tests."""

    group: str
    name: str
    options: dict[str, str] | None = None
    root: Path | None = None

    def save(self) -> None:
        """Persist options to the metadata path used by write_split_tags."""
        if self.root is None:
            return
        metadata_path = self.root / "windows" / self.group / self.name / "metadata.json"
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        metadata_path.write_text(json.dumps({"options": self.options}))


def _split_ids(count: int, prefix: str) -> list[WindowId]:
    return [(prefix, f"w{i}") for i in range(count)]


def test_create_missing_splits_noops_when_all_splits_exist() -> None:
    """All-present splits should be returned unchanged."""
    original = {
        "train": _split_ids(1, "train"),
        "val": _split_ids(1, "val"),
        "test": _split_ids(1, "test"),
    }

    assert splits.create_missing_splits(original) == original
    assert original["val"] == [("val", "w0")]


def test_create_missing_splits_splits_validation_when_test_missing() -> None:
    """A missing test split should be carved out of validation windows."""
    result = splits.create_missing_splits(
        {
            "train": _split_ids(2, "train"),
            "val": _split_ids(4, "val"),
            "test": [],
        },
        val_test_ratio=0.5,
        seed=7,
    )

    assert len(result["train"]) == 2
    assert len(result["val"]) == 2
    assert len(result["test"]) == 2
    assert sorted(result["val"] + result["test"]) == _split_ids(4, "val")


def test_create_missing_splits_resplits_single_bucket() -> None:
    """Single-bucket inputs should be rebalanced into train, val, and test."""
    result = splits.create_missing_splits(
        {
            "train": _split_ids(10, "train"),
            "val": [],
            "test": [],
        },
        train_val_ratio=0.6,
        val_test_ratio=0.5,
        seed=11,
    )

    assert len(result["train"]) == 6
    assert len(result["val"]) == 2
    assert len(result["test"]) == 2
    assert sorted(result["train"] + result["val"] + result["test"]) == _split_ids(
        10,
        "train",
    )


def test_create_missing_splits_does_not_mutate_global_random_state() -> None:
    """Split creation should use local RNG state."""
    random.seed(123)
    state = random.getstate()

    splits.create_missing_splits(
        {
            "train": _split_ids(10, "train"),
            "val": [],
            "test": [],
        },
        seed=7,
    )

    assert random.getstate() == state


def test_count_split_stats() -> None:
    """Split stats should report counts per split."""
    assert splits.count_split_stats(
        {
            "train": _split_ids(3, "train"),
            "val": _split_ids(2, "val"),
        }
    ) == {
        "train": {"count": 3},
        "val": {"count": 2},
    }


def test_scan_windows_and_splits_filters_tags_and_detects_splits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Scan should filter source tags and infer splits from tags or groups."""
    windows = [
        FakeWindow("ignored_group", "explicit", {"split": "test", "region": "west"}),
        FakeWindow("valid", "by_group", {"region": "west"}),
        FakeWindow("other", "default_train", {"region": "west"}),
        FakeWindow("train", "filtered_out", {"region": "east"}),
        FakeWindow("train", "no_options", None),
    ]
    load_calls: list[dict[str, Any]] = []

    class FakeDataset:
        def __init__(self, path: object) -> None:
            self.path = path

        def load_windows(
            self,
            groups: list[str] | None = None,
            workers: int | None = None,
        ) -> list[FakeWindow]:
            load_calls.append({"groups": groups, "workers": workers})
            return windows

    monkeypatch.setattr(splits, "RslearnDataset", FakeDataset)
    monkeypatch.setattr(splits, "NUM_WORKERS", 3)

    result = splits.scan_windows_and_splits(
        "/dataset",
        source_groups=["g1"],
        source_tags={"region": "west"},
    )

    assert load_calls == [{"groups": ["g1"], "workers": 3}]
    assert result == {
        "train": [("other", "default_train")],
        "val": [("valid", "by_group")],
        "test": [("ignored_group", "explicit")],
    }


def test_write_split_tags_persists_eval_split_tags(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Split tag writing should persist eval split tags through window save."""
    windows = [
        FakeWindow("train", "w1", {}, tmp_path),
        FakeWindow("val", "w2", None, tmp_path),
    ]
    for window in windows:
        window.save()

    class FakeDataset:
        def __init__(self, path: object) -> None:
            self.path = path

        def load_windows(self, workers: int | None = None) -> list[FakeWindow]:
            return windows

    monkeypatch.setattr(splits, "RslearnDataset", FakeDataset)

    splits.write_split_tags(
        str(tmp_path),
        {
            "train": [("train", "w1")],
            "val": [("val", "w2")],
            "test": [("missing", "w3")],
        },
    )

    train_metadata = json.loads(
        (tmp_path / "windows" / "train" / "w1" / "metadata.json").read_text()
    )
    val_metadata = json.loads(
        (tmp_path / "windows" / "val" / "w2" / "metadata.json").read_text()
    )

    assert train_metadata["options"][splits.EVAL_SPLIT_TAG_KEY] == "train"
    assert val_metadata["options"][splits.EVAL_SPLIT_TAG_KEY] == "val"
