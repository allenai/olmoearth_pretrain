"""Tests for Studio ingest metadata helpers."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from olmoearth_pretrain.evals.studio_ingest import ingest
from olmoearth_pretrain.evals.task_types import TaskType


def _layer(max_matches: int | None) -> SimpleNamespace:
    data_source = None
    if max_matches is not None:
        data_source = SimpleNamespace(
            query_config=SimpleNamespace(max_matches=max_matches)
        )
    return SimpleNamespace(data_source=data_source)


def test_extract_modality_metadata_skips_unusable_layers() -> None:
    """Modality metadata should skip output, unmapped, and no-source layers."""
    dataset_config = SimpleNamespace(
        layers={
            "sentinel2": _layer(1),
            "pre_sentinel1": _layer(3),
            "output": _layer(None),
            "unknown": _layer(5),
        }
    )

    metadata = ingest._extract_modality_metadata(dataset_config)

    assert metadata.modalities == ["sentinel2_l2a", "sentinel1"]
    assert metadata.timeseries is True
    assert metadata.num_timesteps == 3


def test_extract_modality_metadata_defaults_to_single_timestep() -> None:
    """No mapped source layers should default to a single timestep."""
    dataset_config = SimpleNamespace(layers={"output": _layer(None)})

    metadata = ingest._extract_modality_metadata(dataset_config)

    assert metadata.modalities == []
    assert metadata.timeseries is False
    assert metadata.num_timesteps == 1


def test_extract_task_metadata_handles_single_task_num_classes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Single task configs should use explicit num_classes."""
    monkeypatch.setattr(ingest, "instantiate_from_config", lambda config: object())
    monkeypatch.setattr(
        ingest,
        "rslearn_task_type_to_olmoearth_task_type",
        lambda task: TaskType.SEGMENTATION,
    )

    metadata = ingest._extract_task_metadata(
        {
            "data": {
                "init_args": {
                    "task": {
                        "class_path": "example.Task",
                        "init_args": {"num_classes": 4},
                    }
                }
            }
        }
    )

    assert metadata.task_type == TaskType.SEGMENTATION
    assert metadata.num_classes == 4
    assert metadata.label_values == ["0", "1", "2", "3"]


def test_extract_task_metadata_handles_one_nested_task_classes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One nested task should use class-list length for num_classes."""
    seen_configs: list[dict[str, Any]] = []

    def fake_instantiate(config: dict[str, Any]) -> object:
        seen_configs.append(config)
        return object()

    monkeypatch.setattr(ingest, "instantiate_from_config", fake_instantiate)
    monkeypatch.setattr(
        ingest,
        "rslearn_task_type_to_olmoearth_task_type",
        lambda task: TaskType.CLASSIFICATION,
    )

    task_config = {
        "class_path": "example.Task",
        "init_args": {"classes": ["forest", "water"]},
    }
    metadata = ingest._extract_task_metadata(
        {
            "data": {
                "init_args": {
                    "task": {
                        "class_path": "example.Wrapper",
                        "init_args": {"tasks": {"landcover": task_config}},
                    }
                }
            }
        }
    )

    assert seen_configs == [task_config]
    assert metadata.task_type == TaskType.CLASSIFICATION
    assert metadata.num_classes == 2
    assert metadata.label_values == ["0", "1"]


def test_extract_task_metadata_rejects_multiple_nested_tasks() -> None:
    """Multiple nested tasks should remain unsupported."""
    with pytest.raises(NotImplementedError, match="Multiple tasks not supported"):
        ingest._extract_task_metadata(
            {
                "data": {
                    "init_args": {
                        "task": {
                            "class_path": "example.Wrapper",
                            "init_args": {
                                "tasks": {
                                    "a": {"class_path": "example.A"},
                                    "b": {"class_path": "example.B"},
                                }
                            },
                        }
                    }
                }
            }
        )


def test_extract_task_metadata_requires_class_count(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Task configs without a class count should fail clearly."""
    monkeypatch.setattr(ingest, "instantiate_from_config", lambda config: object())

    with pytest.raises(ValueError, match="Could not determine num_classes"):
        ingest._extract_task_metadata(
            {
                "data": {
                    "init_args": {
                        "task": {
                            "class_path": "example.Task",
                            "init_args": {},
                        }
                    }
                }
            }
        )


def test_extract_window_size_prefers_crop_size(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Window size should prefer configured crop_size over metadata inference."""
    monkeypatch.setattr(
        ingest,
        "_infer_window_size",
        lambda dataset_path: pytest.fail("should not infer when crop_size exists"),
    )

    assert (
        ingest._extract_window_size(
            {"data": {"init_args": {"default_config": {"crop_size": 128}}}},
            "/dataset",
        )
        == 128
    )


def test_extract_window_size_falls_back_to_window_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Window size should fall back to metadata inference when crop_size is absent."""
    monkeypatch.setattr(ingest, "_infer_window_size", lambda dataset_path: 96)

    assert (
        ingest._extract_window_size(
            {"data": {"init_args": {"default_config": {}}}},
            "/dataset",
        )
        == 96
    )


def test_load_dataset_config_removes_output_layer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Dataset config loading should persist removal of the deprecated output layer."""
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "layers": {
                    "sentinel2": {"data_source": None},
                    "output": {"format": {"name": "geojson"}},
                }
            }
        )
    )
    captured: list[dict[str, Any]] = []

    class FakeDatasetConfig:
        @classmethod
        def model_validate(cls, value: dict[str, Any]) -> dict[str, Any]:
            captured.append(value)
            return value

    monkeypatch.setattr(ingest, "DatasetConfig", FakeDatasetConfig)

    result = ingest._load_dataset_config(str(tmp_path))

    assert result == captured[0]
    assert set(captured[0]["layers"]) == {"sentinel2"}
    persisted = json.loads(config_path.read_text())
    assert set(persisted["layers"]) == {"sentinel2"}
