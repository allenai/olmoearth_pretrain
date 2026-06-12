"""Test eval dataset factory imports."""

import subprocess
import sys
from collections.abc import Iterator
from typing import Any

import pytest

import olmoearth_pretrain.evals.datasets as datasets
from olmoearth_pretrain.evals.datasets.configs import DATASET_TO_CONFIG
from olmoearth_pretrain.evals.datasets.registry import (
    BUILTIN_EVAL_DATASET_SPECS,
    get_builtin_eval_dataset_spec,
)


def test_eval_dataset_package_does_not_eager_import_adapters() -> None:
    """Importing the package should not import every dataset adapter."""
    script = """
import sys

import olmoearth_pretrain.evals.datasets as datasets

adapter_modules = [
    "olmoearth_pretrain.evals.datasets.breizhcrops",
    "olmoearth_pretrain.evals.datasets.floods_dataset",
    "olmoearth_pretrain.evals.datasets.geobench_dataset",
    "olmoearth_pretrain.evals.datasets.mados_dataset",
    "olmoearth_pretrain.evals.datasets.pastis_dataset",
    "olmoearth_pretrain.evals.datasets.pretrain_subset",
    "olmoearth_pretrain.evals.datasets.rslearn_dataset",
]

loaded = [module for module in adapter_modules if module in sys.modules]
assert loaded == [], loaded

_ = datasets.GeobenchDataset
assert "olmoearth_pretrain.evals.datasets.geobench_dataset" in sys.modules
loaded = [module for module in adapter_modules if module in sys.modules]
assert loaded == ["olmoearth_pretrain.evals.datasets.geobench_dataset"], loaded
"""
    subprocess.run([sys.executable, "-c", script], check=True)


def test_eval_dataset_package_all_is_lightweight() -> None:
    """Dataset package star imports should not load lazy adapters."""
    script = """
import sys

import olmoearth_pretrain.evals.datasets as datasets

namespace = {}
exec("from olmoearth_pretrain.evals.datasets import *", namespace)

expected = {
    "NormMethod",
    "get_eval_dataset",
    "paths",
    "scale_train_samples",
}

assert set(datasets.__all__) == expected
assert expected.issubset(namespace)
assert "GeobenchDataset" not in namespace
assert "olmoearth_pretrain.evals.datasets.geobench_dataset" not in sys.modules
"""
    subprocess.run([sys.executable, "-c", script], check=True)


class RecordingDataset:
    """Lightweight stand-in for dataset adapters in factory tests."""

    def __init__(self, **kwargs: Any) -> None:
        """Record constructor kwargs for assertions."""
        self.kwargs = kwargs


@pytest.fixture(autouse=True)
def _clear_lazy_dataset_globals() -> Iterator[None]:
    """Remove lazy exports that other tests may have resolved on this module."""
    lazy_names = [spec.adapter_class_name for spec in BUILTIN_EVAL_DATASET_SPECS]
    saved = {
        name: datasets.__dict__.pop(name)
        for name in lazy_names
        if name in vars(datasets)
    }
    yield
    for name in lazy_names:
        datasets.__dict__.pop(name, None)
    datasets.__dict__.update(saved)


def test_builtin_dataset_specs_reference_known_configs() -> None:
    """Built-in factory metadata should stay aligned with hardcoded configs."""
    config_names = set(DATASET_TO_CONFIG)

    for spec in BUILTIN_EVAL_DATASET_SPECS:
        assert set(spec.config_names).issubset(config_names)
        for config_name in spec.config_names:
            assert spec.matches(config_name)


def test_path_backed_builtin_dataset_specs_define_path_keys() -> None:
    """Simple path-backed families should carry their path keys in metadata."""
    specs_by_family = {spec.family: spec for spec in BUILTIN_EVAL_DATASET_SPECS}

    assert specs_by_family["geobench"].path_key == "GEOBENCH_DIR"
    assert specs_by_family["mados"].path_key == "MADOS_DIR"
    assert specs_by_family["mados"].pretrain_norm_warning is not None
    assert specs_by_family["sen1floods11"].path_key == "FLOODS_DIR"
    assert specs_by_family["breizhcrops"].path_key == "BREIZHCROPS_DIR"


@pytest.mark.parametrize(
    ("dataset_name", "family"),
    [
        ("pretrain_subset", "pretrain_subset"),
        ("pretrain_subset_worldcover", "pretrain_subset"),
        ("m-eurosat", "geobench"),
        ("mados", "mados"),
        ("sen1floods11", "sen1floods11"),
        ("pastis", "pastis"),
        ("pastis128", "pastis"),
        ("breizhcrops", "breizhcrops"),
    ],
)
def test_builtin_dataset_spec_matching(dataset_name: str, family: str) -> None:
    """Dataset names should dispatch to the expected built-in family."""
    spec = get_builtin_eval_dataset_spec(dataset_name)

    assert spec is not None
    assert spec.family == family


@pytest.mark.parametrize(
    ("dataset_name", "adapter_name", "env_key", "path_kwarg"),
    [
        ("m-eurosat", "GeobenchDataset", "GEOBENCH_DIR", "geobench_dir"),
        ("mados", "MADOSDataset", "MADOS_DIR", "path_to_splits"),
        ("sen1floods11", "Sen1Floods11Dataset", "FLOODS_DIR", "path_to_splits"),
        ("breizhcrops", "BreizhCropsDataset", "BREIZHCROPS_DIR", "path_to_splits"),
    ],
)
def test_get_eval_dataset_resolves_paths_at_call_time(
    monkeypatch: pytest.MonkeyPatch,
    dataset_name: str,
    adapter_name: str,
    env_key: str,
    path_kwarg: str,
) -> None:
    """Factory path kwargs should use the current environment, not import-time values."""
    monkeypatch.setattr(datasets, adapter_name, RecordingDataset, raising=False)
    monkeypatch.setenv(env_key, f"/tmp/{env_key.lower()}")

    dataset = datasets.get_eval_dataset(dataset_name, split="train")

    assert isinstance(dataset, RecordingDataset)
    assert str(dataset.kwargs[path_kwarg]) == f"/tmp/{env_key.lower()}"


def test_get_eval_dataset_warns_for_mados_pretrain_norm(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """MADOS should keep its dataset-specific pretrain-norm warning."""
    monkeypatch.setattr(datasets, "MADOSDataset", RecordingDataset, raising=False)
    monkeypatch.setenv("MADOS_DIR", "/tmp/mados")

    datasets.get_eval_dataset(
        "mados",
        split="train",
        norm_stats_from_pretrained=True,
    )

    assert "MADOS has very different norm stats" in caplog.text


@pytest.mark.parametrize(
    ("dataset_name", "env_key", "expected_kwarg"),
    [
        ("pastis", "PASTIS_DIR", "path_to_splits"),
        ("pastis128", "PASTIS_DIR_ORIG", "path_to_splits"),
    ],
)
def test_get_eval_dataset_resolves_pastis_paths_at_call_time(
    monkeypatch: pytest.MonkeyPatch,
    dataset_name: str,
    env_key: str,
    expected_kwarg: str,
) -> None:
    """PASTIS variants should choose their own split path and shared partition path."""
    monkeypatch.setattr(datasets, "PASTISRDataset", RecordingDataset, raising=False)
    monkeypatch.setenv(env_key, f"/tmp/{env_key.lower()}")
    monkeypatch.setenv("PASTIS_DIR_PARTITION", "/tmp/pastis_partition")

    dataset = datasets.get_eval_dataset(
        dataset_name,
        split="valid",
        input_modalities=["sentinel2_l2a"],
    )

    assert isinstance(dataset, RecordingDataset)
    assert str(dataset.kwargs[expected_kwarg]) == f"/tmp/{env_key.lower()}"
    assert str(dataset.kwargs["dir_partition"]) == "/tmp/pastis_partition"


def test_get_eval_dataset_pretrain_subset_uses_scaled_label_fraction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Pretrain probe kwargs should preserve the old default construction behavior."""
    monkeypatch.setattr(
        datasets, "PretrainSubsetDataset", RecordingDataset, raising=False
    )

    dataset = datasets.get_eval_dataset(
        "pretrain_subset_worldcover",
        split="train",
        h5py_dir="/tmp/h5",
        input_modalities=["sentinel2_l2a"],
        label_fraction=0.5,
        pretrain_train_samples=100,
    )

    assert isinstance(dataset, RecordingDataset)
    assert dataset.kwargs["h5py_dir"] == "/tmp/h5"
    assert dataset.kwargs["training_modalities"] == ["sentinel2_l2a"]
    assert dataset.kwargs["train_samples"] == 50


def test_get_eval_dataset_falls_back_to_studio_registry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Unknown names should continue to use the external dataset registry."""
    entry = object()

    def fake_from_registry_entry(**kwargs: Any) -> dict[str, Any]:
        return kwargs

    monkeypatch.setattr(
        datasets, "get_dataset_entry", lambda name: entry, raising=False
    )
    monkeypatch.setattr(
        datasets, "from_registry_entry", fake_from_registry_entry, raising=False
    )

    dataset = datasets.get_eval_dataset(
        "registry-dataset",
        split="test",
        input_modalities=["sentinel1"],
        label_fraction=0.25,
    )

    assert dataset["entry"] is entry
    assert dataset["split"] == "test"
    assert dataset["norm_stats_from_pretrained"] is None
    assert dataset["input_modalities_override"] == ["sentinel1"]
    assert dataset["label_fraction"] == 0.25
