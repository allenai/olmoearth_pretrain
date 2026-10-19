"""Unit tests for eval dataset config provenance."""

import hashlib
from pathlib import Path
from typing import Any

import pytest

from olmoearth_pretrain.evals.studio_ingest.provenance import (
    DATASET_CONFIGS_DIR,
    packaged_config_name,
    packaged_model_yaml_path,
    verify_config_json_hash,
)
from olmoearth_pretrain.evals.studio_ingest.schema import EvalDatasetEntry


def _make_entry(**overrides: Any) -> EvalDatasetEntry:
    defaults: dict[str, Any] = dict(
        name="prov_test",
        source_path="/tmp/source",
        weka_path="/tmp/weka",
        task_type="classification",
        num_classes=2,
        modalities=["sentinel2_l2a"],
    )
    defaults.update(overrides)
    return EvalDatasetEntry(**defaults)


def test_packaged_config_name_inside_package() -> None:
    """Packaged dataset config dirs map to their name."""
    assert (
        packaged_config_name(str(DATASET_CONFIGS_DIR / "pastis_rslearn"))
        == "pastis_rslearn"
    )


def test_packaged_config_name_outside_package(tmp_path: Path) -> None:
    """Config dirs outside the packaged dataset_configs return None."""
    assert packaged_config_name(str(tmp_path)) is None


def test_packaged_model_yaml_path_existing() -> None:
    """Resolving a packaged config returns an existing path."""
    assert Path(packaged_model_yaml_path("pastis_rslearn")).exists()


def test_packaged_model_yaml_path_missing_raises() -> None:
    """Resolving a missing packaged config raises instead of falling back."""
    with pytest.raises(FileNotFoundError, match="does not exist"):
        packaged_model_yaml_path("does_not_exist")


def test_model_yaml_path_falls_back_to_weka_copy() -> None:
    """Entries without packaged_config keep reading the Weka copy."""
    entry = _make_entry()
    assert entry.model_yaml_path == "/tmp/weka/model.yaml"


def test_model_yaml_path_resolves_packaged_config() -> None:
    """Entries with packaged_config read the packaged model.yaml."""
    entry = _make_entry(packaged_config="pastis_rslearn")
    assert entry.model_yaml_path.endswith("dataset_configs/pastis_rslearn/model.yaml")
    assert Path(entry.model_yaml_path).exists()


def test_model_yaml_path_packaged_config_missing_raises() -> None:
    """A dangling packaged_config raises instead of falling back."""
    entry = _make_entry(packaged_config="nope")
    with pytest.raises(FileNotFoundError):
        _ = entry.model_yaml_path


def test_verify_config_json_hash_match(tmp_path: Path) -> None:
    """A matching recorded hash verifies quietly."""
    (tmp_path / "config.json").write_bytes(b"{}")
    expected = hashlib.sha256(b"{}").hexdigest()
    actual = verify_config_json_hash("ds", str(tmp_path), expected)
    assert actual == expected


def test_verify_config_json_hash_mismatch_raises(tmp_path: Path) -> None:
    """A drifted config.json fails loudly."""
    (tmp_path / "config.json").write_bytes(b"{}")
    with pytest.raises(ValueError, match="drifted"):
        verify_config_json_hash("ds", str(tmp_path), "0" * 64)


def test_verify_config_json_hash_no_recorded_hash_warns_only(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Entries without a recorded hash warn but do not fail."""
    (tmp_path / "config.json").write_bytes(b"{}")
    with caplog.at_level("WARNING"):
        actual = verify_config_json_hash("ds", str(tmp_path), None)
    assert actual == hashlib.sha256(b"{}").hexdigest()
    assert "no config_json_sha256" in caplog.text


def test_verify_config_json_hash_missing_file_warns_only(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """A missing config.json warns but does not fail."""
    with caplog.at_level("WARNING"):
        assert verify_config_json_hash("ds", str(tmp_path), "0" * 64) is None
    assert "no config.json" in caplog.text


def test_registry_entry_roundtrips_provenance_fields() -> None:
    """Provenance fields survive a model_dump/model_validate roundtrip."""
    entry = _make_entry(
        packaged_config="pastis_rslearn",
        config_json_sha256="a" * 64,
    )
    dumped = entry.model_dump(mode="json")
    reloaded = EvalDatasetEntry.model_validate(dumped)
    assert reloaded.packaged_config == entry.packaged_config
    assert reloaded.config_json_sha256 == "a" * 64
