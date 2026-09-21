"""Unit tests for eval dataset config provenance."""

import hashlib
from pathlib import Path
from typing import Any

import pytest

from olmoearth_pretrain.evals.studio_ingest.provenance import (
    REPO_ROOT_ENV_VAR,
    find_repo_root,
    repo_relative_config_dir,
    resolve_repo_config_path,
    sha256_of_file,
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


def test_find_repo_root_locates_checkout() -> None:
    """find_repo_root walks up from the package to the checkout root."""
    root = find_repo_root()
    assert root is not None
    assert (root / "pyproject.toml").exists()
    assert (root / "olmoearth_pretrain").is_dir()


def test_find_repo_root_env_override(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The env var override wins over walking up from __file__."""
    monkeypatch.setenv(REPO_ROOT_ENV_VAR, str(tmp_path))
    assert find_repo_root() == tmp_path


def test_find_repo_root_env_override_missing_dir(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A bad env override fails loudly instead of falling back."""
    monkeypatch.setenv(REPO_ROOT_ENV_VAR, "/nonexistent/nowhere")
    with pytest.raises(FileNotFoundError):
        find_repo_root()


def test_sha256_of_file(tmp_path: Path) -> None:
    """sha256_of_file matches hashlib over the raw bytes."""
    f = tmp_path / "config.json"
    f.write_bytes(b'{"layers": {}}')
    assert sha256_of_file(f) == hashlib.sha256(b'{"layers": {}}').hexdigest()


def test_repo_relative_config_dir_inside_repo() -> None:
    """Config dirs inside the checkout become repo-relative paths."""
    root = find_repo_root()
    assert root is not None
    inside = root / "data" / "rslearn_dataset_configs" / "pastis_rslearn"
    assert (
        repo_relative_config_dir(str(inside))
        == "data/rslearn_dataset_configs/pastis_rslearn"
    )


def test_repo_relative_config_dir_outside_repo(tmp_path: Path) -> None:
    """Config dirs outside the checkout return None."""
    assert repo_relative_config_dir(str(tmp_path)) is None


def test_resolve_repo_config_path_existing() -> None:
    """Resolving a committed config returns an existing absolute path."""
    resolved = resolve_repo_config_path(
        "data/rslearn_dataset_configs/pastis_rslearn", "model.yaml"
    )
    assert Path(resolved).exists()


def test_resolve_repo_config_path_missing_file() -> None:
    """Resolving a missing repo config raises instead of falling back."""
    with pytest.raises(FileNotFoundError, match="does not exist"):
        resolve_repo_config_path(
            "data/rslearn_dataset_configs/does_not_exist", "model.yaml"
        )


def test_model_yaml_path_falls_back_to_weka_copy() -> None:
    """Entries without config_repo_dir keep reading the Weka copy."""
    entry = _make_entry()
    assert entry.model_yaml_path == "/tmp/weka/model.yaml"


def test_model_yaml_path_resolves_repo_config() -> None:
    """Entries with config_repo_dir read the git-tracked model.yaml."""
    entry = _make_entry(config_repo_dir="data/rslearn_dataset_configs/pastis_rslearn")
    assert entry.model_yaml_path.endswith(
        "data/rslearn_dataset_configs/pastis_rslearn/model.yaml"
    )
    assert Path(entry.model_yaml_path).exists()


def test_model_yaml_path_repo_config_missing_raises() -> None:
    """A dangling config_repo_dir raises instead of falling back."""
    entry = _make_entry(config_repo_dir="data/rslearn_dataset_configs/nope")
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
        config_repo_dir="data/rslearn_dataset_configs/pastis_rslearn",
        config_json_sha256="a" * 64,
    )
    dumped = entry.model_dump(mode="json")
    reloaded = EvalDatasetEntry.model_validate(dumped)
    assert reloaded.config_repo_dir == entry.config_repo_dir
    assert reloaded.config_json_sha256 == "a" * 64
