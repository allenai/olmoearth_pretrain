"""Tests for Studio ingest copy helpers."""

import json
from pathlib import Path

import pytest
from upath import UPath

from olmoearth_pretrain.evals.studio_ingest import copying
from olmoearth_pretrain.evals.studio_ingest.copying import (
    _copy_directory_recursive,
    _resolve_dataset_root,
    _tar_copy_cmd,
    _window_matches_tags,
    prepare_copied_dataset_config,
)


def test_tar_copy_cmd_optionally_uses_pv() -> None:
    """Streaming copy command should include pv only when requested."""
    assert _tar_copy_cmd("/src", "/dst", use_pv=False) == (
        "tar cf - -C /src . | tar xf - -C /dst"
    )
    assert _tar_copy_cmd("/src", "/dst", use_pv=True) == (
        "tar cf - -C /src . | pv | tar xf - -C /dst"
    )
    assert _tar_copy_cmd("/src with space", "/dst with space", use_pv=False) == (
        "tar cf - -C '/src with space' . | tar xf - -C '/dst with space'"
    )


def test_window_matches_tags_supports_exact_and_key_exists(tmp_path: Path) -> None:
    """Tag matching should support exact values and key-exists checks."""
    metadata_path = tmp_path / "metadata.json"
    metadata_path.write_text(
        json.dumps({"options": {"split": "train", "region": "west"}})
    )

    assert _window_matches_tags(metadata_path, {"split": "train"})
    assert _window_matches_tags(metadata_path, {"region": ""})
    assert not _window_matches_tags(metadata_path, {"split": "val"})
    assert not _window_matches_tags(metadata_path, {"missing": ""})


def test_window_matches_tags_returns_false_for_bad_metadata(tmp_path: Path) -> None:
    """Invalid metadata should be treated as not matching."""
    metadata_path = tmp_path / "metadata.json"
    metadata_path.write_text("{not-json")

    assert not _window_matches_tags(metadata_path, {"split": "train"})


def test_resolve_dataset_root_handles_nested_archive_extract(tmp_path: Path) -> None:
    """Tar extracts with a single nested dataset folder should resolve to it."""
    nested = tmp_path / "dataset"
    (nested / "windows").mkdir(parents=True)

    assert _resolve_dataset_root(str(tmp_path)) == str(nested)
    assert _resolve_dataset_root(str(nested)) == str(nested)


def test_copy_directory_recursive_copies_nested_files(tmp_path: Path) -> None:
    """Generic UPath copy should preserve nested directory contents."""
    src = tmp_path / "src"
    dst = tmp_path / "dst"
    (src / "nested").mkdir(parents=True)
    (src / "root.txt").write_text("root")
    (src / "nested" / "child.txt").write_text("child")

    _copy_directory_recursive(UPath(src), UPath(dst))

    assert (dst / "root.txt").read_text() == "root"
    assert (dst / "nested" / "child.txt").read_text() == "child"


def test_prepare_copied_dataset_config_writes_canonical_configs(
    tmp_path: Path,
) -> None:
    """Copied dataset preparation should provide config.json and model.yaml."""
    dataset_path = tmp_path / "dataset"
    model_config_dir = tmp_path / "model"
    dataset_path.mkdir()
    model_config_dir.mkdir()
    (model_config_dir / "dataset.json").write_text('{"layers": {}}')
    (model_config_dir / "model.yaml").write_text("data: {}\n")

    prepare_copied_dataset_config(str(dataset_path), str(model_config_dir))

    assert (dataset_path / "config.json").read_text() == '{"layers": {}}'
    assert (dataset_path / "model.yaml").read_text() == "data: {}\n"


def test_prepare_copied_dataset_config_accepts_direct_yaml_path(
    tmp_path: Path,
) -> None:
    """Copied dataset preparation should accept a direct YAML config path."""
    dataset_path = tmp_path / "dataset"
    model_config_dir = tmp_path / "model"
    dataset_path.mkdir()
    model_config_dir.mkdir()
    yaml_path = model_config_dir / "olmoearth_run.yaml"
    (model_config_dir / "dataset.json").write_text('{"layers": {}}')
    yaml_path.write_text("data: {}\n")

    prepare_copied_dataset_config(str(dataset_path), str(yaml_path))

    assert (dataset_path / "config.json").read_text() == '{"layers": {}}'
    assert (dataset_path / "model.yaml").read_text() == "data: {}\n"


def test_copy_dataset_refuses_existing_destination_with_filters(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Filtered copies should not silently reuse an existing destination."""
    monkeypatch.setattr(copying, "EVAL_DATASETS_BASE_PATH", str(tmp_path))
    dest = tmp_path / "existing_dataset"
    (dest / "windows").mkdir(parents=True)

    with pytest.raises(ValueError, match="Refusing to reuse an existing copy"):
        copying.copy_dataset(
            "/source",
            "existing_dataset",
            source_groups=["train"],
        )
