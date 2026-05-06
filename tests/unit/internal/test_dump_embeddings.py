"""Unit tests for dump_embeddings partition + override helpers."""

import json
from pathlib import Path

import pytest

from olmoearth_pretrain.internal.dump_embeddings import (
    _load_settings_group,
    _partition_by_norm_mode,
    _per_task_overrides,
)


def _settings(pooling: str, lr: float | None, pretrained: bool) -> dict:
    return {
        "settings": {
            "pooling_type": pooling,
            "probe_lr": lr,
            "norm_stats_from_pretrained": pretrained,
        }
    }


def test_partition_splits_by_norm_mode() -> None:
    """Mixed-mode group is split into per-mode buckets."""
    g = {
        "task_a": _settings("mean", 0.1, True),
        "task_b": _settings("max", 0.01, False),
        "task_c": _settings("mean", None, False),
    }
    parts = _partition_by_norm_mode(g)
    assert parts == {True: ["task_a"], False: ["task_b", "task_c"]}


def test_partition_drops_empty_partitions() -> None:
    """Single-mode group emits only that mode's bucket."""
    g = {"task_a": _settings("mean", None, False)}
    parts = _partition_by_norm_mode(g)
    assert True not in parts
    assert parts[False] == ["task_a"]


def test_per_task_overrides_emits_expected_keys() -> None:
    """Per-task dotlist overrides set the dump fields and skip probe_lr."""
    args = _per_task_overrides(
        task_name="m_eurosat",
        settings={
            "pooling_type": "max",
            "probe_lr": None,
            "norm_stats_from_pretrained": True,
        },
        save_embeddings_dir="/weka/x",
        embedding_dump_dtype="bfloat16",
    )
    base = "--trainer.callbacks.downstream_evaluator.tasks.m_eurosat"
    assert f"{base}.eval_mode=EMBEDDING_DUMP" in args
    assert f"{base}.save_embeddings_dir=/weka/x" in args
    assert f"{base}.embedding_dump_dtype=bfloat16" in args
    assert f"{base}.pooling_type=max" in args
    assert f"{base}.norm_stats_from_pretrained=True" in args
    # probe_lr should NOT be set in dump mode (no probe runs).
    assert all("probe_lr" not in s for s in args)


def test_load_settings_group_raises_for_missing_group(tmp_path: Path) -> None:
    """Unknown group name raises KeyError with the available groups listed."""
    p = tmp_path / "settings.json"
    p.write_text(
        json.dumps({"galileo_base": {"task_a": _settings("mean", 0.1, False)}})
    )
    with pytest.raises(KeyError):
        _load_settings_group(str(p), "does_not_exist")
