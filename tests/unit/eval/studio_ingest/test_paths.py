"""Tests for Studio ingest path helpers."""

import pytest

from olmoearth_pretrain.evals.studio_ingest import ingest, paths


def test_get_eval_datasets_base_path_uses_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Eval dataset base path should default to the Weka URI."""
    monkeypatch.delenv(paths.EVAL_DATASETS_ENV_VAR, raising=False)

    assert paths.get_eval_datasets_base_path() == paths.DEFAULT_WEKA_BASE_PATH
    assert ingest.get_eval_datasets_base_path() == paths.DEFAULT_WEKA_BASE_PATH


def test_get_eval_datasets_base_path_uses_env_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Eval dataset base path should respect the external override env var."""
    monkeypatch.setenv(paths.EVAL_DATASETS_ENV_VAR, "/tmp/eval_datasets")

    assert paths.get_eval_datasets_base_path() == "/tmp/eval_datasets"
    assert ingest.get_eval_datasets_base_path() == "/tmp/eval_datasets"
