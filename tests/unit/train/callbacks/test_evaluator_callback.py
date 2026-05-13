"""Tests for downstream evaluator callback edge cases."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from olmoearth_pretrain.evals.datasets.configs import TaskType
from olmoearth_pretrain.train.callbacks.evaluator_callback import (
    DownstreamEvaluator,
    DownstreamTaskConfig,
    EvalMode,
)


def test_downstream_evaluator_rejects_feature_exit_depth_for_finetune(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Finetune evals should fail fast when feature exit depth is requested."""
    monkeypatch.setattr(
        "olmoearth_pretrain.train.callbacks.evaluator_callback.dataset_to_config",
        lambda _dataset: SimpleNamespace(
            task_type=TaskType.CLASSIFICATION,
            height_width=None,
        ),
    )
    task = DownstreamTaskConfig(
        dataset="dummy",
        eval_mode=EvalMode.FINETUNE,
        ft_lr=1e-3,
        feature_exit_depth=12,
    )

    with pytest.raises(
        ValueError, match="feature_exit_depth is not supported for finetune evals"
    ):
        DownstreamEvaluator(
            evaluation_name="dummy_eval",
            task=task,
            trainer=Mock(),
        )
