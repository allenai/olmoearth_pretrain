"""Tests for olmoearth_pretrain.train.olmo_core_patches."""

from unittest.mock import MagicMock

from olmo_core.train.trainer import Trainer

from olmoearth_pretrain.train.olmo_core_patches import (
    ASYNC_CHECKPOINT_DURATION_METRIC,
    drop_async_checkpoint_duration_metric,
)


def test_drop_async_checkpoint_duration_metric() -> None:
    """The async-save duration metric is dropped; other metrics are still recorded."""
    original = Trainer.record_metric
    try:
        Trainer.record_metric = MagicMock()  # type: ignore[method-assign]
        inner = Trainer.record_metric
        drop_async_checkpoint_duration_metric()
        drop_async_checkpoint_duration_metric()  # idempotent: wraps only once
        trainer = object.__new__(Trainer)
        trainer.record_metric(ASYNC_CHECKPOINT_DURATION_METRIC, 1.0)
        inner.assert_not_called()
        trainer.record_metric("train/loss", 2.0, reduce_type=None)
        inner.assert_called_once_with(trainer, "train/loss", 2.0, reduce_type=None)
    finally:
        Trainer.record_metric = original  # type: ignore[method-assign]
