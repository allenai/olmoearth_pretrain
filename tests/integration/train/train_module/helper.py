"""Helper functions for the train module tests."""

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any
from unittest.mock import MagicMock, patch

import torch


class MockTrainer:
    """Minimal trainer surface used by train module integration tests."""

    def __init__(self) -> None:
        """Initialize the mock trainer."""
        self._metrics: dict[str, float] = {}
        self.global_step = 0
        self.max_steps = 100

    def record_metric(
        self,
        name: str,
        value: float,
        reduce_type: str,
        namespace: str | None = None,
    ) -> None:
        """Record a metric in the mock trainer."""
        self._metrics[name] = value


@contextmanager
def attached_train_module(train_module: Any) -> Iterator[MockTrainer]:
    """Attach a train module to a minimal mocked trainer."""
    with patch("olmoearth_pretrain.train.train_module.train_module.build_world_mesh"):
        mock_trainer = MockTrainer()
        setattr(train_module, "on_attach", MagicMock(return_value=None))
        train_module._attach_trainer(mock_trainer)
        yield mock_trainer


def check_loss_is_a_reasonable_value(loss: torch.Tensor) -> None:
    """Check a tensor doesn't contain NaN or Inf values, and is between 0 and 4."""
    assert not torch.isinf(loss).any()
    assert not torch.isnan(loss).any()
    assert loss < 4
    assert loss > 0
