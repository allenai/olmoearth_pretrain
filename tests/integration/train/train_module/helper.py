"""Helper functions for the train module tests."""

import torch


def check_loss_is_a_reasonable_value(loss: torch.Tensor) -> None:
    """Check a tensor doesn't contain NaN or Inf values, and is between 0 and 4."""
    assert not torch.isinf(loss).any()
    assert not torch.isnan(loss).any()
    assert loss < 4
    assert loss > 0
