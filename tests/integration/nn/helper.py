"""Helper assertions for NN integration tests."""

from torch import nn


def assert_has_parameter_grad(module: nn.Module) -> None:
    """Assert that a backward pass reached at least one trainable parameter."""
    assert any(
        param.grad is not None for param in module.parameters() if param.requires_grad
    )
