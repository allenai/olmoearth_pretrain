"""Tests for the OE_GRAD_DIAGNOSTICS_INTERVAL per-module gradient breakdown."""

from types import SimpleNamespace
from typing import Any, cast

import pytest
import torch
from torch import nn

from olmoearth_pretrain.train.train_module.train_module import (
    OlmoEarthTrainModule,
    _param_group_name,
)


def test_param_group_name_collapses_stacks_and_truncates() -> None:
    """Repeated blocks aggregate; paths truncate to the requested depth."""
    assert (
        _param_group_name("encoder.blocks.0.attn.q.weight", 5)
        == "encoder.blocks.*.attn.q"
    )
    assert (
        _param_group_name("encoder.blocks.11.attn.q.bias", 5)
        == "encoder.blocks.*.attn.q"
    )
    # Per-modality patch embeddings stay separate at the default depth, which is
    # what makes a modality-specific gradient problem visible.
    assert (
        _param_group_name(
            "encoder.patch_embeddings.per_modality_embeddings.landsat"
            ".landsat__0.pixel_proj.0.weight",
            5,
        )
        == "encoder.patch_embeddings.per_modality_embeddings.landsat.landsat__0"
    )
    assert _param_group_name("encoder.norm.weight", 5) == "encoder.norm.weight"


class _TinyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = nn.Module()
        self.encoder.blocks = nn.ModuleList([nn.Linear(4, 4) for _ in range(3)])
        self.head = nn.Linear(4, 2)


def _stub(model: nn.Module, recorded: dict[str, float]) -> OlmoEarthTrainModule:
    """Stand-in exposing only the attributes the diagnostics helpers read.

    Building a real train module needs a distributed context, so the helpers are
    exercised against a namespace and the result is cast to keep call sites typed.
    """

    def record_metric(
        name: str,
        value: Any,
        reduce_type: Any = None,
        namespace: str | None = None,
    ) -> None:
        if namespace is not None:
            name = f"{namespace}/{name}"
        recorded[name] = float(value)

    stub = SimpleNamespace(
        model=model,
        device=torch.device("cpu"),
        is_fsdp=False,
        dp_process_group=None,
        _grad_diagnostics_groups=None,
        _grad_diagnostics_depth=3,
        _grad_diagnostics_interval=1,
        trainer=SimpleNamespace(record_metric=record_metric),
    )
    stub._grad_diagnostic_groups = lambda: OlmoEarthTrainModule._grad_diagnostic_groups(
        cast(OlmoEarthTrainModule, stub)
    )
    return cast(OlmoEarthTrainModule, stub)


def test_grad_diagnostics_attribute_norm_to_the_dominant_module() -> None:
    """Shares sum to one and single out the module holding the gradient."""
    torch.manual_seed(0)
    model = _TinyModel()
    for param in model.parameters():
        param.grad = torch.full_like(param, 0.01)
    # Make one module dominate, as an over-driven patch embedding would.
    model.head.weight.grad = torch.full_like(model.head.weight, 10.0)

    recorded: dict[str, float] = {}
    stub = _stub(model, recorded)
    OlmoEarthTrainModule._log_grad_diagnostics(stub)

    shares = {
        k[len("grad share/") :]: v
        for k, v in recorded.items()
        if k.startswith("grad share/")
    }
    assert set(shares) == {"encoder.blocks.*", "head.weight", "head.bias"}
    assert sum(shares.values()) == pytest.approx(1.0)
    assert shares["head.weight"] > 0.99
    assert recorded["optim/max group grad share"] == pytest.approx(
        shares["head.weight"]
    )

    # Gradient norms are reported per group, alongside the weight norm of the
    # same parameters so the two can be compared.
    expected = torch.linalg.vector_norm(model.head.weight.grad).item()
    assert recorded["grad norm/head.weight"] == pytest.approx(expected, rel=1e-5)
    assert recorded["weight norm/head.weight"] == pytest.approx(
        torch.linalg.vector_norm(model.head.weight).item(), rel=1e-5
    )
    assert recorded["grad over weight/head.weight"] == pytest.approx(
        recorded["grad norm/head.weight"] / recorded["weight norm/head.weight"],
        rel=1e-5,
    )


def test_grad_diagnostics_tolerate_parameters_without_grads() -> None:
    """Params skipped by the backward pass contribute zero, not NaN."""
    model = _TinyModel()
    model.head.weight.grad = torch.full_like(model.head.weight, 2.0)

    recorded: dict[str, float] = {}
    OlmoEarthTrainModule._log_grad_diagnostics(_stub(model, recorded))

    assert recorded["grad share/head.weight"] == pytest.approx(1.0)
    assert recorded["grad norm/encoder.blocks.*"] == 0.0
    # Weight norms are still reported for those groups.
    assert recorded["weight norm/encoder.blocks.*"] > 0.0
    assert recorded["grad over weight/encoder.blocks.*"] == 0.0
