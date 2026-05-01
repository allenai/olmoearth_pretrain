"""Unit tests for the Hyperball optimizer."""

import math

import pytest
import torch
import torch.nn as nn

from olmoearth_pretrain.train.optim import Hyperball, HyperballConfig


def _toy_model(seed: int = 0) -> nn.Module:
    torch.manual_seed(seed)
    return nn.Sequential(
        nn.Linear(8, 16),
        nn.GELU(),
        nn.Linear(16, 4),
    )


def test_frobenius_norm_preserved_on_2d_params() -> None:
    """After many steps, ||W||_F should stay equal to its initial value for 2D matrices."""
    model = _toy_model()
    initial_norms = {
        n: p.detach().norm().item() for n, p in model.named_parameters() if p.dim() >= 2
    }
    opt = Hyperball(model.parameters(), lr=1e-2, apply_hyperball=True)

    for _ in range(20):
        x = torch.randn(32, 8)
        y = model(x).sum()
        opt.zero_grad()
        y.backward()
        opt.step()

    for name, p in model.named_parameters():
        if p.dim() >= 2:
            assert math.isclose(
                p.detach().norm().item(),
                initial_norms[name],
                rel_tol=1e-4,
                abs_tol=1e-5,
            ), (
                f"{name}: norm drifted from {initial_norms[name]} to {p.detach().norm().item()}"
            )


def test_one_d_params_fallback_to_adamw() -> None:
    """1D params (biases) should NOT have their norm constrained — they fall back to AdamW."""
    model = _toy_model()
    bias = next(p for n, p in model.named_parameters() if p.dim() == 1)
    initial_bias_norm = bias.detach().norm().item()

    opt = Hyperball(model.parameters(), lr=1e-2, apply_hyperball=True)
    for _ in range(20):
        x = torch.randn(32, 8)
        y = model(x).sum()
        opt.zero_grad()
        y.backward()
        opt.step()

    # The bias should have moved (training did something).
    assert not math.isclose(
        bias.detach().norm().item(), initial_bias_norm, rel_tol=1e-3
    )


def test_apply_hyperball_false_falls_back_to_adamw() -> None:
    """A param group with apply_hyperball=False should behave as AdamW."""
    torch.manual_seed(42)
    layer_hb = nn.Linear(8, 16)
    layer_no = nn.Linear(8, 16)
    # Same init.
    layer_no.load_state_dict(layer_hb.state_dict())

    opt_hb = Hyperball(
        [{"params": layer_hb.parameters(), "apply_hyperball": True}], lr=1e-2
    )
    opt_no = Hyperball(
        [
            {
                "params": layer_no.parameters(),
                "apply_hyperball": False,
                "weight_decay": 0.0,
            }
        ],
        lr=1e-2,
    )

    initial_norm = layer_hb.weight.detach().norm().item()

    for _ in range(10):
        x = torch.randn(32, 8)
        y_hb = layer_hb(x).sum()
        y_no = layer_no(x).sum()
        opt_hb.zero_grad()
        y_hb.backward()
        opt_hb.step()
        opt_no.zero_grad()
        y_no.backward()
        opt_no.step()

    hb_norm = layer_hb.weight.detach().norm().item()
    no_norm = layer_no.weight.detach().norm().item()

    assert math.isclose(hb_norm, initial_norm, rel_tol=1e-4)
    assert not math.isclose(no_norm, initial_norm, rel_tol=1e-3)


def test_step_size_is_lr_times_R() -> None:
    """Per the paper, the per-step displacement before projection is lr*R in -Normalize(u_t)."""
    torch.manual_seed(0)
    layer = nn.Linear(8, 16, bias=False)
    R0 = layer.weight.detach().norm().item()
    lr = 0.1
    opt = Hyperball(layer.parameters(), lr=lr, apply_hyperball=True)

    # One step
    x = torch.randn(32, 8)
    y = layer(x).sum()
    opt.zero_grad()
    y.backward()
    W_before = layer.weight.detach().clone()
    opt.step()
    W_after = layer.weight.detach().clone()

    # ||W_after - W_before|| should be <= lr*R (the chord of the sphere step)
    # and W_after should be on the sphere of radius R0.
    chord = (W_after - W_before).norm().item()
    assert chord <= lr * R0 + 1e-4
    assert math.isclose(W_after.norm().item(), R0, rel_tol=1e-5)


def test_no_nan_on_zero_gradient() -> None:
    """If the gradient is zero, the optimizer must not produce NaN or move the params."""
    layer = nn.Linear(8, 16)
    opt = Hyperball(layer.parameters(), lr=1e-2, apply_hyperball=True)

    W_before = layer.weight.detach().clone()
    # No backward -> grad is None initially.
    layer.weight.grad = torch.zeros_like(layer.weight)
    layer.bias.grad = torch.zeros_like(layer.bias)
    opt.step()

    assert not torch.isnan(layer.weight).any()
    assert torch.allclose(layer.weight.detach(), W_before, atol=1e-6)


def test_config_builds_optimizer() -> None:
    """HyperballConfig.optimizer() should return the Hyperball class for olmo-core."""
    cfg = HyperballConfig(lr=2.5e-3)
    assert cfg.optimizer() is Hyperball


def test_loss_decreases_on_toy_problem() -> None:
    """End-to-end sanity: on a tiny regression task, loss should decrease."""
    torch.manual_seed(123)
    model = nn.Sequential(nn.Linear(4, 8), nn.GELU(), nn.Linear(8, 1))
    opt = Hyperball(model.parameters(), lr=5e-3, apply_hyperball=True)

    X = torch.randn(64, 4)
    target = (X.sum(dim=-1, keepdim=True) > 0).float()

    losses = []
    for _ in range(200):
        pred = model(X)
        loss = (pred - target).pow(2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())

    assert losses[-1] < losses[0], (
        f"Loss did not decrease (start={losses[0]:.4f}, end={losses[-1]:.4f})"
    )


@pytest.mark.parametrize("apply_hyperball", [True, False])
def test_resume_from_state_dict(apply_hyperball: bool) -> None:
    """state_dict / load_state_dict round-trip should preserve optimizer state."""
    torch.manual_seed(0)
    layer1 = nn.Linear(8, 16)
    opt1 = Hyperball(layer1.parameters(), lr=1e-2, apply_hyperball=apply_hyperball)
    # Capture all batches up front so RNG drift can't contaminate the comparison.
    xs = [torch.randn(32, 8) for _ in range(6)]

    for x in xs[:5]:
        y = layer1(x).sum()
        opt1.zero_grad()
        y.backward()
        opt1.step()

    state = opt1.state_dict()

    layer2 = nn.Linear(8, 16)
    layer2.load_state_dict(layer1.state_dict())
    opt2 = Hyperball(layer2.parameters(), lr=1e-2, apply_hyperball=apply_hyperball)
    opt2.load_state_dict(state)

    # 6th step on identical x for both.
    y1 = layer1(xs[5]).sum()
    opt1.zero_grad()
    y1.backward()
    opt1.step()
    y2 = layer2(xs[5]).sum()
    opt2.zero_grad()
    y2.backward()
    opt2.step()

    assert torch.allclose(layer1.weight, layer2.weight, atol=1e-6), (
        f"max diff: {(layer1.weight - layer2.weight).abs().max().item()}"
    )
