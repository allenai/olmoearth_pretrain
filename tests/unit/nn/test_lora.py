"""Unit tests for TaskLoRALinear."""

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from helios.nn.lora import TaskLoRALinear


def _copy_linear_params(dst: TaskLoRALinear, src: nn.Linear) -> None:
    """Copy base Linear params into TaskLoRALinear.

    Args:
        dst: TaskLoRALinear module to copy parameters to.
        src: nn.Linear module to copy parameters from.
    """
    with torch.no_grad():
        dst.weight.copy_(src.weight)
        if dst.bias is not None and src.bias is not None:
            dst.bias.copy_(src.bias)


def test_matches_linear_when_no_task() -> None:
    """Behaves like nn.Linear when task_emb=None."""
    torch.manual_seed(0)
    in_f, out_f = 16, 12
    x = torch.randn(3, 7, in_f)

    base = nn.Linear(in_f, out_f)
    tl = TaskLoRALinear(
        in_f, out_f, task_dim=8, lora_rank=4, lora_alpha=8.0, lora_dropout=0.0
    )

    _copy_linear_params(tl, base)

    y_base = base(x)
    y_tl = tl(x, task_emb=None)
    assert torch.allclose(y_base, y_tl, atol=1e-6, rtol=1e-6)


def test_zero_init_delta() -> None:
    """Zero-init generators => ΔW=0 so outputs match base at start."""
    torch.manual_seed(1)
    in_f, out_f, task_d = 8, 8, 6
    B = 2
    x = torch.randn(B, 5, in_f)
    task = torch.randn(B, task_d)  # (batch, task_dim)

    base = nn.Linear(in_f, out_f)
    tl = TaskLoRALinear(
        in_f, out_f, task_dim=task_d, lora_rank=2, lora_alpha=2.0, lora_dropout=0.0
    )

    _copy_linear_params(tl, base)

    y_base = base(x)
    y_tl = tl(x, task_emb=task)
    assert torch.allclose(y_base, y_tl, atol=1e-6, rtol=1e-6)


def test_learns_nonzero_delta_after_optimization() -> None:
    """Training changes outputs (ΔW learns)."""
    torch.manual_seed(2)
    in_f, out_f, task_d = 10, 7, 5
    B = 4
    x = torch.randn(B, 3, in_f)
    task = torch.randn(B, task_d)  # (batch, task_dim)
    target = torch.randn(B, 3, out_f)

    tl = TaskLoRALinear(
        in_f, out_f, task_dim=task_d, lora_rank=3, lora_alpha=6.0, lora_dropout=0.0
    )

    y0 = tl(x, task_emb=task).detach()
    opt = torch.optim.Adam(tl.parameters(), lr=1e-2)
    for _ in range(8):
        opt.zero_grad(set_to_none=True)
        loss = F.mse_loss(tl(x, task_emb=task), target)
        loss.backward()
        opt.step()
    y1 = tl(x, task_emb=task).detach()
    assert not torch.allclose(y0, y1)


def test_high_rank_input_5d() -> None:
    """Supports extra leading dims (5D)."""
    torch.manual_seed(4)
    B, T, H, W, in_f, out_f, task_d = 1, 2, 3, 4, 6, 9, 5
    x = torch.randn(B, T, H, W, in_f)
    task = torch.randn(B, task_d)  # (batch, task_dim) with batch=B

    tl = TaskLoRALinear(
        in_f, out_f, task_dim=task_d, lora_rank=2, lora_alpha=2.0, lora_dropout=0.0
    )
    y = tl(x, task_emb=task)
    assert y.shape == (B, T, H, W, out_f)


def test_invalid_last_dim_raises() -> None:
    """Raises if last dim != in_features."""
    torch.manual_seed(5)
    B, H, N, in_f, out_f, task_d = 2, 2, 4, 8, 8, 4
    x = torch.randn(B, H, N, in_f + 1)  # wrong last dim
    task = torch.randn(B, task_d)  # (batch, task_dim)

    tl = TaskLoRALinear(in_f, out_f, task_dim=task_d, lora_rank=2)
    with pytest.raises(ValueError):
        _ = tl(x, task_emb=task)


def test_alpha_scaling_effect() -> None:
    """Delta magnitude grows with alpha (roughly)."""
    # From the lora paper, delta should somewhat scale the learning rate
    torch.manual_seed(6)
    in_f, out_f, task_d, r = 8, 8, 4, 2
    B = 3
    x = torch.randn(B, 7, in_f)
    task = torch.randn(B, task_d)  # (batch, task_dim)

    def make(alpha: float) -> TaskLoRALinear:
        tl = TaskLoRALinear(
            in_f,
            out_f,
            task_dim=task_d,
            lora_rank=r,
            lora_alpha=alpha,
            lora_dropout=0.0,
        )
        # Force nonzero generators for test (override zero-init)
        with torch.no_grad():
            la = tl.lora_gen_a[-1]
            lb = tl.lora_gen_b[-1]
            nn.init.uniform_(la.weight, -0.1, 0.1)
            nn.init.uniform_(la.bias, -0.1, 0.1)
            nn.init.uniform_(lb.weight, -0.1, 0.1)
            nn.init.uniform_(lb.bias, -0.1, 0.1)
        return tl

    # Same base weights for fair comparison
    base = nn.Linear(in_f, out_f)
    tl1, tl2 = make(2.0), make(6.0)
    with torch.no_grad():
        tl1.weight.copy_(base.weight)
        tl1.bias.copy_(base.bias)
        tl2.weight.copy_(base.weight)
        tl2.bias.copy_(base.bias)

    yb = base(x)
    d1 = (tl1(x, task_emb=task) - yb).norm()
    d2 = (tl2(x, task_emb=task) - yb).norm()
    ratio = (d2 / (d1 + 1e-8)).item()
    assert 2.0 <= ratio <= 4.0  # allow slack


def test_gradients_flow() -> None:
    """Sanity check that gradients reach base and generator params."""
    torch.manual_seed(7)
    in_f, out_f, task_d = 8, 8, 3
    B = 2
    x = torch.randn(B, 5, in_f)
    task = torch.randn(B, task_d)  # (batch, task_dim)
    target = torch.randn(B, 5, out_f)

    tl = TaskLoRALinear(
        in_f, out_f, task_dim=task_d, lora_rank=2, lora_alpha=2.0, lora_dropout=0.0
    )
    loss = F.mse_loss(tl(x, task_emb=task), target)
    loss.backward()

    assert tl.weight.grad is not None
    if tl.bias is not None:
        assert tl.bias.grad is not None
    assert all(p.grad is not None for p in tl.lora_gen_a.parameters())
    assert all(p.grad is not None for p in tl.lora_gen_b.parameters())
