"""Unit tests for TaskLoRALinear (new API with precomputed base_out)."""

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from helios.nn.lora import TaskLoRALinear


def test_matches_linear_when_no_task() -> None:
    """Returns base_out unchanged when task_emb=None."""
    torch.manual_seed(0)
    in_f, out_f = 16, 12
    x = torch.randn(3, 7, in_f)

    base = nn.Linear(in_f, out_f)
    base_out = base(x)

    tl = TaskLoRALinear(in_f, out_f, task_dim=8, rank=4, alpha=8.0, dropout=0.0)
    y_tl = tl(x, base_out, task_emb=None)

    assert torch.allclose(base_out, y_tl, atol=1e-6, rtol=1e-6)


def test_learns_nonzero_delta_after_optimization() -> None:
    """Training updates generator params so outputs change (ΔW learns)."""
    torch.manual_seed(2)
    in_f, out_f, task_d = 10, 7, 5
    B = 4
    x = torch.randn(B, 3, in_f)
    task = torch.randn(B, task_d)  # (batch, task_dim)
    target = torch.randn(B, 3, out_f)

    # Keep a fixed base_out; optimization only updates the LoRA generators.
    base = nn.Linear(in_f, out_f)
    with torch.no_grad():
        base_out = base(x)

    tl = TaskLoRALinear(in_f, out_f, task_dim=task_d, rank=3, alpha=6.0, dropout=0.0)

    with torch.no_grad():
        loss0 = F.mse_loss(tl(x, base_out, task_emb=task), target)

    opt = torch.optim.Adam(tl.parameters(), lr=1e-2)
    for _ in range(1):
        opt.zero_grad(set_to_none=True)
        y = tl(x, base_out, task_emb=task)
        loss = F.mse_loss(y, target)
        loss.backward()
        opt.step()

    loss1 = F.mse_loss(tl(x, base_out, task_emb=task), target)
    assert loss1 < loss0


def test_high_rank_input_5d() -> None:
    """Supports extra leading dims (5D)."""
    torch.manual_seed(4)
    B, T, H, W, in_f, out_f, task_d = 1, 2, 3, 4, 6, 9, 5
    x = torch.randn(B, T, H, W, in_f)
    task = torch.randn(B, task_d)  # (batch, task_dim) with batch=B

    # Provide any precomputed base_out with matching leading dims.
    base_out = torch.zeros(B, T, H, W, out_f)

    tl = TaskLoRALinear(in_f, out_f, task_dim=task_d, rank=2, alpha=2.0, dropout=0.0)
    y = tl(x, base_out, task_emb=task)

    assert y.shape == (B, T, H, W, out_f)


def test_invalid_last_dim_raises() -> None:
    """Raises if x's last dim != in_features."""
    torch.manual_seed(5)
    B, H, N, in_f, out_f, task_d = 2, 2, 4, 8, 8, 4
    x = torch.randn(B, H, N, in_f + 1)  # wrong last dim
    task = torch.randn(B, task_d)  # (batch, task_dim)

    # base_out with the same leading dims as x, last dim out_f
    base_out = torch.randn(B, H, N, out_f)

    tl = TaskLoRALinear(in_f, out_f, task_dim=task_d, rank=2)
    with pytest.raises(ValueError):
        _ = tl(x, base_out, task_emb=task)


def test_gradients_flow() -> None:
    """Sanity check that gradients reach generator params."""
    torch.manual_seed(7)
    in_f, out_f, task_d = 8, 8, 3
    B = 2
    x = torch.randn(B, 5, in_f)
    task = torch.randn(B, task_d)  # (batch, task_dim)
    target = torch.randn(B, 5, out_f)

    # Using zeros for base_out keeps the focus purely on the LoRA delta path.
    base_out = torch.zeros(B, 5, out_f, requires_grad=False)

    tl = TaskLoRALinear(in_f, out_f, task_dim=task_d, rank=2, alpha=2.0, dropout=0.0)
    y = tl(x, base_out, task_emb=task)
    loss = F.mse_loss(y, target)
    loss.backward()

    # Generators must receive gradients.
    assert all(p.grad is not None for p in tl.gen_a.parameters())
    assert all(p.grad is not None for p in tl.gen_b.parameters())
