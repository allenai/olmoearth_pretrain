"""Hyperball optimizer (Wen et al., 2025).

Replaces weight decay with an explicit Frobenius-norm projection on weight matrices.
Per the paper:

  W_{t+1} = R * Normalize( W_t - eta * R * Normalize(u_t) )

where:
  - u_t is the base-optimizer update direction (Adam moment ratio here)
  - R = ||W_0||_F captured at parameter initialization
  - Normalize(x) = x / ||x||_F

Applied to non-embedding 2D+ weight matrices. Falls back to AdamW for 1D
parameters (biases, RMSNorm gains) and for any param group with
`apply_hyperball=False` (typically embeddings + LM head).

Reference: "Fantastic Pretraining Optimizers and Where to Find Them 2.1:
Hyperball Optimization" — https://tinyurl.com/muonh
"""

from collections.abc import Callable, Iterable
from dataclasses import dataclass

import torch
import torch.distributed as dist
from olmo_core.optim.config import OptimConfig


def _global_frobenius_norm(t: torch.Tensor) -> torch.Tensor:
    """Frobenius norm computed over the global tensor.

    For DTensor-sharded params (FSDP / TP), all-reduces the sum of squares
    of local shards. For ordinary tensors (DDP / single GPU), this is just
    the local Frobenius norm.
    """
    local = t
    if hasattr(t, "to_local"):  # DTensor
        local = t.to_local()
    sq_sum = local.float().pow(2).sum()
    if dist.is_available() and dist.is_initialized() and hasattr(t, "to_local"):
        # Sum across the device mesh used by the DTensor.
        mesh = getattr(t, "device_mesh", None)
        if mesh is not None:
            for i in range(mesh.ndim):
                dist.all_reduce(sq_sum, op=dist.ReduceOp.SUM, group=mesh.get_group(i))
    return sq_sum.sqrt()


class Hyperball(torch.optim.Optimizer):
    """Hyperball optimizer with Adam as the base update direction.

    Per param group, set ``apply_hyperball=True`` (default) to enforce the
    Frobenius-norm constraint, or ``apply_hyperball=False`` to fall back to
    plain AdamW (with ``weight_decay``). Embeddings and LM head should use
    ``apply_hyperball=False`` per the paper's empirical tips.

    Note: 1D parameters (biases, RMSNorm gains) automatically fall back to
    AdamW regardless of group setting, since Frobenius normalization on a
    1D vector reduces to a sign+scale hack that has no theoretical basis
    in the paper.
    """

    def __init__(
        self,
        params: Iterable,
        lr: float = 5e-3,
        betas: tuple[float, float] = (0.9, 0.95),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        apply_hyperball: bool = True,
    ) -> None:
        """Initialize the Hyperball optimizer."""
        if lr <= 0.0:
            raise ValueError(f"lr must be > 0, got {lr}")
        if not all(0.0 <= b <= 1.0 for b in betas):
            raise ValueError(f"betas must be in [0,1], got {betas}")
        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            apply_hyperball=apply_hyperball,
        )
        super().__init__(params, defaults)

    def load_state_dict(self, state_dict: dict) -> None:
        """Load state, cloning tensors so optimizer instances don't alias each other.

        torch.optim's default load_state_dict re-uses tensor refs from the saved
        dict when dtype/device match, which means two optimizers loaded from the
        same state_dict will mutate the *same* exp_avg tensors and stomp on each
        other.  Clone every tensor to break aliasing.
        """
        import copy

        cloned = copy.deepcopy(state_dict)
        for ps in cloned.get("state", {}).values():
            for k, v in list(ps.items()):
                if torch.is_tensor(v):
                    ps[k] = v.clone()
        super().load_state_dict(cloned)

    @torch.no_grad()
    def step(self, closure: Callable | None = None) -> torch.Tensor | None:
        """Perform a single optimization step."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr: float = group["lr"]
            beta1, beta2 = group["betas"]
            eps: float = group["eps"]
            weight_decay: float = group["weight_decay"]
            group_apply_hb: bool = group.get("apply_hyperball", True)

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                # Per-param decision: hyperball only on 2D+ matrices.
                apply_hb = group_apply_hb and p.dim() >= 2

                state = self.state[p]
                if len(state) == 0:
                    # `step` and `R` are stored as Python scalars so they are
                    # deep-copied through state_dict / load_state_dict (torch.optim
                    # *re-uses* tensor refs across optimizers on load, which would
                    # cause aliasing bugs for these scalars).
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p)
                    state["exp_avg_sq"] = torch.zeros_like(p)
                    if apply_hb:
                        state["R"] = float(_global_frobenius_norm(p).item())

                state["step"] += 1
                step = state["step"]
                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]

                # Adam moments
                exp_avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)

                bias_correction1 = 1.0 - beta1**step
                bias_correction2 = 1.0 - beta2**step

                denom = (exp_avg_sq.sqrt() / (bias_correction2**0.5)).add_(eps)
                # u_t: Adam update direction (no lr, no sign flip).
                # Standard Adam applies -lr * u_t.
                u = (exp_avg / bias_correction1) / denom

                if apply_hb:
                    R: float = float(state["R"])
                    # Normalize u_t to the unit sphere.
                    u_norm = _global_frobenius_norm(u)
                    if float(u_norm) > 0.0:
                        u_unit = u / u_norm
                    else:
                        u_unit = u  # zero update; nothing to normalize
                    # W_t - eta * R * Normalize(u_t)
                    w_new = p.data - lr * R * u_unit
                    # Project back onto sphere of radius R.
                    w_norm = _global_frobenius_norm(w_new)
                    if float(w_norm) > 0.0:
                        p.data.copy_(R * w_new / w_norm)
                    else:
                        p.data.copy_(w_new)
                else:
                    # AdamW fallback (used for embeddings / LM head / 1D params).
                    if weight_decay != 0.0:
                        p.data.mul_(1.0 - lr * weight_decay)
                    p.data.add_(u, alpha=-lr)

        return loss


@dataclass
class HyperballConfig(OptimConfig):
    """Config for the Hyperball optimizer.

    Set ``apply_hyperball=True`` (default) for the main param group. Use
    ``group_overrides`` to carve out embeddings + LM head with
    ``apply_hyperball=False`` (and a non-zero ``weight_decay`` for the
    AdamW fallback).
    """

    lr: float = 5e-3
    betas: tuple[float, float] = (0.9, 0.95)
    eps: float = 1e-8
    weight_decay: float = 0.0
    apply_hyperball: bool = True

    @classmethod
    def optimizer(cls) -> type[Hyperball]:
        """Return the Hyperball optimizer class for olmo-core to instantiate."""
        return Hyperball
