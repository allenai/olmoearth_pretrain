"""use_recompute_mlp_activation: same MLP outputs and gradients, activation recomputed."""

import pytest
import torch

from olmoearth_pretrain.nn import attention
from olmoearth_pretrain.nn.attention import Mlp, use_recompute_mlp_activation


def _run(
    recompute: bool, autocast: bool
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    torch.manual_seed(0)
    mlp = Mlp(16, hidden_features=64)
    x = torch.randn(3, 5, 16, requires_grad=True)
    use_recompute_mlp_activation(recompute)
    try:
        with torch.autocast("cpu", dtype=torch.bfloat16, enabled=autocast):
            out = mlp(x)
        out.float().pow(2).sum().backward()
    finally:
        use_recompute_mlp_activation(False)
    grads = {name: p.grad.clone() for name, p in mlp.named_parameters()}
    assert x.grad is not None
    grads["x"] = x.grad.clone()
    return out.detach().float(), grads


@pytest.mark.parametrize("autocast", [False, True])
def test_recompute_matches_the_standard_mlp(autocast: bool) -> None:
    """Outputs and every gradient match (exactly in fp32, closely under bf16)."""
    ref_out, ref_grads = _run(False, autocast)
    out, grads = _run(True, autocast)
    tol = {} if not autocast else {"atol": 2e-2, "rtol": 2e-2}
    torch.testing.assert_close(out, ref_out, **tol)
    for name, grad in ref_grads.items():
        torch.testing.assert_close(grads[name].float(), grad.float(), **tol, msg=name)
    assert not attention._RECOMPUTE_MLP_ACTIVATION
