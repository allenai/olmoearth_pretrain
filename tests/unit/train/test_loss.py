"""Test losses."""

import torch

from helios.train.loss import PatchDiscriminationLoss
from helios.train.model import TokensAndMasks


def test_patch_disc_loss() -> None:
    """Just test that it runs as expected."""
    b, t, t_h, t_w, d = 3, 2, 4, 4, 2

    preds = TokensAndMasks(
        s2=torch.ones((b, t, t_h, t_w, d)),
        s2_mask=torch.ones((b, t, t_h, t_w)) * 2,
        latlon=torch.ones((b, 1, d)),
        latlon_mask=torch.ones((b, 1)) * 2,
    )
    targets = TokensAndMasks(
        s2=torch.ones((b, t, t_h, t_w, d)),
        s2_mask=torch.zeros((b, t, t_h, t_w)),
        latlon=torch.ones((b, 1, d)),
        latlon_mask=torch.zeros((b, 1)),
    )
    loss = PatchDiscriminationLoss()
    loss_value = loss.compute(preds, targets)
    # not very good! since they are all the same
    # predictions and values
    assert loss_value > 1
