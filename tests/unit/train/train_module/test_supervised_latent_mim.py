"""Unit tests for the supervised latent MIM Training Module."""

import torch
from einops import repeat
from torch.nn.functional import one_hot

from helios.data.constants import MISSING_VALUE
from helios.train.train_module.supervised_latent_mim import (
    SupervisedLatentMIMTrainModule,
)


class TestSupervisedLatentMIMUnit:
    """Unit tests for the supervised latent MIM Training Module."""

    def test_supervised_loss(self) -> None:
        """Test the supervised loss."""
        max_patch_size = 12
        batch_patch_size = 2
        b = 1
        supervisory_modalities = {
            "worldcover": repeat(
                torch.tensor([[1, 2], [3, MISSING_VALUE]]),
                "h w -> b (h p1) (w p2) t d",
                b=b,
                p1=batch_patch_size,
                p2=batch_patch_size,
                t=1,
                d=1,
            ),
            "gse": torch.randn((b, 2 * batch_patch_size, 2 * batch_patch_size, 1, 64)),
        }
        # 1s where the value is present
        probe_outputs = {
            "worldcover_0": repeat(
                # times 100 since this is unnormalized from the perspective of the ce loss
                one_hot(torch.tensor([[1, 2], [3, 4]]), num_classes=12).float() * 100,
                "h w d -> b (h p1) (w p2) d",
                b=b,
                p1=max_patch_size,
                p2=max_patch_size,
            ),
            "gse_0": torch.randn(b, 2 * max_patch_size, 2 * max_patch_size, 64),
        }
        org_sup_loss, org_sup_acc = SupervisedLatentMIMTrainModule.supervisory_losses(
            supervisory_modalities, probe_outputs, compute_accuracies=True
        )
        assert torch.allclose(org_sup_loss, torch.tensor(1.6), atol=0.1)
        assert org_sup_acc["worldcover_0"] == 1
        # all the contribution of the loss is from GSE
        assert torch.allclose(org_sup_acc["gse_0"], torch.tensor(1.6), atol=0.1)
