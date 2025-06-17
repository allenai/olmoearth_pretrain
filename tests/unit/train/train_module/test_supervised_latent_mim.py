"""Unit tests for the supervised latent MIM Training Module."""

import torch
from einops import repeat
from torch.nn.functional import one_hot

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
                torch.tensor([[1, 2], [3, 4]]),
                "h w -> b (h p1) (w p2) t d",
                b=b,
                p1=batch_patch_size,
                p2=batch_patch_size,
                t=1,
                d=1,
            )
        }
        # 1s where the value is present
        probe_outputs = {
            # the mask has the INPUT size, not the OUTPUT size
            # it is 1 where the tokens are *present*
            "mask": repeat(
                torch.tensor([[0, 1], [1, 0]], dtype=torch.bool),
                "h w -> b (h p1) (w p2)",
                b=b,
                p1=batch_patch_size,
                p2=batch_patch_size,
            ),
            "worldcover_0": repeat(
                # times 100 since this is unnormalized from the perspective of the ce loss
                one_hot(torch.tensor([[5, 2], [3, 6]]), num_classes=12).float() * 100,
                "h w d -> b (h p1) (w p2) d",
                b=b,
                p1=max_patch_size,
                p2=max_patch_size,
            ),
        }
        print(probe_outputs["mask"])
        org_loss = torch.tensor(0).float()
        org_sup_loss = torch.tensor(0).float()
        org_sup_acc = {"worldcover_0": torch.tensor(0).float()}
        loss, org_sup_loss, org_sup_acc = (
            SupervisedLatentMIMTrainModule.supervisory_losses(
                supervisory_modalities,
                probe_outputs,
                org_loss,
                org_sup_loss,
                org_sup_acc,
            )
        )
        assert loss == 0
        assert org_sup_loss == 0
        assert org_sup_acc["worldcover_0"] == 1
