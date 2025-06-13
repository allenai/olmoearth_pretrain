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
        batch_patch_size = 3
        b = 1
        supervisory_modalities = {
            # 95 will be assigned class 11
            "worldcover": repeat(
                torch.tensor([[95, 95], [95, 70]]),
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
            "mask": repeat(
                torch.tensor([[1, 0], [0, 1]]),
                "h w -> b (h p1) (w p2)",
                b=b,
                p1=batch_patch_size,
                p2=batch_patch_size,
            ),
            "worldcover_0": repeat(
                # times 100 since this is unnormalized from the perspective of the ce loss
                one_hot(torch.tensor([[4, 11], [11, 7]]), num_classes=12).float() * 100,
                "h w d -> b (h p1) (w p2) d",
                b=b,
                p1=max_patch_size,
                p2=max_patch_size,
            ),
        }
        org_loss = torch.tensor(0).float()
        loss = SupervisedLatentMIMTrainModule.supervisory_losses(
            supervisory_modalities, probe_outputs, org_loss
        )
        assert loss == 0
