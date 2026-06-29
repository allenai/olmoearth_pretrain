"""Lightweight integration tests for finetune checkpoint resume functionality."""

import os
import tempfile

import torch
import torch.nn as nn

from olmoearth_pretrain.evals.finetune.checkpoint import (
    load_training_checkpoint,
    save_training_checkpoint,
)
from olmoearth_pretrain.evals.finetune.model import snapshot_state_dict


class DummyModel(nn.Module):
    """Minimal model for testing checkpoint functionality."""

    def __init__(self) -> None:
        """Initialize with a simple linear layer."""
        super().__init__()
        self.fc = nn.Linear(32, 32)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.fc(x)


def test_resume_from_checkpoint() -> None:
    """Test that resume from checkpoint works correctly."""
    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_path = os.path.join(tmpdir, "last.pt")
        device = torch.device("cpu")

        model = DummyModel()
        opt = torch.optim.AdamW(model.parameters(), lr=0.001)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max")

        # Save checkpoint at epoch 2
        save_training_checkpoint(
            path=ckpt_path,
            epoch=2,
            model_state=snapshot_state_dict(model),
            optimizer_state=opt.state_dict(),
            scheduler_state=scheduler.state_dict(),
            best_state=snapshot_state_dict(model),
            best_val_metric=0.5,
            backbone_unfrozen=False,
        )

        # Load and verify start_epoch would be 3
        ckpt = load_training_checkpoint(ckpt_path, device)
        assert ckpt is not None
        assert ckpt["epoch"] + 1 == 3
        assert ckpt["backbone_unfrozen"] is False
        assert ckpt["best_val_metric"] == 0.5
