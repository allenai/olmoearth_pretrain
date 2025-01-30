"""Trainer based on olmo_core."""

from logging import getLogger
from typing import Any

import torch
from olmo_core.train.trainer import Trainer

from helios.data.collator import PerModalityCollatedOutput

logger = getLogger(__name__)


def move_to_device_helios(
    batch: PerModalityCollatedOutput, device: torch.device, non_blocking: bool = True
) -> PerModalityCollatedOutput:
    """Move the batch to the device."""
    return PerModalityCollatedOutput(
        sentinel2=batch.sentinel2.to(device, non_blocking=non_blocking),
        naip=batch.naip.to(device, non_blocking=non_blocking),
        worldcover=batch.worldcover.to(device, non_blocking=non_blocking),
        sample_metadata=batch.sample_metadata,
    )


class HeliosTrainer(Trainer):
    """Trainer for Helios."""

    def model_forward(self, micro_batch: dict[str, Any]) -> torch.Tensor:
        """Run a forward pass on a micro-batch, returning the logits."""
        raise NotImplementedError("model forward helios")

    def get_losses(
        self,
    ) -> dict[str, Any]:
        """Compute the losses for a micro-batch and logits."""
        raise NotImplementedError("get losses helios")

    def eval_batch(self, batch: dict[str, Any]) -> dict[str, Any]:
        """Evaluate a batch."""
        raise NotImplementedError("eval batch helios")

    def _validate_batch(self, batch: dict[str, Any]) -> int:
        """Validate the data in a batch and return the global total number of tokens in the batch."""
        return self.data_loader.global_batch_size
