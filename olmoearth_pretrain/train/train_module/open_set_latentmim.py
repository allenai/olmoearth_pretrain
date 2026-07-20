"""Contrastive latent-MIM train module with a supervised open-set probe.

Extends :class:`ContrastiveLatentMIMTrainModule` by adding a supervised
segmentation + regression loss (see
:class:`olmoearth_pretrain.train.open_set_probe.OpenSetProbe`) on top of the
self-supervised objective. The probe reads the *online* encoder output, so the
supervised gradient flows back into the encoder.

The probe itself lives inside the model
(:class:`olmoearth_pretrain.nn.open_set_latent_mim.OpenSetLatentMIM`) so that the
DDP gradient all-reduce and the optimizer cover its parameters.
"""

from dataclasses import dataclass
from logging import getLogger
from typing import Any

import torch
from olmo_core.train.common import ReduceType

from olmoearth_pretrain.datatypes import MaskedOlmoEarthSample, TokensAndMasks
from olmoearth_pretrain.train.train_module.contrastive_latentmim import (
    ContrastiveLatentMIMTrainModule,
    ContrastiveLatentMIMTrainModuleConfig,
)

logger = getLogger(__name__)


class OpenSetLatentMIMTrainModule(ContrastiveLatentMIMTrainModule):
    """Contrastive latent-MIM plus a supervised open-set probe loss."""

    def __init__(self, *args: Any, sup_loss_weight: float = 1.0, **kwargs: Any) -> None:
        """Initialize, extracting the supervised loss weight.

        Args:
            *args: Positional arguments forwarded to the base train module.
            sup_loss_weight: Scalar weight applied to the combined supervised
                (CE + MSE) loss when added to the self-supervised objective.
            **kwargs: Keyword arguments forwarded to the base train module.
        """
        super().__init__(*args, **kwargs)
        self.sup_loss_weight = sup_loss_weight

    def model_forward(
        self,
        batch: MaskedOlmoEarthSample,
        patch_size: int,
        token_exit_cfg: dict[str, int],
    ) -> tuple[
        torch.Tensor, TokensAndMasks, TokensAndMasks, TokensAndMasks, torch.Tensor
    ]:
        """Run the base forward, then add the supervised probe loss."""
        loss, latent, decoded, target_output, pooled = super().model_forward(
            batch, patch_size, token_exit_cfg
        )
        # The probe lives inside the model so DDP/optimizer cover its params. It always
        # returns a probe-connected loss (a zero-touch term when a rank has no labeled
        # patches) so every rank produces gradients for the probe params each step.
        # Re-enter the model forward context because the base method has already exited
        # it. Production DDP uses bf16 autocast, and the probe's fp32 parameters must be
        # autocast together with the encoder's bf16 latent representations.
        with self._model_forward_context():
            sup_loss, sup_metrics = self.model.open_set_probe(latent, batch)
        loss = loss + self.sup_loss_weight * sup_loss
        if sup_metrics:
            self.log_extra_metrics(
                {f"train/{key}": value for key, value in sup_metrics.items()},
                reduce_type=ReduceType.mean,
            )
        return loss, latent, decoded, target_output, pooled


@dataclass
class OpenSetLatentMIMTrainModuleConfig(ContrastiveLatentMIMTrainModuleConfig):
    """Configuration for :class:`OpenSetLatentMIMTrainModule`."""

    sup_loss_weight: float = 1.0

    def build(
        self,
        model: Any,
        device: torch.device | None = None,
    ) -> "OpenSetLatentMIMTrainModule":
        """Build the corresponding :class:`OpenSetLatentMIMTrainModule`."""
        kwargs = self.prepare_kwargs()
        return OpenSetLatentMIMTrainModule(
            model=model,
            device=device,
            **kwargs,
        )
