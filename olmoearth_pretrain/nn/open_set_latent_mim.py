"""Latent-MIM model with a supervised open-set probe head on the register grid.

``OpenSetLatentMIM`` is a thin extension of :class:`LatentMIM` that additionally
owns an :class:`OpenSetProbe`. The probe lives *inside the model* on purpose: the
replicated-DDP path in the train module broadcasts parameters and all-reduces
gradients by iterating ``self.model.parameters()``, and the optimizer is likewise
built from ``self.model``. A probe attached to the train module (rather than the
model) would therefore never be synced or optimized. (Only the replicated DDP path
is supported: under FSDP the probe would need its own ``fully_shard`` unit.)

The probe reads the encoder's Perceiver register grid (``[B, n_h, n_w, D]``, the
same grid the v1.3 map supervision heads and the distilled student read), so the
model REQUIRES ``encoder_config.perceiver_config``. The probe is not part of the
self-supervised ``forward``; the train module calls
``model.open_set_probe(model.last_register_grid, batch)`` after the usual
latent-MIM forward.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import torch
import torch.nn as nn

from olmoearth_pretrain.nn.latent_mim import LatentMIM, LatentMIMConfig
from olmoearth_pretrain.nn.register_distillation_head import RegisterDistillationHead
from olmoearth_pretrain.nn.supervision_head import SupervisionHead
from olmoearth_pretrain.train.open_set_probe import OpenSetProbe, OpenSetProbeConfig

logger = logging.getLogger(__name__)


class OpenSetLatentMIM(LatentMIM):
    """A :class:`LatentMIM` that also owns a supervised :class:`OpenSetProbe`."""

    def __init__(
        self,
        encoder: nn.Module,
        decoder: nn.Module,
        open_set_probe: OpenSetProbe,
        reconstructor: torch.nn.Module | None = None,
        supervision_head: SupervisionHead | None = None,
        register_distillation_head: RegisterDistillationHead | None = None,
        projection_only_target: bool = False,
    ):
        """Initialize the model and attach the probe as a submodule."""
        super().__init__(
            encoder=encoder,
            decoder=decoder,
            reconstructor=reconstructor,
            supervision_head=supervision_head,
            register_distillation_head=register_distillation_head,
            projection_only_target=projection_only_target,
        )
        self.open_set_probe = open_set_probe


@dataclass
class OpenSetLatentMIMConfig(LatentMIMConfig):
    """Configuration for :class:`OpenSetLatentMIM`."""

    open_set_probe_config: OpenSetProbeConfig | None = None

    def validate(self) -> None:
        """Validate the configuration."""
        super().validate()
        if self.open_set_probe_config is None:
            raise ValueError("open_set_probe_config is required for OpenSetLatentMIM")
        if self.encoder_config.perceiver_config is None:
            raise ValueError(
                "OpenSetLatentMIM requires the encoder Perceiver: the open-set probe "
                "reads the register grid it produces (set "
                "encoder_config.perceiver_config)"
            )

    def build(self) -> OpenSetLatentMIM:
        """Build the model, including the supervised probe head."""
        self.validate()
        assert self.open_set_probe_config is not None
        base = super().build()
        # The probe reads the register grid, so its input dim is the register
        # (bottleneck) width, not the encoder token dim.
        perceiver_config = self.encoder_config.perceiver_config
        assert perceiver_config is not None
        probe = self.open_set_probe_config.build(
            embedding_size=perceiver_config.register_dim
        )
        return OpenSetLatentMIM(
            encoder=base.encoder,
            decoder=base.decoder,
            open_set_probe=probe,
            reconstructor=base.reconstructor,
            supervision_head=base.supervision_head,
            register_distillation_head=base.register_distillation_head,
            projection_only_target=self.projection_only_target,
        )
