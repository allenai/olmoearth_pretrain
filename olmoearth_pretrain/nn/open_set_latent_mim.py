"""Latent-MIM model with a supervised open-set probe head.

``OpenSetLatentMIM`` is a thin extension of :class:`LatentMIM` that additionally
owns an :class:`OpenSetProbe`. The probe lives *inside the model* on purpose: the
DDP data-parallel path in the train module broadcasts parameters and all-reduces
gradients by iterating ``self.model.parameters()``, and the optimizer is likewise
built from ``self.model``. A probe attached to the train module (rather than the
model) would therefore never be synced or optimized.

The probe is not part of the self-supervised ``forward``; the train module calls
``model.open_set_probe(latent, batch)`` after the usual latent-MIM forward.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.distributed import DeviceMesh
from torch.distributed.fsdp import fully_shard, register_fsdp_forward_method

from olmoearth_pretrain.nn.latent_mim import LatentMIM, LatentMIMConfig
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
        projection_only_target: bool = False,
    ):
        """Initialize the model and attach the probe as a submodule."""
        super().__init__(
            encoder=encoder,
            decoder=decoder,
            reconstructor=reconstructor,
            projection_only_target=projection_only_target,
        )
        self.open_set_probe = open_set_probe

    def apply_fsdp(
        self,
        dp_mesh: DeviceMesh | None = None,
        param_dtype: torch.dtype | None = None,
        reduce_dtype: torch.dtype = torch.float32,
        prefetch_factor: int = 0,
    ) -> None:
        """Apply FSDP, sharding the probe as its own unit before the outer shard."""
        from torch.distributed.fsdp import MixedPrecisionPolicy

        mp_policy = MixedPrecisionPolicy(
            param_dtype=param_dtype, reduce_dtype=reduce_dtype
        )
        # Shard the probe as its own unit and register its forward so FSDP unshards
        # it when the train module calls ``model.open_set_probe(...)`` directly.
        fully_shard(self.open_set_probe, mesh=dp_mesh, mp_policy=mp_policy)
        register_fsdp_forward_method(self.open_set_probe, "forward")
        super().apply_fsdp(
            dp_mesh=dp_mesh,
            param_dtype=param_dtype,
            reduce_dtype=reduce_dtype,
            prefetch_factor=prefetch_factor,
        )

    def apply_compile(self) -> None:
        """Apply torch.compile to the underlying latent-MIM modules (not the probe)."""
        super().apply_compile()


@dataclass
class OpenSetLatentMIMConfig(LatentMIMConfig):
    """Configuration for :class:`OpenSetLatentMIM`."""

    open_set_probe_config: OpenSetProbeConfig | None = None

    def validate(self) -> None:
        """Validate the configuration."""
        super().validate()
        if self.open_set_probe_config is None:
            raise ValueError("open_set_probe_config is required for OpenSetLatentMIM")

    def build(self) -> OpenSetLatentMIM:
        """Build the model, including the supervised probe head."""
        self.validate()
        assert self.open_set_probe_config is not None
        encoder = self.encoder_config.build()
        decoder = self.decoder_config.build()
        reconstructor = (
            self.reconstructor_config.build()
            if self.reconstructor_config is not None
            else None
        )
        embedding_size = (
            self.encoder_config.output_embedding_size
            or self.encoder_config.embedding_size
        )
        probe = self.open_set_probe_config.build(embedding_size=embedding_size)
        return OpenSetLatentMIM(
            encoder=encoder,
            decoder=decoder,
            open_set_probe=probe,
            reconstructor=reconstructor,
            projection_only_target=self.projection_only_target,
        )
