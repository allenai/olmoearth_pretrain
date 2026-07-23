"""Latent-MIM model with a supervised open-set probe head.

``OpenSetLatentMIM`` is a thin extension of :class:`LatentMIM` that additionally
owns an :class:`OpenSetProbe`. The probe lives *inside the model* on purpose: the
DDP data-parallel path in the train module broadcasts parameters and all-reduces
gradients by iterating ``self.model.parameters()``, and the optimizer is likewise
built from ``self.model``. A probe attached to the train module (rather than the
model) would therefore never be synced or optimized.

The probe reads the encoder's *spatial latent grid* (the Perceiver/register
bottleneck output exposed by ``LatentMIM.forward`` as ``last_register_grid``),
so the encoder must be configured with ``use_register_bottleneck=True``. The
probe is not part of the self-supervised ``forward``; the train module calls
``model.open_set_probe(register_grid, batch)`` after the usual latent-MIM
forward.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.distributed import DeviceMesh
from torch.distributed.fsdp import fully_shard, register_fsdp_forward_method

from olmoearth_pretrain.nn.latent_mim import LatentMIM, LatentMIMConfig
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
        projection_only_target: bool = False,
    ):
        """Initialize the model and attach the probe as a submodule."""
        super().__init__(
            encoder=encoder,
            decoder=decoder,
            reconstructor=reconstructor,
            supervision_head=supervision_head,
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
        if not getattr(self.encoder_config, "use_register_bottleneck", False):
            raise ValueError(
                "OpenSetLatentMIM requires the encoder register bottleneck: the "
                "open-set probe reads the spatial latent grid it produces"
            )

    def build(self) -> OpenSetLatentMIM:
        """Build the model, including the supervised probe head."""
        self.validate()
        assert self.open_set_probe_config is not None
        base = super().build()
        # The probe reads the spatial latent grid, so its input dim is the
        # register (bottleneck) dim, not the encoder token dim.
        register_dim = self.encoder_config.register_dim or (
            self.encoder_config.embedding_size // 2
        )
        probe = self.open_set_probe_config.build(embedding_size=register_dim)
        return OpenSetLatentMIM(
            encoder=base.encoder,
            decoder=base.decoder,
            open_set_probe=probe,
            reconstructor=base.reconstructor,
            supervision_head=base.supervision_head,
            projection_only_target=self.projection_only_target,
        )
