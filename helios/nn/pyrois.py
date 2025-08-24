"""Simple set up of latent predictor."""

import logging
from copy import deepcopy
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
from olmo_core.config import Config
from torch.distributed import DeviceMesh
from torch.distributed.fsdp import (
    MixedPrecisionPolicy,
    fully_shard,
    register_fsdp_forward_method,
)

from helios.nn.utils import DistributedMixins
from helios.train.masking import MaskedHeliosSample

logger = logging.getLogger(__name__)


class Pyrois(nn.Module, DistributedMixins):
    """Latent MIM Style."""

    def __init__(
        self,
        encoder: nn.Module,
        decoder: nn.Module,
        reconstructor: torch.nn.Module,
    ):
        """Initialize the Latent MIM Style.

        Args:
            encoder: The encoder to use.
            decoder: The decoder to use.
            reconstructor: Optional reconstructor for auto-encoding.
        """
        super().__init__()
        self.encoder = encoder
        self.projector = deepcopy(encoder.patch_embeddings)

        self.decoder = decoder
        self.proj_decoder = deepcopy(decoder)
        self.reconstructor = reconstructor

        for p in self.projector.parameters():
            p.requires_grad = False

    def forward(self, x: MaskedHeliosSample, patch_size: int) -> dict[str, Any]:
        """Forward pass for the Latent MIM Style.

        Returns:
            latent: embeddings from encoder
            decoded: predictions from decoder for masked tokens
            latent_projected_and_pooled: pooled tokens for contrastive loss
            reconstructed: MAE predictions if enabled
        """
        # TODO: Input And outputs here are not consistent between encoder and decoder need a tokensandmaks++
        out = {}
        out["latent"], out["pooled"] = self.encoder(x, patch_size=patch_size)
        out["reconstructed"] = self.reconstructor(
            out["latent"], x.timestamps, patch_size
        )
        out["decoded"] = self.decoder(
            out["latent"], timestamps=x.timestamps, patch_size=patch_size
        )
        out["decoded_proj"] = self.proj_decoder(
            out["latent"], timestamps=x.timestamps, patch_size=patch_size
        )
        return out

    def apply_fsdp(
        self,
        dp_mesh: DeviceMesh | None = None,
        param_dtype: torch.dtype | None = None,
        reduce_dtype: torch.dtype = torch.float32,
        prefetch_factor: int = 0,
    ) -> None:
        """Apply FSDP to the model."""
        mp_policy = MixedPrecisionPolicy(
            param_dtype=param_dtype, reduce_dtype=reduce_dtype
        )
        fsdp_config = dict(mesh=dp_mesh, mp_policy=mp_policy)

        self.encoder.apply_fsdp(**fsdp_config)
        self.decoder.apply_fsdp(**fsdp_config)
        self.proj_decoder.apply_fsdp(**fsdp_config)
        self.reconstructor.apply_fsdp(**fsdp_config)
        # TODO: More finegrained wrapping of the encoder transformer layers next time
        fully_shard(self.projector, **fsdp_config)
        fully_shard(self, **fsdp_config)
        register_fsdp_forward_method(self.projector, "forward")
        register_fsdp_forward_method(self.encoder, "forward")

    def apply_compile(self) -> None:
        """Apply torch.compile to the model."""
        logger.info("Applying torch.compile to the model")
        self.encoder.apply_compile()
        logger.info("Applied torch.compile to the encoder")
        self.decoder.apply_compile()
        logger.info("Applied torch.compile to the decoder")


@dataclass
class PyroisConfig(Config):
    """Configuration for the Latent Predictor."""

    encoder_config: Config
    decoder_config: Config
    reconstructor_config: Config | None = None

    def validate(self) -> None:
        """Validate the configuration."""
        if (
            self.encoder_config.supported_modalities
            != self.decoder_config.supported_modalities
        ):
            raise ValueError("Encoder and decoder must support the same modalities")
        if (
            self.encoder_config.max_sequence_length
            != self.decoder_config.max_sequence_length
        ):
            raise ValueError(
                "Encoder and decoder must have the same max sequence length"
            )
        if (
            self.encoder_config.embedding_size
            != self.decoder_config.encoder_embedding_size
        ):
            raise ValueError("Encoder embedding size must be consistent!")

    def build(self) -> "Pyrois":
        """Build the Latent Predictor."""
        self.validate()
        encoder = self.encoder_config.build()
        decoder = self.decoder_config.build()
        reconstructor = (
            self.reconstructor_config.build()
            if self.reconstructor_config is not None
            else None
        )
        return Pyrois(
            encoder=encoder,
            decoder=decoder,
            reconstructor=reconstructor,
        )
