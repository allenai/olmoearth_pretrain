"""Simple set up of latent predictor with two predictors, following Galileo."""

import logging
from copy import deepcopy
from dataclasses import dataclass

import torch
import torch.nn as nn
from olmo_core.config import Config
from torch.distributed import DeviceMesh
from torch.distributed.fsdp import (
    MixedPrecisionPolicy,
    fully_shard,
    register_fsdp_forward_method,
)

from olmoearth_pretrain.nn.flexi_vit import TokensAndMasks
from olmoearth_pretrain.nn.utils import DistributedMixins, unpack_encoder_output
from olmoearth_pretrain.train.masking import MaskedOlmoEarthSample

logger = logging.getLogger(__name__)


class Galileo(nn.Module, DistributedMixins):
    """Galileo Style."""

    supports_multiple_modalities_at_once = True

    def __init__(
        self,
        encoder: torch.nn.Module,
        decoder: torch.nn.Module,
        reconstructor: torch.nn.Module | None = None,
    ):
        """Initialize the Galileo Style.

        Args:
            encoder: The encoder to use.
            decoder: The decoder to use.
            reconstructor: Optional reconstructor for auto-encoding.
        """
        super().__init__()
        self.encoder = encoder
        self.decoder_a = decoder
        self.decoder_b = deepcopy(decoder)
        self.target_encoder = deepcopy(self.encoder)
        self.reconstructor = reconstructor
        for p in self.target_encoder.parameters():
            p.requires_grad = False

    def forward_a(
        self, x: MaskedOlmoEarthSample, patch_size: int
    ) -> tuple[TokensAndMasks, TokensAndMasks, torch.Tensor, TokensAndMasks | None]:
        """Forward pass for the Latent MIM Style.

        Returns:
            latent: embeddings from encoder
            decoded: predictions from decoder for masked tokens
            latent_projected_and_pooled: pooled tokens for contrastive loss
            reconstructed: MAE predictions if enabled
        """
        # TODO: Input And outputs here are not consistent between encoder and decoder need a tokensandmaks++
        output_dict = self.encoder(x, patch_size=patch_size)
        latent, latent_projected_and_pooled, decoder_kwargs = unpack_encoder_output(
            output_dict
        )
        reconstructed = None
        if self.reconstructor:
            reconstructed = self.reconstructor(latent, x.timestamps, patch_size)
        decoded = self.decoder_a(
            latent, timestamps=x.timestamps, patch_size=patch_size, **decoder_kwargs
        )
        return latent, decoded, latent_projected_and_pooled, reconstructed

    def forward_b(
        self, x: MaskedOlmoEarthSample, patch_size: int
    ) -> tuple[TokensAndMasks, TokensAndMasks, torch.Tensor, TokensAndMasks | None]:
        """Forward pass for the Latent MIM Style.

        Returns:
            latent: embeddings from encoder
            decoded: predictions from decoder for masked tokens
            latent_projected_and_pooled: pooled tokens for contrastive loss
            reconstructed: MAE predictions if enabled
        """
        # TODO: Input And outputs here are not consistent between encoder and decoder need a tokensandmaks++
        output_dict = self.encoder(x, patch_size=patch_size)
        latent, latent_projected_and_pooled, decoder_kwargs = unpack_encoder_output(
            output_dict
        )
        reconstructed = None
        if self.reconstructor:
            reconstructed = self.reconstructor(latent, x.timestamps, patch_size)
        decoded = self.decoder_b(
            latent, timestamps=x.timestamps, patch_size=patch_size, **decoder_kwargs
        )
        return latent, decoded, latent_projected_and_pooled, reconstructed

    def forward(
        self,
        input_a: MaskedOlmoEarthSample,
        input_b: MaskedOlmoEarthSample,
        patch_size: int,
    ) -> dict[
        str, tuple[TokensAndMasks, TokensAndMasks, torch.Tensor, TokensAndMasks | None]
    ]:
        """Forward pass for the Galileo Style."""
        return {
            "a": self.forward_a(input_a, patch_size),
            "b": self.forward_b(input_b, patch_size),
        }

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
        self.decoder_a.apply_fsdp(**fsdp_config)
        self.decoder_b.apply_fsdp(**fsdp_config)
        self.target_encoder.apply_fsdp(**fsdp_config)
        if self.reconstructor:
            self.reconstructor.apply_fsdp(**fsdp_config)
        # TODO: More finegrained wrapping of the encoder transformer layers next time
        fully_shard(self, **fsdp_config)
        register_fsdp_forward_method(self.target_encoder, "forward")

    def apply_compile(self) -> None:
        """Apply torch.compile to the model."""
        self.encoder.apply_compile()
        self.decoder_a.apply_compile()
        self.decoder_b.apply_compile()
        self.target_encoder.apply_compile()
        if self.reconstructor is not None:
            self.reconstructor.apply_compile()


@dataclass
class GalileoConfig(Config):
    """Configuration for the Galileo model."""

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

    def build(self) -> "Galileo":
        """Build the Galileo model."""
        self.validate()
        encoder = self.encoder_config.build()
        decoder = self.decoder_config.build()
        reconstructor = (
            self.reconstructor_config.build()
            if self.reconstructor_config is not None
            else None
        )
        return Galileo(
            encoder=encoder,
            decoder=decoder,
            reconstructor=reconstructor,
        )
