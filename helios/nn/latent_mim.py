"""Simple set up of latent predictor."""

from copy import deepcopy
from dataclasses import dataclass

import torch
import torch.nn as nn
from olmo_core.config import Config
from torch.distributed import DeviceMesh
from torch.distributed.fsdp import fully_shard, register_fsdp_forward_method

from helios.nn.flexihelios import EncoderConfig, PredictorConfig, TokensAndMasks
from helios.nn.utils import DistributedMixins
from helios.train.masking import MaskedHeliosSample


class LatentMIM(nn.Module, DistributedMixins):
    """Latent MIM Style."""

    def __init__(
        self,
        encoder: nn.Module,
        decoder: nn.Module,
    ):
        """Initialize the Latent MIM Style.

        Args:
            encoder: The encoder to use.
            decoder: The decoder to use.
        """
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.target_encoder = deepcopy(self.encoder)
        for p in self.target_encoder.parameters():
            p.requires_grad = False

    def forward(self, x: MaskedHeliosSample, patch_size: int) -> TokensAndMasks:
        """Forward pass for the Latent MIM Style."""
        # TODO: Input And outputs here are not consistent between encoder and decoder need a tokensandmaks++
        latent = self.encoder(x, patch_size=patch_size)
        decoded = self.decoder(latent, timestamps=x.timestamps, patch_size=patch_size)
        return decoded

    def apply_fsdp(
        self,
        dp_mesh: DeviceMesh | None = None,
        param_dtype: torch.dtype | None = None,
        reduce_dtype: torch.dtype = torch.float32,
        prefetch_factor: int = 0,
    ) -> None:
        """Apply FSDP to the model."""
        fsdp_config = dict(mesh=dp_mesh)

        self.encoder.apply_fsdp(**fsdp_config)
        self.decoder.apply_fsdp(**fsdp_config)
        self.target_encoder.apply_fsdp(**fsdp_config)
        # TODO: More finegrained wrapping of the encoder transformer layers next time
        fully_shard(self, **fsdp_config)
        register_fsdp_forward_method(self.target_encoder, "forward")


@dataclass
class LatentMIMConfig(Config):
    """Configuration for the Latent Predictor."""

    encoder_config: "EncoderConfig"
    decoder_config: "PredictorConfig"

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

    def build(self) -> "LatentMIM":
        """Build the Latent Predictor."""
        self.validate()
        encoder = self.encoder_config.build()
        decoder = self.decoder_config.build()
        return LatentMIM(
            encoder=encoder,
            decoder=decoder,
        )
