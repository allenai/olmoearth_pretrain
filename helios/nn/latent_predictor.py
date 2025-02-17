"""Simple set up of latent predictor."""

from copy import deepcopy
from dataclasses import dataclass

import torch.nn as nn
from olmo_core.config import Config

from helios.nn.flexihelios import EncoderConfig, PredictorConfig, TokensAndMasks
from helios.train.masking import MaskedHeliosSample


class LatentMIM(nn.Module):
    """Latent MIM Style."""

    def __init__(self, encoder: nn.Module, decoder: nn.Module):
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

    def forward(
        self,
        x: MaskedHeliosSample,
        patch_size: int,
    ) -> TokensAndMasks:
        """Forward pass for the Latent MIM Style."""
        # TODO: Input And outputs here are not consistent between encoder and decoder need a tokensandmaks++
        latent = self.encoder(x, patch_size=patch_size)
        decoded = self.decoder(latent, timestamps=x.timestamps, patch_size=patch_size)
        return decoded


@dataclass
class LatentMIMConfig(Config):
    """Configuration for the Latent Predictor."""

    encoder_config: EncoderConfig
    decoder_config: PredictorConfig

    def validate(self) -> None:
        """Validate the configuration."""
        self.encoder_config.validate()
        self.decoder_config.validate()
        if (
            self.encoder_config.supported_modalities
            != self.decoder_config.supported_modalities
        ):
            raise ValueError("Encoder and decoder must support the same modalities")
        if self.encoder_config.max_patch_size != self.decoder_config.max_patch_size:
            raise ValueError("Encoder and decoder must have the same max patch size")
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
            raise ValueError("Encoder and decoder must have the same embedding size")

    def build(self) -> "LatentMIM":
        """Build the Latent Predictor."""
        encoder = self.encoder_config.build()
        decoder = self.decoder_config.build()
        return LatentMIM(encoder, decoder)
