"""Simple set up of latent predictor."""

from copy import deepcopy
from dataclasses import dataclass

import torch.nn as nn
from olmo_core.config import Config

from helios.nn.flexihelios import EncoderConfig, PredictorConfig, TokensAndMasks
from helios.train.masking import MaskedHeliosSample


class LatentMIM(nn.Module):
    """Latent MIM Style."""

    def __init__(
        self,
        encoder: nn.Module,
        decoder: nn.Module,
        token_budget: int = 1500,
        h_w_to_sample_min: int = 2,
        h_w_to_sample_max: int = 13,
    ):
        """Initialize the Latent MIM Style.

        Args:
            encoder: The encoder to use.
            decoder: The decoder to use.
            token_budget: The token budget to use.
            h_w_to_sample_min: The minimum height and width to sample.
            h_w_to_sample_max: The maximum height and width to sample.
        """
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.target_encoder = deepcopy(self.encoder)
        for p in self.target_encoder.parameters():
            p.requires_grad = False
        self.token_budget = token_budget
        self.h_w_to_sample_min = h_w_to_sample_min
        self.h_w_to_sample_max = h_w_to_sample_max

    def forward(self, x: MaskedHeliosSample, patch_size: int) -> TokensAndMasks:
        """Forward pass for the Latent MIM Style."""
        # TODO: Input And outputs here are not consistent between encoder and decoder need a tokensandmaks++
        latent = self.encoder(x, patch_size=patch_size)
        decoded = self.decoder(latent, timestamps=x.timestamps, patch_size=patch_size)
        return decoded


@dataclass
class LatentMIMConfig(Config):
    """Configuration for the Latent Predictor."""

    encoder_config: "EncoderConfig"
    decoder_config: "PredictorConfig"
    token_budget: int = 1500
    h_w_to_sample_min: int = 2
    h_w_to_sample_max: int = 13

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
