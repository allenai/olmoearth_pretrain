"""Simple set up of latent predictor."""

from dataclasses import dataclass

import torch.nn as nn
from olmo_core.config import Config

from helios.data.transform import Transform, TransformConfig
from helios.nn.flexihelios import EncoderConfig, PredictorConfig, TokensAndMasks
from helios.nn.utils import DistributedMixins
from helios.train.masking import MaskedHeliosSample


class MAE(nn.Module, DistributedMixins):
    """Masked Auto-Encoder Module."""

    def __init__(
        self,
        encoder: nn.Module,
        decoder: nn.Module,
        transform: Transform,
        token_budget: int = 1500,
        h_w_to_sample_min: int = 8,
        h_w_to_sample_max: int = 8,
    ):
        """Initialize the MAE Module.

        Args:
            encoder: The encoder to use.
            decoder: The decoder to use.
            transform: The transform to use.
            token_budget: The token budget to use.
            h_w_to_sample_min: The minimum height and width to sample.
            h_w_to_sample_max: The maximum height and width to sample.
        """
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.token_budget = token_budget
        self.transform = transform
        self.h_w_to_sample_min = h_w_to_sample_min
        self.h_w_to_sample_max = h_w_to_sample_max

    def forward(self, x: MaskedHeliosSample, patch_size: int) -> TokensAndMasks:
        """Forward pass for the MAE Module."""
        # TODO: Input And outputs here are not consistent between encoder and decoder need a tokensandmaks++
        latent = self.encoder(x, patch_size=patch_size)
        decoded = self.decoder(latent, timestamps=x.timestamps, patch_size=patch_size)
        return decoded


@dataclass
class MAEConfig(Config):
    """Configuration for the MAE."""

    encoder_config: "EncoderConfig"
    decoder_config: "PredictorConfig"
    transform_type: str = "no_transform"
    token_budget: int = 1500
    h_w_to_sample_min: int = 8
    h_w_to_sample_max: int = 8

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

    def build(self) -> "MAE":
        """Build the MAE Predictor."""
        self.validate()
        encoder = self.encoder_config.build()
        decoder = self.decoder_config.build()
        transform = TransformConfig(transform_type=self.transform_type).build()
        return MAE(
            encoder=encoder,
            decoder=decoder,
            transform=transform,
            token_budget=self.token_budget,
            h_w_to_sample_min=self.h_w_to_sample_min,
            h_w_to_sample_max=self.h_w_to_sample_max,
        )
