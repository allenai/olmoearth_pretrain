"""Simple set up of latent predictor."""

from dataclasses import dataclass

import torch.nn as nn
from olmo_core.config import Config

from helios.nn.flexihelios import (
    EncoderConfig,
    PredictorConfig,
    ReconstructorConfig,
    TokensAndMasks,
)
from helios.nn.utils import DistributedMixins
from helios.train.masking import MaskedHeliosSample


class MAE(nn.Module, DistributedMixins):
    """Masked Auto-Encoder Module."""

    def __init__(
        self,
        encoder: nn.Module,
        decoder: nn.Module,
        reconstructor: nn.Module,
    ):
        """Initialize the MAE Module.

        Args:
            encoder: The encoder to use.
            decoder: The decoder to use.
            reconstructor: The reconstructor to use.
        """
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.reconstructor = reconstructor

    def forward(
        self, x: MaskedHeliosSample, patch_size: int
    ) -> tuple[TokensAndMasks, TokensAndMasks]:
        """Forward pass for the MAE Module."""
        latent = self.encoder(x, patch_size=patch_size)
        decoded = self.decoder(latent, timestamps=x.timestamps, patch_size=patch_size)
        reconstructed = self.reconstructor(decoded, patch_size=patch_size)
        return latent, reconstructed

    def apply_compile(self) -> None:
        """Apply torch.compile to the model."""
        self.encoder.apply_compile()
        self.decoder.apply_compile()
        # TODO: add aaply for constructor


@dataclass
class MAEConfig(Config):
    """Configuration for the MAE."""

    encoder_config: "EncoderConfig"
    decoder_config: "PredictorConfig"
    reconstructor_config: "ReconstructorConfig"

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
        reconstructor = self.reconstructor_config.build()
        return MAE(
            encoder=encoder,
            decoder=decoder,
            reconstructor=reconstructor,
        )
