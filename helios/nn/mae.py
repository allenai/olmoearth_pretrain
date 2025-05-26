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
        decoder: nn.Module | None = None,
        reconstructor: nn.Module | None = None,
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
    ) -> tuple[TokensAndMasks, TokensAndMasks | None, TokensAndMasks | None]:
        """Forward pass for the MAE Module."""
        latent, _ = self.encoder(x, patch_size=patch_size)
        decoded = self.decoder and self.decoder(
            latent, timestamps=x.timestamps, patch_size=patch_size
        )
        reconstructed = self.reconstructor and self.reconstructor(
            latent, timestamps=x.timestamps, patch_size=patch_size
        )

        return latent, decoded, reconstructed

    def apply_compile(self) -> None:
        """Apply torch.compile to the model."""
        self.encoder.apply_compile()
        if self.decoder is not None:
            self.decoder.apply_compile()
        if self.reconstructor is not None:
            self.reconstructor.apply_compile()
        # TODO: add aaply for constructor


@dataclass
class MAEConfig(Config):
    """Configuration for the MAE."""

    encoder_config: EncoderConfig
    decoder_config: PredictorConfig | None = None
    reconstructor_config: ReconstructorConfig | None = None

    def validate(self) -> None:
        """Validate the configuration."""
        if self.decoder_config is not None:
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
        if self.reconstructor_config is not None:
            if (
                self.encoder_config.supported_modalities
                != self.reconstructor_config.supported_modalities
            ):
                raise ValueError(
                    "Encoder and reconstructor must support the same modalities"
                )
            if (
                self.encoder_config.max_sequence_length
                != self.reconstructor_config.decoder_config.max_sequence_length
            ):
                raise ValueError(
                    "Encoder and reconstructor must have the same max sequence length"
                )
            if (
                self.encoder_config.embedding_size
                != self.reconstructor_config.decoder_config.encoder_embedding_size
            ):
                raise ValueError("Encoder embedding size must be consistent!")

    def build(self) -> "MAE":
        """Build the MAE Predictor."""
        self.validate()
        encoder = self.encoder_config.build()
        decoder = (
            self.decoder_config.build() if self.decoder_config is not None else None
        )
        reconstructor = (
            self.reconstructor_config.build()
            if self.reconstructor_config is not None
            else None
        )
        return MAE(
            encoder=encoder,
            decoder=decoder,
            reconstructor=reconstructor,
        )
