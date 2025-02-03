"""Simple set up of latent predictor."""

from copy import deepcopy

import torch
import torch.nn as nn


class LatentMIMStyle(nn.Module):
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for the Latent MIM Style."""
        latent = self.encoder(x)
        decoded = self.decoder(latent)
        return decoded
