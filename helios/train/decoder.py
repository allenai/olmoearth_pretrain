"""Decoder code for the Latent MIM loss set up."""

import torch
from torch import nn


class SimpleLatentDecoder(nn.Module):
    """Simple decoder module for reconstructing latent representations."""

    def __init__(
        self, embed_dim: int = 768, mlp_ratio: float = 4.0, dropout: float = 0.1
    ):
        """Initialize the SimpleLatentDecoder.

        Args:
            embed_dim: Dimension of the input embeddings.
            mlp_ratio: Multiplier for hidden dimension size.
            dropout: Dropout probability.
        """
        super().__init__()

        # Simple MLP to reconstruct latents
        mlp_hidden_dim = int(embed_dim * mlp_ratio)
        self.decoder = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, encoded: torch.Tensor) -> torch.Tensor:
        """Process the encoded tensor through the decoder.

        Args:
            encoded: Tensor of shape (B, N, C) where:
                B is batch size
                N is number of patches
                C is embedding dimension

        Returns:
            Reconstructed latents of the same shape
        """
        return self.decoder(encoded)
