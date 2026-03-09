"""Spectral mixer module for cross-band interaction before patch embedding."""

import torch
import torch.nn as nn


class SpectralMixer(nn.Module):
    """Lightweight cross-band MLP mixer applied pixel-wise before patch embedding.

    Learns non-linear spectral combinations (e.g., NDVI-like ratios) across all
    bands before spatial aggregation by the patch embedding Conv2d. This restores
    cross-spectral learning that is otherwise lost when using a single flat
    bandset, since the Conv2d can only learn linear band combinations.

    Applied after band dropout (if any) so the mixer also learns to be robust
    to partial band observations.

    Initialized as identity (zero-init on the output projection) so training
    starts from the same point as a model without the mixer.

    Args:
        num_bands: Number of spectral bands in this bandset.
        expansion: Hidden dim multiplier for the inner MLP. Default: 4.
    """

    def __init__(self, num_bands: int, expansion: int = 4) -> None:
        """Initialize SpectralMixer."""
        super().__init__()
        hidden = num_bands * expansion
        self.norm = nn.LayerNorm(num_bands)
        self.fc1 = nn.Linear(num_bands, hidden)
        self.fc2 = nn.Linear(hidden, num_bands)
        self.act = nn.GELU()
        # Zero-init so the mixer starts as a pure residual (identity transform).
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply cross-band mixing.

        Args:
            x: Any-shape tensor with bands in the last dimension [..., num_bands].

        Returns:
            Spectrally mixed tensor of the same shape.
        """
        return x + self.fc2(self.act(self.fc1(self.norm(x))))
