"""Model code for the Helios model."""

from typing import NamedTuple

import torch
from torch import Tensor, nn

from helios.train.masking import MaskedHeliosSample


class TokensAndMasks(NamedTuple):
    """Output to compute the loss on.

    Args:
        s2: sentinel 2 data of shape (B, C_G, T, P_H, P_W, D)
        s2_mask: sentinel 2 mask indicating which tokens are masked/unmasked
        latlon: lat lon data containing geographical coordinates (B, 1, D)
        latlon_mask: lat lon mask indicating which coordinates are masked/unmasked
    """

    s2: Tensor  # (B, C_G, T, P_H, P_W, D)
    s2_mask: Tensor
    latlon: Tensor
    latlon_mask: Tensor

    @property
    def device(self) -> torch.device:
        """Get the device of the tokens and masks."""
        return self.s2.device

    @property
    def data_fields(self) -> list[str]:
        """Return all data fields."""
        return [x for x in self._fields if not x.endswith("mask")]


class Encoder(nn.Module):
    """Encoder module that processes masked input samples into token representations."""

    def forward(self, x: MaskedHeliosSample, patch_size: int) -> TokensAndMasks:
        """Process masked input samples into token representations.

        Args:
            x: Masked input sample containing the data to be encoded
            patch_size: Size of patches to divide the input into

        Returns:
            TokensAndMasks containing the encoded representations and their masks
        """
        raise NotImplementedError


class Predictor(nn.Module):
    """Predictor module that generates predictions from encoded tokens."""

    def forward(self, x: TokensAndMasks) -> TokensAndMasks:
        """Generate predictions from encoded token representations.

        Args:
            x: TokensAndMasks containing the encoded tokens to make predictions from

        Returns:
            TokensAndMasks containing the predicted tokens and their masks
        """
        raise NotImplementedError
