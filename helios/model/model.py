from typing import NamedTuple

from torch import Tensor
from .masking import MaskedDataContract


class TokensAndMasks(NamedTuple):
    s2: Tensor   # (B, C_G, T, P_H, P_W)
    s2_mask: Tensor
    latlon: Tensor
    latlon_mask: Tensor


class Encoder(nn.Module):

    def forward(x: MaskedDataContract, patch_size: int) -> TokensAndMasks:
        raise NotImplementedError


class Predictor(nn.Module):

    def forward(x: TokensAndMasks) -> TokensAndMasks:
        raise NotImplementedError
