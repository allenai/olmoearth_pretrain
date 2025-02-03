from typing import NamedTuple

from helios.data_contract import DataContract, ArrayTensor


class MaskedDataContract(NamedTuple):
    s2: ArrayTensor  # [B, len(S2_bands), T H, W]
    s2_mask: ArrayTensor
    latlon: ArrayTensor # [B, 2]
    latlon_mask: ArrayTensor
    timestamps: ArrayTensor  # [B, D=3, T], where D=[day, month, year]


def apply_mask(x: DataContract, patch_size: int) -> MaskedDataContract:
    raise NotImplementedError
