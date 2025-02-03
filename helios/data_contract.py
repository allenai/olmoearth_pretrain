from collections.abc import Sequence
from typing import NamedTuple, Union

import numpy as np
import torch
from torch.utils.data import default_collate

NODATAVALUE = 65535

S2_BANDS = [
    "B1",
    "B2",
    "B3",
    "B4",
    "B5",
    "B6",
    "B7",
    "B8",
    "B8A",
    "B9",
    "B10",
    "B11",
    "B12",
]
LATLON_BANDS = ["lat", "lon"]
TIMESTAMPS = ["day", "month", "year"]

ArrayTensor = Union[np.ndarray, torch.Tensor]


REFERENCE_RESOLUTION = 10


class DataContract(NamedTuple):
    # if an attribute is added here, its bands must also
    # be added to attribute_to_bands

    # input shape is (B, C, T, H, W)
    s2: ArrayTensor | None = None  # [B, len(S2_bands), T H, W]
    latlon: ArrayTensor | None = None  # [B, 2]
    timestamps: ArrayTensor | None = None  # [B, D=3, T], where D=[day, month, year]

    def as_dict(self, ignore_nones: bool = True):
        return_dict = {}
        for field in self._fields:
            val = getattr(self, field)
            if ignore_nones and (val is None):
                continue
            else:
                return_dict[field] = val
        return return_dict

    @staticmethod
    def attribute_to_bands() -> dict[str, list[str]]:
        return {"s2": S2_BANDS, "latlon": LATLON_BANDS, "timestamps": TIMESTAMPS}

    @property
    def t(self):
        return self.timestamps.shape[-1]

    @property
    def h(self):
        # if we had NAIP only, we'd do something like
        # if s2 is None:
        #     naip_height = self.naip[-2]
        # return naip_height / 10  (since we have a reference res of 10)
        return self.s2.shape[-2]

    @property
    def w(self):
        return self.s2.shape[-1]


def collate_fn(batch: Sequence[DataContract]):
    # we assume that the same values are consistently None within a batch
    collated_dict = default_collate([i.as_dict(ignore_nones=True) for i in batch])
    return DataContract(**collated_dict)
