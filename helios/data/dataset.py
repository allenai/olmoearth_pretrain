"""Dataset module for helios."""

import logging
from typing import Any, NamedTuple

import numpy as np
from olmo_core.data.numpy_dataset import NumpyDatasetBase
from torch.utils.data import Dataset
from upath import UPath

from helios.constants import LATLON_BANDS, S2_BANDS, TIMESTAMPS
from helios.data.data_source_io import DataSourceReader, DataSourceReaderRegistry
from helios.dataset.index import SampleInformation
from helios.types import ArrayTensor

logger = logging.getLogger(__name__)


# TODO: THIS SHOULD BE THE OUTPUT OF THE DATASET GET ITEM
class HeliosSample(NamedTuple):
    """A sample of the data from the Helios dataset.

    This is a namedtuple that contains the data for a single sample from the Helios dataset.
    For each modality. we have an ArrayTensor named by modality, positions in lat lon of each sample and
    timestamps of each sample.

    Args:
        s2: ArrayTensor | None = None  # [B, len(S2_bands), T H, W]
        latlon: ArrayTensor | None = None  # [B, 2]
        timestamps: ArrayTensor | None = None  # [B, D=3, T], where D=[day, month, year]
    """

    # if an attribute is added here, its bands must also
    # be added to attribute_to_bands

    # input shape is (B, C, T, H, W)
    s2: ArrayTensor | None = None  # [B, len(S2_bands), T H, W]
    latlon: ArrayTensor | None = None  # [B, 2]
    timestamps: ArrayTensor | None = None  # [B, D=3, T], where D=[day, month, year]

    def as_dict(self, ignore_nones: bool = True) -> dict[str, Any]:
        """Convert the namedtuple to a dictionary.

        Args:
            ignore_nones: Whether to ignore None values.

        Returns:
            Dictionary representation of the namedtuple.
        """
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
        """Get the bands for each attribute.

        Returns:
            A dictionary mapping attribute names to their corresponding bands.
        """
        return {"s2": S2_BANDS, "latlon": LATLON_BANDS, "timestamps": TIMESTAMPS}

    @property
    def b(self) -> int:
        """Get the batch size.

        Returns:
            The batch size of the sample.
        """
        if self.s2 is None:
            raise ValueError("S2 is not present in the sample")
        if len(self.s2.shape) == 5:
            return self.s2.shape[0]
        else:
            raise ValueError("This is a single sample and not a batch")

    @property
    def t(self) -> int:
        """Get the number of timesteps.

        Returns:
            The number of timesteps in the sample.
        """
        if self.s2 is None:
            raise ValueError("S2 is not present in the sample")
        return self.s2.shape[-3]

    @property
    def h(self) -> int:
        """Get the height of the image.

        Returns:
            The height of the image in the sample.
        """
        if self.s2 is None:
            raise ValueError("S2 is not present in the sample")
        return self.s2.shape[-2]

    @property
    def w(self) -> int:
        """Get the width of the image.

        Returns:
            The width of the image in the sample.
        """
        if self.s2 is None:
            raise ValueError("S2 is not present in the sample")
        return self.s2.shape[-1]


class HeliosDataset(NumpyDatasetBase, Dataset):
    """Helios dataset."""

    def __init__(
        self,
        *samples: SampleInformation,
        dtype: np.dtype,
    ):
        """Initialize the dataset.
        
        Args:
            samples: The samples to include in the dataset.
            dtype: The dtype of the data.
        """
        self.samples = samples
        self.dtype = dtype

    def __len__(self) -> int:
        """Get the length of the dataset."""
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        """Get the item at the given index."""
        sample: SampleInformation = self.samples[index]
        # TODO: add data loading and processing here
        return sample
