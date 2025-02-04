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
        ignore_data_sources: list[str] = [],
        filter_samples_with_missing_inputs: bool = False,
        dtype: np.dtype,
    ):
        """Initialize the dataset.

        Things that would need to be optional or should be forgotten about, or changed
        - paths would need to ba dictionary or collection of paths for this to work
        - pad_token_id: int,
        - eos_token_id: int,
        - vocab_size: int,
        """
        self.ignore_data_sources = ignore_data_sources
        if filter_samples_with_missing_inputs:
            filtered_samples = [
                sample
                for sample in samples
                if not sample.sample_metadata["has_missing_inputs"]
            ]
        else:
            filtered_samples = list(samples)
        super().__init__(
            *filtered_samples,
            dtype=dtype,
            pad_token_id=-1,  # Not needed only LM
            eos_token_id=-1,  # Not needed only LM
            vocab_size=-1,  # Not needed only LM
        )

    @property
    def max_sequence_length(self) -> int:
        """Max sequence length."""
        # NOT SUPER needed
        # instances are always based on batch size
        return 1

    @property
    def fingerprint(self) -> str:
        """Fingerprint of the dataset."""
        import hashlib

        sha256_hash = hashlib.sha256()
        sha256_hash.update(f"dtype={self.dtype}".encode())
        logger.warning("Fingerprint: is not yet meaningful")
        return sha256_hash.hexdigest()

    def __len__(self) -> int:
        """Get the length of the dataset."""
        return len(self.paths)

    def _load_data_source(
        self, file_path: UPath | str, data_source: str
    ) -> np.ndarray | dict[str, Any]:
        """Load data from a data source using the appropriate reader.

        Args:
            file_path: Path to the data file
            data_source: Name of the data source

        Returns:
            Either a numpy array or a dictionary of data
        """
        try:
            reader: DataSourceReader = DataSourceReaderRegistry.get_class(data_source)
            return reader.load(file_path)
        except Exception as e:
            logger.error(f"Error loading {file_path} from {data_source}: {e}")
            raise

    def __getitem__(self, index: int) -> dict[str, Any]:
        """Get the item at the given index."""
        sample: SampleInformation = self.paths[index]
        data_source_paths = sample.data_source_paths
        data_inputs = {}
        max_num_timesteps = 1
        for data_source, file_path in data_source_paths.items():
            if data_source in self.ignore_data_sources:
                continue
            data_input = self._load_data_source(file_path, data_source)
            if isinstance(data_input, np.ndarray):
                max_num_timesteps = max(max_num_timesteps, data_input.shape[2])
                # make the data all have same dtype
                data_input = data_input.astype(self.dtype)
            data_inputs[data_source] = data_input

        return {
            "data_inputs": data_inputs,
            "sample_metadata": sample.sample_metadata,
            "num_timesteps": max_num_timesteps,
            "data_source_metadata": sample.data_source_metadata,
        }
