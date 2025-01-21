"""Dataset module for helios."""

import logging
from typing import Any

import numpy as np
import rasterio
from einops import rearrange
from olmo_core.data.numpy_dataset import NumpyDatasetBase
from torch.utils.data import Dataset
from upath import UPath

from helios.data.constants import DATA_SOURCE_TO_VARIATION_TYPE, S2_BANDS

logger = logging.getLogger(__name__)


class HeliosDataset(NumpyDatasetBase, Dataset):
    """Helios dataset."""

    def __init__(self, *samples: dict[str, Any], dtype: np.dtype):
        """Initialize the dataset.

        Things that would need to be optional or should be forgotten about, or changed
        - paths would need to ba dictionary or collection of paths for this to work
        - pad_token_id: int,
        - eos_token_id: int,
        - vocab_size: int,
        """
        super().__init__(
            *samples,
            dtype=dtype,
            pad_token_id=-1,  # Not needed only LM
            eos_token_id=-1,  # Not needed only LM
            vocab_size=-1,  # Not needed only LM
        )
        #
        # What does it look like for me to access paths?

        # After init we have
        # paths to samples
        # numpy data type
        pass

    @property
    def max_sequence_length(self) -> int:
        """Max sequence length."""
        # NOT SUPER needed
        return max(item["num_timesteps"] for item in self.paths)

    @property
    def fingerprint(self) -> str:
        """Fingerprint of the dataset."""
        # LM specific
        raise NotImplementedError("Fingerprint not implemented")

    def __len__(self) -> int:
        """Get the length of the dataset."""
        return len(self.paths)

    def _tif_to_array(
        self, tif_path: UPath | str, data_source: str
    ) -> tuple[np.ndarray, float]:
        """Convert a tif file to an array.

        Args:
            tif_path: The path to the tif file.
            data_source: The data source string to load the correct datasource
        Returns:
            The array from the tif file.
        """
        if data_source == "sentinel2":
            space_bands = S2_BANDS
        else:
            raise ValueError(f"Unknown data source: {data_source}")
        # We will need different ingestion logic for different data sources at this point

        variation_type = DATA_SOURCE_TO_VARIATION_TYPE[data_source]
        if variation_type == "space_time_varying":
            with rasterio.open(tif_path) as data:
                values = data.read()

            num_timesteps = values.shape[0] / len(space_bands)
            assert (
                num_timesteps % 1 == 0
            ), f"{tif_path} has incorrect number of channels {space_bands} \
                {values.shape[0]=} {len(space_bands)=}"
            space_time_x = rearrange(
                values, "(t c) h w -> h w t c", c=len(space_bands), t=int(num_timesteps)
            )
            return space_time_x, num_timesteps
        else:
            raise NotImplementedError(f"Unknown variation type: {variation_type}")

    def _tif_to_array_with_checks(
        self, tif_path: UPath | str, data_source: str
    ) -> tuple[np.ndarray, float]:
        """Load the tif file and return the array.

        Args:
            tif_path: The path to the tif file.
            data_source: The data source.

        Returns:
            The array from the tif file.
        """
        try:
            output = self._tif_to_array(tif_path, data_source)
            return output
        except Exception as e:
            logger.error(f"Replacing tif {tif_path} due to {e}")
            # TODO: Implement behavior so that we don't hit bad indices again
            # For now just raise the error so it fails obviously
            raise e

    def __getitem__(self, index: int) -> dict[str, Any]:
        """Get the item at the given index."""
        sample = self.paths[index]  # THis really is the dict of all the sample info
        data_source_paths = sample["data_source_paths"]
        data_arrays = {}
        for data_source, tif_path in data_source_paths.items():
            data_source_array, num_timesteps = self._tif_to_array_with_checks(
                tif_path, data_source
            )
            data_arrays[data_source] = data_source_array
        output_dict: dict[str, Any] = {"data_arrays": data_arrays}
        output_dict["sample_metadata"] = sample["sample_metadata"]
        output_dict["num_timesteps"] = num_timesteps
        output_dict["data_source_metadata"] = sample["data_source_metadata"]
        return output_dict
