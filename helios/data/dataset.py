"""Dataset module for helios."""

from typing import Literal, NamedTuple, cast

import numpy as np
import rioxarray
import xarray as xr
from einops import rearrange
from torch.utils.data import Dataset as PyTorchDataset
from upath import UPath

from helios.data.utils import (
    load_data_index,
    load_sentinel2_frequency_metadata,
    load_sentinel2_monthly_metadata,
)

# TODO: Move these to a .sources folder specific to each data source
# WARNING: TEMPORARY BANDS: We forgot to pull B9, B10 from the export
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
    "B11",
    "B12",
]


class ArrayWithMetadata(NamedTuple):
    """A named tuple for storing the output of the dataset to the model (a single sample)."""

    array: np.ndarray
    metadata: dict


class DatasetOutput(NamedTuple):
    """A named tuple for storing the output of the dataset to the model (a single sample).

    The output is a dictionary of data sources with the array and metadata.
    """

    sentinel2: ArrayWithMetadata
    sample_metadata: dict


# TODO: Adding a Dataset specific fingerprint is probably good for an evolving dataset
# TODO: We want to make what data sources and examples we use configuration drivend

# Quick and dirty interface for data sources
ALL_DATA_SOURCES = ["sentinel2"]

DATA_FREQUENCY_TYPES = ["freq", "monthly"]

LOAD_DATA_SOURCE_METADATA_FUNCTIONS = {
    "sentinel2": {
        "freq": load_sentinel2_frequency_metadata,
        "monthly": load_sentinel2_monthly_metadata,
    },
}

# Quick and dirty interface for data source variation types
DATA_SOURCE_VARIATION_TYPES = Literal[
    "space_time_varying", "time_varying_only", "space_varying_only", "static_only"
]

DATA_SOURCE_TO_VARIATION_TYPE = {
    "sentinel2": "space_time_varying",
}


# Expected types of Data Sources
# Space-Time varying
# Time varying only
# Space varying only
# Static only
# For a given location and or time we want to be able to coalesce the data sources
class HeliosDataset(PyTorchDataset):
    """Helios dataset."""

    def __init__(self, data_index_path: UPath | str):
        """Initialize the dataset."""
        self.data_sources = ALL_DATA_SOURCES
        self.data_index_path = UPath(data_index_path)
        self.root_dir = self.data_index_path.parent
        # Using a df as initial ingest due to ease of inspection and manipulation,
        self.data_index_df = load_data_index(data_index_path)
        self.example_id_to_index_metadata_dict = self.data_index_df.set_index(
            "example_id"
        ).to_dict("index")

        self.freq_metadata_df_dict = {}
        self.monthly_metadata_df_dict = {}
        for data_source in self.data_sources:
            self.freq_metadata_df_dict[data_source] = (
                LOAD_DATA_SOURCE_METADATA_FUNCTIONS[
                    data_source
                ]["freq"](self.get_path_to_data_source_metadata(data_source, "freq"))
            )
            self.monthly_metadata_df_dict[data_source] = (
                LOAD_DATA_SOURCE_METADATA_FUNCTIONS[
                    data_source
                ][
                    "monthly"
                ](self.get_path_to_data_source_metadata(data_source, "monthly"))
            )

        # Intersect available data sources with index column names

        assert (
            len(self.data_sources) > 0
        ), "No data sources found in index, check naming of columns"

        # Get example IDs where at least one data source has monthly data
        monthly_mask = (
            self.data_index_df[
                [col for col in self.data_index_df.columns if "monthly" in col]
            ]
            .eq("y")
            .any(axis=1)
        )
        monthly_example_ids = self.data_index_df.loc[
            monthly_mask, "example_id"
        ].to_numpy(dtype=str)
        freq_mask = (
            self.data_index_df[
                [col for col in self.data_index_df.columns if "freq" in col]
            ]
            .eq("y")
            .any(axis=1)
        )
        freq_example_ids = self.data_index_df.loc[freq_mask, "example_id"].to_numpy(
            dtype=str
        )

        self.root_dir = self.data_index_path.parent

        # Store the example IDs and create indices
        self.monthly_example_ids = monthly_example_ids
        self.freq_example_ids = freq_example_ids

        # Create separate indices for monthly and frequency data
        self.monthly_indices = np.arange(len(monthly_example_ids))
        self.freq_indices = np.arange(len(freq_example_ids))

    def get_path_to_data_source_metadata(
        self, data_source: str, frequency_type: str
    ) -> UPath:
        """Get the path to the data source metadata."""
        return self.root_dir / f"{data_source}_{frequency_type}.csv"

    def __len__(self) -> int:
        """Get the length of the dataset."""
        return len(self.monthly_example_ids) + len(self.freq_example_ids)

    def get_example_from_index(
        self, index: int
    ) -> tuple[str, Literal["monthly", "freq"]]:
        """Convert a global index to an example ID and its type.

        Args:
            index: Global index between 0 and len(dataset)-1

        Returns:
            tuple: (example_id, data_type)
        """
        if index < len(self.monthly_example_ids):
            return self.monthly_example_ids[index], "monthly"
        else:
            freq_index = index - len(self.monthly_example_ids)
            return self.freq_example_ids[freq_index], "freq"

    def _tif_to_array(self, tif_path: UPath | str, data_source: str) -> np.ndarray:
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
            with cast(xr.Dataset, rioxarray.open_rasterio(tif_path)) as data:
                # [all_combined_bands, H, W]
                # all_combined_bands includes all dynamic-in-time bands
                # interleaved for all timesteps
                # followed by the static-in-time bands
                values = cast(np.ndarray, data.values)
                # lon = np.mean(cast(np.ndarray, data.x)).item()
                # lat = np.mean(cast(np.ndarray, data.y)).item()

            num_timesteps = values.shape[0] / len(space_bands)
            assert (
                num_timesteps % 1 == 0
            ), f"{tif_path} has incorrect number of channels {space_bands} \
                {values.shape[0]=} {len(space_bands)=}"
            dynamic_in_time_x = rearrange(
                values, "(t c) h w -> h w t c", c=len(space_bands), t=int(num_timesteps)
            )
            return dynamic_in_time_x
        else:
            raise NotImplementedError(f"Unknown variation type: {variation_type}")

    def _tif_to_array_with_checks(
        self, tif_path: UPath | str, data_source: str
    ) -> np.ndarray:
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
            print(f"Replacing tif {tif_path} due to {e}")
            raise e

    def _get_tif_path(
        self,
        data_source: str,
        example_id: str,
        frequency_type: Literal["monthly", "freq"],
    ) -> UPath:
        return self.root_dir / f"{data_source}_{frequency_type}" / f"{example_id}.tif"

    def _get_metadata_for_sample(
        self, data_source: str, example_id: str, frequency_type: str
    ) -> dict:
        """Get the metadata for a sample."""
        metadata_df = self.freq_metadata_df_dict[data_source]
        meta_dict_records = metadata_df[
            metadata_df["example_id"] == example_id
        ].to_dict(orient="records")
        # TURN INto single dict without example_id
        meta_dict = {}
        for record in meta_dict_records:
            image_idx = record.pop("image_idx")
            record.pop("example_id")
            meta_dict[image_idx] = record
        print(
            f"Metadata for {example_id} from {data_source} {frequency_type}: {meta_dict}"
        )
        return meta_dict

    def __getitem__(self, index: int) -> DatasetOutput:
        """Get the item at the given index."""
        example_id, data_frequency_type = self.get_example_from_index(index)
        sample_metadata = self.example_id_to_index_metadata_dict[example_id]
        data_source_output_dict = {}
        for data_source in self.data_sources:
            tif_path = self._get_tif_path(data_source, example_id, data_frequency_type)
            data_source_array = self._tif_to_array_with_checks(tif_path, data_source)
            # Probably there is a better way to have direct access to the metadata but will leave for now
            metadata_dict = self._get_metadata_for_sample(
                data_source, example_id, data_frequency_type
            )
            data_source_output_dict[data_source] = ArrayWithMetadata(
                array=data_source_array,
                metadata=metadata_dict,
            )
        sample_metadata["frequency_type"] = data_frequency_type
        return DatasetOutput(**data_source_output_dict, sample_metadata=sample_metadata)


if __name__ == "__main__":
    # TODO: Make this work for remote files likely want to use rslearn utils
    data_index_path = "gs://ai2-helios/data/20250113-sample-dataset-helios/index.csv"
    dataset = HeliosDataset(data_index_path)
    print(f"Dataset length: {len(dataset)}")
    import time

    time_to_load_sample = []
    for i in np.random.randint(0, len(dataset), size=10):
        start_time = time.time()
        dataset[i]
        end_time = time.time()
        time_taken = end_time - start_time
        print(f"Time taken: {time_taken} seconds")
        time_to_load_sample.append(time_taken)
    print(
        f"Time taken: {np.mean(time_to_load_sample)} seconds and {np.std(time_to_load_sample)} seconds"
    )
