"""Module for navigating Dataset Index.

TODO: Add the assumed dataset organizing format and rules
"""

from typing import NamedTuple

import numpy as np
from upath import UPath

from helios.data.constants import ALL_DATA_SOURCES
from helios.dataset.utils import (
    DataSourceMetadataRegistry,
    FrequencyType,
    load_data_index,
)


class SampleInformation(NamedTuple):
    """Information about a sample.

    Attributes:
        data_source_metadata: Metadata for each of the data sources.
        data_source_paths: Paths to the data sources.
        sample_metadata: Metadata for the sample.
    """

    data_source_metadata: dict[str, dict]
    data_source_paths: dict[str, dict]
    sample_metadata: dict[str, dict]


class DatasetIndexParser:
    """Parses the dataset index and provides paths to individual samples along with sample_me."""

    def __init__(self, data_index_path: UPath | str):
        """Initialize the dataset index parser."""
        self.data_sources = ALL_DATA_SOURCES
        self.data_index_path = UPath(data_index_path)
        self.root_dir = self.data_index_path.parent
        self.data_index_df = load_data_index(data_index_path)
        self.example_id_to_sample_metadata_dict = self.data_index_df.set_index(
            "example_id"
        ).to_dict("index")

        self.freq_metadata_df_dict = {}
        self.monthly_metadata_df_dict = {}
        for data_source in self.data_sources:
            metadata_path = self.get_path_to_data_source_metadata(data_source, "freq")
            self.freq_metadata_df_dict[data_source] = (
                DataSourceMetadataRegistry.load_and_validate(
                    data_source,
                    "freq",
                    sentinel2_frequency_metadata_path=metadata_path
                    if data_source == "sentinel2"
                    else None,
                )
            )

            metadata_path = self.get_path_to_data_source_metadata(
                data_source, "monthly"
            )
            self.monthly_metadata_df_dict[data_source] = (
                DataSourceMetadataRegistry.load_and_validate(
                    data_source,
                    "monthly",
                    sentinel2_monthly_metadata_path=metadata_path
                    if data_source == "sentinel2"
                    else None,
                )
            )

        # Intersect available data sources with index column names

        assert (
            len(self.data_sources) > 0
        ), "No data sources found in index, check naming of columns"

        self.monthly_example_ids = self.get_example_ids_by_frequency_type("monthly")
        self.freq_example_ids = self.get_example_ids_by_frequency_type("freq")

        # SO {paths: {data_source: path}, data_source_metadata: {data_source: metadata}, sample_metadata: {sample_metadata}}
        samples = self.get_sample_information_from_example_id_list(
            self.monthly_example_ids, "monthly"
        ) + self.get_sample_information_from_example_id_list(
            self.freq_example_ids, "freq"
        )

        self._samples = samples

        self.root_dir = self.data_index_path.parent

    def get_sample_information_from_example_id(
        self, example_id: str, freq_type: FrequencyType
    ) -> SampleInformation:
        """Get the sample information from an example ID.

        Args:
            example_id: The example ID to get information for.
            freq_type: The frequency type to get information for.

        Returns:
            SampleInformation: Information about the sample.
        """
        data_source_metadata = {}
        data_source_paths = {}
        sample_metadata = self.example_id_to_sample_metadata_dict[example_id]
        sample_metadata["example_id"] = example_id
        example_row = self.data_index_df[
            self.data_index_df["example_id"] == example_id
        ].iloc[0]
        for data_source in self.data_sources:
            if example_row[f"{data_source}_{freq_type}"] != "y":
                continue
            data_source_metadata[data_source] = (
                self.get_metadata_for_data_source_in_sample(
                    data_source, example_id, freq_type
                )
            )
            data_source_paths[data_source] = self.get_tif_path(
                data_source, example_id, freq_type
            )
        return SampleInformation(
            data_source_metadata=data_source_metadata,
            data_source_paths=data_source_paths,
            sample_metadata=sample_metadata,
        )

    def get_sample_information_from_example_id_list(
        self, example_ids: list[str], freq_type: FrequencyType
    ) -> list[SampleInformation]:
        """Get the sample information from a list of example IDs."""
        return [
            self.get_sample_information_from_example_id(example_id, freq_type)
            for example_id in example_ids
        ]

    def get_example_ids_by_frequency_type(
        self, frequency_type: FrequencyType
    ) -> np.ndarray:
        """Get the example IDs by frequency type."""
        frequency_type_mask = (
            self.data_index_df[
                [col for col in self.data_index_df.columns if frequency_type in col]
            ]
            .eq("y")
            .any(axis=1)
        )
        example_ids = self.data_index_df.loc[
            frequency_type_mask, "example_id"
        ].to_numpy(dtype=str)
        return example_ids

    def get_path_to_data_source_metadata(
        self, data_source: str, frequency_type: str
    ) -> UPath:
        """Get the path to the data source metadata."""
        return self.root_dir / f"{data_source}_{frequency_type}.csv"

    def get_tif_path(
        self,
        data_source: str,
        example_id: str,
        frequency_type: FrequencyType,
    ) -> UPath:
        """Get the path to the tif file."""
        return self.root_dir / f"{data_source}_{frequency_type}" / f"{example_id}.tif"

    def get_metadata_for_data_source_in_sample(
        self, data_source: str, example_id: str, frequency_type: FrequencyType
    ) -> dict:
        """Get the metadata for a sample."""
        if frequency_type == "monthly":
            metadata_df = self.monthly_metadata_df_dict[data_source]
        else:
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
        return meta_dict

    def __len__(self) -> int:
        """Get the number of unique samples {co-located multimodal data}."""
        return len(self.samples)

    @property
    def samples(self) -> list[SampleInformation]:
        """Get the samples."""
        return self._samples
