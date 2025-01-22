"""Module for navigating Dataset Index.

TODO: Add the assumed dataset organizing format and rules
"""

from typing import NamedTuple

import numpy as np
from pandera.typing import DataFrame
from upath import UPath

from helios.dataset.schemas import TrainingDataIndexDataModel
from helios.dataset.utils import (
    FrequencyType,
    load_data_index,
    load_data_source_metadata,
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

    # TODO: THis should be stored in a single place that is shared with the dataset creation code

    EXTENSIONS = {
        "naip": "tif",
        "openstreetmap": "geojson",
        "sentinel2_freq": "tif",
        "sentinel2_monthly": "tif",
        "worldcover": "tif",
    }

    def __init__(self, data_index_path: UPath | str):
        """Initialize the dataset index parser."""
        self.data_index_path = UPath(data_index_path)
        self.root_dir = self.data_index_path.parent
        self.data_index_df: DataFrame[TrainingDataIndexDataModel] = load_data_index(
            data_index_path
        )
        self.example_id_to_sample_metadata_dict = self.data_index_df.set_index(
            "example_id"
        ).to_dict("index")
        self.data_source_and_freq_types = self._get_data_sources_and_freq_types()
        assert (
            len(self.data_source_and_freq_types) > 0
        ), "No data sources found in index, check naming of columns"
        (
            self.freq_metadata_df_dict,
            self.monthly_metadata_df_dict,
            self.static_metadata_df_dict,
        ) = self._load_all_data_source_metadata()

        # Intersect available data sources with index column names

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

    def _get_data_sources_and_freq_types(
        self,
    ) -> list[tuple[str, FrequencyType | None]]:
        """Get the data sources and frequency types."""
        data_source_columns_and_freq_types = []
        for (
            column_name,
            field,
        ) in TrainingDataIndexDataModel.to_schema().columns.items():
            metadata = getattr(field, "metadata", {})
            if not metadata:
                continue
            if metadata.get("is_data_source", False):
                continue
            # No Frequency Type means static data source
            freq_type = metadata.get("frequency_type", None)
            data_source_columns_and_freq_types.append((column_name, freq_type))
        return data_source_columns_and_freq_types

    def _load_all_data_source_metadata(
        self,
    ) -> tuple[dict[str, DataFrame], dict[str, DataFrame], dict[str, DataFrame]]:
        """Load all data source metadata."""
        static_metadata_df_dict = {}
        freq_metadata_df_dict = {}
        monthly_metadata_df_dict = {}
        for data_source, freq_type in self.data_source_and_freq_types:
            metadata_path = self.get_path_to_data_source_metadata(
                data_source, freq_type
            )
            data_source_metadata_df = load_data_source_metadata(metadata_path)
            if freq_type is None:
                static_metadata_df_dict[data_source] = data_source_metadata_df
            elif freq_type == "freq":
                freq_metadata_df_dict[data_source] = data_source_metadata_df
            elif freq_type == "monthly":
                monthly_metadata_df_dict[data_source] = data_source_metadata_df
            else:
                raise ValueError(f"Unknown frequency type: {freq_type}")
        return static_metadata_df_dict, freq_metadata_df_dict, monthly_metadata_df_dict

    def get_sample_information_from_example_id(
        self, example_id: str, freq_type: FrequencyType
    ) -> SampleInformation:
        """Get the sample information from an example ID.

        For a given frequency type, we will get all the data sources that are available for that frequency type
        and all static data sources associated with that example_id.

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
        for data_source, _ in self.data_source_and_freq_types:
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
        self, data_source: str, frequency_type: str | None
    ) -> UPath:
        """Get the path to the data source metadata."""
        if frequency_type is None:
            return self.root_dir / f"{data_source}.csv"
        else:
            return self.root_dir / f"{data_source}_{frequency_type}.csv"

    def get_tif_path(
        self,
        data_source: str,
        example_id: str,
        frequency_type: FrequencyType | None,
    ) -> UPath:
        """Get the path to the tif file."""
        extension = self.EXTENSIONS[data_source]
        if frequency_type is None:
            return self.root_dir / data_source / f"{example_id}.{extension}"
        else:
            return (
                self.root_dir
                / f"{data_source}_{frequency_type}"
                / f"{example_id}.{extension}"
            )

    def get_metadata_for_data_source_in_sample(
        self, data_source: str, example_id: str, frequency_type: FrequencyType | None
    ) -> dict:
        """Get the metadata for a sample."""
        if frequency_type == "monthly":
            metadata_df = self.monthly_metadata_df_dict[data_source]
        elif frequency_type == "freq":
            metadata_df = self.freq_metadata_df_dict[data_source]
        elif frequency_type is None:
            metadata_df = self.static_metadata_df_dict[data_source]
        else:
            raise ValueError(f"Unknown frequency type: {frequency_type}")
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
