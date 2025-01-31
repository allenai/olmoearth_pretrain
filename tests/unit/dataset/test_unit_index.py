"""Unit tests for parsing the dataset index."""

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from upath import UPath

from helios.dataset.index import DatasetIndexParser, SampleInformation


# Probably closer to integration tests
@pytest.fixture
def sample_index_path() -> str:
    """Fixture providing path to test dataset index."""
    return "tests/fixtures/sample-dataset/index.csv"


class TestDatasetIndex:
    """Unit tests for the DatasetIndex class."""

    def test_get_data_sources_and_freq_types(self) -> None:
        """Test the get_data_sources_and_freq_types method."""
        expected_data_sources_and_freq_types = set(
            [
                ("naip", None),
                ("openstreetmap", None),
                ("sentinel2", "freq"),
                ("worldcover", None),
                ("sentinel2", "monthly"),
            ]
        )
        data_source_and_freq_types = (
            DatasetIndexParser._get_data_sources_and_freq_types()
        )
        assert len(data_source_and_freq_types) == 5
        assert set(data_source_and_freq_types) == expected_data_sources_and_freq_types

    def test_load_all_data_source_metadata(self) -> None:
        """Test the _load_all_data_source_metadata method."""
        with patch.object(DatasetIndexParser, "__init__", lambda x, y: None):
            parser = DatasetIndexParser("dummy_path")

            # Manually set the data_source_and_freq_types attribute
            parser.data_source_and_freq_types = [
                ("naip", None),
                ("sentinel2", "freq"),
                ("sentinel2", "monthly"),
            ]

            # Mock the methods
            with (
                patch.object(
                    parser,
                    "get_path_to_data_source_metadata",
                    return_value="dummy_path",
                ) as mock_get_path,
                patch(
                    "helios.dataset.index.load_data_source_metadata",
                    return_value={"key": "value"},
                ) as mock_load_metadata,
            ):
                # Call the method
                metadata = parser._load_all_data_source_metadata()

                # Assertions
                assert mock_get_path.call_count == 3
                assert mock_load_metadata.call_count == 3

                assert metadata.static == {"naip": {"key": "value"}}
                assert metadata.freq == {"sentinel2": {"key": "value"}}
                assert metadata.monthly == {"sentinel2": {"key": "value"}}

    def test_get_sample_information_from_example_id(self) -> None:
        """Test the get_sample_information_from_example_id method."""
        with patch.object(DatasetIndexParser, "__init__", lambda x, y: None):
            parser = DatasetIndexParser("dummy_path")
            parser.root_dir = UPath("/dummy_root_dir")

            # Mock the necessary attributes and methods
            parser.example_id_to_sample_metadata_dict = {
                "example_001": {"some_metadata_key": "some_metadata_value"}
            }
            parser.data_index_df = pd.DataFrame(
                {
                    "example_id": ["example_001", "example_002", "example_003"],
                    "sentinel2_freq": ["y", "n", "y"],
                    "worldcover": ["y", "y", "n"],
                    "sentinel2_monthly": ["n", "y", "y"],
                }
            )
            parser.freq_metadata_df_dict = {
                "sentinel2": pd.DataFrame(
                    {
                        "example_id": ["example_001"],
                        "meta_key": ["meta_value"],
                        "image_idx": [0],
                    }
                )
            }
            parser.static_metadata_df_dict = {
                "worldcover": pd.DataFrame(
                    {
                        "example_id": ["example_001"],
                        "meta_key": ["meta_value"],
                        "image_idx": [0],
                    }
                )
            }

            parser.data_source_and_freq_types = [
                ("sentinel2", "freq"),
                ("worldcover", None),
                ("sentinel2", "monthly"),
            ]

            # Call the method
            sample_info = parser.get_sample_information_from_example_id(
                "example_001", "freq"
            )

            # Assertions
            assert isinstance(sample_info, SampleInformation)
            assert sample_info.sample_metadata["example_id"] == "example_001"
            assert sample_info.data_source_metadata == {
                "sentinel2": {0: {"meta_key": "meta_value"}},
                "worldcover": {0: {"meta_key": "meta_value"}},
            }
            assert sample_info.data_source_paths == {
                "sentinel2": UPath("/dummy_root_dir/sentinel2_freq/example_001.tif"),
                "worldcover": UPath("/dummy_root_dir/worldcover/example_001.tif"),
            }

    def test_get_example_ids_by_frequency_type(self) -> None:
        """Test the get_example_ids_by_frequency_type method."""
        with patch.object(DatasetIndexParser, "__init__", lambda x, y: None):
            parser = DatasetIndexParser("dummy_path")
            parser.data_index_df = pd.DataFrame(
                {
                    "example_id": ["example_001", "example_002"],
                    "sentinel2_freq": ["y", "n"],
                    "sentinel2_monthly": ["n", "y"],
                }
            )
            assert parser.get_example_ids_by_frequency_type("freq") == np.array(
                ["example_001"]
            )
            assert parser.get_example_ids_by_frequency_type("monthly") == np.array(
                ["example_002"]
            )
            with pytest.raises(ValueError):
                parser.get_example_ids_by_frequency_type("unknown")  # type: ignore

    def test_get_path_to_data_source_metadata(self) -> None:
        """Test the get_path_to_data_source_metadata method."""
        with patch.object(DatasetIndexParser, "__init__", lambda x, y: None):
            parser = DatasetIndexParser("dummy_path")
            parser.root_dir = UPath("/dummy_root_dir")
            assert parser.get_path_to_data_source_metadata(
                "sentinel2", "freq"
            ) == UPath("/dummy_root_dir/sentinel2_freq.csv")
            assert parser.get_path_to_data_source_metadata(
                "sentinel2", "monthly"
            ) == UPath("/dummy_root_dir/sentinel2_monthly.csv")
            assert parser.get_path_to_data_source_metadata("worldcover", None) == UPath(
                "/dummy_root_dir/worldcover.csv"
            )

    def test_get_tif_path(self) -> None:
        """Test the get_tif_path method."""
        with patch.object(DatasetIndexParser, "__init__", lambda x, y: None):
            parser = DatasetIndexParser("dummy_path")
            parser.root_dir = UPath("/dummy_root_dir")
            assert parser.get_tif_path("sentinel2", "example_001", "freq") == UPath(
                "/dummy_root_dir/sentinel2_freq/example_001.tif"
            )
            assert parser.get_tif_path("sentinel2", "example_001", "monthly") == UPath(
                "/dummy_root_dir/sentinel2_monthly/example_001.tif"
            )
            assert parser.get_tif_path("worldcover", "example_001", None) == UPath(
                "/dummy_root_dir/worldcover/example_001.tif"
            )

    def test_get_metadata_for_data_source_in_sample(self) -> None:
        """Test the get_metadata_for_data_source_in_sample method."""
        with patch.object(DatasetIndexParser, "__init__", lambda x, y: None):
            parser = DatasetIndexParser("dummy_path")
            parser.root_dir = UPath("/dummy_root_dir")
            parser.freq_metadata_df_dict = {
                "sentinel2": pd.DataFrame(
                    {
                        "example_id": ["example_001"],
                        "meta_key": ["meta_value"],
                        "image_idx": [0],
                    }
                )
            }
            assert parser.get_metadata_for_data_source_in_sample(
                "sentinel2", "example_001", "freq"
            ) == {0: {"meta_key": "meta_value"}}
