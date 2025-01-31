"""Integration tests for the dataset index."""

import pytest

from helios.dataset.index import DatasetIndexParser


@pytest.fixture
def sample_index_path() -> str:
    """Fixture providing path to test dataset index."""
    return "tests/fixtures/sample-dataset/index.csv"


def test_dataset_index(sample_index_path: str) -> None:
    """Test the dataset index."""
    index_parser = DatasetIndexParser(sample_index_path)
    assert len(index_parser.samples) == 4

    # Access the first sample's information
    first_sample_info = index_parser.samples[0]

    # Check the keys in the first sample's data_source_paths
    expected_data_sources = {"sentinel2", "naip", "worldcover", "openstreetmap"}
    assert set(first_sample_info.data_source_paths.keys()) == expected_data_sources

    # Define expected paths (split into multiple lines to fix line length)
    expected_paths = {
        "sentinel2": (
            "tests/fixtures/sample-dataset/sentinel2_monthly/" "example_001.tif"
        ),
        "naip": "tests/fixtures/sample-dataset/naip/example_001.tif",
        "worldcover": "tests/fixtures/sample-dataset/worldcover/example_001.tif",
        "openstreetmap": "tests/fixtures/sample-dataset/openstreetmap/example_001.geojson",
    }

    # Compare actual paths with expected paths
    actual_paths = first_sample_info.data_source_paths
    assert len(expected_paths) == len(actual_paths)
    for data_source, expected_path in expected_paths.items():
        assert data_source in actual_paths
        assert str(actual_paths[data_source]) == expected_path

    # Check the keys in the first sample's data_source_metadata
    assert set(first_sample_info.data_source_metadata.keys()) == expected_data_sources

    # Check the keys in the first sample's sample_metadata
    assert "example_id" in first_sample_info.sample_metadata
