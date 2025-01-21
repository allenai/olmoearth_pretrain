"""Unit tests for parsing the dataset index."""

import pytest

from helios.data.index import DatasetIndexParser


@pytest.fixture
def sample_index_path() -> str:
    """Fixture providing path to test dataset index."""
    return "tests/fixtures/sample-dataset/index.csv"


def test_dataset_index(sample_index_path: str) -> None:
    """Test the dataset index."""
    index_parser = DatasetIndexParser(sample_index_path)
    assert len(index_parser.samples) == 4
    assert set(index_parser.samples[0].keys()) == {
        "data_source_paths",
        "data_source_metadata",
        "sample_metadata",
    }
    expected_paths = {
        "sentinel2": "tests/fixtures/sample-dataset/sentinel2_monthly/example_001.tif"
    }
    actual_paths = index_parser.samples[0]["data_source_paths"]
    assert len(expected_paths) == len(actual_paths)
    for data_source, expected_path in expected_paths.items():
        assert data_source in actual_paths
        assert str(actual_paths[data_source]) == expected_path
