"""Data schemas for helios training data index and metadata files."""

from typing import Any

import pandas as pd
import pandera as pa
from pandera import DataFrameModel
from pandera.typing import Series


class BaseDataModel(DataFrameModel):
    """Base schema for accompanying data files."""

    class Config:
        """Config for BaseDataModel."""

        coerce = True  # ensure that columns are coerced to the correct type
        strict = False  # allow extra columns


# Should we have a different metadata model for stuff with no time axis?


class DataSourceMetadataModel(BaseDataModel):
    """Data source metadata schema."""

    example_id: Series[str] = pa.Field(
        description="Unique identifier for the example, name of the file",
        nullable=False,
    )

    image_idx: Series[pd.Int64Dtype] = pa.Field(
        description="Index of the image on the time axis for the geotiff",
        nullable=True,
    )
    # TODO: Add check that if the column has a null it is all null

    start_time: Series[str] = pa.Field(
        description="Start time in ISO format",
        nullable=False,
        regex=r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\+\d{2}:\d{2}",
    )

    end_time: Series[str] = pa.Field(
        description="End time in ISO format",
        nullable=False,
        regex=r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\+\d{2}:\d{2}",
    )


def is_data_source_field(is_data_source: bool, *args: Any, **kwargs: Any) -> pa.Field:
    """Helper function to create a field with is_data_source set to the given value.

    Args:
        is_data_source: Whether the field is a data source
        *args: Positional arguments to pass to the Field constructor
        **kwargs: Keyword arguments to pass to the Field constructor

    Returns:
        A Field object with the is_data_source metadata set to the given value
    """
    metadata = kwargs.get("metadata", {})
    metadata["is_data_source"] = is_data_source
    kwargs["metadata"] = metadata
    return pa.Field(*args, **kwargs)


class TrainingDataIndexModel(BaseDataModel):
    """Schema for training data index files.

    This file contains metadata about the training data, including the example_id, projection,
    resolution, start_column, start_row, and time.

    The fields include metadata indicating whether they are data sources or not.
    They also include metadata indicating the frequency type of the data source.
    """

    example_id: Series[str] = is_data_source_field(
        is_data_source=False,
        description="Unique identifier for the example",
        nullable=False,
        unique=True,
    )

    projection: Series[str] = is_data_source_field(
        is_data_source=False,
        description="EPSG projection code",
        nullable=False,
    )

    resolution: Series[int] = is_data_source_field(
        is_data_source=False,
        description="Resolution in meters",
        nullable=False,
        isin=[1, 10, 250],  # Based on the values in the CSV
    )
    start_column: Series[int] = is_data_source_field(
        is_data_source=False,
        description="Starting column UTM coordinate",
        nullable=False,
    )

    start_row: Series[int] = is_data_source_field(
        is_data_source=False,
        description="Starting row UTM coordinate",
        nullable=False,
    )

    time: Series[str] = is_data_source_field(
        is_data_source=False,
        description="Timestamp of the following: \
            (a) for two-week data, the two week period starts at that time; \
            (b) for one-year data, the one year period is roughly centered at that time.",
        nullable=False,
        regex=r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\+\d{2}:\d{2}",  # ISO timestamp format
    )

    sentinel2_freq: Series[str] = is_data_source_field(
        is_data_source=True,
        description="Whether the example_id is available in the Sentinel-2 Frequency dataset",
        nullable=False,
        isin=["y", "n"],  # TODO: potentially might want this to be a boolean
        metadata={"frequency_type": "freq"},
    )

    sentinel2_monthly: Series[str] = is_data_source_field(
        is_data_source=True,
        description="Whether the example_id is available in the Sentinel-2 Monthly dataset",
        nullable=False,
        isin=["y", "n"],
        metadata={"frequency_type": "monthly"},
    )

    naip: Series[str] = is_data_source_field(
        is_data_source=True,
        description="Whether the example_id is available in the NAIP dataset",
        nullable=False,
        isin=["y", "n"],
    )

    worldcover: Series[str] = is_data_source_field(
        is_data_source=True,
        description="Whether the example_id is available in the WorldCover dataset",
        nullable=False,
        isin=["y", "n"],
    )

    openstreetmap: Series[str] = is_data_source_field(
        is_data_source=True,
        description="Whether the example_id is available in the OpenStreetMap dataset",
        nullable=False,
        isin=["y", "n"],
    )
