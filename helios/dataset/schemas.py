"""Data schemas for helios training data index and metadata files."""

import pandera as pa
from pandera import DataFrameModel
from pandera.typing import Series


class BaseDataModel(DataFrameModel):
    """Base schema for accompanying data files."""

    class Config:
        """Config for BaseDataModel."""

        coerce = True  # ensure that columns are coerced to the correct type
        strict = False  # allow extra columns


class GeoTiffTimeSeriesMetadataModel(BaseDataModel):
    """Base schema for time series metadata files."""

    example_id: Series[str] = pa.Field(
        description="Unique identifier for the example, name of the file",
        nullable=False,
    )

    image_idx: Series[int] = pa.Field(
        description="Index of the image on the time axis for the geotiff",
        nullable=False,
    )

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


class Sentinel2FrequencyMetadataDataModel(GeoTiffTimeSeriesMetadataModel):
    """Schema for Sentinel-2 Frequency metadata files."""

    # Inherits all fields from TimeSeriesMetadataModel
    pass


class Sentinel2MonthlyMetadataDataModel(GeoTiffTimeSeriesMetadataModel):
    """Schema for Sentinel-2 Monthly metadata files."""

    # Inherits all fields from TimeSeriesMetadataModel
    # Could override image_idx description if needed:
    image_idx: Series[int] = pa.Field(
        description="Index of the monthly mosaic on the time axis for the geotiff",
        nullable=False,
    )


class TrainingDataIndexDataModel(BaseDataModel):
    """Schema for training data index files.

    This file contains metadata about the training data, including the example_id, projection,
    resolution, start_column, start_row, and time.
    """

    example_id: Series[str] = pa.Field(
        description="Unique identifier for the example",
        nullable=False,
        unique=True,
    )

    projection: Series[str] = pa.Field(
        description="EPSG projection code",
        nullable=False,
    )

    resolution: Series[int] = pa.Field(
        description="Resolution in meters",
        nullable=False,
        isin=[1, 10, 250],  # Based on the values in the CSV
    )

    start_column: Series[int] = pa.Field(
        description="Starting column UTM coordinate",
        nullable=False,
    )

    start_row: Series[int] = pa.Field(
        description="Starting row UTM coordinate",
        nullable=False,
    )

    time: Series[str] = pa.Field(
        description="Timestamp of the following: \
            (a) for two-week data, the two week period starts at that time; \
            (b) for one-year data, the one year period is roughly centered at that time.",
        nullable=False,
        regex=r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\+\d{2}:\d{2}",  # ISO timestamp format
    )

    sentinel2_freq: Series[str] = pa.Field(
        description="Whether the example_id is available in the Sentinel-2 Frequency dataset",
        nullable=False,
        isin=["y", "n"],  # TODO: potentially might want this to be a boolean
    )

    sentinel2_monthly: Series[str] = pa.Field(
        description="Whether the example_id is available in the Sentinel-2 Monthly dataset",
        nullable=False,
        isin=["y", "n"],
    )
