"""Utility functions for the data module."""

from pathlib import Path
from typing import Any, Literal, TypeVar

import pandas as pd
import pandera as pa
from pandera.typing import DataFrame

from helios.dataset.schemas import DataSourceMetadataModel, TrainingDataIndexModel

T = TypeVar("T")
FrequencyType = Literal["monthly", "freq"]


def load_metadata(path: Path | str, schema: type[T], **kwargs: Any) -> DataFrame[T]:
    """Load metadata from a file with the specified schema.

    Args:
        path: Path to the file
        schema: Pandera schema class to validate against
        **kwargs: Additional arguments passed to pd.read_csv
    """
    # check the extension of the file say not implemented if not csv
    file_extension = (
        path.split(".")[-1] if isinstance(path, str) else path.name.split(".")[-1]
    )
    if file_extension not in ["csv", "parquet"]:
        raise NotImplementedError(f"File extension {file_extension} not supported")
    if file_extension == "csv":
        return pd.read_csv(path, **kwargs)
    elif file_extension == "parquet":
        return pd.read_parquet(path, **kwargs)
    else:
        raise NotImplementedError(f"File extension {file_extension} not supported")


@pa.check_types
def load_data_index(
    data_index_path: Path | str, **kwargs: Any
) -> DataFrame[TrainingDataIndexModel]:
    """Load the data index from a csv file."""
    return load_metadata(data_index_path, TrainingDataIndexModel, **kwargs)


@pa.check_types
def load_data_source_metadata(
    data_source_metadata_path: Path | str, **kwargs: Any
) -> DataFrame[DataSourceMetadataModel]:
    """Load the data source metadata from a csv file."""
    return load_metadata(data_source_metadata_path, DataSourceMetadataModel, **kwargs)
