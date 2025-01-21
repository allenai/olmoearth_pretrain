"""Constants for the data module."""

from typing import Literal

ALL_DATA_SOURCES = ["sentinel2"]

DATA_FREQUENCY_TYPES = ["freq", "monthly"]

DATA_SOURCE_VARIATION_TYPES = Literal[
    "space_time_varying", "time_varying_only", "space_varying_only", "static_only"
]

# THe data can have values that change across different dimennsions each source always varies in one of these ways
DATA_SOURCE_TO_VARIATION_TYPE = {
    "sentinel2": "space_time_varying",
}

# WARNING: TEMPORARY BANDS: We forgot to pull B9, B10 from the export for the initial sample dataset
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
