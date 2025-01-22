"""Constants for the data module."""

from typing import Literal

DATA_SOURCE_VARIATION_TYPES = Literal[
    "space_time_varying", "time_varying_only", "space_varying_only", "static_only"
]

# THe data can have values that change across different dimennsions each source always varies in one of these ways
DATA_SOURCE_TO_VARIATION_TYPE = {
    "sentinel2": "space_time_varying",
}
