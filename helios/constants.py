"""Constants shared across the helios package."""

from typing import Literal

# TODO: data related constants should be moved to the data folder.
NAIP_BANDS = ["R", "G", "B", "IR"]

S2_BANDS = [
    "B01",
    "B02",
    "B03",
    "B04",
    "B05",
    "B06",
    "B07",
    "B08",
    "B8A",
    "B09",
    "B10",
    "B11",
    "B12",
]

LATLON_BANDS = ["lat", "lon"]
TIMESTAMPS = ["day", "month", "year"]

WORLDCOVER_BANDS = ["B1"]

VARIATION_TYPES = Literal[
    "space_time_varying", "time_varying", "space_varying", "static"
]

DATA_SOURCE_TO_VARIATION_TYPE = {
    "sentinel2": "space_time_varying",
    "worldcover": "space_varying",
    "naip": "space_varying",
    "openstreetmap": "space_varying",  # How do we want to use open Streetmap
}
