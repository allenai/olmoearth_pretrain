"""Compatibility imports for shared OlmoEarth modality definitions.

New code should import these from :mod:`olmoearth_pretrain.modalities`. This
module remains for older callers that still use the historical data package path.
"""

from olmoearth_pretrain.modalities import (
    BASE_GSD,
    BASE_RESOLUTION,
    IMAGE_TILE_SIZE,
    LATLON,
    MAX_SEQUENCE_LENGTH,
    MISSING_VALUE,
    PROJECTION_CRS,
    SENTINEL1_NODATA,
    TIMESTAMPS,
    YEAR_NUM_TIMESTEPS,
    BandSet,
    Modality,
    ModalitySpec,
    TimeSpan,
    get_modality_specs_from_names,
    get_resolution,
)

__all__ = [
    "BASE_GSD",
    "BASE_RESOLUTION",
    "BandSet",
    "IMAGE_TILE_SIZE",
    "LATLON",
    "MAX_SEQUENCE_LENGTH",
    "MISSING_VALUE",
    "Modality",
    "ModalitySpec",
    "PROJECTION_CRS",
    "SENTINEL1_NODATA",
    "TIMESTAMPS",
    "TimeSpan",
    "YEAR_NUM_TIMESTEPS",
    "get_modality_specs_from_names",
    "get_resolution",
]
