"""Constants shared across the helios package.

Warning: this is only developed for raster data currently.
"""

from dataclasses import dataclass
from enum import Enum

# The highest resolution that we are working at.
# Everything else is a factor (which is a power of 2) coarser than this resolution.
BASE_RESOLUTION = 0.625

# The default image tile size.
# Some images may be smaller if they are stored at a coarser resolution compared to the
# resolution that the grid is based on.
IMAGE_TILE_SIZE = 256


@dataclass
class BandSet:
    """A group of bands that is stored at the same resolution.

    Many modalities only have one band set, but some have different bands at different
    resolutions.
    """

    # List of band names.
    bands: list[str]

    # Resolution is BASE_RESOLUTION // resolution_factor.
    resolution_factor: int


@dataclass
class ModalitySpec:
    """Specification of one modality."""

    name: str

    # Resolution by which the grid is defined for this modality.
    tile_resolution_factor: int

    # Band sets in this modality.
    band_sets: list[BandSet]


# Modalities supported by helios
class Modality(Enum):
    """Modality information."""

    NAIP = ModalitySpec(
        name="naip",
        tile_resolution_factor=1,
        band_sets=[BandSet(["R", "G", "B", "IR"], 1)],
    )

    S1 = ModalitySpec(
        name="sentinel1",
        tile_resolution_factor=16,
        band_sets=[BandSet(["VV", "VH"], 16)],
    )

    S2 = ModalitySpec(
        name="sentinel2",
        tile_resolution_factor=16,
        band_sets=[
            # 10 m/pixel bands.
            BandSet(["B02", "B03", "B04", "B08"], 16),
            # 20 m/pixel bands.
            BandSet(["B05", "B06", "B07", "B8A", "B11", "B12"], 32),
            # 60 m/pixel bands that we store at 40 m/pixel.
            BandSet(["B01", "B09", "B10"], 64),
        ],
    )

    LANDSAT = ModalitySpec(
        name="landsat",
        tile_resolution_factor=16,
        band_sets=[
            # 15 m/pixel bands that we store at 10 m/pixel.
            BandSet(["B8"], 16),
            # 30 m/pixel bands that we store at 20 m/pixel.
            BandSet(["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B9", "B10", "B11"], 16),
        ],
    )

    WORLDCOVER = ModalitySpec(
        name="worldcover",
        tile_resolution_factor=16,
        band_sets=[BandSet(["B1"], 16)],
    )

    OSM = ModalitySpec(
        name="openstreetmap",
        # OpenStreetMap is gridded at 10 m/pixel, but the data itself is rasterized at
        # 2.5 m/pixel (so each tile contains 1024x1024 instead of 256x256).
        tile_resolution_factor=16,
        band_sets=[
            BandSet(
                [
                    "aerialway_pylon",
                    "aerodrome",
                    "airstrip",
                    "amenity_fuel",
                    "building",
                    "chimney",
                    "communications_tower",
                    "crane",
                    "flagpole",
                    "fountain",
                    "generator_wind",
                    "helipad",
                    "highway",
                    "leisure",
                    "lighthouse",
                    "obelisk",
                    "observatory",
                    "parking",
                    "petroleum_well",
                    "power_plant",
                    "power_substation",
                    "power_tower",
                    "river",
                    "runway",
                    "satellite_dish",
                    "silo",
                    "storage_tank",
                    "taxiway",
                    "water_tower",
                    "works",
                ],
                4,
            )
        ],
    )
