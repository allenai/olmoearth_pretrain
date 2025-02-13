"""Constants shared across the helios package.

Warning: this is only developed for raster data currently.
"""

from collections.abc import Sequence
from dataclasses import dataclass

# The highest resolution that we are working at.
# Everything else is a factor (which is a power of 2) coarser than this resolution.
BASE_RESOLUTION = 0.625

# The default image tile size.
# Some images may be smaller if they are stored at a coarser resolution compared to the
# resolution that the grid is based on.
IMAGE_TILE_SIZE = 256


def get_resolution(resolution_factor: int) -> float | int:
    """Compute the resolution.

    If it is an integer, then we cast it to int so that it works with the raw Helios
    dataset, where some files are named based on the integer. We may want to change
    this in the future to avoid the extra code here.
    """
    resolution = BASE_RESOLUTION * resolution_factor
    if float(int(resolution)) == resolution:
        return int(resolution)
    return resolution


@dataclass(frozen=True)
class BandSet:
    """A group of bands that is stored at the same resolution.

    Many modalities only have one band set, but some have different bands at different
    resolutions.
    """

    # List of band names.
    bands: Sequence[str]

    # Resolution is BASE_RESOLUTION * resolution_factor.
    # If resolution == 0, this means the data
    # does not vary in space (e.g. latlons)
    resolution_factor: int

    def __hash__(self) -> int:
        """Hash this BandSet."""
        return hash((tuple(self.bands), self.resolution_factor))

    def get_resolution(self) -> float:
        """Compute the resolution."""
        return get_resolution(self.resolution_factor)


@dataclass(frozen=True)
class ModalitySpec:
    """Modality specification."""

    name: str
    tile_resolution_factor: int
    band_sets: Sequence[BandSet]
    is_multitemporal: bool

    def __hash__(self) -> int:
        """Hash this Modality."""
        return hash(self.name)

    def get_tile_resolution(self) -> float:
        """Compute the tile resolution."""
        return get_resolution(self.tile_resolution_factor)

    def bandsets_as_indices(self) -> list[list[int]]:
        """Return the band sets as indices."""
        # TODO: Add Integration test that we actually load the data in the correct order from the band sets
        band_specs_as_indices = []
        for band_set in self.band_sets:
            # TODO: I think the bands are not actually in the order of the old constant but stacked succesively from band sets
            band_specs_as_indices.append(list(range(len(band_set.bands))))
        return band_specs_as_indices

    @property
    def band_order(self) -> list[str]:
        """Get the band order."""
        return [b for band_set in self.band_sets for b in band_set.bands]

    @property
    def num_band_sets(self) -> int:
        """Get the number of band sets."""
        return len(self.band_sets)

    @property
    def num_channels(self) -> int:
        """Get the number of channels.

        The number of channels is the sum of the number of bands in all the band sets.
        """
        return sum(len(band_set.bands) for band_set in self.band_sets)


class Modality:
    """Enum-like access to ModalitySpecs."""

    NAIP = ModalitySpec(
        name="naip",
        tile_resolution_factor=1,
        band_sets=[BandSet(["R", "G", "B", "IR"], 1)],
        is_multitemporal=False,
    )

    SENTINEL1 = ModalitySpec(
        name="sentinel1",
        tile_resolution_factor=16,
        band_sets=[BandSet(["VV", "VH"], 16)],
        is_multitemporal=True,
    )

    SENTINEL2 = ModalitySpec(
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
        is_multitemporal=True,
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
        is_multitemporal=True,
    )

    WORLDCOVER = ModalitySpec(
        name="worldcover",
        tile_resolution_factor=16,
        band_sets=[BandSet(["B1"], 16)],
        is_multitemporal=False,
    )

    OPENSTREETMAP = ModalitySpec(
        name="openstreetmap",
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
        is_multitemporal=False,
    )

    # TODO: decide if we want to include latlon as a modality
    # The issue is that parse_modality_csv will search for the csv file and relevant ModalityTile
    LATLON = ModalitySpec(
        name="latlon",
        tile_resolution_factor=0,
        band_sets=[BandSet(["lat", "lon"], 0)],
        is_multitemporal=False,
    )

    @classmethod
    def get(self, name: str) -> ModalitySpec:
        """Get the ModalitySpec with the specified name."""
        modality = getattr(Modality, name.upper())
        assert modality.name == name
        return modality

    @classmethod
    def values(self) -> list[ModalitySpec]:
        """Get all of the ModalitySpecs."""
        modalities = []
        for k in dir(Modality):
            modality = getattr(Modality, k)
            if not isinstance(modality, ModalitySpec):
                continue
            modalities.append(modality)
        return modalities


# TODO: change this to other name to avoid confusion
SUPPORTED_MODALITIES = [
    Modality.SENTINEL1,
    Modality.SENTINEL2,
    Modality.WORLDCOVER,
]
LATLON = ["lat", "lon"]
TIMESTAMPS = ["day", "month", "year"]
