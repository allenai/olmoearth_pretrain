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

PROJECTION_CRS = "EPSG:4326"

MISSING_VALUE = -99999


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
    bands: list[str]

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

    def get_expected_image_size(self, modality_resolution_factor: int) -> int:
        """Get the expected size of images containing these bands.

        Args:
            modality_resolution_factor: the resolution factor of the modality.

        Returns:
            the expected image size.
        """
        return IMAGE_TILE_SIZE // (self.resolution_factor // modality_resolution_factor)


class TimeSpan(str, Enum):
    """Enum to distinguish data that is valid for different time ranges."""

    # Only one data point (not time series).
    STATIC = "static"

    # Monthly over one year.
    YEAR = "year"

    # Every data point in a two-week period.
    TWO_WEEK = "two_week"

    def get_suffix(self) -> str:
        """Returns the suffix used for this timespan in raw Helios dataset."""
        if self == TimeSpan.STATIC:
            return ""
        if self == TimeSpan.YEAR:
            return "_monthly"
        if self == TimeSpan.TWO_WEEK:
            return "_freq"
        raise ValueError("invalid TimeSpan")


@dataclass(frozen=True)
class ModalitySpec:
    """Modality specification.

    Args:
        name: the name of the modality.
        tile_resolution_factor: the factor of how much more ground area is covered by the tile compared with a tile
                        of IMAGE_TILE_SIZE x IMAGE_TILE_SIZE pixels at the base resolution.
        band_sets: the band sets of the modality, ie the units of tokenization.
        is_multitemporal: whether the modality is multitemporal.
        ignore_when_parsing: whether to ignore the modality when parsing the data form the csv file.
    """

    name: str
    tile_resolution_factor: int
    band_sets: list[BandSet]
    is_multitemporal: bool
    ignore_when_parsing: bool  # If true this modality is not parsed from the csv file and not loaded form a file

    def __hash__(self) -> int:
        """Hash this Modality."""
        return hash(self.name)

    def get_tile_resolution(self) -> float:
        """Compute the tile resolution."""
        return get_resolution(self.tile_resolution_factor)

    def bandsets_as_indices(self) -> list[list[int]]:
        """Return band sets as indices."""
        indices = []
        offset = 0
        for band_set in self.band_sets:
            num_bands = len(band_set.bands)
            indices.append(list(range(offset, offset + num_bands)))
            offset += num_bands
        return indices

    @property
    def band_order(self) -> list[str]:
        """Get band order."""
        return sum((list(band_set.bands) for band_set in self.band_sets), [])

    @property
    def num_band_sets(self) -> int:
        """Get the number of band sets."""
        return len(self.band_sets)

    @property
    def num_bands(self) -> int:
        """Get the number of channels.

        The number of channels is the sum of the number of bands in all the band sets.
        """
        return sum(len(band_set.bands) for band_set in self.band_sets)

    @property
    def is_spatial(self) -> bool:
        """Does the modality have spatial data."""
        return self.get_tile_resolution() > 0

    @property
    def is_spacetime_varying(self) -> bool:
        """Does the modality vary in space and time."""
        return self.is_spatial and self.is_multitemporal

    @property
    def is_space_only_varying(self) -> bool:
        """Does the modality vary in space and not time."""
        return self.is_spatial and not self.is_multitemporal

    @property
    def is_time_only_varying(self) -> bool:
        """Does the modality vary in time and not space."""
        return not self.is_spatial and self.is_multitemporal

    @property
    def is_static_in_space_and_time(self) -> bool:
        """Does the modality vary in neither space or space."""
        return not self.is_spatial and not self.is_multitemporal


class Modality:
    """Enum-like access to ModalitySpecs."""

    NAIP = ModalitySpec(
        name="naip",
        tile_resolution_factor=1,
        band_sets=[BandSet(["R", "G", "B", "IR"], 1)],
        is_multitemporal=False,
        ignore_when_parsing=False,
    )

    # NAIP_10 is the NAIP data that covers the same extent as a IMAGE_TILE_SIZE x IMAGE_TILE_SIZE tile
    # at 10 m/pixel resolution but is still stored at NAIP resolution.
    NAIP_10 = ModalitySpec(
        name="naip_10",
        tile_resolution_factor=16,
        band_sets=[BandSet(["R", "G", "B", "IR"], 1)],
        is_multitemporal=False,
        ignore_when_parsing=False,
    )

    SENTINEL1 = ModalitySpec(
        name="sentinel1",
        tile_resolution_factor=16,
        band_sets=[BandSet(["vv", "vh"], 16)],
        is_multitemporal=True,
        ignore_when_parsing=False,
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
        ignore_when_parsing=False,
    )

    SENTINEL2_L2A = ModalitySpec(
        name="sentinel2_l2a",
        tile_resolution_factor=16,
        band_sets=[
            # 10 m/pixel bands.
            BandSet(["B02", "B03", "B04", "B08"], 16),
            # 20 m/pixel bands.
            BandSet(["B05", "B06", "B07", "B8A", "B11", "B12"], 32),
            # 60 m/pixel bands that we store at 40 m/pixel.
            BandSet(["B01", "B09"], 64),
        ],
        is_multitemporal=True,
        ignore_when_parsing=False,
    )

    LANDSAT = ModalitySpec(
        name="landsat",
        tile_resolution_factor=16,
        band_sets=[
            # 15 m/pixel bands that we store at 10 m/pixel.
            BandSet(["B8"], 16),
            # 30 m/pixel bands that we store at 20 m/pixel.
            BandSet(["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B9", "B10", "B11"], 32),
        ],
        is_multitemporal=True,
        ignore_when_parsing=False,
    )

    WORLDCOVER = ModalitySpec(
        name="worldcover",
        tile_resolution_factor=16,
        band_sets=[BandSet(["B1"], 16)],
        is_multitemporal=False,
        ignore_when_parsing=False,
    )

    SRTM = ModalitySpec(
        name="srtm",
        tile_resolution_factor=16,
        band_sets=[BandSet(["srtm"], 16)],
        is_multitemporal=False,
        ignore_when_parsing=False,
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
                1,
            )
        ],
        is_multitemporal=False,
        ignore_when_parsing=True,
    )

    OPENSTREETMAP_RASTER = ModalitySpec(
        name="openstreetmap_raster",
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
        ignore_when_parsing=False,
    )

    ERA5 = ModalitySpec(
        name="era5",
        tile_resolution_factor=256,
        band_sets=[
            BandSet(
                [
                    "snow-cover",
                    "snow-depth",
                    "soil-temperature-level-1",
                    "10m-u-component-of-wind",
                    "10m-v-component-of-wind",
                    "2m-temperature",
                    "skin-temperature",
                    "total-precipitation",
                ],
                256,
            ),
        ],
        is_multitemporal=True,
        ignore_when_parsing=True,
    )

    LATLON = ModalitySpec(
        name="latlon",
        tile_resolution_factor=0,
        band_sets=[BandSet(["lat", "lon"], 0)],
        is_multitemporal=False,
        ignore_when_parsing=True,
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

    @classmethod
    def names(self) -> list[str]:
        """Get all of the modality names."""
        return [modality.name for modality in self.values()]


# Latlon and timestamps
LATLON = ["lat", "lon"]
TIMESTAMPS = ["day", "month", "year"]
