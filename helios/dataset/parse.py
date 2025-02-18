"""Parse the Helios dataset."""

import csv
import logging
from dataclasses import dataclass
from datetime import datetime

from upath import UPath

from helios.data.constants import (
    BASE_RESOLUTION,
    BandSet,
    Modality,
    ModalitySpec,
    TimeSpan,
)

from .util import WindowMetadata, get_modality_fname

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ModalityImage:
    """Information about one image contained within a modality tile.

    The tile contains a stacked image time series. So this is the start and end time of
    each image in the series.
    """

    start_time: datetime
    end_time: datetime

    # Add this to see if there are two ModalityImage objects that are the same
    def __eq__(self, other: object) -> bool:
        """Check if two ModalityImage objects are the same."""
        if not isinstance(other, ModalityImage):
            return False
        return self.start_time == other.start_time and self.end_time == other.end_time


@dataclass(frozen=True)
class GridTile:
    """The position of a tile along a grid of a certain resolution."""

    # The CRS e.g. EPSG:32610.
    crs: str

    # The factor at which this tile is stored relative to BASE_RESOLUTION.
    resolution_factor: int

    # The column and row along the grid defined based on the resolution factor.
    col: int
    row: int


@dataclass
class ModalityTile:
    """Information about one tile pertaining to a modality."""

    grid_tile: GridTile
    images: list[ModalityImage]

    # The center time that defines the time ranges for this tile.
    center_time: datetime

    # The band sets along with the file containing them.
    band_sets: dict[BandSet, UPath]

    def get_flat_bands(self) -> list[str]:
        """Get the names of the bands as a flat list.

        This would correspond to the order of the bands in any function that combines
        the band sets into a single tensor.
        """
        bands: list[str] = []
        for band_set in self.band_sets:
            bands.extend(band_set.bands)
        return bands


def parse_modality_csv(
    helios_path: UPath, modality: ModalitySpec, time_span: TimeSpan, csv_path: UPath
) -> list[ModalityTile]:
    """Parse CSV for one modality and time span.

    Args:
        helios_path: the Helios dataset path.
        modality: the modality to parse.
        time_span: the time span to parse.
        csv_path: the CSV path.

    Returns:
        list of ModalityTiles.
    """
    # First get the tiles, and images in each tile.
    # We fill in the band sets and image paths next.
    modality_tiles: dict[GridTile, ModalityTile] = {}
    with csv_path.open() as f:
        reader = csv.DictReader(f)
        for csv_row in reader:
            grid_tile = GridTile(
                crs=csv_row["crs"],
                resolution_factor=modality.tile_resolution_factor,
                col=int(csv_row["col"]),
                row=int(csv_row["row"]),
            )
            image = ModalityImage(
                start_time=datetime.fromisoformat(csv_row["start_time"]),
                end_time=datetime.fromisoformat(csv_row["end_time"]),
            )
            image_idx = int(csv_row["image_idx"])
            if grid_tile not in modality_tiles:
                modality_tiles[grid_tile] = ModalityTile(
                    grid_tile=grid_tile,
                    images=[],
                    center_time=datetime.fromisoformat(csv_row["tile_time"]),
                    band_sets={},
                )

            # This image should appear at the index above. But the indexes should be in
            # order in the CSV.
            if image_idx != len(modality_tiles[grid_tile].images):
                # This should be an error but currently I realized there are one or two
                # tiles that actually have two timestamps in the original rslearn
                # dataset, which means the Helios dataset has two sets of entries in
                # the CSV but there is really only one file.
                # raise ValueError(
                #    "expected image index to be in increasing order and contiguous"
                # )
                continue
            modality_tiles[grid_tile].images.append(image)

    # Now we can fill in the band sets.
    # We also double check that there are no None in the image lists.
    for tile in modality_tiles.values():
        grid_tile = tile.grid_tile
        window_metadata = WindowMetadata(
            crs=grid_tile.crs,
            resolution=BASE_RESOLUTION * grid_tile.resolution_factor,
            col=grid_tile.col,
            row=grid_tile.row,
            time=tile.center_time,
        )
        for band_set in modality.band_sets:
            fname = get_modality_fname(
                helios_path,
                modality,
                time_span,
                window_metadata,
                band_set.get_resolution(),
                "tif",
            )
            tile.band_sets[band_set] = fname

    return list(modality_tiles.values())


def parse_helios_dataset(
    helios_path: UPath, supported_modalities: list[ModalitySpec] = Modality.values()
) -> dict[ModalitySpec, dict[TimeSpan, list[ModalityTile]]]:
    """Parse the various per-modality tiles present in a Helios dataset.

    Returns:
        a mapping from modality -> time span (e.g. yearly / two-week) -> list of tiles.
    """
    tiles: dict[ModalitySpec, dict[TimeSpan, list[ModalityTile]]] = {}

    for modality in Modality.values():
        if modality.ignore_when_parsing:
            continue
        if modality not in supported_modalities:
            logger.warning(
                f"ignoring modality {modality.name} not in supported_modalities"
            )
            continue
        # TODO: there's N/A in the image_idx column for openstreetmap
        if modality.name == "openstreetmap":
            continue

        if modality.is_multitemporal:
            # We need to load the one-year and two-week data separately.
            time_spans = [TimeSpan.YEAR, TimeSpan.TWO_WEEK]

        else:
            # Just need to load the static data.
            time_spans = [TimeSpan.STATIC]

        # For each possible time span available for this modality, parse the associated
        # CSV to get the ModalityTiles under that time span.
        tiles[modality] = {}
        for time_span in time_spans:
            # Reconstruct the CSV filename from the grid resolution, modality, and time span.
            tile_resolution = modality.get_tile_resolution()
            csv_fname = (
                helios_path
                / f"{tile_resolution}_{modality.name}{time_span.get_suffix()}.csv"  # type: ignore
            )
            logger.info(f"Parsing {modality.name} {time_span} {csv_fname}")
            tiles[modality][time_span] = parse_modality_csv(  # type: ignore
                helios_path,
                modality,
                time_span,  # type: ignore
                csv_fname,
            )

    return tiles
