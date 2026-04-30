"""Parse the OlmoEarth Pretrain dataset."""

import csv
import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from upath import UPath

from olmoearth_pretrain.data.constants import (
    BASE_RESOLUTION,
    IMAGE_TILE_SIZE,
    BandSet,
    Modality,
    ModalitySpec,
    TimeSpan,
)

from .utils import WindowMetadata, get_modality_dir, get_modality_fname

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
    """Reference to a raw tile.

    Legacy datasets identify tiles by ``crs/resolution_factor/col/row``. Newer datasets
    can opt out of that and instead carry ``sample_id`` plus explicit geometry.
    """

    # The CRS e.g. EPSG:32610.
    crs: str

    # The factor at which this tile is stored relative to BASE_RESOLUTION.
    resolution_factor: int

    # The column and row along the grid defined based on the resolution factor.
    col: int | None = None
    row: int | None = None
    sample_id: str | None = None
    bounds: tuple[int, int, int, int] | None = None
    x_resolution: float | None = None
    y_resolution: float | None = None
    use_grid_reference: bool = True

    def get_x_resolution(self) -> float:
        """Get the x resolution for this tile."""
        if self.x_resolution is not None:
            return self.x_resolution
        return BASE_RESOLUTION * self.resolution_factor

    def get_y_resolution(self) -> float:
        """Get the y resolution for this tile."""
        if self.y_resolution is not None:
            return self.y_resolution
        return -self.get_x_resolution()

    def get_tile_size(self) -> int:
        """Get the nominal tile size in pixels."""
        if self.bounds is not None:
            return self.bounds[2] - self.bounds[0]
        return IMAGE_TILE_SIZE

    def get_bounds(self) -> tuple[int, int, int, int]:
        """Get pixel-space bounds for this tile."""
        if self.bounds is not None:
            return self.bounds
        if self.col is None or self.row is None:
            raise ValueError("tile is missing both bounds and grid reference")
        tile_size = self.get_tile_size()
        return (
            self.col * tile_size,
            self.row * tile_size,
            (self.col + 1) * tile_size,
            (self.row + 1) * tile_size,
        )

    def get_projected_extent(self) -> tuple[float, float, float, float]:
        """Get the tile extent in projected coordinates."""
        bounds = self.get_bounds()
        x_resolution = self.get_x_resolution()
        y_resolution = self.get_y_resolution()
        x0 = bounds[0] * x_resolution
        x1 = bounds[2] * x_resolution
        y0 = bounds[1] * y_resolution
        y1 = bounds[3] * y_resolution
        return (min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1))

    def contains(self, other: "GridTile", epsilon: float = 1e-6) -> bool:
        """Check if this tile fully contains another tile geometrically."""
        if self.crs != other.crs:
            return False
        sx0, sy0, sx1, sy1 = self.get_projected_extent()
        ox0, oy0, ox1, oy1 = other.get_projected_extent()
        return (
            sx0 - epsilon <= ox0
            and sy0 - epsilon <= oy0
            and sx1 + epsilon >= ox1
            and sy1 + epsilon >= oy1
        )

    def area(self) -> float:
        """Get the projected area of the tile."""
        x0, y0, x1, y1 = self.get_projected_extent()
        return (x1 - x0) * (y1 - y0)

    def get_crop_window(
        self, child: "GridTile", raster_width: int, raster_height: int
    ) -> tuple[int, int, int, int]:
        """Get the raster window containing ``child`` within this tile."""
        if not self.contains(child):
            raise ValueError("child tile must be contained within parent tile")

        px0, py0, px1, py1 = self.get_projected_extent()
        cx0, cy0, cx1, cy1 = child.get_projected_extent()
        parent_width = px1 - px0
        parent_height = py1 - py0
        if parent_width <= 0 or parent_height <= 0:
            raise ValueError("parent tile must have positive projected size")

        col_off = round((cx0 - px0) / parent_width * raster_width)
        row_off = round((py1 - cy1) / parent_height * raster_height)
        width = round((cx1 - cx0) / parent_width * raster_width)
        height = round((cy1 - cy0) / parent_height * raster_height)
        return (col_off, row_off, width, height)


def _parse_optional_int(value: str | None) -> int | None:
    if value is None or value == "":
        return None
    return int(value)


def _parse_optional_float(value: str | None) -> float | None:
    if value is None or value == "":
        return None
    return float(value)


def _parse_optional_str(value: str | None) -> str | None:
    if value is None or value == "":
        return None
    return value


def _parse_use_grid_reference(csv_row: dict[str, Any]) -> bool:
    raw_value = csv_row.get("use_grid_reference")
    if raw_value is None or raw_value == "":
        return True
    return str(raw_value).strip().lower() in {"1", "true", "yes", "y", "on"}


def _parse_bounds(csv_row: dict[str, Any]) -> tuple[int, int, int, int] | None:
    keys = ("bounds_left", "bounds_bottom", "bounds_right", "bounds_top")
    raw_values = [csv_row.get(key) for key in keys]
    if all(value in (None, "") for value in raw_values):
        return None
    if any(value in (None, "") for value in raw_values):
        raise ValueError("bounds must either be fully specified or fully omitted")
    return (
        int(str(raw_values[0])),
        int(str(raw_values[1])),
        int(str(raw_values[2])),
        int(str(raw_values[3])),
    )


@dataclass
class ModalityTile:
    """Information about one tile pertaining to a modality."""

    grid_tile: GridTile
    images: list[ModalityImage]

    # The center time that defines the time ranges for this tile.
    center_time: datetime

    # The band sets along with the file containing them.
    band_sets: dict[BandSet, UPath]

    modality: ModalitySpec

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
    path: UPath, modality: ModalitySpec, time_span: TimeSpan, csv_path: UPath
) -> list[ModalityTile]:
    """Parse CSV for one modality and time span.

    Args:
        path: the OlmoEarth Pretrain dataset path.
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
            use_grid_reference = _parse_use_grid_reference(csv_row)
            grid_tile = GridTile(
                crs=csv_row["crs"],
                resolution_factor=modality.tile_resolution_factor,
                col=_parse_optional_int(csv_row.get("col")),
                row=_parse_optional_int(csv_row.get("row")),
                sample_id=_parse_optional_str(csv_row.get("sample_id")),
                bounds=_parse_bounds(csv_row),
                x_resolution=_parse_optional_float(csv_row.get("x_resolution")),
                y_resolution=_parse_optional_float(csv_row.get("y_resolution")),
                use_grid_reference=use_grid_reference,
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
                    modality=modality,
                )

            # This image should appear at the index above. But the indexes should be in
            # order in the CSV.
            if image_idx != len(modality_tiles[grid_tile].images):
                # This should be an error but currently I realized there are one or two
                # tiles that actually have two timestamps in the original rslearn
                # dataset, which means the OlmoEarth Pretrain dataset has two sets of entries in
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
            resolution=abs(grid_tile.get_x_resolution()),
            time=tile.center_time,
            col=grid_tile.col,
            row=grid_tile.row,
            sample_id=grid_tile.sample_id,
            use_grid_reference=grid_tile.use_grid_reference,
            x_resolution=grid_tile.get_x_resolution(),
            y_resolution=grid_tile.get_y_resolution(),
            bounds=grid_tile.get_bounds(),
        )
        for band_set in modality.band_sets:
            fname = get_modality_fname(
                path,
                modality,
                time_span,
                window_metadata,
                band_set.get_resolution(),
                "tif",
            )
            tile.band_sets[band_set] = fname

    return list(modality_tiles.values())


def parse_dataset(
    path: UPath,
    supported_modalities: list[ModalitySpec] = Modality.values(),
    multitemporal_time_spans: list[TimeSpan] | None = None,
) -> dict[ModalitySpec, dict[TimeSpan, list[ModalityTile]]]:
    """Parse the various per-modality tiles present in a OlmoEarth Pretrain dataset.

    Returns:
        a mapping from modality -> time span (e.g. yearly / two-week) -> list of tiles.
    """
    tiles: dict[ModalitySpec, dict[TimeSpan, list[ModalityTile]]] = {}
    if multitemporal_time_spans is None:
        multitemporal_time_spans = [TimeSpan.YEAR]

    for modality in Modality.values():
        if modality.ignore_when_parsing:
            continue
        if modality not in supported_modalities:
            logger.warning(
                f"ignoring modality {modality.name} not in supported_modalities"
            )
            continue

        if modality.is_multitemporal:
            time_spans = multitemporal_time_spans
        else:
            # Just need to load the static data.
            time_spans = [TimeSpan.STATIC]

        # For each possible time span available for this modality, parse the associated
        # CSV to get the ModalityTiles under that time span.
        tiles[modality] = {}
        for time_span in time_spans:
            # Reconstruct the CSV filename from the grid resolution, modality, and time span.
            modality_dir = get_modality_dir(path, modality, time_span)
            csv_fname = path / f"{modality_dir.name}.csv"
            logger.debug(f"Parsing {modality.name} {time_span} {csv_fname}")
            tiles[modality][time_span] = parse_modality_csv(  # type: ignore
                path,
                modality,
                time_span,  # type: ignore
                csv_fname,
            )

    return tiles
