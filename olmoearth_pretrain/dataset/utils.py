"""Utilities and base classes relating to the OlmoEarth Pretrain dataset structure."""

from dataclasses import dataclass
from datetime import datetime
from typing import cast

from upath import UPath

from olmoearth_pretrain.data.constants import (
    BASE_RESOLUTION,
    IMAGE_TILE_SIZE,
    ModalitySpec,
    TimeSpan,
)


@dataclass(frozen=True)
class WindowMetadata:
    """Class to represent the metadata associated with an rslearn window used for OlmoEarth Pretrain.

    Historically the raw dataset contract used ``crs/col/row`` as the canonical
    identifier. Newer datasets can opt out of that and instead use ``sample_id`` along
    with geometry fields persisted in the per-modality CSVs.
    """

    crs: str
    resolution: float
    time: datetime
    col: int | None = None
    row: int | None = None
    sample_id: str | None = None
    use_grid_reference: bool = True
    x_resolution: float | None = None
    y_resolution: float | None = None
    bounds: tuple[int, int, int, int] | None = None

    def __post_init__(self) -> None:  # noqa: D105
        if self.use_grid_reference and (self.col is None or self.row is None):
            raise ValueError("grid-referenced metadata requires both col and row")
        if not self.use_grid_reference and not self.sample_id:
            raise ValueError("sample-id metadata requires sample_id")
        if self.bounds is not None and len(self.bounds) != 4:
            raise ValueError("bounds must contain exactly four values")

    def get_window_name(self) -> str:
        """Encode the metadata back to a window name."""
        if self.use_grid_reference:
            if self.col is None or self.row is None:
                raise ValueError("grid-referenced metadata is missing col/row")
            return f"{self.crs}_{self.resolution}_{self.col}_{self.row}"
        return cast(str, self.sample_id)

    def get_example_id(self) -> str:
        """Get the identifier shared across raw data files for this sample."""
        if self.use_grid_reference:
            if self.col is None or self.row is None:
                raise ValueError("grid-referenced metadata is missing col/row")
            return f"{self.crs}_{self.col}_{self.row}"
        return cast(str, self.sample_id)

    def get_resolution_factor(self) -> int:
        """Get the resolution factor.

        See helios.data.constants.
        """
        return round(self.resolution / BASE_RESOLUTION)

    def get_x_resolution(self) -> float:
        """Get the x resolution in projection units."""
        if self.x_resolution is not None:
            return self.x_resolution
        return self.resolution

    def get_y_resolution(self) -> float:
        """Get the y resolution in projection units."""
        if self.y_resolution is not None:
            return self.y_resolution
        return -self.resolution

    def get_tile_size(self) -> int:
        """Get the raw tile size in pixels along one axis."""
        if self.bounds is None:
            return IMAGE_TILE_SIZE
        return self.bounds[2] - self.bounds[0]


def get_modality_dir(path: UPath, modality: ModalitySpec, time_span: TimeSpan) -> UPath:
    """Get the directory where data should be stored for the specified modality.

    Args:
        path: the OlmoEarth Pretrain dataset root.
        modality: the modality.
        time_span: the time span, which determines suffix of directory name.

    Returns:
        directory within path to store the modality.
    """
    suffix = time_span.get_suffix()
    dir_name = f"{modality.get_tile_resolution()}_{modality.name}{suffix}"
    return path / dir_name


def list_examples_for_modality(
    path: UPath, modality: ModalitySpec, time_span: TimeSpan
) -> list[str]:
    """List the example IDs available for the specified modality.

    This is determined by listing the contents of the modality directory. Index and
    metadata CSVs are not used.

    Args:
        path: the OlmoEarth Pretrain dataset root.
        modality: the modality to check.
        time_span: the time span to check.

    Returns:
        a list of example IDs
    """
    modality_dir = get_modality_dir(path, modality, time_span)
    if not modality_dir.exists():
        return []

    # We just list the directory and strip the extension.
    example_ids = []
    for fname in modality_dir.iterdir():
        example_ids.append(fname.name.split(".")[0])
    return example_ids


def get_modality_fname(
    path: UPath,
    modality: ModalitySpec,
    time_span: TimeSpan,
    window_metadata: WindowMetadata,
    resolution: float,
    ext: str,
) -> UPath:
    """Get the filename where to store data for the specified window and modality.

    Args:
        path: the OlmoEarth Pretrain dataset root.
        modality: the modality.
        time_span: the time span of this data.
        window_metadata: details extracted from the window name.
        resolution: the resolution of this band. This should be a power of 2 multiplied
            by the window resolution.
        ext: the filename extension, like "tif" or "geojson".

    Returns:
        the filename to store the data in.
    """
    modality_dir = get_modality_dir(path, modality, time_span)
    fname = f"{window_metadata.get_example_id()}_{resolution}.{ext}"
    return modality_dir / fname
