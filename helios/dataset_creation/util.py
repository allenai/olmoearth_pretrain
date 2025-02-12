"""Utilities related to dataset creation."""

from datetime import datetime

from rslearn.dataset import Window
from upath import UPath

from helios.data.constants import BASE_RESOLUTION, ModalitySpec, TimeSpan

from .constants import WINDOW_DURATION


class WindowMetadata:
    """Class to represent the metadata associated with an rslearn window used for Helios.

    The window name specifies the CRS, column, row, resolution, and timestamp.
    These can also be derived from the rslearn window metadata.
    """

    def __init__(
        self,
        crs: str,
        resolution: float,
        col: int,
        row: int,
        time: datetime,
    ):
        """Create a new WindowMetadata.

        Args:
            crs: the UTM CRS that the example is in.
            resolution: the resolution of the grid that this window is on.
            col: the column of the tile in the grid.
            row: the row of the tile in the grid.
            time: the center time used at this tile.
        """
        self.crs = crs
        self.resolution = resolution
        self.col = col
        self.row = row
        self.time = time

    def get_window_name(self) -> str:
        """Encode the metadata back to a window name."""
        return f"{self.crs}_{self.resolution}_{self.col}_{self.row}"

    def get_resolution_factor(self) -> int:
        """Get the resolution factor.

        See helios.data.constants.
        """
        return round(self.resolution / BASE_RESOLUTION)


def get_window_metadata(window: Window) -> WindowMetadata:
    """Extract metadata about a window from the window.

    Args:
        window: the Window.

    Returns:
        WindowMetadata object containing the Helios metadata encoded within the window
    """
    crs, resolution, col, row = window.name.split("_")
    center_time = window.time_range[0] + WINDOW_DURATION // 2
    return WindowMetadata(
        crs,
        float(resolution),
        int(col),
        int(row),
        center_time,
    )


def get_modality_dir(
    helios_path: UPath, modality: ModalitySpec, time_span: TimeSpan
) -> UPath:
    """Get the directory where data should be stored for the specified modality.

    Args:
        helios_path: the Helios dataset root.
        modality: the modality.
        time_span: the time span, which determines suffix of directory name.

    Returns:
        directory within helios_path to store the modality.
    """
    suffix = time_span.get_suffix()
    dir_name = f"{modality.get_tile_resolution()}_{modality.name}{suffix}"
    return helios_path / dir_name


def list_examples_for_modality(
    helios_path: UPath, modality: ModalitySpec, time_span: TimeSpan
) -> list[str]:
    """List the example IDs available for the specified modality.

    This is determined by listing the contents of the modality directory. Index and
    metadata CSVs are not used.

    Args:
        helios_path: the Helios dataset root.
        modality: the modality to check.
        time_span: the time span to check.

    Returns:
        a list of example IDs
    """
    modality_dir = get_modality_dir(helios_path, modality, time_span)
    if not modality_dir.exists():
        return []

    # We just list the directory and strip the extension.
    example_ids = []
    for fname in modality_dir.iterdir():
        example_ids.append(fname.name.split(".")[0])
    return example_ids


def get_modality_fname(
    helios_path: UPath,
    modality: ModalitySpec,
    time_span: TimeSpan,
    window_metadata: WindowMetadata,
    resolution: float,
    ext: str,
) -> UPath:
    """Get the filename where to store data for the specified window and modality.

    Args:
        helios_path: the Helios dataset root.
        modality: the modality.
        time_span: the time span of this data.
        window_metadata: details extracted from the window name.
        resolution: the resolution of this band. This should be a power of 2 multiplied
            by the window resolution.
        ext: the filename extension, like "tif" or "geojson".

    Returns:
        the filename to store the data in.
    """
    modality_dir = get_modality_dir(helios_path, modality, time_span)
    crs = window_metadata.crs
    col = window_metadata.col
    row = window_metadata.row
    fname = f"{crs}_{col}_{row}_{resolution}.{ext}"
    return modality_dir / fname


def get_modality_temp_meta_dir(
    helios_path: UPath, modality: ModalitySpec, time_span: TimeSpan
) -> UPath:
    """Get the directory to store per-example metadata files for a given modality.

    Args:
        helios_path: the Helios dataset root.
        modality: the modality.
        time_span: the time span of this data.

    Returns:
        the directory to store the metadata files.
    """
    modality_dir = get_modality_dir(helios_path, modality, time_span)
    return helios_path / (modality_dir.name + "_meta")


def get_modality_temp_meta_fname(
    helios_path: UPath, modality: ModalitySpec, time_span: TimeSpan, example_id: str
) -> UPath:
    """Get the temporary filename to store the metadata for an example and modality.

    This is created by the helios.dataset_creation.rslearn_to_helios scripts. It will
    then be read by helios.dataset_creation.make_meta_summary to create the final
    metadata CSV.

    Args:
        helios_path: the Helios dataset root.
        modality: the modality name.
        time_span: the TimeSpan.
        example_id: the example ID.

    Returns:
        the filename for the per-example metadata CSV.
    """
    temp_meta_dir = get_modality_temp_meta_dir(helios_path, modality, time_span)
    return temp_meta_dir / f"{example_id}.csv"
