"""Utilities related to dataset creation."""

from datetime import datetime

from upath import UPath


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
        return (
            f"{self.crs}_{self.resolution}_"
            + f"{self.col}_{self.row}_"
            + self.time.isoformat()
        )


def parse_window_name(window_name: str) -> WindowMetadata:
    """Parse the specified window name, extracting the encoded metadata.

    Args:
        window_name: the window name to parse.

    Returns:
        WindowMetadata object containing the metadata encoded within the window name
    """
    crs, resolution, col, row, time = window_name.split("_")
    return WindowMetadata(
        crs,
        float(resolution),
        int(col),
        int(row),
        datetime.fromisoformat(time),
    )


def get_modality_dir(helios_path: UPath, modality: str) -> UPath:
    """Get the directory where data should be stored for the specified modality.

    Args:
        helios_path: the Helios dataset root.
        modality: the modality.

    Returns:
        directory within helios_path to store the modality.
    """
    return helios_path / modality


def list_examples_for_modality(helios_path: UPath, modality: str) -> list[str]:
    """List the example IDs available for the specified modality.

    This is determined by listing the contents of the modality directory. Index and
    metadata CSVs are not used.

    Args:
        helios_path: the Helios dataset root.
        modality: the modality to check.

    Returns:
        a list of example IDs
    """
    modality_dir = get_modality_dir(helios_path, modality)
    if not modality_dir.exists():
        return []

    # We just list the directory and strip the extension.
    example_ids = []
    for fname in modality_dir.iterdir():
        example_ids.append(fname.name.split(".")[0])
    return example_ids


def get_modality_fname(
    helios_path: UPath,
    modality: str,
    window_metadata: WindowMetadata,
    resolution: float,
    ext: str,
) -> UPath:
    """Get the filename where to store data for the specified window and modality.

    Args:
        helios_path: the Helios dataset root.
        modality: the modality name.
        window_metadata: details extracted from the window name.
        resolution: the resolution of this band. This should be a power of 2 multiplied
            by the window resolution.
        ext: the filename extension, like "tif" or "geojson".

    Returns:
        the filename to store the data in.
    """
    crs = window_metadata.crs
    col = window_metadata.col
    row = window_metadata.row
    fname = f"{crs}_{col}_{row}_{resolution}.{ext}"
    return helios_path / modality / fname


def get_modality_temp_meta_dir(helios_path: UPath, modality: str) -> UPath:
    """Get the directory to store per-example metadata files for a given modality.

    Args:
        helios_path: the Helios dataset root.
        modality: the modality name.

    Returns:
        the directory to store the metadata files.
    """
    return helios_path / f"{modality}_meta"


def get_modality_temp_meta_fname(
    helios_path: UPath, modality: str, example_id: str
) -> UPath:
    """Get the temporary filename to store the metadata for an example and modality.

    This is created by the helios.dataset_creation.rslearn_to_helios scripts. It will
    then be read by helios.dataset_creation.make_meta_summary to create the final
    metadata CSV.

    Args:
        helios_path: the Helios dataset root.
        modality: the modality name.
        example_id: the example ID.

    Returns:
        the filename for the per-example metadata CSV.
    """
    return get_modality_temp_meta_dir(helios_path, modality) / f"{example_id}.csv"
