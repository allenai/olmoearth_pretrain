"""Utilities related to dataset creation."""

from rslearn.dataset import Window
from upath import UPath

from helios.data.constants import ModalitySpec, TimeSpan
from helios.dataset.util import WindowMetadata, get_modality_dir

from .constants import WINDOW_DURATION


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
