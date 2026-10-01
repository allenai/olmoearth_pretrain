"""Utilities related to dataset creation."""

from datetime import datetime

from rslearn.dataset import Window
from upath import UPath

from olmoearth_pretrain.data.constants import ModalitySpec, TimeSpan
from olmoearth_pretrain.dataset.utils import WindowMetadata, get_modality_dir

from .constants import WINDOW_DURATION


def get_window_metadata(window: Window) -> WindowMetadata:
    """Extract metadata about a window from the window.

    Args:
        window: the Window.

    Returns:
        WindowMetadata object containing the OlmoEarth Pretrain metadata encoded within the window
    """
    parts = window.name.split("_")
    if len(parts) == 4:
        try:
            crs, resolution, col, row = parts
            center_time = window.time_range[0] + WINDOW_DURATION // 2
            return WindowMetadata(
                crs,
                float(resolution),
                int(col),
                int(row),
                center_time,
            )
        except ValueError:
            # Not a grid tile name (crs_res_col_row); fall through to options.
            pass

    # Windows that are centered on a sample rather than snapped to a global grid
    # (e.g. the open-set segmentation dataset) are named slug_sampleid and cannot be
    # parsed as crs_res_col_row. They instead carry the metadata explicitly in
    # window.options.
    opts = window.options
    time = opts["time"]
    center_time = datetime.fromisoformat(time) if isinstance(time, str) else time
    return WindowMetadata(
        crs=opts["crs"],
        resolution=float(opts["resolution"]),
        col=int(opts["col"]),
        row=int(opts["row"]),
        time=center_time,
        example_id=opts.get("example_id"),
    )


def get_modality_temp_meta_dir(
    olmoearth_path: UPath, modality: ModalitySpec, time_span: TimeSpan
) -> UPath:
    """Get the directory to store per-example metadata files for a given modality.

    Args:
        olmoearth_path: the OlmoEarth Pretrain dataset root.
        modality: the modality.
        time_span: the time span of this data.

    Returns:
        the directory to store the metadata files.
    """
    modality_dir = get_modality_dir(olmoearth_path, modality, time_span)
    return olmoearth_path / (modality_dir.name + "_meta")


def get_modality_temp_meta_fname(
    olmoearth_path: UPath, modality: ModalitySpec, time_span: TimeSpan, example_id: str
) -> UPath:
    """Get the temporary filename to store the metadata for an example and modality.

    This is created by the olmoearth_pretrain.dataset_creation.rslearn_to_olmoearth scripts. It will
    then be read by olmoearth_prertain.dataset_creation.make_meta_summary to create the final
    metadata CSV.

    Args:
        olmoearth_path: the OlmoEarth Pretrain dataset root.
        modality: the modality name.
        time_span: the TimeSpan.
        example_id: the example ID.

    Returns:
        the filename for the per-example metadata CSV.
    """
    temp_meta_dir = get_modality_temp_meta_dir(olmoearth_path, modality, time_span)
    return temp_meta_dir / f"{example_id}.csv"
