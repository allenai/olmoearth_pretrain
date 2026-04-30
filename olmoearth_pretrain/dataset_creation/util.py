"""Utilities related to dataset creation."""

import csv
from datetime import datetime

from rslearn.dataset import Window
from upath import UPath

from olmoearth_pretrain.data.constants import ModalitySpec, TimeSpan
from olmoearth_pretrain.dataset.utils import WindowMetadata, get_modality_dir

from .constants import (
    DEFAULT_USE_GRID_REFERENCE,
    SAMPLE_ID_OPTION,
    USE_GRID_REFERENCE_OPTION,
    WINDOW_SIZE,
)


def parse_bool(value: object, default: bool = False) -> bool:
    """Parse a permissive boolean value."""
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "y", "on"}:
            return True
        if normalized in {"0", "false", "no", "n", "off"}:
            return False
    return bool(value)


def derive_grid_reference(bounds: tuple[int, int, int, int]) -> tuple[int, int]:
    """Derive the legacy OlmoEarth grid reference from window bounds."""
    width = bounds[2] - bounds[0]
    height = bounds[3] - bounds[1]
    if width != WINDOW_SIZE or height != WINDOW_SIZE:
        raise ValueError(
            f"legacy grid reference requires {WINDOW_SIZE}x{WINDOW_SIZE} windows but got {width}x{height}"
        )
    if bounds[0] % WINDOW_SIZE != 0 or bounds[1] % WINDOW_SIZE != 0:
        raise ValueError(
            "legacy grid reference requires bounds aligned to the OlmoEarth grid"
        )
    return (bounds[0] // WINDOW_SIZE, bounds[1] // WINDOW_SIZE)


def get_window_metadata(window: Window) -> WindowMetadata:
    """Extract metadata about a window from the window.

    Args:
        window: the Window.

    Returns:
        WindowMetadata object containing the OlmoEarth metadata derived from the window.
    """
    if window.time_range is None:
        raise ValueError(f"window {window.name} must include a time_range")

    use_grid_reference = parse_bool(
        window.options.get(USE_GRID_REFERENCE_OPTION), DEFAULT_USE_GRID_REFERENCE
    )
    center_time = (
        window.time_range[0] + (window.time_range[1] - window.time_range[0]) / 2
    )
    sample_id = window.options.get(SAMPLE_ID_OPTION)
    if sample_id is None and not use_grid_reference:
        sample_id = window.name

    metadata_kwargs = dict(
        crs=str(window.projection.crs),
        resolution=abs(window.projection.x_resolution),
        time=center_time,
        sample_id=str(sample_id) if sample_id is not None else None,
        use_grid_reference=use_grid_reference,
        x_resolution=window.projection.x_resolution,
        y_resolution=window.projection.y_resolution,
        bounds=window.bounds,
    )
    if use_grid_reference:
        col, row = derive_grid_reference(window.bounds)
        metadata_kwargs["col"] = col
        metadata_kwargs["row"] = row

    return WindowMetadata(**metadata_kwargs)


def _serialize_time(value: datetime | str) -> str:
    if isinstance(value, datetime):
        return value.isoformat()
    return value


def get_metadata_row(
    window_metadata: WindowMetadata,
    image_idx: int | str,
    start_time: datetime | str,
    end_time: datetime | str,
) -> dict[str, str | int | float]:
    """Build a metadata CSV row for one output image."""
    bounds = window_metadata.bounds or ("", "", "", "")
    return dict(
        use_grid_reference=str(window_metadata.use_grid_reference).lower(),
        sample_id=window_metadata.sample_id or "",
        crs=window_metadata.crs,
        x_resolution=window_metadata.get_x_resolution(),
        y_resolution=window_metadata.get_y_resolution(),
        bounds_left=bounds[0],
        bounds_bottom=bounds[1],
        bounds_right=bounds[2],
        bounds_top=bounds[3],
        col="" if window_metadata.col is None else window_metadata.col,
        row="" if window_metadata.row is None else window_metadata.row,
        tile_time=window_metadata.time.isoformat(),
        image_idx=image_idx,
        start_time=_serialize_time(start_time),
        end_time=_serialize_time(end_time),
    )


def write_metadata_rows(
    metadata_fname: UPath, rows: list[dict[str, str | int | float]]
) -> None:
    """Write one or more metadata rows to a CSV file."""
    from .constants import METADATA_COLUMNS

    metadata_fname.parent.mkdir(parents=True, exist_ok=True)
    with metadata_fname.open("w") as f:
        writer = csv.DictWriter(f, fieldnames=METADATA_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def write_single_metadata_row(
    metadata_fname: UPath,
    window_metadata: WindowMetadata,
    image_idx: int | str,
    start_time: datetime | str,
    end_time: datetime | str,
) -> None:
    """Write a single metadata row to a CSV file."""
    write_metadata_rows(
        metadata_fname,
        [get_metadata_row(window_metadata, image_idx, start_time, end_time)],
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
