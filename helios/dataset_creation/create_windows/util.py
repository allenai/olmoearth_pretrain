"""Utilities for creating windows."""

from datetime import timedelta

from rasterio.crs import CRS
from rslearn.dataset import Window
from rslearn.utils.geometry import Projection
from upath import UPath

from ..util import WindowMetadata

# List of resolutions that are needed.
# When creating a window at a given resolution, we ensure that it is covered at every
# coarser resolution too.
WINDOW_RESOLUTIONS = [0.625, 10, 160]

WINDOW_DURATION = timedelta(days=14)
WINDOW_SIZE = 256


def create_window(ds_path: UPath, metadata: WindowMetadata) -> list[Window]:
    """Create one or more rslearn windows for ingesting data for Helios.

    A window is created at each predefined resolution that is equal to or coarser than
    the provided resolution. This way, lower resolution data is included at all
    locations where higher resolution data is ingested.

    Args:
        ds_path: the rslearn dataset path.
        metadata: the metadata that defines the window.

    Returns:
        the new windows.
    """
    windows = []
    for resolution in WINDOW_RESOLUTIONS:
        # Only create windows at resolutions equal to or coarser than the provided one.
        if resolution < metadata.resolution:
            continue

        # Adjust the metadata for this resolution (i.e., compute the window that is
        # aligned with the grid in case the resolution is coarser).
        factor = round(resolution / metadata.resolution)
        cur_metadata = WindowMetadata(
            metadata.crs,
            resolution,
            metadata.col // factor,
            metadata.row // factor,
            metadata.time,
        )

        # Compute the window attributes based on the WindowMetadata.
        group = f"res_{resolution}"
        window_name = cur_metadata.get_window_name()
        bounds = (
            cur_metadata.col * WINDOW_SIZE,
            cur_metadata.row * WINDOW_SIZE,
            (cur_metadata.col + 1) * WINDOW_SIZE,
            (cur_metadata.row + 1) * WINDOW_SIZE,
        )
        time_range = (
            cur_metadata.time - WINDOW_DURATION // 2,
            cur_metadata.time + WINDOW_DURATION // 2,
        )
        projection = Projection(
            CRS.from_string(cur_metadata.crs), resolution, -resolution
        )

        # Create the window.
        window = Window(
            path=Window.get_window_root(ds_path, group, window_name),
            group=group,
            name=window_name,
            projection=projection,
            bounds=bounds,
            time_range=time_range,
        )
        window.save()
        windows.append(window)

    return windows
