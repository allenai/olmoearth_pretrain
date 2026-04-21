"""Constants related to OlmoEarth Pretrain dataset creation."""

from datetime import timedelta

from rslearn.utils.raster_format import GeotiffRasterFormat

# List of resolutions that are needed.
# When creating a window at a given resolution, we ensure that it is covered at every
# coarser resolution too.
WINDOW_RESOLUTIONS = [0.625, 10, 160]

WINDOW_DURATION = timedelta(days=14)
WINDOW_SIZE = 256

# Columns in the per-modality metadata CSVs.
METADATA_COLUMNS = [
    "use_grid_reference",
    "sample_id",
    "crs",
    "x_resolution",
    "y_resolution",
    "bounds_left",
    "bounds_bottom",
    "bounds_right",
    "bounds_top",
    "col",
    "row",
    "tile_time",
    "image_idx",
    "start_time",
    "end_time",
]

GEOTIFF_BLOCK_SIZE = 32
GEOTIFF_RASTER_FORMAT = GeotiffRasterFormat(
    block_size=GEOTIFF_BLOCK_SIZE, always_enable_tiling=True
)

# Window option keys used to carry the raw dataset contract through rslearn windows.
USE_GRID_REFERENCE_OPTION = "use_grid_reference"
SAMPLE_ID_OPTION = "sample_id"

DEFAULT_USE_GRID_REFERENCE = True
