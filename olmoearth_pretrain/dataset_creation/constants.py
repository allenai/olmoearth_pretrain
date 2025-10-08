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
    "crs",
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
