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
# ``sun_elevation`` and ``platform`` are only populated for Landsat (used to
# convert DN to reflectance / brightness temperature at h5-creation); other
# modalities leave them empty (csv.DictWriter fills missing keys with restval).
METADATA_COLUMNS = [
    "crs",
    "col",
    "row",
    "tile_time",
    "image_idx",
    "start_time",
    "end_time",
    "sun_elevation",
    "platform",
]

GEOTIFF_BLOCK_SIZE = 32
GEOTIFF_RASTER_FORMAT = GeotiffRasterFormat(
    block_size=GEOTIFF_BLOCK_SIZE, always_enable_tiling=True
)
