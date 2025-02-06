"""Constants related to Helios dataset creation."""

from rslearn.utils.raster_format import GeotiffRasterFormat

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
