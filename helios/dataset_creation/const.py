"""Constants related to Helios dataset creation."""

MODALITIES = [
    "landsat_freq",
    "landsat_monthly",
    "naip",
    "openstreetmap",
    "sentinel2_freq",
    "sentinel2_monthly",
    "worldcover",
]

# Columns in the per-modality metadata CSVs.
METADATA_COLUMNS = [
    "example_id",
    "image_idx",
    "start_time",
    "end_time",
]

# Columns in the index CSV.
# It is the columns listed below providing metadata about the example, plus one column
# per modality indicating whether data for that modality is present at this example.
INDEX_COLUMNS = [
    "example_id",
    "projection",
    "resolution",
    "start_column",
    "start_row",
    "time",
] + MODALITIES
