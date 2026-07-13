"""Constants for turning the open-set label bank into a pretraining dataset.

These are distinct from :mod:`olmoearth_pretrain.open_set_segmentation_data.io`
(which describes the *label bank* on-disk format). The constants here describe the
*pretraining* dataset built on top of the label bank: window geometry, the combined
``open_set`` / ``open_set_regression`` label layers, and datasets that are excluded
because they are held-out evaluations.
"""

# Source datasets excluded from the open-set pretraining dataset because they are
# held-out evaluation datasets (the GeoBench EuroSAT / So2Sat LCZ42 evals) whose
# contamination cannot be removed geographically -- they have no reliable per-sample
# geocoordinates. The whole source dataset is dropped from class assembly and window
# creation. Other evals (PASTIS, yemen_crop) are excluded geographically instead, via
# an exclusion GeoJSON of their val/test extents (see generate_eval_exclusion_geojson).
EXCLUDED_SLUGS = frozenset({"eurosat", "so2sat_lcz42"})

# Open-set windows are 128x128 at 10 m/pixel, centered on each label sample.
OPEN_SET_WINDOW_SIZE = 128
OPEN_SET_RESOLUTION = 10  # meters/pixel

# A single rslearn group holds all open-set windows.
OPEN_SET_GROUP = "open_set"

# Combined classification label layer. Single-band, globally-unique class ids.
# dtype is uint16 because the combined class count across all datasets can exceed 255.
OPEN_SET_LAYER = "open_set"
OPEN_SET_DTYPE = "uint16"
OPEN_SET_NODATA = 65535

# Synthetic training group that merges all presence-only datasets so that, at train
# time, each presence-only class supplies negatives for the others.
PRESENCE_ONLY_GROUP = "__presence_only__"

# Combined regression label layer. Two bands, both uint16:
#   band 0: regression dataset id (1-based; 0 = no regression label at this pixel).
#   band 1: value linearly remapped from the dataset's [min, max] to [1, 65535]
#           (0 = nodata).
OPEN_SET_REGRESSION_LAYER = "open_set_regression"
OPEN_SET_REGRESSION_DTYPE = "uint16"
REGRESSION_DATASET_ID_NODATA = 0
REGRESSION_VALUE_NODATA = 0
REGRESSION_VALUE_MIN_OUT = 1
REGRESSION_VALUE_MAX_OUT = 65535
