"""Paths, grids and constants shared by the snow dataset stages."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

# The ERA5 side is shared with the FLUXNET and USDM evals: same mirror, band
# order, window length, layer names and no-data value.
from ..fluxnet.common import (  # noqa: F401
    ERA5_LAYER,
    ERA5_MIRROR,
    ERA5L_BANDS,
    LABEL_LAYER,
    NODATA,
    WINDOW_DAYS,
)

DEFAULT_ROOT = Path("/weka/dfive-default/rslearn-eai/datasets/snow")

EPOCH = datetime(1970, 1, 1, tzinfo=UTC)

# MODIS/Terra Snow Cover 8-Day L3 Global 500 m SIN Grid, collection 6.1, as the
# Planetary Computer's per-asset COGs (anonymous read with a SAS token).
PRODUCT = "MOD10A2"
BLOB_ROOT = "https://modiseuwest.blob.core.windows.net/modis-061-cogs"
SAS_URL = "https://planetarycomputer.microsoft.com/api/sas/v1/token/modiseuwest/modis-061-cogs"
ASSET = "Maximum_Snow_Extent"

# MODIS sinusoidal grid: 36 x 18 tiles of 2400 x 2400 pixels (~463 m).
SPHERE_RADIUS = 6371007.181
GRID_X_MIN = -20015109.355798
GRID_Y_MAX = 10007554.677899
TILE_SIZE_M = 1111950.5197665
TILE_PX = 2400

# Maximum_Snow_Extent classes (MOD10A2 C6.1 user guide). Every other value is a
# land pixel whose state is unknown: 0 missing, 1 no decision, 11 night,
# 50 cloud, 254 detector saturated.
SNOW = 200
NO_SNOW = 25
WATER = (37, 39, 100)  # inland water, ocean, lake ice
FILL = 255

# Label grid: ERA5-Land cell centres, multiples of 0.1 deg. A cell is keyed by
# its centre in tenths of a degree, longitude in [-1800, 1800).
CELL_RES = 0.1
