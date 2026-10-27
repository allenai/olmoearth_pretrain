"""Paths, grid and constants shared by the USDM dataset stages."""

from __future__ import annotations

from datetime import UTC, date, datetime
from pathlib import Path

import numpy as np

# The ERA5 side is shared with the FLUXNET eval: same mirror, band order,
# window length, layer names and no-data value.
from ..fluxnet.common import (  # noqa: F401
    ERA5_LAYER,
    ERA5_MIRROR,
    ERA5L_BANDS,
    LABEL_LAYER,
    NODATA,
    WINDOW_DAYS,
    haversine_km,
)

DEFAULT_ROOT = Path("/weka/dfive-default/rslearn-eai/datasets/usdm")

# Weekly USDM maps (valid on Tuesdays, 8 a.m. EST), one zipped shapefile each:
# EPSG:4326, one multipolygon per category DM = 0 (D0) .. 4 (D4), categories
# mutually exclusive; land inside the US and outside every polygon is "none".
USDM_URL = "https://droughtmonitor.unl.edu/data/shapefiles_m/USDM_{ymd}_M.zip"
FIRST_MAP = date(2000, 1, 4)
# State boundaries (Census cartographic boundary, 1:20M) for the CONUS mask.
STATES_URL = "https://www2.census.gov/geo/tiger/GENZ2018/shp/cb_2018_us_state_20m.zip"
NON_CONUS_STATES = ("AK", "HI", "PR")

# Label grid: ERA5-Land cell centres (multiples of 0.1 deg) over CONUS, rows
# north to south. Each cell is rasterized on SUB x SUB sub-cells, so a cell's
# severity is an area average with 1 / SUB**2 resolution.
GRID_RES = 0.1
GRID_NORTH = 50.0
GRID_SOUTH = 24.0
GRID_WEST = -125.0
GRID_EAST = -66.5
SUB = 4
GRID_ROWS = int(round((GRID_NORTH - GRID_SOUTH) / GRID_RES)) + 1
GRID_COLS = int(round((GRID_EAST - GRID_WEST) / GRID_RES)) + 1
# A cell is in the domain when at least this many of its SUB**2 sub-cells are
# inside CONUS (drops cells that are mostly sea or Canada / Mexico).
MIN_INSIDE_SUBCELLS = 12
# Value stored for cells outside the domain in the uint8 severity grid.
OUTSIDE = 255

EPOCH = datetime(1970, 1, 1, tzinfo=UTC)
EPOCH_DATE = date(1970, 1, 1)

# Cell-centre latitude of each grid row (north to south) and longitude of each
# column (west to east, -180..180).
GRID_LATS = np.round(GRID_NORTH - np.arange(GRID_ROWS) * GRID_RES, 1)
GRID_LONS = np.round(GRID_WEST + np.arange(GRID_COLS) * GRID_RES, 1)


def grid_index(lat: float, lon: float) -> tuple[int, int] | None:
    """Grid cell holding a point.

    Args:
        lat: Latitude (deg).
        lon: Longitude (deg, -180..180).

    Returns:
        ``(row, col)``, or None outside the grid.
    """
    row = int(np.floor((GRID_NORTH + GRID_RES / 2 - lat) / GRID_RES))
    col = int(np.floor((lon - (GRID_WEST - GRID_RES / 2)) / GRID_RES))
    if 0 <= row < GRID_ROWS and 0 <= col < GRID_COLS:
        return row, col
    return None
