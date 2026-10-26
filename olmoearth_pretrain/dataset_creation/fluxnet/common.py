"""Paths and constants shared by the FLUXNET dataset stages."""

from __future__ import annotations

from pathlib import Path

import numpy as np

DEFAULT_ROOT = Path("/weka/dfive-default/rslearn-eai/datasets/fluxnet")

# ERA5-Land daily UTC mirror on weka (1980-01-01 .. 2026-02-28, 0.1 deg,
# longitudes in [0, 360)), chunked (14 bands, 3650 days, 10 lat, 10 lon) so one
# site's full record is a handful of chunk reads.
ERA5_MIRROR = Path("/weka/dfive-default/rslearn-eai/datasets/era5_land_global.zarr")

# Band order of the era5d_448d_historical / era5_daily layers (ERA5L_DAY_10).
ERA5L_BANDS = (
    "d2m",
    "e",
    "pev",
    "ro",
    "sp",
    "ssr",
    "ssrd",
    "str",
    "swvl1",
    "swvl2",
    "t2m",
    "tp",
    "u10",
    "v10",
)
NODATA = -9999.0
WINDOW_DAYS = 448

ERA5_LAYER = "era5_daily"
LABEL_LAYER = "label"

EARTH_RADIUS_KM = 6371.0


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance.

    Args:
        lat1: First latitude (deg).
        lon1: First longitude (deg).
        lat2: Second latitude (deg).
        lon2: Second longitude (deg).

    Returns:
        Distance in km.
    """
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dp, dl = p2 - p1, np.radians(lon2 - lon1)
    a = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return float(2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(a)))
