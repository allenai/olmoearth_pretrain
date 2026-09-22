"""Shared constants and helpers for the CY-Bench ERA5-Land export.

The pipeline (see README.md in this directory) turns the CY-Bench crop-yield
benchmark (WUR-AI/AgML-CY-Bench, Zenodo record 17279151) into an rslearn
dataset consumable by the ERA5 daily encoder evals:

1. ``build_weights``  — region x ERA5-Land-cell weights (polygon coverage x
   crop area fraction), one row per (crop, adm_id, cell).
2. ``aggregate_era5`` — stream ERA5-Land daily Zarr chunks from EarthDataHub
   and reduce them to per-region weighted means (no tile store, no raw grid
   kept on disk).
3. ``write_windows``  — slice 448-day windows ending at each label's crop
   calendar end-of-season and write them as materialized rslearn windows with
   a ``yield`` label layer.

Everything is keyed on our own canonical ERA5-Land 0.1 deg grid (cell centers
at lat = 90 - 0.1 i, lon = 0.1 j with lon in [0, 360)); ``aggregate_era5``
maps that grid onto the Zarr coordinate arrays at runtime, so nothing here
assumes the Zarr's axis order.
"""

from __future__ import annotations

import logging
import math
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ERA5-Land band order — must match Modality.ERA5L_DAY_10 and the pretraining
# ``era5d_448d_historical`` layer so the eval loader's band check passes.
ERA5L_BANDS: list[str] = [
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
]
NODATA = -9999.0
WINDOW_DAYS = 448

# Canonical ERA5-Land 0.1 deg grid.
CELL_DEG = 0.1
N_LAT = 1801  # 90 .. -90
N_LON = 3600  # 0 .. 359.9
# Fine sub-grid used to rasterize polygons for coverage fractions
# (10 x 10 sub-pixels per ERA5 cell).
FINE_PER_CELL = 10
FINE_DEG = CELL_DEG / FINE_PER_CELL

# Layer / group naming in the output rslearn dataset.
ERA5_LAYER = "era5_daily"
LABEL_LAYER = "label"

# CY-Bench crop -> WorldCereal area-fraction image (in the AgML-CY-Bench repo,
# data_preparation/global_crop_AFIs_ESA_WC/).
CROP_TO_AFI = {
    "maize": "crop_mask_maize_WC.tif",
    "wheat": "crop_mask_winter_spring_cereals_WC.tif",
}

# Default weka layout.
DEFAULT_ROOT = Path("/weka/dfive-default/helios/dataset/cybench")
DEFAULT_LABELS_ROOT = DEFAULT_ROOT / "raw" / "cybench-data" / "cybench-data"
DEFAULT_POLYGONS_ROOT = DEFAULT_ROOT / "raw" / "polygons" / "polygons"

# Year-based split tags written on every window (harvest_year based). CY-Bench
# itself uses leave-one-year-out; this fixed split is only for the in-loop
# probe. All years are also tagged individually so other splits can be built
# with tag filters.
SPLIT_TRAIN_MAX_YEAR = 2017
SPLIT_VAL_YEARS = (2018, 2019)


def split_for_year(year: int) -> str:
    """Map a harvest year to the fixed train/val/test split tag."""
    if year <= SPLIT_TRAIN_MAX_YEAR:
        return "train"
    if year in SPLIT_VAL_YEARS:
        return "val"
    return "test"


# ---------------------------------------------------------------------------
# Grid helpers
# ---------------------------------------------------------------------------
def lat_to_i(lat: float) -> int:
    """Canonical row index of the cell whose center is nearest to ``lat``."""
    return int(np.clip(round((90.0 - lat) / CELL_DEG), 0, N_LAT - 1))


def lon_to_j(lon: float) -> int:
    """Canonical column index of the cell whose center is nearest to ``lon``."""
    return int(round((lon % 360.0) / CELL_DEG)) % N_LON


def i_to_lat(i: np.ndarray | int) -> np.ndarray | float:
    """Cell-center latitude for canonical row index."""
    return 90.0 - CELL_DEG * np.asarray(i, dtype=np.float64)


def j_to_lon360(j: np.ndarray | int) -> np.ndarray | float:
    """Cell-center longitude in [0, 360) for canonical column index."""
    return CELL_DEG * np.asarray(j, dtype=np.float64)


# ---------------------------------------------------------------------------
# CY-Bench labels + crop calendars
# ---------------------------------------------------------------------------
def list_crop_countries(labels_root: Path) -> list[tuple[str, str]]:
    """Return all (crop, country_code) pairs that have a yield csv."""
    pairs = []
    for yf in sorted(labels_root.glob("*/*/yield_*.csv")):
        crop = yf.parent.parent.name
        cc = yf.parent.name
        pairs.append((crop, cc))
    return pairs


def load_labels(labels_root: Path, crop: str, cc: str) -> pd.DataFrame:
    """Load the yield csv for (crop, cc) with CY-Bench's own filters applied.

    Returns columns ``adm_id, harvest_year, yield`` (yield in t/ha), after
    dropping NaNs and non-positive yields exactly like
    ``cybench.datasets.configured.load_dfs``.
    """
    path = labels_root / crop / cc / f"yield_{crop}_{cc}.csv"
    df = pd.read_csv(path)
    df = df.rename(columns={"harvest_year": "year"})[["adm_id", "year", "yield"]]
    df = df.dropna()
    df = df[df["yield"] > 0.0]
    df["year"] = df["year"].astype(int)
    df["adm_id"] = df["adm_id"].astype(str)
    return df.reset_index(drop=True)


def load_calendar(labels_root: Path, crop: str, cc: str) -> pd.DataFrame:
    """Load the crop calendar (``adm_id, sos, eos`` day-of-year ints)."""
    path = labels_root / crop / cc / f"crop_calendar_{crop}_{cc}.csv"
    if not path.exists():
        return pd.DataFrame(columns=["adm_id", "sos", "eos"])
    cal = pd.read_csv(path)[["adm_id", "sos", "eos"]].dropna()
    cal["adm_id"] = cal["adm_id"].astype(str)
    # CY-Bench rounds/clips these to [1, 366]; ESA WorldCereal calendars ship
    # fractional day-of-year values after admin-unit averaging.
    cal["sos"] = cal["sos"].round().clip(1, 366).astype(int)
    cal["eos"] = cal["eos"].round().clip(1, 366).astype(int)
    return cal.drop_duplicates("adm_id").reset_index(drop=True)


def eos_date(year: int, eos_doy: int) -> datetime:
    """End-of-season date (UTC midnight) for a harvest year and calendar DOY.

    Day-of-year 366 in a non-leap year is clamped to Dec 31.
    """
    days_in_year = 366 if _is_leap(year) else 365
    doy = min(int(eos_doy), days_in_year)
    return datetime(year, 1, 1, tzinfo=UTC) + timedelta(days=doy - 1)


def _is_leap(year: int) -> bool:
    return year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)


def window_range(year: int, eos_doy: int) -> tuple[datetime, datetime]:
    """The 448-day input window ``[eos - 448 d, eos)`` for one label."""
    end = eos_date(year, eos_doy)
    return end - timedelta(days=WINDOW_DAYS), end


def region_key(crop: str, cc: str, adm_id: str) -> str:
    """Stable region identifier used across the three stages."""
    return f"{crop}|{cc}|{adm_id}"


def safe_name(s: str) -> str:
    """Make an identifier filesystem-safe for window names."""
    return "".join(ch if (ch.isalnum() or ch in "-_.") else "_" for ch in s)


def ceil_div(a: int, b: int) -> int:
    """Integer ceiling division."""
    return -(-a // b)


def chunk_index_range(
    time_vals: np.ndarray, start: datetime, end: datetime, time_cs: int
) -> range:
    """Time-chunk indices overlapping ``[start, end)`` given the Zarr time axis."""
    tv = time_vals.astype("datetime64[ns]")
    s = np.datetime64(start.replace(tzinfo=None), "ns")
    e = np.datetime64(end.replace(tzinfo=None), "ns")
    i0 = int(np.searchsorted(tv, s, side="right") - 1)
    i1 = int(np.searchsorted(tv, e, side="left"))  # exclusive
    i0 = max(i0, 0)
    i1 = min(max(i1, i0 + 1), len(tv))
    return range(i0 // time_cs, ceil_div(i1, time_cs))


def fmt_chunk(tc: int, latc: int, lonc: int) -> str:
    """Partial-file stem for one (time, lat, lon) chunk triple."""
    return f"t{tc}_y{latc}_x{lonc}"


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance, used only for sanity logging."""
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = p2 - p1
    dl = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))
