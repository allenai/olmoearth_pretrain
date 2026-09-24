"""Shared constants and helpers for the CY-Bench ERA5-Land export.

Pipeline (see README.md): ``build_weights`` -> ``aggregate_era5`` ->
``write_windows`` -> ``tag_eval_subset``. Everything is keyed on a canonical
ERA5-Land 0.1 deg grid (cell centers at lat = 90 - 0.1 i, lon = 0.1 j with lon
in [0, 360)); ``aggregate_era5`` maps that grid onto the Zarr coordinates at
runtime.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

# ERA5-Land band order: must match Modality.ERA5L_DAY_10 and the pretraining
# layer so the eval loader's band check passes.
ERA5L_BANDS: list[str] = [
    "d2m", "e", "pev", "ro", "sp", "ssr", "ssrd", "str",
    "swvl1", "swvl2", "t2m", "tp", "u10", "v10",
]  # fmt: skip
NODATA = -9999.0
WINDOW_DAYS = 448

# Canonical ERA5-Land 0.1 deg grid and the 0.01 deg sub-grid used to
# rasterize polygons (10 x 10 sub-pixels per cell).
CELL_DEG = 0.1
N_LAT = 1801  # 90 .. -90
N_LON = 3600  # 0 .. 359.9
FINE_PER_CELL = 10
FINE_DEG = CELL_DEG / FINE_PER_CELL

# Layer names in the output rslearn dataset.
ERA5_LAYER = "era5_daily"
LABEL_LAYER = "label"

# CY-Bench crop -> WorldCereal area-fraction image
# (AgML-CY-Bench repo, data_preparation/global_crop_AFIs_ESA_WC/).
CROP_TO_AFI = {
    "maize": "crop_mask_maize_WC.tif",
    "wheat": "crop_mask_winter_spring_cereals_WC.tif",
}

# Weka layout.
DEFAULT_ROOT = Path("/weka/dfive-default/helios/dataset/cybench")
DEFAULT_LABELS_ROOT = DEFAULT_ROOT / "raw" / "cybench-data" / "cybench-data"
DEFAULT_POLYGONS_ROOT = DEFAULT_ROOT / "raw" / "polygons" / "polygons"

# Fixed harvest-year split written on every window (CY-Bench itself is
# leave-one-year-out; this is only for the in-loop probe).
SPLIT_TRAIN_MAX_YEAR = 2017
SPLIT_VAL_YEARS = (2018, 2019)


def split_for_year(year: int) -> str:
    """Map a harvest year to the fixed train/val/test split tag."""
    if year <= SPLIT_TRAIN_MAX_YEAR:
        return "train"
    if year in SPLIT_VAL_YEARS:
        return "val"
    return "test"


# --- canonical grid -------------------------------------------------------
def i_to_lat(i: np.ndarray | int) -> np.ndarray | float:
    """Cell-center latitude for canonical row index."""
    return 90.0 - CELL_DEG * np.asarray(i, dtype=np.float64)


def j_to_lon360(j: np.ndarray | int) -> np.ndarray | float:
    """Cell-center longitude in [0, 360) for canonical column index."""
    return CELL_DEG * np.asarray(j, dtype=np.float64)


# --- CY-Bench labels and crop calendars ----------------------------------
def list_crop_countries(labels_root: Path) -> list[tuple[str, str]]:
    """All (crop, country_code) pairs that have a yield csv."""
    return [
        (yf.parent.parent.name, yf.parent.name)
        for yf in sorted(labels_root.glob("*/*/yield_*.csv"))
    ]


def load_labels(labels_root: Path, crop: str, cc: str) -> pd.DataFrame:
    """Yield labels for (crop, cc) with CY-Bench's own filters applied.

    Columns ``adm_id, year, yield`` (t/ha); NaNs and non-positive yields are
    dropped exactly like ``cybench.datasets.configured.load_dfs``.
    """
    path = labels_root / crop / cc / f"yield_{crop}_{cc}.csv"
    df = pd.read_csv(path).rename(columns={"harvest_year": "year"})
    df = df[["adm_id", "year", "yield"]].dropna()
    df = df[df["yield"] > 0.0]
    df["year"] = df["year"].astype(int)
    df["adm_id"] = df["adm_id"].astype(str)
    return df.reset_index(drop=True)


def load_calendar(labels_root: Path, crop: str, cc: str) -> pd.DataFrame:
    """Crop calendar for (crop, cc): ``adm_id, sos, eos`` as integer day-of-year.

    WorldCereal calendars ship fractional DOYs after admin-unit averaging;
    CY-Bench rounds and clips them to [1, 366], so we do the same.
    """
    path = labels_root / crop / cc / f"crop_calendar_{crop}_{cc}.csv"
    if not path.exists():
        return pd.DataFrame(columns=["adm_id", "sos", "eos"])
    cal = pd.read_csv(path)[["adm_id", "sos", "eos"]].dropna()
    cal["adm_id"] = cal["adm_id"].astype(str)
    for col in ("sos", "eos"):
        cal[col] = cal[col].round().clip(1, 366).astype(int)
    return cal.drop_duplicates("adm_id").reset_index(drop=True)


def window_range(year: int, eos_doy: int) -> tuple[datetime, datetime]:
    """The 448-day input window ``[eos - 448 d, eos)`` for one label.

    ``eos`` is the calendar end-of-season DOY placed in the harvest year
    (DOY 366 in a non-leap year is clamped to Dec 31).
    """
    days_in_year = 366 if _is_leap(year) else 365
    end = datetime(year, 1, 1, tzinfo=UTC) + timedelta(
        days=min(eos_doy, days_in_year) - 1
    )
    return end - timedelta(days=WINDOW_DAYS), end


def _is_leap(year: int) -> bool:
    return year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)


# --- naming ---------------------------------------------------------------
def region_key(crop: str, cc: str, adm_id: str) -> str:
    """Stable region identifier shared by all stages."""
    return f"{crop}|{cc}|{adm_id}"


def window_name(crop: str, cc: str, adm_id: str, year: int) -> str:
    """Rslearn window name for one label (filesystem-safe)."""
    raw = f"{crop}_{cc}_{adm_id}_{year}"
    return "".join(ch if (ch.isalnum() or ch in "-_.") else "_" for ch in raw)


def partial_name(tc: int, latc: int, lonc: int) -> str:
    """File stem of the aggregation partial for one Zarr chunk."""
    return f"t{tc}_y{latc}_x{lonc}"
