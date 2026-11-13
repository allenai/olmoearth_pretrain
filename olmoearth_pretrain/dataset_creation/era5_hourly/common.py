"""Paths, bands and EarthDataHub access shared by the hourly corpus stages."""

from __future__ import annotations

import base64
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ..fluxnet.common import ERA5_MIRROR, NODATA  # noqa: F401

# The daily pretraining corpus. Read-only: windows are taken exactly as defined.
CORPUS_ROOT = Path("/weka/dfive-default/helios/dataset/era5enc_pretrain")
WINDOWS_DIR = CORPUS_ROOT / "rslearn_dataset" / "windows" / "default"
DEFAULT_ROOT = CORPUS_ROOT / "era5L_hourly_365d"

# EarthDataHub ERA5-Land stores (zarr v3). The hourly one starts 1950-01-01T00,
# covers 90N..57.1S on the 0.1 deg grid of the weka daily mirror (longitudes in
# [0, 360)) and is chunked (1440 h, 50 lat, 100 lon). Values are bit-rounded to
# 13 mantissa bits by the provider.
HOURLY_URL = "https://data.earthdatahub.destine.eu/era5/era5-land-v0.zarr"
DAILY_URL = "https://data.earthdatahub.destine.eu/era5/era5-land-daily-utc-v1.zarr"
EPOCH = np.datetime64("1950-01-01T00", "h")
N_LAT, N_LON = 1472, 3600
GRID_DEG = 0.1
TIME_CHUNK, LAT_CHUNK, LON_CHUNK = 1440, 50, 100

WINDOW_HOURS = 365 * 24
SHARD_WINDOWS = 10_000


@dataclass(frozen=True)
class Band:
    """One stored channel.

    Args:
        name: ECMWF short name of the physical quantity stored.
        source: Field read from the store. ERA5-Land's archive has three
            evaporation components swapped (ECMWF known issue, checked on the
            EarthDataHub store): ``evabs`` holds transpiration, ``evaow``
            bare-soil evaporation and ``evavt`` open-water evaporation.
        accumulated: Running total from 00 UTC in the store; stored as the
            amount over the hour ending at each timestamp.
        units: Units as stored (per hour for accumulated bands).
        description: What the band is.
    """

    name: str
    source: str
    accumulated: bool
    units: str
    description: str


# The 14 daily bands first, in ERA5L_DAY_10 order, then the 13 additions.
BANDS = (
    Band("d2m", "d2m", False, "K", "2 m dewpoint temperature"),
    Band("e", "e", True, "m of water equivalent", "total evaporation (negative = evaporation)"),
    Band("pev", "pev", True, "m", "potential evaporation"),
    Band("ro", "ro", True, "m", "runoff (= sro + ssro)"),
    Band("sp", "sp", False, "Pa", "surface pressure"),
    Band("ssr", "ssr", True, "J m-2", "surface net shortwave radiation"),
    Band("ssrd", "ssrd", True, "J m-2", "surface shortwave radiation downwards"),
    Band("str", "str", True, "J m-2", "surface net longwave radiation"),
    Band("swvl1", "swvl1", False, "m3 m-3", "volumetric soil water, 0-7 cm"),
    Band("swvl2", "swvl2", False, "m3 m-3", "volumetric soil water, 7-28 cm"),
    Band("t2m", "t2m", False, "K", "2 m temperature"),
    Band("tp", "tp", True, "m", "total precipitation"),
    Band("u10", "u10", False, "m s-1", "10 m u wind"),
    Band("v10", "v10", False, "m s-1", "10 m v wind"),
    Band("sf", "sf", True, "m of water equivalent", "snowfall"),
    Band("strd", "strd", True, "J m-2", "surface longwave radiation downwards"),
    Band("sshf", "sshf", True, "J m-2", "surface sensible heat flux"),
    Band("evavt", "evabs", True, "m of water equivalent", "transpiration (store field evabs)"),
    Band("evabs", "evaow", True, "m of water equivalent", "bare-soil evaporation (store field evaow)"),
    Band("sro", "sro", True, "m", "surface runoff"),
    Band("swvl3", "swvl3", False, "m3 m-3", "volumetric soil water, 28-100 cm"),
    Band("swvl4", "swvl4", False, "m3 m-3", "volumetric soil water, 100-289 cm"),
    Band("skt", "skt", False, "K", "skin temperature"),
    Band("stl1", "stl1", False, "K", "soil temperature, 0-7 cm"),
    Band("stl4", "stl4", False, "K", "soil temperature, 100-289 cm"),
    Band("sd", "sd", False, "m of water equivalent", "snow depth (water equivalent)"),
    Band("smlt", "smlt", True, "m of water equivalent", "snowmelt"),
)  # fmt: skip

# Time-invariant fields, from the daily store (lshf there has a time axis;
# its first step is taken).
STATIC_FIELDS = (
    "z",
    "lsm",
    "slt",
    "cvh",
    "cvl",
    "tvh",
    "tvl",
    "glm",
    "cl",
    "dl",
    "lshf",
)


def hour_index(t: np.datetime64) -> int:
    """Index of an hour on the store's time axis.

    Args:
        t: A time (floored to the hour).

    Returns:
        Hours since ``EPOCH``.
    """
    return int((np.datetime64(t, "h") - EPOCH) / np.timedelta64(1, "h"))


def grid_cell(lat: np.ndarray, lon: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The ERA5-Land cell (grid point) each location falls in.

    Args:
        lat: Latitudes (deg).
        lon: Longitudes (deg, any convention).

    Returns:
        ``(ilat, ilon)`` store indices; ``ilat >= N_LAT`` south of the store.
    """
    ilat = np.rint((90.0 - np.asarray(lat)) / GRID_DEG).astype(np.int64)
    ilon = np.rint((np.asarray(lon) % 360.0) / GRID_DEG).astype(np.int64) % N_LON
    return ilat, ilon


def deaccumulate(totals: np.ndarray, first_hour: int) -> np.ndarray:
    """Hourly amounts from ERA5-Land running totals.

    The store holds totals accumulated from 00 UTC: the 01 UTC value is the
    first hour's amount and the 00 UTC value closes the previous day.

    Args:
        totals: ``(T, ...)`` totals at consecutive hours.
        first_hour: Store hour index of ``totals[0]``.

    Returns:
        ``(T, ...)`` amount over the hour ending at each step; step 0 is NaN
        unless it is 01 UTC (its previous total is not in ``totals``).
    """
    out = np.empty_like(totals)
    out[0] = np.nan
    out[1:] = totals[1:] - totals[:-1]
    restart = (first_hour + np.arange(len(totals))) % 24 == 1
    out[restart] = totals[restart]
    return out


def edh_storage_options() -> dict[str, Any]:
    """Storage options carrying the EarthDataHub token (``EARTHDATAHUB_TOKEN``).

    Returns:
        Storage options with a basic-auth header, as rslearn's EarthDataHub
        source builds it.
    """
    token = os.environ.get("EARTHDATAHUB_TOKEN")
    if not token:
        raise RuntimeError("EARTHDATAHUB_TOKEN is not set")
    credentials = base64.b64encode(f"edh:{token}".encode()).decode()
    return {"headers": {"Authorization": f"Basic {credentials}"}}


def open_store(url: str = HOURLY_URL) -> Any:
    """Open an EarthDataHub zarr store read-only.

    Args:
        url: Store URL.

    Returns:
        The root ``zarr.Group``.
    """
    import zarr  # not in the lock; jobs install it

    store = zarr.storage.FsspecStore.from_url(
        url, storage_options=edh_storage_options(), read_only=True
    )
    return zarr.open_group(store, mode="r")
