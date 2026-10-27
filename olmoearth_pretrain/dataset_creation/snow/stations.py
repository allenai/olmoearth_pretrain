"""Independent check of the MODIS melt-out labels against GHCN-Daily snow depth.

GHCN-Daily (NOAA, public on AWS: ``noaa-ghcn-pds``) reports daily snow depth
(``SNWD``, mm) at thousands of stations. For every station inside a cell of the
phenology table with ``SNWD`` over at least ``MIN_STATION_YEARS`` of the MODIS
water years: per water year (the phenology table's hemisphere convention), the
station's snow disappearance date is the day after the last day of its longest
run of days with at least ``SNOW_MM`` of snow (gaps of up to ``MAX_GAP_DAYS``
unreported days bridged; the year needs ``MIN_COVERAGE`` of its days
reported), and its anomaly is against the station's other years
(leave-one-year-out, as for the cells).

A station measures one point and a cell's SDD is its median pixel's, so absolute
dates differ (elevation, canopy, wind drift); the anomalies should agree when
both track the same season. The comparison table
(``meta/stations_vs_modis.parquet``: station x water year with both SDDs and
anomalies) and a summary (``meta/stations_summary.json``: pooled and
per-region correlations, per-station median correlation) give the eval's label
noise against ground truth.

Usage::

    python -m olmoearth_pretrain.dataset_creation.snow.stations --workers 32
"""

from __future__ import annotations

import argparse
import io
import json
import logging
import multiprocessing
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests

from .common import DEFAULT_ROOT
from .phenology import MIN_CLIM_YEARS, water_years
from .write_windows import region_of

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

GHCN_ROOT = "https://noaa-ghcn-pds.s3.amazonaws.com"
SNOW_MM = 10
MAX_GAP_DAYS = 5
MIN_COVERAGE = 0.8
MIN_STATION_YEARS = 10
FIRST_YEAR, LAST_YEAR = 2000, 2026


def inventory() -> pd.DataFrame:
    """Stations reporting ``SNWD`` over the MODIS years.

    Returns:
        ``station, lat, lon, first, last``.
    """
    text = requests.get(f"{GHCN_ROOT}/ghcnd-inventory.txt", timeout=300).text
    frame = pd.read_fwf(
        io.StringIO(text), colspecs=[(0, 11), (12, 20), (21, 30), (31, 35), (36, 40), (41, 45)],
        names=["station", "lat", "lon", "element", "first", "last"],
    )  # fmt: skip
    frame = frame[frame["element"] == "SNWD"]
    span = (
        frame["last"].clip(upper=LAST_YEAR) - frame["first"].clip(lower=FIRST_YEAR) + 1
    )
    return frame[span >= MIN_STATION_YEARS].drop(columns="element")


def station_years(job: tuple[str, bool, str]) -> list[dict[str, Any]]:
    """Worker: per water year SDD of one station.

    Args:
        job: ``(station id, northern hemisphere, cache dir)``.

    Returns:
        One row per water year with enough data.
    """
    station, north, cache_dir = job
    cache = Path(cache_dir) / f"{station}.parquet"
    if cache.exists():
        snwd = pd.read_parquet(cache)
    else:
        try:
            response = requests.get(
                f"{GHCN_ROOT}/csv/by_station/{station}.csv", timeout=300
            )
            response.raise_for_status()
        except requests.RequestException as exc:
            logger.warning("%s: %s", station, type(exc).__name__)
            return []
        frame = pd.read_csv(io.StringIO(response.text), usecols=["DATE", "ELEMENT", "DATA_VALUE", "Q_FLAG"],
                            dtype={"DATE": str, "Q_FLAG": str})  # fmt: skip
        frame = frame[(frame["ELEMENT"] == "SNWD") & frame["Q_FLAG"].isna()]
        snwd = pd.DataFrame({
            "day": (pd.to_datetime(frame["DATE"], format="%Y%m%d")
                    - pd.Timestamp("1970-01-01")).dt.days.astype(np.int32),
            "mm": frame["DATA_VALUE"].astype(np.float32),
        })  # fmt: skip
        snwd = snwd[snwd["day"] >= (pd.Timestamp(f"{FIRST_YEAR - 1}-01-01")
                                    - pd.Timestamp("1970-01-01")).days]  # fmt: skip
        snwd.to_parquet(cache)
    depth = dict(zip(snwd["day"].to_numpy(), snwd["mm"].to_numpy()))
    rows = []
    for wy, first, end in water_years(north):
        days = np.arange(first, end)
        values = np.array([depth.get(int(d), np.nan) for d in days])
        reported = np.isfinite(values)
        if reported.mean() < MIN_COVERAGE:
            continue
        snowy = values >= SNOW_MM
        # Longest run of snow days, bridging short reporting gaps.
        best, best_end, start, last_snow, gap = 0, -1, -1, -1, 0
        for k in range(len(days)):
            if snowy[k]:
                if start < 0 or gap > MAX_GAP_DAYS:
                    start = k
                last_snow, gap = k, 0
                if last_snow - start + 1 > best:
                    best, best_end = last_snow - start + 1, last_snow
            elif reported[k]:
                start, gap = -1, 0
            else:
                gap += 1
        if best < 30 or best_end == len(days) - 1:
            continue
        rows.append(dict(station=station, wy=wy, sdd=float(days[best_end] + 1), run_days=best,
                         sdd_dowy=float(best_end + 2)))  # fmt: skip
    return rows


def agreement(frame: pd.DataFrame) -> dict[str, float]:
    """Agreement of station and cell anomalies.

    Args:
        frame: Matched station x water year rows.

    Returns:
        ``n``, Pearson ``r`` and RMSE of the anomalies, their SDs, and the mean
        SDD difference (station minus cell, days).
    """
    a, b = frame["sdd_dowy_anom_stn"], frame["sdd_dowy_anom_modis"]
    return dict(n=len(frame), r=float(np.corrcoef(a, b)[0, 1]),
                rmse=float(np.sqrt(((a - b) ** 2).mean())),
                sd_stn=float(a.std()), sd_modis=float(b.std()),
                bias=float((frame["sdd_dowy_stn"] - frame["sdd_dowy_modis"]).mean()))  # fmt: skip


def run(root: Path, workers: int) -> None:
    """Station SDDs, their anomalies, and the comparison with the MODIS cells.

    Args:
        root: Dataset root.
        workers: Download processes.
    """
    meta = root / "meta"
    cache_dir = root / "raw" / "ghcnd"
    cache_dir.mkdir(parents=True, exist_ok=True)
    phen = pd.read_parquet(meta / "phenology.parquet",
                           columns=["cell", "lat10", "lon10", "wy", "sdd", "sdd_dowy", "sdd_dowy_anom", "sdd_gap"])  # fmt: skip
    stations = inventory()
    lat10 = np.rint(stations["lat"].to_numpy() * 10).astype(int)
    lon10 = np.rint(stations["lon"].to_numpy() * 10).astype(int)
    lon10 = np.where(lon10 >= 1800, lon10 - 3600, lon10)
    stations["cell"] = [f"c{a:+05d}{b:+06d}" for a, b in zip(lat10, lon10)]
    stations = stations[stations["cell"].isin(set(phen["cell"]))]
    logger.info("%d SNWD stations inside seasonal-snow cells", len(stations))
    jobs = [
        (s, bool(lat > 0), str(cache_dir))
        for s, lat in zip(stations["station"], stations["lat"])
    ]
    rows: list[dict[str, Any]] = []
    with multiprocessing.get_context("spawn").Pool(workers) as pool:
        for k, part in enumerate(pool.imap_unordered(station_years, jobs), start=1):
            rows += part
            if k % 500 == 0:
                logger.info("[%d/%d stations]", k, len(jobs))
    years = pd.DataFrame(rows)
    defined = years.groupby("station")["sdd_dowy"].transform("count")
    total = years.groupby("station")["sdd_dowy"].transform("sum")
    others = defined - 1
    years["sdd_dowy_clim"] = ((total - years["sdd_dowy"]) / others).where(
        others >= MIN_CLIM_YEARS
    )
    years["sdd_dowy_anom"] = years["sdd_dowy"] - years["sdd_dowy_clim"]
    both = years.merge(stations[["station", "cell", "lat", "lon"]], on="station").merge(
        phen, on=["cell", "wy"], suffixes=("_stn", "_modis")
    )
    both = both[both["sdd_dowy_anom_stn"].notna() & both["sdd_dowy_anom_modis"].notna()]
    both["region"] = [region_of(a, b) for a, b in zip(both["lat"], both["lon"])]
    both.to_parquet(meta / "stations_vs_modis.parquet")
    per_station = both.groupby("station").filter(lambda f: len(f) >= MIN_CLIM_YEARS)
    station_r = per_station.groupby("station").apply(
        lambda f: np.corrcoef(f["sdd_dowy_anom_stn"], f["sdd_dowy_anom_modis"])[0, 1]
    )
    summary = dict(
        pooled=agreement(both), by_region={r: agreement(f) for r, f in both.groupby("region")},
        stations=int(both["station"].nunique()), cells=int(both["cell"].nunique()),
        per_station_r_median=float(station_r.median()), per_station_n=int(len(station_r)),
    )  # fmt: skip
    (meta / "stations_summary.json").write_text(json.dumps(summary, indent=2))
    logger.info("stations vs MODIS: %s", json.dumps(summary["pooled"]))


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--root", default=str(DEFAULT_ROOT))
    ap.add_argument("--workers", type=int, default=32)
    args = ap.parse_args()
    run(Path(args.root), args.workers)


if __name__ == "__main__":
    main()
