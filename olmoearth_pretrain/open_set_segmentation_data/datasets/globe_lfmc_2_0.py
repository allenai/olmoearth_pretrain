"""Process Globe-LFMC 2.0 into a point-table regression dataset (live fuel moisture content).

Source: Globe-LFMC 2.0 (Yebra et al. 2024, Scientific Data 11:332), figshare dataset
DOI 10.6084/m9.figshare.25413790 (article 25413790, single file
``Globe-LFMC-2.0 final.xlsx``, 72 MB, CC-BY-4.0). No credential required. It is a global
compilation of >280,000 in-situ Live Fuel Moisture Content (LFMC) field measurements at
>2,000 sites in 15 countries, 1977-2023. Each row of the "LFMC data" sheet is one dated,
georeferenced destructive field sample: WGS84 lat/lon, sampling date (YYYYMMDD), species,
functional type, and the LFMC value (%) = 100 * (fresh - dry) / dry weight.

REGRESSION TARGET -- Live Fuel Moisture Content (%). Sparse dated points, so we write one
dataset-wide GeoJSON point table (points.geojson, spec 2a), NOT per-point GeoTIFFs.

Key processing decisions (see summary for full rationale):
  * LFMC is a *rapidly-varying condition*, not a static annual label: it is only valid for
    a short period around the sampling date. So each sample gets a SHORT time window
    centered on its sampling date (+/-15 days => ~1 month, via io.centered_time_range),
    NOT a static year. change_time stays null (this is a condition, not a dated change
    event / where-mask).
  * Post-2016 only (spec 8.2). Globe-LFMC spans 1977-2023; ~118k of ~294k rows are
    2016-01-01 or later. We keep only those (Sentinel era) and drop the pre-2016 majority.
  * Each dated measurement is a separate sample keyed by (site, date). A site+date often
    has several per-species measurements at identical coordinates; we aggregate them to the
    mean LFMC for that (location, day) so each sample is one (pixel, time, value). This
    yields ~51.6k site+date samples.
  * Value QC: drop physically-implausible values / sentinels. LFMC realistically sits in
    ~[10, 400] %; the raw column has clear error sentinels (up to 599999) and a heavy tail
    above 500. We keep measurements in [10, 400] % (>99% of rows) before aggregation.
  * The site+date-mean distribution is right-skewed (median ~101, tail to 400), so we
    bucket-balance across the value range (spec 5) when sampling down to the 5000-sample
    regression cap, giving even coverage of low/median/high LFMC.

Run:  python3 -m olmoearth_pretrain.open_set_segmentation_data.datasets.globe_lfmc_2_0
"""

import argparse

import numpy as np
import pandas as pd

from olmoearth_pretrain.open_set_segmentation_data import download, io, manifest
from olmoearth_pretrain.open_set_segmentation_data.sampling import (
    bucket_balance_regression,
)

SLUG = "globe_lfmc_2_0"
NAME = "Globe-LFMC 2.0"
FIGSHARE_ARTICLE = "25413790"
FIGSHARE_FILE_URL = "https://ndownloader.figshare.com/files/45049786"
XLSX_NAME = "Globe-LFMC-2.0_final.xlsx"
SHEET = "LFMC data"

MIN_YEAR = 2016  # Sentinel era; keep >= 2016-01-01 (§8.2)
LFMC_MIN, LFMC_MAX = 10.0, 400.0  # plausible LFMC (%) range; drops sentinels/outliers
MAX_REGRESSION = 5000
N_BUCKETS = 10
HALF_WINDOW_DAYS = 15  # +/-15 days around sampling date => ~1-month validity window
SEED = 42

COL_LAT = "Latitude (WGS84, EPSG:4326)"
COL_LON = "Longitude (WGS84, EPSG:4326)"
COL_DATE = "Sampling date (YYYYMMDD)"
COL_VAL = "LFMC value (%)"
COL_SITE = "Site name"
CORE_COLS = [
    "Sorting ID",
    COL_SITE,
    "Country",
    COL_LAT,
    COL_LON,
    COL_DATE,
    COL_VAL,
    "Species collected",
    "Species functional type",
    "Individual sample or mean value",
    "IGBP Land Cover",
]


def _ensure_table() -> pd.DataFrame:
    """Return the core LFMC columns as a DataFrame, caching a parquet for fast reruns.

    Downloads the figshare xlsx to raw/{slug}/ (idempotent), then extracts the needed
    columns of the "LFMC data" sheet once into lfmc_core.parquet (reading the 72 MB xlsx
    takes ~80 s, so we cache).
    """
    raw = io.raw_dir(SLUG)
    raw.mkdir(parents=True, exist_ok=True)
    xlsx = raw / XLSX_NAME
    download.download_http(FIGSHARE_FILE_URL, xlsx)
    with (raw / "SOURCE.txt").open("w") as f:
        f.write(
            f"{NAME}\n"
            f"figshare article {FIGSHARE_ARTICLE} "
            "(DOI 10.6084/m9.figshare.25413790)\n"
            f"file: {XLSX_NAME} <- {FIGSHARE_FILE_URL}\n"
            "paper: Yebra et al. 2024, Sci Data 11:332, "
            "https://doi.org/10.1038/s41597-024-03159-6\n"
            "license: CC-BY-4.0\n"
            f"regression target: Live Fuel Moisture Content (%), sheet '{SHEET}'\n"
        )
    parquet = raw / "lfmc_core.parquet"
    if parquet.exists():
        return pd.read_parquet(parquet.path)
    print(f"reading {XLSX_NAME} (slow, ~80 s) and caching parquet ...")
    df = pd.read_excel(xlsx.path, sheet_name=SHEET, usecols=CORE_COLS)
    df.to_parquet(parquet.path)
    return df


def build_site_date_records(df: pd.DataFrame) -> list[dict]:
    """Filter to post-2016 plausible measurements and aggregate to one record per
    (site, date): mean LFMC over the species sampled there that day.
    """
    lat = pd.to_numeric(df[COL_LAT], errors="coerce")
    lon = pd.to_numeric(df[COL_LON], errors="coerce")
    date = pd.to_datetime(df[COL_DATE], errors="coerce")
    val = pd.to_numeric(df[COL_VAL], errors="coerce")

    keep = (
        lat.notna()
        & lon.notna()
        & date.notna()
        & val.notna()
        & lat.between(-90, 90)
        & lon.between(-180, 180)
        & (date >= pd.Timestamp(MIN_YEAR, 1, 1))
        & (val >= LFMC_MIN)
        & (val <= LFMC_MAX)
    )
    work = pd.DataFrame(
        {
            "site": df[COL_SITE].astype(str),
            "lat": lat,
            "lon": lon,
            "date": date.dt.floor("D"),
            "val": val,
        }
    )[keep].reset_index(drop=True)

    grouped = work.groupby(["site", "date", "lat", "lon"], as_index=False).agg(
        val=("val", "mean"), n=("val", "size")
    )
    recs: list[dict] = []
    for row in grouped.itertuples(index=False):
        recs.append(
            {
                "lon": float(row.lon),
                "lat": float(row.lat),
                "value": float(row.val),
                "date": row.date.to_pydatetime(),
                "n_species": int(row.n),
                "source_id": f"{row.site}@{row.date.strftime('%Y%m%d')}",
            }
        )
    return recs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-samples", type=int, default=MAX_REGRESSION)
    parser.add_argument("--n-buckets", type=int, default=N_BUCKETS)
    parser.parse_args()

    io.check_disk()
    manifest.write_registry_entry(SLUG, "in_progress")

    df = _ensure_table()
    print(f"raw rows: {len(df)}")
    recs = build_site_date_records(df)
    print(
        f"post-2016 site+date samples (LFMC in [{LFMC_MIN},{LFMC_MAX}] %): {len(recs)}"
    )

    selected, edges = bucket_balance_regression(
        recs, "value", total=MAX_REGRESSION, n_buckets=N_BUCKETS, seed=SEED
    )
    print(f"selected {len(selected)} (bucket-balanced, {N_BUCKETS} buckets)")

    points = []
    for i, r in enumerate(selected):
        points.append(
            {
                "id": f"{i:06d}",
                "lon": r["lon"],
                "lat": r["lat"],
                "label": r["value"],
                "time_range": io.centered_time_range(r["date"], HALF_WINDOW_DAYS),
                "change_time": None,
                "source_id": r["source_id"],
            }
        )
    io.write_points_table(SLUG, "regression", points)

    vals = np.array([p["label"] for p in points], dtype=float)
    hist_counts, hist_edges = np.histogram(vals, bins=10)
    io.write_dataset_metadata(
        SLUG,
        {
            "dataset": SLUG,
            "name": NAME,
            "task_type": "regression",
            "source": "Globe-LFMC 2.0 (figshare 25413790; Yebra et al. 2024, Sci Data)",
            "license": "CC-BY-4.0",
            "provenance": {
                "url": "https://doi.org/10.6084/m9.figshare.25413790",
                "paper": "https://doi.org/10.1038/s41597-024-03159-6",
                "have_locally": False,
                "annotation_method": "in-situ field plot (destructive fresh/dry sampling)",
            },
            "sensors_relevant": ["sentinel2", "sentinel1", "landsat"],
            "regression": {
                "name": "live_fuel_moisture_content",
                "description": (
                    "Live Fuel Moisture Content of vegetation, LFMC[%] = 100 * (Wf - Wd) "
                    "/ Wd, where Wf/Wd are the fresh/oven-dry weights of a destructively "
                    "sampled field plant sample. Per (site, day) we use the mean LFMC over "
                    "the species sampled there that day. A rapidly-varying condition valid "
                    "only near its sampling date."
                ),
                "unit": "percent",
                "dtype": "float32",
                "value_range": [float(vals.min()), float(vals.max())],
                "nodata_value": io.REGRESSION_NODATA,
                "buckets": [round(float(e), 3) for e in edges],
            },
            "num_samples": len(points),
            "value_histogram": {
                "bin_edges": [float(e) for e in hist_edges],
                "counts": [int(c) for c in hist_counts],
            },
            "notes": (
                "Point-table regression (spec 2a); label = mean LFMC (%) at a (site, date). "
                "Source: Globe-LFMC 2.0 figshare xlsx 'LFMC data' sheet (293,796 rows, "
                "1977-2023). Kept only post-2016 measurements (Sentinel era, spec 8.2): "
                "~118k of ~294k rows. Value QC: kept LFMC in [10,400] % (drops error "
                "sentinels up to 599999 and an implausible tail >400%; >99% of rows kept). "
                "Aggregated per-species rows to one mean value per (site, date) -> ~51.6k "
                "site+date samples, then bucket-balanced across the value range to the "
                f"{MAX_REGRESSION}-sample regression cap ({N_BUCKETS} buckets, seed {SEED}) "
                "because the distribution is right-skewed (median ~101%, tail to 400%). "
                "TIME RANGE: LFMC is a rapidly-varying condition, so each sample gets a "
                f"SHORT +/-{HALF_WINDOW_DAYS}-day (~1-month) window centered on its sampling "
                "date rather than a static year; change_time=null (condition, not a dated "
                "change/where-mask). Coordinates are WGS84 site locations (~5 decimals, "
                "~1 m); a field sample represents plot-scale vegetation, approximately "
                "placed on the 10 m grid."
            ),
        },
    )
    manifest.write_registry_entry(
        SLUG, "completed", task_type="regression", num_samples=len(points)
    )
    print(f"done num_samples={len(points)} task_type=regression")


if __name__ == "__main__":
    main()
