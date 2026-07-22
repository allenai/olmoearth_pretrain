"""Process GLORIA into a point-table regression dataset (in-situ water quality).

Source: GLORIA -- "A global dataset of remote sensing reflectance and water quality from
inland and coastal waters" (Lehmann et al. 2023, Scientific Data), PANGAEA
DOI 10.1594/PANGAEA.948492, single archive ``GLORIA-2022.zip`` (~59 MB, CC-BY-4.0). No
credential required (PANGAEA HTTP; a Firefox User-Agent avoids the UA-fingerprint block).
GLORIA is 7,572 curated in-situ hyperspectral remote-sensing-reflectance measurements from
450 water bodies worldwide, each with at least one co-located water-quality measurement:
chlorophyll-a (Chla), total suspended solids (TSS), CDOM absorption at 440 nm (aCDOM440),
and Secchi disk depth. We use only the ``GLORIA_meta_and_lab.csv`` table (coords + dates +
lab values); the bulky per-wavelength radiometry CSVs are not needed for the label signal.

REGRESSION TARGET -- chlorophyll-a concentration (Chla, mg m-3). Chla is chosen as the
PRIMARY target because it is the most standard water-quality variable retrievable from
S2/S1/Landsat water color AND the best-populated post-2016 column here (1,635 usable points
vs 1,530 TSS / 1,568 Secchi / 980 aCDOM440). TSS, aCDOM440, and Secchi_depth are carried as
AUXILIARY per-point properties (present where measured), plus water-body type / country /
turbidity context. Sparse dated point measurements -> one dataset-wide GeoJSON point table
(points.geojson, spec 2a), NOT per-point GeoTIFFs.

Key processing decisions (see summary for full rationale):
  * Post-2016 only (spec 8.2). GLORIA spans 1990-2022; ~2,411 of 7,572 rows are 2016-01-01
    or later. We keep only those (Sentinel era) with valid WGS84 lat/lon and a non-null
    Chla value -> 1,635 samples.
  * Chla is an instantaneous match-up quantity tied to the acquisition date and varies on
    days-to-weeks timescales (algal blooms), so each sample gets a SHORT time window centered
    on its measurement date (+/-15 days => ~1 month, via io.centered_time_range), NOT a
    static year. change_time stays null (a water-column state at a time, not a dated change /
    where-mask).
  * Value QC: GLORIA is quality-controlled; the post-2016 Chla column has no zeros, negatives
    or sentinel values (range 0.05-659 mg m-3), so no extra value filtering is applied.
  * The Chla distribution is strongly right-skewed / roughly log-distributed (median ~6.8,
    tail to 659 mg m-3). At 1,635 points it is already well under the 5,000-sample regression
    cap, so we keep ALL of it rather than downsampling; decile bucket edges of the value
    distribution are recorded in metadata.json for reference (spec 5).

Run:  python3 -m olmoearth_pretrain.open_set_segmentation_data.datasets.gloria_global_hyperspectral_water_quality
"""

import argparse
import zipfile

import numpy as np
import pandas as pd

from olmoearth_pretrain.open_set_segmentation_data import download, io, manifest

SLUG = "gloria_global_hyperspectral_water_quality"
NAME = "GLORIA (Global hyperspectral water quality)"
ZIP_URL = "https://download.pangaea.de/dataset/948492/files/GLORIA-2022.zip"
ZIP_NAME = "GLORIA-2022.zip"
META_MEMBER = "GLORIA_2022/GLORIA_meta_and_lab.csv"
META_NAME = "GLORIA_meta_and_lab.csv"
# Firefox UA: PANGAEA blocks generic urllib User-Agents (UA-fingerprint 403).
UA = "Mozilla/5.0 (X11; Linux x86_64; rv:128.0) Gecko/20100101 Firefox/128.0"

MIN_YEAR = 2016  # Sentinel era; keep >= 2016-01-01 (spec 8.2)
MAX_REGRESSION = 5000
N_BUCKETS = 10
HALF_WINDOW_DAYS = 15  # +/-15 days around measurement date => ~1-month match-up window
SEED = 42


def _ensure_meta_csv() -> str:
    """Download the GLORIA zip (idempotent) and extract only the meta+lab CSV. Returns path."""
    raw = io.raw_dir(SLUG)
    raw.mkdir(parents=True, exist_ok=True)
    zip_path = raw / ZIP_NAME
    download.download_http(ZIP_URL, zip_path, headers={"User-Agent": UA})
    with (raw / "SOURCE.txt").open("w") as f:
        f.write(
            f"{NAME}\n"
            "PANGAEA DOI 10.1594/PANGAEA.948492\n"
            f"file: {ZIP_NAME} <- {ZIP_URL}\n"
            "paper: Lehmann et al. 2023, Sci Data 10:100, "
            "https://doi.org/10.1038/s41597-023-01973-y\n"
            "license: CC-BY-4.0\n"
            f"labels extracted from member: {META_MEMBER}\n"
            "regression target: chlorophyll-a (Chla, mg m-3); aux: TSS, aCDOM440, Secchi_depth\n"
        )
    csv_path = raw / META_NAME
    if not csv_path.exists():
        with zipfile.ZipFile(zip_path.path) as zf, zf.open(META_MEMBER) as src:
            data = src.read()
        tmp = raw / (META_NAME + ".tmp")
        with tmp.open("wb") as f:
            f.write(data)
        tmp.rename(csv_path)
    return csv_path.path


def build_records(csv_path: str) -> list[dict]:
    """Filter to post-2016 valid-coord rows with a non-null Chla value; one record each."""
    df = pd.read_csv(csv_path, low_memory=False)
    lat = pd.to_numeric(df["Latitude"], errors="coerce")
    lon = pd.to_numeric(df["Longitude"], errors="coerce")
    dt = pd.to_datetime(df["Date_Time_UTC"], errors="coerce", utc=True)
    chla = pd.to_numeric(df["Chla"], errors="coerce")
    tss = pd.to_numeric(df["TSS"], errors="coerce")
    cdom = pd.to_numeric(df["aCDOM440"], errors="coerce")
    secchi = pd.to_numeric(df["Secchi_depth"], errors="coerce")
    turb = pd.to_numeric(df["Turbidity"], errors="coerce")

    keep = (
        lat.notna()
        & lon.notna()
        & lat.between(-90, 90)
        & lon.between(-180, 180)
        & dt.notna()
        & (dt >= pd.Timestamp(MIN_YEAR, 1, 1, tz="UTC"))
        & chla.notna()
        & (chla > 0)
    )
    recs: list[dict] = []
    for i in df.index[keep]:
        rec = {
            "lon": float(lon[i]),
            "lat": float(lat[i]),
            "value": float(chla[i]),
            "date": dt[i].to_pydatetime(),
            "source_id": str(df.at[i, "GLORIA_ID"]),
            "water_body_type": _clean(df.at[i, "Water_body_type"]),
            "country": _clean(df.at[i, "Country"]),
        }
        # Auxiliary regression targets, present where measured.
        if pd.notna(tss[i]):
            rec["tss"] = float(tss[i])
        if pd.notna(cdom[i]):
            rec["acdom440"] = float(cdom[i])
        if pd.notna(secchi[i]):
            rec["secchi_depth"] = float(secchi[i])
        if pd.notna(turb[i]):
            rec["turbidity"] = float(turb[i])
        recs.append(rec)
    return recs


def _clean(v: object) -> object:
    """JSON-safe scalar: NaN -> None, else str/int as-is."""
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return None
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        return int(v) if float(v).is_integer() else float(v)
    return str(v)


def main() -> None:
    argparse.ArgumentParser().parse_args()

    io.check_disk()
    manifest.write_registry_entry(SLUG, "in_progress")

    csv_path = _ensure_meta_csv()
    recs = build_records(csv_path)
    print(f"post-{MIN_YEAR} valid-coord Chla samples: {len(recs)}")

    # 1,635 points are already under the 5,000-sample regression cap, so keep all (no
    # downsampling). Record decile bucket edges of the (skewed) value distribution.
    recs.sort(key=lambda r: r["source_id"])  # deterministic id assignment
    vals = np.array([r["value"] for r in recs], dtype=float)
    edges = list(np.quantile(vals, np.linspace(0, 1, N_BUCKETS + 1)))

    points = []
    for i, r in enumerate(recs):
        p = {
            "id": f"{i:06d}",
            "lon": r["lon"],
            "lat": r["lat"],
            "label": r["value"],
            "time_range": io.centered_time_range(r["date"], HALF_WINDOW_DAYS),
            "change_time": None,
            "source_id": r["source_id"],
            "water_body_type": r["water_body_type"],
            "country": r["country"],
        }
        for k in ("tss", "acdom440", "secchi_depth", "turbidity"):
            if k in r:
                p[k] = r[k]
        points.append(p)
    io.write_points_table(SLUG, "regression", points)

    hist_counts, hist_edges = np.histogram(vals, bins=10)
    io.write_dataset_metadata(
        SLUG,
        {
            "dataset": SLUG,
            "name": NAME,
            "task_type": "regression",
            "source": "GLORIA (PANGAEA 948492; Lehmann et al. 2023, Sci Data)",
            "license": "CC-BY-4.0",
            "provenance": {
                "url": "https://doi.org/10.1594/PANGAEA.948492",
                "paper": "https://doi.org/10.1038/s41597-023-01973-y",
                "have_locally": False,
                "annotation_method": "in-situ field sampling + laboratory water-quality analysis",
            },
            "sensors_relevant": ["sentinel2", "landsat"],
            "regression": {
                "name": "chlorophyll_a",
                "description": (
                    "Chlorophyll-a concentration (Chla) of the surface water, measured "
                    "in-situ by laboratory analysis of collected water samples (spectrophotometric "
                    "/ HPLC / fluorometric methods vary by contributor). The primary optically "
                    "active phytoplankton pigment and the standard water-quality / trophic-state "
                    "indicator retrievable from water color. Instantaneous value at the sample "
                    "location and date."
                ),
                "unit": "mg m-3",
                "dtype": "float32",
                "value_range": [float(vals.min()), float(vals.max())],
                "nodata_value": io.REGRESSION_NODATA,
                "buckets": [round(float(e), 4) for e in edges],
            },
            "auxiliary_fields": {
                "tss": "Total suspended solids (g m-3), where measured (GLORIA 'TSS').",
                "acdom440": "CDOM absorption coefficient at 440 nm (m-1), where measured "
                "(GLORIA 'aCDOM440').",
                "secchi_depth": "Secchi disk depth / water clarity (m), where measured "
                "(GLORIA 'Secchi_depth').",
                "turbidity": "Turbidity (NTU), where measured (GLORIA 'Turbidity').",
                "water_body_type": "GLORIA water-body-type code (1=lake/reservoir, 3=river, "
                "4=estuary, 5=coastal/ocean, ...).",
                "country": "Country of the sampling site.",
            },
            "num_samples": len(points),
            "value_histogram": {
                "bin_edges": [float(e) for e in hist_edges],
                "counts": [int(c) for c in hist_counts],
            },
            "notes": (
                "Point-table regression (spec 2a); label = in-situ chlorophyll-a (mg m-3) at a "
                "sampling location/date. Source: GLORIA 'GLORIA_meta_and_lab.csv' (7,572 rows, "
                "1990-2022). Kept only post-2016 rows (Sentinel era, spec 8.2) with valid WGS84 "
                "lat/lon and a non-null Chla value -> 1,635 samples (of 2,411 post-2016 rows). "
                "No extra value QC needed (curated; no zeros/sentinels; range 0.05-659 mg m-3). "
                "PRIMARY target Chla chosen as the most standard water-color quantity and the "
                "best-populated post-2016 column; TSS / aCDOM440 / Secchi_depth carried as "
                "auxiliary per-point fields where measured (present in ~1140 / ~675 / ~1295 of "
                "the 1,635 points). Distribution is strongly right-skewed (median ~6.8, tail to "
                "659); at 1,635 < 5,000-sample cap we keep all points (no bucket downsampling) "
                "and record decile bucket edges for reference. TIME RANGE: Chla is an "
                f"instantaneous match-up quantity, so each sample gets a SHORT +/-{HALF_WINDOW_DAYS}"
                "-day (~1-month) window centered on its measurement date rather than a static "
                "year; change_time=null (a water-column state, not a dated change/where-mask). "
                "Weak label for 10-30 m water color: a single-pixel surface-water point "
                "measurement, not a full-water-body segmentation."
            ),
        },
    )
    manifest.write_registry_entry(
        SLUG, "completed", task_type="regression", num_samples=len(points)
    )
    print(f"done num_samples={len(points)} task_type=regression")


if __name__ == "__main__":
    main()
