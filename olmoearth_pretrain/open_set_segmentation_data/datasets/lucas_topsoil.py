"""Process LUCAS 2018 Topsoil into a point-table regression dataset (topsoil SOC stock).

Source: the LUCAS (Land Use/Cover Area frame Survey) Soil Module. The authoritative
LUCAS 2018 TOPSOIL laboratory dataset (18,984 in-situ samples with pH, organic carbon
content, CaCO3, N, P, K, EC, oxalate Fe/Al, bulk density, coarse fragments) is
distributed by EC JRC / ESDAC and is **registration-gated**
(https://esdac.jrc.ec.europa.eu/content/lucas-2018-topsoil-data). No ESDAC credential is
present in .env, so the raw multi-property CSV is not directly
reachable.

Per the task's instruction to first check for an OPEN mirror, we use the CC-BY-4.0 Zenodo
release of Chen et al. (2024), "European soil bulk density and organic carbon stock
database using LUCAS Soil 2018" (Zenodo record 10211884, DOI 10.5281/zenodo.10211884;
ESSD 16:2367-2383, doi:10.5194/essd-16-2367-2024). It republishes LUCAS Soil 2018 topsoil
(0-20 cm) at the in-situ GPS sampling locations with:
    POINTID, Bdfine (g cm-3), SOCS (kg m-2)*, coarse_vol, GPS_LAT, GPS_LONG, BDfine method
(* the CSV header prints "kg cm-2" which is a typo; the values, 0.4-62, are kg m-2 =
10x Mg ha-1 for the 0-20 cm layer.) 15,389 points carry a soil organic carbon stock
(SOCS) value with real field GPS coordinates. SOC stock = measured LUCAS topsoil organic
carbon content x fine-earth bulk density x 0.2 m x (1 - coarse fragment volume); the bulk
density is measured for 5,163 points and locally-predicted (random forest PTF) for the
remaining 10,226.

REGRESSION TARGET -- topsoil soil organic carbon stock (SOCS), 0-20 cm, kg m-2.
The task recommends soil organic carbon as the primary target; the open mirror provides
the closely-related, policy-relevant SOC *stock* (0-20 cm) rather than the raw OC content
in g/kg (which stays behind ESDAC registration). SOC stock is a clean, bounded, widely
EO-modelled continuous soil-carbon quantity, so we regress it directly. Fine-earth bulk
density and coarse-fragment volume fraction are carried alongside as auxiliary point
properties. The other LUCAS properties (pH, N/P/K, CaCO3, CEC, texture) require ESDAC
registration and are noted as available-on-request in the summary.

Each label is a single point with a continuous value -> REGRESSION written to a
dataset-wide point table (points.geojson, spec 2a), NOT per-point GeoTIFFs.

Time range: LUCAS Soil 2018 was surveyed Apr-Oct 2018 (Sentinel-2 era). Topsoil SOC stock
is a quasi-static property, so per spec 5 (static labels) we anchor a representative
1-year window on the survey year (2018) for every point.

The SOC-stock distribution is strongly right-skewed (median ~4.1, mean ~5.4, max ~62
kg m-2, with a long organic/peat-soil tail), so we bucket-balance across the value range
(spec 5) when sampling down to the 5000-sample regression cap, giving even coverage of the
full carbon range.

Run:
  python3 -m olmoearth_pretrain.open_set_segmentation_data.datasets.lucas_topsoil
"""

import argparse

import numpy as np
import pandas as pd

from olmoearth_pretrain.open_set_segmentation_data import download, io, manifest
from olmoearth_pretrain.open_set_segmentation_data.sampling import (
    bucket_balance_regression,
)

SLUG = "lucas_topsoil"
NAME = "LUCAS Topsoil"

# Open CC-BY-4.0 mirror of LUCAS Soil 2018 topsoil (Chen et al. 2024).
ZENODO_RECORD = "10211884"
PRIMARY_CSV = "LUCAS SOIL 2018 BD SOCS Local-RFFRFS.csv"

# CSV columns (stripped of trailing spaces).
COL_POINTID = "POINTID"
COL_BD = "Bdfine (g cm-3)"
COL_SOCS = "SOCS (kg cm-2)"  # header typo; values are kg m-2 (0-20 cm)
COL_COARSE = "coarse_vol"
COL_LAT = "GPS_LAT"
COL_LON = "GPS_LONG"
COL_BDMETHOD = "BDfine  method"

SURVEY_YEAR = 2018  # LUCAS Soil 2018 surveyed Apr-Oct 2018
MAX_REGRESSION = 5000
N_BUCKETS = 10
SEED = 42

# EU / study-area bounds sanity gate (LUCAS covers the EU + a few neighbours).
LON_MIN, LON_MAX = -32.0, 45.0
LAT_MIN, LAT_MAX = 27.0, 72.0


def load_socs() -> pd.DataFrame:
    """Return quality-filtered LUCAS 2018 topsoil SOC-stock point records."""
    csv = io.raw_dir(SLUG) / PRIMARY_CSV
    df = pd.read_csv(csv.path)
    df.columns = [c.strip() for c in df.columns]
    for c in df.columns:
        if df[c].dtype == object:
            df[c] = df[c].astype(str).str.strip()
    df["socs"] = pd.to_numeric(df[COL_SOCS], errors="coerce")
    df["bd"] = pd.to_numeric(df[COL_BD], errors="coerce")
    df["coarse"] = pd.to_numeric(df[COL_COARSE], errors="coerce")
    df["lat"] = pd.to_numeric(df[COL_LAT], errors="coerce")
    df["lon"] = pd.to_numeric(df[COL_LON], errors="coerce")
    df["bd_method"] = df[COL_BDMETHOD]
    df = df.dropna(subset=["socs", "lat", "lon"])
    # Positive, physically-plausible SOC stock; valid EU coordinates.
    df = df[df["socs"] > 0]
    df = df[df["lon"].between(LON_MIN, LON_MAX) & df["lat"].between(LAT_MIN, LAT_MAX)]
    df = df[(df["lon"] != 0) | (df["lat"] != 0)]
    return df.reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-samples", type=int, default=MAX_REGRESSION)
    parser.add_argument("--n-buckets", type=int, default=N_BUCKETS)
    args = parser.parse_args()

    io.check_disk()
    manifest.write_registry_entry(SLUG, "in_progress")

    raw = io.raw_dir(SLUG)
    raw.mkdir(parents=True, exist_ok=True)
    download.download_zenodo(ZENODO_RECORD, raw)
    with (raw / "SOURCE.txt").open("w") as f:
        f.write(
            "LUCAS 2018 Topsoil (LUCAS Soil Module).\n"
            "Authoritative raw multi-property lab CSV: EC JRC / ESDAC, registration-gated:\n"
            "  https://esdac.jrc.ec.europa.eu/content/lucas-2018-topsoil-data\n"
            "  (DOI 10.2905/JRC.J2EXD50) -- NOT used (no ESDAC credential in .env).\n"
            "OPEN mirror used here (CC-BY-4.0): Chen et al. (2024),\n"
            "  'European soil bulk density and organic carbon stock database using LUCAS Soil 2018'\n"
            f"  Zenodo record {ZENODO_RECORD}, DOI 10.5281/zenodo.{ZENODO_RECORD}\n"
            "  paper: ESSD 16:2367-2383, https://doi.org/10.5194/essd-16-2367-2024\n"
            f"regression target: topsoil SOC stock (0-20 cm), from '{PRIMARY_CSV}'\n"
        )

    df = load_socs()
    print(f"{len(df)} quality-filtered LUCAS 2018 topsoil SOC-stock points")

    recs = df.to_dict("records")
    selected, edges = bucket_balance_regression(
        recs, "socs", total=args.max_samples, n_buckets=args.n_buckets, seed=SEED
    )
    print(f"selected {len(selected)} (bucket-balanced, {args.n_buckets} buckets)")

    tr = io.year_range(SURVEY_YEAR)
    points = []
    for i, r in enumerate(selected):
        points.append(
            {
                "id": f"{i:06d}",
                "lon": float(r["lon"]),
                "lat": float(r["lat"]),
                "label": float(r["socs"]),
                "time_range": tr,
                "change_time": None,
                "source_id": f"lucas_pointid_{int(r[COL_POINTID])}",
                # auxiliary in-situ / derived soil properties (spec 2a extra props).
                "bulk_density_g_cm3": float(r["bd"]) if not pd.isna(r["bd"]) else None,
                "coarse_fragment_vol_frac": (
                    float(r["coarse"]) if not pd.isna(r["coarse"]) else None
                ),
                "bd_method": str(r["bd_method"]),
            }
        )
    io.write_points_table(SLUG, "regression", points)

    vals = np.array([p["label"] for p in points], dtype=float)
    hist_counts, hist_edges = np.histogram(vals, bins=np.linspace(0, 65, 14))
    io.write_dataset_metadata(
        SLUG,
        {
            "dataset": SLUG,
            "name": NAME,
            "task_type": "regression",
            "source": "EC JRC / ESDAC (LUCAS Soil 2018); open mirror Chen et al. 2024 (Zenodo 10211884)",
            "license": "CC-BY-4.0",
            "provenance": {
                "url": "https://esdac.jrc.ec.europa.eu/projects/lucas",
                "have_locally": False,
                "annotation_method": (
                    "in-situ LUCAS 2018 field topsoil sampling (0-20 cm) + lab analysis; "
                    "SOC stock derived from measured organic carbon x fine-earth bulk "
                    "density (measured or local-RF-predicted) x depth x (1 - coarse "
                    "fragment volume)"
                ),
            },
            "sensors_relevant": ["sentinel2", "sentinel1", "landsat"],
            "regression": {
                "name": "soil_organic_carbon_stock_topsoil",
                "description": (
                    "Topsoil soil organic carbon stock (SOCS) for the 0-20 cm layer, "
                    "from LUCAS Soil 2018 in-situ sampling points (Chen et al. 2024, "
                    "ESSD 16:2367-2383). SOCS = measured LUCAS topsoil organic carbon "
                    "content x fine-earth bulk density x 0.2 m x (1 - coarse fragment "
                    "volume fraction). Bulk density is measured for 5,163 points and "
                    "locally-predicted (random-forest pedotransfer function) for the "
                    "rest. Note: the source CSV header labels the unit 'kg cm-2' which "
                    "is a typo; the correct unit is kg m-2 (= 10x Mg ha-1)."
                ),
                "unit": "kg m-2 (0-20 cm)",
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
                "Point-table regression (spec 2a); label = topsoil SOC stock (0-20 cm, "
                "kg m-2). The authoritative LUCAS 2018 TOPSOIL lab dataset (18,984 "
                "samples with pH-H2O/CaCl2, organic carbon content g/kg, CaCO3, N, P, K, "
                "EC, oxalate Fe/Al, bulk density, coarse fragments) is EC JRC/ESDAC "
                "registration-gated and no ESDAC credential is in .env; per the task we "
                "instead used the OPEN CC-BY-4.0 mirror (Chen et al. 2024, Zenodo "
                f"{ZENODO_RECORD}), which republishes LUCAS Soil 2018 topsoil (0-20 cm) "
                "at the in-situ GPS locations with SOC stock, fine-earth bulk density, "
                "and coarse-fragment volume. 15,389 points carry SOCS with valid GPS "
                f"coords; bucket-balanced across the SOC-stock range to {len(points)} "
                f"samples ({args.n_buckets} buckets, seed {SEED}) because the "
                "distribution is strongly right-skewed (organic/peat-soil tail). Time "
                "range = 1-year window on the 2018 survey year (LUCAS Soil 2018 sampled "
                "Apr-Oct 2018; SOC stock quasi-static). GPS coordinates are the true "
                "field sampling locations. Auxiliary bulk_density_g_cm3, "
                "coarse_fragment_vol_frac, and bd_method are attached per point. Raw OC "
                "content (g/kg) and the full multi-property suite are available from "
                "ESDAC after free registration (needs-credential)."
            ),
        },
    )
    manifest.write_registry_entry(
        SLUG, "completed", task_type="regression", num_samples=len(points)
    )
    print(f"done num_samples={len(points)} task_type=regression")


if __name__ == "__main__":
    main()
