"""Process WoSIS Soil Profiles into a point-table regression dataset (topsoil pH-H2O).

Source: ISRIC WoSIS "World Soil Information Service" 2023 snapshot (December 2023),
the standardised global compilation of legacy soil-profile point data (228k profiles /
174 countries). Openly downloadable, CC-BY-4.0, from
https://files.isric.org/public/wosis_snapshot/WoSIS_2023_December.zip
(DOI 10.17027/isric-wdcsoils-20231130). No credential required.

The snapshot ships one TSV per standardised soil property. Each per-property TSV is a
layer-level (horizon-level) table that already carries the profile's lon/lat, sampling
date, positional uncertainty, and the standardised value -- so no join to the profiles
file is needed. Columns of interest:
    profile_id, upper_depth, lower_depth, value_avg (the standardised numeric value),
    longitude, latitude, positional_uncertainty, date.

REGRESSION TARGET -- topsoil pH in water (PHAQ / pH-H2O).
We pick pH-H2O because, of the manifest's suggested continuous candidates (organic
carbon vs pH), PHAQ is the most-populated: 140,326 profiles / 655,336 layers vs ORGC's
135,655 profiles (from wosis_202312_observations.tsv). pH is a clean, bounded, widely
modelled continuous soil property.

"Topsoil" = the shallowest sampled layer of each profile with ``upper_depth < 30`` cm
(the 0-30 cm topsoil convention); one value per profile. This yields ~137k profiles.

Each label is a single point with a continuous value -> REGRESSION written to a
dataset-wide point table (points.json, spec 2a), NOT per-point GeoTIFFs.

Quality filters:
  - keep pH in [2, 12] (drops 3 physically-impossible outliers);
  - keep only profiles located to <= ~1 km ("Circa 100 m" or "100 m - 1 km"); WoSIS
    also carries "1 km - 10 km" / "Over 10 km" profiles whose coordinates are too coarse
    to place meaningfully on a 10 m Sentinel grid, so we drop those (~18.5k profiles).

Time range: WoSIS is legacy in-situ data -- most sampling dates are pre-Sentinel (median
~1991; only ~100 profiles sampled >= 2016) and ~38% carry no date. Soil pH is a
quasi-static property (SoilGrids et al. routinely learn it from recent imagery over
legacy profiles), so per spec 5 (static labels) we anchor a single representative
Sentinel-era 1-year window (2020) on every point rather than the historical sample date.
This is the main caveat and is documented in the summary.

The pH distribution is only moderately skewed but has long acidic/alkaline tails, so we
bucket-balance across the value range (spec 5) to give even coverage of the full pH range
when sampling down to the 5000-sample regression cap.

Run:
  python3 -m olmoearth_pretrain.open_set_segmentation_data.datasets.wosis_soil_profiles
"""

import argparse

import numpy as np
import pandas as pd

from olmoearth_pretrain.open_set_segmentation_data import io, manifest
from olmoearth_pretrain.open_set_segmentation_data.sampling import (
    bucket_balance_regression,
)

SLUG = "wosis_soil_profiles"
NAME = "WoSIS Soil Profiles"

# ISRIC WoSIS 2023 snapshot (December 2023), extracted TSVs live under raw_dir.
SNAPSHOT_DIR = "wosis_202312/WoSIS_2023_December"
PROPERTY_TSV = "wosis_202312_phaq.tsv"  # pH in water (pH-H2O)
DOWNLOAD_URL = "https://files.isric.org/public/wosis_snapshot/WoSIS_2023_December.zip"

TOPSOIL_MAX_UPPER_DEPTH = 30  # cm; 0-30 cm topsoil convention
PH_MIN, PH_MAX = 2.0, 12.0
GOOD_UNCERTAINTY = {"Circa 100 m", "100 m - 1 km"}
REPRESENTATIVE_YEAR = 2020  # Sentinel-2 era anchor for quasi-static soil labels
MAX_REGRESSION = 5000
N_BUCKETS = 10
SEED = 42


def load_topsoil_ph() -> pd.DataFrame:
    """Return one-topsoil-pH-per-profile records (quality filtered)."""
    tsv = io.raw_dir(SLUG) / SNAPSHOT_DIR / PROPERTY_TSV
    cols = [
        "profile_id",
        "upper_depth",
        "lower_depth",
        "value_avg",
        "longitude",
        "latitude",
        "positional_uncertainty",
        "date",
    ]
    df = pd.read_csv(tsv.path, sep="\t", usecols=cols, low_memory=False)
    df = df.dropna(subset=["value_avg", "longitude", "latitude", "upper_depth"])
    # Topsoil: shallowest layer with upper_depth < 30 cm, one per profile.
    df = df[df["upper_depth"] < TOPSOIL_MAX_UPPER_DEPTH]
    df = (
        df.sort_values(["profile_id", "upper_depth"])
        .groupby("profile_id", as_index=False)
        .first()
    )
    df["value_avg"] = df["value_avg"].astype(float)
    df = df[(df["value_avg"] >= PH_MIN) & (df["value_avg"] <= PH_MAX)]
    df = df[df["positional_uncertainty"].isin(GOOD_UNCERTAINTY)]
    df = df[(df["longitude"].between(-180, 180)) & (df["latitude"].between(-90, 90))]
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
    with (raw / "SOURCE.txt").open("w") as f:
        f.write(
            "ISRIC WoSIS 2023 snapshot (December 2023).\n"
            f"download: {DOWNLOAD_URL}\n"
            "DOI: https://doi.org/10.17027/isric-wdcsoils-20231130\n"
            "paper: https://doi.org/10.5194/essd-16-4735-2024\n"
            "license: CC-BY-4.0\n"
            f"regression target: topsoil pH in water, from {SNAPSHOT_DIR}/{PROPERTY_TSV}\n"
        )

    df = load_topsoil_ph()
    print(f"{len(df)} quality-filtered topsoil pH profiles")

    recs = df.to_dict("records")
    selected, edges = bucket_balance_regression(
        recs, "value_avg", total=args.max_samples, n_buckets=args.n_buckets, seed=SEED
    )
    print(f"selected {len(selected)} (bucket-balanced, {args.n_buckets} buckets)")

    tr = io.year_range(REPRESENTATIVE_YEAR)
    points = []
    for i, r in enumerate(selected):
        points.append(
            {
                "id": f"{i:06d}",
                "lon": float(r["longitude"]),
                "lat": float(r["latitude"]),
                "label": float(r["value_avg"]),
                "time_range": tr,
                "change_time": None,
                "source_id": f"profile_{int(r['profile_id'])}",
            }
        )
    io.write_points_table(SLUG, "regression", points)

    vals = np.array([p["label"] for p in points], dtype=float)
    hist_counts, hist_edges = np.histogram(vals, bins=np.arange(2.0, 12.5, 1.0))
    io.write_dataset_metadata(
        SLUG,
        {
            "dataset": SLUG,
            "name": NAME,
            "task_type": "regression",
            "source": "ISRIC WoSIS 2023 snapshot (December 2023)",
            "license": "CC-BY-4.0",
            "provenance": {
                "url": "https://www.isric.org/explore/wosis",
                "have_locally": False,
                "annotation_method": "field soil profiles + ISRIC standardisation",
            },
            "sensors_relevant": ["sentinel2", "sentinel1", "landsat"],
            "regression": {
                "name": "soil_ph_h2o_topsoil",
                "description": (
                    "Topsoil pH measured in a water (H2O) suspension (WoSIS property "
                    "PHAQ): the negative log10 activity of H+ ions. Taken from the "
                    "shallowest sampled layer with upper depth < 30 cm of each WoSIS "
                    "profile (0-30 cm topsoil), using the ISRIC-standardised value_avg. "
                    "Dimensionless pH units (~1.5-11)."
                ),
                "unit": "pH (dimensionless)",
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
                "Point-table regression (spec 2a); label = topsoil pH-H2O. Source: WoSIS "
                "2023 December snapshot, per-property file wosis_202312_phaq.tsv (PHAQ is "
                "the most-populated of the organic-carbon/pH candidates: 140,326 profiles "
                "vs ORGC 135,655). One value per profile = shallowest layer with "
                "upper_depth < 30 cm. Kept pH in [2,12] and only profiles located to "
                "<= ~1 km ('Circa 100 m' / '100 m - 1 km'); dropped ~18.5k coarser "
                "('1 km - 10 km' / 'Over 10 km') profiles. ~137k profiles passed filters; "
                f"bucket-balanced across the pH range to {len(points)} samples "
                f"({args.n_buckets} buckets, seed {SEED}). CAVEAT: WoSIS sampling dates "
                "are legacy (median ~1991; only ~100 profiles >= 2016) and ~38% undated, "
                "so we anchor a representative Sentinel-era 1-year window (2020) on every "
                "point rather than the historical date -- valid because topsoil pH is "
                "quasi-static. Positional uncertainty (~100 m for most kept profiles) is "
                "coarser than a 10 m pixel; treat as approximately located."
            ),
        },
    )
    manifest.write_registry_entry(
        SLUG, "completed", task_type="regression", num_samples=len(points)
    )
    print(f"done num_samples={len(points)} task_type=regression")


if __name__ == "__main__":
    main()
