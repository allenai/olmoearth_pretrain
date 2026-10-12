"""Process SustainBench Poverty (Asset Wealth Index) into a point-table regression dataset.

Source: SustainBench (Stanford Sustainability & AI Lab), the DHS survey-derived poverty
benchmark. Cluster-level labels are published as ``dhs_final_labels.csv`` on the project
Google Drive (folder 1tzWDfd4Y5MvJnJb-lHieOuD-aVcUqzcu). Each row is one DHS survey
cluster ("enumeration area", roughly a village / neighborhood) with:
  - ``lat`` / ``lon``  : cluster centroid (WGS84; DHS jitters urban clusters up to 2 km
                         and rural up to ~5-10 km for privacy -- the standard public
                         geocoords),
  - ``asset_index``    : the cluster-mean asset wealth index (PCA over household assets;
                         higher = wealthier). This is the SustainBench regression target.
  - ``year`` / ``cname`` / ``urban``.

The underlying DHS household microdata is registration-gated, but the aggregated
cluster-level wealth index + centroid coordinates are the *SustainBench label* and are
public -- so we use them directly (no DHS credential needed).

Each label is a single point with a continuous value -> REGRESSION written to a
dataset-wide point table (points.json, spec 2a), NOT per-point GeoTIFFs.

We restrict to survey year >= 2016 (Sentinel-2 era; also the manifest's [2016, 2019]
window) so each cluster gets a valid 1-year Sentinel-era time range, then randomly
sample down to the 5000-sample regression cap. The value distribution is not strongly
skewed, so we use a plain seeded random sample (no bucket balancing).

Run:
  python3 -m olmoearth_pretrain.open_set_segmentation_data.datasets.sustainbench_poverty_asset_wealth_index
"""

import argparse
import random

import numpy as np
import pandas as pd

from olmoearth_pretrain.open_set_segmentation_data import io, manifest

SLUG = "sustainbench_poverty_asset_wealth_index"
NAME = "SustainBench Poverty (Asset Wealth Index)"

# dhs_final_labels.csv lives in the SustainBench poverty Google Drive folder.
GDRIVE_FOLDER = "1tzWDfd4Y5MvJnJb-lHieOuD-aVcUqzcu"
GDRIVE_FILE_ID = "16OORDhlm5OufImAIRGRNW0kZc3rowrks"
CSV_NAME = "dhs_final_labels.csv"

MIN_YEAR = 2016  # Sentinel-2 era + manifest [2016, 2019] window
MAX_REGRESSION = 5000
SEED = 42


def ensure_raw() -> str:
    """Download dhs_final_labels.csv to raw_dir if absent; return its local path."""
    raw = io.raw_dir(SLUG)
    raw.mkdir(parents=True, exist_ok=True)
    csv_path = raw / CSV_NAME
    if not csv_path.exists():
        import gdown

        gdown.download(id=GDRIVE_FILE_ID, output=str(csv_path), quiet=False)
    with (raw / "SOURCE.txt").open("w") as f:
        f.write(
            "SustainBench Poverty (Asset Wealth Index), DHS cluster labels.\n"
            f"file: {CSV_NAME}\n"
            f"google drive folder: https://drive.google.com/drive/folders/{GDRIVE_FOLDER}\n"
            f"google drive file id: {GDRIVE_FILE_ID}\n"
            "docs: https://sustainlab-group.github.io/sustainbench/docs/datasets/sdg1/change_in_poverty.html\n"
        )
    return str(csv_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--min-year", type=int, default=MIN_YEAR)
    parser.add_argument("--max-samples", type=int, default=MAX_REGRESSION)
    args = parser.parse_args()

    io.check_disk()
    manifest.write_registry_entry(SLUG, "in_progress")

    csv_path = ensure_raw()
    df = pd.read_csv(csv_path)

    # Keep clusters with a valid asset wealth index and real coordinates.
    df = df.dropna(subset=["asset_index", "lat", "lon"])
    df = df[(df["lat"] != 0) | (df["lon"] != 0)]
    df = df[df["year"] >= args.min_year]
    print(
        f"{len(df)} clusters with asset_index and year >= {args.min_year} "
        f"({df['cname'].nunique()} countries)"
    )

    # Plain seeded random sample down to the regression cap (distribution ~symmetric).
    recs = df.to_dict("records")
    rng = random.Random(SEED)
    rng.shuffle(recs)
    selected = recs[: args.max_samples]
    print(f"selected {len(selected)} clusters (cap {args.max_samples})")

    points = []
    for i, r in enumerate(selected):
        year = int(r["year"])
        points.append(
            {
                "id": f"{i:06d}",
                "lon": float(r["lon"]),
                "lat": float(r["lat"]),
                "label": float(r["asset_index"]),
                "time_range": io.year_range(year),
                "source_id": str(r["DHSID_EA"]),
            }
        )
    io.write_points_table(SLUG, "regression", points)

    vals = np.array([p["label"] for p in points], dtype=float)
    io.write_dataset_metadata(
        SLUG,
        {
            "dataset": SLUG,
            "name": NAME,
            "task_type": "regression",
            "source": "SustainBench (Stanford Sustainability & AI Lab)",
            "license": "research; DHS registration (aggregated cluster labels public)",
            "provenance": {
                "url": "https://sustainlab-group.github.io/sustainbench/docs/datasets/sdg1/change_in_poverty.html",
                "have_locally": False,
                "annotation_method": "survey-derived (DHS household asset PCA, cluster-mean)",
            },
            "sensors_relevant": ["sentinel2", "sentinel1", "landsat"],
            "regression": {
                "name": "asset_wealth_index",
                "description": (
                    "Cluster-mean asset wealth index from DHS surveys: a scalar per "
                    "household computed by PCA over asset ownership / housing-quality "
                    "variables, averaged over households in a survey cluster (enumeration "
                    "area). Higher = wealthier. Dimensionless (standardized). Cluster "
                    "centroids are DHS-jittered up to 2 km (urban) / ~5-10 km (rural)."
                ),
                "unit": "index (standardized, dimensionless)",
                "dtype": "float32",
                "value_range": [float(vals.min()), float(vals.max())],
                "nodata_value": io.REGRESSION_NODATA,
            },
            "num_samples": len(points),
            "notes": (
                f"Point-table regression (spec 2a); label = asset_index. Restricted to "
                f"DHS survey year >= {args.min_year} (Sentinel-2 era / manifest "
                f"[2016,2019]); {len(points)} clusters randomly sampled (seed {SEED}) "
                f"from the >= {args.min_year} pool. 1-year time range anchored on the "
                f"survey year. All source splits used. Full public label file "
                f"dhs_final_labels.csv has 86,936 clusters over 1996-2019 / 56 countries; "
                f"we use the Sentinel-era subset."
            ),
        },
    )
    manifest.write_registry_entry(
        SLUG, "completed", task_type="regression", num_samples=len(points)
    )
    print("done")


if __name__ == "__main__":
    main()
