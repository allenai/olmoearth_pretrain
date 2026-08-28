"""Process the public GLanCE Global Land Cover Training Data into label points.

Source: Boston University GLanCE project, distributed on Source Cooperative
(``boston-university/bu-glance``). ~1.87M globally distributed 30 m training units, each
labeled by manual on-screen photointerpretation for the NASA MEaSUREs GLanCE product
(Stanimirova et al. 2023, Sci Data). Each row carries a lon/lat, a segment year range
[Start_Year, End_Year] (1984-2020), and a GLanCE Level-1 land-cover class (1-7). This is a
pure sparse-point segmentation dataset -> one dataset-wide point table (points.geojson,
spec 2a), balanced to <=1000 per class.

Relationship to ``olmoearth_glance_land_cover``: that dataset is the local OlmoEarth
*derived-product* eval (11-class OlmoEarth legend, distinct integer legend). THIS dataset
is the upstream manually-photointerpreted *reference* behind GLanCE (7-class GLanCE Level-1
legend), the full public V1 release (~1.9M points vs the olmoearth eval subset). The
manifest marks it "prefer over the map". They are different releases with different
legends, so this is processed on its own terms; the overlap is noted in the summary.

Design decisions (see AGENT_SUMMARY.md sections 2a, 4, 5):
- Classification, 7 GLanCE Level-1 classes mapped to ids 0-6.
- Post-2016 rule: keep only records whose segment reaches the Sentinel era (End_Year >=
  2016). Each label is a stable land-cover state over its segment, so we assign a 1-year
  window uniformly sampled from the post-2016 portion [max(Start_Year, 2016),
  min(End_Year, 2020)] (deterministic, seeded per Glance_ID).
- Change labels: rows with Change==True denote land-cover change somewhere in a multi-year
  segment; the change date is NOT resolvable to within ~1-2 months, so per spec 5 they are
  NOT usable as change labels. We drop them and keep only stable (Change==False) segments,
  whose recorded class holds across the whole segment (safe for any in-segment 1-year
  window). Documented in the summary.
"""

import argparse
import random
from collections import Counter
from typing import Any

from olmoearth_pretrain.open_set_segmentation_data import download, io, manifest
from olmoearth_pretrain.open_set_segmentation_data.sampling import balance_by_class

SLUG = "glance_global_land_cover_training_data"
S3_BUCKET = "boston-university"
S3_KEY = "bu-glance/bu_glance_training_dataV1.parquet"
S3_ENDPOINT = "https://data.source.coop"
PARQUET_NAME = "bu_glance_training_dataV1.parquet"
PER_CLASS = 1000
SENTINEL_START = 2016
GLANCE_MAX_YEAR = 2020
SEED = 42

# GLanCE Level-1 class id (from the README legend) -> our contiguous id (0-based). Order
# matches the manifest class list: Water, Ice/Snow, Developed, Barren, Trees, Shrub,
# Herbaceous.
GLANCE_TO_ID = {1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 7: 6}
CLASSES = [
    (
        "Water",
        "Open water: oceans, lakes, rivers, reservoirs and other permanent or seasonal "
        "water bodies (GLanCE Level-1 class 1).",
    ),
    (
        "Ice/Snow",
        "Permanent snow and ice: glaciers, ice sheets and perennial snow "
        "(GLanCE Level-1 class 2).",
    ),
    (
        "Developed",
        "Developed / built-up: impervious human-made surfaces such as urban areas, roads "
        "and buildings (GLanCE Level-1 class 3).",
    ),
    (
        "Barren",
        "Barren or sparsely vegetated land: bare soil, rock, and sand/beach with little to "
        "no vegetation (GLanCE Level-1 class 4).",
    ),
    (
        "Trees",
        "Tree-dominated vegetation: forest and woodland (deciduous, evergreen or mixed) "
        "(GLanCE Level-1 class 5).",
    ),
    ("Shrub", "Shrub-dominated vegetation (GLanCE Level-1 class 6)."),
    (
        "Herbaceous",
        "Herbaceous vegetation: grassland and other non-woody herbaceous cover, including "
        "cultivated cropland (GLanCE Level-1 class 7).",
    ),
]


def _download_raw() -> str:
    """Download the GeoParquet training table to raw_dir (idempotent). Returns local path."""
    raw = io.raw_dir(SLUG)
    raw.mkdir(parents=True, exist_ok=True)
    dst = raw / PARQUET_NAME
    download.download_s3_unsigned(S3_BUCKET, S3_KEY, dst, endpoint_url=S3_ENDPOINT)
    return dst.path


def scan_records(parquet_path: str) -> list[dict[str, Any]]:
    """Read the parquet, filter to usable stable post-2016 points, build flat records.

    A single ~73 MB parquet read (fast, columnar); no per-file weka I/O, so no Pool needed
    (unlike window-dir scans). Each output record: lon, lat, label (0-based class id),
    year (chosen 1-year window start), time_range, source_id.
    """
    import pandas as pd

    cols = [
        "Lat",
        "Lon",
        "Start_Year",
        "End_Year",
        "Glance_Class_ID_level1",
        "Change",
        "Glance_ID",
    ]
    df = pd.read_parquet(parquet_path, columns=cols)

    # Keep only stable segments (Change==False): their class holds across the whole
    # segment, so any in-segment 1-year window is valid. Change==True segments have an
    # unknown-within-1-2-months change date -> not usable (spec 5), dropped.
    df = df[df["Change"] == False]  # noqa: E712
    # Post-2016 rule: keep segments that reach the Sentinel era.
    df = df[df["End_Year"] >= SENTINEL_START]
    df = df[df["Glance_Class_ID_level1"].isin(GLANCE_TO_ID)]

    records: list[dict[str, Any]] = []
    for lat, lon, sy, ey, cls, gid in zip(
        df["Lat"].to_numpy(),
        df["Lon"].to_numpy(),
        df["Start_Year"].to_numpy(),
        df["End_Year"].to_numpy(),
        df["Glance_Class_ID_level1"].to_numpy(),
        df["Glance_ID"].to_numpy(),
    ):
        lo = max(int(sy), SENTINEL_START)
        hi = min(int(ey), GLANCE_MAX_YEAR)
        if hi < lo:
            continue
        # Deterministic per-record uniform sample of a 1-year window in the valid span.
        year = random.Random(int(gid)).randint(lo, hi)
        records.append(
            {
                "lon": float(lon),
                "lat": float(lat),
                "label": GLANCE_TO_ID[int(cls)],
                "year": year,
                "source_id": f"glance_id/{int(gid)}",
            }
        )
    return records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--per-class", type=int, default=PER_CLASS)
    args = parser.parse_args()

    io.check_disk()
    manifest.write_registry_entry(SLUG, "in_progress")

    parquet_path = _download_raw()
    io.check_disk()

    recs = scan_records(parquet_path)
    print(f"scanned {len(recs)} usable stable post-2016 points")

    selected = balance_by_class(recs, "label", per_class=args.per_class)
    print(f"selected {len(selected)} (<= {args.per_class}/class, 25k cap)")

    points = []
    for i, r in enumerate(selected):
        points.append(
            {
                "id": f"{i:06d}",
                "lon": r["lon"],
                "lat": r["lat"],
                "label": r["label"],
                "time_range": io.year_range(r["year"]),
                "source_id": r["source_id"],
            }
        )
    io.write_points_table(SLUG, "classification", points)

    counts = Counter(r["label"] for r in selected)
    io.write_dataset_metadata(
        SLUG,
        {
            "dataset": SLUG,
            "name": "GLanCE Global Land Cover Training Data",
            "task_type": "classification",
            "source": "Source Cooperative (boston-university/bu-glance)",
            "license": "CC-BY-4.0",
            "provenance": {
                "url": "https://source.coop/boston-university/bu-glance",
                "have_locally": False,
                "annotation_method": "manual photointerpretation (GLanCE reference)",
            },
            "sensors_relevant": ["sentinel2", "sentinel1", "landsat"],
            "classes": [
                {"id": i, "name": name, "description": desc}
                for i, (name, desc) in enumerate(CLASSES)
            ],
            "nodata_value": io.CLASS_NODATA,
            "num_samples": len(selected),
            "class_counts": {
                name: counts.get(i, 0) for i, (name, _d) in enumerate(CLASSES)
            },
            "notes": (
                "Manual photointerpreted reference behind the NASA GLanCE product "
                "(GLanCE Level-1 7-class legend). 1x1 sparse-point classification "
                "labels. Kept only stable segments (Change==0) reaching the Sentinel era "
                "(End_Year>=2016); dropped Change==1 segments (change date not resolvable "
                "to ~1-2 months). Each point gets a 1-year window uniformly sampled from "
                "its post-2016 segment span. Distinct from the derived-product "
                "olmoearth_glance_land_cover (11-class OlmoEarth legend); this is the "
                "preferred manual reference (full public V1 release)."
            ),
        },
    )
    manifest.write_registry_entry(
        SLUG, "completed", task_type="classification", num_samples=len(selected)
    )
    print("done")


if __name__ == "__main__":
    main()
