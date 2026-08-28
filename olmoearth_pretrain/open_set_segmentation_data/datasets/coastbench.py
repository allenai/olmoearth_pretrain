"""Process CoastBench coastal transect labels into an open-set-segmentation point table.

Source: CoastBench (Deltares / TU Delft), Zenodo record 15800285, CC-BY-4.0.
~1,763 expert-labeled coastal transects, each anchored to the Global Coastal Transect
System (GCTS) grid with a WGS84 lon/lat origin. Each transect carries several attribute
dimensions; we take ``coastal_type`` (the coastal landform/type classification) as the
primary per-point class and attach the other attributes (shore/sediment type, coastal
defense presence, built-environment presence) as auxiliary point properties.

Sparse point labels -> one dataset-wide GeoJSON point table (spec 2a), not per-point tifs.
Coastal type is quasi-static, so each point gets a representative 1-year Sentinel-era time
range and change_time=null.

Only the tiny ``labels.parquet`` (~340 KB) is needed; it lives inside a 16.6 GB release
zip alongside imagery + model checkpoints, so we selectively extract just that member via
HTTP range requests rather than downloading the whole archive.

Run: python3 -m olmoearth_pretrain.open_set_segmentation_data.datasets.coastbench
"""

import argparse
from collections import Counter

import pandas as pd

from olmoearth_pretrain.open_set_segmentation_data import io, manifest
from olmoearth_pretrain.open_set_segmentation_data.download import HttpRangeFile
from olmoearth_pretrain.open_set_segmentation_data.sampling import balance_by_class

SLUG = "coastbench"
ZENODO_RECORD = "15800285"
ZIP_URL = (
    "https://zenodo.org/api/records/15800285/files/"
    "coastbench-release-2025-04-09.zip/content"
)
LABELS_MEMBER = "labels.parquet"
PER_CLASS = 1000

# Representative Sentinel-era 1-year window. Transects were expert-labeled Aug 2024 - Apr
# 2025 from recent satellite composites; coastal type is quasi-static, so we anchor a
# single representative recent year rather than a per-sample date (change_time=null).
ANCHOR_YEAR = 2023

# Primary classification: coastal_type (coastal landform/type). Ordered by frequency
# (descending) -> class id, with short definitions from the CoastPy coastal typology.
CLASSES = [
    (
        "sediment_plain",
        "Low-relief coast built of unconsolidated sediment (sand/gravel/mud) forming a plain.",
    ),
    (
        "cliffed_or_steep",
        "Cliffed or steeply-sloping coast in rock or consolidated material.",
    ),
    (
        "moderately_sloped",
        "Coast of moderate slope, intermediate between a flat sediment plain and a cliff.",
    ),
    (
        "engineered_structures",
        "Coast dominated by human-made / engineered structures (revetments, seawalls, ports).",
    ),
    ("bedrock_plain", "Low-relief coast on exposed bedrock or a rock shore platform."),
    ("dune", "Coastal dune system (aeolian sand dunes backing the shore)."),
    ("wetland", "Coastal wetland: salt marsh, mangrove, or vegetated tidal flat."),
    ("inlet", "Coastal inlet, estuary mouth, or tidal channel / entrance."),
    ("coral", "Coral-reef-dominated coast."),
]
NAME_TO_ID = {name: i for i, (name, _d) in enumerate(CLASSES)}

# Auxiliary attribute dimensions carried per-point (not the primary class). shore_type is
# the sediment/shore-material class; has_defense and is_built_environment are booleans.
AUX_STR_COLS = ["shore_type", "confidence"]
AUX_BOOL_COLS = ["has_defense", "is_built_environment"]


def _to_bool(v: object) -> bool | None:
    if isinstance(v, str):
        s = v.strip().lower()
        if s == "true":
            return True
        if s == "false":
            return False
        return None
    if isinstance(v, bool):
        return v
    return None


def extract_labels() -> "pd.DataFrame":
    """Selectively extract labels.parquet from the release zip (range requests)."""
    import zipfile

    raw = io.raw_dir(SLUG)
    raw.mkdir(parents=True, exist_ok=True)
    out = raw / LABELS_MEMBER
    if not out.exists():
        rf = HttpRangeFile(ZIP_URL)
        try:
            zf = zipfile.ZipFile(rf)
            data = zf.read(LABELS_MEMBER)
        finally:
            rf.close()
        tmp = raw / (LABELS_MEMBER + ".tmp")
        with tmp.open("wb") as f:
            f.write(data)
        tmp.rename(out)
        print(f"extracted {LABELS_MEMBER}: {len(data)} bytes")
    with (raw / "SOURCE.txt").open("w") as f:
        f.write(
            f"Zenodo record {ZENODO_RECORD} ({ZIP_URL})\n"
            f"Only {LABELS_MEMBER} extracted (via HTTP range requests) from the 16.6 GB "
            f"release zip; imagery/model checkpoints not needed for labels.\n"
        )
    return pd.read_parquet(str(out))


def main() -> None:
    argparse.ArgumentParser().parse_args()

    io.check_disk()
    manifest.write_registry_entry(SLUG, "in_progress")

    df = extract_labels()
    print(f"loaded {len(df)} transects")

    recs = []
    for _, row in df.iterrows():
        ctype = row["coastal_type"]
        if ctype not in NAME_TO_ID:
            continue
        lon, lat = row["lon"], row["lat"]
        if pd.isna(lon) or pd.isna(lat):
            continue
        rec = {
            "lon": float(lon),
            "lat": float(lat),
            "label_name": ctype,
            "source_id": str(row["transect_id"]),
        }
        for c in AUX_STR_COLS:
            v = row.get(c)
            rec[c] = (
                None if (v is None or (isinstance(v, float) and pd.isna(v))) else str(v)
            )
        for c in AUX_BOOL_COLS:
            rec[c] = _to_bool(row.get(c))
        recs.append(rec)
    print(f"usable records: {len(recs)}")

    selected = balance_by_class(recs, "label_name", per_class=PER_CLASS)
    print(f"selected {len(selected)} (<= {PER_CLASS}/class)")

    points = []
    for i, r in enumerate(selected):
        points.append(
            {
                "id": f"{i:06d}",
                "lon": r["lon"],
                "lat": r["lat"],
                "label": NAME_TO_ID[r["label_name"]],
                "time_range": io.year_range(ANCHOR_YEAR),
                "change_time": None,
                "source_id": r["source_id"],
                # auxiliary attribute dimensions
                "shore_type": r["shore_type"],
                "has_defense": r["has_defense"],
                "is_built_environment": r["is_built_environment"],
                "confidence": r["confidence"],
            }
        )
    io.write_points_table(SLUG, "classification", points)

    counts = Counter(r["label_name"] for r in selected)
    io.write_dataset_metadata(
        SLUG,
        {
            "dataset": SLUG,
            "name": "CoastBench",
            "task_type": "classification",
            "source": "Deltares / TU Delft (Zenodo 15800285)",
            "license": "CC-BY-4.0",
            "provenance": {
                "url": "https://zenodo.org/records/15800285",
                "have_locally": False,
                "annotation_method": "manual (expert web-labeled coastal transects)",
            },
            "sensors_relevant": ["sentinel2", "sentinel1", "landsat"],
            "classes": [
                {"id": i, "name": name, "description": desc}
                for i, (name, desc) in enumerate(CLASSES)
            ],
            "nodata_value": io.CLASS_NODATA,
            "num_samples": len(selected),
            "class_counts": {name: counts.get(name, 0) for name, _ in CLASSES},
            "auxiliary_attributes": {
                "shore_type": "Shore/sediment material class (sandy_gravel_or_small_boulder_sediments, "
                "no_sediment_or_shore_platform, rocky_shore_platform_or_large_boulders, muddy_sediments, ice_or_tundra).",
                "has_defense": "Whether a human-made coastal defense is present at the transect (bool).",
                "is_built_environment": "Whether built environment is present at the transect (bool).",
                "confidence": "Annotator confidence (low/medium/high).",
            },
            "notes": (
                "Sparse coastal-transect point labels; primary class = coastal_type (coastal "
                "landform/type). Secondary attributes (shore/sediment type, coastal defense, "
                "built environment) carried per-point in points.geojson properties. Coastal "
                f"type is quasi-static: representative 1-year window ({ANCHOR_YEAR}), "
                "change_time=null. All samples post-2016. No per-class truncation "
                "(max class 366 < 1000 cap)."
            ),
        },
    )
    manifest.write_registry_entry(
        SLUG, "completed", task_type="classification", num_samples=len(selected)
    )
    print("done")


if __name__ == "__main__":
    main()
