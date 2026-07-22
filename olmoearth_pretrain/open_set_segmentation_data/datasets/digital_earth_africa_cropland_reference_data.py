"""Process Digital Earth Africa Cropland Reference Data into open-set-segmentation labels.

Source: Digital Earth Africa crop-mask GitHub repo
(github.com/digitalearthafrica/crop-mask). We use the merged, continent-wide reference
set ``testing/combined_training_data.geojson`` (30,448 features): manually
photo-interpreted crop / non-crop reference samples collected via Collect Earth Online
across African agro-ecological zones, plus a few pre-existing crop reference sets, merged
by the project's ``Reference_data_merge`` step. Each feature is a small field-scale
Polygon with a single ``Class`` attribute (1 = cropland, 0 = non-cropland).

These are sparse reference samples (a class-per-location, field-scale footprint), so per
the manifest and spec 2a we represent each as a POINT at the polygon centroid and write one
dataset-wide point table (points.geojson), rather than per-feature GeoTIFFs. Binary class
map: id 0 = non-cropland, id 1 = cropland (ids match the source ``Class`` value). Cropland
is a seasonal/annual state, so each point gets a 1-year time range in the Sentinel era
anchored on 2019 (the DE Africa crop-mask reference campaign period; within the manifest's
2018-2020 range). Balanced to <= 1000/class (2000 total, well under the 25k dataset cap).

Run: python3 -m olmoearth_pretrain.open_set_segmentation_data.datasets.digital_earth_africa_cropland_reference_data
"""

import argparse
import json
import multiprocessing
from collections import Counter
from typing import Any

from shapely.geometry import shape

from olmoearth_pretrain.open_set_segmentation_data import download, io, manifest
from olmoearth_pretrain.open_set_segmentation_data.sampling import balance_by_class

SLUG = "digital_earth_africa_cropland_reference_data"
NAME = "Digital Earth Africa Cropland Reference Data"
SOURCE_URL = (
    "https://raw.githubusercontent.com/digitalearthafrica/crop-mask/main/"
    "testing/combined_training_data.geojson"
)
REPO_URL = "https://github.com/digitalearthafrica/crop-mask"
PER_CLASS = 1000
# Representative Sentinel-era year for this static/annual cropland state; within the
# manifest's 2018-2020 valid range and matching the DE Africa reference campaign period.
LABEL_YEAR = 2019

# Source ``Class`` value -> (id, name, description). Ids match the source Class value so
# label provenance is exact.
CLASSES = [
    (
        0,
        "non-cropland",
        "Any land cover that is not actively cultivated cropland (natural vegetation, "
        "bare ground, water, built-up, etc.), as interpreted from high-resolution imagery.",
    ),
    (
        1,
        "cropland",
        "Actively cultivated cropland / farmland (annual and perennial crops), as "
        "interpreted from high-resolution imagery via Collect Earth Online.",
    ),
]
ID_TO_NAME = {i: name for i, name, _ in CLASSES}


def _centroid_record(feature: dict[str, Any]) -> dict[str, Any] | None:
    """Turn one reference Polygon feature into a centroid point record, or None if
    the geometry is empty/degenerate.
    """
    cls = feature.get("properties", {}).get("Class")
    if cls not in ID_TO_NAME:
        return None
    try:
        geom = shape(feature["geometry"])
    except Exception:
        return None
    if geom.is_empty:
        return None
    if not geom.is_valid:
        geom = geom.buffer(0)
        if geom.is_empty:
            return None
    c = geom.centroid
    if c.is_empty:
        return None
    lon, lat = float(c.x), float(c.y)
    if not (-180 <= lon <= 180 and -90 <= lat <= 90):
        return None
    return {"lon": lon, "lat": lat, "label": int(cls)}


def load_records(path: str) -> list[dict[str, Any]]:
    """Read the reference geojson and reduce each feature to a centroid point record."""
    with open(path) as f:
        fc = json.load(f)
    feats = fc["features"]
    with multiprocessing.Pool(64) as p:
        recs = [r for r in p.map(_centroid_record, feats, chunksize=256) if r]
    return recs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=64)
    parser.parse_args()

    io.check_disk()
    manifest.write_registry_entry(SLUG, "in_progress")

    raw = io.raw_dir(SLUG)
    raw.mkdir(parents=True, exist_ok=True)
    geojson_path = raw / "combined_training_data.geojson"
    download.download_http(SOURCE_URL, geojson_path)
    with (raw / "SOURCE.txt").open("w") as f:
        f.write(f"repo: {REPO_URL}\n")
        f.write(f"file: {SOURCE_URL}\n")
        f.write(
            "Merged continent-wide crop/non-crop reference samples (Collect Earth "
            "Online photointerpretation + pre-existing crop reference sets), field-scale "
            "polygons with a single 'Class' attribute (1=crop, 0=non-crop).\n"
        )

    recs = load_records(str(geojson_path))
    print(f"loaded {len(recs)} reference points (from centroids)")
    print("raw class dist:", Counter(r["label"] for r in recs))

    selected = balance_by_class(recs, "label", per_class=PER_CLASS)
    print(f"selected {len(selected)} (<= {PER_CLASS}/class, 25k cap)")

    points = []
    for i, r in enumerate(selected):
        points.append(
            {
                "id": f"{i:06d}",
                "lon": r["lon"],
                "lat": r["lat"],
                "label": r["label"],
                "time_range": io.year_range(LABEL_YEAR),
                "source_id": None,
            }
        )
    io.write_points_table(SLUG, "classification", points)

    counts = Counter(r["label"] for r in selected)
    io.write_dataset_metadata(
        SLUG,
        {
            "dataset": SLUG,
            "name": NAME,
            "task_type": "classification",
            "source": "Digital Earth Africa (GitHub)",
            "license": "CC-BY-4.0",
            "provenance": {
                "url": REPO_URL,
                "have_locally": False,
                "annotation_method": "manual photointerpretation (Collect Earth Online)",
            },
            "sensors_relevant": ["sentinel2", "sentinel1", "landsat"],
            "classes": [
                {"id": i, "name": name, "description": desc}
                for i, name, desc in CLASSES
            ],
            "nodata_value": io.CLASS_NODATA,
            "num_samples": len(selected),
            "class_counts": {ID_TO_NAME[i]: counts.get(i, 0) for i, _, _ in CLASSES},
            "notes": (
                "1x1 point-segmentation labels (cropland/non-cropland) across African "
                "agro-ecological zones, taken as centroids of the field-scale reference "
                "polygons in combined_training_data.geojson. Ids match source Class "
                "(0=non-cropland, 1=cropland). ~1-year time range anchored on 2019 "
                "(reference campaign period, within 2018-2020). Balanced to <=1000/class "
                "(2000 total; raw pool was 12,422 cropland / 18,026 non-cropland)."
            ),
        },
    )
    manifest.write_registry_entry(
        SLUG, "completed", task_type="classification", num_samples=len(selected)
    )
    print("done")


if __name__ == "__main__":
    multiprocessing.set_start_method("forkserver", force=True)
    main()
