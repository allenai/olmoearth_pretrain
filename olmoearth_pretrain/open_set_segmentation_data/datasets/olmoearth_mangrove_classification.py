"""Process OlmoEarth mangrove classification into open-set-segmentation label points.

Source: local rslearn eval at rslearn-eai/datasets/mangrove/classification/20250626. Each
window is a 32x32 @10m patch labeled with a single, uniform class (Mangrove / Water /
Other) derived from Global Mangrove Watch, stored in window metadata.json ``options.label``
(and duplicated as a single-category polygon covering the whole window in the ``label``
vector layer). Because every window carries one uniform class, this is effectively sparse
point segmentation, so we write one dataset-wide point table (points.geojson, spec 2a),
balanced to <=1000 per class.

Two window groups are both used (all source splits per spec 5): ``reference`` (test split,
Mangrove/Other only) and ``sample_100K`` (train/val, all three classes). The point location
is the WGS84 center of each window's UTM bounds. All windows share a 30-day source time
range (2020-06-15..2020-07-15); since these are static/annual GMW land-cover labels we
assign the spec-5 default 1-year window anchored on the labeled year (2020).
"""

import argparse
import json
import multiprocessing
import os
from typing import Any

import shapely
from rasterio.crs import CRS
from rslearn.const import WGS84_PROJECTION
from rslearn.utils.geometry import Projection, STGeometry

from olmoearth_pretrain.open_set_segmentation_data import io, manifest
from olmoearth_pretrain.open_set_segmentation_data.sampling import balance_by_class

SLUG = "olmoearth_mangrove_classification"
SOURCE = "/weka/dfive-default/rslearn-eai/datasets/mangrove/classification/20250626"
PER_CLASS = 1000
LABEL_YEAR = 2020

# Class ordering follows the manifest -> id, with short definitions.
CLASSES = [
    (
        "Mangrove",
        "Mangrove forest / mangrove-covered tidal wetland (per Global Mangrove Watch).",
    ),
    ("Water", "Open water: ocean, tidal channels, and other permanent/standing water."),
    (
        "Other",
        "Any non-mangrove, non-water land cover (vegetation, bare soil, built-up, etc.).",
    ),
]
NAME_TO_ID = {name: i for i, (name, _desc) in enumerate(CLASSES)}


def scan_records() -> list[dict[str, Any]]:
    """Parallel-read window metadata into flat records (one per labeled window)."""
    jobs = []
    windows_root = os.path.join(SOURCE, "windows")
    for group in os.listdir(windows_root):
        gd = os.path.join(windows_root, group)
        if os.path.isdir(gd):
            for name in os.listdir(gd):
                jobs.append(os.path.join(gd, name))
    with multiprocessing.Pool(64) as p:
        recs = [r for r in p.map(_read_one, jobs, chunksize=128) if r]
    return recs


def _read_one(path: str) -> dict[str, Any] | None:
    try:
        with open(os.path.join(path, "metadata.json")) as f:
            md = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return None
    opt = md.get("options", {})
    if opt.get("label") not in NAME_TO_ID:
        return None
    # Point location = WGS84 center of the window's UTM bounds (exact labeled pixel).
    b = md["bounds"]
    proj = Projection(CRS.from_string(md["projection"]["crs"]), 10, -10)
    cx = (b[0] + b[2]) / 2.0
    cy = (b[1] + b[3]) / 2.0
    geom = STGeometry(proj, shapely.Point(cx, cy), None).to_projection(WGS84_PROJECTION)
    group = os.path.basename(os.path.dirname(path))
    name = os.path.basename(path)
    return {
        "lon": float(geom.shp.x),
        "lat": float(geom.shp.y),
        "label": opt["label"],
        "source_id": f"{group}/{name}",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=64)
    args = parser.parse_args()

    io.check_disk()
    manifest.write_registry_entry(SLUG, "in_progress")

    raw = io.raw_dir(SLUG)
    raw.mkdir(parents=True, exist_ok=True)
    with (raw / "SOURCE.txt").open("w") as f:
        f.write(f"local rslearn dataset: {SOURCE}\n")

    recs = scan_records()
    print(f"scanned {len(recs)} labeled windows")
    selected = balance_by_class(recs, "label", per_class=PER_CLASS)
    print(f"selected {len(selected)} (<= {PER_CLASS}/class)")

    # Sparse point dataset -> one dataset-wide point table (spec 2a), not per-point tifs.
    time_range = io.year_range(LABEL_YEAR)
    points = []
    for i, r in enumerate(selected):
        cid = NAME_TO_ID[r["label"]]
        points.append(
            {
                "id": f"{i:06d}",
                "lon": r["lon"],
                "lat": r["lat"],
                "label": cid,
                "time_range": time_range,
                "source_id": r["source_id"],
            }
        )
    io.write_points_table(SLUG, "classification", points)

    from collections import Counter

    counts = Counter(r["label"] for r in selected)
    io.write_dataset_metadata(
        SLUG,
        {
            "dataset": SLUG,
            "name": "OlmoEarth mangrove classification",
            "task_type": "classification",
            "source": "olmoearth",
            "license": "internal",
            "provenance": {
                "url": SOURCE,
                "have_locally": True,
                "annotation_method": "derived (Global Mangrove Watch)",
            },
            "sensors_relevant": ["sentinel2", "sentinel1", "landsat"],
            "classes": [
                {"id": i, "name": name, "description": desc}
                for i, (name, desc) in enumerate(CLASSES)
            ],
            "nodata_value": io.CLASS_NODATA,
            "num_samples": len(selected),
            "class_counts": {name: counts.get(name, 0) for name, _ in CLASSES},
            "notes": (
                "1x1 point-segmentation labels from uniform-class 32x32 windows; both window "
                "groups (reference/test + sample_100K/train+val) and all splits used. Each "
                "point gets a 1-year window anchored on 2020 (spec 5 land-cover default; the "
                "source windows used a curated 30-day 2020-06-15..2020-07-15 range). All three "
                "classes truncated to 1000 (each had >1000 available)."
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
