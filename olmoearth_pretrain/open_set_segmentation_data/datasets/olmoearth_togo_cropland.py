"""Process OlmoEarth Togo cropland into open-set-segmentation label patches.

Source: local rslearn dataset materialized from the ``togo_cropland`` olmoearth_projects
project (labels from nasaharvest/togo-crop-mask, Zenodo record 3836629). Each window is one
field-surveyed crop / non-crop point in Togo, buffered to a small window. Window
metadata.json carries the crop/non_crop class (``options.lulc_category``), the source split,
and a growing-season 2019 time range; the point location is the center of the window bounds
(EPSG:32631 UTM at 10 m). Sparse point classification, so we write one dataset-wide point
table (points.json, spec 2a), balanced to <=1000 per class. All source splits are used.
"""

import argparse
import multiprocessing
import os
from datetime import datetime
from typing import Any

import shapely
from rasterio.crs import CRS
from rslearn.const import WGS84_PROJECTION
from rslearn.utils.geometry import Projection, STGeometry

from olmoearth_pretrain.open_set_segmentation_data import io, manifest
from olmoearth_pretrain.open_set_segmentation_data.sampling import balance_by_class

SLUG = "olmoearth_togo_cropland"
SOURCE = "/weka/dfive-default/rslearn-eai/datasets/crop/togo_2020/20260127"
PROJECT = "olmoearth_projects/projects/togo_cropland"
PER_CLASS = 1000

# Manifest class order -> id.
CLASSES = [
    (
        "non_crop",
        "Land that is not actively cultivated cropland (natural vegetation, "
        "bare ground, water, settlements).",
    ),
    (
        "crop",
        "Actively cultivated cropland / farmland identified by in-situ field survey.",
    ),
]
NAME_TO_ID = {name: i for i, (name, _desc) in enumerate(CLASSES)}


def scan_records() -> list[dict[str, Any]]:
    """Parallel-read window metadata into flat point records."""
    jobs = []
    windows_root = os.path.join(SOURCE, "windows")
    for group in os.listdir(windows_root):
        gd = os.path.join(windows_root, group)
        if os.path.isdir(gd):
            for name in os.listdir(gd):
                jobs.append(os.path.join(gd, name))
    with multiprocessing.Pool(64) as p:
        recs = [r for r in p.map(_read_one, jobs, chunksize=64) if r]
    return recs


def _read_one(path: str) -> dict[str, Any] | None:
    import json

    try:
        with open(os.path.join(path, "metadata.json")) as f:
            md = json.load(f)
    except FileNotFoundError:
        return None
    opt = md.get("options", {})
    label = opt.get("lulc_category")
    if label not in NAME_TO_ID:
        return None
    proj = md["projection"]
    b = md["bounds"]
    projection = Projection(
        CRS.from_string(proj["crs"]), proj["x_resolution"], proj["y_resolution"]
    )
    cx = (b[0] + b[2]) / 2.0
    cy = (b[1] + b[3]) / 2.0
    geom = STGeometry(projection, shapely.Point(cx, cy), None).to_projection(
        WGS84_PROJECTION
    )
    tr = md.get("time_range")
    return {
        "lon": float(geom.shp.x),
        "lat": float(geom.shp.y),
        "label": label,
        "time_range": tr,  # [iso, iso], already <= 1 year
        "split": opt.get("split"),
        "source_id": f"{os.path.basename(os.path.dirname(path))}/{os.path.basename(path)}",
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
        f.write(f"olmoearth_projects project: {PROJECT}\n")
        f.write("labels: nasaharvest/togo-crop-mask (Zenodo record 3836629)\n")

    recs = scan_records()
    print(f"scanned {len(recs)} labeled points")
    selected = balance_by_class(recs, "label", per_class=PER_CLASS)
    print(f"selected {len(selected)} (<= {PER_CLASS}/class)")

    # Sparse point dataset -> one dataset-wide point table (spec 2a), not per-point tifs.
    points = []
    for i, r in enumerate(selected):
        cid = NAME_TO_ID[r["label"]]
        tr = r["time_range"]
        if tr:
            time_range = [
                datetime.fromisoformat(tr[0]),
                datetime.fromisoformat(tr[1]),
            ]
        else:
            time_range = io.year_range(2019)
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
            "name": "OlmoEarth Togo cropland",
            "task_type": "classification",
            "source": "olmoearth",
            "license": "internal",
            "provenance": {
                "url": PROJECT,
                "have_locally": True,
                "annotation_method": "manual field survey (nasaharvest/togo-crop-mask)",
            },
            "sensors_relevant": ["sentinel2", "sentinel1", "landsat"],
            "classes": [
                {"id": i, "name": name, "description": desc}
                for i, (name, desc) in enumerate(CLASSES)
            ],
            "nodata_value": io.CLASS_NODATA,
            "num_samples": len(selected),
            "class_counts": {name: counts.get(name, 0) for name, _ in CLASSES},
            "notes": "1x1 point-segmentation labels (crop/non_crop) in Togo; all source "
            "splits (train/val/test) used; growing-season 2019 time range "
            "(2019-02-01..2019-09-30) per point.",
        },
    )
    manifest.write_registry_entry(
        SLUG, "completed", task_type="classification", num_samples=len(selected)
    )
    print("done")


if __name__ == "__main__":
    multiprocessing.set_start_method("forkserver", force=True)
    main()
