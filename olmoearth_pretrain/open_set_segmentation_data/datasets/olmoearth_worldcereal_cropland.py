"""Process OlmoEarth WorldCereal cropland into open-set-segmentation label points.

Source: local rslearn eval at rslearn-eai/datasets/crop/worldcereal_cropland/20250422.
Each window is one in-situ WorldCereal RDM reference sample covering a single 10 m pixel
(1x1 bounds). The binary class ("Cropland" / "Non-Cropland") lives in the window's vector
``label`` layer (feature ``properties.category``); the window ``metadata.json`` carries the
UTM projection, the 1x1 pixel bounds, a ~1-month observation time_range, and provenance in
``options`` (WorldCereal ``sample_id``, ``ewoc_code``, quality scores, source split).

Sparse point segmentation -> one dataset-wide point table (points.geojson, spec 2a),
balanced to <=1000 per class. Each label is a seasonal/annual crop observation, so we
anchor a 1-year time range on the labeled year (spec 5).

Run: python3 -m olmoearth_pretrain.open_set_segmentation_data.datasets.olmoearth_worldcereal_cropland
"""

import argparse
import json
import multiprocessing
import os
from collections import Counter
from typing import Any

from olmoearth_pretrain.open_set_segmentation_data import io, manifest
from olmoearth_pretrain.open_set_segmentation_data.sampling import balance_by_class

SLUG = "olmoearth_worldcereal_cropland"
SOURCE = "/weka/dfive-default/rslearn-eai/datasets/crop/worldcereal_cropland/20250422"
PER_CLASS = 1000

# Manifest class order -> id. Binary cropland classification.
CLASSES = [
    (
        "Cropland",
        "Land actively cultivated with annual or perennial crops during the "
        "reference season, per the ESA WorldCereal harmonized in-situ definition (temporary "
        "crops, incl. arable land and managed cropland).",
    ),
    (
        "Non-Cropland",
        "All other land cover (natural vegetation, grassland, forest, built-up, "
        "water, bare) that is not cultivated cropland in the reference season.",
    ),
]
NAME_TO_ID = {name: i for i, (name, _desc) in enumerate(CLASSES)}


def scan_records(workers: int) -> list[dict[str, Any]]:
    """Parallel-read window metadata + label vector into flat records."""
    jobs = []
    windows_root = os.path.join(SOURCE, "windows")
    for group in sorted(os.listdir(windows_root)):
        gd = os.path.join(windows_root, group)
        if os.path.isdir(gd):
            for name in os.listdir(gd):
                jobs.append(os.path.join(gd, name))
    with multiprocessing.Pool(workers) as p:
        recs = [r for r in p.map(_read_one, jobs, chunksize=64) if r]
    return recs


def _read_one(path: str) -> dict[str, Any] | None:
    try:
        with open(os.path.join(path, "metadata.json")) as f:
            md = json.load(f)
        with open(os.path.join(path, "layers", "label", "data.geojson")) as f:
            g = json.load(f)
    except FileNotFoundError:
        return None
    feats = g.get("features") or []
    if not feats:
        return None
    category = feats[0].get("properties", {}).get("category")
    if category not in NAME_TO_ID:
        return None
    tr = md.get("time_range")
    year = int(tr[0][:4]) if tr else None
    if (
        year is None or year < 2016
    ):  # Sentinel era only (all WorldCereal samples are 2017+)
        return None
    crs = md["projection"]["crs"]
    bounds = md["bounds"]
    lon, lat = io.pixel_center_lonlat(crs, bounds)
    opt = md.get("options", {})
    return {
        "lon": lon,
        "lat": lat,
        "label": category,
        "year": year,
        "source_id": opt.get("sample_id")
        or f"{os.path.basename(os.path.dirname(path))}/{os.path.basename(path)}",
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

    recs = scan_records(args.workers)
    print(f"scanned {len(recs)} labeled points")
    selected = balance_by_class(recs, "label", per_class=PER_CLASS)
    print(f"selected {len(selected)} (<= {PER_CLASS}/class)")

    # Sparse point dataset -> one dataset-wide point table (spec 2a).
    points = []
    for i, r in enumerate(selected):
        points.append(
            {
                "id": f"{i:06d}",
                "lon": r["lon"],
                "lat": r["lat"],
                "label": NAME_TO_ID[r["label"]],
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
            "name": "OlmoEarth WorldCereal cropland",
            "task_type": "classification",
            "source": "olmoearth",
            "license": "CC-BY-4.0",
            "provenance": {
                "url": SOURCE,
                "have_locally": True,
                "annotation_method": "in-situ / harmonized reference (ESA WorldCereal RDM)",
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
                "1x1 point-segmentation labels from ESA WorldCereal in-situ reference; "
                "binary cropland vs non-cropland. All source splits (train/val) used. "
                "1-year time range anchored on each sample's labeled year (2017-2023)."
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
