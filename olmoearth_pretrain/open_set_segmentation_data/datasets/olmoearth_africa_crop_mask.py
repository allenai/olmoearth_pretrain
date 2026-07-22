"""Process OlmoEarth Africa crop mask into open-set-segmentation label patches.

Source: local rslearn eval at olmoearth_evals/africa_crop_mask. Each window carries one
manually / photo-interpreted point over Africa labeled crop vs not_crop, with the class
name, lon/lat, and a ~1-year time range stored in window metadata.json ``options`` (also
mirrored in a single-point label vector layer and a 32x32 label_raster where the center
pixel holds the class id and the rest is 255 nodata). Sparse point segmentation, so we
write one dataset-wide point table (points.geojson, spec 2a), balanced to <=1000 per class.
"""

import argparse
import multiprocessing
import os
from collections import Counter
from typing import Any

from olmoearth_pretrain.open_set_segmentation_data import io, manifest
from olmoearth_pretrain.open_set_segmentation_data.sampling import balance_by_class

SLUG = "olmoearth_africa_crop_mask"
SOURCE = "/weka/dfive-default/rslearn-eai/datasets/olmoearth_evals/africa_crop_mask"
PER_CLASS = 1000

# Manifest class ordering -> id. Confirmed against the source label_raster values
# (not_crop pixels = 0, crop pixels = 1).
CLASSES = [
    (
        "not_crop",
        "Non-agricultural land: any land cover that is not actively cultivated cropland (natural vegetation, water, built-up, bare, etc.).",
    ),
    ("crop", "Agricultural cropland: land actively used for growing crops."),
]
NAME_TO_ID = {name: i for i, (name, _desc) in enumerate(CLASSES)}


def scan_records() -> list[dict[str, Any]]:
    """Parallel-read window metadata into flat records."""
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
    tr = md.get("time_range")
    if opt.get("lon") is None or opt.get("label") not in NAME_TO_ID:
        return None
    return {
        "lon": opt["lon"],
        "lat": opt["lat"],
        "label": opt["label"],
        "year": int(tr[0][:4]) if tr else 2019,
        "source_id": f"{os.path.basename(os.path.dirname(path))}/{os.path.basename(path)}",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=64)
    parser.parse_args()

    io.check_disk()
    manifest.write_registry_entry(SLUG, "in_progress")

    raw = io.raw_dir(SLUG)
    raw.mkdir(parents=True, exist_ok=True)
    with (raw / "SOURCE.txt").open("w") as f:
        f.write(f"local rslearn dataset: {SOURCE}\n")

    recs = scan_records()
    print(f"scanned {len(recs)} labeled points")
    selected = balance_by_class(recs, "label", per_class=PER_CLASS)
    print(f"selected {len(selected)} (<= {PER_CLASS}/class)")

    # Sparse point dataset -> one dataset-wide point table (spec 2a), not per-point tifs.
    points = []
    for i, r in enumerate(selected):
        cid = NAME_TO_ID[r["label"]]
        points.append(
            {
                "id": f"{i:06d}",
                "lon": r["lon"],
                "lat": r["lat"],
                "label": cid,
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
            "name": "OlmoEarth Africa crop mask",
            "task_type": "classification",
            "source": "olmoearth",
            "license": "internal",
            "provenance": {
                "url": SOURCE,
                "have_locally": True,
                "annotation_method": "manual / photointerpretation",
            },
            "sensors_relevant": ["sentinel2", "sentinel1", "landsat"],
            "classes": [
                {"id": i, "name": name, "description": desc}
                for i, (name, desc) in enumerate(CLASSES)
            ],
            "nodata_value": io.CLASS_NODATA,
            "num_samples": len(selected),
            "class_counts": {name: counts.get(name, 0) for name, _ in CLASSES},
            "notes": "1x1 point-segmentation patches over Africa; both train+test source splits used; ~1-year time range per point (labeled year, 2019 or 2020). not_crop capped at 1000; crop kept in full.",
        },
    )
    manifest.write_registry_entry(
        SLUG, "completed", task_type="classification", num_samples=len(selected)
    )
    print("done")


if __name__ == "__main__":
    multiprocessing.set_start_method("forkserver", force=True)
    main()
