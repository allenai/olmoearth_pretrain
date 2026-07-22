"""Process OlmoEarth Canada crops (fine) into open-set-segmentation labels.

Source: local rslearn eval at olmoearth_evals/canada_crops_fine. Each window is one
fine-grained crop-type point over Canadian farmland, derived from the AAFC Annual Crop
Inventory (ACI). The class name, lon/lat, and a ~1-year time range are stored in the
window metadata.json ``options`` (and mirrored in the label vector layer). Sparse point
segmentation, so we write one dataset-wide point table (points.geojson, spec §2a),
balanced to <=1000 per class. All 24 fine classes are post-2016; no filtering needed.
"""

import argparse
import json
import multiprocessing
import os
from collections import Counter
from typing import Any

from olmoearth_pretrain.open_set_segmentation_data import io, manifest
from olmoearth_pretrain.open_set_segmentation_data.sampling import balance_by_class

SLUG = "olmoearth_canada_crops_fine"
SOURCE = "/weka/dfive-default/rslearn-eai/datasets/olmoearth_evals/canada_crops_fine"
PER_CLASS = 1000

# Class ids in descending observed frequency, with short AAFC ACI class definitions.
CLASSES = [
    (
        "Mixed Forage",
        "Fields of mixed forage crops (grasses and legumes grown for hay/silage).",
    ),
    ("Soybeans", "Soybean (Glycine max) cropland."),
    ("Corn", "Maize / corn (Zea mays) cropland."),
    ("Pasture", "Managed (improved) pasture and grazing land."),
    ("Winter Wheat", "Fall-seeded winter wheat cropland."),
    ("Mixedwood", "Mixedwood forest (mixed coniferous and deciduous trees)."),
    ("Urban", "Developed / built-up land (urban areas and infrastructure)."),
    ("Alfalfa", "Alfalfa (Medicago sativa) forage cropland."),
    (
        "Unimproved Pasture",
        "Native / unimproved pasture not managed by seeding or fertilization.",
    ),
    ("Shrubland", "Shrub-dominated natural vegetation."),
    (
        "Wetland",
        "Wetlands (marsh, bog, swamp and other seasonally/permanently saturated land).",
    ),
    (
        "Abandoned (Overgrown)",
        "Abandoned farmland reverting to grass/herbaceous vegetation (overgrown).",
    ),
    ("Abandoned (Shrubs)", "Abandoned farmland reverting to shrub cover."),
    ("Coniferous", "Coniferous (evergreen needleleaf) forest."),
    ("Potatoes", "Potato (Solanum tuberosum) cropland."),
    ("Barren", "Barren land with little to no vegetation."),
    ("Oats", "Oat (Avena sativa) cropland."),
    ("Blueberry (Undiff)", "Blueberry cultivation (undifferentiated)."),
    ("Barley (Undiff)", "Barley (Hordeum vulgare) cropland (undifferentiated)."),
    ("Water", "Open water."),
    ("Spring Wheat", "Spring-seeded wheat cropland."),
    ("Pasture/Forage", "Pasture and forage land (undifferentiated)."),
    ("Canola/Rapeseed", "Canola / rapeseed (Brassica) oilseed cropland."),
    ("Native Grassland", "Native grassland (natural herbaceous vegetation)."),
]
NAME_TO_ID = {name: i for i, (name, _desc) in enumerate(CLASSES)}


def _read_one(path: str) -> dict[str, Any] | None:
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

    # Sparse point dataset -> one dataset-wide point table (spec §2a), not per-point tifs.
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
            "name": "OlmoEarth Canada crops (fine)",
            "task_type": "classification",
            "source": "olmoearth",
            "license": "internal",
            "provenance": {
                "url": SOURCE,
                "have_locally": True,
                "annotation_method": "derived-product (AAFC Annual Crop Inventory)",
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
                "1x1 point-segmentation labels over Canadian farmland; all source splits "
                "(train+test) used; ~1-year time range per point anchored on the ACI "
                "labeled year (2016-2021). Class ids ordered by descending frequency."
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
