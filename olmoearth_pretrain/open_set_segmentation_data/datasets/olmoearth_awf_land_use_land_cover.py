"""Process OlmoEarth AWF land-use/land-cover into open-set-segmentation labels.

Source: local rslearn eval at rslearn-eai/datasets/crop/awf_2023. Ground-truth reference
points live only in the ``20250822`` window group (1459 windows); the ``amboseli`` and
``kenya`` groups are unlabeled prediction/eval tiles (no label layer) and are skipped.

Each labeled window is a 32x32 (320 m) tile whose ``label_raster`` is background (0)
everywhere except a single center pixel (16,16) carrying the reference class id (1-9); the
window metadata ``options`` carry the class string (``lulc``) plus the reference point's
lon/lat. This is manually-annotated sparse-point land cover, so we write one dataset-wide
point table (points.geojson, spec 2a), one Point feature per reference point. Labels are
from 2023 imagery -> a 1-year (2023) time range per point. 9 classes, all kept.
"""

import argparse
import json
import multiprocessing
import os
from collections import Counter
from typing import Any

from olmoearth_pretrain.open_set_segmentation_data import io, manifest
from olmoearth_pretrain.open_set_segmentation_data.sampling import balance_by_class

SLUG = "olmoearth_awf_land_use_land_cover"
SOURCE = "/weka/dfive-default/rslearn-eai/datasets/crop/awf_2023"
LABELED_GROUP = "20250822"
PER_CLASS = 1000
YEAR = 2023

# Class ids in manifest order (0-based). Source ``lulc`` strings map here; the source
# label_raster uses 1-based ids that we do not reuse.
CLASSES = [
    (
        "Agriculture/Settlement",
        "Cultivated cropland and rural settlement / smallholder farming mosaic.",
    ),
    ("Grassland/barren", "Open grassland and sparsely-vegetated / bare ground."),
    (
        "Herbaceous wetland",
        "Seasonally or permanently waterlogged land with herbaceous (non-woody) vegetation.",
    ),
    (
        "Lava forest",
        "Forest established on volcanic lava substrate (distinctive AWF landscape class).",
    ),
    ("Montane forest", "Closed-canopy forest on higher-elevation montane terrain."),
    ("Open water", "Lakes, rivers, ponds and other standing/flowing surface water."),
    ("Shrubland/Savanna", "Shrub-dominated bushland and wooded savanna."),
    (
        "Urban/dense development",
        "Built-up urban areas and dense impervious development.",
    ),
    (
        "Woodland forest (>40% canopy)",
        "Woodland with greater than 40% tree canopy cover.",
    ),
]
# Map source lulc string -> manifest class id (0-based).
NAME_TO_ID = {name: i for i, (name, _desc) in enumerate(CLASSES)}
# Manifest uses a shortened woodland name; accept both spellings.
NAME_TO_ID["Woodland forest"] = NAME_TO_ID["Woodland forest (>40% canopy)"]


def _read_one(path: str) -> dict[str, Any] | None:
    try:
        with open(os.path.join(path, "metadata.json")) as f:
            md = json.load(f)
    except FileNotFoundError:
        return None
    opt = md.get("options", {})
    lulc = opt.get("lulc")
    if lulc not in NAME_TO_ID:
        return None
    if opt.get("longitude") is None or opt.get("latitude") is None:
        return None
    return {
        "lon": opt["longitude"],
        "lat": opt["latitude"],
        "label": NAME_TO_ID[lulc],
        "lulc": lulc,
        "source_id": f"{os.path.basename(os.path.dirname(path))}/{os.path.basename(path)}",
    }


def scan_records() -> list[dict[str, Any]]:
    gd = os.path.join(SOURCE, "windows", LABELED_GROUP)
    jobs = [os.path.join(gd, name) for name in os.listdir(gd)]
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
        f.write(
            f"labeled window group: {LABELED_GROUP} (only group with ground-truth labels)\n"
        )

    recs = scan_records()
    print(f"scanned {len(recs)} labeled reference points")
    selected = balance_by_class(recs, "label", per_class=PER_CLASS)
    print(f"selected {len(selected)} (<= {PER_CLASS}/class, cap 25000)")

    points = []
    for i, r in enumerate(selected):
        points.append(
            {
                "id": f"{i:06d}",
                "lon": r["lon"],
                "lat": r["lat"],
                "label": r["label"],
                "time_range": io.year_range(YEAR),
                "source_id": r["source_id"],
            }
        )
    io.write_points_table(SLUG, "classification", points)

    counts = Counter(r["label"] for r in selected)
    io.write_dataset_metadata(
        SLUG,
        {
            "dataset": SLUG,
            "name": "OlmoEarth AWF land use/land cover",
            "task_type": "classification",
            "source": "olmoearth",
            "license": "internal",
            "provenance": {
                "url": SOURCE,
                "have_locally": True,
                "annotation_method": "manual reference points",
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
                "Sparse-point land cover (1x1 points) over an African Wildlife Foundation "
                "landscape in Kenya; labels from 2023 imagery -> 1-year (2023) time range. "
                "Only the '20250822' window group carries ground-truth labels; 'amboseli' "
                "and 'kenya' groups are unlabeled prediction tiles and are excluded. "
                "Rare classes kept (Lava forest 18, Herbaceous wetland 49, Open water 55)."
            ),
        },
    )
    manifest.write_registry_entry(
        SLUG, "completed", task_type="classification", num_samples=len(selected)
    )
    print("done:", dict(counts))


if __name__ == "__main__":
    multiprocessing.set_start_method("forkserver", force=True)
    main()
