"""Process OlmoEarth Descals oil palm into open-set-segmentation label patches.

Source: local rslearn eval at olmoearth_evals/descals. Each window is one photo-interpreted
validation point over the Descals et al. (2021) global oil-palm map, tagged with an oil-palm
plantation-type class (Industrial / Smallholder / Other), lon/lat, and a ~1-year time range
(2019-2021) in window metadata.json ``options``. This is sparse point segmentation, so we
write one dataset-wide point table (points.geojson, spec 2a), balanced to <=1000 per class.
"""

import argparse
import multiprocessing
import os
from collections import Counter
from typing import Any

from olmoearth_pretrain.open_set_segmentation_data import io, manifest
from olmoearth_pretrain.open_set_segmentation_data.sampling import balance_by_class

SLUG = "olmoearth_descals_oil_palm"
SOURCE = "/weka/dfive-default/rslearn-eai/datasets/olmoearth_evals/descals"
PER_CLASS = 1000

# Class ordering (manifest order) -> id, with Descals oil-palm plantation-type definitions.
CLASSES = [
    (
        "Industrial oil palm",
        "Large-scale industrial closed-canopy oil-palm plantations with regular, "
        "geometric planting patterns (Descals et al. global oil-palm class 1).",
    ),
    (
        "Smallholder oil palm",
        "Smallholder oil-palm plantations, typically smaller and with less-regular "
        "planting patterns than industrial estates (Descals et al. class 2).",
    ),
    (
        "Other",
        "Non-oil-palm land (any other land cover); background/negative class.",
    ),
]
NAME_TO_ID = {name: i for i, (name, _desc) in enumerate(CLASSES)}


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
        "year": int(tr[0][:4]) if tr else 2020,
        "source_id": f"{os.path.basename(os.path.dirname(path))}/{os.path.basename(path)}",
    }


def scan_records(workers: int) -> list[dict[str, Any]]:
    """Parallel-read window metadata into flat records."""
    jobs = []
    windows_root = os.path.join(SOURCE, "windows")
    for group in os.listdir(windows_root):
        gd = os.path.join(windows_root, group)
        if os.path.isdir(gd):
            for name in os.listdir(gd):
                jobs.append(os.path.join(gd, name))
    with multiprocessing.Pool(workers) as p:
        recs = [r for r in p.map(_read_one, jobs, chunksize=64) if r]
    return recs


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
    raw_counts = Counter(r["label"] for r in recs)
    print(f"raw class counts: {dict(raw_counts)}")

    selected = balance_by_class(recs, "label", per_class=PER_CLASS)
    print(f"selected {len(selected)} (<= {PER_CLASS}/class)")

    # Sparse point dataset -> one dataset-wide point table (spec 2a), not per-point tifs.
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
            "name": "OlmoEarth Descals oil palm",
            "task_type": "classification",
            "source": "olmoearth",
            "license": "CC-BY-4.0",
            "provenance": {
                "url": SOURCE,
                "have_locally": True,
                "annotation_method": "derived-product with photointerpreted validation",
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
                "1x1 point-segmentation patches; all source splits (test+train) used; "
                "~1-year time range per point (Descals labeled year 2019-2021). "
                "'Other' capped at 1000; oil-palm classes kept in full (rare-class "
                "preservation, spec 5)."
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
