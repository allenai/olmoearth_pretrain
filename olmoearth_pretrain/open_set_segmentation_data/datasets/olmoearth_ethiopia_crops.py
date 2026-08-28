"""Process OlmoEarth Ethiopia crops into open-set-segmentation label points.

Source: local rslearn eval at olmoearth_evals/ethiopia_crops. Each window is one
manually field-surveyed crop-type point (wheat / barley / maize / teff), with the class
name, lon/lat, and a ~1-year growing-season time range stored in window metadata.json
``options`` (and duplicated in the ``label`` vector layer). Sparse point segmentation, so
we write one dataset-wide point table (points.geojson, spec 2a), balanced to <=1000 per
class. Every window's time range is a single leap-year growing-season window anchored on
the labeled 2019/2020 season, which we preserve verbatim (better than snapping to Jan-Jan).
"""

import argparse
import multiprocessing
import os
from typing import Any

from olmoearth_pretrain.open_set_segmentation_data import io, manifest
from olmoearth_pretrain.open_set_segmentation_data.sampling import balance_by_class

SLUG = "olmoearth_ethiopia_crops"
SOURCE = "/weka/dfive-default/rslearn-eai/datasets/olmoearth_evals/ethiopia_crops"
PER_CLASS = 1000

# Class ordering follows the manifest -> id, with short Ethiopian staple-cereal definitions.
CLASSES = [
    ("wheat", "Wheat (Triticum spp.) cereal fields; a dominant Ethiopian staple crop."),
    (
        "barley",
        "Barley (Hordeum vulgare) cereal fields, common in the Ethiopian highlands.",
    ),
    ("maize", "Maize / corn (Zea mays) fields."),
    ("teff", "Teff (Eragrostis tef), a fine-grain cereal staple endemic to Ethiopia."),
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
        # Preserve the source growing-season window verbatim (already <=~1 year).
        "time_range": tr,
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
                "time_range": r["time_range"],
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
            "name": "OlmoEarth Ethiopia crops",
            "task_type": "classification",
            "source": "olmoearth",
            "license": "internal",
            "provenance": {
                "url": SOURCE,
                "have_locally": True,
                "annotation_method": "manual field survey",
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
                "1x1 point-segmentation labels; all source splits (train+test) used; "
                "each point keeps its source ~1-year growing-season time range anchored "
                "on the labeled 2019/2020 season. wheat truncated to 1000 (2077 available); "
                "barley/maize/teff kept in full (rare classes retained per spec 5)."
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
