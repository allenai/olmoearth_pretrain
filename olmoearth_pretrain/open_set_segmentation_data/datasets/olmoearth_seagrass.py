"""Process OlmoEarth seagrass point supervision into open-set-segmentation labels.

Source: local rslearn project at ``/weka/dfive-default/piperw/rslearn_projects/data/
seagrass``. The ``baleares_official_2025`` window group holds ~40k manually-annotated
Sentinel-2 point labels over the Balearic Islands: each 64x64 window has exactly one
labeled 10 m pixel (the rest is nodata=255), with the class id, class name, and the
point's lon/lat stored in window ``metadata.json`` ``options``. Two classes are present in
the point data: ``background`` (source label 0) and ``dense_seagrass`` (source label 2);
we remap to contiguous ids 0/1 (the config also lists an unused ``sparse_seagrass``).

This is a pure sparse-point dataset, so we write one dataset-wide GeoJSON point table
(points.geojson, spec 2a), balanced to <=1000 per class. The separate
``baleares_official_eval`` group (276 dense 512x512 polygon-derived tiles, held-out
evaluation) is a different label modality and is not included in the point table.
"""

import argparse
import json
import multiprocessing
import os
from collections import Counter
from typing import Any

from olmoearth_pretrain.open_set_segmentation_data import io, manifest
from olmoearth_pretrain.open_set_segmentation_data.sampling import balance_by_class

SLUG = "olmoearth_seagrass"
SOURCE = "/weka/dfive-default/piperw/rslearn_projects/data/seagrass"
POINT_GROUP = "baleares_official_2025"
PER_CLASS = 1000

# Source label id -> (new contiguous id, name, description).
CLASSES = [
    (0, "background", "Non-seagrass sea/land background pixel (manually annotated)."),
    (
        1,
        "dense_seagrass",
        "Dense seagrass meadow (primarily Posidonia oceanica) visible in Sentinel-2, "
        "manually annotated over the Balearic Islands.",
    ),
]
SOURCE_LABEL_TO_ID = {0: 0, 2: 1}
ID_TO_NAME = {i: name for i, name, _ in CLASSES}


def _read_one(path: str) -> dict[str, Any] | None:
    try:
        with open(os.path.join(path, "metadata.json")) as f:
            md = json.load(f)
    except FileNotFoundError:
        return None
    opt = md.get("options", {})
    tr = md.get("time_range")
    lon, lat = opt.get("longitude"), opt.get("latitude")
    src_label = opt.get("label")
    if lon is None or lat is None or src_label not in SOURCE_LABEL_TO_ID or not tr:
        return None
    return {
        "lon": lon,
        "lat": lat,
        "label": SOURCE_LABEL_TO_ID[src_label],
        "time_range": tr,
        "source_id": f"{os.path.basename(os.path.dirname(path))}/{os.path.basename(path)}",
    }


def scan_records(workers: int) -> list[dict[str, Any]]:
    group_dir = os.path.join(SOURCE, "windows", POINT_GROUP)
    jobs = [
        os.path.join(group_dir, name)
        for name in os.listdir(group_dir)
        if os.path.isdir(os.path.join(group_dir, name))
    ]
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
        f.write(f"point group used: {POINT_GROUP}\n")

    recs = scan_records(args.workers)
    print(f"scanned {len(recs)} labeled points")
    print("raw distribution:", Counter(ID_TO_NAME[r["label"]] for r in recs))

    selected = balance_by_class(recs, "label", per_class=PER_CLASS)
    counts = Counter(r["label"] for r in selected)
    print(
        f"selected {len(selected)} (<= {PER_CLASS}/class):",
        {ID_TO_NAME[k]: v for k, v in counts.items()},
    )

    points = [
        {
            "id": f"{i:06d}",
            "lon": r["lon"],
            "lat": r["lat"],
            "label": r["label"],
            "time_range": r["time_range"],
            "source_id": r["source_id"],
        }
        for i, r in enumerate(selected)
    ]
    io.write_points_table(SLUG, "classification", points)

    io.write_dataset_metadata(
        SLUG,
        {
            "dataset": SLUG,
            "name": "OlmoEarth seagrass",
            "task_type": "classification",
            "source": "olmoearth",
            "license": "internal",
            "provenance": {
                "url": SOURCE,
                "have_locally": True,
                "annotation_method": "manual annotation",
            },
            "sensors_relevant": ["sentinel2", "sentinel1", "landsat"],
            "classes": [
                {"id": i, "name": name, "description": desc}
                for i, name, desc in CLASSES
            ],
            "nodata_value": io.CLASS_NODATA,
            "num_samples": len(selected),
            "class_counts": {ID_TO_NAME[k]: v for k, v in counts.items()},
            "notes": (
                "Sparse point segmentation (1x1 labels) over the Balearic Islands; "
                "one point label per source window. Two classes present in the point "
                "data (background, dense_seagrass); config's unused sparse_seagrass "
                "dropped. ~1-year time range per point (2025) from source window. "
                "Held-out dense polygon eval tiles (baleares_official_eval) excluded."
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
