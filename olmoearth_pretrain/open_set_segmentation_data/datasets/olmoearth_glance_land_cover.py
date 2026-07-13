"""Process OlmoEarth GLanCE land cover into open-set-segmentation label patches.

Source: local rslearn eval at olmoearth_evals/glance. Each window is a 32x32 UTM context
tile whose single center pixel carries one land-cover class (GLanCE-derived, 11-class
OlmoEarth legend). The class name, lon/lat, and a ~1-year time range live in the window
metadata.json ``options`` (and are mirrored in the label / label_raster layers). This is a
pure sparse-point segmentation dataset, so we write one dataset-wide point table
(points.geojson, spec 2a) balanced to <=1000 per class.

Relationship note: this is the GLanCE *map/derived* product. The manifest also lists the
upstream manually-photointerpreted reference ("GLanCE Global Land Cover Training Data",
have_locally=false, 7-class GLanCE legend). That reference is a separate, external dataset
with a different legend; this local eval is a distinct 11-class OlmoEarth product and is
processed on its own (the pairing is recorded in the summary).
"""

import argparse
import multiprocessing
import os
from collections import Counter
from typing import Any

from olmoearth_pretrain.open_set_segmentation_data import io, manifest
from olmoearth_pretrain.open_set_segmentation_data.sampling import balance_by_class

SLUG = "olmoearth_glance_land_cover"
SOURCE = "/weka/dfive-default/rslearn-eai/datasets/olmoearth_evals/glance"
PER_CLASS = 1000

# Manifest class ordering -> id. The source label_raster values already equal these ids
# (verified: options.label name maps to the same integer as the raster pixel value).
CLASSES = [
    ("water", "Open water bodies (lakes, rivers, ocean, reservoirs)."),
    ("evergreen", "Evergreen tree-dominated forest / woodland."),
    ("deciduous", "Deciduous tree-dominated forest / woodland."),
    ("agriculture", "Cultivated cropland and managed agricultural land."),
    ("grassland", "Herbaceous grass-dominated vegetation."),
    ("mixed", "Mixed vegetation (mixed forest / heterogeneous vegetated cover)."),
    (
        "developed",
        "Human-made structures / impervious surfaces (urban, roads, buildings).",
    ),
    ("sand", "Sand surfaces (beaches, dunes, sandy barren)."),
    ("shrub", "Shrub-dominated vegetation."),
    ("rock", "Exposed rock / bedrock barren."),
    ("soil", "Bare soil / barren ground."),
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
        # Preserve the source's own ~1-year window (already an ISO [start, end) pair).
        "time_range": tr,
        "year": int(tr[0][:4]) if tr else 2019,
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
                "time_range": r["time_range"] or io.year_range(r["year"]),
                "source_id": r["source_id"],
            }
        )
    io.write_points_table(SLUG, "classification", points)

    counts = Counter(r["label"] for r in selected)
    io.write_dataset_metadata(
        SLUG,
        {
            "dataset": SLUG,
            "name": "OlmoEarth GLanCE land cover",
            "task_type": "classification",
            "source": "olmoearth",
            "license": "internal",
            "provenance": {
                "url": SOURCE,
                "have_locally": True,
                "annotation_method": "derived-product (GLanCE land cover)",
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
                "1x1 point-segmentation labels (single labeled center pixel per source "
                "window); all source splits (train+test) used; ~1-year time range per point "
                "taken from the source window (labeled year 2017-2020)."
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
