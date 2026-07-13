"""Process OlmoEarth land-cover change (deforestation & urban expansion) into
open-set-segmentation label patches.

Source: two local rslearn evals under olmoearth_evals -- ``lcc_deforestation`` and
``lcc_urban_expansion``. They share the exact same 13,812 window locations/times (same
group+name, CRS, bounds and 1-year post time_range); each window carries a per-tile binary
change ``category`` (positive/negative) in its metadata ``options``, and the label layer is
a full-window polygon of that category. The two datasets differ only in which change type
(deforestation vs urban expansion) each window is scored for.

We combine both source datasets into ONE binary change dataset (spec sec 5, "combine into
one dataset with a unified class scheme") by joining on (group, name):

    label = positive (1) if the window is a deforestation-positive OR an urban-expansion-
            positive tile (union), else negative (0).

The union keeps negatives coherent -- a negative tile is negative in BOTH source datasets,
so it genuinely has neither change type (avoids the contradiction that would arise from
treating each source separately, where a deforestation-negative tile can be urban-positive).

Each label is a coherent full-tile (64x64, 640 m) change annotation, so we write per-window
single-band uint8 GeoTIFFs (dense/scene-level, spec sec 2/4), NOT a point table. Every pixel
of a tile carries the tile's class id (0 negative, 1 positive).

CHANGE handling (spec sec 5): each window's stored time_range is a 1-year "post"
observation window; we set ``change_time`` to its midpoint (so time_range is centered on
change_time). NOTE / caveat: the annotation compares the post mosaic against a pre mosaic
taken ~3 years earlier (config ``time_offset: -1095d``), so the precise transition moment is
NOT resolvable to within this 1-year window -- change_time anchors the post-observation year
(when the changed state is fully visible), not an exact event date. See the summary.

Run: ``python3 -m olmoearth_pretrain.open_set_segmentation_data.datasets.olmoearth_land_cover_change_deforestation_urban_expansion``
"""

import argparse
import json
import multiprocessing
import os
from collections import Counter
from datetime import datetime
from typing import Any

import numpy as np
from rasterio.crs import CRS as RioCRS
from rslearn.utils.geometry import Projection

from olmoearth_pretrain.open_set_segmentation_data import io, manifest

SLUG = "olmoearth_land_cover_change_deforestation_urban_expansion"
SOURCES = {
    "deforestation": "/weka/dfive-default/rslearn-eai/datasets/olmoearth_evals/lcc_deforestation",
    "urban_expansion": "/weka/dfive-default/rslearn-eai/datasets/olmoearth_evals/lcc_urban_expansion",
}
PER_CLASS = 1000

CLASSES = [
    (
        "negative",
        "No land-cover change: the tile shows neither deforestation nor urban expansion "
        "between the pre mosaic (~3 years earlier) and the post mosaic. Negative in both "
        "the deforestation and urban-expansion source annotations.",
    ),
    (
        "positive",
        "Land-cover change present: deforestation and/or urban expansion occurred somewhere "
        "in the 640 m tile between the pre and post mosaics (union of the deforestation- and "
        "urban-expansion-positive annotations). The whole tile is labeled positive (tile-level "
        "change annotation).",
    ),
]


def _read_ds(source: str) -> dict[tuple[str, str], dict[str, Any]]:
    """Parallel-read one source dataset's window metadata keyed by (group, name)."""
    windows_root = os.path.join(source, "windows")
    jobs = []
    for group in os.listdir(windows_root):
        gd = os.path.join(windows_root, group)
        if os.path.isdir(gd):
            for name in os.listdir(gd):
                jobs.append((group, os.path.join(gd, name)))
    with multiprocessing.Pool(48) as p:
        recs = p.map(_read_one, jobs, chunksize=64)
    out = {}
    for r in recs:
        if r:
            out[(r["group"], r["name"])] = r
    return out


def _read_one(job: tuple[str, str]) -> dict[str, Any] | None:
    group, path = job
    try:
        with open(os.path.join(path, "metadata.json")) as f:
            md = json.load(f)
    except FileNotFoundError:
        return None
    opt = md.get("options", {})
    cat = opt.get("category")
    tr = md.get("time_range")
    proj = md.get("projection", {})
    bounds = md.get("bounds")
    if cat not in ("positive", "negative") or tr is None or bounds is None:
        return None
    return {
        "group": group,
        "name": os.path.basename(path),
        "category": cat,
        "crs": proj["crs"],
        "bounds": tuple(bounds),
        "time_range": tr,  # [iso, iso]
    }


def join_records() -> list[dict[str, Any]]:
    """Join the two source datasets on (group, name) into unified binary records."""
    defor = _read_ds(SOURCES["deforestation"])
    urban = _read_ds(SOURCES["urban_expansion"])
    keys = set(defor) | set(urban)
    records = []
    geo_mismatch = 0
    for k in keys:
        d = defor.get(k)
        u = urban.get(k)
        base = d or u  # both should exist; guard anyway
        if d and u and (d["bounds"] != u["bounds"] or d["crs"] != u["crs"]):
            geo_mismatch += 1
            # fall back to deforestation geometry as canonical
        types = []
        if d and d["category"] == "positive":
            types.append("deforestation")
        if u and u["category"] == "positive":
            types.append("urban_expansion")
        label = 1 if types else 0
        records.append(
            {
                "group": base["group"],
                "name": base["name"],
                "crs": base["crs"],
                "bounds": base["bounds"],
                "time_range": base["time_range"],
                "label": label,
                "types": types,
            }
        )
    if geo_mismatch:
        print(f"WARNING: {geo_mismatch} windows had cross-dataset geometry mismatch")
    return records


def _write_one(job: tuple[str, dict[str, Any]]) -> int:
    sample_id, r = job
    tif = io.locations_dir(SLUG) / f"{sample_id}.tif"
    if tif.exists():
        return r["label"]
    x_min, y_min, x_max, y_max = r["bounds"]
    w, h = x_max - x_min, y_max - y_min
    w, h = min(w, io.MAX_TILE), min(h, io.MAX_TILE)
    bounds = (x_min, y_min, x_min + w, y_min + h)
    projection = Projection(RioCRS.from_string(r["crs"]), io.RESOLUTION, -io.RESOLUTION)
    arr = np.full((h, w), r["label"], dtype=np.uint8)

    tr = r["time_range"]
    t0 = datetime.fromisoformat(tr[0])
    t1 = datetime.fromisoformat(tr[1])
    change_time = t0 + (t1 - t0) / 2

    source_id = f"{r['group']}/{r['name']}"
    if r["types"]:
        source_id += ";types=" + "+".join(r["types"])

    io.write_label_geotiff(SLUG, sample_id, arr, projection, bounds)
    io.write_sample_json(
        SLUG,
        sample_id,
        projection,
        bounds,
        (t0, t1),
        change_time=change_time,
        source_id=source_id,
        classes_present=[r["label"]],
    )
    return r["label"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=48)
    args = parser.parse_args()

    io.check_disk()
    manifest.write_registry_entry(SLUG, "in_progress")

    raw = io.raw_dir(SLUG)
    raw.mkdir(parents=True, exist_ok=True)
    with (raw / "SOURCE.txt").open("w") as f:
        f.write(
            "Local rslearn datasets (combined into one binary change dataset):\n"
            f"  deforestation: {SOURCES['deforestation']}\n"
            f"  urban_expansion: {SOURCES['urban_expansion']}\n"
            "Same 13,812 windows shared across both; label = union of positives "
            "(0 negative, 1 positive). Full-tile 64x64 change annotations.\n"
        )

    records = join_records()
    total_counts = Counter(r["label"] for r in records)
    print(f"joined {len(records)} windows; label counts {dict(total_counts)}")

    from olmoearth_pretrain.open_set_segmentation_data.sampling import balance_by_class

    selected = balance_by_class(records, "label", per_class=PER_CLASS)
    print(f"selected {len(selected)} (<= {PER_CLASS}/class)")

    jobs = [(f"{i:06d}", r) for i, r in enumerate(selected)]
    with multiprocessing.Pool(args.workers) as p:
        labels = p.map(_write_one, jobs, chunksize=32)
    counts = Counter(labels)
    print(f"wrote {len(labels)} patches; selected counts {dict(counts)}")

    io.write_dataset_metadata(
        SLUG,
        {
            "dataset": SLUG,
            "name": "OlmoEarth land-cover change (deforestation & urban expansion)",
            "task_type": "classification",
            "source": "olmoearth",
            "license": "internal",
            "provenance": {
                "url": "/weka/dfive-default/rslearn-eai/datasets/olmoearth_evals/{lcc_deforestation,lcc_urban_expansion}",
                "have_locally": True,
                "annotation_method": "manual annotation (per-tile binary change; pre vs post Sentinel-2 mosaics)",
            },
            "sensors_relevant": ["sentinel2", "sentinel1", "landsat"],
            "classes": [
                {"id": i, "name": name, "description": desc}
                for i, (name, desc) in enumerate(CLASSES)
            ],
            "nodata_value": io.CLASS_NODATA,
            "num_samples": len(selected),
            "class_counts": {
                CLASSES[k][0]: counts.get(k, 0) for k in range(len(CLASSES))
            },
            "available_before_balancing": {
                CLASSES[k][0]: total_counts.get(k, 0) for k in range(len(CLASSES))
            },
            "notes": (
                "Binary pre/post land-cover-change classification combining the olmoearth "
                "lcc_deforestation and lcc_urban_expansion evals (identical 13,812 window "
                "set). label = union of positives. Full-tile 64x64 uint8 patches (uniform "
                "class; tile-level change annotation, not a precise sub-tile change mask). "
                "change_time = midpoint of each window's 1-year post time_range; the actual "
                "transition happened between a pre mosaic ~3 years earlier and the post "
                "mosaic, so change_time is a coarse (post-year) anchor, not an exact date "
                "-- see summary."
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
