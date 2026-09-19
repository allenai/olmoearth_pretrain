"""Process OlmoEarth Tolbi agroforestry into open-set-segmentation labels.

Source: the Tolbi tropical tree-crop / agroforestry project over Ivory Coast (West
Africa). The manifest url ``/weka/dfive-default/rslearn-eai/datasets/tolbi`` (a weka
mirror of ``gs://rslearn-eai/datasets/tolbi``) is not present on disk, but the same
dataset is materialized on weka as the eval dataset ``eval_datasets/tolbi_crop`` (group
``20251210``), created by ``rslp/tolbi/create_windows.py``.

Each window is a 31x31 px UTM-10m tile whose label_raster carries the class id at the
single center pixel (rest 0); the class name and split are in window ``options`` and the
reference year in ``time_range``. So although the manifest calls this ``polygons``, the
materialized labels are effectively SPARSE POINTS: positive cash-crop points (cacao,
rubber) sampled from the Tolbi team's ground-truth polygons (reference year 2024), plus
negative land-cover points (tree, shrub, others) from ESA WorldCover reference clusters
(reference year 2016). We therefore emit one dataset-wide point table (points.geojson,
spec 2a), balanced to <=1000 per class.

Notes on the class set: the manifest lists 6 classes (cacao, palmoil, rubber, tree,
shrub, others) but the materialized dataset contains NO ``palmoil`` samples, so palmoil
is documented and omitted. The 5 present classes are all in the Sentinel era (2016+).
"""

import argparse
import json
import multiprocessing
import os
from collections import Counter
from typing import Any

from pyproj import CRS, Transformer

from olmoearth_pretrain.open_set_segmentation_data import io, manifest
from olmoearth_pretrain.open_set_segmentation_data.sampling import balance_by_class

SLUG = "olmoearth_tolbi_agroforestry"
SOURCE = "/weka/dfive-default/olmoearth/eval_datasets/tolbi_crop"
GROUP = "20251210"
PER_CLASS = 1000

# Class ordering -> id (0-indexed), with definitions. Only classes actually present in
# the materialized dataset are included (palmoil listed in the manifest is absent).
CLASSES = [
    (
        "cacao",
        "Cacao (Theobroma cacao) tree-crop plots, from Tolbi ground-truth polygons.",
    ),
    (
        "rubber",
        "Rubber (Hevea brasiliensis) plantation, from Tolbi ground-truth polygons.",
    ),
    ("tree", "Natural / non-cash-crop tree cover (ESA WorldCover reference negative)."),
    ("shrub", "Shrubland (ESA WorldCover reference negative)."),
    (
        "others",
        "Other land cover: water, built-up, bare soil, cropland, etc. (ESA WorldCover reference negative).",
    ),
]
NAME_TO_ID = {name: i for i, (name, _d) in enumerate(CLASSES)}

_TRANSFORMERS: dict[str, Transformer] = {}


def _to_lonlat(crs_str: str, x: float, y: float) -> tuple[float, float]:
    tf = _TRANSFORMERS.get(crs_str)
    if tf is None:
        tf = Transformer.from_crs(
            CRS.from_string(crs_str), CRS.from_epsg(4326), always_xy=True
        )
        _TRANSFORMERS[crs_str] = tf
    lon, lat = tf.transform(x, y)
    return lon, lat


def _read_one(path: str) -> dict[str, Any] | None:
    try:
        with open(os.path.join(path, "metadata.json")) as f:
            md = json.load(f)
    except FileNotFoundError:
        return None
    opt = md.get("options", {})
    cat = opt.get("category")
    if cat not in NAME_TO_ID:
        return None
    proj = md["projection"]
    crs_str = proj["crs"]
    xr, yr = proj["x_resolution"], proj["y_resolution"]
    b = md["bounds"]
    # Center pixel of the window; geo coord = pixel-center * resolution.
    ccol = (b[0] + b[2]) / 2.0
    crow = (b[1] + b[3]) / 2.0
    gx = (ccol + 0.5) * xr
    gy = (crow + 0.5) * yr
    lon, lat = _to_lonlat(crs_str, gx, gy)
    tr = md.get("time_range")
    year = int(tr[0][:4]) if tr else 2020
    return {
        "lon": lon,
        "lat": lat,
        "label": cat,
        "year": year,
        "source_id": f"{GROUP}/{os.path.basename(path)}",
    }


def scan_records() -> list[dict[str, Any]]:
    windows_root = os.path.join(SOURCE, "windows", GROUP)
    jobs = [os.path.join(windows_root, n) for n in os.listdir(windows_root)]
    jobs = [j for j in jobs if os.path.isdir(j)]
    with multiprocessing.Pool(64) as p:
        recs = [r for r in p.map(_read_one, jobs, chunksize=128) if r]
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
        f.write(
            "Tolbi agroforestry (Ivory Coast tree-crop mapping).\n"
            "manifest url (not on disk): /weka/dfive-default/rslearn-eai/datasets/tolbi "
            "(mirror of gs://rslearn-eai/datasets/tolbi)\n"
            f"materialized copy used: {SOURCE} group {GROUP}\n"
            "Each window carries a single center-pixel class label (sparse points).\n"
        )

    recs = scan_records()
    print(f"scanned {len(recs)} labeled points")
    raw_counts = Counter(r["label"] for r in recs)
    print("raw class counts:", dict(raw_counts))

    selected = balance_by_class(recs, "label", per_class=PER_CLASS)
    print(f"selected {len(selected)} (<= {PER_CLASS}/class)")

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
            "name": "OlmoEarth Tolbi agroforestry",
            "task_type": "classification",
            "source": "olmoearth",
            "license": "internal",
            "provenance": {
                "url": "/weka/dfive-default/rslearn-eai/datasets/tolbi",
                "have_locally": True,
                "annotation_method": "manual annotation (ground-truth cash-crop polygons) + ESA WorldCover reference negatives",
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
                "Sparse 1x1 point-segmentation labels (materialized center-pixel labels; "
                "manifest label_type 'polygons' but on-disk labels are points). All source "
                "splits used. Time range = reference year (1-year window): positives (cacao, "
                "rubber) year 2024 from Tolbi ground-truth polygons; negatives (tree, shrub, "
                "others) year 2016 from ESA WorldCover reference clusters. Manifest class "
                "'palmoil' has zero materialized samples and is omitted. lon/lat computed from "
                "each window's center pixel in its UTM projection."
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
