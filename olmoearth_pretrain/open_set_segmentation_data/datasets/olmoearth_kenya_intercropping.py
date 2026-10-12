"""Process OlmoEarth Kenya intercropping into open-set-segmentation label points.

Source: local rslearn eval at ``olmoearth_evals/kenya_intercropping`` (manifest
``have_locally: true``; ``url`` points at
``/weka/dfive-default/rslearn-eai/datasets/olmoearth_evals/kenya_intercropping/``). Kenyan
smallholder-field cropping-system labels (monocrop vs intercrop vs other) derived from a
manual field survey (Copernicus4GEOGLAM ground points), one class per surveyed field.

On-disk form: 8,285 windows across train/val/test groups, each a 64x64 patch at 10 m in a
local UTM CRS (EPSG:32736, UTM 36S). The registry tags this ``dense_raster``, but every
window's ``label_raster`` layer labels **exactly one pixel** — the surveyed point at the
window center (row 32, col 32) — with the rest = 255 (nodata). The ``label`` vector layer
is only the window-footprint box, not a real field boundary. So the honest ground truth is
a single 10 m pixel per window: this is a **pure sparse-point** dataset (spec 2/2a), NOT a
dense raster. We therefore write ONE dataset-wide GeoJSON point table
(``points.geojson``) — writing 64x64 tiles would fabricate labels for unobserved
neighboring pixels, which spec 2 forbids.

Task: per-pixel **classification** (cropping system). Class ids follow the source
``label_raster`` encoding verbatim: 0=intercrop, 1=monocrop, 2=other. (These map 1:1 to the
window ``category`` string: intercrop->0, monocrop->1, other->2.) The manifest's
``["background","monocrop","intercrop"]`` blurb is superseded by the actual on-disk
categories (``other`` in place of ``background``); "other" is a real surveyed non-inter/mono
class, so we keep it rather than fabricating negatives (assembly adds cross-dataset
negatives, spec 5).

Point location: the exact center of the single labeled pixel, transformed from the window's
UTM projection to WGS84 lon/lat (GeoJSON native CRS; pretraining reprojects onto the S2
grid).

Time range: crop labels are seasonal -> the source growing-season window
[2022-10-01, 2023-03-31) (~6 months, <= 1 year), preserved verbatim. Post-2016. (The
manifest's 2019-2021 hint does not match the on-disk windows, which are the 2022/23 short-
rains season; we trust the on-disk time_range.)

Sampling: classification, up to 1000 locations per class, class-balanced (spec 5); 3
classes, well under the 25k cap and the 254-class uint8 cap.

Reproduce (idempotent; regenerates points.geojson deterministically):
    python3 -m olmoearth_pretrain.open_set_segmentation_data.datasets.olmoearth_kenya_intercropping
"""

import argparse
import json
import multiprocessing
import os
from collections import Counter
from typing import Any

import numpy as np
import shapely
import tqdm
from rasterio.crs import CRS
from rslearn.const import WGS84_PROJECTION
from rslearn.utils.geometry import Projection, STGeometry
from rslearn.utils.mp import star_imap_unordered
from rslearn.utils.raster_format import GeotiffRasterFormat
from upath import UPath

from olmoearth_pretrain.open_set_segmentation_data import io, manifest
from olmoearth_pretrain.open_set_segmentation_data.sampling import balance_by_class

SLUG = "olmoearth_kenya_intercropping"
NAME = "OlmoEarth Kenya intercropping"
SOURCE = "/weka/dfive-default/rslearn-eai/datasets/olmoearth_evals/kenya_intercropping"
PER_CLASS = 1000
RES = io.RESOLUTION  # 10 m

# Source label_raster value -> (class name, description). Ids ARE the source encoding.
CLASSES = [
    (
        0,
        "intercrop",
        "Smallholder field with two or more crops interplanted in the same plot "
        "(intercropping), from a Kenyan Copernicus4GEOGLAM field survey.",
    ),
    (1, "monocrop", "Smallholder field planted with a single crop (monocropping)."),
    (
        2,
        "other",
        "Other land cover — a surveyed point that is neither a monocropped nor an "
        "intercropped field (the field survey's residual/'other' class).",
    ),
]
VALID_VALUES = {c[0] for c in CLASSES}


def _list_windows() -> list[str]:
    windows_root = os.path.join(SOURCE, "windows")
    jobs = []
    for group in sorted(os.listdir(windows_root)):
        gd = os.path.join(windows_root, group)
        if os.path.isdir(gd):
            for name in sorted(os.listdir(gd)):
                jobs.append(os.path.join(gd, name))
    return jobs


def _read_one(path: str) -> dict[str, Any] | None:
    """Read one window: the single labeled pixel (class + exact lon/lat) + time range."""
    try:
        with open(os.path.join(path, "metadata.json")) as f:
            md = json.load(f)
    except FileNotFoundError:
        return None
    proj_md = md["projection"]
    crs = proj_md["crs"]
    bounds = md[
        "bounds"
    ]  # pixel coords under the projection [x_min, y_min, x_max, y_max]
    tr = md.get("time_range")
    projection = Projection(CRS.from_string(crs), RES, -RES)

    # Read the label_raster aligned to the window; find the single labeled pixel.
    raster_dir = UPath(path) / "layers" / "label_raster" / "label"
    try:
        raster = GeotiffRasterFormat().decode_raster(
            raster_dir, projection, tuple(bounds)
        )
    except Exception:
        return None
    arr = raster.get_chw_array()[0]  # (H, W)
    rows, cols = np.where(arr != io.CLASS_NODATA)
    if len(rows) == 0:
        return None
    # Use the first (expected: only) labeled pixel.
    lr, lc = int(rows[0]), int(cols[0])
    value = int(arr[lr, lc])
    if value not in VALID_VALUES:
        return None
    # Absolute pixel coords = window origin + local index; center of pixel -> WGS84.
    abs_col = bounds[0] + lc + 0.5
    abs_row = bounds[1] + lr + 0.5
    geom = STGeometry(projection, shapely.Point(abs_col, abs_row), None).to_projection(
        WGS84_PROJECTION
    )
    return {
        "lon": float(geom.shp.x),
        "lat": float(geom.shp.y),
        "label": value,
        "time_range": tr,
        "source_id": f"{os.path.basename(os.path.dirname(path))}/{os.path.basename(path)}",
    }


def scan_records(workers: int) -> list[dict[str, Any]]:
    jobs = _list_windows()
    recs: list[dict[str, Any]] = []
    with multiprocessing.Pool(workers) as pool:
        for r in tqdm.tqdm(
            star_imap_unordered(pool, _read_one, [dict(path=p) for p in jobs]),
            total=len(jobs),
        ):
            if r:
                recs.append(r)
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
    print("raw class counts:", Counter(r["label"] for r in recs))

    selected = balance_by_class(recs, "label", per_class=PER_CLASS)
    print(f"selected {len(selected)} (<= {PER_CLASS}/class)")

    id_to_name = {cid: name for cid, name, _ in CLASSES}
    points = []
    for i, r in enumerate(selected):
        points.append(
            {
                "id": f"{i:06d}",
                "lon": r["lon"],
                "lat": r["lat"],
                "label": r["label"],
                "time_range": r["time_range"],
                "source_id": r["source_id"],
            }
        )
    io.write_points_table(SLUG, "classification", points)

    counts = Counter(r["label"] for r in selected)
    io.write_dataset_metadata(
        SLUG,
        {
            "dataset": SLUG,
            "name": NAME,
            "task_type": "classification",
            "source": "olmoearth",
            "license": "internal",
            "provenance": {
                "url": SOURCE,
                "have_locally": True,
                "annotation_method": "manual field survey (Copernicus4GEOGLAM ground points)",
            },
            "sensors_relevant": ["sentinel2", "sentinel1", "landsat"],
            "classes": [
                {"id": cid, "name": name, "description": desc}
                for cid, name, desc in CLASSES
            ],
            "nodata_value": io.CLASS_NODATA,
            "num_samples": len(selected),
            "class_counts": {name: counts.get(cid, 0) for cid, name, _ in CLASSES},
            "notes": (
                "Single-pixel (sparse-point) crop-system labels -> points.geojson (spec 2a); "
                "the source label_raster labels only the surveyed center pixel per 64x64 "
                "window, so a per-tile GeoTIFF would fabricate labels for unobserved pixels. "
                "Class ids follow the source label_raster encoding (0=intercrop, 1=monocrop, "
                "2=other). All train/val/test splits used. Each point keeps its source "
                "growing-season window [2022-10-01, 2023-03-31) verbatim (~6 months, post-2016). "
                "Balanced to <=1000/class (all three classes truncated from "
                f"{Counter(r['label'] for r in recs)} available)."
            ),
        },
    )
    print("class counts (selected):", {id_to_name[k]: v for k, v in counts.items()})
    manifest.write_registry_entry(
        SLUG, "completed", task_type="classification", num_samples=len(selected)
    )
    print("done")


if __name__ == "__main__":
    multiprocessing.set_start_method("forkserver", force=True)
    main()
