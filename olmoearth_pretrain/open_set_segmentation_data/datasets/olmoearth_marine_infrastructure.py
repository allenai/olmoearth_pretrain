"""Process OlmoEarth marine infrastructure into open-set-segmentation detection tiles.

Source: local rslearn dataset (have_locally=true, not copied)
``/weka/dfive-default/rslearn-eai/datasets/marine_infra/dataset_v1/20250605``.
This is the existing OlmoEarth / Satlas offshore marine-infrastructure detection eval:
manually annotated windows, each a specific-image crop already in a **local UTM
projection at 10 m/pixel** (~855x855 px) with a ~220-day monthly-composite ``time_range``
(all windows post-2016).

Label layer ``label`` (vector, GeoJSON): one ``Point`` feature per annotated object,
``properties.category`` in {platform, turbine, vessel, power, aerialway, (unknown)}.
Coordinates are in the window's projection (pixel) coordinates (x_resolution=10, matching
the window ``bounds``), so no reprojection is needed. ``metadata.options.has_objects``
flags windows that contain >=1 object.

Encoding (label_type bboxes -> detection, spec section 4). The manifest calls these
"bboxes" but the on-disk annotations are **points** (object centroids). We use the tunable
detection encoding with a UNIFIED two-class marine-infrastructure scheme (spec section 5,
multi-target -> one class map):
  * classes: background(0), platform(1), turbine(2) -- the manifest targets;
  * one 32x32 (DET_TILE) context tile per platform/turbine detection, centered on it,
    written in the window's own UTM projection so georeferencing is exact;
  * the detection is a 1x1 positive of its class id, ringed by a 10 px nodata (255) buffer
    (centroids are not pixel-exact), all other pixels background (class 0 = ocean);
  * every other platform/turbine of the same window falling inside the tile is also marked
    positive;
  * non-target annotated objects (vessel/power/aerialway/unknown) falling inside a tile are
    marked as **nodata (255)** ignore (with the same buffer), so they are neither called a
    wrong class nor a false background.
We additionally emit background-only NEGATIVE tiles from object-free windows
(``has_objects == false``) so the background class has spatially-meaningful negatives
(spec section 5, detection exception).

Groups: the dataset has a single ``label`` group; all windows/splits are used (splits are
pretraining-agnostic, spec section 5).

Time range: each sample uses its window's own ~220-day monthly-composite ``time_range``
(< 1 year; marine infrastructure is static across that window, spec section 5). Two target
classes, so PER_CLASS=1000 positive tiles per class + N_NEGATIVES background tiles.

Run (idempotent; skips already-written tiles):
  python3 -m olmoearth_pretrain.open_set_segmentation_data.datasets.olmoearth_marine_infrastructure
"""

import argparse
import json
import math
import multiprocessing
import os
import random
from collections import Counter
from datetime import datetime
from typing import Any

import numpy as np
import tqdm
from rasterio.crs import CRS
from rslearn.utils.geometry import Projection
from rslearn.utils.mp import star_imap_unordered

from olmoearth_pretrain.open_set_segmentation_data import io, manifest
from olmoearth_pretrain.open_set_segmentation_data.sampling import (
    balance_by_class,
    encode_detection_tile,
)

SLUG = "olmoearth_marine_infrastructure"
NAME = "OlmoEarth marine infrastructure"
SOURCE = "/weka/dfive-default/rslearn-eai/datasets/marine_infra/dataset_v1/20250605"
WINDOWS_ROOT = os.path.join(SOURCE, "windows")

BACKGROUND_ID = 0
PLATFORM_ID = 1
TURBINE_ID = 2
# Manifest targets. Everything else annotated (vessel/power/aerialway/unknown) is ignored.
TARGET_CATEGORIES = {"platform": PLATFORM_ID, "turbine": TURBINE_ID}
CLASS_NAMES = {
    BACKGROUND_ID: "background",
    PLATFORM_ID: "platform",
    TURBINE_ID: "turbine",
}

# Detection encoding parameters (spec section 4).
DET_TILE = 32
DET_POS_SIZE = 1
DET_BUFFER = 10

PER_CLASS = 1000  # positive tiles per target class (spec section 5)
N_NEGATIVES = 1000  # background-only tiles from object-free windows
SEED = 42


def _list_windows() -> list[tuple[str, str]]:
    """(group, name) for every window."""
    out: list[tuple[str, str]] = []
    for g in sorted(os.listdir(WINDOWS_ROOT)):
        gd = os.path.join(WINDOWS_ROOT, g)
        if not os.path.isdir(gd):
            continue
        for name in os.listdir(gd):
            out.append((g, name))
    return out


def _scan_window(group: str, name: str) -> dict[str, Any] | None:
    """Read one window's metadata (+ label objects if any). Lightweight record."""
    wdir = os.path.join(WINDOWS_ROOT, group, name)
    try:
        md = json.load(open(os.path.join(wdir, "metadata.json")))
    except (FileNotFoundError, json.JSONDecodeError):
        return None
    tr = md.get("time_range")
    if not tr or tr[0] is None:
        return None  # no acquisition time -> cannot assign a time range
    proj = md["projection"]
    if abs(proj.get("x_resolution", 0)) != io.RESOLUTION:
        return None
    rec: dict[str, Any] = {
        "crs": proj["crs"],
        "wbounds": md["bounds"],
        "tr": tr,
        "src": f"{group}/{name}",
    }
    has_obj = bool(md.get("options", {}).get("has_objects"))
    if not has_obj:
        rec["kind"] = "neg"
        return rec
    try:
        lab = json.load(open(os.path.join(wdir, "layers", "label", "data.geojson")))
    except (FileNotFoundError, json.JSONDecodeError):
        return None
    # (x, y, class_id) for target objects; (x, y) for ignore objects.
    targets: list[tuple[float, float, int]] = []
    ignores: list[tuple[float, float]] = []
    for f in lab.get("features", []):
        geom = f.get("geometry") or {}
        if geom.get("type") != "Point":
            continue
        cat = f.get("properties", {}).get("category")
        x, y = geom["coordinates"][:2]
        if cat in TARGET_CATEGORIES:
            targets.append((float(x), float(y), TARGET_CATEGORIES[cat]))
        else:
            ignores.append((float(x), float(y)))
    if not targets:
        # No platform/turbine (only vessels/etc.) -> not a usable positive, and unsafe as a
        # negative (it does contain annotated objects), so drop it.
        return None
    rec["kind"] = "pos"
    rec["targets"] = targets
    rec["ignores"] = ignores
    return rec


def _projection(crs: str) -> Projection:
    return Projection(CRS.from_string(crs), io.RESOLUTION, -io.RESOLUTION)


def _time_range(tr: list[str]) -> tuple[datetime, datetime]:
    return (datetime.fromisoformat(tr[0]), datetime.fromisoformat(tr[1]))


def _write_positive(rec: dict[str, Any]) -> str:
    sample_id = rec["sample_id"]
    if (io.locations_dir(SLUG) / f"{sample_id}.tif").exists():
        return "skip"
    proj = _projection(rec["crs"])
    cx, cy = rec["center"]
    px, py = int(math.floor(cx)), int(math.floor(cy))
    bounds = io.centered_bounds(px, py, DET_TILE, DET_TILE)
    x_min, y_min = bounds[0], bounds[1]
    positives: list[tuple[int, int, int]] = []
    # Target objects (platform/turbine) of this window that fall inside the tile.
    for tx, ty, cid in rec["targets"]:
        lc = int(math.floor(tx)) - x_min
        lr = int(math.floor(ty)) - y_min
        if 0 <= lc < DET_TILE and 0 <= lr < DET_TILE:
            positives.append((lr, lc, cid))
    # Non-target objects -> nodata ignore (encode as a "positive" of class 255 so the
    # shared encoder paints its center + buffer as nodata; real positives are painted last
    # among same-class ties are irrelevant since these are 255).
    for ix, iy in rec["ignores"]:
        lc = int(math.floor(ix)) - x_min
        lr = int(math.floor(iy)) - y_min
        if 0 <= lc < DET_TILE and 0 <= lr < DET_TILE:
            positives.append((lr, lc, io.CLASS_NODATA))
    # Order so real target positives are painted AFTER ignore centers (later wins ties).
    positives.sort(key=lambda p: p[2] == io.CLASS_NODATA, reverse=True)
    arr = encode_detection_tile(
        positives,
        tile_size=DET_TILE,
        positive_size=DET_POS_SIZE,
        buffer_size=DET_BUFFER,
        nodata=io.CLASS_NODATA,
        background=BACKGROUND_ID,
    )
    io.write_label_geotiff(SLUG, sample_id, arr, proj, bounds, nodata=io.CLASS_NODATA)
    classes_present = sorted(
        set(np.unique(arr).tolist()) - {io.CLASS_NODATA, BACKGROUND_ID}
    )
    io.write_sample_json(
        SLUG,
        sample_id,
        proj,
        bounds,
        _time_range(rec["tr"]),
        source_id=rec["src"],
        classes_present=classes_present,
    )
    return "pos"


def _write_negative(rec: dict[str, Any]) -> str:
    sample_id = rec["sample_id"]
    if (io.locations_dir(SLUG) / f"{sample_id}.tif").exists():
        return "skip"
    proj = _projection(rec["crs"])
    x0, y0, x1, y1 = rec["wbounds"]
    half = DET_TILE // 2
    rng = random.Random(hash(rec["src"]) & 0xFFFFFFFF)
    cx = rng.randint(x0 + half, max(x0 + half, x1 - half))
    cy = rng.randint(y0 + half, max(y0 + half, y1 - half))
    bounds = io.centered_bounds(cx, cy, DET_TILE, DET_TILE)
    arr = encode_detection_tile(
        [],
        tile_size=DET_TILE,
        positive_size=DET_POS_SIZE,
        buffer_size=DET_BUFFER,
        nodata=io.CLASS_NODATA,
        background=BACKGROUND_ID,
    )
    io.write_label_geotiff(SLUG, sample_id, arr, proj, bounds, nodata=io.CLASS_NODATA)
    io.write_sample_json(
        SLUG,
        sample_id,
        proj,
        bounds,
        _time_range(rec["tr"]),
        source_id=rec["src"],
        classes_present=[],
    )
    return "neg"


def _dispatch(rec: dict[str, Any]) -> str:
    return _write_positive(rec) if rec["kind"] == "pos" else _write_negative(rec)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=64)
    args = parser.parse_args()

    io.check_disk()
    manifest.write_registry_entry(SLUG, "in_progress")

    raw = io.raw_dir(SLUG)
    raw.mkdir(parents=True, exist_ok=True)
    if not (raw / "SOURCE.txt").exists():
        with (raw / "SOURCE.txt").open("w") as f:
            f.write(
                "local rslearn dataset (have_locally=true, not copied):\n"
                f"{SOURCE}\n"
                "OlmoEarth/Satlas offshore marine-infrastructure detection eval. Windows in "
                "local UTM @ 10 m with ~220-day time_range; layer 'label' (vector GeoJSON): "
                "Point per object, properties.category in {platform,turbine,vessel,power,"
                "aerialway}. Targets = platform/turbine; others -> nodata ignore. "
                "has_objects flags object-bearing windows.\n"
            )

    windows = _list_windows()
    print(f"scanning {len(windows)} windows", flush=True)
    records: list[dict[str, Any]] = []
    with multiprocessing.Pool(args.workers) as p:
        for rec in tqdm.tqdm(
            star_imap_unordered(
                p, _scan_window, [dict(group=g, name=n) for g, n in windows]
            ),
            total=len(windows),
        ):
            if rec is not None:
                records.append(rec)

    pos_windows = [r for r in records if r["kind"] == "pos"]
    neg_windows = [r for r in records if r["kind"] == "neg"]
    n_platform = sum(
        sum(1 for _, _, c in r["targets"] if c == PLATFORM_ID) for r in pos_windows
    )
    n_turbine = sum(
        sum(1 for _, _, c in r["targets"] if c == TURBINE_ID) for r in pos_windows
    )
    print(
        f"positive windows: {len(pos_windows)} "
        f"({n_platform} platforms, {n_turbine} turbines); "
        f"object-free windows: {len(neg_windows)}",
        flush=True,
    )

    io.check_disk()

    # One positive tile per target detection, tagged with its center class for balancing.
    pos_records: list[dict[str, Any]] = []
    for r in pos_windows:
        for tx, ty, cid in r["targets"]:
            pos_records.append(
                {
                    "kind": "pos",
                    "crs": r["crs"],
                    "tr": r["tr"],
                    "src": r["src"],
                    "center": (tx, ty),
                    "center_class": cid,
                    "targets": r["targets"],
                    "ignores": r["ignores"],
                    "wbounds": r["wbounds"],
                }
            )
    # Balance to <= PER_CLASS tiles per target class (by the tile's center class).
    selected_pos = balance_by_class(
        pos_records,
        key="center_class",
        per_class=PER_CLASS,
        seed=SEED,
    )
    rng = random.Random(SEED)
    rng.shuffle(neg_windows)
    selected_neg = neg_windows[:N_NEGATIVES]
    all_recs = selected_pos + selected_neg
    for i, r in enumerate(all_recs):
        r["sample_id"] = f"{i:06d}"
    sel_by_class = Counter(r["center_class"] for r in selected_pos)
    print(
        f"selected {len(selected_pos)} positive tiles "
        f"(platform={sel_by_class[PLATFORM_ID]}, turbine={sel_by_class[TURBINE_ID]}) "
        f"+ {len(selected_neg)} negatives = {len(all_recs)} samples",
        flush=True,
    )

    results: Counter = Counter()
    with multiprocessing.Pool(args.workers) as p:
        for res in tqdm.tqdm(
            star_imap_unordered(p, _dispatch, [dict(rec=r) for r in all_recs]),
            total=len(all_recs),
        ):
            results[res] += 1
    print("write results:", dict(results), flush=True)

    io.check_disk()

    io.write_dataset_metadata(
        SLUG,
        {
            "dataset": SLUG,
            "name": NAME,
            "task_type": "classification",  # detection encoded as per-pixel classes
            "source": "olmoearth",
            "license": "ODbL/internal",
            "provenance": {
                "url": SOURCE,
                "have_locally": True,
                "annotation_method": "manual annotation (Satlas)",
            },
            "sensors_relevant": ["sentinel2", "sentinel1", "landsat"],
            "classes": [
                {
                    "id": BACKGROUND_ID,
                    "name": "background",
                    "description": "Open water / non-infrastructure ocean surface within the tile.",
                },
                {
                    "id": PLATFORM_ID,
                    "name": "platform",
                    "description": "Offshore platform (e.g. oil/gas platform, offshore "
                    "substation) manually annotated as a point in Sentinel-2 imagery.",
                },
                {
                    "id": TURBINE_ID,
                    "name": "turbine",
                    "description": "Offshore wind turbine manually annotated as a point in "
                    "Sentinel-2 imagery.",
                },
            ],
            "nodata_value": io.CLASS_NODATA,
            "detection_encoding": {
                "tile_size": DET_TILE,
                "positive_size": DET_POS_SIZE,
                "buffer_size": DET_BUFFER,
            },
            "num_samples": len(all_recs),
            "class_tile_counts": {
                "platform_positive_tiles": sel_by_class[PLATFORM_ID],
                "turbine_positive_tiles": sel_by_class[TURBINE_ID],
                "background_negative_tiles": len(selected_neg),
            },
            "notes": (
                "Local OlmoEarth/Satlas offshore marine-infrastructure rslearn dataset. "
                "Manifest label_type='bboxes' but on-disk annotations are object-centroid "
                "POINTS. Unified two-target class scheme: background(0), platform(1), "
                "turbine(2). Detection encoding: 32x32 UTM 10 m context tile per "
                "platform/turbine, 1 px positive of its class + 10 px nodata (255) buffer "
                "ring, rest background (0); all targets of the source window falling in a "
                "tile are marked. Non-target categories (vessel/power/aerialway/unknown) "
                "falling in a tile are marked nodata (255) ignore. Written in each window's "
                "own UTM projection (source already local UTM @ 10 m, no reprojection). "
                "Negatives: background-only tiles from object-free windows "
                "(has_objects==false). All splits used (pretraining-agnostic). Time range "
                "= each window's own ~220-day monthly-composite window (< 1 year, marine "
                "infra static across it, spec section 5). Two classes -> up to 1000 "
                "positive tiles per class + 1000 negatives; all labels post-2016."
            ),
        },
    )
    manifest.write_registry_entry(
        SLUG, "completed", task_type="classification", num_samples=len(all_recs)
    )
    print(f"done: {len(all_recs)} samples", flush=True)


if __name__ == "__main__":
    multiprocessing.set_start_method("forkserver", force=True)
    main()
