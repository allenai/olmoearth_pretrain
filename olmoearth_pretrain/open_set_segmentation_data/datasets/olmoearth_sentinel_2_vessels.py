"""Process OlmoEarth Sentinel-2 vessels into open-set-segmentation detection tiles.

Source: local rslearn dataset (have_locally=true, not copied)
``/weka/dfive-default/rslearn-eai/datasets/sentinel2_vessels/dataset_v1/20250213``.
This is the existing OlmoEarth / Satlas Sentinel-2 vessel-detection eval: manually
annotated windows, each a specific-image crop already in a **local UTM projection at
10 m/pixel** (~780x780 px) with a ~10-minute Sentinel-2 acquisition ``time_range``.

Label layer ``label`` (vector, GeoJSON): one ``Point`` feature per annotated vessel,
``properties.category == "vessel"``. Coordinates are in the window's projection
(pixel) coordinates (x_resolution=10, matching the window ``bounds``), so no reprojection
is needed. ``metadata.options.has_objects`` flags windows that contain >=1 vessel.

Encoding (label_type bboxes -> detection, spec section 4). The manifest calls these
"bboxes" but the on-disk annotations are **points** (vessel centroids). We use the tunable
detection encoding:
  * one 32x32 (DET_TILE) context tile per vessel, centered on the vessel pixel, written in
    the window's own UTM projection so georeferencing is exact;
  * the vessel is a 1x1 positive (class 1 = vessel), ringed by a 10 px nodata (255) buffer
    (vessel centroids are not pixel-exact), all other pixels background (class 0 = ocean);
  * every other annotated vessel of the same window that falls inside the tile is also
    marked positive.
We additionally emit background-only NEGATIVE tiles from windows annotated as vessel-free
(``has_objects == false``, incl. the dedicated train-bg / valid-bg groups) so the
background class has spatially-meaningful negatives (spec section 5, detection exception).

Groups: all vessel groups are used (splits are pretraining-agnostic, spec section 5). The
``sargassum_train`` / ``sargassum_val`` groups are EXCLUDED: they belong to a different
(sargassum) task where vessels were not annotated, so they are unsafe as vessel negatives.

Time range: each sample uses its window's own ~10-minute acquisition ``time_range`` (well
under the ~1 hour specific-image budget, spec section 5). Single class, so PER_CLASS=1000
positive vessel tiles + N_NEGATIVES background tiles.

Run (idempotent; skips already-written tiles):
  python3 -m olmoearth_pretrain.open_set_segmentation_data.datasets.olmoearth_sentinel_2_vessels
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
from olmoearth_pretrain.open_set_segmentation_data.sampling import encode_detection_tile

SLUG = "olmoearth_sentinel_2_vessels"
NAME = "OlmoEarth Sentinel-2 vessels"
SOURCE = (
    "/weka/dfive-default/rslearn-eai/datasets/sentinel2_vessels/dataset_v1/20250213"
)
WINDOWS_ROOT = os.path.join(SOURCE, "windows")

# Groups excluded from vessel processing (different task; vessels not annotated there).
EXCLUDE_GROUP_PREFIXES = ("sargassum",)

BACKGROUND_ID = 0
VESSEL_ID = 1
CLASS_NAMES = {BACKGROUND_ID: "background", VESSEL_ID: "vessel"}

# Detection encoding parameters (spec section 4).
DET_TILE = 32
DET_POS_SIZE = 1
DET_BUFFER = 10

PER_CLASS = 1000  # positive vessel tiles (single class, spec section 5)
N_NEGATIVES = 1000  # background-only tiles from vessel-free windows
SEED = 42


def _list_windows() -> list[tuple[str, str]]:
    """(group, name) for every window, excluding non-vessel groups."""
    out: list[tuple[str, str]] = []
    for g in sorted(os.listdir(WINDOWS_ROOT)):
        if g.startswith(EXCLUDE_GROUP_PREFIXES):
            continue
        gd = os.path.join(WINDOWS_ROOT, g)
        if not os.path.isdir(gd):
            continue
        for name in os.listdir(gd):
            out.append((g, name))
    return out


def _scan_window(group: str, name: str) -> dict[str, Any] | None:
    """Read one window's metadata (+ label if it has vessels). Lightweight record."""
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
    crs = proj["crs"]
    bounds = md["bounds"]
    has_obj = bool(md.get("options", {}).get("has_objects"))
    rec: dict[str, Any] = {
        "crs": crs,
        "wbounds": bounds,
        "tr": tr,
        "src": f"{group}/{name}",
    }
    if not has_obj:
        rec["kind"] = "neg"
        return rec
    try:
        lab = json.load(open(os.path.join(wdir, "layers", "label", "data.geojson")))
    except (FileNotFoundError, json.JSONDecodeError):
        return None
    vessels: list[tuple[float, float]] = []
    for f in lab.get("features", []):
        geom = f.get("geometry") or {}
        if geom.get("type") != "Point":
            continue
        if f.get("properties", {}).get("category") not in (None, "vessel"):
            continue
        x, y = geom["coordinates"][:2]
        vessels.append((float(x), float(y)))
    if not vessels:
        rec["kind"] = "neg"  # flagged has_objects but no usable point -> treat as neg
        return rec
    rec["kind"] = "pos"
    rec["vessels"] = vessels
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
    for vx, vy in rec["vessels"]:
        lc = int(math.floor(vx)) - x_min
        lr = int(math.floor(vy)) - y_min
        if 0 <= lc < DET_TILE and 0 <= lr < DET_TILE:
            positives.append((lr, lc, VESSEL_ID))
    arr = encode_detection_tile(
        positives,
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
        classes_present=sorted(set(np.unique(arr).tolist()) - {io.CLASS_NODATA}),
    )
    return "pos"


def _write_negative(rec: dict[str, Any]) -> str:
    sample_id = rec["sample_id"]
    if (io.locations_dir(SLUG) / f"{sample_id}.tif").exists():
        return "skip"
    proj = _projection(rec["crs"])
    x0, y0, x1, y1 = rec["wbounds"]
    # Random tile center with a half-tile margin so the tile fits inside the window.
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
        classes_present=[BACKGROUND_ID],
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
    with (raw / "SOURCE.txt").open("w") as f:
        f.write(
            "local rslearn dataset (have_locally=true, not copied):\n"
            f"{SOURCE}\n"
            "OlmoEarth/Satlas Sentinel-2 vessel-detection eval. Each window is a specific "
            "S2 image crop in a local UTM projection @ 10 m with a ~10-min time_range.\n"
            "layer 'label' (vector GeoJSON): Point per vessel, properties.category='vessel', "
            "coords in window projection (pixel) units.\n"
            "metadata.options.has_objects flags vessel-bearing windows.\n"
            "sargassum_train/sargassum_val groups excluded (different task; unsafe negatives).\n"
        )

    windows = _list_windows()
    print(f"scanning {len(windows)} windows (excl. sargassum)", flush=True)
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
    n_vessels = sum(len(r["vessels"]) for r in pos_windows)
    print(
        f"positive windows: {len(pos_windows)} ({n_vessels} vessels); "
        f"vessel-free windows: {len(neg_windows)}",
        flush=True,
    )

    io.check_disk()

    # One positive tile per annotated vessel (each tile also marks nearby vessels of the
    # same window). Shuffle and take up to PER_CLASS.
    pos_records: list[dict[str, Any]] = []
    for r in pos_windows:
        for vx, vy in r["vessels"]:
            pos_records.append(
                {
                    "kind": "pos",
                    "crs": r["crs"],
                    "tr": r["tr"],
                    "src": r["src"],
                    "center": (vx, vy),
                    "vessels": r["vessels"],
                    "wbounds": r["wbounds"],
                }
            )
    rng = random.Random(SEED)
    rng.shuffle(pos_records)
    rng.shuffle(neg_windows)
    selected_pos = pos_records[:PER_CLASS]
    selected_neg = neg_windows[:N_NEGATIVES]
    all_recs = selected_pos + selected_neg
    for i, r in enumerate(all_recs):
        r["sample_id"] = f"{i:06d}"
    print(
        f"selected {len(selected_pos)} positive tiles + {len(selected_neg)} negatives "
        f"= {len(all_recs)} samples",
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
            "license": "internal",
            "provenance": {
                "url": SOURCE,
                "have_locally": True,
                "annotation_method": "manual annotation (Satlas/OlmoEarth vessel eval)",
            },
            "sensors_relevant": ["sentinel2", "sentinel1", "landsat"],
            "classes": [
                {
                    "id": BACKGROUND_ID,
                    "name": "background",
                    "description": "Open water / non-vessel ocean surface within the tile.",
                },
                {
                    "id": VESSEL_ID,
                    "name": "vessel",
                    "description": "Ship / vessel detected in Sentinel-2 imagery "
                    "(manually annotated point/centroid).",
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
                "vessel_positive_tiles": len(selected_pos),
                "background_negative_tiles": len(selected_neg),
            },
            "notes": (
                "Local OlmoEarth/Satlas Sentinel-2 vessel-detection rslearn dataset. "
                "Manifest label_type='bboxes' but on-disk annotations are vessel-centroid "
                "POINTS. Detection encoding: 32x32 UTM 10 m context tile per vessel, "
                "1 px positive (id 1) + 10 px nodata (255) buffer ring, rest background "
                "(id 0); all vessels of the source window falling in a tile are marked. "
                "Written in each window's own UTM projection (source already local UTM @ "
                "10 m, no reprojection). Negatives: background-only tiles from vessel-free "
                "windows (has_objects==false, incl. train-bg/valid-bg). sargassum_train/"
                "sargassum_val groups excluded (different task; vessels not annotated -> "
                "unsafe negatives). All vessel splits used (pretraining-agnostic). Time "
                "range = each window's own ~10-min S2 acquisition window (specific-image, "
                "spec section 5). Single class -> up to 1000 positive tiles + 1000 negatives."
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
