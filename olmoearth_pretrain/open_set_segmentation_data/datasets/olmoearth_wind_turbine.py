"""Process OlmoEarth wind turbine into open-set-segmentation detection tiles.

Source: local rslearn dataset (have_locally=true, not copied)
``/weka/dfive-default/rslearn-eai/datasets/wind_turbine/dataset_v1/20260122``.
This is the existing OlmoEarth / Satlas wind-turbine detection eval: manually annotated
windows, each a small crop already in a **local UTM projection at 10 m/pixel** (~360-420 px
square) with a Satlas seasonal-mosaic ``time_range`` (~90 days). Two window groups
(``label`` and ``naip``) are both used (splits are pretraining-agnostic, spec section 5).

Label layer ``label`` (vector, GeoJSON): one ``Point`` feature per annotated turbine,
``properties.category == "turbine"``. The two groups store coordinates DIFFERENTLY: the
``label`` group uses the window's projection (pixel) coordinates (FeatureCollection
``properties.crs`` = window UTM CRS, x_resolution=10, matching the window ``bounds``), while
the ``naip`` group uses WGS84 lon/lat (top-level GeoJSON ``crs`` = EPSG:4326). We detect the
coordinate system per file and reproject lon/lat into window-projection pixel coords so both
groups are handled uniformly. A window is a POSITIVE window if its ``label`` layer has >=1
turbine point, otherwise a turbine-free NEGATIVE window.

Encoding (label_type bboxes -> detection, spec section 4). The manifest calls these
"bboxes" but the on-disk annotations are **points** (turbine centroids). We use the tunable
detection encoding:
  * one 64x64 (DET_TILE) context tile per turbine, centered on the turbine pixel but
    CLAMPED to lie fully inside the source window, written in the window's own UTM
    projection so georeferencing is exact;
  * the turbine is a 1x1 positive (class 1 = turbine), ringed by a 10 px nodata (255)
    buffer (turbine centroids are not pixel-exact), all other pixels background
    (class 0 = non-turbine ground/sea);
  * every other annotated turbine of the same window that falls inside the tile is also
    marked positive. Clamping the tile inside the window guarantees we know every turbine
    in the tile (turbines are only annotated within the window), so background pixels are
    true negatives (no unlabeled turbines leak in from outside the window).
We additionally emit background-only NEGATIVE tiles from turbine-free windows so the
background class has spatially-meaningful negatives (spec section 5, detection exception).

Classes: 0 background, 1 turbine.

Sampling: single turbine class, so PER_CLASS=1000 positive turbine tiles + N_NEGATIVES=1000
background tiles (well under the 25k cap), matching the vessel-detection precedent.

Time range: each sample uses its window's own Satlas seasonal-mosaic ``time_range``
(~90 days, <= 1 year, spec section 5). Turbines are persistent structures, so a seasonal
window is a valid observation period. All source labels are 2017-2022 (post-2016).

Run (idempotent; skips already-written tiles):
  python3 -m olmoearth_pretrain.open_set_segmentation_data.datasets.olmoearth_wind_turbine
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
import shapely
import tqdm
from rasterio.crs import CRS
from rslearn.const import WGS84_PROJECTION
from rslearn.utils.geometry import Projection, STGeometry
from rslearn.utils.mp import star_imap_unordered

from olmoearth_pretrain.open_set_segmentation_data import io, manifest
from olmoearth_pretrain.open_set_segmentation_data.sampling import encode_detection_tile

SLUG = "olmoearth_wind_turbine"
NAME = "OlmoEarth wind turbine"
SOURCE = "/weka/dfive-default/rslearn-eai/datasets/wind_turbine/dataset_v1/20260122"
WINDOWS_ROOT = os.path.join(SOURCE, "windows")

BACKGROUND_ID = 0
TURBINE_ID = 1
CLASS_NAMES = {BACKGROUND_ID: "background", TURBINE_ID: "turbine"}

# Detection encoding parameters (spec section 4). 64 px tile gives ample background
# context and captures neighboring turbines of dense wind farms in one tile.
DET_TILE = 64
DET_POS_SIZE = 1
DET_BUFFER = 10

PER_CLASS = 1000  # positive turbine tiles (single class, spec section 5)
N_NEGATIVES = 1000  # background-only tiles from turbine-free windows
SEED = 42


def _list_windows() -> list[tuple[str, str]]:
    """(group, name) for every window in the source dataset."""
    out: list[tuple[str, str]] = []
    for g in sorted(os.listdir(WINDOWS_ROOT)):
        gd = os.path.join(WINDOWS_ROOT, g)
        if not os.path.isdir(gd):
            continue
        for name in os.listdir(gd):
            out.append((g, name))
    return out


def _geojson_is_wgs84(lab: dict[str, Any]) -> bool:
    """True if the GeoJSON coordinates are WGS84 lon/lat rather than window-projection
    pixel coords. The naip group declares a top-level ``crs`` (EPSG:4326 / WGS 84); the
    label group instead carries the window UTM CRS under ``properties.crs``. Fall back to a
    coordinate-magnitude test (lon/lat fit in [-180,180]/[-90,90]; UTM pixel coords do not).
    """
    crs_member = lab.get("crs")
    if isinstance(crs_member, dict):
        blob = json.dumps(crs_member).lower()
        if "4326" in blob or "wgs 84" in blob or "wgs_1984" in blob:
            return True
    for f in lab.get("features", []):
        geom = f.get("geometry") or {}
        if geom.get("type") == "Point":
            x, y = geom["coordinates"][:2]
            return abs(float(x)) <= 180.0 and abs(float(y)) <= 90.0
    return False


def _scan_window(group: str, name: str) -> dict[str, Any] | None:
    """Read one window's metadata + label layer. Lightweight pos/neg record."""
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
    try:
        lab = json.load(open(os.path.join(wdir, "layers", "label", "data.geojson")))
    except (FileNotFoundError, json.JSONDecodeError):
        return None
    # Detect coordinate system: naip group is WGS84 lon/lat (top-level GeoJSON "crs" =
    # EPSG:4326); label group is already in window-projection pixel coords.
    is_wgs84 = _geojson_is_wgs84(lab)
    win_proj = _projection(crs) if is_wgs84 else None
    turbines: list[tuple[float, float]] = []
    for f in lab.get("features", []):
        geom = f.get("geometry") or {}
        if geom.get("type") != "Point":
            continue
        if f.get("properties", {}).get("category") not in (None, "turbine"):
            continue
        x, y = geom["coordinates"][:2]
        x, y = float(x), float(y)
        if is_wgs84:
            g = STGeometry(WGS84_PROJECTION, shapely.Point(x, y), None).to_projection(
                win_proj
            )
            x, y = float(g.shp.x), float(g.shp.y)
        turbines.append((x, y))
    rec: dict[str, Any] = {
        "crs": crs,
        "wbounds": bounds,
        "tr": tr,
        "src": f"{group}/{name}",
    }
    if turbines:
        rec["kind"] = "pos"
        rec["turbines"] = turbines
    else:
        rec["kind"] = "neg"
    return rec


def _projection(crs: str) -> Projection:
    return Projection(CRS.from_string(crs), io.RESOLUTION, -io.RESOLUTION)


def _time_range(tr: list[str]) -> tuple[datetime, datetime]:
    return (datetime.fromisoformat(tr[0]), datetime.fromisoformat(tr[1]))


def _clamped_bounds(px: int, py: int, wbounds: list[int]) -> tuple[int, int, int, int]:
    """A DET_TILE x DET_TILE tile centered on (px, py) but shifted to lie fully inside the
    source window bounds when possible (so every turbine in the tile is known).
    """
    x0, y0, x1, y1 = wbounds
    tx = px - DET_TILE // 2
    ty = py - DET_TILE // 2
    # Clamp start so tile stays within window; if window smaller than tile, pin to x0/y0.
    if x1 - x0 >= DET_TILE:
        tx = min(max(tx, x0), x1 - DET_TILE)
    else:
        tx = x0
    if y1 - y0 >= DET_TILE:
        ty = min(max(ty, y0), y1 - DET_TILE)
    else:
        ty = y0
    return (tx, ty, tx + DET_TILE, ty + DET_TILE)


def _write_positive(rec: dict[str, Any]) -> str:
    sample_id = rec["sample_id"]
    if (io.locations_dir(SLUG) / f"{sample_id}.tif").exists():
        return "skip"
    proj = _projection(rec["crs"])
    cx, cy = rec["center"]
    px, py = int(math.floor(cx)), int(math.floor(cy))
    bounds = _clamped_bounds(px, py, rec["wbounds"])
    x_min, y_min = bounds[0], bounds[1]
    positives: list[tuple[int, int, int]] = []
    for tx, ty in rec["turbines"]:
        lc = int(math.floor(tx)) - x_min
        lr = int(math.floor(ty)) - y_min
        if 0 <= lc < DET_TILE and 0 <= lr < DET_TILE:
            positives.append((lr, lc, TURBINE_ID))
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
    bounds = _clamped_bounds(cx, cy, rec["wbounds"])
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
            "OlmoEarth/Satlas wind-turbine detection eval. Each window is a crop in a local "
            "UTM projection @ 10 m (~360-420 px) with a Satlas seasonal-mosaic time_range "
            "(~90 days).\n"
            "layer 'label' (vector GeoJSON): Point per turbine, "
            "properties.category='turbine', coords in window projection (pixel) units.\n"
            "Groups 'label' and 'naip' both used; a window is positive iff its label layer "
            "has >=1 turbine point, else a turbine-free negative window.\n"
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
    n_turbines = sum(len(r["turbines"]) for r in pos_windows)
    print(
        f"positive windows: {len(pos_windows)} ({n_turbines} turbines); "
        f"turbine-free windows: {len(neg_windows)}",
        flush=True,
    )

    io.check_disk()

    # One positive tile per annotated turbine (each tile also marks nearby turbines of the
    # same window). Shuffle and take up to PER_CLASS.
    pos_records: list[dict[str, Any]] = []
    for r in pos_windows:
        for tx, ty in r["turbines"]:
            pos_records.append(
                {
                    "kind": "pos",
                    "crs": r["crs"],
                    "tr": r["tr"],
                    "src": r["src"],
                    "center": (tx, ty),
                    "turbines": r["turbines"],
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
            "license": "ODbL/internal",
            "provenance": {
                "url": SOURCE,
                "have_locally": True,
                "annotation_method": "manual annotation (Satlas/OlmoEarth wind-turbine eval)",
            },
            "sensors_relevant": ["sentinel2", "sentinel1", "landsat"],
            "classes": [
                {
                    "id": BACKGROUND_ID,
                    "name": "background",
                    "description": "Non-turbine ground / sea surface within the tile.",
                },
                {
                    "id": TURBINE_ID,
                    "name": "turbine",
                    "description": "Wind turbine detected in Sentinel-2/Sentinel-1 imagery "
                    "(manually annotated point/centroid; onshore and offshore).",
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
                "turbine_positive_tiles": len(selected_pos),
                "background_negative_tiles": len(selected_neg),
            },
            "available": {
                "positive_windows": len(pos_windows),
                "turbine_points": n_turbines,
                "turbine_free_windows": len(neg_windows),
            },
            "notes": (
                "Local OlmoEarth/Satlas wind-turbine detection rslearn dataset. Manifest "
                "label_type='bboxes' but on-disk annotations are turbine-centroid POINTS. "
                "Detection encoding: 64x64 UTM 10 m context tile per turbine, 1 px positive "
                "(id 1) + 10 px nodata (255) buffer ring, rest background (id 0); the tile "
                "is clamped inside the source window and all turbines of that window falling "
                "in the tile are marked (so background pixels are true negatives). Written "
                "in each window's own UTM projection (source already local UTM @ 10 m, no "
                "reprojection). Negatives: background-only tiles from turbine-free windows. "
                "Groups 'label' and 'naip' both used (splits pretraining-agnostic). Time "
                "range = each window's own Satlas seasonal-mosaic window (~90 days, spec "
                "section 5); turbines are persistent structures. All source labels 2017-2022 "
                "(post-2016). Single class -> up to 1000 positive tiles + 1000 negatives."
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
