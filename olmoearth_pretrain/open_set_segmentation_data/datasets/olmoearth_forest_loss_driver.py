"""Process the OlmoEarth forest-loss-driver eval into open-set-segmentation tiles.

Source: local rslearn dataset at
``/weka/dfive-default/rslearn-eai/datasets/forest_loss_driver/dataset_v1/combined``.
Each window is one manually-annotated GLAD forest-loss event in the Amazon basin (Peru /
Brazil / Colombia). The ``label`` vector layer (``layers/label/data.geojson``) carries a
WGS84 **polygon footprint** of the loss patch plus a ``new_label`` driver class
(agriculture / mining / airstrip / road / logging / burned / landslide / hurricane /
river / none). Each event has an approximate date (``info.json`` ``pixel_date``/``date``,
else the window metadata time_range midpoint).

Because the label has a real (multi-pixel) footprint and is a dated **change** event, we
emit small rasterized-polygon GeoTIFF tiles (driver class inside the footprint, 255 =
nodata/ignore elsewhere), sized to the footprint + context and capped at 64x64. Each
sample sets ``change_time`` = event date and ``time_range`` = a 1-year window centered on
it (spec section 5). Balanced to <=1000 samples per class.
"""

import argparse
import json
import math
import multiprocessing
import os
from datetime import UTC, datetime, timedelta
from typing import Any

import shapely
from rslearn.const import WGS84_PROJECTION

from olmoearth_pretrain.open_set_segmentation_data import io, manifest
from olmoearth_pretrain.open_set_segmentation_data.rasterize import (
    geom_to_pixels,
    rasterize_shapes,
)
from olmoearth_pretrain.open_set_segmentation_data.sampling import balance_by_class

SLUG = "olmoearth_forest_loss_driver"
SOURCE = (
    "/weka/dfive-default/rslearn-eai/datasets/forest_loss_driver/dataset_v1/combined"
)
PER_CLASS = 1000
MARGIN_PX = 10  # context ring around the footprint
MIN_TILE = 32
MAX_TILE = io.MAX_TILE  # 64

# Driver taxonomy (manifest order) -> class id, with short definitions.
CLASSES = [
    (
        "agriculture",
        "Forest cleared for cultivated cropland / managed agriculture (incl. rice, smallholder, Mennonite colonies).",
    ),
    (
        "mining",
        "Forest loss from mineral / gold mining operations (pits, tailings, dredging).",
    ),
    ("airstrip", "Forest cleared to build a landing strip / airstrip."),
    ("road", "Forest loss along a newly cut road or track."),
    ("logging", "Selective or clear-cut timber logging."),
    ("burned", "Forest loss from fire / burn scar."),
    ("landslide", "Natural forest loss from a landslide / mass wasting."),
    ("hurricane", "Wind / storm (hurricane) blowdown of forest."),
    ("river", "Forest loss from river channel migration / flooding / erosion."),
    ("none", "No identifiable anthropogenic driver / natural or non-driver loss."),
]
NAME_TO_ID = {name: i for i, (name, _d) in enumerate(CLASSES)}

# Map raw source ``new_label`` values to the canonical taxonomy above. Ambiguous /
# free-text values (unlabeled, unknown, "Anthropic - Unknown", "General deforestation
# (Clearing)", "natural", "Natural - Unknown") are intentionally dropped so the class
# scheme stays 1:1 with the manifest.
LABEL_MAP = {name: name for name, _ in CLASSES}


def _event_time(wd: str, md: dict[str, Any]) -> datetime | None:
    """Best-estimate event datetime for a window (see module docstring)."""
    ip = os.path.join(wd, "info.json")
    if os.path.exists(ip):
        try:
            info = json.load(open(ip))
        except Exception:
            info = {}
        for k in ("pixel_date", "date"):
            if info.get(k):
                try:
                    return datetime.fromisoformat(info[k])
                except ValueError:
                    pass
    tr = md.get("time_range")
    if tr:
        try:
            a = datetime.fromisoformat(tr[0])
            b = datetime.fromisoformat(tr[1])
            return a + (b - a) / 2
        except (ValueError, TypeError):
            return None
    return None


def _read_one(wd: str) -> dict[str, Any] | None:
    """Read one window -> flat record (or None to skip)."""
    try:
        with open(os.path.join(wd, "layers", "label", "data.geojson")) as f:
            gj = json.load(f)
    except FileNotFoundError:
        return None
    feats = gj.get("features") or []
    if not feats:
        return None
    raw = feats[0]["properties"].get("new_label")
    name = LABEL_MAP.get(raw)
    if name is None:
        return None
    try:
        with open(os.path.join(wd, "metadata.json")) as f:
            md = json.load(f)
    except FileNotFoundError:
        return None
    geom = shapely.geometry.shape(feats[0]["geometry"])
    if geom.is_empty:
        return None
    cx, cy = geom.centroid.x, geom.centroid.y
    if not (-180 <= cx <= 180 and -90 <= cy <= 90):
        return None  # geojson not in lon/lat as expected
    et = _event_time(wd, md)
    if et is None:
        return None
    if et.tzinfo is None:
        et = et.replace(tzinfo=UTC)
    return {
        "label": name,
        "lon": cx,
        "lat": cy,
        "wkt": geom.wkt,
        "change_time": et.isoformat(),
        "source_id": f"{os.path.basename(os.path.dirname(wd))}/{os.path.basename(wd)}",
    }


def scan_records() -> list[dict[str, Any]]:
    jobs = []
    root = os.path.join(SOURCE, "windows")
    for g in sorted(os.listdir(root)):
        gd = os.path.join(root, g)
        if os.path.isdir(gd):
            for w in os.listdir(gd):
                jobs.append(os.path.join(gd, w))
    with multiprocessing.Pool(64) as p:
        recs = [r for r in p.map(_read_one, jobs, chunksize=32) if r]
    return recs


def _write_one(args: tuple[int, dict[str, Any]]) -> tuple[int, int] | None:
    """Rasterize one event's polygon into a tile and write tif + json."""
    idx, rec = args
    sample_id = f"{idx:06d}"
    out_tif = io.locations_dir(SLUG) / f"{sample_id}.tif"
    if out_tif.exists():
        return (idx, NAME_TO_ID[rec["label"]])

    cid = NAME_TO_ID[rec["label"]]
    poly = shapely.from_wkt(rec["wkt"])
    proj = io.utm_projection_for_lonlat(rec["lon"], rec["lat"])
    _, col, row = io.lonlat_to_utm_pixel(rec["lon"], rec["lat"], proj)

    poly_px = geom_to_pixels(poly, WGS84_PROJECTION, proj)
    minx, miny, maxx, maxy = poly_px.bounds
    fw = maxx - minx
    fh = maxy - miny
    size = int(math.ceil(max(fw, fh))) + 2 * MARGIN_PX
    size = max(MIN_TILE, min(MAX_TILE, size))

    bounds = io.centered_bounds(col, row, size, size)
    arr = rasterize_shapes(
        [(poly_px, cid)],
        bounds,
        fill=io.CLASS_NODATA,
        dtype="uint8",
        all_touched=True,
    )
    if not (arr == cid).any():
        # Footprint fell outside the tile (huge/offset polygon); force center pixel.
        h, w = arr.shape[1], arr.shape[2]
        arr[0, h // 2, w // 2] = cid

    ct = datetime.fromisoformat(rec["change_time"])
    half = timedelta(days=180)
    time_range = (ct - half, ct + half)

    io.write_label_geotiff(SLUG, sample_id, arr, proj, bounds, nodata=io.CLASS_NODATA)
    io.write_sample_json(
        SLUG,
        sample_id,
        proj,
        bounds,
        time_range,
        change_time=ct,
        source_id=rec["source_id"],
        classes_present=[cid],
    )
    return (idx, cid)


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
    print(f"scanned {len(recs)} labeled forest-loss events")
    selected = balance_by_class(recs, "label", per_class=PER_CLASS)
    print(f"selected {len(selected)} (<= {PER_CLASS}/class)")

    io.locations_dir(SLUG).mkdir(parents=True, exist_ok=True)
    jobs = list(enumerate(selected))
    with multiprocessing.Pool(args.workers) as p:
        results = [r for r in p.map(_write_one, jobs, chunksize=16) if r]

    from collections import Counter

    counts = Counter(cid for _, cid in results)
    id_to_name = {i: name for i, (name, _d) in enumerate(CLASSES)}
    class_counts = {name: counts.get(i, 0) for i, (name, _d) in enumerate(CLASSES)}
    print("class counts:", class_counts)

    io.write_dataset_metadata(
        SLUG,
        {
            "dataset": SLUG,
            "name": "OlmoEarth forest loss driver",
            "task_type": "classification",
            "source": "olmoearth",
            "license": "internal",
            "provenance": {
                "url": SOURCE,
                "have_locally": True,
                "annotation_method": "manual annotation on GLAD alerts",
            },
            "sensors_relevant": ["sentinel2", "sentinel1", "landsat"],
            "classes": [
                {"id": i, "name": name, "description": desc}
                for i, (name, desc) in enumerate(CLASSES)
            ],
            "nodata_value": io.CLASS_NODATA,
            "num_samples": len(results),
            "class_counts": class_counts,
            "notes": (
                "Rasterized forest-loss polygon footprints (driver class inside, 255 "
                "nodata elsewhere), tiles sized to footprint + 10px context, capped 64x64. "
                "Change labels: change_time = event date (info.json pixel_date/date, else "
                "window time_range midpoint); time_range = 1-year window centered on it. "
                "All source splits/groups used. Dropped ambiguous source labels "
                "(unlabeled, unknown, 'Anthropic - Unknown', 'General deforestation "
                "(Clearing)', natural, 'Natural - Unknown')."
            ),
        },
    )
    manifest.write_registry_entry(
        SLUG, "completed", task_type="classification", num_samples=len(results)
    )
    print("done", len(results))


if __name__ == "__main__":
    multiprocessing.set_start_method("forkserver", force=True)
    main()
