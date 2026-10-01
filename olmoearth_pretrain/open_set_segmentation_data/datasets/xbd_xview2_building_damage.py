"""xBD / xView2 (building damage) -> open-set-segmentation change/damage masks.

Source: xView2 / xBD Building Damage Assessment Dataset (Gupta et al. 2019), built on
Maxar Open Data VHR imagery. ~850k manually annotated (expert-reviewed) building polygons
with pre/post-disaster image pairs across 6 natural-disaster types worldwide and a 4-level
damage scale (no-damage / minor-damage / major-damage / destroyed). Official distribution
(https://xview2.org/dataset) is behind a free-signup portal; we use a public Hugging Face
mirror that bundles the labels alongside imagery. We stream/download only the tier1
``train`` + ``test`` archives and extract **only the post-disaster label JSONs** (a few tens
of MB) -- the VHR image pixels are NOT used, since pretraining supplies its own S2/S1/Landsat
imagery. Each xBD label JSON carries, per building, a WGS84 (lng/lat) WKT polygon and a
``subtype`` damage class, plus per-image ``metadata.capture_date`` (day-resolved) and the
disaster name/type. So georeferencing comes directly from the label file (no image headers
needed).

CHANGE dataset (spec S5). This is a disaster before->after damage dataset. xBD gives, per
post-disaster image, a satellite ``capture_date`` resolvable to the day, tasked within days
of the event (a post-event date). We set ``change_time`` to that per-image post-disaster
capture date and emit two independent six-month windows via
``io.pre_post_time_ranges(change_time, pre_offset_days=45)``: ``post_time_range`` starts at
``change_time`` and runs ~6 months (<=183 days) forward, and ``pre_time_range`` ends 45 days
before ``change_time`` (a guard offset, since the rapid post-disaster imagery follows the
event by weeks) and spans ~6 months (<=183 days) backward from there, placing the pre window
before the event. ``time_range`` is null; pretraining pairs a "before" stack with an "after"
stack and probes on their difference. The damage (major/destroyed) mask is the "where the
change occurred" signal. This easily meets the S5 timing-precision requirement (event known
to << 1-2 months).

Damage-class scheme at 10 m (observability, spec S4). Buildings are ~0.4-1.4 m native GSD
(sub-pixel at 10 m), so a single building's *damage level* is not discriminable at 10 m from
Sentinel-scale imagery. Per spec S4/the manifest note we therefore COLLAPSE the 4-level scale
to a 2-class building-presence-x-damage scheme that is observable when damage clusters
(razed neighborhoods, tsunami-swept / burned zones):
    0 = intact_building    (xBD no-damage + minor-damage)
    1 = damaged_building   (xBD major-damage + destroyed)
The finer 4-level counts are retained in ``metadata.json``. ``un-classified`` buildings
(damage not assessable) are dropped (not painted). Non-building ground stays nodata (255):
this is a positive-only dataset (spec S5) -- assembly supplies negatives from other datasets;
we do NOT fabricate a background class. Per pixel the most-severe class wins (damaged painted
after intact via all_touched rasterization).

Post-2016 filter (spec S2): kept disasters are filtered to capture_date year >= 2016. The
tier1 (train+test) disasters are all 2016-2019 (hurricane-matthew is Oct 2016), so none are
dropped here. The ``tier3`` archive (18 GB) -- which additionally holds the pre-2016 tornado
events (joplin 2011, moore 2013, tuscaloosa 2011, pinery 2015) plus a few redundant post-2016
events -- is NOT downloaded: tier1 already covers all 6 disaster types post-2016 and yields
ample class-balanced tiles, so pulling 18 GB more is unwarranted (spec S8 download economy).

Run: python3 -m olmoearth_pretrain.open_set_segmentation_data.datasets.xbd_xview2_building_damage
"""

import argparse
import glob
import json
import math
import multiprocessing
import os
import tarfile
from collections import Counter
from datetime import UTC, datetime
from typing import Any

import numpy as np
import shapely
import shapely.wkt
from rslearn.const import WGS84_PROJECTION
from rslearn.utils.geometry import Projection
from rslearn.utils.mp import star_imap_unordered

from olmoearth_pretrain.open_set_segmentation_data import (
    download,
    io,
    manifest,
    rasterize,
    sampling,
)

SLUG = "xbd_xview2_building_damage"
NAME = "xBD / xView2 (building damage)"

# Public HF mirror (bundles labels + imagery). We extract only post-disaster label JSONs.
HF_BASE = (
    "https://huggingface.co/datasets/WayBob/"
    "Disaster_Recognition_RemoteSense_EN_CN_JA/resolve/main"
)
ARCHIVES = ["xview2_train.tar.gz", "xview2_test.tar.gz"]  # tier1; tier3 (18 GB) skipped
UA = {"User-Agent": "Mozilla/5.0 (olmoearth-open-set-seg)"}

TILE = 64
PER_CLASS = 1000
MIN_YEAR = 2016  # Sentinel era; drop any image whose post capture predates this.

# 2-class collapsed damage scheme (see module docstring).
CLASSES = [
    (
        "intact_building",
        "Building footprint assessed as no-damage or minor-damage in the post-disaster VHR "
        "image (structure intact; at most superficial damage). xBD subtypes no-damage + "
        "minor-damage. At 10 m an isolated building is sub-pixel (all_touched rasterization).",
    ),
    (
        "damaged_building",
        "Building footprint assessed as major-damage or destroyed (partial/complete structural "
        "collapse). xBD subtypes major-damage + destroyed. Damaged buildings cluster in disaster "
        "zones (razed neighborhoods, tsunami-swept / burned areas), aggregating into "
        "damage-extent patches observable at 10 m -- the change signal.",
    ),
]
NUM_CLASSES = len(CLASSES)

# xBD subtype -> our class id. un-classified (unassessable) -> None (skip).
SUBTYPE_MAP = {
    "no-damage": 0,
    "minor-damage": 0,
    "major-damage": 1,
    "destroyed": 1,
    "un-classified": None,
}
# Paint order: intact first, damaged last so the most-severe (change) class wins overlaps.
PAINT_RANK = {0: 0, 1: 1}


def _labels_dir() -> str:
    return str(io.raw_dir(SLUG) / "labels")


def _ensure_labels() -> list[str]:
    """Download tier1 archives (idempotent) and extract post-disaster label JSONs.

    Returns the list of extracted label-file paths. Only ``*/labels/*_post_disaster.json``
    members are written to disk; imagery/masks in the archive are skipped.
    """
    raw = io.raw_dir(SLUG)
    raw.mkdir(parents=True, exist_ok=True)
    out = _labels_dir()
    os.makedirs(out, exist_ok=True)
    for arc in ARCHIVES:
        zpath = raw / arc
        if not zpath.exists():
            print(f"downloading {arc} ...")
            download.download_http(f"{HF_BASE}/{arc}", zpath, headers=UA)
        # Skip extraction only if we've already pulled labels for this archive.
        marker = raw / (arc + ".labels_done")
        if marker.exists():
            continue
        n = 0
        with tarfile.open(str(zpath), "r:gz") as tf:
            for m in tf:
                if not (
                    m.isfile()
                    and "/labels/" in m.name
                    and m.name.endswith("_post_disaster.json")
                ):
                    continue
                dst = os.path.join(out, os.path.basename(m.name))
                if not os.path.exists(dst):
                    with tf.extractfile(m) as src, open(dst, "wb") as f:
                        f.write(src.read())
                n += 1
        marker.touch()
        print(f"{arc}: extracted {n} post-disaster label files")
    with (raw / "SOURCE.txt").open("w") as f:
        f.write(
            "xBD / xView2 Building Damage Assessment labels.\n"
            "Official: https://xview2.org/dataset (free signup). Public mirror used: "
            f"{HF_BASE} (archives {ARCHIVES}; CC-BY-NC-SA-4.0).\n"
            "Only post-disaster building label JSONs are extracted (WGS84 lng/lat WKT "
            "polygons + damage subtype + per-image capture_date). VHR imagery is NOT used; "
            "pretraining supplies its own imagery. tier3 archive (18 GB) not downloaded "
            "(tier1 covers all 6 disaster types post-2016).\n"
        )
    return sorted(glob.glob(os.path.join(out, "*.json")))


def _parse_capture_date(s: str | None) -> datetime | None:
    if not s:
        return None
    try:
        dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt


def _scan_file(path: str) -> list[dict[str, Any]]:
    """Rasterize one post-disaster label file's buildings into <=64x64 UTM 10 m patches."""
    with open(path) as f:
        j = json.load(f)
    meta = j.get("metadata", {})
    change_time = _parse_capture_date(meta.get("capture_date"))
    if change_time is None or change_time.year < MIN_YEAR:
        return []
    disaster = meta.get("disaster", "")
    disaster_type = meta.get("disaster_type", "")

    shapes: list[tuple[Any, int]] = []
    for ft in j.get("features", {}).get("lng_lat", []):
        props = ft.get("properties", {})
        if props.get("feature_type") != "building":
            continue
        cid = SUBTYPE_MAP.get(props.get("subtype"))
        if cid is None:
            continue
        wkt = ft.get("wkt")
        if not wkt:
            continue
        try:
            g = shapely.wkt.loads(wkt)
        except Exception:
            continue
        if g.is_empty:
            continue
        shapes.append((g, cid))
    if not shapes:
        return []

    union = shapely.unary_union([g for g, _ in shapes])
    c = union.centroid
    proj = io.utm_projection_for_lonlat(float(c.x), float(c.y))

    px_shapes: list[tuple[Any, int]] = []
    xs: list[float] = []
    ys: list[float] = []
    for g, cid in shapes:
        try:
            pg = rasterize.geom_to_pixels(g, WGS84_PROJECTION, proj)
        except Exception:
            continue
        if pg.is_empty:
            continue
        px_shapes.append((pg, cid))
        x0, y0, x1, y1 = pg.bounds
        xs += [x0, x1]
        ys += [y0, y1]
    if not px_shapes:
        return []
    px_shapes.sort(key=lambda s: PAINT_RANK[s[1]])

    x_min = int(math.floor(min(xs)))
    y_min = int(math.floor(min(ys)))
    x_max = int(math.ceil(max(xs)))
    y_max = int(math.ceil(max(ys)))
    w = max(1, x_max - x_min)
    h = max(1, y_max - y_min)
    stem = os.path.basename(path)[: -len(".json")]

    recs: list[dict[str, Any]] = []
    for ti in range(math.ceil(w / TILE)):
        for tj in range(math.ceil(h / TILE)):
            bx0 = x_min + ti * TILE
            by0 = y_min + tj * TILE
            bounds = (bx0, by0, bx0 + TILE, by0 + TILE)
            arr = rasterize.rasterize_shapes(
                px_shapes, bounds, fill=io.CLASS_NODATA, dtype="uint8", all_touched=True
            )[0]
            present = sorted(int(v) for v in np.unique(arr) if v != io.CLASS_NODATA)
            if not present:
                continue
            recs.append(
                {
                    "array": arr,
                    "crs": proj.crs.to_string(),
                    "bounds": bounds,
                    "classes_present": present,
                    "disaster": disaster,
                    "disaster_type": disaster_type,
                    "change_time": change_time,
                    "source_id": f"{stem}/{ti}_{tj}",
                }
            )
    return recs


def _write_one(rec: dict[str, Any]) -> int:
    from rasterio.crs import CRS

    sample_id = rec["sample_id"]
    if (io.locations_dir(SLUG) / f"{sample_id}.tif").exists():
        return 0
    proj = Projection(CRS.from_string(rec["crs"]), io.RESOLUTION, -io.RESOLUTION)
    bounds = tuple(rec["bounds"])
    ct = rec["change_time"]
    pre_range, post_range = io.pre_post_time_ranges(ct, pre_offset_days=45)
    time_range = (pre_range[0], post_range[1])
    io.write_label_geotiff(
        SLUG, sample_id, rec["array"], proj, bounds, nodata=io.CLASS_NODATA
    )
    io.write_sample_json(
        SLUG,
        sample_id,
        proj,
        bounds,
        time_range,
        change_time=ct,
        source_id=rec["source_id"],
        classes_present=rec["classes_present"],
        pre_time_range=pre_range,
        post_time_range=post_range,
    )
    return 1


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=64)
    parser.add_argument(
        "--probe", action="store_true", help="scan/report only, no writes"
    )
    args = parser.parse_args()

    io.check_disk()
    manifest.write_registry_entry(SLUG, "in_progress")

    files = _ensure_labels()
    print(f"{len(files)} post-disaster label files")

    io.check_disk()
    all_recs: list[dict[str, Any]] = []
    with multiprocessing.Pool(args.workers) as p:
        for recs in star_imap_unordered(p, _scan_file, [{"path": f} for f in files]):
            all_recs.extend(recs)
    print(f"scanned {len(all_recs)} non-empty tiles from {len(files)} label files")

    # Report raw source distributions for the summary.
    dis_counts: Counter = Counter(r["disaster"] for r in all_recs)
    print("candidate tiles per disaster:", dict(dis_counts))

    selected = sampling.select_tiles_per_class(
        all_recs, classes_key="classes_present", per_class=PER_CLASS
    )
    selected.sort(key=lambda r: r["source_id"])
    for i, r in enumerate(selected):
        r["sample_id"] = f"{i:06d}"
    print(f"selected {len(selected)} tiles (of {len(all_recs)})")

    tile_counts = {i: 0 for i in range(NUM_CLASSES)}
    dis_sel: Counter = Counter()
    dtype_sel: Counter = Counter()
    for r in selected:
        dis_sel[r["disaster"]] += 1
        dtype_sel[r["disaster_type"]] += 1
        for c in r["classes_present"]:
            tile_counts[c] += 1
    print(
        "tiles-per-class:", {CLASSES[i][0]: tile_counts[i] for i in range(NUM_CLASSES)}
    )
    print("tiles-per-disaster:", dict(dis_sel))
    print("tiles-per-disaster-type:", dict(dtype_sel))

    if args.probe:
        print("probe only; exiting before writes")
        return

    written = 0
    with multiprocessing.Pool(args.workers) as p:
        for n in star_imap_unordered(p, _write_one, [{"rec": r} for r in selected]):
            written += n
    print(f"wrote {written} new tiles ({len(selected)} total selected)")

    io.write_dataset_metadata(
        SLUG,
        {
            "dataset": SLUG,
            "name": NAME,
            "task_type": "classification",
            "source": "xView2 / xBD (Maxar Open Data)",
            "license": "CC-BY-NC-SA-4.0",
            "provenance": {
                "url": "https://xview2.org/dataset",
                "have_locally": False,
                "annotation_method": "manual, expert-reviewed (VHR pre/post-disaster pairs)",
                "mirror": f"{HF_BASE} (archives {ARCHIVES})",
            },
            "sensors_relevant": ["sentinel2", "sentinel1", "landsat"],
            "classes": [
                {"id": i, "name": name, "description": desc}
                for i, (name, desc) in enumerate(CLASSES)
            ],
            "nodata_value": io.CLASS_NODATA,
            "num_samples": len(selected),
            "tiles_per_class": {
                CLASSES[i][0]: tile_counts[i] for i in range(NUM_CLASSES)
            },
            "tiles_per_disaster": dict(dis_sel),
            "tiles_per_disaster_type": dict(dtype_sel),
            "damage_scale_source": {
                "note": "xBD native 4-level scale collapsed to 2 classes for 10 m "
                "observability; finer scale retained here.",
                "mapping": {
                    "no-damage": "intact_building",
                    "minor-damage": "intact_building",
                    "major-damage": "damaged_building",
                    "destroyed": "damaged_building",
                    "un-classified": "dropped",
                },
            },
            "notes": (
                "xBD/xView2 VHR (~0.4-1.4 m Maxar) post-disaster building-damage labels "
                "rasterized to local-UTM 10 m <=64x64 patches (all_touched). CHANGE dataset: "
                "change_time = per-image post-disaster capture_date (day-resolved), time_range "
                "= 360-day window centered on it. 4-level damage scale collapsed to intact "
                "(no+minor) vs damaged (major+destroyed) building presence for 10 m "
                "observability (damaged painted last so it wins overlaps); un-classified "
                "buildings dropped. Positive-only (non-building = nodata 255); assembly supplies "
                "negatives. Only post-2016 kept (all tier1 disasters are 2016-2019). tier1 "
                "(train+test) only; tier3 (18 GB, incl. pre-2016 tornadoes) not downloaded."
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
