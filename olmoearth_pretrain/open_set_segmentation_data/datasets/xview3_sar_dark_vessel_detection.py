"""Process xView3-SAR (dark vessel detection) into open-set-segmentation detection tiles.

Source: DIU / Global Fishing Watch xView3-SAR challenge (https://iuu.xview.us/).
~1,000 large Sentinel-1 SAR scenes (GRD, ~29,400 x 24,400 px each) with 220k+
georeferenced maritime-object annotations. label_type=points, object detection.

ACCESS / CREDENTIAL GATE
------------------------
The xView3 labels and imagery are distributed ONLY behind a registration/login wall at
https://iuu.xview.us/signup (DrivenData-style account). As of processing, no
unauthenticated open mirror of the LABEL CSVs exists:
  * iuu.xview.us  -> "Register/Login to Download Data" (account required);
  * allenai/sar_vessel_detect, DIUx-xView/xview3-reference, DIUx-xView/SARFish
    -> all point back to iuu.xview.us for the labels;
  * ConnorLuckettDSTG/SARFishSample (public HF) -> only 1 sample GRD + 1 SLC scene,
    NO label CSVs;
  * ConnorLuckettDSTG/SARFish (HF) -> imagery gated + labels still from xView3 site;
  * ai2-prior-sarfish S3 -> only a model checkpoint is public (listing denied).
Per the task SOP this is a `needs-credential` rejection: the user supplies the
registered label CSVs (and scene acquisition times) out of band, after which re-running
this script processes them (it is idempotent).

We only need the LABEL CSVs + per-scene acquisition times to place labels; we do NOT need
the multi-TB SAR imagery. Each label row already carries WGS84 detect_lat / detect_lon, so
tiles are built directly in a local UTM projection from lon/lat -- no scene raster read.

RETRY INPUTS (drop these into raw/xview3_sar_dark_vessel_detection/ once registered):
  * label CSVs: any of GRD_train.csv, GRD_validation.csv (SLC_* also accepted; identical
    label schema). Columns used: scene_id, detect_lat, detect_lon, is_vessel, is_fishing,
    vessel_length_m, confidence.
  * scenes.csv: two columns `scene_id,acquisition_time` (ISO 8601). xView3 scene ids do
    NOT embed the timestamp, so this mapping is required to honor the specific-image
    ~1-hour time range (spec section 5). The Sentinel-1 product names in the challenge
    file listing embed the acquisition datetime (S1x_IW_GRDH_..._YYYYMMDDThhmmss_...), so
    this CSV is trivially derivable from the download manifest.

ENCODING (spec section 4, detection; mirrors olmoearth_sentinel_2_vessels)
--------------------------------------------------------------------------
Classes (unified scheme; background is spatially meaningful within a tile):
  0 background, 1 fishing_vessel, 2 non_fishing_vessel, 3 fixed_structure.
Mapping from the label CSV:
  is_vessel=True  & is_fishing=True   -> 1 fishing_vessel
  is_vessel=True  & is_fishing=False  -> 2 non_fishing_vessel
  is_vessel=False                     -> 3 fixed_structure
Rows with is_vessel unknown (NaN), or a vessel with is_fishing unknown, are dropped for
class purity (recorded in the summary). Detection encoding: a DET_TILE (64) UTM 10 m
context tile per detection, centered on the detection pixel; 1 px positive of the class id,
ringed by a 10 px nodata (255) buffer (coords are not pixel-exact), rest background (0).
Other detections of the SAME scene that fall inside the tile are also marked. We also emit
background-only NEGATIVE tiles at ocean points far (>= NEG_MIN_PX) from every detection,
sampled inside each scene's detection bounding box, so class 0 has real negatives.

Sampling (spec section 5): tiles-per-class balanced, up to PER_CLASS per class, hard cap
25,000. Time range: each detection uses its scene's ~1-hour acquisition window
(specific-image). All splits used (pretraining-agnostic).

Run (idempotent; skips already-written tiles):
  python3 -m olmoearth_pretrain.open_set_segmentation_data.datasets.xview3_sar_dark_vessel_detection
"""

import argparse
import csv
import math
import multiprocessing
import random
from collections import Counter
from datetime import UTC, datetime, timedelta
from typing import Any

import numpy as np
import shapely
import tqdm
from rslearn.const import WGS84_PROJECTION
from rslearn.utils.geometry import STGeometry
from rslearn.utils.mp import star_imap_unordered

from olmoearth_pretrain.open_set_segmentation_data import io, manifest
from olmoearth_pretrain.open_set_segmentation_data.sampling import (
    MAX_SAMPLES_PER_DATASET,
    encode_detection_tile,
    select_tiles_per_class,
)

SLUG = "xview3_sar_dark_vessel_detection"
NAME = "xView3-SAR (dark vessel detection)"
SOURCE_URL = "https://iuu.xview.us/"

BACKGROUND_ID = 0
FISHING_ID = 1
NON_FISHING_ID = 2
STRUCTURE_ID = 3
CLASS_NAMES = {
    BACKGROUND_ID: "background",
    FISHING_ID: "fishing_vessel",
    NON_FISHING_ID: "non_fishing_vessel",
    STRUCTURE_ID: "fixed_structure",
}
CLASS_DESCRIPTIONS = {
    BACKGROUND_ID: "Open water / non-object ocean surface within the tile.",
    FISHING_ID: "Vessel engaged in fishing (is_vessel & is_fishing), incl. dark/"
    "non-broadcasting fishing vessels correlated via AIS + manual SAR analysis.",
    NON_FISHING_ID: "Non-fishing vessel (is_vessel & not is_fishing): cargo, tanker, "
    "passenger, etc.",
    STRUCTURE_ID: "Fixed ocean structure (is_vessel=False): oil/gas platform, wind "
    "turbine, aquaculture, or other persistent installation.",
}

# Detection encoding (spec section 4).
DET_TILE = 64
DET_POS_SIZE = 1
DET_BUFFER = 10
NEG_MIN_PX = 30  # negatives kept >= this many px from any detection
PER_CLASS = 1000
NEG_PER_CLASS_FACTOR = 1.0  # negatives target ~ PER_CLASS
SEED = 42

LABEL_CSV_NAMES = (
    "GRD_train.csv",
    "GRD_validation.csv",
    "SLC_train.csv",
    "SLC_validation.csv",
)
SCENES_CSV = "scenes.csv"


# --------------------------------------------------------------------------- parsing


def _as_bool(v: str) -> bool | None:
    s = (v or "").strip().lower()
    if s in ("true", "1", "t", "yes"):
        return True
    if s in ("false", "0", "f", "no"):
        return False
    return None


def _class_for(is_vessel: bool | None, is_fishing: bool | None) -> int | None:
    if is_vessel is None:
        return None
    if not is_vessel:
        return STRUCTURE_ID
    if is_fishing is None:
        return None  # vessel of unknown fishing status -> drop for class purity
    return FISHING_ID if is_fishing else NON_FISHING_ID


def _load_scene_times(raw) -> dict[str, tuple[datetime, datetime]]:
    """scene_id -> ~1-hour (start, end) window from scenes.csv (acquisition_time ISO)."""
    path = raw / SCENES_CSV
    out: dict[str, tuple[datetime, datetime]] = {}
    if not path.exists():
        return out
    with path.open() as f:
        for row in csv.DictReader(f):
            sid = row.get("scene_id")
            ts = row.get("acquisition_time") or row.get("timestamp")
            if not sid or not ts:
                continue
            t = datetime.fromisoformat(ts.replace("Z", "+00:00"))
            if t.tzinfo is None:
                t = t.replace(tzinfo=UTC)
            out[sid] = (t - timedelta(minutes=30), t + timedelta(minutes=30))
    return out


def _load_detections(raw) -> list[dict[str, Any]]:
    """Read all present label CSVs into detection records (class-mapped, geolocated)."""
    dets: list[dict[str, Any]] = []
    for name in LABEL_CSV_NAMES:
        path = raw / name
        if not path.exists():
            continue
        with path.open() as f:
            for row in csv.DictReader(f):
                try:
                    lat = float(row["detect_lat"])
                    lon = float(row["detect_lon"])
                except (KeyError, ValueError, TypeError):
                    continue
                cls = _class_for(
                    _as_bool(row.get("is_vessel", "")),
                    _as_bool(row.get("is_fishing", "")),
                )
                if cls is None:
                    continue
                dets.append(
                    {
                        "scene_id": row.get("scene_id", ""),
                        "lat": lat,
                        "lon": lon,
                        "cls": cls,
                        "detect_id": row.get("detect_id", ""),
                    }
                )
    return dets


# --------------------------------------------------------------------------- encoding


def _write_tile(rec: dict[str, Any]) -> str:
    sample_id = rec["sample_id"]
    if (io.locations_dir(SLUG) / f"{sample_id}.tif").exists():
        return "skip"
    proj, cx, cy = io.lonlat_to_utm_pixel(rec["lon"], rec["lat"])
    bounds = io.centered_bounds(cx, cy, DET_TILE, DET_TILE)
    x_min, y_min = bounds[0], bounds[1]
    positives: list[tuple[int, int, int]] = []
    # Mark this detection plus any same-scene detection landing inside the tile.
    for d in rec["scene_dets"]:
        g = STGeometry(WGS84_PROJECTION, shapely.Point(d["lon"], d["lat"]), None)
        p = g.to_projection(proj).shp
        lc = int(math.floor(p.x)) - x_min
        lr = int(math.floor(p.y)) - y_min
        if 0 <= lc < DET_TILE and 0 <= lr < DET_TILE:
            positives.append((lr, lc, d["cls"]))
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
        rec["time_range"],
        source_id=rec["source_id"],
        classes_present=sorted(set(np.unique(arr).tolist()) - {io.CLASS_NODATA}),
    )
    return "pos"


def _write_negative(rec: dict[str, Any]) -> str:
    sample_id = rec["sample_id"]
    if (io.locations_dir(SLUG) / f"{sample_id}.tif").exists():
        return "skip"
    proj, cx, cy = io.lonlat_to_utm_pixel(rec["lon"], rec["lat"])
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
        rec["time_range"],
        source_id=rec["source_id"],
        classes_present=[BACKGROUND_ID],
    )
    return "neg"


def _dispatch(rec: dict[str, Any]) -> str:
    return _write_negative(rec) if rec["kind"] == "neg" else _write_tile(rec)


# --------------------------------------------------------------------------- negatives


def _make_negatives(by_scene: dict[str, list[dict]], scene_times, rng, n_target):
    """Background-only tiles at ocean points far from any detection, inside each scene's
    detection bbox. Approximate degrees->px via a coarse 10 m ~ 1e-4 deg conversion; the
    NEG_MIN_PX guard uses a generous margin so it is robust to that approximation.
    """
    negs: list[dict[str, Any]] = []
    deg_per_px = 10.0 / 111_000.0  # ~10 m at the equator, adequate for a spacing guard
    min_deg = NEG_MIN_PX * deg_per_px
    scenes = list(by_scene)
    rng.shuffle(scenes)
    for sid in scenes:
        if len(negs) >= n_target:
            break
        dets = by_scene[sid]
        if sid not in scene_times or len(dets) < 2:
            continue
        lats = [d["lat"] for d in dets]
        lons = [d["lon"] for d in dets]
        lo_lat, hi_lat, lo_lon, hi_lon = min(lats), max(lats), min(lons), max(lons)
        if hi_lat - lo_lat < 4 * min_deg or hi_lon - lo_lon < 4 * min_deg:
            continue
        for _ in range(6):  # a few tries per scene
            plat = rng.uniform(lo_lat, hi_lat)
            plon = rng.uniform(lo_lon, hi_lon)
            if all(
                abs(plat - d["lat"]) > min_deg or abs(plon - d["lon"]) > min_deg
                for d in dets
            ):
                negs.append(
                    {
                        "kind": "neg",
                        "lat": plat,
                        "lon": plon,
                        "time_range": scene_times[sid],
                        "source_id": f"{sid}/background",
                    }
                )
                break
    return negs


# --------------------------------------------------------------------------- main


def _reject_needs_credential() -> None:
    notes = (
        "needs-credential: xView3-SAR labels require registration/login at "
        "https://iuu.xview.us/signup. Tried open mirrors (allenai/sar_vessel_detect, "
        "DIUx-xView/xview3-reference, DIUx-xView/SARFish, ConnorLuckettDSTG/SARFishSample "
        "(public, sample imagery only, no label CSVs), ai2-prior-sarfish S3) -- none host "
        "the label CSVs unauthenticated. To process: place GRD_train.csv / "
        "GRD_validation.csv and scenes.csv (scene_id,acquisition_time) in "
        f"raw/{SLUG}/ and re-run; the script is idempotent."
    )
    manifest.write_registry_entry(SLUG, "rejected", notes=notes)
    print("REJECTED:", notes, flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=64)
    args = parser.parse_args()

    io.check_disk()
    manifest.write_registry_entry(SLUG, "in_progress")

    raw = io.raw_dir(SLUG)
    raw.mkdir(parents=True, exist_ok=True)

    present = [n for n in LABEL_CSV_NAMES if (raw / n).exists()]
    scene_times = _load_scene_times(raw)
    if not present or not scene_times:
        # Credential gate: label CSVs (and scene acquisition times) not available.
        _reject_needs_credential()
        return

    dets = _load_detections(raw)
    print(f"loaded {len(dets)} class-mapped detections from {present}", flush=True)
    by_scene: dict[str, list[dict]] = {}
    for d in dets:
        by_scene.setdefault(d["scene_id"], []).append(d)

    # Positive candidates: one tile per detection whose scene has a known acquisition time.
    cands: list[dict[str, Any]] = []
    for d in dets:
        if d["scene_id"] not in scene_times:
            continue
        cands.append(
            {
                "kind": "pos",
                "lat": d["lat"],
                "lon": d["lon"],
                "classes_present": [d["cls"]],
                "scene_dets": by_scene[d["scene_id"]],
                "time_range": scene_times[d["scene_id"]],
                "source_id": f"{d['scene_id']}/{d['detect_id']}",
            }
        )
    selected = select_tiles_per_class(
        cands,
        classes_key="classes_present",
        per_class=PER_CLASS,
        total_cap=MAX_SAMPLES_PER_DATASET,
        seed=SEED,
    )
    io.check_disk()

    rng = random.Random(SEED)
    n_neg = int(PER_CLASS * NEG_PER_CLASS_FACTOR)
    negs = _make_negatives(by_scene, scene_times, rng, n_neg)

    all_recs = selected + negs
    for i, r in enumerate(all_recs):
        r["sample_id"] = f"{i:06d}"
    print(
        f"selected {len(selected)} positive + {len(negs)} negative = {len(all_recs)}",
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

    with (raw / "SOURCE.txt").open("w") as f:
        f.write(
            f"xView3-SAR labels (registered download) processed from {present}.\n"
            f"scenes.csv provided acquisition times for {len(scene_times)} scenes.\n"
        )

    io.write_dataset_metadata(
        SLUG,
        {
            "dataset": SLUG,
            "name": NAME,
            "task_type": "classification",
            "source": "DIU / Global Fishing Watch (xView3-SAR)",
            "license": "open (non-commercial research)",
            "provenance": {
                "url": SOURCE_URL,
                "have_locally": False,
                "annotation_method": "manual + AIS-assisted",
            },
            "sensors_relevant": ["sentinel1"],
            "classes": [
                {"id": i, "name": CLASS_NAMES[i], "description": CLASS_DESCRIPTIONS[i]}
                for i in sorted(CLASS_NAMES)
            ],
            "nodata_value": io.CLASS_NODATA,
            "detection_encoding": {
                "tile_size": DET_TILE,
                "positive_size": DET_POS_SIZE,
                "buffer_size": DET_BUFFER,
            },
            "num_samples": len(all_recs),
            "notes": "Sentinel-1 SAR dark-vessel detections. Tiles built directly from WGS84 "
            "detect_lat/detect_lon (no SAR raster read). Specific-image ~1-hour time "
            "range per scene acquisition. See dataset summary.",
        },
    )
    manifest.write_registry_entry(
        SLUG, "completed", task_type="classification", num_samples=len(all_recs)
    )
    print(f"done: {len(all_recs)} samples", flush=True)


if __name__ == "__main__":
    multiprocessing.set_start_method("forkserver", force=True)
    main()
