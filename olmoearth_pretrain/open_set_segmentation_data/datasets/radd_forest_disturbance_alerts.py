"""Process RADD (RAdar for Detecting Deforestation) forest-disturbance alerts.

Source: Wageningen University (WUR) RADD alerts, contributed to WRI Global Forest Watch,
served in Google Earth Engine as the ImageCollection ``projects/radar-wur/raddalert/v1``.
RADD provides near-real-time forest-disturbance alerts for the humid tropics at 10 m,
derived from cloud-penetrating Sentinel-1 C-band radar. Coverage: South America (``sa``),
Africa / Congo Basin (``africa``) and insular Southeast Asia (``asia``) -- the three main
RADD geographies used here.

Each geography's latest (cumulative) alert image has two bands:
  * ``Alert`` -- 2 = unconfirmed (low confidence), 3 = confirmed (high confidence).
  * ``Date``  -- date of first detected disturbance, encoded **YYDOY**: value // 1000 =
                 (year - 2000), value % 1000 = day-of-year. e.g. 24184 -> 2024, DOY 184
                 (2024-07-02); 22230 -> 2022, DOY 230. This is DAY-precise, well within the
                 spec's ~1-2 month change-timing requirement.
A per-geography ``forest_baseline`` image (band ``constant`` = 1 over the primary-forest
baseline extent, masked elsewhere) delimits the valid forest area.

This is a genuine dated CHANGE dataset (forest -> disturbed). Per spec §5 we use the
change_time scheme: each disturbance tile's ``change_time`` is the representative (median)
decoded disturbance date of the confirmed alerts forming a single temporally-coherent event
within the tile. That ``change_time`` splits the tile into two adjacent six-month windows
(via ``io.pre_post_time_ranges``): ``pre_time_range`` = the ~6 months (<=183 days)
immediately before it and ``post_time_range`` = the ~6 months (<=183 days) immediately after,
with ``time_range`` = null. The label is a **mask of where** the disturbance occurred, and
pretraining pairs the "before" image stack with the "after" stack and probes on their
difference.

Label scheme (uint8, single band, local UTM 10 m):
  0   stable forest   (forest-baseline pixel with no alert) -- background.
  1   disturbance     (confirmed alert, Alert==3, whose decoded date lies within the tile's
                       event window [change_time ± EVENT_HALF_DAYS]).
  255 nodata/ignore   (outside the forest baseline; low-confidence alerts (Alert==2);
                       confirmed alerts of a *different* date than this tile's event).

Only confirmed (high-confidence, Alert==3) alerts define the positive class; low-confidence
alerts are ignored (255) to keep the change mask clean. Disturbance tiles require at least
MIN_DISTURBED confirmed in-window pixels so the mask carries real signal. We also emit
stable-forest background negatives (all class 0 within the forest baseline, no alert) with
``change_time=null`` and a static representative 1-year window.

Post-2016 rule: RADD begins ~2019 (S. America/Asia baseline 2019, Africa 2018), so every
alert is well inside the Sentinel era; no pre-2016 filtering is needed.

Access: Earth Engine service-account credentials from ``.env``
(spec §8 authorizes ``.env`` creds; the GFW data-lake S3 mirror is requester-pays). We do
NOT bulk-download the pan-tropical tiles: candidate centers are sampled with EE
``stratifiedSample`` over a grid of 2° cells across each geography, then each ≤64×64 label
patch is fetched directly (reprojected to local UTM at 10 m, nearest-neighbour) via
``ee.data.computePixels``.

Run: ``python3 -m olmoearth_pretrain.open_set_segmentation_data.datasets.radd_forest_disturbance_alerts``
Idempotent: candidate samples are cached to raw/, and existing locations/{id}.tif are skipped.
"""

import argparse
import json
import multiprocessing
import random
import time
from collections import Counter, defaultdict
from datetime import UTC, datetime, timedelta
from typing import Any

import numpy as np
import tqdm

from olmoearth_pretrain.open_set_segmentation_data import io

SLUG = "radd_forest_disturbance_alerts"
NAME = "RADD Forest Disturbance Alerts"
URL = "https://data.globalforestwatch.org/datasets/gfw::deforestation-alerts-radd/about"
EE_COLLECTION = "projects/radar-wur/raddalert/v1"
GEE_KEY = "/etc/credentials/gcp_credentials.json"

# Three main RADD geographies -> friendly region names.
REGIONS = {
    "sa": "Amazon / South America",
    "africa": "Congo Basin / Africa",
    "asia": "Insular Southeast Asia",
}
# Forest-focused bounding boxes (lon_min, lat_min, lon_max, lat_max) per geography; we grid
# these into CELL-degree cells for spatially-distributed sampling. Chosen to cover the humid
# tropical forest belt while keeping the number of (mostly cheap) cell queries bounded.
REGION_BBOX = {
    "sa": (-79.0, -18.0, -44.0, 10.0),
    "africa": (-16.0, -13.0, 42.0, 12.0),
    "asia": (95.0, -11.0, 141.0, 8.0),
}

TILE = 64  # 64 px @ 10 m = 640 m
CELL = 2.0  # sampling grid cell size, degrees
SAMPLE_SCALE = 30  # coarser scale for candidate discovery (fast); date re-read at 10 m
CELL_POINTS = 80  # confirmed-alert candidate points requested per cell
CELL_BG_POINTS = 12  # stable-forest background candidate points per cell
SEED = 42

STABLE_ID = 0
DISTURB_ID = 1
EVENT_HALF_DAYS = 45  # tile event window = change_time ± 45 d (single coherent event)
WINDOW_HALF_DAYS = 180  # time_range = change_time ± 180 d (360-day pairing window)
MIN_DISTURBED = (
    20  # >= 20 confirmed in-window px (~0.2 ha) required for a disturbance tile
)
MAX_BG_ALERT_PX = (
    5  # a background tile may contain at most this many confirmed alert px
)
BG_STATIC_YEAR = 2022  # representative 1-year window for background negatives

DISTURB_TARGET_PER_REGION = 1000
BG_TARGET_PER_REGION = 300

CLASS_DEFS = [
    (
        0,
        "stable_forest",
        "Primary humid-tropical-forest baseline pixel with no RADD disturbance alert: "
        "standing forest that was not flagged as disturbed. Background/negative class.",
    ),
    (
        1,
        "forest_disturbance",
        "Confirmed (high-confidence) RADD forest-disturbance alert derived from Sentinel-1 "
        "radar: forest cleared/degraded (deforestation, logging, etc.) whose first detection "
        "date falls within this tile's temporally-coherent event window. Low-confidence "
        "(unconfirmed) alerts are ignored (nodata) rather than labeled disturbance.",
    ),
]

# ----------------------------------------------------------------------------------------
# Earth Engine helpers (lazily initialised once per process; EE objects are not picklable).
# ----------------------------------------------------------------------------------------
_EE_READY = False
_IMG_CACHE: dict[str, Any] = {}


def _ensure_ee() -> None:
    global _EE_READY
    if _EE_READY:
        return
    import ee

    info = json.load(open(GEE_KEY))
    ee.Initialize(ee.ServiceAccountCredentials(info["client_email"], GEE_KEY))
    _EE_READY = True


def _region_images(region: str):
    """Return (comb, conf3_pts, stable_pts) EE images for a geography (cached per process).

    comb: (Date int32, Alert int16, forest uint8) for label-patch fetching.
    conf3_pts: Date band masked to confirmed alerts + a constant class band 'c' for sampling.
    stable_pts: constant class band 's' over stable forest (forest & no alert) for sampling.
    """
    if region in _IMG_CACHE:
        return _IMG_CACHE[region]
    import ee

    _ensure_ee()
    radd = ee.ImageCollection(EE_COLLECTION)
    alert = ee.Image(
        radd.filterMetadata("layer", "contains", "alert")
        .filterMetadata("geography", "equals", region)
        .sort("system:time_end", False)
        .first()
    )
    fb = ee.Image(
        radd.filterMetadata("layer", "contains", "forest_baseline")
        .filterMetadata("geography", "equals", region)
        .first()
    )
    date = alert.select("Date").toInt32().rename("Date")
    conf = alert.select("Alert").toInt16().rename("Alert")
    forest = fb.select(0).unmask(0).toUint8().rename("forest")
    comb = date.addBands(conf).addBands(forest)

    conf3 = conf.eq(3)
    conf3_pts = (
        date.updateMask(conf3)
        .rename("Date")
        .addBands(ee.Image(1).updateMask(conf3).toUint8().rename("c"))
    )
    conf_any = conf.unmask(-1)
    stable = forest.eq(1).And(conf_any.neq(2)).And(conf_any.neq(3))
    stable_pts = stable.selfMask().toUint8().rename("s")

    _IMG_CACHE[region] = (comb, conf3_pts, stable_pts)
    return _IMG_CACHE[region]


# ----------------------------------------------------------------------------------------
# Candidate sampling (one task per grid cell).
# ----------------------------------------------------------------------------------------
def _sample_cell(region: str, lo: float, la: float, kind: str) -> list[dict[str, Any]]:
    import ee

    _ensure_ee()
    comb, conf3_pts, stable_pts = _region_images(region)
    reg = ee.Geometry.Rectangle([lo, la, lo + CELL, la + CELL], None, False)
    for attempt in range(4):
        try:
            if kind == "dist":
                fc = conf3_pts.stratifiedSample(
                    numPoints=CELL_POINTS,
                    classBand="c",
                    region=reg,
                    scale=SAMPLE_SCALE,
                    seed=SEED,
                    geometries=True,
                    tileScale=8,
                )
            else:
                fc = stable_pts.stratifiedSample(
                    numPoints=CELL_BG_POINTS,
                    classBand="s",
                    region=reg,
                    scale=SAMPLE_SCALE,
                    seed=SEED,
                    geometries=True,
                    tileScale=8,
                )
            feats = fc.getInfo()["features"]
            break
        except Exception as e:  # noqa: BLE001
            if attempt == 3:
                print(
                    f"  cell sample failed {region} ({lo},{la}) {kind}: {e}", flush=True
                )
                return []
            time.sleep(2 * (attempt + 1))
    out = []
    for f in feats:
        g = f.get("geometry")
        if not g:
            continue
        lon, lat = g["coordinates"][0], g["coordinates"][1]
        d = f["properties"].get("Date") if kind == "dist" else None
        out.append({"region": region, "lon": lon, "lat": lat, "date": d, "kind": kind})
    return out


def _cells(region: str) -> list[tuple[float, float]]:
    lo0, la0, lo1, la1 = REGION_BBOX[region]
    cells = []
    la = la0
    while la < la1:
        lo = lo0
        while lo < lo1:
            cells.append((round(lo, 3), round(la, 3)))
            lo += CELL
        la += CELL
    return cells


def _decode_yydoy(v: int) -> datetime | None:
    """Decode a RADD YYDOY Date value -> UTC datetime, or None if implausible."""
    if v is None or v <= 0:
        return None
    yy, doy = divmod(int(v), 1000)
    year = 2000 + yy
    if not (2015 <= year <= 2030) or not (1 <= doy <= 366):
        return None
    try:
        return datetime(year, 1, 1, tzinfo=UTC) + timedelta(days=doy - 1)
    except Exception:  # noqa: BLE001
        return None


# ----------------------------------------------------------------------------------------
# Tile fetching + label construction.
# ----------------------------------------------------------------------------------------
def _fetch_window(region: str, lon: float, lat: float):
    """Fetch a 64x64 (Date, Alert, forest) window reprojected to local UTM 10 m.

    Returns (arr_dict, projection, bounds) or None on failure.
    """
    import ee

    _ensure_ee()
    comb, _, _ = _region_images(region)
    proj, col, row = io.lonlat_to_utm_pixel(lon, lat)
    bounds = io.centered_bounds(col, row, TILE, TILE)
    xmin, ymin, _xmax, _ymax = bounds
    req = {
        "expression": comb,
        "fileFormat": "NUMPY_NDARRAY",
        "grid": {
            "dimensions": {"width": TILE, "height": TILE},
            "affineTransform": {
                "scaleX": io.RESOLUTION,
                "shearX": 0,
                "translateX": io.RESOLUTION * xmin,
                "shearY": 0,
                "scaleY": -io.RESOLUTION,
                "translateY": -io.RESOLUTION * ymin,
            },
            "crsCode": proj.crs.to_string(),
        },
    }
    for attempt in range(4):
        try:
            arr = ee.data.computePixels(req)
            return arr, proj, bounds
        except Exception as e:  # noqa: BLE001
            if attempt == 3:
                print(
                    f"  computePixels failed {region} ({lon:.3f},{lat:.3f}): {e}",
                    flush=True,
                )
                return None
            time.sleep(2 * (attempt + 1))
    return None


def _write_disturbance_tile(rec: dict[str, Any]) -> dict[str, Any] | None:
    sample_id = rec["sample_id"]
    tif = io.locations_dir(SLUG) / f"{sample_id}.tif"
    if tif.exists():
        return {"sample_id": sample_id, "kind": "dist", "cached": True}
    seed_dt = _decode_yydoy(rec["date"])
    if seed_dt is None:
        return None
    fetched = _fetch_window(rec["region"], rec["lon"], rec["lat"])
    if fetched is None:
        return None
    arr, proj, bounds = fetched
    D = arr["Date"].astype(np.int64)
    A = arr["Alert"].astype(np.int64)
    F = arr["forest"].astype(np.int64)

    conf3 = A == 3
    alert_any = (A == 2) | (A == 3)
    valid_date = conf3 & (D > 0)

    # Decode confirmed-alert dates to ordinals (few unique values in a 64x64 tile).
    ord_arr = np.zeros(D.shape, dtype=np.int64)
    for v in np.unique(D[valid_date]):
        dt = _decode_yydoy(int(v))
        if dt is None:
            valid_date &= D != v
            continue
        ord_arr[(D == v) & valid_date] = dt.toordinal()

    seed_ord = seed_dt.toordinal()
    in_window = valid_date & (np.abs(ord_arr - seed_ord) <= EVENT_HALF_DAYS)
    n_dist = int(in_window.sum())
    if n_dist < MIN_DISTURBED:
        return None
    change_ord = int(np.median(ord_arr[in_window]))
    change_dt = datetime.fromordinal(change_ord).replace(tzinfo=UTC)

    mask = np.full(D.shape, io.CLASS_NODATA, dtype=np.uint8)
    stable = (F == 1) & (~alert_any)
    mask[stable] = STABLE_ID
    mask[in_window] = DISTURB_ID
    # Everything else (non-forest, low-conf alerts, out-of-window confirmed alerts) stays 255.

    pre_range, post_range = io.pre_post_time_ranges(change_dt)
    tr = (pre_range[0], post_range[1])  # outer bounding span
    present = [DISTURB_ID]
    if bool((mask == STABLE_ID).any()):
        present.insert(0, STABLE_ID)
    io.write_label_geotiff(SLUG, sample_id, mask, proj, bounds, nodata=io.CLASS_NODATA)
    io.write_sample_json(
        SLUG,
        sample_id,
        proj,
        bounds,
        tr,
        change_time=change_dt,
        source_id=f"radd:{rec['region']}:{rec['lon']:.4f},{rec['lat']:.4f}",
        classes_present=present,
        pre_time_range=pre_range,
        post_time_range=post_range,
    )
    return {
        "sample_id": sample_id,
        "kind": "dist",
        "region": rec["region"],
        "year": change_dt.year,
        "n_dist": n_dist,
        "classes": present,
    }


def _write_background_tile(rec: dict[str, Any]) -> dict[str, Any] | None:
    sample_id = rec["sample_id"]
    tif = io.locations_dir(SLUG) / f"{sample_id}.tif"
    if tif.exists():
        return {"sample_id": sample_id, "kind": "bg", "cached": True}
    fetched = _fetch_window(rec["region"], rec["lon"], rec["lat"])
    if fetched is None:
        return None
    arr, proj, bounds = fetched
    A = arr["Alert"].astype(np.int64)
    F = arr["forest"].astype(np.int64)
    conf3 = A == 3
    if int(conf3.sum()) > MAX_BG_ALERT_PX:
        return None  # not a clean negative
    if int((F == 1).sum()) < MIN_DISTURBED:
        return None  # too little forest to be a useful stable-forest negative
    alert_any = (A == 2) | (A == 3)
    mask = np.full(F.shape, io.CLASS_NODATA, dtype=np.uint8)
    mask[(F == 1) & (~alert_any)] = STABLE_ID
    tr = io.year_range(BG_STATIC_YEAR)
    io.write_label_geotiff(SLUG, sample_id, mask, proj, bounds, nodata=io.CLASS_NODATA)
    io.write_sample_json(
        SLUG,
        sample_id,
        proj,
        bounds,
        tr,
        change_time=None,
        source_id=f"radd-bg:{rec['region']}:{rec['lon']:.4f},{rec['lat']:.4f}",
        classes_present=[STABLE_ID],
    )
    return {
        "sample_id": sample_id,
        "kind": "bg",
        "region": rec["region"],
        "classes": [STABLE_ID],
    }


# ----------------------------------------------------------------------------------------
# Selection helpers.
# ----------------------------------------------------------------------------------------
def _snap(lon: float, lat: float) -> tuple[int, int]:
    """Snap to a ~640 m grid so we don't pick heavily-overlapping tiles."""
    step = TILE * io.RESOLUTION / 111320.0  # ~degrees per tile at equator
    return (int(round(lon / step)), int(round(lat / step)))


def _dedupe(cands: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[tuple[str, int, int]] = set()
    out = []
    for c in cands:
        k = (c["region"], *_snap(c["lon"], c["lat"]))
        if k in seen:
            continue
        seen.add(k)
        out.append(c)
    return out


def _select_disturbance(
    cands: list[dict[str, Any]], target: int
) -> list[dict[str, Any]]:
    """Round-robin across (region, year) for spatial + temporal diversity."""
    rng = random.Random(SEED)
    buckets: dict[tuple[str, int], list] = defaultdict(list)
    for c in cands:
        dt = _decode_yydoy(c["date"])
        if dt is None:
            continue
        c["year"] = dt.year
        buckets[(c["region"], dt.year)].append(c)
    keys = sorted(buckets)
    for k in keys:
        buckets[k].sort(key=lambda c: (c["lon"], c["lat"]))
        rng.shuffle(buckets[k])
    idx = {k: 0 for k in keys}
    picked: list[dict[str, Any]] = []
    limit = target * len(REGIONS)
    while len(picked) < limit:
        progressed = False
        for k in keys:
            if idx[k] < len(buckets[k]):
                picked.append(buckets[k][idx[k]])
                idx[k] += 1
                progressed = True
        if not progressed:
            break
    return picked


# ----------------------------------------------------------------------------------------
def _write_source_txt() -> None:
    io.raw_dir(SLUG).mkdir(parents=True, exist_ok=True)
    with (io.raw_dir(SLUG) / "SOURCE.txt").open("w") as f:
        f.write(
            "RADD (RAdar for Detecting Deforestation) forest-disturbance alerts.\n"
            "Wageningen University (WUR) / WRI Global Forest Watch.\n"
            f"Earth Engine collection: {EE_COLLECTION}\n"
            f"Portal: {URL}\n"
            "Bands: Alert (2=low/unconfirmed, 3=high/confirmed), Date (YYDOY: "
            "value//1000 = year-2000, value%1000 = day-of-year).\n"
            "Geographies used: sa, africa, asia. Accessed via EE service account "
            "(.env). Candidate centers sampled with stratifiedSample over 2-deg cells; "
            "label patches fetched with ee.data.computePixels (reprojected to UTM 10 m).\n"
        )


def _gather_candidates(pool, region: str) -> tuple[list, list]:
    """Sample disturbance + background candidates for one region (cached to raw/)."""
    from rslearn.utils.mp import star_imap_unordered

    cache = io.raw_dir(SLUG) / f"candidates_{region}.json"
    if cache.exists():
        with cache.open() as f:
            data = json.load(f)
        return data["dist"], data["bg"]

    cells = _cells(region)
    rng = random.Random(SEED)
    rng.shuffle(cells)
    dist_args = [dict(region=region, lo=lo, la=la, kind="dist") for lo, la in cells]
    bg_args = [dict(region=region, lo=lo, la=la, kind="bg") for lo, la in cells]

    dist: list[dict[str, Any]] = []
    for res in tqdm.tqdm(
        star_imap_unordered(pool, _sample_cell, dist_args),
        total=len(dist_args),
        desc=f"{region} dist-cells",
    ):
        dist.extend(res)
    bg: list[dict[str, Any]] = []
    for res in tqdm.tqdm(
        star_imap_unordered(pool, _sample_cell, bg_args),
        total=len(bg_args),
        desc=f"{region} bg-cells",
    ):
        bg.extend(res)

    dist = _dedupe(dist)
    bg = _dedupe(bg)
    io.raw_dir(SLUG).mkdir(parents=True, exist_ok=True)
    tmp = cache.parent / (cache.name + ".tmp")
    with tmp.open("w") as f:
        json.dump({"dist": dist, "bg": bg}, f)
    tmp.rename(cache)
    return dist, bg


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=32)
    args = parser.parse_args()

    from rslearn.utils.mp import star_imap_unordered

    from olmoearth_pretrain.open_set_segmentation_data import manifest

    manifest.write_registry_entry(SLUG, "in_progress")
    io.check_disk()
    _write_source_txt()

    # ---- Candidate sampling ----
    all_dist: list[dict[str, Any]] = []
    all_bg: list[dict[str, Any]] = []
    with multiprocessing.Pool(args.workers, initializer=_ensure_ee) as pool:
        for region in REGIONS:
            dist, bg = _gather_candidates(pool, region)
            print(
                f"{region}: {len(dist)} dist candidates, {len(bg)} bg candidates",
                flush=True,
            )
            all_dist.extend(dist)
            all_bg.extend(bg)

    io.check_disk()

    # ---- Selection ----
    sel_dist = _select_disturbance(all_dist, DISTURB_TARGET_PER_REGION)
    rng = random.Random(SEED)
    bg_by_region: dict[str, list] = defaultdict(list)
    for c in all_bg:
        bg_by_region[c["region"]].append(c)
    sel_bg: list[dict[str, Any]] = []
    for region, items in bg_by_region.items():
        items.sort(key=lambda c: (c["lon"], c["lat"]))
        rng.shuffle(items)
        sel_bg.extend(items[:BG_TARGET_PER_REGION])

    # Deterministic sample_id assignment: disturbance tiles first, then background.
    for i, r in enumerate(sel_dist):
        r["sample_id"] = f"{i:06d}"
    for j, r in enumerate(sel_bg):
        r["sample_id"] = f"{len(sel_dist) + j:06d}"
    print(
        f"selected {len(sel_dist)} disturbance + {len(sel_bg)} background candidates",
        flush=True,
    )

    # ---- Fetch + write tiles ----
    io.check_disk()
    with multiprocessing.Pool(args.workers, initializer=_ensure_ee) as pool:
        dist_results = list(
            tqdm.tqdm(
                star_imap_unordered(
                    pool, _write_disturbance_tile, [dict(rec=r) for r in sel_dist]
                ),
                total=len(sel_dist),
                desc="disturbance tiles",
            )
        )
        bg_results = list(
            tqdm.tqdm(
                star_imap_unordered(
                    pool, _write_background_tile, [dict(rec=r) for r in sel_bg]
                ),
                total=len(sel_bg),
                desc="background tiles",
            )
        )

    dist_ok = [r for r in dist_results if r]
    bg_ok = [r for r in bg_results if r]
    n_dist = len(dist_ok)
    n_bg = len(bg_ok)
    total = n_dist + n_bg

    region_counts = Counter(r.get("region") for r in dist_ok if r.get("region"))
    year_counts = Counter(r.get("year") for r in dist_ok if r.get("year"))
    bg_region_counts = Counter(r.get("region") for r in bg_ok if r.get("region"))
    # class-level tile counts (tiles containing each class)
    n_tiles_with_stable = (
        sum(1 for r in dist_ok if STABLE_ID in r.get("classes", [])) + n_bg
    )
    n_tiles_with_disturb = n_dist

    io.write_dataset_metadata(
        SLUG,
        {
            "dataset": SLUG,
            "name": NAME,
            "task_type": "classification",
            "source": "Wageningen University (WUR) RADD / WRI Global Forest Watch",
            "license": "CC-BY-4.0",
            "provenance": {
                "url": URL,
                "have_locally": False,
                "annotation_method": "derived-product (Sentinel-1 radar, RADD algorithm)",
                "ee_collection": EE_COLLECTION,
            },
            "sensors_relevant": ["sentinel2", "sentinel1", "landsat"],
            "classes": [
                {"id": cid, "name": name, "description": desc}
                for cid, name, desc in CLASS_DEFS
            ],
            "nodata_value": io.CLASS_NODATA,
            "num_samples": total,
            "class_tile_counts": {
                "stable_forest": n_tiles_with_stable,
                "forest_disturbance": n_tiles_with_disturb,
            },
            "disturbance_tiles": n_dist,
            "background_tiles": n_bg,
            "disturbance_tiles_per_region": dict(region_counts),
            "background_tiles_per_region": dict(bg_region_counts),
            "disturbance_tiles_per_year": {
                str(y): year_counts[y] for y in sorted(year_counts)
            },
            "tile_size": TILE,
            "change_time_scheme": True,
            "event_window_days": 2 * EVENT_HALF_DAYS,
            "time_range_days": 2 * WINDOW_HALF_DAYS,
            "date_encoding": "YYDOY: value//1000 = year-2000, value%1000 = day-of-year",
            "notes": (
                "64x64 UTM 10 m tiles from WUR RADD Sentinel-1 forest-disturbance alerts "
                "(EE projects/radar-wur/raddalert/v1), geographies sa/africa/asia. "
                "Class 1 = confirmed (Alert==3) disturbance whose decoded YYDOY date lies "
                f"within change_time +/- {EVENT_HALF_DAYS} d (single coherent event); class 0 "
                "= stable forest baseline w/o alert; 255 = nodata (non-forest, low-confidence "
                "alerts, or confirmed alerts of a different date). Dated change labels: "
                "change_time = median decoded date of in-window disturbed pixels (day-precise), "
                f"time_range = +/-{WINDOW_HALF_DAYS} d centered on it. Disturbance tiles require "
                f">= {MIN_DISTURBED} in-window confirmed px. Background negatives: stable-forest "
                f"tiles (<= {MAX_BG_ALERT_PX} confirmed alert px), change_time=null, static "
                f"{BG_STATIC_YEAR} 1-year window. All alerts post-2016 (RADD starts ~2018-2019)."
            ),
        },
    )

    manifest.write_registry_entry(
        SLUG, "completed", task_type="classification", num_samples=total
    )
    print(
        f"done: {total} tiles ({n_dist} disturbance + {n_bg} background); "
        f"per-region={dict(region_counts)}; per-year={dict(sorted(year_counts.items()))}",
        flush=True,
    )


if __name__ == "__main__":
    multiprocessing.set_start_method("forkserver", force=True)
    main()
