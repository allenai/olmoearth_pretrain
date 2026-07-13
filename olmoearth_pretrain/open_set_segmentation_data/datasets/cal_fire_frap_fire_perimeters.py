"""Process CAL FIRE FRAP historical California wildland-fire perimeters into burned-area
segmentation label tiles.

Source: CAL FIRE Fire and Resource Assessment Program (FRAP) "California Fire Perimeters
(all)" -- the authoritative historical wildland-fire perimeter polygon layer for
California, updated annually. Public domain. The legacy File-Geodatabase download at
frap.fire.ca.gov is now gated/redirected, so we pull the identical layer from CAL FIRE's
public hosted ArcGIS Feature Service (no credentials):

    https://services1.arcgis.com/jUJYIo9tSA7EHvfZ/arcgis/rest/services/
        California_Historic_Fire_Perimeters/FeatureServer/0   (name: "California Fire Perimeters (all)")

Each feature is one fire perimeter polygon with attributes YEAR_, ALARM_DATE (ignition
date, epoch ms), CONT_DATE, CAUSE (ignition-cause code), GIS_ACRES, FIRE_NAME, etc.

Task: **binary burned-area segmentation** (label_type: polygons):

    0 = background  (outside the fire perimeter, i.e. unburned in this fire's window)
    1 = fire        (burned area inside a FRAP fire perimeter)

Ignition CAUSE and acreage are per-fire attributes, NOT observable per-pixel from 10-30 m
S2/S1/Landsat imagery (a burn scar looks the same regardless of ignition cause), so they
are recorded as provenance metadata only, not as label classes.

Change semantics: a fire is a dated CHANGE event. We set the per-sample ``change_time``
to the fire's ALARM_DATE and make ``time_range`` a ~1-year window CENTERED on it (spec
§5), so pretraining can pair the burned-area mask with imagery spanning the fire (before
+ after the scar appears). Only fires with YEAR_ >= 2016 (Sentinel era) are used; FRAP's
pre-2016 perimeters are filtered out.

Tiling: perimeters are reprojected to a local UTM projection at 10 m/pixel. A fire whose
footprint fits in a 64x64 tile (640 m) yields one centered tile; larger fires are gridded
into non-overlapping 64x64 windows, keeping windows that actually intersect the perimeter,
and randomly sampling up to MAX_TILES_PER_FIRE of them for geographic spread. Inside the
polygon -> 1, outside -> 0 (nodata 255 unused). Selection is round-robin across fires
(every fire contributes >=1 tile before big fires add more) capped at 25,000 tiles total.

Run: python3 -m olmoearth_pretrain.open_set_segmentation_data.datasets.cal_fire_frap_fire_perimeters
Idempotent: existing locations/{id}.tif are skipped.
"""

import argparse
import json
import math
import multiprocessing
import random
import time
import urllib.request
from collections import Counter
from datetime import UTC, datetime, timedelta
from typing import Any

import numpy as np
import shapely
import shapely.geometry
import shapely.wkb
import tqdm
from rslearn.const import WGS84_PROJECTION
from rslearn.utils.geometry import Projection
from rslearn.utils.mp import star_imap_unordered

from olmoearth_pretrain.open_set_segmentation_data import io, sampling

SLUG = "cal_fire_frap_fire_perimeters"
NAME = "CAL FIRE FRAP Fire Perimeters"

SERVICE = (
    "https://services1.arcgis.com/jUJYIo9tSA7EHvfZ/arcgis/rest/services/"
    "California_Historic_Fire_Perimeters/FeatureServer/0"
)
URL = "https://frap.fire.ca.gov/data/frapgisdata-sw-fireperimeters_download"

RAW_GEOJSON = io.raw_dir(SLUG) / "perimeters_2016plus.geojson"

TILE = 64
MIN_YEAR = 2016
MAX_TILES_PER_FIRE = 40
MAX_SAMPLES = sampling.MAX_SAMPLES_PER_DATASET  # 25000
HALF_WINDOW = timedelta(
    days=180
)  # +/-180 d => 360-day (<=1 year) window centered on fire

BG, FIRE = 0, 1
CLASSES = [
    (
        "background",
        "Outside the mapped fire perimeter: land not burned by this fire during its "
        "~1-year label window. (The FRAP perimeter authoritatively delimits the fire's "
        "burned extent, so nearby out-of-perimeter pixels are genuine non-fire context; "
        "no synthetic far negatives are added.)",
    ),
    (
        "fire",
        "Burned area inside a CAL FIRE FRAP wildland-fire perimeter -- the mapped extent "
        "of a wildfire that ignited at change_time (ALARM_DATE). Perimeters are agency "
        "mapped via GPS / photointerpretation / satellite.",
    ),
]

# FRAP CAUSE domain (recorded as metadata only; not a per-pixel class).
CAUSE_CODES = {
    1: "Lightning",
    2: "Equipment Use",
    3: "Smoking",
    4: "Campfire",
    5: "Debris",
    6: "Railroad",
    7: "Arson",
    8: "Playing with fire",
    9: "Miscellaneous",
    10: "Vehicle",
    11: "Powerline",
    12: "Firefighter Training",
    13: "Non-Firefighter Training",
    14: "Unknown/Unidentified",
    15: "Structure",
    16: "Aircraft",
    17: "Volcanic",
    18: "Escaped Prescribed Burn",
    19: "Illegal Alien Campfire",
}


# --------------------------------------------------------------------------- download


def _fetch(url: str, retries: int = 6, timeout: int = 180) -> dict[str, Any]:
    """GET a JSON URL with retries. Raises after exhausting retries (transient failure)."""
    last: Exception | None = None
    for a in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.loads(r.read())
        except Exception as e:  # noqa: BLE001
            last = e
            time.sleep(min(30, 2**a))
    raise RuntimeError(f"failed to fetch {url}: {last}")


def download_perimeters() -> None:
    """Download all YEAR_ >= 2016 perimeters (EPSG:4326) to one GeoJSON, paginated."""
    io.raw_dir(SLUG).mkdir(parents=True, exist_ok=True)
    with (io.raw_dir(SLUG) / "SOURCE.txt").open("w") as f:
        f.write(
            "CAL FIRE FRAP 'California Fire Perimeters (all)' (public domain).\n"
            f"Landing page: {URL}\n"
            f"Hosted layer (no credentials): {SERVICE}\n"
            "Downloaded: all features with YEAR_ >= 2016, geometry in EPSG:4326, "
            "-> perimeters_2016plus.geojson\n"
        )
    if RAW_GEOJSON.exists():
        print(f"  [skip] {RAW_GEOJSON.name} present")
        return

    out_fields = "OBJECTID,YEAR_,ALARM_DATE,CONT_DATE,CAUSE,GIS_ACRES,FIRE_NAME,INC_NUM"
    page = 1000
    offset = 0
    features: list[dict[str, Any]] = []
    while True:
        io.check_disk()
        q = (
            f"{SERVICE}/query?where=YEAR_%3E%3D{MIN_YEAR}"
            f"&outFields={out_fields.replace(',', '%2C')}"
            f"&outSR=4326&f=geojson&returnGeometry=true"
            f"&orderByFields=OBJECTID&resultOffset={offset}&resultRecordCount={page}"
        )
        d = _fetch(q)
        feats = d.get("features", [])
        features.extend(feats)
        print(f"  fetched {len(feats)} (offset {offset}); total {len(features)}")
        if len(feats) < page:  # short page => no more records
            break
        offset += len(feats)
    fc = {"type": "FeatureCollection", "features": features}
    tmp = RAW_GEOJSON.parent / (RAW_GEOJSON.name + ".tmp")
    with tmp.open("w") as f:
        json.dump(fc, f)
    tmp.rename(RAW_GEOJSON)
    print(f"  wrote {len(features)} perimeters to {RAW_GEOJSON.name}")


# --------------------------------------------------------------------------- tiling


def _fire_candidates(fire: dict[str, Any]) -> list[dict[str, Any]]:
    """Return candidate tile records for one fire (bounds + clipped pixel geom as WKB)."""
    from shapely.geometry import box
    from shapely.prepared import prep

    from olmoearth_pretrain.open_set_segmentation_data.rasterize import geom_to_pixels

    geom = shapely.wkb.loads(fire["wkb"])
    if geom.is_empty:
        return []
    c = geom.centroid
    proj = io.utm_projection_for_lonlat(float(c.x), float(c.y))
    px = geom_to_pixels(geom, WGS84_PROJECTION, proj)
    if px.is_empty or px.area <= 0:
        return []
    minx, miny, maxx, maxy = px.bounds
    w, h = maxx - minx, maxy - miny
    crs = proj.crs.to_string()

    base = {
        "crs": crs,
        "year": fire["year"],
        "alarm_ms": fire["alarm_ms"],
        "source_id": fire["source_id"],
    }
    out: list[dict[str, Any]] = []

    if w <= TILE and h <= TILE:
        col = round((minx + maxx) / 2.0)
        row = round((miny + maxy) / 2.0)
        b = io.centered_bounds(col, row, TILE, TILE)
        clip = px.intersection(box(*b))
        if not clip.is_empty and clip.area > 0:
            out.append({**base, "bounds": b, "clip_wkb": shapely.wkb.dumps(clip)})
        return out

    # Large fire: grid the bbox into non-overlapping 64x64 windows; keep intersecting ones.
    x0, y0 = math.floor(minx), math.floor(miny)
    cells = []
    x = x0
    while x < maxx:
        y = y0
        while y < maxy:
            cells.append((x, y, x + TILE, y + TILE))
            y += TILE
        x += TILE
    rng = random.Random(fire["idx"])
    rng.shuffle(cells)
    prepared = prep(px)
    for b in cells:
        if len(out) >= MAX_TILES_PER_FIRE:
            break
        bx = box(*b)
        if not prepared.intersects(bx):
            continue
        clip = px.intersection(bx)
        if clip.is_empty or clip.area <= 0:
            continue
        out.append({**base, "bounds": b, "clip_wkb": shapely.wkb.dumps(clip)})
    return out


def _write_one(rec: dict[str, Any]) -> str | None:
    from rasterio.crs import CRS

    from olmoearth_pretrain.open_set_segmentation_data.rasterize import rasterize_shapes

    sample_id = rec["sample_id"]
    if (io.locations_dir(SLUG) / f"{sample_id}.tif").exists():
        return None

    proj = Projection(CRS.from_string(rec["crs"]), io.RESOLUTION, -io.RESOLUTION)
    bounds = tuple(rec["bounds"])
    clip = shapely.wkb.loads(rec["clip_wkb"])
    label = rasterize_shapes(
        [(clip, FIRE)], bounds, fill=BG, dtype="uint8", all_touched=False
    )[0]

    change_time = datetime.fromtimestamp(rec["alarm_ms"] / 1000.0, tz=UTC)
    time_range = (change_time - HALF_WINDOW, change_time + HALF_WINDOW)

    present = sorted(int(v) for v in np.unique(label))
    io.write_label_geotiff(SLUG, sample_id, label, proj, bounds, nodata=io.CLASS_NODATA)
    io.write_sample_json(
        SLUG,
        sample_id,
        proj,
        bounds,
        time_range,
        change_time=change_time,
        source_id=rec["source_id"],
        classes_present=present,
    )
    return "with_bg" if BG in present else "fire_only"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=64)
    args = ap.parse_args()

    io.check_disk()
    download_perimeters()
    io.check_disk()

    # ---- load perimeters
    with RAW_GEOJSON.open() as f:
        fc = json.load(f)
    fires: list[dict[str, Any]] = []
    for i, feat in enumerate(fc["features"]):
        if feat.get("geometry") is None:
            continue
        props = feat["properties"]
        year = props.get("YEAR_")
        alarm = props.get("ALARM_DATE")
        if year is None or year < MIN_YEAR or alarm is None:
            continue
        geom = shapely.geometry.shape(feat["geometry"])
        if geom.is_empty:
            continue
        name = props.get("FIRE_NAME") or "UNNAMED"
        oid = props.get("OBJECTID")
        fires.append(
            {
                "idx": i,
                "wkb": shapely.wkb.dumps(geom),
                "year": int(year),
                "alarm_ms": int(alarm),
                "source_id": f"OBJECTID={oid}:{name}:{int(year)}",
            }
        )
    print(f"{len(fires)} fires with YEAR_ >= {MIN_YEAR} and ALARM_DATE")

    # ---- Phase B: per-fire candidate tiles (parallel)
    io.check_disk()
    per_fire: list[list[dict[str, Any]]] = []
    with multiprocessing.Pool(args.workers) as p:
        for cands in tqdm.tqdm(
            star_imap_unordered(p, _fire_candidates, [dict(fire=fr) for fr in fires]),
            total=len(fires),
            desc="candidates",
        ):
            if cands:
                per_fire.append(cands)
    total_cand = sum(len(c) for c in per_fire)
    print(f"{total_cand} candidate tiles across {len(per_fire)} fires")

    # ---- Phase C: round-robin selection across fires, capped at MAX_SAMPLES
    rng = random.Random(42)
    for lst in per_fire:
        rng.shuffle(lst)
    rng.shuffle(per_fire)
    selected: list[dict[str, Any]] = []
    i = 0
    active = [lst for lst in per_fire if lst]
    while active and len(selected) < MAX_SAMPLES:
        lst = active[i % len(active)]
        selected.append(lst.pop())
        i += 1
        if i % len(active) == 0:
            active = [lst for lst in active if lst]
    for j, r in enumerate(selected):
        r["sample_id"] = f"{j:06d}"
    print(f"selected {len(selected)} tiles (cap {MAX_SAMPLES})")

    # ---- Phase D: write tiles (parallel)
    io.check_disk()
    counts: Counter = Counter()
    with multiprocessing.Pool(args.workers) as p:
        for res in tqdm.tqdm(
            star_imap_unordered(p, _write_one, [dict(rec=r) for r in selected]),
            total=len(selected),
            desc="write tiles",
        ):
            if res is not None:
                counts[res] += 1

    n_written = len(list(io.locations_dir(SLUG).glob("*.tif")))
    years = Counter(int(r["source_id"].rsplit(":", 1)[1]) for r in selected)
    io.write_dataset_metadata(
        SLUG,
        {
            "dataset": SLUG,
            "name": NAME,
            "task_type": "classification",
            "source": "CAL FIRE FRAP",
            "license": "public domain",
            "provenance": {
                "url": URL,
                "service": SERVICE,
                "have_locally": False,
                "annotation_method": "agency mapping (GPS/photointerpretation/satellite)",
                "cause_codes": CAUSE_CODES,
            },
            "sensors_relevant": ["sentinel2", "sentinel1", "landsat"],
            "classes": [
                {"id": i, "name": name, "description": desc}
                for i, (name, desc) in enumerate(CLASSES)
            ],
            "nodata_value": io.CLASS_NODATA,
            "num_samples": n_written,
            "tile_counts": {
                "tiles_with_background": counts.get("with_bg", 0),
                "fire_only_tiles": counts.get("fire_only", 0),
            },
            "samples_per_year": dict(sorted(years.items())),
            "is_change_dataset": True,
            "notes": (
                "Binary burned-area segmentation from CAL FIRE FRAP 'California Fire "
                "Perimeters (all)'. 64x64 uint8 tiles, local UTM at 10 m; class 0 "
                "background, 1 fire (255 nodata, unused). Fire is a dated CHANGE event: "
                "change_time = ALARM_DATE, time_range = +/-180 d (360-day window) "
                "centered on it. Only YEAR_ >= 2016 fires used (pre-2016 filtered out). "
                "Ignition CAUSE and GIS_ACRES are per-fire attributes not observable "
                "per-pixel, so kept as metadata only (see provenance.cause_codes), not "
                "classes. Small fires -> 1 centered tile; large fires gridded into "
                f"non-overlapping 64x64 windows, up to {MAX_TILES_PER_FIRE} intersecting "
                "windows sampled per fire. Round-robin selection across fires (every fire "
                f">=1 tile) capped at {MAX_SAMPLES}. Pulled from the public hosted ArcGIS "
                "layer because the legacy frap.fire.ca.gov File-GDB download is now "
                "gated/redirected."
            ),
        },
    )
    print("tile counts:", dict(counts))
    print("samples per year:", dict(sorted(years.items())))
    print("total tif on disk:", n_written)
    print("done")


if __name__ == "__main__":
    multiprocessing.set_start_method("forkserver", force=True)
    main()
