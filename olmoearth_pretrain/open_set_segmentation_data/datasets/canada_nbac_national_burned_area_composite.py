"""Process Canada NBAC (National Burned Area Composite) fire perimeters into
burned-area segmentation label tiles.

Source: Natural Resources Canada / Canadian Forest Service, National Burned Area
Composite (NBAC) -- the authoritative annual "best-available" fire-perimeter polygon
layer for all of Canada. For each fire NBAC selects the best available mapping among
agency perimeters, satellite hotspot delineation, and Landsat/Sentinel-2 burned-area
imagery. Open Government Licence - Canada. Distributed as per-year shapefile ZIPs at

    https://cwfis.cfs.nrcan.gc.ca/downloads/nbac/NBAC_{YEAR}_{VERSION}.zip

(no credentials). We pull only YEAR >= 2016 (Sentinel era, spec §8.2).

Each feature is one fire polygon (Polygon/MultiPolygon) in Canada Lambert Conformal
Conic (NAD83) with attributes: YEAR, NFIREID, POLY_HA / ADJ_HA (burned area), FIRECAUS
(cause), PRESCRIBED, and several dates -- HS_SDATE/HS_EDATE (satellite hotspot fire
start/end, day-resolved), AG_SDATE/AG_EDATE (agency fire start/end, day-resolved), and
CAPDATE (burned-area mapping-image capture date).

Task: **binary burned-area segmentation** (label_type: polygons):

    0 = background  (outside the fire perimeter -- unburned in this fire's window)
    1 = fire        (burned area inside an NBAC fire perimeter)

FIRECAUS, POLY_HA, PRESCRIBED are per-fire attributes not observable per-pixel from
10-30 m S2/S1/Landsat imagery (a burn scar looks the same regardless of cause), so they
are recorded as provenance metadata only, not as label classes.

Change semantics (spec §5): a fire is a dated CHANGE event. NBAC fires carry
day-resolved fire dates, so we set the per-sample ``change_time`` to the fire's start
date -- ``HS_SDATE`` (satellite hotspot start) preferred, else ``AG_SDATE`` (agency
start), else ``CAPDATE`` (same-year mapping capture) -- and make ``time_range`` a
360-day window CENTERED on it, so pretraining can pair the burned-area mask with imagery
spanning the fire (before + after the scar appears). This meets the hard
timing-precision rule (event known to <= ~1-2 months): HS/AG start dates are exact fire
dates; the small CAPDATE fallback (~1% of fires) is a same-fire-year capture of the
scar. A fire is dropped only if it has no date at all, or its only dates fall outside
[YEAR-1, YEAR+1] (71 of 15,272 fires; data-quality outliers). Only YEAR >= 2016 fires
are used; NBAC's 1972-2015 perimeters are filtered out.

Tiling (mirrors cal_fire_frap_fire_perimeters): perimeters are reprojected to a local
UTM projection at 10 m/pixel. A fire whose footprint fits in a 64x64 tile (640 m) yields
one centered tile; larger fires are gridded into non-overlapping 64x64 windows, keeping
windows that intersect the perimeter and randomly sampling up to MAX_TILES_PER_FIRE of
them for geographic spread. Inside polygon -> 1, outside -> 0 (nodata 255 unused).
Selection is round-robin across fires (every fire contributes >=1 tile before big fires
add more) capped at 25,000 tiles total.

Run: python3 -m olmoearth_pretrain.open_set_segmentation_data.datasets.canada_nbac_national_burned_area_composite
Idempotent: existing locations/{id}.tif are skipped.
"""

import argparse
import datetime as dt
import math
import multiprocessing
import random
import zipfile
from collections import Counter
from datetime import datetime, timedelta
from typing import Any

import fiona
import numpy as np
import shapely
import shapely.geometry
import shapely.wkb
import tqdm
from rasterio.crs import CRS
from rslearn.utils.geometry import Projection
from rslearn.utils.mp import star_imap_unordered

from olmoearth_pretrain.open_set_segmentation_data import download, io, sampling

SLUG = "canada_nbac_national_burned_area_composite"
NAME = "Canada NBAC (National Burned Area Composite)"

URL = "https://cwfis.cfs.nrcan.gc.ca/datamart/download/nbac"
DOWNLOAD_BASE = "https://cwfis.cfs.nrcan.gc.ca/downloads/nbac"
VERSION = "20260513"  # NBAC release version stamp on the ZIP filenames

TILE = 64
MIN_YEAR = 2016
MAX_YEAR = 2025
MAX_TILES_PER_FIRE = 40
MAX_SAMPLES = sampling.MAX_SAMPLES_PER_DATASET  # 25000
HALF_WINDOW = timedelta(
    days=180
)  # +/-180 d => 360-day (<=1 year) window centered on fire

BG, FIRE = 0, 1
CLASSES = [
    (
        "background",
        "Outside the mapped NBAC fire perimeter: land not burned by this fire during its "
        "~1-year label window. (The NBAC perimeter authoritatively delimits the fire's "
        "burned extent, so nearby out-of-perimeter pixels are genuine non-fire context; "
        "no synthetic far negatives are added.)",
    ),
    (
        "fire",
        "Burned area inside an NBAC (National Burned Area Composite) fire perimeter -- the "
        "best-available mapped extent of a wildfire (or, rarely, prescribed burn) that "
        "burned at change_time (fire start date). Perimeters combine agency mapping, "
        "satellite hotspot delineation, and Landsat/Sentinel-2 burned-area imagery.",
    ),
]


# --------------------------------------------------------------------------- download


def download_years() -> None:
    """Download + extract the per-year NBAC shapefile ZIPs for MIN_YEAR..MAX_YEAR."""
    rd = io.raw_dir(SLUG)
    rd.mkdir(parents=True, exist_ok=True)
    with (rd / "SOURCE.txt").open("w") as f:
        f.write(
            "Canada NBAC (National Burned Area Composite), Natural Resources Canada / "
            "Canadian Forest Service (Open Government Licence - Canada).\n"
            f"Landing page: {URL}\n"
            f"Per-year shapefile ZIPs: {DOWNLOAD_BASE}/NBAC_{{YEAR}}_{VERSION}.zip\n"
            f"Downloaded years {MIN_YEAR}..{MAX_YEAR} (Sentinel era); pre-2016 filtered out.\n"
        )
    for year in range(MIN_YEAR, MAX_YEAR + 1):
        stem = f"NBAC_{year}_{VERSION}"
        zpath = rd / f"{stem}.zip"
        shp = rd / stem / f"{stem}.shp"
        if not zpath.exists():
            io.check_disk()
            download.download_http(f"{DOWNLOAD_BASE}/{stem}.zip", zpath)
        if not shp.exists():
            (rd / stem).mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(str(zpath)) as zf:
                zf.extractall(str(rd / stem))
        print(f"  [ok] {stem}")


def _shp_path(year: int):
    stem = f"NBAC_{year}_{VERSION}"
    return io.raw_dir(SLUG) / stem / f"{stem}.shp"


# --------------------------------------------------------------------------- date anchor


def _pick_change_time(props: dict[str, Any], year: int) -> datetime | None:
    """Return the fire's change_time (UTC) or None to drop the fire.

    Priority HS_SDATE (satellite hotspot start) > AG_SDATE (agency start) > CAPDATE
    (mapping capture). The chosen date must parse and fall within [year-1, year+1]
    (rejects a handful of data-quality outliers whose dates are years off).
    """
    for key in ("HS_SDATE", "AG_SDATE", "CAPDATE"):
        v = props.get(key)
        if not v:
            continue
        try:
            d = dt.date.fromisoformat(str(v)[:10])
        except ValueError:
            continue
        if year - 1 <= d.year <= year + 1:
            return datetime(d.year, d.month, d.day, tzinfo=dt.UTC)
    return None


# --------------------------------------------------------------------------- tiling


def _fire_candidates(fire: dict[str, Any]) -> list[dict[str, Any]]:
    """Return candidate tile records for one fire (bounds + clipped pixel geom as WKB)."""
    from rslearn.const import WGS84_PROJECTION
    from shapely.geometry import box
    from shapely.prepared import prep

    from olmoearth_pretrain.open_set_segmentation_data.rasterize import geom_to_pixels

    geom = shapely.wkb.loads(fire["wkb"])
    if geom.is_empty:
        return []
    # Geometry is in the source CRS (Canada LCC, metres); a Projection with resolution
    # (1, 1) means the geometry coordinates are interpreted directly as CRS units
    # (easting, northing) with no sign flip before reprojection.
    src_proj = Projection(CRS.from_wkt(fire["src_crs"]), 1, 1)
    # Centroid -> lon/lat to choose the local UTM zone.
    ll = geom_to_pixels(geom.centroid, src_proj, WGS84_PROJECTION)
    proj = io.utm_projection_for_lonlat(float(ll.x), float(ll.y))
    px = geom_to_pixels(geom, src_proj, proj)
    if px.is_empty or px.area <= 0:
        return []
    if not px.is_valid:
        px = shapely.make_valid(px)
    minx, miny, maxx, maxy = px.bounds
    w, h = maxx - minx, maxy - miny
    crs = proj.crs.to_string()

    base = {
        "crs": crs,
        "year": fire["year"],
        "change_ts": fire["change_ts"],
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

    change_time = datetime.fromtimestamp(rec["change_ts"], tz=dt.UTC)
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
    download_years()
    io.check_disk()

    # ---- load perimeters across all years
    fires: list[dict[str, Any]] = []
    n_total = n_nodate = 0
    for year in range(MIN_YEAR, MAX_YEAR + 1):
        with fiona.open(str(_shp_path(year))) as c:
            src_crs = c.crs_wkt
            for feat in c:
                n_total += 1
                if feat["geometry"] is None:
                    continue
                props = dict(feat["properties"])
                ct = _pick_change_time(props, year)
                if ct is None:
                    n_nodate += 1
                    continue
                geom = shapely.geometry.shape(feat["geometry"])
                if geom.is_empty:
                    continue
                gid = props.get("GID") or f"{year}_{props.get('NFIREID')}"
                fires.append(
                    {
                        "idx": len(fires),
                        "wkb": shapely.wkb.dumps(geom),
                        "src_crs": src_crs,
                        "year": year,
                        "change_ts": ct.timestamp(),
                        "source_id": f"GID={gid}:{year}",
                    }
                )
    print(
        f"{len(fires)} fires kept ({MIN_YEAR}..{MAX_YEAR}); {n_nodate} dropped "
        f"for no usable date; {n_total} scanned"
    )

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
            "source": "Natural Resources Canada (Canadian Forest Service)",
            "license": "OGL-Canada",
            "provenance": {
                "url": URL,
                "download_base": DOWNLOAD_BASE,
                "version": VERSION,
                "have_locally": False,
                "annotation_method": (
                    "best-available composite: agency mapping + satellite hotspot "
                    "delineation + Landsat/Sentinel-2 burned-area imagery"
                ),
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
                "Binary burned-area segmentation from Canada NBAC (National Burned Area "
                "Composite). 64x64 uint8 tiles, local UTM at 10 m; class 0 background, "
                "1 fire (255 nodata, unused). Fire is a dated CHANGE event: change_time "
                "= fire start date (HS_SDATE preferred, else AG_SDATE, else same-year "
                "CAPDATE), time_range = +/-180 d (360-day window) centered on it. Only "
                "YEAR >= 2016 fires used (pre-2016 filtered out); fires with no date in "
                "[year-1, year+1] dropped. FIRECAUS / POLY_HA / PRESCRIBED are per-fire "
                "attributes not observable per-pixel, so kept out of the class scheme. "
                "Small fires -> 1 centered tile; large fires gridded into non-overlapping "
                f"64x64 windows, up to {MAX_TILES_PER_FIRE} intersecting windows sampled "
                "per fire. Round-robin selection across fires (every fire >=1 tile) "
                f"capped at {MAX_SAMPLES}. Source CRS Canada Lambert Conformal Conic "
                "(NAD83), reprojected per-fire to local UTM."
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
