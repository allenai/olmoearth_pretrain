"""Process FORWIND (European forest wind disturbance) polygons into windthrow
segmentation label tiles.

Source: FORWIND -- "A spatially-explicit database of wind disturbances in European
forests over the period 2000-2018" (Forzieri et al., ESSD 2020). figshare record
https://doi.org/10.6084/m9.figshare.9555008 (v2, CC-BY-4.0). One zipped shapefile
(FORWIND_v2.shp/.shx/.dbf/.prj, geometry in WGS84 / EPSG:4326) of >89,000 polygons,
each a spatially-delineated forest area disturbed by wind/tornado/windstorm, with
attributes: Id_poly, EventDate, StormName, EventType, Country, Area [ha], Perimeter [m],
Damage_deg [fraction 0-1], Methods, Dataprovid, Source. Missing values = -999.

Task: **windthrow presence segmentation** (label_type: polygons), positive-only:

    0 = wind_disturbance   (inside a FORWIND wind-damaged forest polygon)
    255 = nodata/ignore    (everything outside the polygon)

We use a single foreground class rather than encoding Damage_deg as per-pixel classes
because Damage_deg is a continuous, stand-level (per-polygon) attribute that is (a) present
for only ~57% of the post-2016 polygons and (b) constant within each polygon, so it does
NOT create meaningful within-tile class structure. It is kept as per-sample provenance
(source_id) instead -- analogous to how cal_fire keeps per-fire CAUSE as metadata.
Background is nodata (not class 0): FORWIND is a compilation of mapped disturbances, not an
exhaustive damage/no-damage map of every forest, so out-of-polygon pixels are "unmapped",
not authoritatively undamaged; the assembly step supplies negatives from other datasets
(spec §5 positive-only handling).

Change semantics: each polygon is dated to a named windstorm event. Post-2016 FORWIND
records are all resolved to the exact day (Vaia 2018-10-28/29, Friederike 2018-01-18,
Xavier 2017-11-10, plus a few day-dated 2017 events), well within the spec's ~1-2 month
change-timing requirement. So we set ``change_time`` = EventDate, which splits the sample
into two adjacent six-month windows (via ``io.pre_post_time_ranges``): ``pre_time_range`` =
the ~6 months (<=183 days) immediately before the storm and ``post_time_range`` = the
~6 months (<=183 days) immediately after, with ``time_range`` = null (spec §5). Pretraining
pairs the "before" image stack with the "after" stack and probes on their difference (forest
before vs blowdown after).

Only records with EventDate year >= 2016 (Sentinel era) are used; FORWIND's 2000-2015
polygons are filtered out (spec §8).

Tiling: each polygon is reprojected to a local UTM projection at 10 m/pixel. A polygon
whose footprint fits in <=64x64 pixels yields one tile tightly framed on its bounding box;
larger polygons are gridded into non-overlapping 64x64 windows, keeping windows that
intersect the polygon and sampling up to MAX_TILES_PER_POLY of them. Inside polygon -> 0,
outside -> 255. Round-robin selection across polygons (every polygon contributes >=1 tile)
capped at 25,000 tiles total.

Run: python3 -m olmoearth_pretrain.open_set_segmentation_data.datasets.forwind_european_forest_wind_disturbance
Idempotent: existing locations/{id}.tif are skipped.
"""

import argparse
import math
import multiprocessing
import random
from collections import Counter
from datetime import UTC, datetime
from typing import Any

import fiona
import numpy as np
import shapely
import shapely.geometry
import shapely.wkb
import tqdm
from rslearn.const import WGS84_PROJECTION
from rslearn.utils.geometry import Projection
from rslearn.utils.mp import star_imap_unordered

from olmoearth_pretrain.open_set_segmentation_data import (
    download,
    io,
    sampling,
)

SLUG = "forwind_european_forest_wind_disturbance"
NAME = "FORWIND (European forest wind disturbance)"

URL = "https://doi.org/10.6084/m9.figshare.9555008"
FIGSHARE_FILES = {
    "FORWIND_v2.shp": "https://ndownloader.figshare.com/files/20325300",
    "FORWIND_v2.shx": "https://ndownloader.figshare.com/files/20325282",
    "FORWIND_v2.dbf": "https://ndownloader.figshare.com/files/20325288",
    "FORWIND_v2.prj": "https://ndownloader.figshare.com/files/20325291",
    "readme.txt": "https://ndownloader.figshare.com/files/20325762",
}
# Some figshare CDN nodes 403 the default urllib UA; a Firefox UA is accepted.
UA = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64; rv:120.0) Gecko/20100101 Firefox/120.0"
}

SHP = io.raw_dir(SLUG) / "FORWIND_v2.shp"

TILE = 64
MIN_YEAR = 2016
MAX_TILES_PER_POLY = 40
MAX_SAMPLES = sampling.MAX_SAMPLES_PER_DATASET  # 25000

WIND, NODATA = 0, io.CLASS_NODATA
CLASSES = [
    (
        "wind_disturbance",
        "Forest stand disturbed (blown down / snapped / uprooted) by a wind event -- a "
        "windstorm or tornado -- as delineated in the FORWIND database from "
        "aerial/satellite photointerpretation and field survey. The polygon marks the "
        "extent of the wind-damaged forest; the blowdown gap is a persistent post-event "
        "state observable in S2/S1/Landsat. change_time is the storm's EventDate. "
        "Per-polygon damage degree (fraction of stand affected) is recorded in source_id, "
        "not as a separate class (it is a stand-level, often-missing, per-polygon value).",
    ),
]


# --------------------------------------------------------------------------- download


def download_source() -> None:
    io.raw_dir(SLUG).mkdir(parents=True, exist_ok=True)
    with (io.raw_dir(SLUG) / "SOURCE.txt").open("w") as f:
        f.write(
            "FORWIND: A spatially-explicit database of wind disturbances in European "
            "forests over 2000-2018 (Forzieri et al., ESSD 2020). CC-BY-4.0.\n"
            f"Landing page (DOI): {URL}\n"
            "figshare files (no credentials): FORWIND_v2.shp/.shx/.dbf/.prj + readme.txt\n"
            "Geometry: WGS84 / EPSG:4326 polygons. Attributes include EventDate, "
            "StormName, EventType, Country, Area[ha], Damage_deg[0-1] (missing=-999).\n"
        )
    for name, url in FIGSHARE_FILES.items():
        io.check_disk()
        p = download.download_http(url, io.raw_dir(SLUG) / name, headers=UA)
        print(f"  have {name} ({p.stat().st_size} bytes)")


# --------------------------------------------------------------------------- dates


def _parse_year(ed: str | None) -> int | None:
    if not ed:
        return None
    s = ed.strip()
    if len(s) >= 4 and s[:4].isdigit():
        return int(s[:4])
    return None


def _parse_event_date(ed: str) -> datetime:
    """Parse an EventDate string ('YYYY-MM-DD' or 'YYYY/MM/DD') to a UTC datetime."""
    s = ed.strip().replace("/", "-")
    return datetime.strptime(s[:10], "%Y-%m-%d").replace(tzinfo=UTC)


# --------------------------------------------------------------------------- tiling


def _poly_candidates(poly: dict[str, Any]) -> list[dict[str, Any]]:
    """Return candidate tile records for one polygon (bounds + clipped pixel geom as WKB)."""
    from shapely.geometry import box
    from shapely.prepared import prep

    from olmoearth_pretrain.open_set_segmentation_data.rasterize import geom_to_pixels

    geom = shapely.wkb.loads(poly["wkb"])
    if geom.is_empty:
        return []
    if not geom.is_valid:
        geom = geom.buffer(0)
        if geom.is_empty or not geom.is_valid:
            return []
    c = geom.centroid
    proj = io.utm_projection_for_lonlat(float(c.x), float(c.y))
    px = geom_to_pixels(geom, WGS84_PROJECTION, proj)
    if px.is_empty or px.area <= 0:
        return []
    minx, miny, maxx, maxy = px.bounds
    fx0, fy0 = math.floor(minx), math.floor(miny)
    w = int(math.ceil(maxx)) - fx0
    h = int(math.ceil(maxy)) - fy0
    crs = proj.crs.to_string()

    base = {
        "crs": crs,
        "change_ts": poly["change_ts"],
        "source_id": poly["source_id"],
    }
    out: list[dict[str, Any]] = []

    if w <= TILE and h <= TILE:
        # Tight bounding-box tile framing the whole polygon (>=1 px each side).
        w = max(w, 1)
        h = max(h, 1)
        b = (fx0, fy0, fx0 + w, fy0 + h)
        clip = px.intersection(box(*b))
        if not clip.is_empty and clip.area > 0:
            out.append({**base, "bounds": b, "clip_wkb": shapely.wkb.dumps(clip)})
        return out

    # Large polygon: grid the bbox into non-overlapping 64x64 windows; keep intersecting.
    cells = []
    x = fx0
    while x < maxx:
        y = fy0
        while y < maxy:
            cells.append((x, y, x + TILE, y + TILE))
            y += TILE
        x += TILE
    rng = random.Random(poly["idx"])
    rng.shuffle(cells)
    prepared = prep(px)
    for b in cells:
        if len(out) >= MAX_TILES_PER_POLY:
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
        [(clip, WIND)], bounds, fill=NODATA, dtype="uint8", all_touched=True
    )[0]

    change_time = datetime.fromtimestamp(rec["change_ts"], tz=UTC)
    pre_range, post_range = io.pre_post_time_ranges(change_time)
    time_range = (pre_range[0], post_range[1])  # outer bounding span

    present = sorted(int(v) for v in np.unique(label) if int(v) != NODATA)
    io.write_label_geotiff(SLUG, sample_id, label, proj, bounds, nodata=NODATA)
    io.write_sample_json(
        SLUG,
        sample_id,
        proj,
        bounds,
        time_range,
        change_time=change_time,
        source_id=rec["source_id"],
        classes_present=present,
        pre_time_range=pre_range,
        post_time_range=post_range,
    )
    return "ok" if present else "empty"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=64)
    args = ap.parse_args()

    io.check_disk()
    download_source()
    io.check_disk()

    # ---- load polygons (filter to Sentinel era)
    polys: list[dict[str, Any]] = []
    storm_counts: Counter = Counter()
    with fiona.open(str(SHP)) as src:
        for i, feat in enumerate(src):
            if feat["geometry"] is None:
                continue
            p = feat["properties"]
            year = _parse_year(p.get("EventDate"))
            if year is None or year < MIN_YEAR:
                continue
            try:
                change_time = _parse_event_date(p["EventDate"])
            except Exception:  # noqa: BLE001
                continue
            geom = shapely.geometry.shape(feat["geometry"])
            if geom.is_empty:
                continue
            storm = p.get("StormName") or "unknown"
            dmg = p.get("Damage_deg")
            dmg_s = f"{dmg:.4f}" if (dmg is not None and dmg != -999.0) else "NA"
            oid = p.get("Id_poly")
            country = p.get("Country") or "NA"
            storm_counts[storm] += 1
            polys.append(
                {
                    "idx": i,
                    "wkb": shapely.wkb.dumps(geom),
                    "change_ts": change_time.timestamp(),
                    "source_id": (
                        f"Id_poly={oid}:{storm}:{p['EventDate'].strip()}:"
                        f"{country}:dmg={dmg_s}"
                    ),
                }
            )
    print(f"{len(polys)} polygons with EventDate year >= {MIN_YEAR}")
    print("storms:", dict(storm_counts))

    # ---- Phase B: per-polygon candidate tiles (parallel)
    io.check_disk()
    per_poly: list[list[dict[str, Any]]] = []
    with multiprocessing.Pool(args.workers) as pool:
        for cands in tqdm.tqdm(
            star_imap_unordered(
                pool, _poly_candidates, [dict(poly=pl) for pl in polys]
            ),
            total=len(polys),
            desc="candidates",
        ):
            if cands:
                per_poly.append(cands)
    total_cand = sum(len(c) for c in per_poly)
    print(f"{total_cand} candidate tiles across {len(per_poly)} polygons")

    # ---- Phase C: round-robin selection across polygons, capped at MAX_SAMPLES
    rng = random.Random(42)
    for lst in per_poly:
        rng.shuffle(lst)
    rng.shuffle(per_poly)
    selected: list[dict[str, Any]] = []
    i = 0
    active = [lst for lst in per_poly if lst]
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
    with multiprocessing.Pool(args.workers) as pool:
        for res in tqdm.tqdm(
            star_imap_unordered(pool, _write_one, [dict(rec=r) for r in selected]),
            total=len(selected),
            desc="write tiles",
        ):
            if res is not None:
                counts[res] += 1

    n_written = len(list(io.locations_dir(SLUG).glob("*.tif")))
    years = Counter(int(r["source_id"].split(":")[2][:4]) for r in selected)
    io.write_dataset_metadata(
        SLUG,
        {
            "dataset": SLUG,
            "name": NAME,
            "task_type": "classification",
            "source": "figshare / ESSD (Forzieri et al. 2020)",
            "license": "CC-BY-4.0",
            "provenance": {
                "url": URL,
                "have_locally": False,
                "annotation_method": "aerial/satellite photointerpretation + field survey",
                "attributes": (
                    "Id_poly, EventDate, StormName, EventType, Country, Area[ha], "
                    "Perimeter[m], Damage_deg[0-1], Methods, Dataprovid, Source "
                    "(missing=-999)"
                ),
            },
            "sensors_relevant": ["sentinel2", "sentinel1", "landsat"],
            "classes": [
                {"id": i, "name": name, "description": desc}
                for i, (name, desc) in enumerate(CLASSES)
            ],
            "nodata_value": NODATA,
            "num_samples": n_written,
            "samples_per_year": dict(sorted(years.items())),
            "storm_counts_polygons": dict(storm_counts),
            "is_change_dataset": True,
            "notes": (
                "Windthrow presence segmentation from FORWIND v2 polygons. <=64x64 uint8 "
                "tiles, local UTM at 10 m; single foreground class 0 = wind_disturbance, "
                "255 = nodata (positive-only: background is unmapped, supplied by assembly). "
                "Each polygon is a dated windstorm event: change_time = EventDate, "
                "time_range = +/-180 d (360-day window) centered on it. All post-2016 "
                "EventDates are day-precision (Vaia 2018-10, Friederike 2018-01, Xavier "
                "2017-11, plus day-dated 2017 events), meeting the ~1-2 month change-timing "
                "requirement. Only EventDate year >= 2016 used (2000-2015 filtered out). "
                "Damage_deg is a continuous, per-polygon, ~57%-missing stand-level value, "
                "so it is kept in source_id as provenance rather than as per-pixel classes. "
                "Small polygons -> 1 tile tightly framed on the bbox; large polygons gridded "
                f"into non-overlapping 64x64 windows, up to {MAX_TILES_PER_POLY} intersecting "
                "windows per polygon. Round-robin selection across polygons (every polygon "
                f">=1 tile) capped at {MAX_SAMPLES}."
            ),
        },
    )
    print("write results:", dict(counts))
    print("samples per year:", dict(sorted(years.items())))
    print("total tif on disk:", n_written)
    print("done")


if __name__ == "__main__":
    multiprocessing.set_start_method("forkserver", force=True)
    main()
