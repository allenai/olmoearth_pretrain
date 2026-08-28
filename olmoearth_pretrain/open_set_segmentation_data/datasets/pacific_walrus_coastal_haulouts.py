"""Process USGS Pacific Walrus coastal-haulout herd outlines into walrus-haulout
presence segmentation label tiles.

Source: USGS Alaska Science Center data release "Pacific Walrus Coastal Haulout
Occurrences Interpreted from Satellite Imagery" (Fischbach & Douglas 2022, ver 6.0
December 2025), doi:10.5066/P9CSM0KN, distributed openly on ScienceBase
(item 6441f010d34ee8d4ade7edcc). License: CC0 1.0 (public domain).

Trained interpreters delineated the extent of walrus herds resting on shore ("coastal
haulouts") apparent in individual satellite images at eight known Chukchi Sea haulout
sites (Alaska + Chukotka, Russia), autumn 2017-2025. Imagery sources span Sentinel-2,
Sentinel-1, PlanetScope, Maxar, TerraSAR-X, RADARSAT-2, Umbra, Capella and Iceye. The
release organises data into per-site, per-year ZIP packages, each containing a
``walrus_dailySatelliteHauloutOutlines/shape`` folder of Esri shapefiles -- one shapefile
per satellite image in which a walrus herd was apparent, each holding one or more herd
sub-group polygons, geocoded in the site's local UTM CRS. Shapefile filenames encode the
image acquisition time + mission: ``[YYYYMMDD]T[hhmm(ss)]Z_[mission]`` (e.g.
``20221008T234631Z_S2``). A top-level CSV (walrus_hauloutAreaEstimates_chukchi.csv) lists
every image examined and the summed herd area; it is downloaded for provenance only.

Task: **single-class walrus-haulout / herd-extent segmentation** (label_type: polygons).
Positive-only foreground dataset (herds were outlined where walruses were apparent; the
absence of an outline is not a verified "no walrus"), so per spec section 5 we do NOT
fabricate negatives:

    0   = walrus haulout / herd extent  (inside a mapped herd outline)
    255 = nodata/ignore                 (everything else)

The assembly step supplies negatives from other datasets.

Time range (spec section 5, specific-image labels): each herd outline was interpreted
from ONE dated satellite image, and walruses are mobile -- the extent is only valid at
that acquisition instant (a year-long window would pair imagery showing no walruses).
So each sample uses a ~1-hour window CENTERED on the image acquisition datetime parsed
from the shapefile filename (analogous to the Sentinel-2 vessels dataset's per-image
window). change_time is left null (this is dated presence, not a change event).

Tiling (mirrors spot6_avalanche_outlines_swiss_alps): the herd polygons of one image are
unioned and reprojected to a local UTM projection at 10 m/pixel. A herd whose footprint
fits in a 64x64 tile (640 m) yields one centered tile; larger/elongated haulouts (many
are long thin beach strips, up to ~2.5 km) are gridded into non-overlapping 64x64 windows
and up to MAX_TILES_PER_IMAGE intersecting windows are kept. Inside outline -> 0, else
255. Selection is round-robin across images (every image contributes >=1 tile before
large ones add more), capped at 25,000 tiles (never reached; ~1-2k tiles).

Caveats:
- USGS notes image georeferencing across missions can be shifted by tens to >100 m, so
  outlines may sit slightly off the true coastline. Not corrected here (recorded in the
  summary).
- 9 per-site/year ZIPs return HTTP 404 and CapeSerdtseKamen_2025.zip is 0 bytes on the
  server (source-side, transient). Of these only Vankarem_2023 (4 herd images) and
  CapeSerdtseKamen_2025 (~12 herd images) carry labels; the other 7 sites/years had no
  walruses (empty outline folders). Re-run once the source restores those files to add
  the ~16 missing herd images.
- 7 shapefiles are named "ESRI Shapefile.shp" (a provider export artifact) and carry no
  parseable acquisition timestamp; they are skipped to preserve per-image time integrity.

Run: python3 -m olmoearth_pretrain.open_set_segmentation_data.datasets.pacific_walrus_coastal_haulouts
Idempotent: existing locations/{id}.tif are skipped; downloaded ZIPs/shapefiles reused.
"""

import argparse
import json
import math
import multiprocessing
import os
import random
import re
import urllib.request
from collections import Counter
from datetime import UTC, datetime, timedelta
from typing import Any

import numpy as np
import shapely
import shapely.geometry
import shapely.ops
import shapely.wkb
import tqdm
from pyproj import Transformer
from rslearn.const import WGS84_PROJECTION
from rslearn.utils.geometry import Projection
from rslearn.utils.mp import star_imap_unordered

from olmoearth_pretrain.open_set_segmentation_data import io, sampling

SLUG = "pacific_walrus_coastal_haulouts"
NAME = "Pacific Walrus Coastal Haulouts"

URL = "https://doi.org/10.5066/P9CSM0KN"
SB_ITEM = "6441f010d34ee8d4ade7edcc"  # ScienceBase parent item
SB_ITEM_API = "https://www.sciencebase.gov/catalog/item/{}?format=json"
SB_CHILDREN_API = (
    "https://www.sciencebase.gov/catalog/items?parentId={}&format=json&max=100"
    "&fields=title,files"
)

TILE = 64
MAX_TILES_PER_IMAGE = 20
MAX_SAMPLES = sampling.MAX_SAMPLES_PER_DATASET  # 25000
HALF_WINDOW = timedelta(
    minutes=30
)  # +/-30 min => ~1-hour specific-image window (spec 5)

HAULOUT, NODATA = 0, io.CLASS_NODATA
CLASSES = [
    (
        "walrus haulout / herd extent",
        "Interior of a herd-extent polygon: the land area occupied by a Pacific walrus "
        "(Odobenus rosmarus divergens) herd resting on shore at a coastal haulout, as "
        "delineated by trained USGS interpreters from a single satellite image "
        "(Sentinel-2/-1, PlanetScope, Maxar, TerraSAR-X, RADARSAT-2, Umbra, Capella or "
        "Iceye). Aggregations range from ~0.08 to ~19 ha. Positive-only: everything "
        "outside a mapped outline is nodata (255), not a verified negative.",
    ),
]

# Mission code (filename suffix) -> human-readable sensor, for provenance.
MISSION_CODES = {
    "S1": "Sentinel-1 C-band SAR (VH), 10 m",
    "S2": "Sentinel-2 optical, 10 m",
    "PS": "PlanetScope optical, ~3 m",
    "US": "Umbra X-band SAR, <=0.5 m",
    "TS": "TerraSAR-X X-band SAR, ~1 m",
    "RS": "RADARSAT-2 C-band SAR, ~1 m",
    "CS": "Capella X-band SAR, <=0.5 m",
    "DG": "Maxar / DigitalGlobe optical, sub-metre",
    "IE": "Iceye X-band SAR, <=0.5 m",
}

_RAW = io.raw_dir(SLUG)
_ZIP_DIR = _RAW / "zips"
_OUT_DIR = _RAW / "outlines"


# --------------------------------------------------------------------------- download


def _sb_get(url: str, retries: int = 5) -> bytes:
    import time as _time

    last = ""
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=300) as r:
                return r.read()
        except Exception as e:  # noqa: BLE001 - retry transient network/HTTP errors
            last = repr(e)
            _time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"GET failed after {retries}: {url}: {last}")


def download() -> dict[str, Any]:
    """Download the ScienceBase package (per-site/year ZIPs + CSV + metadata) and extract
    the outline shapefiles. Returns a small dict of source-availability stats.

    Reproducible: enumerates the parent item's child items via the ScienceBase JSON API
    at runtime (no hard-coded disk-hash URLs). Idempotent: skips ZIPs already on disk and
    shapefiles already extracted. Only the thin vector outlines are needed; the JPG maps
    inside each ZIP are extracted-past (kept inside the raw ZIPs for provenance).
    """
    import zipfile

    _RAW.mkdir(parents=True, exist_ok=True)
    _ZIP_DIR.mkdir(parents=True, exist_ok=True)
    _OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Top-level metadata + CSV (provenance).
    item = json.loads(_sb_get(SB_ITEM_API.format(SB_ITEM)))
    for f in item.get("files", []):
        nm = f.get("name", "")
        if nm.endswith((".csv", ".xml", ".html", ".txt")):
            dst = _RAW / nm
            if not dst.exists():
                data = _sb_get(f["url"])
                tmp = _RAW / (nm + ".tmp")
                with tmp.open("wb") as out:
                    out.write(data)
                tmp.rename(dst)

    with (_RAW / "SOURCE.txt").open("w") as f:
        f.write(
            "Pacific Walrus Coastal Haulout Occurrences Interpreted from Satellite "
            "Imagery (Fischbach & Douglas 2022, ver 6.0 Dec 2025).\n"
            f"USGS Alaska Science Center data release, doi:10.5066/P9CSM0KN.\n"
            f"ScienceBase item: {SB_ITEM}\nLanding page: {URL}\n"
            "License: CC0 1.0 (public domain).\n"
            "Per-site/year ZIPs each hold walrus_dailySatelliteHauloutOutlines/shape/*.shp "
            "(herd outline polygons, local UTM) + JPG maps + methods PDF.\n"
        )

    # Enumerate child items (haulout sites), download each per-year ZIP, extract shapes.
    children = json.loads(_sb_get(SB_CHILDREN_API.format(SB_ITEM)))
    stats = {"zips_ok": 0, "zips_missing": [], "zips_empty": []}
    for child in children.get("items", []):
        cid = child["id"]
        ci = json.loads(_sb_get(SB_ITEM_API.format(cid)))
        for f in ci.get("files", []):
            nm = f.get("name", "")
            if not nm.endswith(".zip"):
                continue
            base = nm[:-4]  # site_year
            size = f.get("size", 0)
            if size == 0:
                stats["zips_empty"].append(nm)
                continue
            dst = _ZIP_DIR / nm
            if not (dst.exists() and dst.stat().st_size == size):
                io.check_disk()
                try:
                    data = _sb_get(f["url"], retries=4)
                except RuntimeError:
                    stats["zips_missing"].append(nm)
                    continue
                if len(data) != size:
                    stats["zips_missing"].append(nm)
                    continue
                tmp = _ZIP_DIR / (nm + ".tmp")
                with tmp.open("wb") as out:
                    out.write(data)
                tmp.rename(dst)
            stats["zips_ok"] += 1
            # Extract only the outline shapefile members.
            out_sub = _OUT_DIR / base
            try:
                with zipfile.ZipFile(str(dst)) as z:
                    members = [
                        m
                        for m in z.namelist()
                        if "/shape/" in m and not m.endswith("/")
                    ]
                    for m in members:
                        fn = os.path.basename(m)
                        tgt = out_sub / fn
                        if tgt.exists():
                            continue
                        out_sub.mkdir(parents=True, exist_ok=True)
                        with (
                            z.open(m) as src,
                            (out_sub / (fn + ".tmp")).open("wb") as o,
                        ):
                            o.write(src.read())
                        (out_sub / (fn + ".tmp")).rename(tgt)
            except zipfile.BadZipFile:
                stats["zips_missing"].append(nm)

    with (_RAW / "download_stats.json").open("w") as f:
        json.dump(stats, f, indent=2)
    print("download stats:", stats)
    return stats


# --------------------------------------------------------------------------- parsing


def _parse_time(fname: str) -> datetime | None:
    """Acquisition datetime (UTC) from a shapefile stem like 20221008T234631Z_S2.

    Handles second-precision (hhmmss) and minute-precision (hhmm) filenames. Returns
    None for names without a parseable timestamp (e.g. the "ESRI Shapefile" artifacts).
    """
    m = re.match(r"(\d{8})T(\d{4}|\d{6})Z", fname)
    if not m:
        return None
    date, tod = m.group(1), m.group(2)
    fmt = "%Y%m%d%H%M%S" if len(tod) == 6 else "%Y%m%d%H%M"
    try:
        return datetime.strptime(date + tod, fmt).replace(tzinfo=UTC)
    except ValueError:
        return None


def _mission(fname: str) -> str:
    m = re.match(r"\d{8}T(?:\d{4}|\d{6})Z_([A-Za-z0-9]+)", fname)
    return m.group(1) if m else "NA"


def _load_images() -> list[dict[str, Any]]:
    """Load every outline shapefile as one image record (herd polygons -> WGS84 WKB)."""
    import glob as _glob

    import fiona

    records: list[dict[str, Any]] = []
    n_skip_notime = 0
    n_skip_empty = 0
    shps = sorted(_glob.glob(str(_OUT_DIR / "*" / "*.shp")))
    for shp in shps:
        stem = os.path.basename(shp)[:-4]
        t = _parse_time(stem)
        if t is None:
            n_skip_notime += 1
            continue
        site = os.path.basename(os.path.dirname(shp))
        with fiona.open(shp) as src:
            epsg = src.crs.to_epsg()
            geoms = [
                shapely.geometry.shape(f["geometry"])
                for f in src
                if f["geometry"] is not None
            ]
        geoms = [g for g in geoms if not g.is_empty]
        if not geoms:
            n_skip_empty += 1
            continue
        # Reproject to WGS84 lon/lat (native for the shared UTM helpers).
        if epsg and epsg != 4326:
            tr = Transformer.from_crs(f"EPSG:{epsg}", "EPSG:4326", always_xy=True)
            geoms = [
                shapely.ops.transform(lambda xs, ys: tr.transform(xs, ys), g)
                for g in geoms
            ]
        union = shapely.ops.unary_union(geoms)
        if union.is_empty:
            n_skip_empty += 1
            continue
        records.append(
            {
                "wkb": shapely.wkb.dumps(union),
                "time": t.isoformat(),
                "source_id": f"{site}/{stem}:mission={_mission(stem)}",
            }
        )
    print(
        f"{len(records)} image records loaded "
        f"(skipped {n_skip_notime} no-timestamp, {n_skip_empty} empty)"
    )
    return records


# --------------------------------------------------------------------------- tiling


def _image_candidates(rec: dict[str, Any]) -> list[dict[str, Any]]:
    """Candidate 64x64 tiles for one image's unioned herd geometry (WGS84 WKB)."""
    from shapely.geometry import box
    from shapely.prepared import prep

    from olmoearth_pretrain.open_set_segmentation_data.rasterize import geom_to_pixels

    geom = shapely.wkb.loads(rec["wkb"])
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
    base = {"crs": crs, "source_id": rec["source_id"], "time": rec["time"]}
    out: list[dict[str, Any]] = []

    if w <= TILE and h <= TILE:
        col = round((minx + maxx) / 2.0)
        row = round((miny + maxy) / 2.0)
        b = io.centered_bounds(col, row, TILE, TILE)
        clip = px.intersection(box(*b))
        if not clip.is_empty and clip.area > 0:
            out.append({**base, "bounds": b, "clip_wkb": shapely.wkb.dumps(clip)})
        return out

    # Large/elongated haulout: grid the bbox into non-overlapping 64x64 windows.
    x0, y0 = math.floor(minx), math.floor(miny)
    cells = []
    x = x0
    while x < maxx:
        y = y0
        while y < maxy:
            cells.append((x, y, x + TILE, y + TILE))
            y += TILE
        x += TILE
    rng = random.Random(hash(rec["source_id"]) & 0xFFFFFFFF)
    rng.shuffle(cells)
    prepared = prep(px)
    for b in cells:
        if len(out) >= MAX_TILES_PER_IMAGE:
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
    # Positive-only: herd interior -> 0, everything else -> 255 (nodata). all_touched so
    # small/thin herd polygons stay visible at 10 m.
    label = rasterize_shapes(
        [(clip, HAULOUT)], bounds, fill=NODATA, dtype="uint8", all_touched=True
    )[0]
    present = sorted(int(v) for v in np.unique(label) if int(v) != NODATA)
    if not present:
        return "empty"

    acq = datetime.fromisoformat(rec["time"])
    time_range = (acq - HALF_WINDOW, acq + HALF_WINDOW)
    io.write_label_geotiff(SLUG, sample_id, label, proj, bounds, nodata=io.CLASS_NODATA)
    io.write_sample_json(
        SLUG,
        sample_id,
        proj,
        bounds,
        time_range,
        change_time=None,
        source_id=rec["source_id"],
        classes_present=present,
    )
    return "ok"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=64)
    args = ap.parse_args()

    io.check_disk()
    stats = download()
    io.check_disk()

    images = _load_images()
    if not images:
        raise RuntimeError("no image outline records found")

    # ---- Phase B: per-image candidate tiles (parallel)
    per_image: list[list[dict[str, Any]]] = []
    with multiprocessing.Pool(args.workers) as p:
        for cands in tqdm.tqdm(
            star_imap_unordered(p, _image_candidates, [dict(rec=r) for r in images]),
            total=len(images),
            desc="candidates",
        ):
            if cands:
                per_image.append(cands)
    total_cand = sum(len(c) for c in per_image)
    print(f"{total_cand} candidate tiles across {len(per_image)} images")

    # ---- Phase C: round-robin selection across images, capped at MAX_SAMPLES
    rng = random.Random(42)
    for lst in per_image:
        rng.shuffle(lst)
    rng.shuffle(per_image)
    selected: list[dict[str, Any]] = []
    i = 0
    active = [lst for lst in per_image if lst]
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
    io.write_dataset_metadata(
        SLUG,
        {
            "dataset": SLUG,
            "name": NAME,
            "task_type": "classification",
            "source": "USGS Alaska Science Center (ScienceBase)",
            "license": "CC0 1.0 (public domain)",
            "provenance": {
                "url": URL,
                "sciencebase_item": SB_ITEM,
                "have_locally": False,
                "annotation_method": (
                    "expert photointerpretation of walrus herds in individual satellite "
                    "images (optical + SAR); herd extent digitised as polygons"
                ),
                "citation": (
                    "Fischbach, A.S., Douglas, D.C., 2022. Pacific Walrus coastal haulout "
                    "occurrences interpreted from satellite imagery (ver 6.0, December "
                    "2025): U.S. Geological Survey data release, "
                    "https://doi.org/10.5066/P9CSM0KN"
                ),
                "mission_codes": MISSION_CODES,
                "download_stats": stats,
            },
            "sensors_relevant": ["sentinel2", "sentinel1", "landsat"],
            "classes": [
                {"id": i, "name": name, "description": desc}
                for i, (name, desc) in enumerate(CLASSES)
            ],
            "nodata_value": io.CLASS_NODATA,
            "num_samples": n_written,
            "is_change_dataset": False,
            "notes": (
                "Single-class walrus-haulout / herd-extent segmentation from USGS "
                "expert-interpreted satellite outlines at Chukchi Sea coastal haulouts "
                "(Alaska + Chukotka), autumn 2017-2025. 64x64 uint8 tiles, local UTM at "
                "10 m; class 0 = herd extent, 255 = nodata (positive-only foreground -- "
                "no fabricated negatives, per spec section 5; assembly adds negatives "
                "from other datasets). Per-image time_range: +/-30 min (~1 hour) centered "
                "on each source image's acquisition datetime (specific-image label; "
                "walruses are mobile so a yearly window would be misleading), change_time "
                "null. One shapefile = one dated image = one herd observation; its herd "
                "sub-polygons are unioned. Small haulouts -> 1 centered tile; large/"
                f"elongated ones (up to ~2.5 km) gridded into 64x64 windows, up to "
                f"{MAX_TILES_PER_IMAGE} per image; round-robin selection capped at "
                f"{MAX_SAMPLES}. all_touched rasterization keeps small herds visible. "
                "Caveats: (1) USGS notes cross-mission georeferencing can shift outlines "
                "tens to >100 m from the true coastline (uncorrected). (2) Source-side: "
                "Vankarem_2023.zip 404s and CapeSerdtseKamen_2025.zip is 0 bytes, so ~16 "
                "herd images are temporarily unavailable -- re-run to add them. (3) 7 "
                "'ESRI Shapefile.shp' artifacts lack a parseable timestamp and are skipped."
            ),
        },
    )
    print("write results:", dict(counts))
    print("total tif on disk:", n_written)
    print("done")


if __name__ == "__main__":
    multiprocessing.set_start_method("forkserver", force=True)
    main()
