"""Process the World Settlement Footprint (WSF) 2019 into open-set-segmentation labels.

Source: DLR EOC Geoservice, WSF 2019 (https://geoservice.dlr.de/web/maps/eoc:wsf2019,
download https://download.geoservice.dlr.de/WSF2019/files/). WSF 2019 is a GLOBAL 10 m
binary human-settlement mask derived from 2019 multitemporal Sentinel-1 + Sentinel-2
imagery. Pixel values in the source GeoTIFFs:

    255  settlement
    0    non-settlement (everything else)

The product is distributed as 5138 GeoTIFF tiles in EPSG:4326, each covering a 2x2 degree
area (~222x222 km, ~22488x22488 px at ~8.98e-5 deg/px ~= 10 m) with a 0.1 deg overlap
buffer. Tiles are named by their lower-left corner, e.g. WSF2019_v1_12_18.tif covers
(12E,18N)-(14E,20N).

Label choice (points vs dense_raster): the manifest notes ~1M crowdsourced photointerpreted
*validation* points (collected with Google / MapSwipe support). Those validation points are
NOT publicly released as a downloadable file -- the WSF2019 download directory contains only
the raster tiles, a global COG, thumbnails, and STAC sidecars (verified 2026-07). So we fall
back to BOUNDED dense_raster tiling of the WSF mask itself (spec 5, spec 4 dense_raster),
which is the intended fallback.

Task: binary per-pixel classification, class ids
    0  non_settlement   (source value 0)
    1  settlement        (source value 255)
Both are meaningful classes (settlement vs non-settlement), so neither is nodata; nodata
(255) is only used for pixels with no source coverage (e.g. reproject fill at a tile edge).

Sampling (spec 5, bounded regional): WSF is a global derived-product raster, so we download
only 34 representative 2x2 degree tiles -- major cities on every inhabited continent (diverse
settlement morphologies) plus rural / arid / boreal / forest tiles (clean non-settlement
landscapes) -- and cut 64x64 @ 10 m windows in local UTM (reprojected from EPSG:4326 with
NEAREST resampling, categorical). We build two window pools:
  * settlement windows  -- centred on settlement pixels, settlement fraction >= 5% (carry the
    settlement footprint and its boundary; count toward class 1, and toward class 0 too when
    they also contain non-settlement pixels, which nearly all do);
  * non_settlement windows -- pure background (0% settlement), drawn across all tiles for a
    clean, homogeneous, high-confidence negative-region label (count toward class 0 only).
Up to 1000 windows per class (spec 5), balanced, giving up to ~2000 samples. Static 2019
product -> 1-year time range [2019-01-01, 2020-01-01), change_time=null.

Run:  python3 -m olmoearth_pretrain.open_set_segmentation_data.datasets.world_settlement_footprint_2019
"""

import argparse
import math
import multiprocessing
import urllib.error
from collections import Counter
from typing import Any

import numpy as np
import rasterio
import tqdm
from rasterio.warp import Resampling, reproject, transform_bounds
from rasterio.windows import from_bounds
from rslearn.utils.mp import star_imap_unordered
from rslearn.utils.raster_format import get_transform_from_projection_and_bounds

from olmoearth_pretrain.open_set_segmentation_data import (
    download,
    io,
    manifest,
    sampling,
)

SLUG = "world_settlement_footprint_2019"

YEAR = 2019  # WSF 2019 epoch.
PER_CLASS = 1000
TILE = 64
SRC_CRS = "EPSG:4326"
SETTLE_VAL = 255  # source value for settlement
# out-of-coverage fill sentinel for boundless reads (source uses only 0/255).
FILL = 254

# minimum settlement fraction for a window to count as a "settlement" window.
MIN_SETTLE_FRAC = 0.05
# candidate window centres to draw per tile before balancing.
CAND_SETTLE_PER_TILE = 500
CAND_BG_PER_TILE = 200
# keep candidate centres this far (native px) from tile edges.
EDGE_MARGIN = 96

BASE_URL = "https://download.geoservice.dlr.de/WSF2019/files/WSF2019_v1_{lon}_{lat}.tif"

# Representative global spread of city centres (lon, lat) -> region label. Each falls inside
# one 2x2 deg WSF tile (computed by floor); city tiles give diverse settlement morphology,
# the rural/natural tiles give clean non-settlement landscapes. Tile existence verified.
CITIES: dict[str, tuple[float, float]] = {
    "London / W Europe": (-0.13, 51.5),
    "Paris / W Europe": (2.35, 48.85),
    "Berlin / C Europe": (13.40, 52.52),
    "Moscow / Russia": (37.62, 55.75),
    "Istanbul / Turkey": (28.98, 41.0),
    "Cairo / Egypt": (31.24, 30.05),
    "Lagos / Nigeria": (3.38, 6.52),
    "Nairobi / Kenya": (36.82, -1.29),
    "Johannesburg / S Africa": (28.05, -26.20),
    "Kinshasa / DR Congo": (15.31, -4.32),
    "New York / US NE": (-74.0, 40.71),
    "Los Angeles / US SW": (-118.24, 34.05),
    "Mexico City / Mexico": (-99.13, 19.43),
    "Chicago / US Midwest": (-87.63, 41.88),
    "Sao Paulo / Brazil": (-46.63, -23.55),
    "Bogota / Colombia": (-74.07, 4.71),
    "Lima / Peru": (-77.04, -12.05),
    "Buenos Aires / Argentina": (-58.38, -34.60),
    "Delhi / N India": (77.21, 28.61),
    "Mumbai / W India": (72.88, 19.08),
    "Dhaka / Bangladesh": (90.41, 23.81),
    "Beijing / N China": (116.40, 39.90),
    "Shanghai / E China": (121.47, 31.23),
    "Tokyo / Japan": (139.65, 35.68),
    "Jakarta / Indonesia": (106.85, -6.21),
    "Bangkok / Thailand": (100.50, 13.76),
    "Sydney / Australia": (151.21, -33.87),
    "Dubai / Gulf": (55.27, 25.20),
    # rural / natural / arid / boreal tiles (dominantly non-settlement)
    "US Great Plains (rural)": (-98.0, 38.5),
    "Amazon (rural)": (-60.0, -3.0),
    "Sahel (arid)": (5.0, 15.0),
    "Australia outback (arid)": (133.0, -25.0),
    "Siberia (boreal)": (90.0, 60.0),
    "Canada prairie (rural)": (-106.0, 52.0),
}


def tile_corner(lon: float, lat: float) -> tuple[int, int]:
    """Lower-left corner (even lon, even lat) of the 2x2 deg WSF tile containing lon/lat."""
    return int(math.floor(lon / 2) * 2), int(math.floor(lat / 2) * 2)


# region label -> (lon_corner, lat_corner); deduped so each tile is downloaded once.
TILES: dict[tuple[int, int], str] = {}
for _region, (_lon, _lat) in CITIES.items():
    TILES.setdefault(tile_corner(_lon, _lat), _region)


def tile_path(lon: int, lat: int):
    return io.raw_dir(SLUG) / f"WSF2019_v1_{lon}_{lat}.tif"


# Class id (position) -> (name, description).
CLASSES: list[tuple[str, str]] = [
    (
        "non_settlement",
        "Non-settlement: any 10 m cell not classified as human settlement in WSF 2019 "
        "(vegetation, bare soil, water, agriculture, etc.). Source value 0.",
    ),
    (
        "settlement",
        "Human settlement: 10 m cells covered by any kind of building / built structure, as "
        "detected from 2019 multitemporal Sentinel-1 + Sentinel-2. Source value 255.",
    ),
]


def _download_one(rc: tuple[int, int]) -> tuple[tuple[int, int], str]:
    """Download one tile. Returns (rc, status): "ok", "404", or "err:<msg>".

    Never raises across the pool boundary (HTTPError holds an unpicklable file object),
    so all failure info is returned as a plain string.
    """
    import time as _time

    lon, lat = rc
    io.check_disk()
    last = ""
    for attempt in range(6):
        try:
            download.download_http(
                BASE_URL.format(lon=lon, lat=lat), tile_path(lon, lat), timeout=300
            )
            return rc, "ok"
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return rc, "404"
            last = f"HTTP {e.code}"  # 503 rate-limit etc: retry with backoff
        except Exception as e:  # noqa: BLE001
            last = repr(e)
        _time.sleep(2.0 * (attempt + 1))
    return rc, f"err:{last}"


def download_source(workers: int) -> None:
    args = [dict(rc=rc) for rc in TILES]
    missing, errors = [], []
    workers = min(workers, 4)  # DLR server 503s under heavy concurrency; be gentle
    with multiprocessing.Pool(min(workers, len(args))) as p:
        for rc, status in tqdm.tqdm(
            star_imap_unordered(p, _download_one, args),
            total=len(args),
            desc="download",
        ):
            if status == "404":
                missing.append(rc)
            elif status != "ok":
                errors.append((rc, status))
    for rc in missing:
        print(f"  WARNING tile {rc} missing (404); skipping")
        TILES.pop(rc, None)
    if errors:
        raise RuntimeError(f"download errors (transient?): {errors}")


def _scan_tile(rc: tuple[int, int]) -> list[dict[str, Any]]:
    """Load a tile; draw settlement- and background-centred candidate window centres.

    classes_present / settlement fraction are computed on the native (EPSG:4326) 64x64
    window; the reprojected UTM label (written in _write_one) is nearest-resampled at the
    same 10 m so the class content matches closely.
    """
    lon, lat = rc
    with rasterio.open(tile_path(lon, lat)) as ds:
        raw = ds.read(1)  # (H, W) uint8, 0 / 255
        transform = ds.transform
    settle = (raw == SETTLE_VAL).astype(np.uint8)
    del raw
    h, w = settle.shape
    half = TILE // 2
    # WSF is in EPSG:4326: a native pixel spans ~10 m in latitude but ~10*cos(lat) m in
    # longitude, so a fixed 64-col native window is narrower on the ground than the 640 m
    # UTM output. Widen the native column window by 1/cos(lat) (using the tile-centre
    # latitude) so the settlement fraction / classes we tag match the reprojected UTM label.
    lat_c = lat + 1.0  # tile centre latitude (2 deg tile)
    col_span = max(TILE, int(round(TILE / max(0.2, math.cos(math.radians(lat_c))))))
    col_half = col_span // 2
    margin = max(EDGE_MARGIN, col_half + 1)
    rng = np.random.default_rng((lon + 200) * 1000 + (lat + 200))

    def win_at(row: int, col: int) -> np.ndarray:
        return settle[
            row - half : row - half + TILE, col - col_half : col - col_half + col_span
        ]

    def center_lonlat(row: int, col: int) -> tuple[float, float]:
        x, y = transform * (col + 0.5, row + 0.5)
        return float(x), float(y)

    cands: list[dict[str, Any]] = []

    # --- settlement-centred candidates ---
    idx = np.flatnonzero(settle.reshape(-1))
    if idx.size:
        if idx.size > CAND_SETTLE_PER_TILE * 20:
            idx = rng.choice(idx, CAND_SETTLE_PER_TILE * 20, replace=False)
        rows = (idx // w).astype(np.int64)
        cols = (idx % w).astype(np.int64)
        keep = (
            (rows >= margin)
            & (rows < h - margin)
            & (cols >= margin)
            & (cols < w - margin)
        )
        rows, cols = rows[keep], cols[keep]
        order = rng.permutation(rows.size)
        n = 0
        for i in order:
            if n >= CAND_SETTLE_PER_TILE:
                break
            row, col = int(rows[i]), int(cols[i])
            win = win_at(row, col)
            frac = float(win.mean())
            if frac < MIN_SETTLE_FRAC:
                continue
            lo, la = center_lonlat(row, col)
            present = [0, 1] if frac < 1.0 else [1]
            cands.append(
                {
                    "lon_t": lon,
                    "lat_t": lat,
                    "row": row,
                    "col": col,
                    "lon": lo,
                    "lat": la,
                    "category": "settlement",
                    "classes_present": present,
                    "source_id": f"WSF2019_v1_{lon}_{lat}_r{row}_c{col}",
                }
            )
            n += 1

    # --- background-centred candidates (pure non-settlement) ---
    bg = np.flatnonzero((settle == 0).reshape(-1))
    if bg.size:
        pick = rng.choice(bg, min(bg.size, CAND_BG_PER_TILE * 20), replace=False)
        rows = (pick // w).astype(np.int64)
        cols = (pick % w).astype(np.int64)
        keep = (
            (rows >= margin)
            & (rows < h - margin)
            & (cols >= margin)
            & (cols < w - margin)
        )
        rows, cols = rows[keep], cols[keep]
        order = rng.permutation(rows.size)
        n = 0
        for i in order:
            if n >= CAND_BG_PER_TILE:
                break
            row, col = int(rows[i]), int(cols[i])
            win = win_at(row, col)
            if win.any():
                continue  # not pure background
            lo, la = center_lonlat(row, col)
            cands.append(
                {
                    "lon_t": lon,
                    "lat_t": lat,
                    "row": row,
                    "col": col,
                    "lon": lo,
                    "lat": la,
                    "category": "non_settlement",
                    "classes_present": [0],
                    "source_id": f"WSF2019_v1_{lon}_{lat}_r{row}_c{col}",
                }
            )
            n += 1

    del settle
    return cands


# ---- writer: cache open source datasets per process ----
_DS: dict[tuple[int, int], Any] = {}


def _src(lon: int, lat: int):
    key = (lon, lat)
    if key not in _DS:
        _DS[key] = rasterio.open(tile_path(lon, lat))
    return _DS[key]


def _write_one(rec: dict[str, Any]) -> None:
    sample_id = rec["sample_id"]
    if (io.locations_dir(SLUG) / f"{sample_id}.tif").exists():
        return
    lon, lat = rec["lon"], rec["lat"]
    dst_proj, col, row = io.lonlat_to_utm_pixel(lon, lat)
    bounds = io.centered_bounds(col, row, TILE, TILE)
    dst_transform = get_transform_from_projection_and_bounds(dst_proj, bounds)

    # UTM tile extent (metres) -> EPSG:4326 read window.
    xs = [bounds[0] * io.RESOLUTION, bounds[2] * io.RESOLUTION]
    ys = [bounds[1] * -io.RESOLUTION, bounds[3] * -io.RESOLUTION]
    left, right = min(xs), max(xs)
    bottom, top = min(ys), max(ys)
    l2, b2, r2, t2 = transform_bounds(dst_proj.crs, SRC_CRS, left, bottom, right, top)
    pad = 0.003  # ~300 m of geographic margin

    ds = _src(rec["lon_t"], rec["lat_t"])
    win = from_bounds(l2 - pad, b2 - pad, r2 + pad, t2 + pad, ds.transform)
    src = ds.read(1, window=win, boundless=True, fill_value=FILL)
    win_transform = ds.window_transform(win)

    dst_raw = np.full((TILE, TILE), FILL, dtype=np.uint8)
    reproject(
        source=src,
        destination=dst_raw,
        src_transform=win_transform,
        src_crs=ds.crs,
        dst_transform=dst_transform,
        dst_crs=dst_proj.crs,
        resampling=Resampling.nearest,
        src_nodata=FILL,
        dst_nodata=FILL,
    )
    # map source values -> class ids: 0 -> 0 (non_settlement), 255 -> 1 (settlement),
    # FILL/other -> 255 (nodata).
    out = np.full((TILE, TILE), io.CLASS_NODATA, dtype=np.uint8)
    out[dst_raw == 0] = 0
    out[dst_raw == SETTLE_VAL] = 1

    io.write_label_geotiff(
        SLUG, sample_id, out, dst_proj, bounds, nodata=io.CLASS_NODATA
    )
    present = sorted(int(x) for x in np.unique(out) if x != io.CLASS_NODATA)
    io.write_sample_json(
        SLUG,
        sample_id,
        dst_proj,
        bounds,
        io.year_range(YEAR),
        source_id=rec["source_id"],
        classes_present=present,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=64)
    parser.add_argument("--scan-workers", type=int, default=16)
    args = parser.parse_args()

    io.check_disk()
    manifest.write_registry_entry(SLUG, "in_progress")

    print(f"Downloading {len(TILES)} representative WSF 2019 tiles...")
    download_source(args.workers)
    io.check_disk()

    print("Scanning tiles for candidate windows...")
    cands: list[dict[str, Any]] = []
    with multiprocessing.Pool(args.scan_workers) as p:
        for res in tqdm.tqdm(
            star_imap_unordered(p, _scan_tile, [dict(rc=rc) for rc in TILES]),
            total=len(TILES),
            desc="scan",
        ):
            cands.extend(res)
    cat_counts = Counter(c["category"] for c in cands)
    print(f"  {len(cands)} candidate windows: {dict(cat_counts)}")

    # up to PER_CLASS per category (settlement / non_settlement), balanced + seeded.
    selected = sampling.balance_by_class(
        cands,
        key="category",
        per_class=PER_CLASS,
        total_cap=sampling.MAX_SAMPLES_PER_DATASET,
    )
    for i, rec in enumerate(selected):
        rec["sample_id"] = f"{i:06d}"
    print(f"  selected {len(selected)} windows")

    io.check_disk()
    with multiprocessing.Pool(args.workers) as p:
        for _ in tqdm.tqdm(
            star_imap_unordered(p, _write_one, [dict(rec=r) for r in selected]),
            total=len(selected),
            desc="write",
        ):
            pass

    # tiles-per-class report (a window counts toward every class present in it).
    sel_class_counts: Counter = Counter()
    for rec in selected:
        for cc in set(rec["classes_present"]):
            sel_class_counts[cc] += 1
    class_counts = {
        name: sel_class_counts.get(i, 0) for i, (name, _d) in enumerate(CLASSES)
    }
    cat_selected = Counter(rec["category"] for rec in selected)
    print("selected windows per category:", dict(cat_selected))
    print("selected tiles containing each class:", class_counts)

    io.write_dataset_metadata(
        SLUG,
        {
            "dataset": SLUG,
            "name": "World Settlement Footprint 2019",
            "task_type": "classification",
            "source": "DLR",
            "license": "CC-BY-4.0",
            "provenance": {
                "url": "https://geoservice.dlr.de/web/maps/eoc:wsf2019",
                "download_url": "https://download.geoservice.dlr.de/WSF2019/files/",
                "have_locally": False,
                "annotation_method": (
                    "derived-product (10 m binary settlement mask from 2019 Sentinel-1/2); "
                    "validated by DLR with ~1M crowdsourced photointerpreted reference points "
                    "(not publicly released, so dense_raster tiling of the mask is used)"
                ),
                "epoch": YEAR,
                "native_resolution_m": 10,
                "native_crs": SRC_CRS,
            },
            "sensors_relevant": ["sentinel2", "sentinel1", "landsat"],
            "classes": [
                {"id": i, "name": n, "description": d}
                for i, (n, d) in enumerate(CLASSES)
            ],
            "nodata_value": io.CLASS_NODATA,
            "num_samples": len(selected),
            "tiles_per_class": class_counts,
            "windows_per_category": dict(cat_selected),
            "regions_sampled": sorted(TILES.values()),
            "notes": (
                "Bounded regional tiling (spec 5) of the GLOBAL DLR World Settlement Footprint "
                f"2019, 10 m native, EPSG:4326, epoch {YEAR}. The ~1M crowdsourced validation "
                "points referenced by the manifest are NOT publicly downloadable (the WSF2019 "
                "download directory exposes only raster tiles + a global COG + STAC), so we tile "
                "the mask itself. 34 representative 2x2 deg tiles were downloaded -- major cities "
                "on every inhabited continent (diverse settlement morphology) + rural/arid/boreal "
                "tiles (clean non-settlement) -- and 64x64 @10 m windows cut in local UTM, "
                "reprojected from EPSG:4326 with NEAREST resampling (categorical, no bilinear). "
                "Binary classification: source 0 -> class 0 (non_settlement), 255 -> class 1 "
                "(settlement); reproject fill -> 255 nodata. Two window pools: settlement windows "
                "(centred on settlement, >=5% settlement, carrying the footprint + boundary) and "
                "pure non_settlement windows (0% settlement); up to 1000 per class, balanced. "
                "Static 2019 product -> 1-year window [2019-01-01,2020-01-01), change_time=null."
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
