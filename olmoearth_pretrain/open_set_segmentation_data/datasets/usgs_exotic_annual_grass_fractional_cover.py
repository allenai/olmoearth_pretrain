"""USGS Exotic Annual Grass (EAG) fractional cover -> open-set regression label patches.

Source: USGS EROS / MRLC "RCMAP - Weekly Herbaceous and Exotic Annual Grass (EAG)"
product (https://www.mrlc.gov/data/type/exotic-annual-grass ; data release DOI
10.5066/P13QWBFH, the current 2016-2026 generation; the manifest cites the sibling annual
release 10.5066/P9GC5JVG, 2016-2024). EAG maps the per-pixel percent cover (0-100) of
combined invasive exotic annual grasses -- cheatgrass (Bromus tectorum), medusahead
(Taeniatherum caput-medusae), field brome and ~12 other Bromus/Ventenata species -- across
the western US sagebrush biome at 30 m, derived from Harmonized Landsat-Sentinel (HLS)
imagery via machine-learning regression trained on BLM AIM + RCMAP field plots. We regress
the **Total EAG** component (combined exotic annual grass cover), the manifest's primary
"exotic annual grass cover" class.

This is a *regression* dataset (continuous per-pixel percent cover, 0-100).

Access / "download only what the labels require": the full-resolution annual rasters are
distributed only via a ScienceBase download that is now behind an interactive Captcha, and
the MRLC data-bundle names for EAG are not resolvable. The same Total EAG native (30 m)
rasters are, however, served as an OGC coverage from the USGS MRLC GeoServer
(dmsdata.cr.usgs.gov). We therefore do **bounded WMS GetMap reads** (format=image/geotiff,
which returns the raw 0-100 values, not a styled image) -- the OGC analogue of COG range
reads -- so we never download the whole western-US mosaic. A single decimated full-extent
GetMap picks a spatially-distributed, cover-balanced set of candidate windows; each selected
64x64 tile is then read from the server at native 30 m over just that window.

Only the 2016-2026 generation is currently loaded on the GeoServer (weekly granules for
2026 so far; the "Total EAG" component is cumulative year-to-date, so the latest available
week -- 2026-06-25 -- is the most-complete annual-representative EAG cover map). 2026 is
post-2016 (Sentinel era); we anchor each tile to a 1-year window on 2026.

Output: single-band float32 GeoTIFFs, local UTM, 10 m/pixel, 64x64 (~640 m), nodata -99999
(io.REGRESSION_NODATA). Source is 30 m EPSG:5070 (CONUS Albers); each tile is reprojected /
resampled to local UTM at 10 m (bilinear, continuous cover field) and DOCUMENTED. Source
mask value 101 (non-rangeland / water / outside mapping area) -> nodata. One <=1-year time
range per tile.
"""

import argparse
import multiprocessing
import random
import tempfile
from collections import Counter
from typing import Any

import numpy as np
import rasterio
import tqdm
from rasterio.enums import Resampling
from rslearn.utils.mp import star_imap_unordered
from rslearn.utils.raster_format import GeotiffRasterFormat
from upath import UPath

from olmoearth_pretrain.open_set_segmentation_data import download, io

SLUG = "usgs_exotic_annual_grass_fractional_cover"
NAME = "USGS Exotic Annual Grass Fractional Cover"
URL = "https://www.mrlc.gov/data/type/exotic-annual-grass"
DOI = "https://doi.org/10.5066/P13QWBFH"

# MRLC GeoServer OGC endpoint for the Total EAG native (30 m) western-CONUS coverage.
WMS_BASE = (
    "https://dmsdata.cr.usgs.gov/geoserver/"
    "mrlc_total-eag-native_westernconus_week_data/wms"
)
WMS_LAYER = "total-eag-native_westernconus_week_data"
# Latest available weekly granule; "Total EAG" is cumulative year-to-date so this is the
# most-complete annual EAG cover for 2026.
WMS_TIME = "2026-06-25T00:00:00.000Z"
YEAR = 2026
SRC_CRS = "EPSG:5070"
SRC_RES = 30.0  # native metres/pixel
SRC_NODATA = 101  # values 0..100 valid percent cover; 101 = mask/non-rangeland

# Full mapped extent of the coverage (EPSG:5070 metres), from WCS DescribeCoverage.
EXTENT = (-2362395.0, 309855.0, 592485.0, 3267405.0)  # (minx, miny, maxx, maxy)

TILE = 64  # output tile size (px), 10 m => ~640 m
TOTAL = 5000  # regression per-dataset target (<= 25k cap)
OV_WIDTH = 2000  # decimated overview width (px) ~= 1.48 km/px
WINDOW_M = 1200.0  # native-res WMS window side (m) fetched per tile (covers 640 m tile)
# Fixed percent-cover bucket edges for the zero-inflated EAG distribution (right edge 101 =>
# last bucket [50, 100]). Balancing over these gives an even spread of cover levels -- many
# low/zero-cover tiles AND high-invasion hotspots -- instead of a mostly-0% corpus.
BUCKET_EDGES = [0, 1, 5, 10, 20, 30, 50, 101]
VALID_MAX = 100
# Restrict to the western US drylands (the 100th meridian is the classic arid/humid
# divide). The coverage's rectangular EPSG:5070 extent runs east to ~-90 lon, where the
# margin is 0-cover edge fill outside the sagebrush/rangeland mapping region; keeping
# lon <= -100 confines samples to the Great Basin / Columbia & Colorado Plateaus / Snake
# River & Wyoming Basins / western Great Plains where EAG is actually mapped.
LON_MAX = -100.0
MAX_CANDIDATES = 400000  # cap candidate pool for memory/speed before balancing
SEED = 42


def overview_path() -> UPath:
    return io.raw_dir(SLUG) / "overview_total_eag_2026-06-25.tif"


def fetch_overview() -> UPath:
    """Fetch (idempotently) a decimated full-extent GetMap for candidate selection."""
    dst = overview_path()
    if dst.exists():
        return dst
    minx, miny, maxx, maxy = EXTENT
    height = int(round(OV_WIDTH * (maxy - miny) / (maxx - minx)))
    print(
        f"fetching overview {OV_WIDTH}x{height} (~{(maxx - minx) / OV_WIDTH / 1000:.2f} km/px)"
    )
    data = download.wms_getmap_geotiff(
        WMS_BASE,
        WMS_LAYER,
        EXTENT,
        OV_WIDTH,
        height,
        srs=SRC_CRS,
        time=WMS_TIME,
        timeout=300,
        retries=5,
    )
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.parent / (dst.name + ".tmp")
    with tmp.open("wb") as f:
        f.write(data)
    tmp.rename(dst)
    return dst


def gather_candidates() -> list[dict[str, Any]]:
    """Read the overview; return candidate window centers (5070 x/y + WGS84 lon/lat + value)."""
    from pyproj import Transformer

    with rasterio.open(overview_path().path) as ds:
        arr = ds.read(1)
        transform = ds.transform
    valid = (arr >= 0) & (arr <= VALID_MAX)
    rows, cols = np.where(valid)
    vals = arr[rows, cols].astype(np.float64)
    # Pixel centers -> 5070 metres -> WGS84 lon/lat.
    xs, ys = transform * (cols + 0.5, rows + 0.5)
    xs = np.asarray(xs, dtype=np.float64)
    ys = np.asarray(ys, dtype=np.float64)
    tf = Transformer.from_crs(SRC_CRS, "EPSG:4326", always_xy=True)
    lons, lats = tf.transform(xs, ys)
    # Confine to the western US drylands mapping region (drop eastern edge fill).
    keep = np.asarray(lons) <= LON_MAX
    xs, ys, vals = xs[keep], ys[keep], vals[keep]
    lons, lats = np.asarray(lons)[keep], np.asarray(lats)[keep]
    rng = np.random.default_rng(SEED)
    if xs.size > MAX_CANDIDATES:
        sel = rng.choice(xs.size, size=MAX_CANDIDATES, replace=False)
        xs, ys, vals, lons, lats = xs[sel], ys[sel], vals[sel], lons[sel], lats[sel]
    return [
        {
            "x5070": float(x),
            "y5070": float(y),
            "lon": float(lo),
            "lat": float(la),
            "value": float(v),
        }
        for x, y, lo, la, v in zip(xs, ys, lons, lats, vals)
    ]


def bucket_balance_fixed(
    records: list[dict[str, Any]], edges: list[int], total: int, seed: int = SEED
) -> list[dict[str, Any]]:
    """Balance across fixed [edge_i, edge_{i+1}) value buckets (zero-inflated data).

    (The shared quantile-based helper degenerates on zero-inflated cover, so we use fixed
    cover buckets, as the RCMAP fractional-cover dataset does.) Take up to total//n_buckets
    per bucket, then top up from leftovers until ``total`` is reached.
    """
    n = len(edges) - 1
    buckets: list[list[dict[str, Any]]] = [[] for _ in range(n)]
    for r in records:
        b = int(np.searchsorted(edges, r["value"], side="right")) - 1
        buckets[min(max(b, 0), n - 1)].append(r)
    rng = random.Random(seed)
    for b in buckets:
        rng.shuffle(b)
    per = max(1, total // n)
    selected: list[dict[str, Any]] = []
    leftovers: list[dict[str, Any]] = []
    for b in buckets:
        selected.extend(b[:per])
        leftovers.extend(b[per:])
    if len(selected) < total:
        rng.shuffle(leftovers)
        selected.extend(leftovers[: total - len(selected)])
    rng.shuffle(selected)
    return selected[:total]


def _write_one(rec: dict[str, Any]) -> dict[str, Any] | None:
    sample_id = rec["sample_id"]
    tif_path = io.locations_dir(SLUG) / f"{sample_id}.tif"
    if tif_path.exists():
        with rasterio.open(tif_path.path) as ds:
            ev = ds.read(1)
        good = ev[ev != io.REGRESSION_NODATA]
        if good.size == 0:
            return {"sample_id": sample_id, "n_valid": 0}
        return {
            "sample_id": sample_id,
            "n_valid": int(good.size),
            "mean": float(good.mean()),
            "min": float(good.min()),
            "max": float(good.max()),
        }

    lon, lat = rec["lon"], rec["lat"]
    proj, col, row = io.lonlat_to_utm_pixel(lon, lat)
    bounds = io.centered_bounds(col, row, TILE, TILE)

    # Fetch a native-res 5070 window (covers the 640 m UTM tile with margin) from the server.
    cx, cy = rec["x5070"], rec["y5070"]
    half = WINDOW_M / 2.0
    wbox = (cx - half, cy - half, cx + half, cy + half)
    npix = int(round(WINDOW_M / SRC_RES))
    try:
        data = download.wms_getmap_geotiff(
            WMS_BASE,
            WMS_LAYER,
            wbox,
            npix,
            npix,
            srs=SRC_CRS,
            time=WMS_TIME,
            timeout=120,
            retries=4,
        )
    except Exception as e:  # noqa: BLE001 - a transient per-window failure just drops the tile
        return {"sample_id": sample_id, "n_valid": 0, "error": str(e)[:80]}

    # Write the fetched window to a local temp dir, then reproject 5070 -> UTM 10 m 64x64.
    with tempfile.TemporaryDirectory() as td:
        tdp = UPath(td)
        fname = "win.tif"
        with (tdp / fname).open("wb") as f:
            f.write(data)
        with rasterio.open((tdp / fname).path) as ds:
            wnodata = ds.nodata
        ra = GeotiffRasterFormat().decode_raster(
            tdp, proj, bounds, resampling=Resampling.bilinear, fname=fname
        )

    vals = ra.array[0].astype(np.float32)
    invalid = (~np.isfinite(vals)) | (vals < 0) | (vals > VALID_MAX)
    if wnodata is not None:
        invalid |= np.abs(vals - float(wnodata)) < 0.5
    invalid |= np.abs(vals - float(SRC_NODATA)) < 0.5
    vals[invalid] = io.REGRESSION_NODATA

    good = vals[vals != io.REGRESSION_NODATA]
    if good.size == 0:
        # Window fell entirely on the source mask (non-rangeland / water); an all-nodata
        # tile is not a usable label -- skip writing it (keeps re-runs idempotent: the
        # sample id simply stays absent).
        return {"sample_id": sample_id, "n_valid": 0}

    io.write_label_geotiff(
        SLUG, sample_id, vals, proj, bounds, nodata=io.REGRESSION_NODATA
    )
    io.write_sample_json(
        SLUG,
        sample_id,
        proj,
        bounds,
        io.year_range(YEAR),
        source_id=f"total_eag_{WMS_TIME[:10]}",
    )
    return {
        "sample_id": sample_id,
        "n_valid": int(good.size),
        "mean": float(good.mean()),
        "min": float(good.min()),
        "max": float(good.max()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=64)
    parser.add_argument("--limit", type=int, default=0, help="cap #samples (0 = full)")
    args = parser.parse_args()

    io.check_disk()
    fetch_overview()
    io.check_disk()

    cands = gather_candidates()
    print(f"gathered {len(cands)} candidate windows from overview")
    selected = bucket_balance_fixed(cands, BUCKET_EDGES, TOTAL, seed=SEED)
    if args.limit:
        selected = selected[: args.limit]
    for i, r in enumerate(selected):
        r["sample_id"] = f"{i:06d}"
    sel_vals = np.array([r["value"] for r in selected], dtype=np.float64)
    bcounts = Counter(
        min(
            max(int(np.searchsorted(BUCKET_EDGES, v, side="right")) - 1, 0),
            len(BUCKET_EDGES) - 2,
        )
        for v in sel_vals
    )
    print(
        f"selected {len(selected)} windows; overview-value bucket counts {dict(sorted(bcounts.items()))}"
    )

    io.locations_dir(SLUG).mkdir(parents=True, exist_ok=True)
    io.check_disk()
    stats: list[dict[str, Any]] = []
    with multiprocessing.Pool(args.workers) as p:
        for res in tqdm.tqdm(
            star_imap_unordered(p, _write_one, [dict(rec=r) for r in selected]),
            total=len(selected),
            desc="write",
        ):
            if res is not None:
                stats.append(res)

    valid_stats = [s for s in stats if s.get("n_valid", 0) > 0]
    n_written = len(valid_stats)
    pix_min = min((s["min"] for s in valid_stats), default=0.0)
    pix_max = max((s["max"] for s in valid_stats), default=0.0)
    n_err = sum(1 for s in stats if s.get("error"))
    if n_err:
        print(f"[warn] {n_err} window fetches failed transiently and were dropped")

    io.write_dataset_metadata(
        SLUG,
        {
            "dataset": SLUG,
            "name": NAME,
            "task_type": "regression",
            "source": "USGS EROS / MRLC (RCMAP Exotic Annual Grass)",
            "license": "public domain (US Government work)",
            "provenance": {
                "url": URL,
                "doi": DOI,
                "have_locally": False,
                "annotation_method": (
                    "HLS (Harmonized Landsat-Sentinel) regression / ML trained on BLM AIM + "
                    "RCMAP field plots"
                ),
                "access": (
                    "bounded WMS GetMap (image/geotiff, raw values) from MRLC GeoServer "
                    f"{WMS_BASE} layer {WMS_LAYER}, TIME={WMS_TIME}"
                ),
            },
            "sensors_relevant": ["sentinel2", "sentinel1", "landsat"],
            "regression": {
                "name": "exotic_annual_grass_cover",
                "description": (
                    "Per-pixel percent cover (0-100) of combined invasive exotic annual "
                    "grasses (cheatgrass Bromus tectorum, medusahead Taeniatherum "
                    "caput-medusae, field brome, and ~12 other Bromus/Ventenata species) "
                    "across the western US sagebrush biome, from the USGS/MRLC RCMAP Exotic "
                    "Annual Grass product (Total EAG component), 30 m native, HLS-based ML "
                    "regression trained on BLM AIM + RCMAP field plots. The distribution is "
                    "heavily zero-inflated; tiles were bucket-balanced across fixed cover "
                    "buckets to give an even spread of cover levels."
                ),
                "unit": "percent cover",
                "dtype": "float32",
                "value_range": [round(pix_min, 3), round(pix_max, 3)],
                "nodata_value": io.REGRESSION_NODATA,
                "source_mask_value": SRC_NODATA,
                "buckets": BUCKET_EDGES,
            },
            "num_samples": n_written,
            "notes": (
                "Total EAG (combined exotic annual grass) cover regressed as the manifest's "
                "primary 'exotic annual grass cover' class. Native 30 m EPSG:5070 (CONUS "
                "Albers) reprojected/resampled per-tile to local UTM at 10 m via bilinear "
                "(continuous cover field); source mask value 101 -> nodata -99999. Bounded "
                "WMS GetMap reads (no full-mosaic download). Only the 2016-2026 generation is "
                f"currently served; TIME={WMS_TIME[:10]} (latest weekly granule; Total EAG is "
                "cumulative year-to-date). 1-year time window on " + str(YEAR) + "."
            ),
        },
    )

    year_counts = {YEAR: n_written}
    hist_edges = [0, 1, 5, 10, 20, 30, 50, 100.0001]
    hist, _ = np.histogram(sel_vals, bins=hist_edges)
    print("selected-window overview-value histogram (percent EAG cover):")
    for lo, hi, c in zip(hist_edges[:-1], hist_edges[1:], hist):
        print(f"  [{lo:>6}, {hi:>6}) : {c}")
    print(f"year counts: {year_counts}")
    print(f"per-pixel value range across tiles: [{pix_min:.3f}, {pix_max:.3f}] percent")
    print(f"num_samples={n_written} task_type=regression")
    print("done")


if __name__ == "__main__":
    multiprocessing.set_start_method("forkserver", force=True)
    main()
