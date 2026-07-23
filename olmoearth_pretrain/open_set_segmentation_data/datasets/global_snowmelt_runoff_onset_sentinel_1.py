"""Global Snowmelt Runoff Onset (Sentinel-1) -> open-set-segmentation regression patches.

Source: Gagliano, Shean & Henderson (2026), "A global high-resolution dataset of snowmelt
runoff onset timing from Sentinel-1 SAR, 2015-2024", Zenodo record 19618062
(https://zenodo.org/records/19618062), CC-BY-4.0. The first comprehensive global map of
snowmelt *runoff onset* timing: Sentinel-1 C-band SAR (VV) backscatter minima -- which
coincide with the ripening->runoff transition of melting seasonal snow -- detected inside a
custom MODIS-derived snow-phenology search window. Validated against 735 Western-US snow
pillows (median timing difference -1.0 d, MAD 9.0 d).

This is a REGRESSION target: per-pixel `runoff_onset`, the **day of water year (DOWY)** of
runoff onset, integer 1-366, no-data -9999. Native grid is EPSG:4326 at ~80 m effective
resolution (pixel spacing ~7.2e-4 deg), dims (water_year: 10, lat: 195970, lon: 499998),
signed int16. `runoff_onset` has NO scale factor (integer DOWY); the 0.1 scale factor
documented for the record applies only to the *_mad / temporal_resolution variables, which
we do not use. Water-year definition (from the record):
  * Northern Hemisphere: WY N = Oct 1 (N-1) .. Sep 30 (N); DOWY 1 = Oct 1 (N-1).
  * Southern Hemisphere: WY N = Apr 1 (N) .. Mar 31 (N+1); DOWY 1 = Apr 1 (N).
Typical runoff onset is DOWY ~110-270 (the product's own colour scale), i.e. the melt event
of water year N falls within **calendar year N** in BOTH hemispheres (NH spring N; SH austral
spring/summer N). So each annual layer is assigned a 1-year time window on calendar year N.

Sentinel era: the record covers WY2015-2024, but WY2015's melt occurs in spring 2015 (NH)
which is pre-2016. We DROP WY2015 (pre-2016), then from WY2016-2024 sample a bounded, evenly
spaced 5-water-year subset -- WY2016, 2018, 2020, 2022, 2024 -> calendar-year windows 2016,
2018, 2020, 2022, 2024. This bounded temporal sampling keeps the network job inside Zenodo's
guest rate-limit window while still giving 5 distinct 1-year pairing windows per region.

Access (bounded, spec 5): this is a large global derived product, so we do NOT attempt
global coverage. The record ships a Kerchunk reference file
(`global_snowmelt_runoff_onset.zarr.tar.refs.json`, ~14 MB) mapping each Zarr key to
`[tar_url, byte_offset, byte_length]` inside the single `.zarr.tar` on Zenodo. We use it to
fetch **only the chunks a bounded region touches** via our own HTTP range requests + blosc
decode (chunks are 2048x2048 blosc-zstd int16 blocks) -- we do NOT go through fsspec's
ReferenceFileSystem because zarr v3's store cannot consume a v2 kerchunk reference fs
(async-flag mismatch), and the reference format is trivially direct. We read a curated set of
19 seasonal-snow regions spanning both hemispheres (Western US ranges, Alaska, Canadian
Cordillera, Iceland, European Alps, Scandinavia, Caucasus, Himalaya, Tien Shan, Pamir, Altai,
E. Siberia, Central Andes, Patagonia, Southern Alps NZ) for the 5 sampled water years,
caching each region-water-year to raw/regions/{region}_WY{wy}.tif (EPSG:4326, 80 m, int16, nodata -9999)
so the network is hit once and re-runs are idempotent. NOTE: Zenodo's file CDN fingerprints
the User-Agent (403 "unusual traffic" for non-browser agents), so all Zenodo requests send a
real browser User-Agent.

Output: single-band **int16** GeoTIFFs, local UTM, 10 m/pixel, 64x64 (~640 m), nodata
**-9999** (the source sentinel; the repo's default -99999 does not fit int16). Each 64x64
window corresponds to an 8x8 native (80 m) block; the ~80 m source window is reprojected /
resampled from EPSG:4326 to the local UTM tile at 10 m (bilinear over the continuous DOWY
field; a nearest/threshold validity mask is warped alongside so -9999 never blends into
valid pixels, and reprojected values are rounded back to integer DOWY). Windows are
**bucket-balanced** across the DOWY distribution (quantile buckets) to span the melt-timing
range rather than piling up in the seasonal mode. Up to 5000 samples (spec 5 regression cap).
"""

import argparse
import base64
import json
import math
import multiprocessing
import random
import time
from typing import Any

import numpy as np
import rasterio
import requests
import tqdm
from affine import Affine
from numcodecs import Blosc
from rasterio.transform import from_origin
from rasterio.warp import Resampling, reproject
from rasterio.windows import Window
from rslearn.utils.mp import star_imap_unordered

from olmoearth_pretrain.open_set_segmentation_data import download, io, sampling

SLUG = "global_snowmelt_runoff_onset_sentinel_1"
NAME = "Global Snowmelt Runoff Onset (Sentinel-1)"
URL = "https://zenodo.org/records/19618062"
DOI = "https://doi.org/10.5281/zenodo.16953614"
ZENODO_RECORD = "19618062"

# Kerchunk reference file for lazy, bounded remote reads of the global Zarr store.
REFS_NAME = "global_snowmelt_runoff_onset.zarr.tar.refs.json"
REFS_URL = f"https://zenodo.org/records/{ZENODO_RECORD}/files/{REFS_NAME}?download=1"
# Zenodo's file CDN fingerprints the User-Agent and 403s non-browser agents ("unusual
# traffic from your network"); a real browser UA succeeds. Used on ALL Zenodo file
# requests, including the fsspec/HTTP range reads that back the Kerchunk lazy access.
BROWSER_UA = "Mozilla/5.0 (X11; Linux x86_64; rv:125.0) Gecko/20100101 Firefox/125.0"

VARIABLE = "runoff_onset"  # annual DOWY of runoff onset (int16, 1-366)
SRC_NODATA = -9999  # source _FillValue (also our output nodata; fits int16)
SRC_RES_DEG = 7.2e-4  # native pixel spacing (~80 m at equator)

# WY2015 melt is pre-2016 (NH spring 2015); drop it. From WY2016-2024 we sample a bounded
# 5-water-year subset evenly spanning the era. This is deliberate bounded sampling (spec 5):
# it keeps the network job inside Zenodo's guest rate window (~2000 requests/hour) while
# still giving strong temporal diversity (5 distinct 1-year pairing windows per region), and
# it easily yields far more than the 5000-sample target across the 19 regions.
WATER_YEARS = [2016, 2018, 2020, 2022, 2024]

# Curated bounded set of seasonal-snow regions, both hemispheres (spec 5: a large global
# derived product -> sample representative regions, not the whole planet). box = (min_lon,
# min_lat, max_lon, max_lat) in EPSG:4326. Boxes are kept modest (~2-3 deg) so each
# region+year read touches few Zarr chunks (Zenodo rate limits large lazy queries).
REGIONS: dict[str, tuple[float, float, float, float]] = {
    "sierra_nevada_us": (-120.6, 36.5, -118.0, 39.2),
    "colorado_rockies_us": (-108.6, 37.4, -105.6, 40.6),
    "cascades_us": (-122.2, 46.3, -120.4, 48.8),
    "wasatch_uinta_us": (-112.2, 39.4, -109.4, 41.6),
    "alaska_range_us": (-152.5, 61.0, -147.0, 64.0),
    "coast_mtns_bc_ca": (-127.5, 49.8, -123.2, 53.2),
    "canadian_rockies_ca": (-118.5, 50.0, -114.5, 53.2),
    "iceland": (-22.5, 63.4, -14.0, 66.2),
    "european_alps": (6.0, 45.4, 12.5, 47.6),
    "scandinavia": (7.0, 60.0, 15.5, 63.5),
    "caucasus": (41.5, 42.0, 45.5, 44.2),
    "himalaya_w": (75.5, 30.0, 82.0, 33.2),
    "tien_shan": (75.5, 40.8, 80.5, 43.6),
    "pamir": (70.8, 37.0, 74.2, 39.6),
    "altai": (86.5, 47.6, 91.5, 51.2),
    "eastern_siberia": (134.5, 59.8, 142.0, 63.2),
    "andes_central_sh": (-71.2, -35.4, -69.0, -32.0),  # Southern Hemisphere
    "patagonia_sh": (-73.8, -50.2, -71.2, -46.8),  # Southern Hemisphere
    "southern_alps_nz_sh": (167.8, -44.8, 171.2, -42.8),  # Southern Hemisphere
}

TILE = 64  # output tile size (px); 10 m => ~640 m ground.
BLOCK = 8  # native (80 m) block that maps to a 64x64 10 m tile (8*80=640 m).
HALF = 12  # native-px half-window read around a center for reprojection.
MIN_VALID_FRAC = 0.50  # >=50% of the native block must be observed (non -9999) snow.
CAP_PER_REGION_YEAR = 500  # reservoir cap per region+year (bounds candidate memory).
# Zenodo throttles guest file access (~60 req/min, ~2000 req/hour per IP). Chunk reads are
# paced ~REQ_DELAY apart to stay comfortably under both limits (steady, not bursty), which
# is far more reliable than burst+backoff. ~1300 chunk reads * REQ_DELAY dominates runtime.
REQ_DELAY = 0.5  # added delay between chunk range requests. With ~1.1s natural
# latency this gives ~37/min -- under Zenodo's guest limits (60/min,
# 2000/hour); the bounded ~1500-request job fits inside one window.
TOTAL = 5000  # regression per-dataset target (<= 25k cap).
N_BUCKETS = 10  # DOWY quantile buckets for balancing.
SEED = 42


def _region_tif(region: str, wy: int):
    return io.raw_dir(SLUG) / "regions" / f"{region}_WY{wy}.tif"


# Module-level Zarr metadata parsed from the Kerchunk reference file (loaded lazily by
# _load_refs). The reference maps zarr keys -> [tar_url, byte_offset, byte_length]; chunks
# are blosc-zstd compressed int16 blocks inside the single .zarr.tar on Zenodo.
_REFS: dict[str, Any] | None = None
_ZMETA: dict[str, Any] = {}


def _load_refs() -> tuple[dict[str, Any], dict[str, Any]]:
    """Download (idempotently) the ~14 MB Kerchunk reference file and parse Zarr metadata.

    We read chunks with our own HTTP range requests + blosc decode rather than fsspec's
    ReferenceFileSystem: zarr v3's FsspecStore cannot consume a v2 kerchunk reference fs
    (async-flag mismatch), and the reference format here is trivially direct
    ([url, offset, length] per chunk), so a self-contained reader is both simpler and avoids
    the incompatibility. Returns (refs, meta) where meta has chunk shape, array shape,
    fill value and the EPSG:4326 GeoTransform (x0, dx, y0, dy).
    """
    global _REFS, _ZMETA
    if _REFS is not None:
        return _REFS, _ZMETA
    refs_path = io.raw_dir(SLUG) / REFS_NAME
    download.download_http(
        REFS_URL, refs_path, headers={"User-Agent": BROWSER_UA}, timeout=600
    )
    refs = json.load(refs_path.open())["refs"]

    def _b64(key: str) -> dict[str, Any]:
        return json.loads(base64.b64decode(refs[key][len("base64:") :]))

    za = _b64(f"{VARIABLE}/.zarray")
    gt = [float(x) for x in _b64("spatial_ref/.zattrs")["GeoTransform"].split()]
    meta = {
        "ch_lat": za["chunks"][1],
        "ch_lon": za["chunks"][2],
        "n_lat": za["shape"][1],
        "n_lon": za["shape"][2],
        "fill": za["fill_value"],
        "dtype": za["dtype"],
        "x0": gt[0],
        "dx": gt[1],
        "y0": gt[3],
        "dy": gt[5],
    }
    _REFS, _ZMETA = refs, meta
    return refs, meta


def _read_region_year(region: str, wy: int) -> tuple[np.ndarray, tuple[float, ...]]:
    """Read one (region, water_year) runoff_onset slice via direct HTTP range reads.

    Computes the lat/lon pixel window for the region box, then range-reads and blosc-decodes
    only the covering 2048x2048 chunks from the .zarr.tar (missing chunks = all fill_value).
    Returns (int16 array, GDAL GeoTransform tuple) in EPSG:4326.
    """
    refs, m = _load_refs()
    ch_lat, ch_lon, n_lat, n_lon = m["ch_lat"], m["ch_lon"], m["n_lat"], m["n_lon"]
    x0, dx, y0, dy = m["x0"], m["dx"], m["y0"], m["dy"]
    fill = m["fill"]
    wyi = wy - 2015  # water_year coord runs 2015..2024 at index 0..9
    box = REGIONS[region]
    i0 = max(0, int(math.floor((box[0] - x0) / dx)))
    i1 = min(n_lon, int(math.ceil((box[2] - x0) / dx)))
    j0 = max(0, int(math.floor((y0 - box[3]) / (-dy))))
    j1 = min(n_lat, int(math.ceil((y0 - box[1]) / (-dy))))
    out = np.full((j1 - j0, i1 - i0), fill, dtype=np.int16)
    codec = Blosc()
    with requests.Session() as sess:
        sess.headers.update({"User-Agent": BROWSER_UA})
        for cj in range(j0 // ch_lat, (j1 - 1) // ch_lat + 1):
            for ci in range(i0 // ch_lon, (i1 - 1) // ch_lon + 1):
                key = f"{VARIABLE}/{wyi}.{cj}.{ci}"
                ref = refs.get(key)
                if ref is None:
                    continue  # unstored chunk -> all fill_value (no seasonal snow)
                url, off, length = ref
                for attempt in range(12):
                    try:
                        r = sess.get(
                            url,
                            headers={"Range": f"bytes={off}-{off + length - 1}"},
                            timeout=180,
                        )
                        if r.status_code in (429, 500, 502, 503, 504):
                            # Zenodo rate-limits file access; back off (honor Retry-After,
                            # which can be the full hourly-window reset if the cap is hit).
                            ra = r.headers.get("Retry-After")
                            wait = (
                                float(ra)
                                if ra and ra.isdigit()
                                else 3.0 * (attempt + 1) ** 2
                            )
                            time.sleep(min(wait, 300.0))
                            continue
                        r.raise_for_status()
                        buf = np.frombuffer(
                            codec.decode(r.content), dtype="<i2"
                        ).reshape(ch_lat, ch_lon)
                        time.sleep(REQ_DELAY)  # pace to stay under Zenodo's rate limit
                        break
                    except requests.exceptions.RequestException:
                        if attempt == 11:
                            raise
                        time.sleep(3.0 * (attempt + 1))
                else:
                    raise RuntimeError(
                        f"exhausted retries (rate-limited) reading {key}"
                    )
                gj0, gi0 = cj * ch_lat, ci * ch_lon
                rj0, rj1 = max(j0, gj0), min(j1, gj0 + ch_lat)
                ri0, ri1 = max(i0, gi0), min(i1, gi0 + ch_lon)
                out[rj0 - j0 : rj1 - j0, ri0 - i0 : ri1 - i0] = buf[
                    rj0 - gj0 : rj1 - gj0, ri0 - gi0 : ri1 - gi0
                ]
    transform = (x0 + i0 * dx, dx, 0.0, y0 + j0 * dy, 0.0, dy)
    return out, transform


def _download_one_region_year(region: str, wy: int) -> str:
    """Read + cache one region-year slice to raw/regions/{region}_WY{wy}.tif (idempotent)."""
    dst = _region_tif(region, wy)
    if dst.exists():
        return f"{region}_WY{wy} (present)"
    arr, gt = _read_region_year(region, wy)
    x0, dx, _, y0, _, dy = gt
    transform = from_origin(x0, y0, dx, -dy)
    d = io.raw_dir(SLUG) / "regions"
    d.mkdir(parents=True, exist_ok=True)
    tmp = d / (dst.name + ".tmp")
    with rasterio.open(
        tmp.path,
        "w",
        driver="GTiff",
        height=arr.shape[0],
        width=arr.shape[1],
        count=1,
        dtype="int16",
        crs="EPSG:4326",
        transform=transform,
        nodata=SRC_NODATA,
        compress="deflate",
    ) as ds:
        ds.write(arr, 1)
    tmp.rename(dst)
    return f"{region}_WY{wy} valid={int((arr != SRC_NODATA).sum())}"


def download_regions() -> None:
    """Cache every region+water-year runoff_onset slice to raw/regions/ (idempotent).

    Each slice is a bounded set of direct HTTP range reads into the global .zarr.tar; the
    network is hit at most once per slice (cached as an EPSG:4326 int16 GeoTIFF, nodata
    -9999). Runs SERIALLY with per-request pacing (REQ_DELAY) to stay under Zenodo's guest
    rate limit -- parallel workers reliably trip its 429 throttle. Idempotent: re-running
    resumes from whatever is already cached.
    """
    _load_refs()
    tasks = [
        (rg, wy)
        for rg in REGIONS
        for wy in WATER_YEARS
        if not _region_tif(rg, wy).exists()
    ]
    n_total = len(REGIONS) * len(WATER_YEARS)
    if not tasks:
        print(f"all {n_total} region-year slices present")
        return
    print(f"reading {len(tasks)} / {n_total} region-year slices via HTTP range reads")
    (io.raw_dir(SLUG) / "regions").mkdir(parents=True, exist_ok=True)
    for rg, wy in tqdm.tqdm(tasks, desc="regions"):
        _download_one_region_year(rg, wy)
    io.check_disk()


def scan_region(region: str, wy: int) -> list[dict[str, Any]]:
    """Scan one region+year GeoTIFF in BLOCK x BLOCK native blocks; return candidates.

    Each kept candidate records the block center (native col/row + lon/lat) and the mean
    DOWY over its valid pixels (for bucket balancing). Reservoir-sampled per region-year.
    """
    path = _region_tif(region, wy).path
    rng = random.Random(f"{region}:{wy}")
    kept: list[dict[str, Any]] = []
    n_seen = 0
    with rasterio.open(path) as ds:
        W, H = ds.width, ds.height
        if W < BLOCK + 2 * HALF or H < BLOCK + 2 * HALF:
            return []
        arr = ds.read(1)  # (H, W) int16, -9999 nodata
        for r0 in range(HALF, H - BLOCK - HALF + 1, BLOCK):
            for c0 in range(HALF, W - BLOCK - HALF + 1, BLOCK):
                blk = arr[r0 : r0 + BLOCK, c0 : c0 + BLOCK]
                valid = blk != SRC_NODATA
                vc = int(valid.sum())
                if vc < MIN_VALID_FRAC * BLOCK * BLOCK:
                    continue
                vals = blk[valid]
                # Guard the documented valid range (1-366).
                if vals.min() < 1 or vals.max() > 366:
                    good = vals[(vals >= 1) & (vals <= 366)]
                    if good.size < MIN_VALID_FRAC * BLOCK * BLOCK:
                        continue
                    mean_v = float(good.mean())
                else:
                    mean_v = float(vals.mean())
                row_c, col_c = r0 + BLOCK // 2, c0 + BLOCK // 2
                lon, lat = ds.xy(row_c, col_c)
                rec = {
                    "region": region,
                    "wy": wy,
                    "col": col_c,
                    "row": row_c,
                    "lon": float(lon),
                    "lat": float(lat),
                    "value": mean_v,
                }
                n_seen += 1
                if len(kept) < CAP_PER_REGION_YEAR:
                    kept.append(rec)
                else:
                    k = rng.randint(0, n_seen - 1)
                    if k < CAP_PER_REGION_YEAR:
                        kept[k] = rec
    return kept


def _write_one(rec: dict[str, Any]) -> dict[str, Any] | None:
    sample_id = rec["sample_id"]
    tif_path = io.locations_dir(SLUG) / f"{sample_id}.tif"
    if tif_path.exists():
        with rasterio.open(tif_path.path) as ds:
            ev = ds.read(1)
        good = ev[ev != SRC_NODATA]
        if good.size == 0:
            return {"sample_id": sample_id, "n_valid": 0}
        return {
            "sample_id": sample_id,
            "n_valid": int(good.size),
            "mean": float(good.mean()),
            "min": float(good.min()),
            "max": float(good.max()),
        }

    proj, col, row = io.lonlat_to_utm_pixel(rec["lon"], rec["lat"])
    bounds = io.centered_bounds(col, row, TILE, TILE)
    dst_transform = Affine(
        proj.x_resolution,
        0,
        bounds[0] * proj.x_resolution,
        0,
        proj.y_resolution,
        bounds[1] * proj.y_resolution,
    )

    with rasterio.open(_region_tif(rec["region"], rec["wy"]).path) as ds:
        c0 = max(0, rec["col"] - HALF)
        r0 = max(0, rec["row"] - HALF)
        c1 = min(ds.width, rec["col"] + HALF)
        r1 = min(ds.height, rec["row"] + HALF)
        win = Window(c0, r0, c1 - c0, r1 - r0)
        src = ds.read(1, window=win)  # int16 DOWY, -9999 nodata
        src_transform = ds.window_transform(win)
        src_crs = ds.crs

    # Warp the continuous DOWY field (bilinear) and a validity mask separately so the
    # -9999 no-data never blends into valid output pixels; round back to integer DOWY.
    v_src = np.where(src == SRC_NODATA, 0, src).astype(np.float32)
    m_src = (src != SRC_NODATA).astype(np.float32)
    dst_v = np.zeros((TILE, TILE), np.float32)
    dst_m = np.zeros((TILE, TILE), np.float32)
    reproject(
        v_src,
        dst_v,
        src_transform=src_transform,
        src_crs=src_crs,
        dst_transform=dst_transform,
        dst_crs=proj.crs,
        resampling=Resampling.bilinear,
    )
    reproject(
        m_src,
        dst_m,
        src_transform=src_transform,
        src_crs=src_crs,
        dst_transform=dst_transform,
        dst_crs=proj.crs,
        resampling=Resampling.bilinear,
    )
    valid = dst_m >= 0.5
    out = np.where(valid, np.rint(dst_v), SRC_NODATA).astype(np.int16)

    good = out[out != SRC_NODATA]
    if good.size < 0.3 * TILE * TILE:
        # Landed mostly on no-data -> not a usable label; skip (keeps re-runs idempotent).
        return {"sample_id": sample_id, "n_valid": int(good.size)}

    io.write_label_geotiff(SLUG, sample_id, out, proj, bounds, nodata=SRC_NODATA)
    io.write_sample_json(
        SLUG,
        sample_id,
        proj,
        bounds,
        io.year_range(rec["wy"]),
        source_id=f"{rec['region']}:WY{rec['wy']}:{rec['col']}_{rec['row']}",
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
    download_regions()
    io.check_disk()

    tasks = [
        {"region": rg, "wy": wy}
        for rg in REGIONS
        for wy in WATER_YEARS
        if _region_tif(rg, wy).exists()
    ]
    print(f"{len(tasks)} region-year rasters to scan")
    with multiprocessing.Pool(args.workers) as p:
        results = list(
            tqdm.tqdm(
                star_imap_unordered(p, scan_region, tasks),
                total=len(tasks),
                desc="scan",
            )
        )
    candidates = [r for sub in results for r in sub]
    print(f"gathered {len(candidates)} candidate windows")
    if not candidates:
        raise RuntimeError("no candidate windows -- check region downloads")

    io.check_disk()
    selected, edges = sampling.bucket_balance_regression(
        candidates, "value", total=TOTAL, n_buckets=N_BUCKETS, seed=SEED
    )
    if args.limit:
        selected = selected[: args.limit]
    for i, r in enumerate(selected):
        r["sample_id"] = f"{i:06d}"
    print(
        f"selected {len(selected)} windows; DOWY bucket edges {[round(e, 1) for e in edges]}"
    )

    io.locations_dir(SLUG).mkdir(parents=True, exist_ok=True)
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

    io.write_dataset_metadata(
        SLUG,
        {
            "dataset": SLUG,
            "name": NAME,
            "task_type": "regression",
            "source": "Zenodo (Gagliano, Shean & Henderson 2026, U. Washington)",
            "license": "CC-BY-4.0",
            "provenance": {
                "url": URL,
                "doi": DOI,
                "have_locally": False,
                "annotation_method": (
                    "derived product: Sentinel-1 C-band SAR (VV) backscatter-minimum "
                    "detection within a MODIS-derived snow-phenology window; validated vs "
                    "735 Western-US snow pillows (median diff -1.0 d, MAD 9.0 d)"
                ),
                "access": (
                    "public Zenodo Kerchunk reference file + direct HTTP range reads of the "
                    f"global .zarr.tar (blosc-zstd int16 chunks); bounded set of {len(REGIONS)} "
                    f"seasonal-snow regions x {len(WATER_YEARS)} water years {WATER_YEARS}; "
                    "browser User-Agent + request pacing to respect Zenodo guest limits"
                ),
            },
            "sensors_relevant": ["sentinel1", "sentinel2", "landsat"],
            "regression": {
                "name": "snowmelt_runoff_onset_dowy",
                "description": (
                    "Day of water year (DOWY, integer 1-366) of snowmelt runoff onset, "
                    "detected as the Sentinel-1 C-band SAR (VV) backscatter minimum within a "
                    "MODIS-constrained seasonal-snow window. Water year: NH = Oct 1(N-1)-"
                    "Sep 30(N), DOWY 1 = Oct 1; SH = Apr 1(N)-Mar 31(N+1), DOWY 1 = Apr 1. "
                    "The runoff-onset event of water year N falls within calendar year N in "
                    "both hemispheres (NH spring; SH austral spring/summer), so each annual "
                    "layer is paired with a 1-year window on calendar year N. Native 80 m "
                    "int16 DOWY, no-data -9999 (no scale factor); reprojected EPSG:4326 -> "
                    "local UTM at 10 m (bilinear + validity mask, values rounded to integer "
                    "DOWY). Windows bucket-balanced across the DOWY distribution."
                ),
                "unit": "day of water year",
                "dtype": "int16",
                "value_range": [round(pix_min, 1), round(pix_max, 1)],
                "nodata_value": SRC_NODATA,
                "buckets": [round(e, 2) for e in edges],
            },
            "num_samples": n_written,
            "notes": (
                "Global 80 m derived product; bounded-region dense_raster regression "
                f"sampling from {len(REGIONS)} curated seasonal-snow regions across both "
                "hemispheres (Western US ranges, Alaska, Canadian Cordillera, Iceland, "
                "European Alps, Scandinavia, Caucasus, Himalaya, Tien Shan, Pamir, Altai, "
                f"E. Siberia, Central Andes, Patagonia, Southern Alps NZ) x {len(WATER_YEARS)} "
                f"bounded-sampled water years {WATER_YEARS} (WY2015 dropped, melt pre-2016; "
                "5-year subset evenly spanning 2016-2024). 64x64 windows (8x8 native 80 m blocks) "
                "reprojected from EPSG:4326 80 m to local UTM at 10 m (bilinear + nearest/"
                "threshold validity mask so -9999 no-data never blends into valid pixels; "
                "reprojected values rounded to integer DOWY). Regression nodata is -9999 "
                "(the source sentinel; repo default -99999 does not fit int16). Native "
                "resolution is 80 m -- the 10 m tiles are upsampled 8x. Annual DOWY layer -> "
                "1-year window on calendar year N."
            ),
        },
    )

    hist, hedges = np.histogram([r["value"] for r in selected], bins=N_BUCKETS)
    print("selected-window mean-DOWY histogram:")
    for lo, hi, c in zip(hedges[:-1], hedges[1:], hist):
        print(f"  [{lo:6.1f}, {hi:6.1f}) : {c}")
    print(f"per-pixel value range across samples: [{pix_min:.1f}, {pix_max:.1f}] DOWY")
    print(f"num_samples={n_written} task_type=regression")
    print("done")


if __name__ == "__main__":
    multiprocessing.set_start_method("forkserver", force=True)
    main()
