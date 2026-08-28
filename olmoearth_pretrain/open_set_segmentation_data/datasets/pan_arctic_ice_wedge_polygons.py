"""Pan-Arctic Ice-Wedge Polygons -> open-set-segmentation regression label patches.

Source: Witharana, Liljedahl et al., "Ice-wedge polygon detection in satellite imagery
from pan-Arctic regions, Permafrost Discovery Gateway, 2001-2021", NSF Arctic Data Center
(https://doi.org/10.18739/A2KW57K57), CC-BY-4.0. A deep-learning (MAPLE / CNN) inventory of
>1 billion individual ice-wedge polygons detected in very-high-resolution (Maxar, ~0.5 m)
commercial satellite imagery across the pan-Arctic tundra. The polygon vectors are also
rasterized to an **ice-wedge-polygon coverage-density** raster: each cell's value is the
fraction of the cell area occupied by ice-wedge polygons.

Why regression (not the manifest's low-/high-centered classes): individual polygons are
~10-20 m and are NOT reliably resolvable as objects at 10-30 m Sentinel-2 / Landsat, and the
publicly-served raster product is a **single-band coverage-density** layer (it does not carry
a per-cell low-centered vs high-centered microtopography split; that attribute lives only in
the per-polygon geopackage). Per the manifest note ("Use rasterized density at S2/Landsat
scale"), we therefore build a per-pixel **regression** target = ice-wedge-polygon coverage
density (fraction 0-1) at 10 m -- capturing polygon presence/density, which IS observable as
patterned-ground texture at S2/Landsat scale.

Access / bounded download (this is a huge, >1B-object pan-Arctic product -- we do NOT bulk
download it): the coverage-density GeoTIFFs are published as a WorldCRS1984Quad tile pyramid
(EPSG:4326) at http://arcticdata.io/data/10.18739/A2KW57K57/iwp_geotiff_high/WGS1984Quad/
{z}/{x}/{y}.tif, zoom levels 0-15. We use **zoom 14** as the density source: it is a
properly *averaged* overview at ~4.8 m/px (lat) / ~1.6 m/px (lon) at 70N, with values that
are genuine area fractions (z=15 is the ~2.4 m native level; z=13 and coarser are *summed*
overviews whose values are inflated >>1, so we avoid them). We download z=14 tiles only over
a bounded set of ~10 representative high-IWP tundra regions across Alaska, Arctic Canada, and
Siberia (§5 large-global-product bounded sampling), mosaic each region, reproject to local
UTM at 10 m with **average** resampling (continuous fraction field; nodata-aware so unmapped
gaps stay nodata), clip the fraction to [0, 1] (values >1 are duplicate-scene overlap
artifacts), and cut 64x64 (~640 m) windows.

Output: single-band float32 GeoTIFFs, local UTM, 10 m/pixel, 64x64, nodata -99999
(io.REGRESSION_NODATA). Windows are bucket-balanced across fixed density buckets (the density
distribution is heavily zero-inflated) to up to 5000 samples. The inventory is a multi-year
(2001-2021, mostly 2016-2021) composite of a persistent geomorphic landform; we anchor each
sample to a representative 1-year Sentinel-era window (2020).
"""

import argparse
import math
import multiprocessing
import re
import urllib.error
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import numpy as np
import rasterio
import tqdm
from affine import Affine
from pyproj import Transformer
from rasterio.warp import Resampling, reproject
from rslearn.utils.geometry import Projection
from rslearn.utils.mp import star_imap_unordered

from olmoearth_pretrain.open_set_segmentation_data import download, io, manifest

SLUG = "pan_arctic_ice_wedge_polygons"
NAME = "Pan-Arctic Ice-Wedge Polygons"
URL = "https://doi.org/10.18739/A2KW57K57"
BASE = "https://arcticdata.io/data/10.18739/A2KW57K57/iwp_geotiff_high/WGS1984Quad"

Z = 14  # density source zoom (averaged overview, ~4.8 m/px lat @70N)
TILESPAN = 360.0 / (
    2 ** (Z + 1)
)  # tile side in degrees (square in deg for WGS1984Quad)
PX = TILESPAN / 256.0  # source degrees/pixel
TILE = 64  # output tile size (px); 10 m => ~640 m
TOTAL = 5000  # regression per-dataset target (<= 25k cap)
YEAR = 2020  # representative Sentinel-era 1-year window
MIN_VALID_FRAC = 0.98  # require windows essentially fully within mapped ground
# Fixed coverage-density buckets for the zero-inflated fraction distribution (right edge
# 1.0001 => last bucket [0.5, 1.0]). Balancing over these gives an even spread of density.
BUCKET_EDGES = [0.0, 0.02, 0.05, 0.10, 0.20, 0.35, 0.50, 1.0001]
SEED = 42

# Bounded set of representative high-IWP tundra regions (name, center_lon, center_lat).
# Half-widths below define each bbox. Coverage verified present on the server.
REGION_HALF_LON = 0.20
REGION_HALF_LAT = 0.10
REGIONS = [
    ("alaska_prudhoe", -148.70, 70.20),
    ("alaska_utqiagvik", -156.60, 71.30),
    ("alaska_teshekpuk", -153.20, 70.65),
    ("canada_tuktoyaktuk", -132.90, 69.52),
    ("canada_banks", -122.20, 71.92),
    ("canada_mackenzie", -134.20, 69.42),
    ("russia_lena", 126.10, 72.42),
    ("russia_yamal", 68.80, 70.15),
    ("russia_kolyma", 160.80, 69.42),
    ("russia_indigirka", 147.50, 70.80),
]


def tile_index(lon: float, lat: float) -> tuple[int, int]:
    """WorldCRS1984Quad (x, y) tile index at zoom Z for a lon/lat."""
    x = int((lon + 180.0) / 360.0 * (2 ** (Z + 1)))
    y = int((90.0 - lat) / 180.0 * (2**Z))
    return x, y


def _list_x_tiles(x: int, y0: int, y1: int) -> list[tuple[int, int]]:
    """List available (x, y) tiles in column x within [y0, y1] via HTTP dir listing."""
    url = f"{BASE}/{Z}/{x}/"
    try:
        with urllib.request.urlopen(url, timeout=60) as r:
            html = r.read().decode("utf-8", "replace")
    except urllib.error.HTTPError:
        return []
    except Exception:
        return []
    out = []
    for m in re.findall(r'href="(\d+)\.tif"', html):
        y = int(m)
        if y0 <= y <= y1:
            out.append((x, y))
    return out


def discover_region_tiles(clon: float, clat: float) -> list[tuple[int, int]]:
    """Enumerate present z=14 tiles inside a region bbox around (clon, clat)."""
    lon0, lon1 = clon - REGION_HALF_LON, clon + REGION_HALF_LON
    lat0, lat1 = clat - REGION_HALF_LAT, clat + REGION_HALF_LAT
    x_a, _ = tile_index(lon0, lat1)
    x_b, _ = tile_index(lon1, lat1)
    _, y_a = tile_index(lon0, lat1)  # smaller y = north (lat1)
    _, y_b = tile_index(lon0, lat0)
    x0, x1 = min(x_a, x_b), max(x_a, x_b)
    y0, y1 = min(y_a, y_b), max(y_a, y_b)
    tiles: list[tuple[int, int]] = []
    with ThreadPoolExecutor(32) as ex:
        for part in ex.map(lambda x: _list_x_tiles(x, y0, y1), range(x0, x1 + 1)):
            tiles.extend(part)
    return tiles


def _dl_tile(x: int, y: int) -> tuple[int, int] | None:
    dst = io.raw_dir(SLUG) / "tiles" / str(Z) / str(x) / f"{y}.tif"
    try:
        download.download_http(f"{BASE}/{Z}/{x}/{y}.tif", dst, timeout=180)
        return (x, y)
    except Exception:
        return None


def download_region(tiles: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Download all present tiles for a region (idempotent). Returns tiles on disk."""
    ok: list[tuple[int, int]] = []
    with ThreadPoolExecutor(64) as ex:
        for res in ex.map(lambda t: _dl_tile(*t), tiles):
            if res is not None:
                ok.append(res)
    return ok


def build_mosaic(tiles: list[tuple[int, int]]) -> tuple[np.ndarray, Affine]:
    """Assemble a region's z=14 tiles into an EPSG:4326 float64 mosaic (nodata -9999)."""
    xs = [t[0] for t in tiles]
    ys = [t[1] for t in tiles]
    x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
    W = (x1 - x0 + 1) * 256
    H = (y1 - y0 + 1) * 256
    mosaic = np.full((H, W), -9999.0, dtype=np.float64)
    for x, y in tiles:
        p = io.raw_dir(SLUG) / "tiles" / str(Z) / str(x) / f"{y}.tif"
        try:
            with rasterio.open(p.path) as ds:
                a = ds.read(1).astype(np.float64)
                nd = ds.nodata
        except Exception:
            continue
        if a.shape != (256, 256):
            continue
        if nd is not None:
            a = np.where(a == nd, -9999.0, a)
        r0 = (y - y0) * 256
        c0 = (x - x0) * 256
        mosaic[r0 : r0 + 256, c0 : c0 + 256] = a
    lon_left = -180.0 + x0 * TILESPAN
    top = 90.0 - y0 * TILESPAN
    src_tf = Affine(PX, 0, lon_left, 0, -PX, top)
    return mosaic, src_tf


def reproject_region(
    mosaic: np.ndarray, src_tf: Affine, clon: float, clat: float
) -> tuple[np.ndarray, Projection, int, int]:
    """Reproject a region mosaic to local UTM at 10 m (average, nodata-aware).

    Returns (density_utm, projection, col_min, row_min) where col_min/row_min are the
    global rslearn pixel-grid indices of the destination array's top-left pixel (so window
    pixel_bounds line up with io.lonlat_to_utm_pixel / centered_bounds).
    """
    proj = io.utm_projection_for_lonlat(clon, clat)
    H, W = mosaic.shape
    # Corner lon/lats of the mosaic extent -> UTM easting/northing to size the dst grid.
    lon_left, top = src_tf.c, src_tf.f
    lon_right = lon_left + W * PX
    bottom = top - H * PX
    tf = Transformer.from_crs("EPSG:4326", proj.crs, always_xy=True)
    corner_lons = [lon_left, lon_right, lon_left, lon_right]
    corner_lats = [top, top, bottom, bottom]
    es, ns = tf.transform(corner_lons, corner_lats)
    e_min, e_max = min(es), max(es)
    n_min, n_max = min(ns), max(ns)
    col_min = int(math.floor(e_min / io.RESOLUTION))
    col_max = int(math.ceil(e_max / io.RESOLUTION))
    # rslearn y_resolution = -10 => pixel row = -northing/10; north (max N) => min row.
    row_min = int(math.floor(-n_max / io.RESOLUTION))
    row_max = int(math.ceil(-n_min / io.RESOLUTION))
    out_w = col_max - col_min
    out_h = row_max - row_min
    dst = np.full((out_h, out_w), -9999.0, dtype=np.float64)
    dst_tf = Affine(
        io.RESOLUTION,
        0,
        col_min * io.RESOLUTION,
        0,
        -io.RESOLUTION,
        row_min * (-io.RESOLUTION),
    )
    reproject(
        source=mosaic,
        destination=dst,
        src_transform=src_tf,
        src_crs="EPSG:4326",
        src_nodata=-9999.0,
        dst_transform=dst_tf,
        dst_crs=proj.crs,
        dst_nodata=-9999.0,
        resampling=Resampling.average,
    )
    return dst, proj, col_min, row_min


def windows_from_region(
    region: str, dst: np.ndarray, proj: Projection, col_min: int, row_min: int
) -> list[dict[str, Any]]:
    """Cut non-overlapping 64x64 windows; keep essentially-fully-mapped ones."""
    H, W = dst.shape
    crs_str = proj.crs.to_string()
    recs: list[dict[str, Any]] = []
    for i0 in range(0, H - TILE + 1, TILE):
        for j0 in range(0, W - TILE + 1, TILE):
            block = dst[i0 : i0 + TILE, j0 : j0 + TILE]
            valid = np.isfinite(block) & (block != -9999.0)
            vf = float(valid.mean())
            if vf < MIN_VALID_FRAC:
                continue
            clipped = np.clip(block, 0.0, 1.0).astype(np.float32)
            mean_density = float(clipped[valid].mean()) if valid.any() else 0.0
            arr = np.where(valid, clipped, np.float32(io.REGRESSION_NODATA)).astype(
                np.float32
            )
            x_min = col_min + j0
            y_min = row_min + i0
            recs.append(
                {
                    "region": region,
                    "crs": crs_str,
                    "bounds": (x_min, y_min, x_min + TILE, y_min + TILE),
                    "value": mean_density,
                    "arr": arr,
                    "source_id": f"{region}/z{Z}/px_{i0}_{j0}",
                }
            )
    return recs


def bucket_balance_fixed(
    records: list[dict[str, Any]], edges: list[float], total: int, seed: int = SEED
) -> list[dict[str, Any]]:
    """Balance across fixed [edge_i, edge_{i+1}) value buckets (zero-inflated density)."""
    import random

    n = len(edges) - 1
    buckets: list[list[dict[str, Any]]] = [[] for _ in range(n)]
    for r in records:
        b = int(np.searchsorted(edges, r["value"], side="right")) - 1
        buckets[min(max(b, 0), n - 1)].append(r)
    rng = random.Random(seed)
    for b in buckets:
        b.sort(key=lambda r: r["source_id"])  # deterministic before shuffle
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

    from rasterio.crs import CRS

    proj = Projection(CRS.from_string(rec["crs"]), io.RESOLUTION, -io.RESOLUTION)
    bounds = tuple(rec["bounds"])
    arr = rec["arr"]
    io.write_label_geotiff(
        SLUG, sample_id, arr, proj, bounds, nodata=io.REGRESSION_NODATA
    )
    io.write_sample_json(
        SLUG, sample_id, proj, bounds, io.year_range(YEAR), source_id=rec["source_id"]
    )
    good = arr[arr != io.REGRESSION_NODATA]
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
    manifest.write_registry_entry(SLUG, "in_progress")

    raw = io.raw_dir(SLUG)
    raw.mkdir(parents=True, exist_ok=True)
    with (raw / "SOURCE.txt").open("w") as f:
        f.write(
            "Pan-Arctic Ice-Wedge Polygon coverage-density GeoTIFF tile pyramid\n"
            f"{BASE}/{{z}}/{{x}}/{{y}}.tif (WorldCRS1984Quad, EPSG:4326)\n"
            f"Bounded sample: zoom {Z}, {len(REGIONS)} representative regions.\n"
        )

    all_recs: list[dict[str, Any]] = []
    for region, clon, clat in REGIONS:
        io.check_disk()
        tiles = discover_region_tiles(clon, clat)
        if not tiles:
            print(f"[{region}] no tiles found; skipping")
            continue
        ok = download_region(tiles)
        print(f"[{region}] {len(ok)}/{len(tiles)} tiles on disk")
        if not ok:
            continue
        mosaic, src_tf = build_mosaic(ok)
        dst, proj, col_min, row_min = reproject_region(mosaic, src_tf, clon, clat)
        recs = windows_from_region(region, dst, proj, col_min, row_min)
        print(f"[{region}] {len(recs)} fully-mapped 64x64 candidate windows")
        all_recs.extend(recs)

    print(f"total candidate windows: {len(all_recs)}")
    if not all_recs:
        raise RuntimeError("no candidate windows produced")

    selected = bucket_balance_fixed(all_recs, BUCKET_EDGES, TOTAL, seed=SEED)
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
    reg_counts = Counter(r["region"] for r in selected)
    print(f"selected {len(selected)} windows")
    print(f"  density-bucket counts (by window-mean): {dict(sorted(bcounts.items()))}")
    print(f"  per-region counts: {dict(sorted(reg_counts.items()))}")

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

    io.write_dataset_metadata(
        SLUG,
        {
            "dataset": SLUG,
            "name": NAME,
            "task_type": "regression",
            "source": "NSF Arctic Data Center / Permafrost Discovery Gateway",
            "license": "CC-BY-4.0",
            "provenance": {
                "url": URL,
                "have_locally": False,
                "annotation_method": (
                    "Deep learning (MAPLE / CNN) on very-high-resolution (~0.5 m Maxar) "
                    "commercial satellite imagery; hand-annotated training. Polygon vectors "
                    "rasterized to coverage density."
                ),
                "access": (
                    f"bounded HTTP tile reads from the WorldCRS1984Quad pyramid {BASE}/"
                    f"{{z}}/{{x}}/{{y}}.tif at zoom {Z} over "
                    f"{len(REGIONS)} representative pan-Arctic regions"
                ),
            },
            "sensors_relevant": ["sentinel2", "sentinel1", "landsat"],
            "regression": {
                "name": "ice_wedge_polygon_coverage_density",
                "description": (
                    "Per-pixel fractional coverage (0-1) of ice-wedge polygons at 10 m, from "
                    "the pan-Arctic ice-wedge-polygon inventory (>1 billion polygons detected "
                    "by deep learning on ~0.5 m VHR imagery, 2001-2021; mostly 2016-2021). "
                    "Each cell value is the fraction of the cell area occupied by detected "
                    "ice-wedge polygons. Sourced from the published coverage-density GeoTIFF "
                    "pyramid (zoom 14, an averaged overview at ~4.8 m/px), reprojected to "
                    "local UTM 10 m with average resampling; fraction clipped to [0, 1] "
                    "(values >1 in the source are duplicate-scene overlap artifacts). "
                    "Individual polygons (~10-20 m) are not resolvable as objects at S2/Landsat "
                    "scale, so density is regressed instead; the manifest's low-/high-centered "
                    "microtopography split is not available in the raster product. The "
                    "distribution is heavily zero-inflated; windows were bucket-balanced across "
                    "fixed density buckets for an even spread of density levels."
                ),
                "unit": "fraction (0-1)",
                "dtype": "float32",
                "value_range": [round(pix_min, 4), round(pix_max, 4)],
                "nodata_value": io.REGRESSION_NODATA,
                "buckets": BUCKET_EDGES,
            },
            "num_samples": n_written,
            "regions": [r[0] for r in REGIONS],
            "notes": (
                "Bounded sampling of a >1B-object pan-Arctic product: zoom-14 coverage-density "
                "tiles over 10 representative high-IWP tundra regions (Alaska: prudhoe, "
                "utqiagvik, teshekpuk; Arctic Canada: tuktoyaktuk, banks, mackenzie; Siberia: "
                "lena, yamal, kolyma, indigirka). Regression = ice-wedge-polygon coverage "
                "density (fraction 0-1). Reprojected to local UTM 10 m via nodata-aware average "
                "resampling; unmapped gaps stay nodata and only >=98%-mapped 64x64 windows are "
                "kept. Multi-year (2001-2021) composite of a persistent landform; anchored to a "
                f"1-year Sentinel-era window on {YEAR}."
            ),
        },
    )

    hist_edges = BUCKET_EDGES[:-1] + [1.0001]
    hist, _ = np.histogram(sel_vals, bins=hist_edges)
    print("selected-window mean-density histogram:")
    for lo, hi, c in zip(hist_edges[:-1], hist_edges[1:], hist):
        print(f"  [{lo:.2f}, {hi:.2f}) : {c}")
    print(
        f"per-pixel value range across tiles: [{pix_min:.4f}, {pix_max:.4f}] fraction"
    )
    print(f"num_samples={n_written} task_type=regression")

    manifest.write_registry_entry(
        SLUG, "completed", task_type="regression", num_samples=n_written
    )
    print("done")


if __name__ == "__main__":
    multiprocessing.set_start_method("forkserver", force=True)
    main()
