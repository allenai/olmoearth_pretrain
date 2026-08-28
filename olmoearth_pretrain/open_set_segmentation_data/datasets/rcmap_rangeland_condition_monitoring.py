"""Process RCMAP sagebrush fractional cover into open-set regression label patches.

Source: RCMAP (Rangeland Condition Monitoring Assessment and Projection) Fractional
Component Time-Series (USGS / MRLC), https://www.mrlc.gov/data . RCMAP maps the per-pixel
percent cover (0-100) of ten rangeland components (annual herbaceous, bare ground,
herbaceous, litter, non-sagebrush shrub, perennial herbaceous, sagebrush, shrub, tree,
shrub height) across western North America at 30 m, one map per year (1985-present),
derived from Landsat via regression trained on field plots. Data DOI 10.5066/P13QF8HT
(V7, 1985-2024); the live MRLC bundles served here are the current generation
(2015-2025 interval used for the Sentinel-era subset).

This is a *regression* dataset (continuous per-pixel percent cover). Following the spec's
"pick one primary component" guidance for multi-component fractional products, we regress
the **sagebrush** component -- RCMAP's flagship/namesake product (the project exists to
monitor sagebrush ecosystems). Sagebrush cover is heavily zero-inflated (most of the
mapped extent is not sagebrush steppe), so we **bucket-balance across fixed cover buckets**
to give the label bank a usable spread of cover levels rather than mostly-zero tiles.

Large regional derived-product raster with no in-situ reference alternative -> BOUNDED-TILE
sampling: we download only the current sagebrush decade bundle and draw <=5000 tiles from a
diverse set of years spanning the Sentinel era.

Output: single-band float32 GeoTIFFs, local UTM, 10 m/pixel, 64x64 (~640 m), nodata -99999
(io.REGRESSION_NODATA). One <=1-year time range per tile (the RCMAP product year).
"""

import argparse
import multiprocessing
import zipfile
from collections import Counter
from typing import Any

import numpy as np
import rasterio
import tqdm
from rasterio.enums import Resampling
from rslearn.utils.mp import star_imap_unordered
from rslearn.utils.raster_format import GeotiffRasterFormat

from olmoearth_pretrain.open_set_segmentation_data import io

SLUG = "rcmap_rangeland_condition_monitoring"
NAME = "RCMAP (Rangeland Condition Monitoring)"
URL = "https://www.mrlc.gov/data"
DOI = "https://doi.org/10.5066/P13QF8HT"
COMPONENT = "sagebrush"
ZIP_NAME = "Sagebrush_2015_2025.zip"

TILE = 64
TOTAL = 5000
# Years spanning the Sentinel era (>=2016) for temporal diversity; each within the
# manifest 1985-2024 range and present in the 2015-2025 bundle.
YEARS = [2016, 2018, 2020, 2022, 2024]

# Fixed percent-cover bucket edges for the zero-inflated sagebrush distribution. Right
# edge 101 makes the last bucket [30, 100]. Balancing over these gives an even spread of
# cover levels instead of a corpus dominated by 0% tiles.
BUCKET_EDGES = [0, 1, 5, 10, 20, 30, 101]
VALID_MAX = 100  # values 0..100 are valid percent cover; anything else is mask/nodata
DECIM = 21  # 30 m source * 21 ~= 630 m ~ one candidate per 640 m tile
CAND_PER_YEAR = 60000
CAND_SEED = 42


def tif_dir() -> Any:
    return io.raw_dir(SLUG) / "tifs"


def member_for_year(names: list[str], year: int) -> str | None:
    """Find the zip member (a .tif) for a given year."""
    cands = [n for n in names if n.lower().endswith(".tif") and str(year) in n]
    if not cands:
        return None
    # Prefer the shortest / most specific match.
    return sorted(cands, key=len)[0]


def year_tif_path(year: int) -> Any:
    return tif_dir() / f"rcmap_{COMPONENT}_{year}.tif"


def extract_years() -> dict[int, str]:
    """Extract the per-year sagebrush GeoTIFFs we need from the bundle zip."""
    zpath = io.raw_dir(SLUG) / ZIP_NAME
    tif_dir().mkdir(parents=True, exist_ok=True)
    out: dict[int, str] = {}
    with zipfile.ZipFile(zpath.path) as z:
        names = z.namelist()
        for year in YEARS:
            dst = year_tif_path(year)
            if dst.exists():
                out[year] = str(dst)
                continue
            member = member_for_year(names, year)
            if member is None:
                print(f"[warn] no member for year {year}")
                continue
            print(f"extracting {member} -> {dst}")
            tmp = dst.parent / (dst.name + ".tmp")
            with z.open(member) as src, tmp.open("wb") as f:
                while True:
                    chunk = src.read(1 << 20)
                    if not chunk:
                        break
                    f.write(chunk)
            tmp.rename(dst)
            out[year] = str(dst)
    return out


# ---------------------------------------------------------------------------
# Candidate sampling (decimated read of each year raster)
# ---------------------------------------------------------------------------
def _candidates_one(year: int) -> list[dict[str, Any]]:
    from pyproj import Transformer

    path = year_tif_path(year)
    rng = np.random.default_rng(abs(hash((year, CAND_SEED))) % (2**32))
    with rasterio.open(path.path) as ds:
        src_crs = ds.crs
        transform = ds.transform
        nodata = ds.nodata
        oh = max(1, ds.height // DECIM)
        ow = max(1, ds.width // DECIM)
        arr = ds.read(1, out_shape=(oh, ow), resampling=Resampling.nearest)
        dec_tf = transform * rasterio.Affine.scale(ds.width / ow, ds.height / oh)
    valid = (arr >= 0) & (arr <= VALID_MAX)
    if nodata is not None:
        valid &= arr != nodata
    rows, cols = np.where(valid)
    if rows.size == 0:
        return []
    if rows.size > CAND_PER_YEAR:
        sel = rng.choice(rows.size, size=CAND_PER_YEAR, replace=False)
        rows, cols = rows[sel], cols[sel]
    xs, ys = dec_tf * (cols + 0.5, rows + 0.5)
    xs = np.asarray(xs, dtype=np.float64)
    ys = np.asarray(ys, dtype=np.float64)
    vals = arr[rows, cols].astype(np.float64)
    # Vectorized reprojection of source-CRS (Albers) pixel centers to WGS84 lon/lat.
    tf = Transformer.from_crs(src_crs, "EPSG:4326", always_xy=True)
    lons, lats = tf.transform(xs, ys)
    return [
        {
            "lon": float(lo),
            "lat": float(la),
            "value": float(v),
            "year": year,
            "source_id": f"{COMPONENT}_{year}",
        }
        for lo, la, v in zip(lons, lats, vals)
    ]


def gather_candidates(workers: int) -> list[dict[str, Any]]:
    jobs = [dict(year=y) for y in YEARS]
    out: list[dict[str, Any]] = []
    with multiprocessing.Pool(min(workers, len(jobs))) as p:
        for recs in tqdm.tqdm(
            star_imap_unordered(p, _candidates_one, jobs),
            total=len(jobs),
            desc="candidates",
        ):
            out.extend(recs)
    return out


def bucket_balance_fixed(
    records: list[dict[str, Any]], edges: list[int], total: int, seed: int = 42
) -> list[dict[str, Any]]:
    """Balance across fixed [edge_i, edge_{i+1}) value buckets (zero-inflated data).

    Take up to total//n_buckets from each bucket; then top up from leftover records
    (bucket order) until ``total`` is reached or candidates run out.
    """
    import random

    n = len(edges) - 1
    buckets: list[list[dict[str, Any]]] = [[] for _ in range(n)]
    for r in records:
        v = r["value"]
        b = int(np.searchsorted(edges, v, side="right")) - 1
        b = min(max(b, 0), n - 1)
        buckets[b].append(r)
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


# ---------------------------------------------------------------------------
# Tile writing
# ---------------------------------------------------------------------------
def _write_one(rec: dict[str, Any]) -> dict[str, Any] | None:
    sample_id = rec["sample_id"]
    year = rec["year"]
    tif_path = io.locations_dir(SLUG) / f"{sample_id}.tif"
    if tif_path.exists():
        # Idempotent skip: read back existing tile so metadata stats stay correct.
        with rasterio.open(tif_path.path) as ds:
            ev = ds.read(1)
        good = ev[ev != io.REGRESSION_NODATA]
        if good.size == 0:
            return {"sample_id": sample_id, "year": year, "n_valid": 0}
        return {
            "sample_id": sample_id,
            "year": year,
            "n_valid": int(good.size),
            "mean": float(good.mean()),
            "min": float(good.min()),
            "max": float(good.max()),
        }
    lon, lat = rec["lon"], rec["lat"]
    proj, col, row = io.lonlat_to_utm_pixel(lon, lat)
    bounds = io.centered_bounds(col, row, TILE, TILE)

    src_dir = tif_dir()
    fname = year_tif_path(year).name
    with rasterio.open((src_dir / fname).path) as ds:
        src_nodata = ds.nodata
    ra = GeotiffRasterFormat().decode_raster(
        src_dir, proj, bounds, resampling=Resampling.bilinear, fname=fname
    )
    vals = ra.array[0].astype(np.float32)  # (H, W), percent cover 0..100
    invalid = (~np.isfinite(vals)) | (vals < 0) | (vals > VALID_MAX)
    if src_nodata is not None:
        invalid |= np.abs(vals - float(src_nodata)) < 0.5
    vals[invalid] = io.REGRESSION_NODATA

    io.write_label_geotiff(
        SLUG, sample_id, vals, proj, bounds, nodata=io.REGRESSION_NODATA
    )
    io.write_sample_json(
        SLUG, sample_id, proj, bounds, io.year_range(year), source_id=rec["source_id"]
    )

    good = vals[vals != io.REGRESSION_NODATA]
    if good.size == 0:
        return {"sample_id": sample_id, "year": year, "n_valid": 0}
    return {
        "sample_id": sample_id,
        "year": year,
        "n_valid": int(good.size),
        "mean": float(good.mean()),
        "min": float(good.min()),
        "max": float(good.max()),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=64)
    args = parser.parse_args()

    io.check_disk()
    extract_years()
    io.check_disk()

    cands = gather_candidates(args.workers)
    print(f"gathered {len(cands)} candidate points across years {YEARS}")

    selected = bucket_balance_fixed(cands, BUCKET_EDGES, TOTAL, seed=CAND_SEED)
    for i, r in enumerate(selected):
        r["sample_id"] = f"{i:06d}"
    sel_vals = np.array([r["value"] for r in selected], dtype=np.float64)
    bucket_counts = Counter(
        min(
            max(int(np.searchsorted(BUCKET_EDGES, v, side="right")) - 1, 0),
            len(BUCKET_EDGES) - 2,
        )
        for v in sel_vals
    )
    print(
        f"selected {len(selected)} tiles; center-value bucket counts {dict(sorted(bucket_counts.items()))}"
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

    year_counts = Counter(r["year"] for r in selected)
    valid_stats = [s for s in stats if s.get("n_valid", 0) > 0]
    pix_min = min((s["min"] for s in valid_stats), default=0.0)
    pix_max = max((s["max"] for s in valid_stats), default=0.0)

    io.write_dataset_metadata(
        SLUG,
        {
            "dataset": SLUG,
            "name": NAME,
            "task_type": "regression",
            "source": "USGS / MRLC (RCMAP)",
            "license": "public domain (US Government work)",
            "provenance": {
                "url": URL,
                "doi": DOI,
                "have_locally": False,
                "annotation_method": "Landsat regression trained on field plots (RCMAP)",
            },
            "sensors_relevant": ["sentinel2", "sentinel1", "landsat"],
            "regression": {
                "name": "sagebrush_cover",
                "description": (
                    "Per-pixel percent cover of sagebrush (Artemisia spp.) canopy from the "
                    "RCMAP fractional-component time-series, a Landsat-derived regression "
                    "trained on field plots across western North America (30 m native). "
                    "The sagebrush component is RCMAP's primary/namesake product. Values are "
                    "0-100 percent; the distribution is heavily zero-inflated and was "
                    "bucket-balanced across fixed cover buckets."
                ),
                "unit": "percent cover",
                "dtype": "float32",
                "value_range": [round(pix_min, 3), round(pix_max, 3)],
                "nodata_value": io.REGRESSION_NODATA,
                "buckets": BUCKET_EDGES,
            },
            "num_samples": len(selected),
            "year_counts": dict(sorted(year_counts.items())),
            "notes": (
                "One primary component (sagebrush) chosen from RCMAP's ten fractional "
                "components per the multi-component regression guidance. Bounded-tile "
                "sampling from the current MRLC sagebrush decade bundle (2015-2025); tiles "
                "drawn from years " + str(YEARS) + " for temporal diversity within the "
                "manifest range. 64x64 tiles at 10 m in local UTM (~640 m), source 30 m "
                "Albers (EPSG:5070) reprojected via bilinear resampling. Distribution is "
                "zero-inflated; bucket-balanced across fixed cover buckets "
                + str(BUCKET_EDGES)
                + " percent."
            ),
        },
    )

    hist_edges = [0, 1, 5, 10, 20, 30, 50, 100.0001]
    hist, _ = np.histogram(sel_vals, bins=hist_edges)
    print("selected-tile center-value histogram (percent sagebrush cover):")
    for lo, hi, c in zip(hist_edges[:-1], hist_edges[1:], hist):
        print(f"  [{lo:>6}, {hi:>6}) : {c}")
    print(f"year counts: {dict(sorted(year_counts.items()))}")
    print(f"per-pixel value range across tiles: [{pix_min:.3f}, {pix_max:.3f}] percent")
    print(f"num_samples={len(selected)} task_type=regression")
    print("done")


if __name__ == "__main__":
    multiprocessing.set_start_method("forkserver", force=True)
    main()
