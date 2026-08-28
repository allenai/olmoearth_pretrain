"""Process the 30 m Snow Melt-Out Date (SMOD) dataset into open-set regression patches.

Source: "Gridded snow melt-out date (SMOD) dataset for Pyrenees, European Alps and Greater
Caucasus at 30-m spatial resolution and two periods, 1985-1996 and 2011-2022"
(Zenodo record 13151801, https://zenodo.org/doi/10.5281/zenodo.13151801, CC-BY-4.0,
published in Scientific Data). Derived from Landsat surface-reflectance time series.

Each raster is a **per-period climatology**: one value per 30 m pixel giving the mean
calendar day-of-year (calDoy) of snow melt-out over the period. There is one file per
(period x region). We use ONLY the **2011-2022** period (Sentinel-era, overlaps 2016+);
the 1985-1996 period is entirely pre-2016 and is excluded per spec (all labels pre-2016).
We use the **MASKED** float32 variant (nodata = NaN), which the authors restricted to
reliable, snow-relevant pixels (valid melt-out DOY range ~121-243, i.e. May-Aug) -- the
high-confidence subset preferred for derived-product maps (spec 4).

This is a REGRESSION dataset: we regress the melt-out DOY value directly (float32, day of
year). Global/large derived product with no in-situ reference alternative -> BOUNDED-TILE
sampling across the three mountain regions (spec 5), bucket-balanced across the DOY value
range (the distribution is right-skewed: late-melt high-alpine/glacier pixels are rare).

Melt-out date is an ANNUAL per-pixel value (a day-of-year), not a dated change event, so
there is no change_time. We anchor each tile's time_range on a **snow year** (Sep 1 -> Aug
31 of the following year, so the spring/summer melt-out falls inside the window), spread
across snow years 2016/17..2021/22 for temporal diversity within the product period.

Reprojection: source is EPSG:3035 (LAEA Europe) at 30 m. We reproject to a local UTM 10 m
grid with bilinear resampling (melt-out DOY is a smooth continuous field), cropping 64x64
(~640 m) tiles. NaN (masked) pixels become io.REGRESSION_NODATA (-99999).

Output: single-band float32 GeoTIFFs, local UTM, 10 m/pixel, 64x64, nodata -99999.

Reproduce:
    python3 -m olmoearth_pretrain.open_set_segmentation_data.datasets.\
snow_melt_out_date_european_mountains_30_m
"""

import argparse
import multiprocessing
from collections import Counter
from datetime import UTC, datetime
from typing import Any

import numpy as np
import rasterio
import tqdm
from pyproj import Transformer
from rasterio.enums import Resampling
from rslearn.utils.mp import star_imap_unordered
from rslearn.utils.raster_format import GeotiffRasterFormat

from olmoearth_pretrain.open_set_segmentation_data import io
from olmoearth_pretrain.open_set_segmentation_data.download import download_zenodo
from olmoearth_pretrain.open_set_segmentation_data.sampling import (
    bucket_balance_regression,
)

SLUG = "snow_melt_out_date_european_mountains_30_m"
NAME = "Snow Melt-Out Date, European Mountains (30 m)"
URL = "https://zenodo.org/doi/10.5281/zenodo.13151801"
ZENODO_RECORD = "13151801"

# 2011-2022 MASKED files only (Sentinel-era; high-confidence variant). 1985-1996 excluded.
REGION_FILES: dict[str, str] = {
    "PYRENE": "SMOD_2011_2022_PYRENE_MASKED.tif",
    "EUALPS": "SMOD_2011_2022_EUALPS_MASKED.tif",
    "GRTCAU": "SMOD_2011_2022_GRTCAU_MASKED.tif",
}

TILE = 64
TOTAL = 5000
N_BUCKETS = 10
# ~640 m tile / 30 m source ~= 21 px; decimate candidate scan by this to get ~1/tile.
DECIM = 21
CAND_PER_REGION = 60000
CAND_SEED = 42
# Snow years to spread tiles across (start year Y = window Sep 1 Y .. Aug 31 Y+1).
SNOW_YEARS = [2016, 2017, 2018, 2019, 2020, 2021]


def snow_year_range(start_year: int) -> tuple[datetime, datetime]:
    """A ~1-year snow-year window: Sep 1 of ``start_year`` to Aug 31 of the next year.

    The spring/summer melt-out (DOY ~121-243) falls inside this window. Length is 364
    days (<= 1 year).
    """
    return (
        datetime(start_year, 9, 1, tzinfo=UTC),
        datetime(start_year + 1, 8, 31, tzinfo=UTC),
    )


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------
def download_all() -> None:
    io.check_disk()
    io.raw_dir(SLUG).mkdir(parents=True, exist_ok=True)
    download_zenodo(
        ZENODO_RECORD, io.raw_dir(SLUG), filenames=list(REGION_FILES.values())
    )
    io.check_disk()


# ---------------------------------------------------------------------------
# Candidate sampling (decimated read of each region)
# ---------------------------------------------------------------------------
def _candidates_one(region: str) -> list[dict[str, Any]]:
    path = io.raw_dir(SLUG) / REGION_FILES[region]
    rng = np.random.default_rng(abs(hash((region, CAND_SEED))) % (2**32))
    with rasterio.open(path.path) as ds:
        oh = max(1, ds.height // DECIM)
        ow = max(1, ds.width // DECIM)
        arr = ds.read(1, out_shape=(oh, ow), resampling=Resampling.nearest)
        dec_tf = ds.transform * rasterio.Affine.scale(ds.width / ow, ds.height / oh)
        transformer = Transformer.from_crs(ds.crs, "EPSG:4326", always_xy=True)
    rows, cols = np.where(np.isfinite(arr) & (arr > 0))
    if rows.size == 0:
        return []
    xs, ys = dec_tf * (cols + 0.5, rows + 0.5)  # EPSG:3035 pixel-center coords
    lons, lats = transformer.transform(np.asarray(xs), np.asarray(ys))
    vals = arr[rows, cols].astype(np.float64)
    if rows.size > CAND_PER_REGION:
        sel = rng.choice(rows.size, size=CAND_PER_REGION, replace=False)
        lons, lats, vals = np.asarray(lons)[sel], np.asarray(lats)[sel], vals[sel]
    return [
        {
            "lon": float(lo),
            "lat": float(la),
            "doy": float(v),
            "region": region,
            "source_id": region,
        }
        for lo, la, v in zip(lons, lats, vals)
    ]


def gather_candidates(workers: int) -> list[dict[str, Any]]:
    jobs = [dict(region=r) for r in REGION_FILES]
    out: list[dict[str, Any]] = []
    with multiprocessing.Pool(min(workers, len(jobs))) as p:
        for recs in tqdm.tqdm(
            star_imap_unordered(p, _candidates_one, jobs),
            total=len(jobs),
            desc="candidates",
        ):
            out.extend(recs)
    return out


# ---------------------------------------------------------------------------
# Tile writing
# ---------------------------------------------------------------------------
def _write_one(rec: dict[str, Any]) -> dict[str, Any] | None:
    sample_id = rec["sample_id"]
    tif_path = io.locations_dir(SLUG) / f"{sample_id}.tif"
    if tif_path.exists():
        return None
    lon, lat = rec["lon"], rec["lat"]
    region = rec["region"]
    proj, col, row = io.lonlat_to_utm_pixel(lon, lat)
    bounds = io.centered_bounds(col, row, TILE, TILE)

    src_dir = io.raw_dir(SLUG)
    fname = REGION_FILES[region]
    ra = GeotiffRasterFormat().decode_raster(
        src_dir, proj, bounds, resampling=Resampling.bilinear, fname=fname
    )
    doy = ra.array[0].astype(np.float32)  # (H, W)
    invalid = ~np.isfinite(doy)
    doy[invalid] = io.REGRESSION_NODATA

    io.write_label_geotiff(
        SLUG, sample_id, doy, proj, bounds, nodata=io.REGRESSION_NODATA
    )
    io.write_sample_json(
        SLUG,
        sample_id,
        proj,
        bounds,
        snow_year_range(rec["snow_year"]),
        source_id=rec["source_id"],
    )

    valid = doy[doy != io.REGRESSION_NODATA]
    if valid.size == 0:
        return {"sample_id": sample_id, "region": region, "n_valid": 0}
    return {
        "sample_id": sample_id,
        "region": region,
        "n_valid": int(valid.size),
        "mean": float(valid.mean()),
        "min": float(valid.min()),
        "max": float(valid.max()),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=64)
    parser.add_argument(
        "--skip-download", action="store_true", help="assume raw files already present"
    )
    args = parser.parse_args()

    io.check_disk()
    if not args.skip_download:
        download_all()

    cands = gather_candidates(args.workers)
    print(f"gathered {len(cands)} candidate points from {len(REGION_FILES)} regions")

    # DOY distribution is right-skewed (late-melt high-alpine pixels are rare) -> bucket
    # balance across the value range so the regression target covers late-melt dates.
    selected, edges = bucket_balance_regression(
        cands, "doy", total=TOTAL, n_buckets=N_BUCKETS
    )
    edges = [round(e, 2) for e in edges]
    for i, r in enumerate(selected):
        r["sample_id"] = f"{i:06d}"
        r["snow_year"] = SNOW_YEARS[i % len(SNOW_YEARS)]
    print(f"selected {len(selected)} tiles (<= {TOTAL}); DOY bucket edges {edges}")

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

    sel_doy = np.array([r["doy"] for r in selected], dtype=np.float64)
    region_counts = Counter(r["region"] for r in selected)
    year_counts = Counter(r["snow_year"] for r in selected)
    valid_stats = [s for s in stats if s.get("n_valid", 0) > 0]
    pix_min = min((s["min"] for s in valid_stats), default=0.0)
    pix_max = max((s["max"] for s in valid_stats), default=0.0)

    io.write_dataset_metadata(
        SLUG,
        {
            "dataset": SLUG,
            "name": NAME,
            "task_type": "regression",
            "source": "Zenodo / Scientific Data (record 13151801)",
            "license": "CC-BY-4.0",
            "provenance": {
                "url": URL,
                "have_locally": False,
                "annotation_method": "derived from Landsat 30 m surface-reflectance time series (per-period melt-out DOY climatology)",
            },
            "sensors_relevant": ["sentinel2", "sentinel1", "landsat"],
            "regression": {
                "name": "snow_melt_out_date",
                "description": (
                    "Calendar day-of-year (DOY) of snow melt-out, per-pixel mean over the "
                    "2011-2022 period, from the gridded SMOD product for the Pyrenees, "
                    "European Alps and Greater Caucasus (30 m, Landsat-derived). MASKED "
                    "high-confidence variant (reliable snow pixels, DOY ~121-243). Reprojected "
                    "from EPSG:3035 30 m to local UTM 10 m with bilinear resampling."
                ),
                "unit": "day of year",
                "dtype": "float32",
                "value_range": [round(pix_min, 2), round(pix_max, 2)],
                "nodata_value": io.REGRESSION_NODATA,
                "buckets": edges,
            },
            "num_samples": len(selected),
            "region_counts": dict(sorted(region_counts.items())),
            "snow_year_counts": dict(sorted(year_counts.items())),
            "notes": (
                "Bounded-tile sampling across the 3 mountain regions (no full coverage); "
                "bucket-balanced across DOY deciles (right-skewed distribution). 64x64 tiles "
                "at 10 m in local UTM (~640 m). Only the 2011-2022 period is used (1985-1996 "
                "is pre-Sentinel and excluded). Time range = a snow year (Sep 1 -> Aug 31), "
                "spread across 2016/17..2021/22; melt-out is an annual value, not a dated "
                "change event, so change_time is null. "
                f"selected DOY percentiles: p5={np.percentile(sel_doy, 5):.0f}, "
                f"p50={np.percentile(sel_doy, 50):.0f}, p95={np.percentile(sel_doy, 95):.0f}, "
                f"max={sel_doy.max():.0f}."
            ),
        },
    )
    hist_edges = [120, 135, 150, 165, 180, 195, 210, 225, 244]
    hist, _ = np.histogram(sel_doy, bins=hist_edges)
    print("selected-tile DOY histogram:")
    for lo, hi, c in zip(hist_edges[:-1], hist_edges[1:], hist):
        print(f"  [{lo:>4}, {hi:>4}) : {c}")
    print(f"region counts: {dict(sorted(region_counts.items()))}")
    print(f"snow-year counts: {dict(sorted(year_counts.items()))}")
    print(f"per-pixel value range across tiles: [{pix_min:.1f}, {pix_max:.1f}] DOY")
    print(f"num_samples={len(selected)} task_type=regression")
    print("done")


if __name__ == "__main__":
    multiprocessing.set_start_method("forkserver", force=True)
    main()
