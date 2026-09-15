"""Process WorldPop Global Population Density into open-set regression label patches.

Source: WorldPop "Global per-country 2000-2020" unconstrained population product
(https://hub.worldpop.org/geodata/listing?id=76), distributed as one GeoTIFF per country
per year at https://data.worldpop.org/GIS/Population/Global_2000_2020/{year}/{ISO3}/
{iso3}_ppp_{year}.tif . The native raster is EPSG:4326, ~3 arc-second (~100 m) pixels,
float32, nodata -99999, and stores **persons-per-pixel counts** ("ppp"), produced by a
random-forest dasymetric model.

This is a *regression* dataset (continuous per-pixel population). Global coverage with no
in-situ reference alternative -> we use BOUNDED-TILE sampling from a diverse fixed set of
countries (see COUNTRIES) rather than global coverage, and bucket-balance the extremely
right-skewed distribution across value buckets.

Units / resampling decision (documented in the summary + metadata):
  Per-pixel *counts* are not resolution-invariant, so resampling them is meaningless. We
  therefore convert to **population density in persons per square kilometre** -- an
  intensity that is invariant to grid resolution -- before/while resampling. For a source
  pixel of size (dx, dy) degrees at latitude phi:
      area_km2(phi) = (dx * 111320 * cos(phi)) * (dy * 110574) / 1e6
      density = ppp / area_km2(phi)
  Because the source pixel area is ~constant over a 640 m tile, bilinearly reprojecting the
  count field to a 10 m UTM grid and then dividing by the (fixed) source pixel area is
  equivalent to reprojecting the density field. Bilinear resampling is appropriate for a
  continuous quantity like density.

Output: single-band float32 GeoTIFFs, local UTM, 10 m/pixel, 64x64 (~640 m), nodata
-99999 (io.REGRESSION_NODATA). One <=1-year time range per tile (the WorldPop product year,
spread across 2016-2020 for temporal diversity within the manifest range).
"""

import argparse
import math
import multiprocessing
from collections import Counter
from typing import Any

import numpy as np
import rasterio
import rasterio.vrt
import tqdm
from rasterio.enums import Resampling
from rslearn.utils.mp import star_imap_unordered
from rslearn.utils.raster_format import GeotiffRasterFormat

from olmoearth_pretrain.open_set_segmentation_data import io
from olmoearth_pretrain.open_set_segmentation_data.sampling import (
    bucket_balance_regression,
)

SLUG = "worldpop_global_population_density"
NAME = "WorldPop Global Population Density"
URL = "https://hub.worldpop.org/geodata/listing?id=76"
BASE_URL = "https://data.worldpop.org/GIS/Population/Global_2000_2020"

TILE = 64
TOTAL = 5000
N_BUCKETS = 10
# Decimation for candidate sampling: ~7 * 92 m ~= 640 m, roughly one candidate per tile.
DECIM = 7
CAND_PER_COUNTRY = 30000
CAND_SEED = 42

# Diverse fixed country set (ISO3) with an assigned product year in 2016-2020, spread for
# temporal diversity. Chosen to span continents, development levels, climate zones, and
# settlement patterns while keeping total download volume moderate (~8 GB; the >1.5 GB
# giants -- USA/BRA/CHN/IND/AUS -- are deliberately excluded, medium/small proxies kept).
COUNTRIES: list[tuple[str, int]] = [
    # Africa
    ("KEN", 2016),  # Kenya - East African highlands + arid north
    ("NGA", 2017),  # Nigeria - dense West African urbanization
    ("EGY", 2018),  # Egypt - extreme Nile-valley concentration vs empty desert
    ("ETH", 2019),  # Ethiopia - highland rural + Addis
    ("ZAF", 2020),  # South Africa - dispersed + Johannesburg/Cape Town
    # Asia
    ("IDN", 2016),  # Indonesia - tropical archipelago
    ("VNM", 2017),  # Vietnam - Red River / Mekong deltas
    ("PHL", 2018),  # Philippines - islands + Manila
    ("JPN", 2019),  # Japan - dense developed, mountainous
    ("BGD", 2020),  # Bangladesh - among the densest rural deltas on Earth
    # Europe
    ("DEU", 2016),  # Germany
    ("FRA", 2017),  # France
    ("GBR", 2018),  # United Kingdom
    ("POL", 2019),  # Poland
    # Americas
    ("MEX", 2016),  # Mexico
    ("PER", 2017),  # Peru - Andes + Amazon + coastal Lima
    ("COL", 2018),  # Colombia
    # Oceania
    ("NZL", 2020),  # New Zealand
]

# Earth constants for lat/lon pixel-area (WGS84 mean; ~0.5% accurate, documented).
M_PER_DEG_LAT = 110574.0
M_PER_DEG_LON = 111320.0


def country_url(iso3: str, year: int) -> str:
    return f"{BASE_URL}/{year}/{iso3}/{iso3.lower()}_ppp_{year}.tif"


def raw_fname(iso3: str, year: int) -> str:
    return f"{iso3.lower()}_ppp_{year}.tif"


def pixel_area_km2(dx_deg: float, dy_deg: float, lat: np.ndarray | float):
    """Approximate ground area (km^2) of a (dx_deg x dy_deg) pixel at latitude ``lat``."""
    coslat = np.cos(np.radians(lat))
    width_m = dx_deg * M_PER_DEG_LON * coslat
    height_m = dy_deg * M_PER_DEG_LAT
    return width_m * height_m / 1e6


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------
def _download_one(iso3: str, year: int) -> dict[str, Any]:
    import urllib.request

    dst = io.raw_dir(SLUG) / raw_fname(iso3, year)
    if dst.exists():
        return {"iso3": iso3, "year": year, "path": str(dst), "skipped": True}
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.parent / (dst.name + ".tmp")
    url = country_url(iso3, year)
    with urllib.request.urlopen(url) as r, tmp.open("wb") as f:
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)
    tmp.rename(dst)
    return {"iso3": iso3, "year": year, "path": str(dst), "skipped": False}


def download_all(workers: int) -> None:
    io.check_disk()
    jobs = [dict(iso3=iso3, year=year) for iso3, year in COUNTRIES]
    with multiprocessing.Pool(min(workers, len(jobs))) as p:
        for res in tqdm.tqdm(
            star_imap_unordered(p, _download_one, jobs),
            total=len(jobs),
            desc="download",
        ):
            # Re-check disk as large country rasters land.
            io.check_disk()
            tag = "skip" if res["skipped"] else "got"
            print(f"[{tag}] {res['iso3']} {res['year']}")


# ---------------------------------------------------------------------------
# Candidate sampling (decimated read of each country)
# ---------------------------------------------------------------------------
def _candidates_one(iso3: str, year: int) -> list[dict[str, Any]]:
    path = io.raw_dir(SLUG) / raw_fname(iso3, year)
    rng = np.random.default_rng(abs(hash((iso3, year, CAND_SEED))) % (2**32))
    with rasterio.open(path.path) as ds:
        dx = ds.transform.a
        dy = -ds.transform.e
        nodata = ds.nodata
        oh = max(1, ds.height // DECIM)
        ow = max(1, ds.width // DECIM)
        arr = ds.read(1, out_shape=(oh, ow), resampling=Resampling.nearest)
        # Decimated pixel-center coordinates via a scaled transform.
        dec_tf = ds.transform * rasterio.Affine.scale(ds.width / ow, ds.height / oh)
    rows, cols = np.where((arr != nodata) & np.isfinite(arr) & (arr >= 0))
    if rows.size == 0:
        return []
    # Pixel-center lon/lat.
    xs, ys = dec_tf * (cols + 0.5, rows + 0.5)
    lons = np.asarray(xs)
    lats = np.asarray(ys)
    ppp = arr[rows, cols].astype(np.float64)
    dens = ppp / pixel_area_km2(dx, dy, lats)
    # Subsample per country to bound memory.
    if rows.size > CAND_PER_COUNTRY:
        sel = rng.choice(rows.size, size=CAND_PER_COUNTRY, replace=False)
        lons, lats, dens = lons[sel], lats[sel], dens[sel]
    return [
        {
            "lon": float(lo),
            "lat": float(la),
            "density": float(de),
            "iso3": iso3,
            "year": year,
            "source_id": f"{iso3}_{year}",
        }
        for lo, la, de in zip(lons, lats, dens)
    ]


def gather_candidates(workers: int) -> list[dict[str, Any]]:
    jobs = [dict(iso3=iso3, year=year) for iso3, year in COUNTRIES]
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
    iso3, year = rec["iso3"], rec["year"]
    proj, col, row = io.lonlat_to_utm_pixel(lon, lat)
    bounds = io.centered_bounds(col, row, TILE, TILE)

    src_dir = io.raw_dir(SLUG)
    fname = raw_fname(iso3, year)
    with rasterio.open((src_dir / fname).path) as ds:
        dx = ds.transform.a
        dy = -ds.transform.e
        src_nodata = ds.nodata
    ra = GeotiffRasterFormat().decode_raster(
        src_dir, proj, bounds, resampling=Resampling.bilinear, fname=fname
    )
    ppp = ra.array[0, 0].astype(np.float64)  # (H, W)

    area = pixel_area_km2(dx, dy, lat)
    dens = ppp / area
    invalid = (~np.isfinite(ppp)) | (ppp <= src_nodata + 1) | (ppp < 0)
    dens = dens.astype(np.float32)
    dens[invalid] = io.REGRESSION_NODATA

    io.write_label_geotiff(
        SLUG, sample_id, dens, proj, bounds, nodata=io.REGRESSION_NODATA
    )
    io.write_sample_json(
        SLUG,
        sample_id,
        proj,
        bounds,
        io.year_range(year),
        source_id=rec["source_id"],
    )

    valid = dens[dens != io.REGRESSION_NODATA]
    if valid.size == 0:
        return {"sample_id": sample_id, "iso3": iso3, "n_valid": 0}
    return {
        "sample_id": sample_id,
        "iso3": iso3,
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
        download_all(args.workers)

    cands = gather_candidates(args.workers)
    print(f"gathered {len(cands)} candidate points from {len(COUNTRIES)} countries")

    # Bucket-balance across log10 density (very right-skewed distribution).
    def log_dens(r: dict[str, Any]) -> float:
        return math.log10(max(r["density"], 0.0) + 1.0)

    selected, log_edges = bucket_balance_regression(
        cands, log_dens, total=TOTAL, n_buckets=N_BUCKETS
    )
    density_edges = [round(10.0**e - 1.0, 4) for e in log_edges]
    for i, r in enumerate(selected):
        r["sample_id"] = f"{i:06d}"
    print(
        f"selected {len(selected)} tiles (<= {TOTAL}); density bucket edges {density_edges}"
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

    # Aggregate value distribution over selected candidate densities (fast, complete)
    # and observed per-pixel range from written tiles.
    sel_dens = np.array([r["density"] for r in selected], dtype=np.float64)
    country_counts = Counter(r["iso3"] for r in selected)
    valid_stats = [s for s in stats if s.get("n_valid", 0) > 0]
    pix_min = min((s["min"] for s in valid_stats), default=0.0)
    pix_max = max((s["max"] for s in valid_stats), default=0.0)

    io.write_dataset_metadata(
        SLUG,
        {
            "dataset": SLUG,
            "name": NAME,
            "task_type": "regression",
            "source": "WorldPop",
            "license": "CC-BY-4.0",
            "provenance": {
                "url": URL,
                "have_locally": False,
                "annotation_method": "model-derived (random-forest dasymetric population model)",
            },
            "sensors_relevant": ["sentinel2", "sentinel1", "landsat"],
            "regression": {
                "name": "population_density",
                "description": (
                    "Human population density derived from WorldPop 'Global per-country "
                    "2000-2020' unconstrained persons-per-pixel counts (100 m, random-forest "
                    "dasymetric model). Native counts are converted to persons per square "
                    "kilometre (a resolution-invariant intensity) before resampling to 10 m."
                ),
                "unit": "persons per square kilometre",
                "dtype": "float32",
                "value_range": [round(pix_min, 4), round(pix_max, 4)],
                "nodata_value": io.REGRESSION_NODATA,
                "buckets": density_edges,
            },
            "num_samples": len(selected),
            "country_counts": dict(sorted(country_counts.items())),
            "notes": (
                "Bounded-tile sampling from a fixed diverse country set (no global coverage); "
                "bucket-balanced across log10(density) deciles. 64x64 tiles at 10 m in local "
                "UTM (~640 m). Bilinear resampling of the density field. Time range = the "
                "WorldPop product year (spread across 2016-2020). "
                f"density percentiles among selected: p50={np.percentile(sel_dens, 50):.2f}, "
                f"p90={np.percentile(sel_dens, 90):.2f}, p99={np.percentile(sel_dens, 99):.2f}, "
                f"max={sel_dens.max():.2f} persons/km^2."
            ),
        },
    )
    # Print a value histogram for the report.
    hist_edges = [0, 1, 10, 50, 100, 500, 1000, 5000, 10000, 50000, np.inf]
    hist, _ = np.histogram(sel_dens, bins=hist_edges)
    print("selected-tile density histogram (persons/km^2):")
    for lo, hi, c in zip(hist_edges[:-1], hist_edges[1:], hist):
        print(f"  [{lo:>8}, {hi:>8}) : {c}")
    print(f"country counts: {dict(sorted(country_counts.items()))}")
    print(
        f"per-pixel value range across tiles: [{pix_min:.3f}, {pix_max:.3f}] persons/km^2"
    )
    print(f"num_samples={len(selected)} task_type=regression")
    print("done")


if __name__ == "__main__":
    multiprocessing.set_start_method("forkserver", force=True)
    main()
