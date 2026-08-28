"""Process HP-LSP (HLS-PhenoCam Land Surface Phenology) into open-set regression patches.

Source: "Phenology derived from Satellite Data and PhenoCam across CONUS and Alaska,
2019-2020" (ORNL DAAC, DOI 10.3334/ORNLDAAC/2248; CMR collection C2775078742-ORNL_CLOUD;
license open/EOSDIS). A 30 m land-surface-phenology (LSP) product that fuses Harmonized
Landsat-Sentinel (HLS) EVI2 time series with PhenoCam ground-camera observations. For each
of 78 PhenoCam sites and growing seasons 2019/2020, per-pixel phenological transition dates
were derived over the site's HLS/MGRS (UTM) tile.

Files (from CMR ``GET DATA`` URLs, under the Earthdata-protected path):
  HLS_PhenoCam_A{YEAR}_{SITE}_T{MGRS}_LSP_Date.tif  -- 12-band Int16 transition dates
  HLS_PhenoCam_A{YEAR}_{SITE}_T{MGRS}_LSP_EVI2.tif  -- 122-band EVI2 (ancillary; unused)
We use the **Date** files. Their 12 bands are 4 transition types x up to 3 growing cycles:
  band 1-3   Greenup   (onset)  cycles 1/2/3
  band 4-6   Maturity  (onset)  cycles 1/2/3
  band 7-9   Senescence(onset)  cycles 1/2/3
  band 10-12 Dormancy  (onset)  cycles 1/2/3
Values are day-of-year (DOY). nodata = 32767, native dtype Int16, native CRS UTM (per MGRS
tile). IMPORTANT (verified empirically over all 156 files): the **primary annual growth
cycle is cycle 2, not cycle 1** -- greenup cycle 2 (band 2) has ~92% valid coverage and a
physically-correct progression (greenup DOY ~102 -> maturity ~176 -> senescence ~252 ->
dormancy ~314), whereas cycle 1 (~3% valid, mostly negative DOY) and cycle 3 (~4% valid,
DOY ~300+) are sparse early/late partial cycles that straddle the calendar boundary. So the
canonical start-of-season is band 2.

REGRESSION: primary target = **Greenup onset, cycle 2 (band 2)** -- the canonical
start-of-season LSP metric for the dominant annual cycle. (Maturity/senescence/dormancy are
the other bands and could be emitted as separate regression datasets; we ship greenup onset
per the task instruction.)

Greenup onset is an ANNUAL per-pixel value, not a dated change event, so change_time is
null and time_range is the 1-year calendar window of the labeled season (2019 or 2020).

Sampling: bounded-tile sampling across all 156 site-year Date tiles (spec 5), bucket-
balanced across the greenup-DOY value range to <= 5000 samples. Reprojection: native UTM
30 m -> local UTM 10 m with **nearest** resampling (the int16 32767 nodata sentinel makes
bilinear unsafe -- it would smear real DOY into the fill). 64x64 (~640 m) tiles; 32767
becomes io.REGRESSION_NODATA (-99999).

Access: files are Earthdata-protected. Put NASA Earthdata credentials in ~/.netrc
(machine urs.earthdata.nasa.gov login <user> password <pass>, chmod 600) before running;
download uses download.download_earthdata (requests + netrc, follows URS OAuth).

Reproduce:
    python3 -m olmoearth_pretrain.open_set_segmentation_data.datasets.\
hp_lsp_hls_phenocam_land_surface_phenology
"""

import argparse
import json
import multiprocessing
import re
import urllib.request
from collections import Counter
from typing import Any

import numpy as np
import rasterio
import tqdm
from pyproj import Transformer
from rasterio.enums import Resampling
from rslearn.utils.mp import star_imap_unordered
from rslearn.utils.raster_format import GeotiffRasterFormat

from olmoearth_pretrain.open_set_segmentation_data import io
from olmoearth_pretrain.open_set_segmentation_data.download import download_earthdata
from olmoearth_pretrain.open_set_segmentation_data.sampling import (
    bucket_balance_regression,
)

SLUG = "hp_lsp_hls_phenocam_land_surface_phenology"
NAME = "HP-LSP (HLS-PhenoCam Land Surface Phenology)"
URL = "https://doi.org/10.3334/ORNLDAAC/2248"
CMR_COLLECTION = "C2775078742-ORNL_CLOUD"

NODATA_SRC = 32767  # source Int16 fill
GREENUP_BAND = 2  # band 2 = Greenup onset, cycle 2 = the primary annual cycle (1-based)
TILE = 64
TOTAL = 5000
N_BUCKETS = 10
# 64x64 @10m ~= 640 m ~= 21 px @30m; decimate candidate scan by this for ~1 center/tile.
DECIM = 21
CAND_SEED = 42
# Greenup onset (cycle 2) is a within-year DOY; valid range ~1-365. Filters the rare
# cross-year fill and any out-of-range artefacts.
DOY_MIN, DOY_MAX = 1, 366

FNAME_RE = re.compile(
    r"HLS_PhenoCam_A(?P<year>\d{4})_(?P<site>[A-Za-z]{2}-\d+)_T(?P<tile>\w{5})_LSP_Date\.tif"
)


# ---------------------------------------------------------------------------
# CMR granule listing (no auth needed)
# ---------------------------------------------------------------------------
def list_date_urls() -> list[str]:
    """Return the download URLs of all *_LSP_Date.tif granules from CMR."""
    urls: list[str] = []
    page = 1
    while True:
        api = (
            "https://cmr.earthdata.nasa.gov/search/granules.umm_json"
            f"?collection_concept_id={CMR_COLLECTION}&page_size=200&page_num={page}"
        )
        req = urllib.request.Request(api, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=120) as r:
            d = json.loads(r.read())
        items = d.get("items", [])
        if not items:
            break
        for it in items:
            for u in it["umm"].get("RelatedUrls", []):
                url = u.get("URL", "")
                if u.get("Type") == "GET DATA" and url.endswith("_LSP_Date.tif"):
                    urls.append(url)
        if len(items) < 200:
            break
        page += 1
    return sorted(set(urls))


def _fname_of(url: str) -> str:
    return url.split("/")[-1]


def download_all(workers: int) -> list[str]:
    """Download all LSP_Date tifs to raw_dir; return the local filenames."""
    io.check_disk()
    io.raw_dir(SLUG).mkdir(parents=True, exist_ok=True)
    urls = list_date_urls()
    print(f"CMR lists {len(urls)} LSP_Date granules")
    jobs = [dict(url=u, dst=io.raw_dir(SLUG) / _fname_of(u)) for u in urls]
    fnames: list[str] = []
    with multiprocessing.Pool(min(workers, 32)) as p:
        for res in tqdm.tqdm(
            star_imap_unordered(p, download_earthdata, jobs),
            total=len(jobs),
            desc="download",
        ):
            fnames.append(res.name)
    io.check_disk()
    return sorted(fnames)


def local_date_files() -> list[str]:
    """Filenames of already-downloaded LSP_Date tifs in raw_dir."""
    d = io.raw_dir(SLUG)
    if not d.exists():
        return []
    return sorted(p.name for p in d.iterdir() if FNAME_RE.match(p.name))


# ---------------------------------------------------------------------------
# Candidate sampling (decimated read of each site-year Date tile, band 1)
# ---------------------------------------------------------------------------
def _candidates_one(fname: str) -> list[dict[str, Any]]:
    m = FNAME_RE.match(fname)
    if not m:
        return []
    year = int(m.group("year"))
    site = m.group("site")
    path = io.raw_dir(SLUG) / fname
    with rasterio.open(path.path) as ds:
        oh = max(1, ds.height // DECIM)
        ow = max(1, ds.width // DECIM)
        arr = ds.read(GREENUP_BAND, out_shape=(oh, ow), resampling=Resampling.nearest)
        dec_tf = ds.transform * rasterio.Affine.scale(ds.width / ow, ds.height / oh)
        transformer = Transformer.from_crs(ds.crs, "EPSG:4326", always_xy=True)
    valid = (arr != NODATA_SRC) & (arr >= DOY_MIN) & (arr <= DOY_MAX)
    rows, cols = np.where(valid)
    if rows.size == 0:
        return []
    xs, ys = dec_tf * (cols + 0.5, rows + 0.5)  # native-UTM pixel-center coords
    lons, lats = transformer.transform(np.asarray(xs), np.asarray(ys))
    vals = arr[rows, cols].astype(np.float64)
    return [
        {
            "lon": float(lo),
            "lat": float(la),
            "doy": float(v),
            "year": year,
            "fname": fname,
            "source_id": f"{site}_A{year}",
        }
        for lo, la, v in zip(lons, lats, vals)
    ]


def gather_candidates(fnames: list[str], workers: int) -> list[dict[str, Any]]:
    jobs = [dict(fname=f) for f in fnames]
    out: list[dict[str, Any]] = []
    with multiprocessing.Pool(min(workers, len(jobs) or 1)) as p:
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
    proj, col, row = io.lonlat_to_utm_pixel(lon, lat)
    bounds = io.centered_bounds(col, row, TILE, TILE)

    src_dir = io.raw_dir(SLUG)
    # nearest resampling: 32767 is a hard int fill; bilinear would smear it into real DOY.
    ra = GeotiffRasterFormat().decode_raster(
        src_dir, proj, bounds, resampling=Resampling.nearest, fname=rec["fname"]
    )
    doy = ra.array[GREENUP_BAND - 1].astype(np.float32)  # (H, W)
    invalid = (
        (doy == NODATA_SRC) | (doy < DOY_MIN) | (doy > DOY_MAX) | ~np.isfinite(doy)
    )
    doy[invalid] = io.REGRESSION_NODATA

    io.write_label_geotiff(
        SLUG, sample_id, doy, proj, bounds, nodata=io.REGRESSION_NODATA
    )
    io.write_sample_json(
        SLUG,
        sample_id,
        proj,
        bounds,
        io.year_range(rec["year"]),
        source_id=rec["source_id"],
    )

    valid = doy[doy != io.REGRESSION_NODATA]
    if valid.size == 0:
        return {"sample_id": sample_id, "n_valid": 0}
    return {
        "sample_id": sample_id,
        "n_valid": int(valid.size),
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
    if args.skip_download:
        fnames = local_date_files()
    else:
        fnames = download_all(args.workers)
    print(f"{len(fnames)} LSP_Date tiles available")

    cands = gather_candidates(fnames, args.workers)
    print(f"gathered {len(cands)} candidate points from {len(fnames)} tiles")

    # Bucket-balance across the greenup-DOY range so early- and late-season pixels both
    # appear (avoids the sample being dominated by the modal spring greenup date).
    selected, edges = bucket_balance_regression(
        cands, "doy", total=TOTAL, n_buckets=N_BUCKETS
    )
    edges = [round(e, 2) for e in edges]
    for i, r in enumerate(selected):
        r["sample_id"] = f"{i:06d}"
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
            "source": "ORNL DAAC (DOI 10.3334/ORNLDAAC/2248)",
            "license": "open (EOSDIS / ORNL DAAC)",
            "provenance": {
                "url": URL,
                "have_locally": False,
                "annotation_method": (
                    "30 m HLS (Harmonized Landsat-Sentinel) EVI2 time series fused with "
                    "PhenoCam ground-camera phenology; per-pixel transition dates "
                    "(accuracy <= ~5 days)"
                ),
            },
            "sensors_relevant": ["sentinel2", "sentinel1", "landsat"],
            "regression": {
                "name": "greenup_onset_doy",
                "description": (
                    "Greenup onset day-of-year (start of season) for the primary annual "
                    "growth cycle (cycle 2), from the HP-LSP HLS-PhenoCam land-surface-"
                    "phenology product (band 2 of the 12-band LSP_Date GeoTIFF). Cycle 2 is "
                    "the dominant annual cycle (~92% valid coverage, DOY ~1-365); cycles 1/3 "
                    "are sparse partial cross-year cycles and are not used. Reprojected from "
                    "native 30 m UTM to local UTM 10 m with nearest resampling (32767 = fill)."
                ),
                "unit": "day of year",
                "dtype": "float32",
                "value_range": [round(pix_min, 2), round(pix_max, 2)],
                "nodata_value": io.REGRESSION_NODATA,
                "buckets": edges,
            },
            "num_samples": len(selected),
            "year_counts": dict(sorted(year_counts.items())),
            "notes": (
                "Primary target = greenup onset, cycle 2 (band 2 of LSP_Date; cycle 2 is the "
                "dominant annual cycle, ~92% valid). Bounded-tile "
                "sampling across all 156 site-year Date tiles (78 PhenoCam sites, CONUS + "
                "Alaska, 2019-2020), bucket-balanced across greenup-DOY deciles. 64x64 tiles "
                "at 10 m in local UTM (~640 m), nearest resampling from native 30 m UTM. "
                "Greenup onset is an annual per-pixel value, not a dated change event, so "
                "change_time is null and time_range is the labeled calendar year. Maturity/"
                "senescence/dormancy onset (LSP_Date bands 4-12) and EVI2 (LSP_EVI2) are "
                "available in the source but not emitted here. "
                f"selected DOY percentiles: p5={np.percentile(sel_doy, 5):.0f}, "
                f"p50={np.percentile(sel_doy, 50):.0f}, p95={np.percentile(sel_doy, 95):.0f}."
            ),
        },
    )
    print(f"year counts: {dict(sorted(year_counts.items()))}")
    print(f"per-pixel greenup-DOY range across tiles: [{pix_min:.1f}, {pix_max:.1f}]")
    print(f"num_samples={len(selected)} task_type=regression")
    print("done")


if __name__ == "__main__":
    multiprocessing.set_start_method("forkserver", force=True)
    main()
