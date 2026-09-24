"""Process EuroMineNet into open-set-segmentation label patches.

Source: Yu et al. (2026), "EuroMineNet: A multitemporal Sentinel-2 benchmark for
spatiotemporal mining footprint analysis in the European Union (2015-2024)", ISPRS J.
Photogramm. Remote Sens. (https://doi.org/10.1016/j.isprsjprs.2026.04.046). Data on RODARE
(https://rodare.hzdr.de/record/4656, DOI 10.14278/rodare.4656, CC-BY-4.0) as a single
18.4 GB ``EuroMineNet.zip``. Code: https://github.com/EricYu97/EuroMineNet.

Layout inside the archive: ``EuroMineNet/{Site}/image/Year{YYYY}.tif`` (10-band Sentinel-2,
int16, EPSG:4326, ~10 m) and ``EuroMineNet/{Site}/label/Year{YYYY}.tif`` (single-band uint8
binary mining-footprint mask, values 0=non-mine / 255=mine). 133 mining sites across 14 EU
countries, annual observations 2015-2024. The per-pixel label is a BINARY mining footprint
mask (paper 3.4: "classify each pixel as either mine or non-mine"); the mine-type categories
in the manifest (metallic/coal/non-metallic/quarry) are SITE-level metadata, not per-pixel
classes, and no site->type table ships in the archive, so we use the honest binary scheme:
    id 0 = background (non-mine), id 1 = mining footprint.

Georeferencing (spec 8): the label tifs are written WITHOUT a CRS/geotransform (identity),
but each shares the exact pixel grid of its sibling image tif, which IS georeferenced
(EPSG:4326 + affine transform, constant across a site's years). We recover the label
georeferencing from the image header. To avoid downloading the 18.4 GB archive (the images
are the bulk and pretraining supplies its own imagery), we range-extract only: (a) each
site's image GeoTIFF header (first 64 KB, enough for CRS+transform+size) and (b) the small
label tifs, then write georeferenced binary raw label rasters to raw/. Total pull ~50 MB.

Time (spec 5): annual state maps -> 1-year window anchored on each label year;
change_time=null (footprint-mapping/state task, not the dataset's change-detection task).
We DROP the 2015 year (its 1-year window largely predates usable Sentinel-2, which only
began ramping mid/late-2015) and keep 2016-2024 per the Sentinel-era rule.

Processing (spec 4 dense_raster): scan each raw label in its native EPSG:4326 grid in 64 px
(~640 m) blocks, record class ids present + block-center lon/lat; select tiles-per-class
balanced (rarest first) up to 1000/class under the 25k cap; reproject each selected tile to
a local UTM 64x64 patch at 10 m with NEAREST resampling (categorical); pixels outside the
source site become 255 (nodata). Binary => at most ~2000 samples.

Run:  python3 -m olmoearth_pretrain.open_set_segmentation_data.datasets.eurominenet_sentinel_2_mining_quarry_benchmark
"""

import argparse
import hashlib
import json
import multiprocessing
import random
from collections import Counter, defaultdict
from typing import Any

import numpy as np
import rasterio
import tqdm
from rasterio.io import MemoryFile
from rasterio.warp import Resampling, reproject, transform_bounds
from rasterio.windows import from_bounds
from rslearn.utils.mp import star_imap_unordered
from rslearn.utils.raster_format import get_transform_from_projection_and_bounds

from olmoearth_pretrain.open_set_segmentation_data import download as dl
from olmoearth_pretrain.open_set_segmentation_data import io
from olmoearth_pretrain.open_set_segmentation_data.sampling import (
    select_tiles_per_class,
)

SLUG = "eurominenet_sentinel_2_mining_quarry_benchmark"
URL = "https://rodare.hzdr.de/record/4656/files/EuroMineNet.zip"

YEARS = list(range(2016, 2025))  # drop 2015 (pre-Sentinel-era window)
REF_YEAR_PREF = [2020, 2019, 2021, 2018, 2022, 2017, 2016, 2023, 2024, 2015]
HEADER_BYTES = 65536  # enough of an image tif to read CRS + transform

TILE = 64  # output UTM tile side (10 m px -> 640 m)
BLOCK = 64  # native-block side scanned for composition
PER_CLASS = 1000  # tiles-per-class target
KEEP_PER_CLASS_PER_TASK = 30  # per site-year cap per class (bounds candidate memory)
SENTINEL = 254  # temp dst fill to distinguish uncovered from source values

# Source label codes -> our compact ids. Native 255 (mine) -> id 1; 0 (non-mine) -> id 0.
CLASS_NAMES = {0: "background", 1: "mining"}
CLASS_DESC = {
    0: "Background: non-mining land surface (native EuroMineNet label value 0).",
    1: (
        "Mining footprint: pixels belonging to a mining site's surface extent — the "
        "expert-verified annual mine/non-mine delineation covering open pits, quarries, "
        "waste/tailings deposits and associated disturbed ground (native value 255). "
        "EuroMineNet's 133 EU sites comprise 50 metallic, 56 coal, 8 non-metallic and 19 "
        "large-quarry mines, but this type is a site-level attribute and the per-pixel "
        "annotation is a single binary mining class."
    ),
}


def raw_label_path(site: str, year: int):
    return io.raw_dir(SLUG) / "labels" / site / f"Year{year}.tif"


def _lut() -> np.ndarray:
    """256-entry LUT: source 0->0 (bg), 255->1 (mine), everything else (incl. SENTINEL)->255."""
    lut = np.full(256, io.CLASS_NODATA, dtype=np.uint8)
    lut[0] = 0
    lut[255] = 1
    return lut


# --------------------------------------------------------------------------- download


def _download_site(url: str, index: dict, site: str) -> dict[str, Any]:
    """Recover georef from the site's image header and write georeferenced raw label tifs.

    Returns {site, crs, transform, width, height, years_written}. Idempotent: skips label
    tifs already present.
    """
    # Find a reference image tif to read georeferencing from.
    ref_name = None
    for y in REF_YEAR_PREF:
        n = f"EuroMineNet/{site}/image/Year{y}.tif"
        if n in index:
            ref_name = n
            break
    if ref_name is None:
        raise RuntimeError(f"no image tif for site {site}")
    hdr = dl.extract_remote_zip_member(
        url, index[ref_name], max_uncompressed=HEADER_BYTES
    )
    with MemoryFile(hdr) as mf, mf.open() as ds:
        crs = ds.crs
        transform = ds.transform
        iw, ih = ds.width, ds.height

    years_written = []
    for year in YEARS:
        lname = f"EuroMineNet/{site}/label/Year{year}.tif"
        if lname not in index:
            continue
        out = raw_label_path(site, year)
        if out.exists():
            years_written.append(year)
            continue
        raw = dl.extract_remote_zip_member(url, index[lname])
        with MemoryFile(raw) as mf, mf.open() as ds:
            arr = ds.read(1)
            lw, lh = ds.width, ds.height
        assert (lw, lh) == (iw, ih), (
            f"{site} {year}: label {lw}x{lh} != image {iw}x{ih}"
        )
        out.parent.mkdir(parents=True, exist_ok=True)
        profile = dict(
            driver="GTiff",
            height=lh,
            width=lw,
            count=1,
            dtype="uint8",
            crs=crs,
            transform=transform,
            compress="deflate",
        )
        tmp = out.parent / (out.name + ".tmp")
        with rasterio.open(tmp.path, "w", **profile) as dst:
            dst.write(arr, 1)
        tmp.rename(out)
        years_written.append(year)

    return {
        "site": site,
        "crs": crs.to_string(),
        "transform": list(transform)[:6],
        "width": iw,
        "height": ih,
        "years_written": years_written,
    }


def download_all(url: str, workers: int) -> list[str]:
    """Range-extract label tifs + recover georef for all sites. Returns site list."""
    print("reading remote zip central directory...")
    index = dl.remote_zip_index(url)
    sites = sorted(
        {
            k.split("/")[1]
            for k in index
            if k.count("/") >= 2 and k.split("/")[1] and not k.endswith("/")
        }
    )
    print(f"{len(sites)} sites; extracting labels + georef (idempotent)...")
    tasks = [dict(url=url, index=index, site=s) for s in sites]
    georef: dict[str, Any] = {}
    with multiprocessing.Pool(workers) as p:
        for rec in tqdm.tqdm(
            star_imap_unordered(p, _download_site, tasks), total=len(tasks)
        ):
            georef[rec["site"]] = rec
    gpath = io.raw_dir(SLUG) / "sites_georef.json"
    gpath.parent.mkdir(parents=True, exist_ok=True)
    with gpath.open("w") as f:
        json.dump(georef, f, indent=2)
    n_lab = sum(len(v["years_written"]) for v in georef.values())
    print(f"raw ready: {n_lab} georeferenced label tifs across {len(georef)} sites")
    return sites


# --------------------------------------------------------------------------- scan


def _scan_site_year(site: str, year: int) -> list[dict[str, Any]]:
    """Scan one raw label tif in native 64-blocks; return candidate tile records."""
    path = raw_label_path(site, year)
    if not path.exists():
        return []
    with rasterio.open(path.path) as ds:
        arr = ds.read(1)
        tf = ds.transform
        h, w = ds.height, ds.width

    nby = max(1, -(-h // BLOCK))  # ceil, keep at least one (small sites)
    nbx = max(1, -(-w // BLOCK))
    seed = int(hashlib.md5(f"{site}/{year}".encode()).hexdigest()[:8], 16)
    rng = random.Random(seed)
    idxs = [(bi, bj) for bi in range(nby) for bj in range(nbx)]
    rng.shuffle(idxs)

    kept_per_class: dict[int, int] = defaultdict(int)
    recs: list[dict[str, Any]] = []
    for bi, bj in idxs:
        block = arr[bi * BLOCK : bi * BLOCK + BLOCK, bj * BLOCK : bj * BLOCK + BLOCK]
        if block.size == 0:
            continue
        present = []
        if (block == 0).any():
            present.append(0)
        if (block == 255).any():
            present.append(1)
        if not present:
            continue
        if all(kept_per_class[c] >= KEEP_PER_CLASS_PER_TASK for c in present):
            continue
        for c in present:
            kept_per_class[c] += 1
        # block-center pixel -> lon/lat (b/d terms ~0 for these north-up rasters)
        px = bj * BLOCK + min(BLOCK, w - bj * BLOCK) / 2.0
        py = bi * BLOCK + min(BLOCK, h - bi * BLOCK) / 2.0
        lon = tf.c + tf.a * px + tf.b * py
        lat = tf.f + tf.d * px + tf.e * py
        recs.append(
            {
                "site": site,
                "year": year,
                "lon": float(lon),
                "lat": float(lat),
                "present_ids": sorted(present),
                "source_id": f"{site}/Year{year}/r{bi * BLOCK}_c{bj * BLOCK}",
            }
        )
        if all(kept_per_class[c] >= KEEP_PER_CLASS_PER_TASK for c in (0, 1)):
            break
    return recs


# --------------------------------------------------------------------------- write


def _write_one(rec: dict[str, Any]) -> None:
    sample_id = rec["sample_id"]
    if (io.locations_dir(SLUG) / f"{sample_id}.tif").exists():
        return
    lon, lat = rec["lon"], rec["lat"]
    dst_proj, col, row = io.lonlat_to_utm_pixel(lon, lat)
    bounds = io.centered_bounds(col, row, TILE, TILE)
    dst_transform = get_transform_from_projection_and_bounds(dst_proj, bounds)

    xs = [bounds[0] * io.RESOLUTION, bounds[2] * io.RESOLUTION]
    ys = [bounds[1] * -io.RESOLUTION, bounds[3] * -io.RESOLUTION]
    left, right = min(xs), max(xs)
    bottom, top = min(ys), max(ys)
    l2, b2, r2, t2 = transform_bounds(
        dst_proj.crs, "EPSG:4326", left, bottom, right, top
    )
    pad = 0.003  # ~300 m lon/lat margin so the tile is fully covered

    path = raw_label_path(rec["site"], rec["year"])
    with rasterio.open(path.path) as ds:
        win = from_bounds(l2 - pad, b2 - pad, r2 + pad, t2 + pad, ds.transform)
        src = ds.read(1, window=win, boundless=True, fill_value=SENTINEL)
        win_transform = ds.window_transform(win)
        src_crs = ds.crs

    dst = np.full((TILE, TILE), SENTINEL, np.uint8)
    reproject(
        source=src,
        destination=dst,
        src_transform=win_transform,
        src_crs=src_crs,
        dst_transform=dst_transform,
        dst_crs=dst_proj.crs,
        resampling=Resampling.nearest,
    )
    out = _lut()[dst]  # 0->0, 255->1, uncovered(SENTINEL)/other -> 255

    io.write_label_geotiff(
        SLUG, sample_id, out, dst_proj, bounds, nodata=io.CLASS_NODATA
    )
    present = sorted(int(x) for x in np.unique(out) if x != io.CLASS_NODATA)
    io.write_sample_json(
        SLUG,
        sample_id,
        dst_proj,
        bounds,
        io.year_range(rec["year"]),
        change_time=None,
        source_id=rec["source_id"],
        classes_present=present,
    )


# --------------------------------------------------------------------------- main


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=64)
    args = parser.parse_args()

    io.check_disk()
    sites = download_all(URL, args.workers)
    io.check_disk()

    scan_tasks = [dict(site=s, year=y) for s in sites for y in YEARS]
    print(f"scanning {len(scan_tasks)} site-years for candidate tiles...")
    all_recs: list[dict[str, Any]] = []
    with multiprocessing.Pool(args.workers) as p:
        for recs in tqdm.tqdm(
            star_imap_unordered(p, _scan_site_year, scan_tasks), total=len(scan_tasks)
        ):
            all_recs.extend(recs)
    print(f"scanned {len(all_recs)} candidate tiles")

    cand_freq: Counter = Counter()
    for r in all_recs:
        for c in set(r["present_ids"]):
            cand_freq[c] += 1
    print(
        "candidate tiles per class: "
        + ", ".join(
            f"{CLASS_NAMES[c]}={cand_freq.get(c, 0)}" for c in sorted(CLASS_NAMES)
        )
    )

    selected = select_tiles_per_class(
        all_recs, classes_key="present_ids", per_class=PER_CLASS
    )
    for i, r in enumerate(selected):
        r["sample_id"] = f"{i:06d}"
    print(
        f"selected {len(selected)} tiles (tiles-per-class, <= {PER_CLASS}/class, 25k cap)"
    )

    io.check_disk()
    with multiprocessing.Pool(args.workers) as p:
        for _ in tqdm.tqdm(
            star_imap_unordered(p, _write_one, [dict(rec=r) for r in selected]),
            total=len(selected),
        ):
            pass

    class_counts: dict[str, int] = defaultdict(int)
    year_counts: dict[int, int] = defaultdict(int)
    site_set = set()
    for r in selected:
        year_counts[r["year"]] += 1
        site_set.add(r["site"])
        for cid in r["present_ids"]:
            class_counts[CLASS_NAMES[cid]] += 1
    print("selected tiles per class (candidate ids):", dict(class_counts))
    print("selected tiles per year:", dict(sorted(year_counts.items())))
    print(f"selected tiles span {len(site_set)} distinct sites")

    classes_meta = [
        {"id": cid, "name": CLASS_NAMES[cid], "description": CLASS_DESC[cid]}
        for cid in sorted(CLASS_NAMES)
    ]
    io.write_dataset_metadata(
        SLUG,
        {
            "dataset": SLUG,
            "name": "EuroMineNet (Sentinel-2 Mining/Quarry Benchmark)",
            "task_type": "classification",
            "source": "RODARE (HZDR) / ISPRS J. Photogramm. Remote Sens. (Yu et al. 2026)",
            "license": "CC-BY-4.0",
            "provenance": {
                "url": "https://rodare.hzdr.de/record/4656",
                "doi": "10.14278/rodare.4656",
                "paper_doi": "10.1016/j.isprsjprs.2026.04.046",
                "code": "https://github.com/EricYu97/EuroMineNet",
                "have_locally": False,
                "annotation_method": "expert-verified annual mining-footprint delineation",
                "years": YEARS,
            },
            "sensors_relevant": ["sentinel2", "sentinel1", "landsat"],
            "classes": classes_meta,
            "nodata_value": io.CLASS_NODATA,
            "num_samples": len(selected),
            "class_counts": dict(class_counts),
            "year_counts": {str(k): v for k, v in sorted(year_counts.items())},
            "num_sites": len(site_set),
            "notes": (
                "Binary mining-footprint segmentation (0=background/non-mine, 1=mining) from "
                "EuroMineNet's 133 EU mining sites, annual 2015-2024 (2015 dropped: its 1-year "
                "window largely predates usable Sentinel-2). Per-pixel labels are binary; the "
                "manifest's mine-type list (metallic/coal/non-metallic/quarry) is a site-level "
                "attribute not present per-pixel in the archive. Label tifs ship without a CRS "
                "but share their sibling image tif's pixel grid, so georeferencing (EPSG:4326, "
                "~10 m, constant per site across years) was recovered from the image header via "
                "HTTP range-extraction of the remote 18.4 GB zip (labels + image headers only, "
                "~50 MB pulled; imagery not downloaded). Tiles-per-class balanced (rarest first) "
                "up to 1000/class under the 25k cap; each selected 640 m tile reprojected to "
                "local UTM 10 m (nearest); source-external pixels = 255 nodata. Time range = the "
                "1-year window of the tile's map year; change_time=null (state/footprint task)."
            ),
        },
    )
    print("done")


if __name__ == "__main__":
    multiprocessing.set_start_method("forkserver", force=True)
    main()
