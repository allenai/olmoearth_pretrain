"""Process "Global Sugarcane 10 m" into open-set-segmentation label patches.

Source: Zenodo record 10871164 (Zhang et al., "Mapping sugarcane globally at 10 m
resolution using GEDI and Sentinel-2"), CC-BY-4.0. 10 m sugarcane presence maps for the
top 13 producing countries, derived from GEDI canopy-height metrics + Sentinel-2 and
validated against field data over 2019-2022. One GeoTIFF per country, EPSG:4326 at
~10 m, with FIVE uint8 bands:

    band 1 = n_tallmonths  (count of "tall canopy" months; 0 over ocean/water/unobserved,
                            high (~14-45) over sugarcane -- a good observed-land proxy)
    band 2 = sugarcane     (the product's binary map: 0 = not sugarcane, 1 = sugarcane)
    band 3 = ESA           }
    band 4 = ESRI          } cross-product agreement layers (not used here)
    band 5 = GLAD          }

We use band 2 (sugarcane) as the per-pixel label. Two classes:

    id 0 = other       (observed non-sugarcane land)
    id 1 = sugarcane

This is a global derived-product raster, so we do BOUNDED-TILE dense_raster sampling
(<=1000 tiles per class). We downloaded a bounded, cross-continental subset of the 13
country rasters (see the dataset summary for the exact list; the 3 largest zips --
brazil/india/china, ~25 GB combined -- were skipped to keep the download bounded). Each
country raster is scanned in 64x64 native-pixel blocks split into parallel row chunks;
we keep spatially-homogeneous candidates:

    * sugarcane tile: >= SUGAR_MIN_FRAC of the block's pixels are sugarcane.
    * other tile: ZERO sugarcane AND >= LAND_MIN_FRAC of pixels are observed land
      (band 1 n_tallmonths > 0), which excludes ocean/water/unobserved fill and yields
      genuine non-sugarcane land as the negative class.

Each selected block's center is reprojected to local UTM and written as a 64x64 10 m
label patch (nearest resampling; categorical). Per-pixel values keep the native ids
(0/1); 255 = nodata/ignore. The map is a multi-year (2019-2022) sugarcane extent, so
each tile is assigned a uniformly-sampled 1-year window within that period (sugarcane is
a persistent crop over the window; no change_time -- yearly presence classification).
"""

import argparse
import glob
import multiprocessing
import os
import random
import zipfile
import zlib
from collections import Counter
from typing import Any

import numpy as np
import rasterio
import tqdm
from affine import Affine
from rasterio.warp import Resampling, reproject
from rslearn.utils.mp import star_imap_unordered

from olmoearth_pretrain.open_set_segmentation_data import io

SLUG = "global_sugarcane_10_m"

# Countries downloaded (bounded, cross-continental subset of the 13-country product).
COUNTRIES = [
    "guatemala",
    "colombia",
    "usa",
    "australia",
    "southafrica",
    "indonesia",
    "philippines",
    "mexico",
    "pakistan",
    "thailand",
]

# Band indices (1-based, rasterio) in the source rasters.
BAND_TALLMONTHS = 1
BAND_SUGARCANE = 2
VAL_OTHER = 0
VAL_SUGAR = 1

# Output class ids (kept aligned to native sugarcane-band values).
CLASSES = [
    (
        "other",
        "Observed non-sugarcane land (the sugarcane band's 0 value over pixels with "
        "detected canopy activity, i.e. n_tallmonths > 0 -- excludes ocean/water/"
        "unobserved fill).",
    ),
    (
        "sugarcane",
        "Sugarcane presence (the product's 1 value), detected from GEDI canopy-height "
        "metrics and Sentinel-2 time series and validated against field data.",
    ),
]

# Sampling parameters.
BLOCK = 64  # native-pixel block = output tile size (64 px * ~10 m = ~640 m).
PER_CLASS = 1000
SUGAR_MIN_FRAC = 0.20  # "sugarcane" tile: >=20% of block pixels are sugarcane.
LAND_MIN_FRAC = 0.30  # "other" tile: >=30% of pixels are observed land (band1>0).
CHUNK_ROWS = 6400  # rows per parallel scan chunk (multiple of BLOCK).
# Reservoir caps per scan chunk (bound memory; plenty to balance from across chunks).
CAP_SUGAR_PER_CHUNK = 200
CAP_OTHER_PER_CHUNK = 40
YEARS = [2019, 2020, 2021, 2022]
SEED = 42


def _country_dir(country: str) -> str:
    return os.path.join(str(io.raw_dir(SLUG)), country)


def _country_tifs(country: str) -> list[str]:
    """All sub-tile GeoTIFFs for a country (large countries are GDAL-retiled into
    several `<country>_GEDIS2_v1<ROW>-<COL>.tif` files; small ones are a single tif).
    """
    return sorted(glob.glob(os.path.join(_country_dir(country), "*.tif")))


def _ensure_extracted(country: str) -> None:
    if _country_tifs(country):
        return
    zip_path = os.path.join(str(io.raw_dir(SLUG)), f"{country}_GEDIS2_v1.zip")
    out_dir = _country_dir(country)
    os.makedirs(out_dir, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(out_dir)


def scan_chunk(country: str, path: str, row0: int, nrows: int) -> list[dict[str, Any]]:
    """Scan a row range of one source raster in 64x64 blocks; return candidates."""
    rng = random.Random(zlib.crc32(f"{path}:{row0}".encode()))
    sugar: list[dict[str, Any]] = []
    other: list[dict[str, Any]] = []
    n_sugar_seen = 0
    n_other_seen = 0
    with rasterio.open(path) as ds:
        W = ds.width
        nbx = W // BLOCK
        if nbx == 0:
            return []
        r_end = min(ds.height, row0 + nrows)
        for r0 in range(row0, r_end - BLOCK + 1, BLOCK):
            win = rasterio.windows.Window(0, r0, nbx * BLOCK, BLOCK)
            sug = ds.read(BAND_SUGARCANE, window=win)
            tall = ds.read(BAND_TALLMONTHS, window=win)
            # (BLOCK, nbx, BLOCK) -> (nbx, BLOCK, BLOCK)
            sblk = sug.reshape(BLOCK, nbx, BLOCK).transpose(1, 0, 2).reshape(nbx, -1)
            tblk = tall.reshape(BLOCK, nbx, BLOCK).transpose(1, 0, 2).reshape(nbx, -1)
            npix = BLOCK * BLOCK
            n_sugar = (sblk == VAL_SUGAR).sum(axis=1)
            n_land = (tblk > 0).sum(axis=1)
            for j in range(nbx):
                ns = int(n_sugar[j])
                col_c = j * BLOCK + BLOCK // 2
                row_c = r0 + BLOCK // 2
                if ns >= SUGAR_MIN_FRAC * npix:
                    lon, lat = ds.xy(row_c, col_c)
                    rec = {
                        "country": country,
                        "src": path,
                        "col": col_c,
                        "row": row_c,
                        "lon": float(lon),
                        "lat": float(lat),
                        "label": "sugarcane",
                    }
                    n_sugar_seen += 1
                    if len(sugar) < CAP_SUGAR_PER_CHUNK:
                        sugar.append(rec)
                    else:
                        k = rng.randint(0, n_sugar_seen - 1)
                        if k < CAP_SUGAR_PER_CHUNK:
                            sugar[k] = rec
                elif ns == 0 and int(n_land[j]) >= LAND_MIN_FRAC * npix:
                    lon, lat = ds.xy(row_c, col_c)
                    rec = {
                        "country": country,
                        "src": path,
                        "col": col_c,
                        "row": row_c,
                        "lon": float(lon),
                        "lat": float(lat),
                        "label": "other",
                    }
                    n_other_seen += 1
                    if len(other) < CAP_OTHER_PER_CHUNK:
                        other.append(rec)
                    else:
                        k = rng.randint(0, n_other_seen - 1)
                        if k < CAP_OTHER_PER_CHUNK:
                            other[k] = rec
    return sugar + other


def _write_one(rec: dict[str, Any]) -> None:
    sample_id = rec["sample_id"]
    if (io.locations_dir(SLUG) / f"{sample_id}.tif").exists():
        return

    proj, col, row = io.lonlat_to_utm_pixel(rec["lon"], rec["lat"])
    bounds = io.centered_bounds(col, row, BLOCK, BLOCK)
    dst_transform = Affine(
        proj.x_resolution,
        0,
        bounds[0] * proj.x_resolution,
        0,
        proj.y_resolution,
        bounds[1] * proj.y_resolution,
    )

    half = 130  # native-pixel margin around block center for reprojection source.
    with rasterio.open(rec["src"]) as ds:
        c0 = max(0, rec["col"] - half)
        r0 = max(0, rec["row"] - half)
        c1 = min(ds.width, rec["col"] + half)
        r1 = min(ds.height, rec["row"] + half)
        win = rasterio.windows.Window(c0, r0, c1 - c0, r1 - r0)
        src_arr = ds.read(BAND_SUGARCANE, window=win)
        src_transform = ds.window_transform(win)
        src_crs = ds.crs

    dst = np.full((BLOCK, BLOCK), io.CLASS_NODATA, dtype=np.uint8)
    reproject(
        source=src_arr,
        destination=dst,
        src_transform=src_transform,
        src_crs=src_crs,
        dst_transform=dst_transform,
        dst_crs=proj.crs,
        resampling=Resampling.nearest,
        dst_nodata=io.CLASS_NODATA,
    )
    # Only 0/1 are real classes; everything else -> 255 (ignore).
    dst[(dst != VAL_OTHER) & (dst != VAL_SUGAR)] = io.CLASS_NODATA
    present = sorted(int(v) for v in np.unique(dst) if v != io.CLASS_NODATA)

    io.write_label_geotiff(SLUG, sample_id, dst, proj, bounds, nodata=io.CLASS_NODATA)
    io.write_sample_json(
        SLUG,
        sample_id,
        proj,
        bounds,
        io.year_range(rec["year"]),
        source_id=f"{os.path.basename(rec['src'])}:{rec['col']}_{rec['row']}",
        classes_present=present,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=64)
    args = parser.parse_args()

    io.check_disk()

    # Extract country tifs from the downloaded zips (idempotent).
    for c in COUNTRIES:
        _ensure_extracted(c)

    # Build parallel scan tasks (row-chunk per source sub-tile).
    tasks: list[dict[str, Any]] = []
    n_tifs = 0
    for c in COUNTRIES:
        for path in _country_tifs(c):
            n_tifs += 1
            with rasterio.open(path) as ds:
                H = ds.height
                descs = ds.descriptions
                if descs[BAND_SUGARCANE - 1] not in (None, "sugarcane"):
                    print(
                        f"WARN {path}: band{BAND_SUGARCANE} desc={descs[BAND_SUGARCANE - 1]!r}"
                    )
            for r0 in range(0, H, CHUNK_ROWS):
                tasks.append(
                    {"country": c, "path": path, "row0": r0, "nrows": CHUNK_ROWS}
                )
    print(f"{len(COUNTRIES)} countries, {n_tifs} source tifs, {len(tasks)} scan chunks")

    with multiprocessing.Pool(args.workers) as p:
        results = list(
            tqdm.tqdm(
                star_imap_unordered(p, scan_chunk, tasks),
                total=len(tasks),
                desc="scan",
            )
        )
    candidates = [r for sub in results for r in sub]
    sugar = [r for r in candidates if r["label"] == "sugarcane"]
    other = [r for r in candidates if r["label"] == "other"]
    print(f"candidates: sugarcane={len(sugar)} other={len(other)}")

    io.check_disk()

    rng = random.Random(SEED)
    rng.shuffle(sugar)
    rng.shuffle(other)
    selected = sugar[:PER_CLASS] + other[:PER_CLASS]
    rng.shuffle(selected)
    for i, r in enumerate(selected):
        r["sample_id"] = f"{i:06d}"
        r["year"] = rng.choice(YEARS)
    print(
        f"selected {len(selected)} "
        f"(sugarcane={min(len(sugar), PER_CLASS)}, other={min(len(other), PER_CLASS)})"
    )

    with multiprocessing.Pool(args.workers) as p:
        for _ in tqdm.tqdm(
            star_imap_unordered(p, _write_one, [dict(rec=r) for r in selected]),
            total=len(selected),
            desc="write",
        ):
            pass

    counts = Counter(r["label"] for r in selected)
    io.write_dataset_metadata(
        SLUG,
        {
            "dataset": SLUG,
            "name": "Global Sugarcane 10 m",
            "task_type": "classification",
            "source": "Zenodo (Zhang et al., record 10871164)",
            "license": "CC-BY-4.0",
            "provenance": {
                "url": "https://zenodo.org/records/10871164",
                "have_locally": False,
                "annotation_method": "derived-product (GEDI + Sentinel-2) with field validation",
            },
            "sensors_relevant": ["sentinel2", "sentinel1", "landsat"],
            "classes": [
                {"id": i, "name": name, "description": desc}
                for i, (name, desc) in enumerate(CLASSES)
            ],
            "nodata_value": io.CLASS_NODATA,
            "num_samples": len(selected),
            "class_counts": {
                "other": counts.get("other", 0),
                "sugarcane": counts.get("sugarcane", 0),
            },
            "notes": (
                "Bounded-tile dense_raster sampling from a global 10 m sugarcane map "
                "(top-13-producing-countries product; we sampled a cross-continental "
                "bounded subset of 10 countries: " + ", ".join(COUNTRIES) + "). 64x64 "
                "tiles reprojected to local UTM at 10 m (nearest resampling). Sugarcane "
                "tiles have >=20% sugarcane pixels; 'other' tiles have zero sugarcane and "
                ">=30% observed-land pixels (n_tallmonths band > 0, which excludes ocean/"
                "water/unobserved fill). Per-pixel labels keep native ids (0=other, "
                "1=sugarcane); 255=nodata. Multi-year (2019-2022) sugarcane extent -> each "
                "tile assigned a uniformly-sampled 1-year window in that period; persistent "
                "presence, no change_time."
            ),
        },
    )
    print("done")


if __name__ == "__main__":
    multiprocessing.set_start_method("forkserver", force=True)
    main()
