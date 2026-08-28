"""Process "Long-history Paddy Rice, Northeast China" into open-set-segmentation patches.

Source: figshare 10.6084/m9.figshare.27604839 (ESSD; Long history paddy rice mapping
across Northeast China with deep learning and annual result enhancement, 1985-2023).
The record provides one 30 m annual paddy-rice presence raster per year. The paired
figshare "Training Dataset" (10.6084/m9.figshare.28283606, the DOI in the manifest) is
only a 50-pair Landsat image/mask sample with very few rice pixels, so we instead use the
companion annual MAP rasters, which are the georeferenced dense rasters the manifest
refers to ("30 m annual paddy-rice maps plus a DL training set").

Each annual GeoTIFF is EPSG:32653 (UTM 53N) at 30 m with values:
    0 = non-paddy (observed land), 1 = paddy rice, 3 = nodata (outside study area).

This is a regional derived-product map -> BOUNDED-TILE dense_raster sampling
(tiles-per-class balanced, <=1000 tiles per class). We scan the raster in ~640 m native
blocks (21 px * 30 m ~= 630 m ~= a 64 px @ 10 m output tile), keep spatially-homogeneous
high-confidence blocks, reproject each selected block to local UTM at 10 m (nearest
resampling; categorical), and write a 64x64 label patch. Two classes:

    id 0 = non-paddy   (pure observed non-rice land: 0 rice pixels in the block)
    id 1 = paddy rice  (rice-dominant: >= 50% of observed pixels are rice)

Native ids 0/1 are kept as output ids (no remap); 255 = nodata/ignore. Annual presence,
so each tile gets a 1-year time range anchored on the labeled year (LABELED_YEAR); no
change_time (yearly presence classification, not a dated event).

Labeled year: the maps span 1985-2023; we sample one representative Sentinel-era year
(2020, within the manifest 2016-2023 range). The per-tile source acquisition year is not
recoverable from a single annual composite, so a 1-year window on 2020 is used for all
tiles (documented in the summary).
"""

import argparse
import multiprocessing
import os
import random
import zlib
from typing import Any

import numpy as np
import rasterio
import tqdm
from affine import Affine
from rasterio.warp import Resampling, reproject
from rasterio.warp import transform as warp_transform
from rasterio.windows import Window
from rslearn.utils.mp import star_imap_unordered

from olmoearth_pretrain.open_set_segmentation_data import download, io, manifest

SLUG = "long_history_paddy_rice_northeast_china"
LABELED_YEAR = 2020
# figshare file id for 2020.tif in record 27604839.
SRC_URL = "https://ndownloader.figshare.com/files/50181699"
SRC_NAME = f"{LABELED_YEAR}.tif"

# Native raster encoding.
VAL_NONRICE = 0
VAL_RICE = 1

CLASSES = [
    (
        "non-paddy",
        "Observed land that is not paddy rice in the labeled year (the map's 0 value).",
    ),
    (
        "paddy rice",
        "Flood-irrigated paddy rice presence in the labeled year, mapped from multi-sensor "
        "Landsat with a deep-learning + Annual Result Enhancement method (the map's 1 value).",
    ),
]

# Sampling parameters.
BLOCK = 21  # native 30 m px per block (~630 m ~= a 64 px @ 10 m output tile).
OUT_TILE = 64  # output tile size (64 px * 10 m = 640 m).
PER_CLASS = 1000
MIN_VALID_FRAC = 0.90  # block must be mostly observed land (few nodata=3 pixels).
RICE_MIN_FRAC = 0.50  # "paddy rice" tile: >=50% of observed pixels are rice.
BAND_ROWS = BLOCK * 100  # scan the raster in horizontal bands of this many rows.
CAP_RICE_PER_BAND = 3000
CAP_NONRICE_PER_BAND = 200
SEED = 42


def _src_path() -> str:
    return os.path.join(str(io.raw_dir(SLUG)), SRC_NAME)


def scan_band(r0: int, height: int) -> list[dict[str, Any]]:
    """Scan a horizontal band [r0, r0+height) in BLOCK x BLOCK native blocks."""
    rng = random.Random(zlib.crc32(str(r0).encode()))
    path = _src_path()
    rice: list[dict[str, Any]] = []
    nonrice: list[dict[str, Any]] = []
    n_rice_seen = 0
    n_nonrice_seen = 0
    thr_valid = MIN_VALID_FRAC * BLOCK * BLOCK
    with rasterio.open(path) as ds:
        W = ds.width
        nbx = W // BLOCK
        h = (min(height, ds.height - r0) // BLOCK) * BLOCK
        if h == 0 or nbx == 0:
            return []
        arr = ds.read(1, window=Window(0, r0, nbx * BLOCK, h))
        nby = h // BLOCK
        # (nby, BLOCK, nbx, BLOCK)
        b = arr.reshape(nby, BLOCK, nbx, BLOCK)
        valid = (b == VAL_NONRICE) | (b == VAL_RICE)
        nv = valid.sum(axis=(1, 3))
        nr = (b == VAL_RICE).sum(axis=(1, 3))
        for iy in range(nby):
            for jx in range(nbx):
                nvj = int(nv[iy, jx])
                if nvj < thr_valid:
                    continue
                nrj = int(nr[iy, jx])
                frac = nrj / nvj
                row_c = r0 + iy * BLOCK + BLOCK // 2
                col_c = jx * BLOCK + BLOCK // 2
                rec = {
                    "col": col_c,
                    "row": row_c,
                    "frac": frac,
                }
                if frac >= RICE_MIN_FRAC:
                    rec["label"] = "rice"
                    n_rice_seen += 1
                    if len(rice) < CAP_RICE_PER_BAND:
                        rice.append(rec)
                    else:
                        k = rng.randint(0, n_rice_seen - 1)
                        if k < CAP_RICE_PER_BAND:
                            rice[k] = rec
                elif nrj == 0:
                    rec["label"] = "nonrice"
                    n_nonrice_seen += 1
                    if len(nonrice) < CAP_NONRICE_PER_BAND:
                        nonrice.append(rec)
                    else:
                        k = rng.randint(0, n_nonrice_seen - 1)
                        if k < CAP_NONRICE_PER_BAND:
                            nonrice[k] = rec
    return rice + nonrice


def _write_one(rec: dict[str, Any]) -> None:
    sample_id = rec["sample_id"]
    if (io.locations_dir(SLUG) / f"{sample_id}.tif").exists():
        return

    half = 40  # native-pixel margin around block center for reprojection source.
    with rasterio.open(_src_path()) as ds:
        # Block-center coords in the source (projected UTM) CRS -> WGS84 lon/lat.
        x_c, y_c = ds.xy(rec["row"], rec["col"])
        lons, lats = warp_transform(ds.crs, "EPSG:4326", [x_c], [y_c])
        lon, lat = float(lons[0]), float(lats[0])
        c0 = max(0, rec["col"] - half)
        r0 = max(0, rec["row"] - half)
        c1 = min(ds.width, rec["col"] + half)
        r1 = min(ds.height, rec["row"] + half)
        win = Window(c0, r0, c1 - c0, r1 - r0)
        src_arr = ds.read(1, window=win)
        src_transform = ds.window_transform(win)
        src_crs = ds.crs

    proj, col, row = io.lonlat_to_utm_pixel(lon, lat)
    bounds = io.centered_bounds(col, row, OUT_TILE, OUT_TILE)
    dst_transform = Affine(
        proj.x_resolution,
        0,
        bounds[0] * proj.x_resolution,
        0,
        proj.y_resolution,
        bounds[1] * proj.y_resolution,
    )

    dst = np.full((OUT_TILE, OUT_TILE), io.CLASS_NODATA, dtype=np.uint8)
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
    # Anything not a real class (0/1) -> 255 (ignore); handles source nodata=3.
    dst[(dst != VAL_NONRICE) & (dst != VAL_RICE)] = io.CLASS_NODATA
    present = sorted(int(v) for v in np.unique(dst) if v != io.CLASS_NODATA)

    io.write_label_geotiff(SLUG, sample_id, dst, proj, bounds, nodata=io.CLASS_NODATA)
    io.write_sample_json(
        SLUG,
        sample_id,
        proj,
        bounds,
        io.year_range(LABELED_YEAR),
        source_id=f"{SRC_NAME}:{rec['col']}_{rec['row']}",
        classes_present=present,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=64)
    args = parser.parse_args()

    io.check_disk()
    manifest.write_registry_entry(SLUG, "in_progress")

    raw = io.raw_dir(SLUG)
    raw.mkdir(parents=True, exist_ok=True)
    download.download_http(SRC_URL, raw / SRC_NAME)
    io.check_disk()

    with rasterio.open(_src_path()) as ds:
        H = ds.height
    band_starts = list(range(0, H - BLOCK + 1, BAND_ROWS))
    print(f"scanning {H} rows in {len(band_starts)} bands")

    # Scan phase.
    with multiprocessing.Pool(args.workers) as p:
        results = list(
            tqdm.tqdm(
                star_imap_unordered(
                    p, scan_band, [dict(r0=r0, height=BAND_ROWS) for r0 in band_starts]
                ),
                total=len(band_starts),
                desc="scan",
            )
        )
    candidates = [r for sub in results for r in sub]
    rice = [r for r in candidates if r["label"] == "rice"]
    nonrice = [r for r in candidates if r["label"] == "nonrice"]
    print(f"candidates: rice={len(rice)} nonrice={len(nonrice)}")

    io.check_disk()

    rng = random.Random(SEED)
    rng.shuffle(rice)
    rng.shuffle(nonrice)
    selected = rice[:PER_CLASS] + nonrice[:PER_CLASS]
    rng.shuffle(selected)
    for i, r in enumerate(selected):
        r["sample_id"] = f"{i:06d}"
    n_rice = min(len(rice), PER_CLASS)
    n_nonrice = min(len(nonrice), PER_CLASS)
    print(f"selected {len(selected)} (rice={n_rice}, nonrice={n_nonrice})")

    # Write phase.
    with multiprocessing.Pool(args.workers) as p:
        for _ in tqdm.tqdm(
            star_imap_unordered(p, _write_one, [dict(rec=r) for r in selected]),
            total=len(selected),
            desc="write",
        ):
            pass

    io.write_dataset_metadata(
        SLUG,
        {
            "dataset": SLUG,
            "name": "Long-history Paddy Rice, Northeast China",
            "task_type": "classification",
            "source": "figshare (ESSD; Long history paddy rice mapping, NE China)",
            "license": "CC-BY-4.0",
            "provenance": {
                "url": "https://doi.org/10.6084/m9.figshare.27604839",
                "have_locally": False,
                "annotation_method": (
                    "deep-learning paddy-rice mapping from multi-sensor Landsat with "
                    "Annual Result Enhancement; training labels from field survey + "
                    "photo-interpretation"
                ),
            },
            "sensors_relevant": ["sentinel2", "sentinel1", "landsat"],
            "classes": [
                {"id": i, "name": name, "description": desc}
                for i, (name, desc) in enumerate(CLASSES)
            ],
            "nodata_value": io.CLASS_NODATA,
            "num_samples": len(selected),
            "class_counts": {"non-paddy": n_nonrice, "paddy rice": n_rice},
            "labeled_year": LABELED_YEAR,
            "notes": (
                "Bounded-tile dense_raster sampling from the 30 m annual paddy-rice map "
                f"for {LABELED_YEAR} (companion figshare 27604839; the manifest DOI "
                "28283606 is only a 50-pair training sample and is not used). 64x64 tiles "
                "reprojected from EPSG:32653 30 m to local UTM at 10 m (nearest, "
                "categorical). Paddy-rice tiles have >=50% rice over observed pixels; "
                "non-paddy tiles are pure observed non-rice land (>=90% valid). Per-pixel "
                "labels keep native ids (0=non-paddy, 1=paddy rice); 255=nodata. Annual "
                "presence -> 1-year time range on the labeled year; source per-tile "
                "acquisition year is not recoverable from an annual composite."
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
