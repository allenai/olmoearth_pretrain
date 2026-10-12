"""Process Global-PCG-10 into open-set-segmentation label patches.

Source: figshare 10.6084/m9.figshare.27731148 (ESSD 2025, https://essd.copernicus.org/
articles/17/5065/2025/) -- "Global-PCG-10: a 10-m global map of plastic-covered
greenhouses derived from Sentinel-2 in 2020". The dataset ships as ~1639 GeoTIFFs, each a
1 deg x 1 deg tile (11133 x 11133 px) at ~10 m in EPSG:4326 (WGS84). Only tiles that
contain PCG are released. Pixel encoding is binary uint8:

    0 = non-PCG  (any other land/water in the mapped tile)
    1 = plastic-covered greenhouse (PCG)

There is no explicit nodata band (every pixel is 0 or 1).

This is a global derived-product map, so we do BOUNDED-TILE dense_raster sampling
(tiles-per-class balanced, <=1000 tiles per class): every source tile is scanned in 64x64
native-pixel blocks; we keep spatially-homogeneous / high-confidence candidates
(PCG-rich blocks with >= PCG_MIN_FRAC greenhouse coverage, and pure non-PCG blocks),
reproject each selected block's center to local UTM, and write a 64x64 10 m label patch
(nearest resampling; categorical). Two output classes keep the native ids 0/1; 255 =
nodata/ignore. PCG-rich tiles carry both classes per-pixel (0 and 1), so class 0 is
abundantly covered; pure non-PCG tiles add background diversity.

Labels are an annual 2020 product, so each tile gets a 1-year time range on 2020 (no
change_time -- annual presence classification, not a dated event).
"""

import argparse
import multiprocessing
import os
import random
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

SLUG = "global_plastic_covered_greenhouses_global_pcg_10"
SRC_SUBDIR = "Global_PCG_10_Dataset"

VAL_NONPCG = 0
VAL_PCG = 1

# Output class ids (kept aligned to native raster values).
CLASSES = [
    (
        "non-PCG",
        "Any pixel in the mapped tile that is not a plastic-covered greenhouse "
        "(other land cover or water); the map's 0 value.",
    ),
    (
        "plastic-covered greenhouse",
        "Plastic-covered greenhouse / plasticulture (film-covered protected "
        "cultivation) detected from Sentinel-2 in 2020; the map's 1 value.",
    ),
]

# Sampling parameters.
BLOCK = 64  # native-pixel block = output tile size (64 px * ~10 m = ~640 m).
PER_CLASS = 1000
# A "PCG" block must have clearly-present, resolvable greenhouse coverage. PCG is a small,
# clustered target; 5% of a 64x64 block (~205 px) is a confident, dense-plasticulture
# window. (Block-fraction survey: ~2000 blocks >=5% per 25 tiles -- ample to reach 1000.)
PCG_MIN_FRAC = 0.05
# Reservoir caps per source tile during scan (bound memory; plenty to balance from
# across ~1639 global tiles).
CAP_PCG_PER_TILE = 200
CAP_NONPCG_PER_TILE = 25
SEED = 42


def _src_dir() -> str:
    return os.path.join(str(io.raw_dir(SLUG)), SRC_SUBDIR)


def _list_source_tiles() -> list[str]:
    d = _src_dir()
    out = []
    for root, _dirs, files in os.walk(d):
        for f in files:
            if f.lower().endswith(".tif"):
                out.append(os.path.join(root, f))
    return sorted(out)


def scan_tile(path: str) -> list[dict[str, Any]]:
    """Scan one source tile in 64x64 native blocks; return homogeneous candidates."""
    rng = random.Random(zlib.crc32(os.path.basename(path).encode()))
    pcg: list[dict[str, Any]] = []
    nonpcg: list[dict[str, Any]] = []
    n_pcg_seen = 0
    n_nonpcg_seen = 0
    thr = PCG_MIN_FRAC * BLOCK * BLOCK
    with rasterio.open(path) as ds:
        W, H = ds.width, ds.height
        nbx = W // BLOCK
        if nbx == 0:
            return []
        for row0 in range(0, H - BLOCK + 1, BLOCK):
            strip = ds.read(
                1, window=rasterio.windows.Window(0, row0, nbx * BLOCK, BLOCK)
            )
            if strip.size == 0:
                continue
            # (BLOCK, nbx, BLOCK) -> (nbx, BLOCK, BLOCK)
            blocks = strip.reshape(BLOCK, nbx, BLOCK).transpose(1, 0, 2)
            n_pcg = (blocks == VAL_PCG).reshape(nbx, -1).sum(axis=1)
            for j in range(nbx):
                np_pcg = int(n_pcg[j])
                col_c = j * BLOCK + BLOCK // 2
                row_c = row0 + BLOCK // 2
                if np_pcg >= thr:
                    lon, lat = ds.xy(row_c, col_c)
                    rec = {
                        "src": path,
                        "col": col_c,
                        "row": row_c,
                        "lon": float(lon),
                        "lat": float(lat),
                        "label": "pcg",
                    }
                    n_pcg_seen += 1
                    if len(pcg) < CAP_PCG_PER_TILE:
                        pcg.append(rec)
                    else:
                        k = rng.randint(0, n_pcg_seen - 1)
                        if k < CAP_PCG_PER_TILE:
                            pcg[k] = rec
                elif np_pcg == 0:
                    lon, lat = ds.xy(row_c, col_c)
                    rec = {
                        "src": path,
                        "col": col_c,
                        "row": row_c,
                        "lon": float(lon),
                        "lat": float(lat),
                        "label": "nonpcg",
                    }
                    n_nonpcg_seen += 1
                    if len(nonpcg) < CAP_NONPCG_PER_TILE:
                        nonpcg.append(rec)
                    else:
                        k = rng.randint(0, n_nonpcg_seen - 1)
                        if k < CAP_NONPCG_PER_TILE:
                            nonpcg[k] = rec
    return pcg + nonpcg


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

    half = 110  # native-pixel margin around block center for reprojection source.
    with rasterio.open(rec["src"]) as ds:
        c0 = max(0, rec["col"] - half)
        r0 = max(0, rec["row"] - half)
        c1 = min(ds.width, rec["col"] + half)
        r1 = min(ds.height, rec["row"] + half)
        win = rasterio.windows.Window(c0, r0, c1 - c0, r1 - r0)
        src_arr = ds.read(1, window=win)
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
    # Only 0/1 are real classes; anything else (reproject fill) -> 255 ignore.
    dst[(dst != VAL_NONPCG) & (dst != VAL_PCG)] = io.CLASS_NODATA
    present = sorted(int(v) for v in np.unique(dst) if v != io.CLASS_NODATA)

    io.write_label_geotiff(SLUG, sample_id, dst, proj, bounds, nodata=io.CLASS_NODATA)
    io.write_sample_json(
        SLUG,
        sample_id,
        proj,
        bounds,
        io.year_range(2020),
        source_id=f"{os.path.basename(rec['src'])}:{rec['col']}_{rec['row']}",
        classes_present=present,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=64)
    args = parser.parse_args()

    io.check_disk()

    tiles = _list_source_tiles()
    print(f"{len(tiles)} source tiles")

    # Scan phase.
    with multiprocessing.Pool(args.workers) as p:
        results = list(
            tqdm.tqdm(
                star_imap_unordered(p, scan_tile, [dict(path=t) for t in tiles]),
                total=len(tiles),
                desc="scan",
            )
        )
    candidates = [r for sub in results for r in sub]
    pcg = [r for r in candidates if r["label"] == "pcg"]
    nonpcg = [r for r in candidates if r["label"] == "nonpcg"]
    print(f"candidates: pcg={len(pcg)} nonpcg={len(nonpcg)}")

    io.check_disk()

    rng = random.Random(SEED)
    rng.shuffle(pcg)
    rng.shuffle(nonpcg)
    selected = pcg[:PER_CLASS] + nonpcg[:PER_CLASS]
    rng.shuffle(selected)
    for i, r in enumerate(selected):
        r["sample_id"] = f"{i:06d}"
    print(
        f"selected {len(selected)} "
        f"(pcg={min(len(pcg), PER_CLASS)}, nonpcg={min(len(nonpcg), PER_CLASS)})"
    )

    # Write phase.
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
            "name": "Global Plastic-Covered Greenhouses (Global-PCG-10)",
            "task_type": "classification",
            "source": "figshare (ESSD 2025)",
            "license": "CC-BY-4.0",
            "provenance": {
                "url": "https://doi.org/10.6084/m9.figshare.27731148",
                "have_locally": False,
                "annotation_method": "derived-product (weak labels + deep learning) from Sentinel-2, 2020",
            },
            "sensors_relevant": ["sentinel2", "sentinel1"],
            "classes": [
                {"id": i, "name": name, "description": desc}
                for i, (name, desc) in enumerate(CLASSES)
            ],
            "nodata_value": io.CLASS_NODATA,
            "num_samples": len(selected),
            "class_counts": {
                "non-PCG (tiles that are pure background)": counts.get("nonpcg", 0),
                "plastic-covered greenhouse (tiles containing PCG)": counts.get(
                    "pcg", 0
                ),
            },
            "notes": (
                "Bounded-tile dense_raster sampling from the 10 m global PCG map "
                "(~1639 released 1deg x 1deg tiles, EPSG:4326). 64x64 tiles reprojected "
                "to local UTM at 10 m (nearest resampling; categorical). PCG tiles have "
                ">=5% PCG pixels (clearly present, resolvable plasticulture) and carry "
                "both classes per-pixel; non-PCG tiles are pure background. Per-pixel "
                "labels keep native ids (0=non-PCG, 1=PCG); 255=nodata/ignore. Annual "
                "2020 product -> 1-year time range on 2020; not a dated event. Caveat: "
                "the source has no nodata band, so pure non-PCG tiles may occasionally "
                "fall on water in coastal tiles (no land mask available to exclude them)."
            ),
        },
    )
    print("done")


if __name__ == "__main__":
    multiprocessing.set_start_method("forkserver", force=True)
    main()
