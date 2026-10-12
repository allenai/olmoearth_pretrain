"""Process RapeseedMap10 (Canola Bloom) into open-set-segmentation label patches.

Source: Mendeley Data 10.17632/ydf3m7pd4j.3 (Han et al., "Developing a phenology- and
pixel-based algorithm for mapping rapeseed at 10 m spatial resolution using multi-source
data"). A global-ish 10 m annual rapeseed/canola presence map covering 18 regional tiles
across 3 years (2017, 2018, 2019). Each source GeoTIFF is EPSG:4326 at ~10 m with values
0 = non-rapeseed (observed land), 1 = rapeseed, 3 = nodata (unmapped / ocean).

This is a regional derived-product map, so we do BOUNDED-TILE dense_raster sampling
(<=1000 tiles per class): we scan every source tile in 64x64 native-pixel blocks, keep
spatially-homogeneous candidates (rapeseed-rich blocks and pure non-rapeseed blocks over
observed land), reproject each selected block's center to local UTM, and write a 64x64
10 m label patch (nearest resampling; categorical). Two classes:

    id 0 = non-rapeseed (other observed land cover)
    id 1 = rapeseed (bloom-based canola presence)

We keep the native raster class ids (0/1) so dense per-pixel tile values need no remap;
255 is nodata/ignore. Labels are annual presence, so each tile gets a 1-year time range
anchored on its labeled year (no change_time -- yearly presence classification, not a
dated bloom event).
"""

import argparse
import multiprocessing
import os
import random
from collections import Counter
from typing import Any

import numpy as np
import rasterio
import tqdm
from rasterio.warp import Resampling, reproject
from rslearn.utils.mp import star_imap_unordered

from olmoearth_pretrain.open_set_segmentation_data import io

SLUG = "rapeseedmap10_canola_bloom"
SRC_SUBDIR = "rapeseed map"

# Native raster encoding. Only 0 (non-rapeseed) and 1 (rapeseed) are real classes;
# every other value is nodata/fill. NOTE: the declared nodata value is inconsistent
# across the source tiles (some use 3, some use 255), so we key off the {0,1} class set
# rather than a single nodata sentinel.
VAL_NONRAPE = 0
VAL_RAPE = 1

# Output class ids (kept aligned to native values).
CLASSES = [
    (
        "non-rapeseed",
        "Observed land that is not rapeseed/canola in the labeled year (the map's 0 value).",
    ),
    (
        "rapeseed (bloom-based)",
        "Rapeseed / oilseed canola presence detected from its distinctive flowering "
        "(bloom) signal in multi-source Sentinel-1/-2 time series (the map's 1 value).",
    ),
]

# Sampling parameters.
BLOCK = 64  # native-pixel block = output tile size (64 px * 10 m = 640 m).
PER_CLASS = 1000
MIN_VALID_FRAC = 0.90  # block must be mostly observed land (few nodata pixels).
RAPE_MIN_FRAC = 0.25  # "rapeseed" tile: >=25% of observed pixels are rapeseed.
# Reservoir caps per source tile during scan (bound memory; plenty to balance from).
CAP_RAPE_PER_TILE = 2000
CAP_NONRAPE_PER_TILE = 150
SEED = 42


def _src_dir() -> str:
    return os.path.join(str(io.raw_dir(SLUG)), SRC_SUBDIR)


def _list_source_tiles() -> list[str]:
    d = _src_dir()
    return sorted(
        os.path.join(d, f) for f in os.listdir(d) if f.lower().endswith(".tif")
    )


def _year_of(path: str) -> int:
    # Filenames like '2018Y100W53N.TIF'.
    return int(os.path.basename(path)[:4])


def scan_tile(path: str) -> list[dict[str, Any]]:
    """Scan one source tile in 64x64 native blocks; return homogeneous candidates."""
    import zlib

    rng = random.Random(zlib.crc32(os.path.basename(path).encode()))
    year = _year_of(path)
    rape: list[dict[str, Any]] = []
    nonrape: list[dict[str, Any]] = []
    n_rape_seen = 0
    n_nonrape_seen = 0
    with rasterio.open(path) as ds:
        W, H = ds.width, ds.height
        nbx = W // BLOCK
        thr_valid = MIN_VALID_FRAC * BLOCK * BLOCK
        for row0 in range(0, H - BLOCK + 1, BLOCK):
            strip = ds.read(
                1, window=rasterio.windows.Window(0, row0, nbx * BLOCK, BLOCK)
            )
            valid_strip = (strip == VAL_NONRAPE) | (strip == VAL_RAPE)
            if strip.size == 0 or not valid_strip.any():
                continue
            # (BLOCK, nbx, BLOCK) -> (nbx, BLOCK, BLOCK)
            blocks = strip.reshape(BLOCK, nbx, BLOCK).transpose(1, 0, 2)
            vblocks = valid_strip.reshape(BLOCK, nbx, BLOCK).transpose(1, 0, 2)
            n_valid = vblocks.reshape(nbx, -1).sum(axis=1)
            n_rape = (blocks == VAL_RAPE).reshape(nbx, -1).sum(axis=1)
            for j in range(nbx):
                nv = int(n_valid[j])
                if nv < thr_valid:
                    continue
                nr = int(n_rape[j])
                frac = nr / nv
                col_c = j * BLOCK + BLOCK // 2
                row_c = row0 + BLOCK // 2
                lon, lat = ds.xy(row_c, col_c)
                rec = {
                    "src": path,
                    "col": col_c,
                    "row": row_c,
                    "lon": float(lon),
                    "lat": float(lat),
                    "year": year,
                    "frac": frac,
                }
                if frac >= RAPE_MIN_FRAC:
                    rec["label"] = "rapeseed"
                    n_rape_seen += 1
                    # reservoir sample
                    if len(rape) < CAP_RAPE_PER_TILE:
                        rape.append(rec)
                    else:
                        k = rng.randint(0, n_rape_seen - 1)
                        if k < CAP_RAPE_PER_TILE:
                            rape[k] = rec
                elif nr == 0:
                    rec["label"] = "non_rapeseed"
                    n_nonrape_seen += 1
                    if len(nonrape) < CAP_NONRAPE_PER_TILE:
                        nonrape.append(rec)
                    else:
                        k = rng.randint(0, n_nonrape_seen - 1)
                        if k < CAP_NONRAPE_PER_TILE:
                            nonrape[k] = rec
    return rape + nonrape


def _write_one(rec: dict[str, Any]) -> None:
    sample_id = rec["sample_id"]
    if (io.locations_dir(SLUG) / f"{sample_id}.tif").exists():
        return

    proj, col, row = io.lonlat_to_utm_pixel(rec["lon"], rec["lat"])
    bounds = io.centered_bounds(col, row, BLOCK, BLOCK)
    from affine import Affine

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
    # Anything that is not a real class (0/1) -> 255 (ignore). Handles the source's
    # inconsistent nodata fills (3 or 255) uniformly.
    dst[(dst != VAL_NONRAPE) & (dst != VAL_RAPE)] = io.CLASS_NODATA
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

    raw = io.raw_dir(SLUG)
    raw.mkdir(parents=True, exist_ok=True)

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
    rape = [r for r in candidates if r["label"] == "rapeseed"]
    nonrape = [r for r in candidates if r["label"] == "non_rapeseed"]
    print(f"candidates: rapeseed={len(rape)} non_rapeseed={len(nonrape)}")

    io.check_disk()

    rng = random.Random(SEED)
    rng.shuffle(rape)
    rng.shuffle(nonrape)
    selected = rape[:PER_CLASS] + nonrape[:PER_CLASS]
    rng.shuffle(selected)
    for i, r in enumerate(selected):
        r["sample_id"] = f"{i:06d}"
    print(
        f"selected {len(selected)} "
        f"(rapeseed={min(len(rape), PER_CLASS)}, non_rapeseed={min(len(nonrape), PER_CLASS)})"
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
            "name": "RapeseedMap10 (Canola Bloom)",
            "task_type": "classification",
            "source": "Mendeley Data (Han et al.)",
            "license": "CC-BY-4.0",
            "provenance": {
                "url": "http://dx.doi.org/10.17632/ydf3m7pd4j.3",
                "have_locally": False,
                "annotation_method": "phenology/bloom-signal detection from Sentinel-1/-2, validated",
            },
            "sensors_relevant": ["sentinel2", "sentinel1"],
            "classes": [
                {"id": i, "name": name, "description": desc}
                for i, (name, desc) in enumerate(CLASSES)
            ],
            "nodata_value": io.CLASS_NODATA,
            "num_samples": len(selected),
            "class_counts": {
                "non-rapeseed": counts.get("non_rapeseed", 0),
                "rapeseed (bloom-based)": counts.get("rapeseed", 0),
            },
            "notes": (
                "Bounded-tile dense_raster sampling from a 10 m regional canola map "
                "(18 regions x 2017/2018/2019). 64x64 tiles reprojected to local UTM at "
                "10 m (nearest resampling). Rapeseed tiles have >=25% rapeseed over "
                "observed pixels; non-rapeseed tiles are pure observed non-rapeseed land "
                "(>=90% valid). Per-pixel labels keep native ids (0/1); 255=nodata. "
                "Annual presence -> 1-year time range on labeled year; not a dated event."
            ),
        },
    )
    # NOTE: registry.json is owned/updated by the orchestrator; this script does not
    # write it. Final status: completed, classification, num_samples=len(selected).
    print("done")


if __name__ == "__main__":
    multiprocessing.set_start_method("forkserver", force=True)
    main()
