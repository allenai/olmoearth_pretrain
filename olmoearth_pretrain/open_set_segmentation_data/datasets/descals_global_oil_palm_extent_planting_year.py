"""Process "Descals Global Oil Palm Extent & Planting Year" into label patches.

Source: Zenodo record 13379129 (Descals et al., ESSD, "Global oil palm extent and
planting year from 1990 to 2021"), CC-BY-4.0. The product ships two global tiled layers
over the tropics (609 ~0.9-degree tiles each, EPSG:4326):

    * GlobalOilPalm_OP-extent  -- 10 m 2021 oil-palm EXTENT (uint8):
          0 = background / not oil palm
          1 = industrial oil palm  (large, closed-canopy, geometric estates)
          2 = smallholder oil palm (smaller, less-regular plots)
    * GlobalOilPalm_OP-YoP      -- 30 m PLANTING-YEAR layer (uint16): 0 = none,
          1989..2021 = year of first oil-palm planting.

PRIMARY LABEL = oil-palm TYPE classification (industrial vs smallholder), using the 10 m
extent layer. We keep the native ids as a 3-class scheme:

    id 0 = other        (in-context non-oil-palm land inside an oil-palm-centered tile)
    id 1 = industrial oil palm
    id 2 = smallholder oil palm

The 30 m planting-year (YoP) layer is documented here as an auxiliary age dimension but is
NOT emitted as a second dataset -- keeping ONE clean oil-palm-type classification dataset
(see the dataset summary for the rationale; YoP would be a coarse 30 m regression target
that overlaps poorly with the 10 m type signal).

This is a GLOBAL derived-product raster, so we do BOUNDED-TILE dense_raster sampling
(spec s5) with tiles-per-class balancing (<=1000 tiles/class). The whole product is only
~750 MB uncompressed, so we scan ALL 609 extent tiles (they already ARE the representative
oil-palm regions: SE Asia, W Africa, Latin America) in 64x64 native-pixel blocks and keep
spatially-homogeneous / high-confidence candidates:

    * candidate block: oil-palm pixels (1 or 2) are >= OP_MIN_FRAC of the 64x64 block
      (a strong, contiguous oil-palm signal -- avoids speckle/false positives).
    * a block "contains" a type for balancing if that type is >= TYPE_MIN_FRAC of the
      block's oil-palm pixels (so tiles are labeled by their dominant type; mixed tiles
      can carry both). Background (0) is present in essentially every block.

Each selected block's center is reprojected to local UTM and written as a 64x64 10 m
label patch (nearest resampling; categorical). Values keep native ids (0/1/2); 255 =
nodata/ignore. The extent map is a 2021 product, so each tile gets the 2021 one-year
window (oil palm is a persistent perennial crop; no change_time).
"""

import argparse
import glob
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

from olmoearth_pretrain.open_set_segmentation_data import io, manifest
from olmoearth_pretrain.open_set_segmentation_data.sampling import (
    select_tiles_per_class,
)

SLUG = "descals_global_oil_palm_extent_planting_year"

VAL_BG = 0
VAL_IND = 1
VAL_SMALL = 2

CLASSES = [
    (
        "other",
        "In-context non-oil-palm land: the extent layer's 0 value (any other land cover) "
        "inside a tile centered on oil palm. Spatially-meaningful background/negative.",
    ),
    (
        "industrial oil palm",
        "Large-scale industrial oil-palm plantations: closed-canopy estates with regular, "
        "geometric planting patterns (Descals et al. extent class 1).",
    ),
    (
        "smallholder oil palm",
        "Smallholder oil-palm plantations: typically smaller plots with less-regular "
        "planting patterns than industrial estates (Descals et al. extent class 2).",
    ),
]

# Sampling parameters.
BLOCK = 64  # native-pixel block = output tile size (64 px * ~10 m = ~640 m).
PER_CLASS = 1000
OP_MIN_FRAC = 0.15  # candidate block: >=15% of pixels are oil palm (1 or 2).
TYPE_MIN_FRAC = 0.25  # a type "present" if it is >=25% of the block's oil-palm pixels.
CHUNK_ROWS = 2000  # rows per parallel scan chunk (multiple of BLOCK ideally).
CAP_SMALL_PER_CHUNK = 120  # reservoir cap for smallholder-bearing candidates per chunk.
CAP_IND_PER_CHUNK = 30  # reservoir cap for industrial-only candidates per chunk.
YEAR = 2021  # extent product year.
SEED = 42


def _extent_tifs() -> list[str]:
    return sorted(glob.glob(os.path.join(str(io.raw_dir(SLUG)), "extent", "*.tif")))


def scan_chunk(path: str, row0: int, nrows: int) -> list[dict[str, Any]]:
    """Scan a row range of one extent tile in 64x64 blocks; return candidate records."""
    rng = random.Random(zlib.crc32(f"{path}:{row0}".encode()))
    small: list[dict[str, Any]] = []
    ind: list[dict[str, Any]] = []
    n_small_seen = 0
    n_ind_seen = 0
    npix = BLOCK * BLOCK
    with rasterio.open(path) as ds:
        W = ds.width
        nbx = W // BLOCK
        if nbx == 0:
            return []
        r_end = min(ds.height, row0 + nrows)
        for r0 in range(row0, r_end - BLOCK + 1, BLOCK):
            win = rasterio.windows.Window(0, r0, nbx * BLOCK, BLOCK)
            arr = ds.read(1, window=win)
            # (BLOCK, nbx, BLOCK) -> (nbx, BLOCK*BLOCK)
            blk = arr.reshape(BLOCK, nbx, BLOCK).transpose(1, 0, 2).reshape(nbx, -1)
            n_ind = (blk == VAL_IND).sum(axis=1)
            n_small = (blk == VAL_SMALL).sum(axis=1)
            n_op = n_ind + n_small
            for j in range(nbx):
                nop = int(n_op[j])
                if nop < OP_MIN_FRAC * npix:
                    continue
                ni = int(n_ind[j])
                nsm = int(n_small[j])
                classes = [VAL_BG]
                if ni >= TYPE_MIN_FRAC * nop:
                    classes.append(VAL_IND)
                if nsm >= TYPE_MIN_FRAC * nop:
                    classes.append(VAL_SMALL)
                if len(classes) == 1:
                    continue  # no dominant oil-palm type -> skip
                col_c = j * BLOCK + BLOCK // 2
                row_c = r0 + BLOCK // 2
                lon, lat = ds.xy(row_c, col_c)
                rec = {
                    "src": path,
                    "col": col_c,
                    "row": row_c,
                    "lon": float(lon),
                    "lat": float(lat),
                    "classes_present": classes,
                    "n_ind": ni,
                    "n_small": nsm,
                }
                if VAL_SMALL in classes:
                    n_small_seen += 1
                    if len(small) < CAP_SMALL_PER_CHUNK:
                        small.append(rec)
                    else:
                        k = rng.randint(0, n_small_seen - 1)
                        if k < CAP_SMALL_PER_CHUNK:
                            small[k] = rec
                else:
                    n_ind_seen += 1
                    if len(ind) < CAP_IND_PER_CHUNK:
                        ind.append(rec)
                    else:
                        k = rng.randint(0, n_ind_seen - 1)
                        if k < CAP_IND_PER_CHUNK:
                            ind[k] = rec
    return small + ind


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

    half = 130  # native-pixel margin around block center for the reprojection source.
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
    # Only 0/1/2 are real classes; anything else -> 255 (ignore).
    dst[(dst != VAL_BG) & (dst != VAL_IND) & (dst != VAL_SMALL)] = io.CLASS_NODATA
    present = sorted(int(v) for v in np.unique(dst) if v != io.CLASS_NODATA)

    io.write_label_geotiff(SLUG, sample_id, dst, proj, bounds, nodata=io.CLASS_NODATA)
    io.write_sample_json(
        SLUG,
        sample_id,
        proj,
        bounds,
        io.year_range(YEAR),
        source_id=f"{os.path.basename(rec['src'])}:{rec['col']}_{rec['row']}",
        classes_present=present,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=64)
    args = parser.parse_args()

    io.check_disk()
    manifest.write_registry_entry(SLUG, "in_progress")

    tifs = _extent_tifs()
    print(f"{len(tifs)} extent tiles")

    tasks: list[dict[str, Any]] = []
    for path in tifs:
        with rasterio.open(path) as ds:
            H = ds.height
        for r0 in range(0, H, CHUNK_ROWS):
            tasks.append({"path": path, "row0": r0, "nrows": CHUNK_ROWS})
    print(f"{len(tasks)} scan chunks")

    with multiprocessing.Pool(args.workers) as p:
        results = list(
            tqdm.tqdm(
                star_imap_unordered(p, scan_chunk, tasks),
                total=len(tasks),
                desc="scan",
            )
        )
    candidates = [r for sub in results for r in sub]
    n_small_c = sum(1 for r in candidates if VAL_SMALL in r["classes_present"])
    n_ind_c = sum(1 for r in candidates if VAL_IND in r["classes_present"])
    print(
        f"candidates: {len(candidates)} (with smallholder={n_small_c}, with industrial={n_ind_c})"
    )

    io.check_disk()

    selected = select_tiles_per_class(
        candidates, classes_key="classes_present", per_class=PER_CLASS, seed=SEED
    )
    rng = random.Random(SEED)
    rng.shuffle(selected)
    for i, r in enumerate(selected):
        r["sample_id"] = f"{i:06d}"
    # Report tiles-per-class selected counts.
    sel_class_counts: Counter = Counter()
    for r in selected:
        for c in r["classes_present"]:
            sel_class_counts[c] += 1
    print(f"selected {len(selected)} tiles; tiles-per-class {dict(sel_class_counts)}")

    with multiprocessing.Pool(args.workers) as p:
        for _ in tqdm.tqdm(
            star_imap_unordered(p, _write_one, [dict(rec=r) for r in selected]),
            total=len(selected),
            desc="write",
        ):
            pass

    name_by_id = {i: name for i, (name, _d) in enumerate(CLASSES)}
    io.write_dataset_metadata(
        SLUG,
        {
            "dataset": SLUG,
            "name": "Descals Global Oil Palm Extent & Planting Year",
            "task_type": "classification",
            "source": "Zenodo / ESSD (Descals et al., record 13379129)",
            "license": "CC-BY-4.0",
            "provenance": {
                "url": "https://doi.org/10.5281/zenodo.13379129",
                "have_locally": False,
                "annotation_method": (
                    "derived-product (Sentinel-1 + Landsat time series classification), "
                    "validated against photo-interpreted points"
                ),
            },
            "sensors_relevant": ["sentinel2", "sentinel1", "landsat"],
            "classes": [
                {"id": i, "name": name, "description": desc}
                for i, (name, desc) in enumerate(CLASSES)
            ],
            "nodata_value": io.CLASS_NODATA,
            "num_samples": len(selected),
            "class_counts": {
                name_by_id[c]: sel_class_counts.get(c, 0) for c in sorted(name_by_id)
            },
            "notes": (
                "Bounded-tile dense_raster sampling of the 10 m 2021 oil-palm EXTENT layer "
                "(all 609 global tropical tiles scanned in 64x64 native-pixel blocks). "
                "Primary label = oil-palm TYPE: 0=other (in-context background), "
                "1=industrial, 2=smallholder (native ids). Tiles-per-class balanced "
                "(<=1000/class, rarest-first so smallholder is prioritized). Candidate "
                "blocks have >=15% oil-palm pixels; a type counts if it is >=25% of the "
                "block's oil-palm pixels. 64x64 tiles reprojected to local UTM at 10 m "
                "(nearest resampling; categorical). 2021 one-year window; oil palm is a "
                "persistent perennial crop, no change_time. AUXILIARY: the product's 30 m "
                "planting-year (YoP, 1989-2021) layer was downloaded but intentionally not "
                "emitted as a second dataset (kept one clean type-classification dataset)."
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
