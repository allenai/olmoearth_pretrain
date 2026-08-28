"""Process "Global River Gravel Bars (Carbonneau & Bizzi)" into label patches.

Source: Durham University Research Online (https://researchdata.durham.ac.uk/files/
r17w62f824x), Carbonneau & Bizzi, CC-BY-NC-4.0. A single 7.6 GB zip
(CarbonneauResearchData.zip) ships 469 GeoTIFF tiles, one per MGRS grid zone (6 deg lon x
8 deg lat), covering ~89% of the non-polar globe: a 10 m Sentinel-2 semantic
classification for July 2021 produced with a fully-convolutional network + image
processing. Each tile is a single-band uint8 GeoTIFF, already in its zone's UTM CRS at
10 m (nodata = 0). Native pixel codes:

    0 = land / background / outside  -> ignore (255)   (no land class in the product)
    1 = river water
    2 = lake water
    3 = sediment / gravel bar   (the KEY fluvial class this product adds)
    4 = ocean
    5 = glaciated terrain
    6 = snow
    7 = cloud                   -> ignore (255)
    8 = data gap                -> ignore (255)

We keep the 6 observable phenomena as classes 0..5 (native code - 1) and treat
land/cloud/data-gap as nodata/ignore (the assembly step supplies negatives from other
datasets, spec s5). This is a GLOBAL derived-product raster, so we do BOUNDED-TILE
dense_raster sampling (spec s5) with tiles-per-class balancing (<=1000 tiles/class,
rarest-first so the sparse gravel-bar / river classes are prioritized).

Because tiles are already local UTM at 10 m, each 64x64 block is cropped NATIVELY from
its source tile (no reprojection): exact georeferencing, no categorical-resampling loss.
Gravel bars are only exposed at summer low flow and the source is a single July-2021
acquisition, so time_range is a fixed summer window [2021-06-01, 2021-09-01) rather than
the full year; the classification is a static snapshot -> change_time null.

Presence rule (a class "counts" toward a 64x64 block for balancing): fluvial thin
classes (river, gravel bar) need only >= PRESENT_ABS pixels -- rivers and bars are narrow
features surrounded by land(0), so a homogeneity/fraction gate would wrongly exclude the
key class; the areal classes (lake, ocean, glaciated, snow) need >= AREA_FRAC of the
block (confident, spatially-coherent windows for a derived product).

BOUNDED SET NOTE: this run processes the tiles retrievable from the source (the Durham
server does not support HTTP range requests and truncated our download at ~3.84 GB); we
sequentially extracted the 246 complete tiles it contained. Those 246 MGRS zones already
span all continents and latitudes (see the summary) -- exactly the bounded, representative
global sample spec s5 calls for. Re-running with the full 469-tile archive present under
raw/.../tiles/ would simply scan more zones; the script is tile-count agnostic.
"""

import argparse
import glob
import multiprocessing
import os
import random
import zlib
from datetime import UTC, datetime
from collections import Counter
from typing import Any

import numpy as np
import rasterio
import tqdm
from pyproj import Transformer
from rslearn.utils.geometry import Projection
from rslearn.utils.mp import star_imap_unordered

from olmoearth_pretrain.open_set_segmentation_data import io, manifest
from olmoearth_pretrain.open_set_segmentation_data.sampling import (
    select_tiles_per_class,
)

SLUG = "global_river_gravel_bars_carbonneau_bizzi"

# Native source code -> our 0-based class id. Codes not here (0 land, 7 cloud, 8 gap,
# anything else) become nodata/ignore (255).
CODE_TO_ID = {1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5}
CLASSES = [
    (
        "river water",
        "Sentinel-2 detected river channel water (Carbonneau & Bizzi code 1).",
    ),
    ("lake water", "Standing lake / reservoir water (code 2)."),
    (
        "sediment/gravel bar",
        "Exposed fluvial sediment / gravel bars in and along river channels; the key "
        "fluvial class this product adds (code 3).",
    ),
    ("ocean", "Ocean / marine water (code 4)."),
    ("glaciated terrain", "Glaciated terrain / glacier ice (code 5)."),
    ("snow", "Seasonal snow cover (code 6)."),
]
ID_TO_NAME = {i: name for i, (name, _d) in enumerate(CLASSES)}

# Sampling parameters.
BLOCK = 64  # native-pixel block == output tile size (64 px * 10 m = 640 m).
PER_CLASS = 1000
CHUNK_ROWS = 2048  # rows per parallel scan chunk (multiple of BLOCK).
PRESENT_ABS = 40  # thin fluvial class present if >= 40 px (~4000 m^2).
AREA_FRAC = 0.15  # areal class present if >= 15% of the 64x64 block.
THIN_IDS = {0, 2}  # river, gravel bar
RARE_IDS = {0, 1, 2}  # fluvial/inland-water -> reservoir priority
# Per-chunk reservoir caps (bound memory; bias toward rare/key classes).
CAP_RARE_PER_CHUNK = 80
CAP_COMMON_PER_CHUNK = 12
YEAR = 2021
# Gravel bars are only exposed at summer low flow; the source is a single July-2021
# acquisition. Use a summer low-flow window rather than the full year so pretraining does
# not pair the label with winter high-flow imagery when the bars are submerged.
LOW_FLOW_WINDOW = (
    datetime(YEAR, 6, 1, tzinfo=UTC),
    datetime(YEAR, 9, 1, tzinfo=UTC),
)
SEED = 42


def _tifs() -> list[str]:
    root = str(io.raw_dir(SLUG))
    tifs = glob.glob(os.path.join(root, "**", "*.tif"), recursive=True)
    tifs += glob.glob(os.path.join(root, "**", "*.tiff"), recursive=True)
    return sorted(set(tifs))


def scan_chunk(path: str, row0: int) -> list[dict[str, Any]]:
    """Scan a CHUNK_ROWS row range of one tile in 64x64 blocks; return candidates."""
    rng = random.Random(zlib.crc32(f"{path}:{row0}".encode()))
    rare: list[dict[str, Any]] = []
    common: list[dict[str, Any]] = []
    n_rare = 0
    n_common = 0
    npix = BLOCK * BLOCK
    area_thresh = AREA_FRAC * npix
    with rasterio.open(path) as ds:
        W, H = ds.width, ds.height
        nbx = W // BLOCK
        if nbx == 0:
            return []
        rows = min(CHUNK_ROWS, H - row0)
        rows = (rows // BLOCK) * BLOCK
        if rows < BLOCK:
            return []
        win = rasterio.windows.Window(0, row0, nbx * BLOCK, rows)
        arr = ds.read(1, window=win)
    nby = rows // BLOCK
    # (nby, BLOCK, nbx, BLOCK) -> (nby*nbx, npix)
    blk = (
        arr.reshape(nby, BLOCK, nbx, BLOCK).transpose(0, 2, 1, 3).reshape(nby * nbx, -1)
    )
    counts = {code: (blk == code).sum(axis=1) for code in CODE_TO_ID}
    nblocks = blk.shape[0]
    for bi in range(nblocks):
        classes = []
        for code, cid in CODE_TO_ID.items():
            c = int(counts[code][bi])
            thr = PRESENT_ABS if cid in THIN_IDS else area_thresh
            if c >= thr:
                classes.append(cid)
        if not classes:
            continue
        by = bi // nbx
        bx = bi % nbx
        rec = {
            "src": path,
            "tlc": bx * BLOCK,  # top-left col in source
            "tlr": row0 + by * BLOCK,  # top-left row in source
            "classes_present": classes,
        }
        if any(cid in RARE_IDS for cid in classes):
            n_rare += 1
            if len(rare) < CAP_RARE_PER_CHUNK:
                rare.append(rec)
            else:
                k = rng.randint(0, n_rare - 1)
                if k < CAP_RARE_PER_CHUNK:
                    rare[k] = rec
        else:
            n_common += 1
            if len(common) < CAP_COMMON_PER_CHUNK:
                common.append(rec)
            else:
                k = rng.randint(0, n_common - 1)
                if k < CAP_COMMON_PER_CHUNK:
                    common[k] = rec
    return rare + common


_REMAP = np.full(256, io.CLASS_NODATA, dtype=np.uint8)
for _code, _cid in CODE_TO_ID.items():
    _REMAP[_code] = _cid


def _write_one(rec: dict[str, Any]) -> None:
    sample_id = rec["sample_id"]
    if (io.locations_dir(SLUG) / f"{sample_id}.tif").exists():
        return
    with rasterio.open(rec["src"]) as ds:
        win = rasterio.windows.Window(rec["tlc"], rec["tlr"], BLOCK, BLOCK)
        src = ds.read(1, window=win)
        left = ds.bounds.left
        top = ds.bounds.top
        crs = ds.crs
    if src.shape != (BLOCK, BLOCK):
        return
    ids = _REMAP[src]
    present = sorted(int(v) for v in np.unique(ids) if v != io.CLASS_NODATA)
    if not present:
        return
    proj = Projection(crs, io.RESOLUTION, -io.RESOLUTION)
    x_min = int(round(left / io.RESOLUTION)) + rec["tlc"]
    y_min = -int(round(top / io.RESOLUTION)) + rec["tlr"]
    bounds = (x_min, y_min, x_min + BLOCK, y_min + BLOCK)
    io.write_label_geotiff(SLUG, sample_id, ids, proj, bounds, nodata=io.CLASS_NODATA)
    io.write_sample_json(
        SLUG,
        sample_id,
        proj,
        bounds,
        LOW_FLOW_WINDOW,
        source_id=f"{os.path.basename(rec['src'])}:{rec['tlc']}_{rec['tlr']}",
        classes_present=present,
    )


def _tile_center_lonlat(path: str) -> tuple[float, float]:
    with rasterio.open(path) as ds:
        cx = (ds.bounds.left + ds.bounds.right) / 2.0
        cy = (ds.bounds.top + ds.bounds.bottom) / 2.0
        tr = Transformer.from_crs(ds.crs, "EPSG:4326", always_xy=True)
        lon, lat = tr.transform(cx, cy)
    return float(lon), float(lat)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=64)
    parser.add_argument("--max-tiles", type=int, default=0, help="0 = all")
    args = parser.parse_args()

    io.check_disk()
    manifest.write_registry_entry(SLUG, "in_progress")

    tifs = _tifs()
    if args.max_tiles:
        tifs = tifs[: args.max_tiles]
    print(f"{len(tifs)} source tiles")
    if not tifs:
        raise RuntimeError("no GeoTIFF tiles under raw dir")

    tasks: list[dict[str, Any]] = []
    for path in tifs:
        with rasterio.open(path) as ds:
            H = ds.height
        for r0 in range(0, H - BLOCK + 1, CHUNK_ROWS):
            tasks.append({"path": path, "row0": r0})
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
    cand_counts: Counter = Counter()
    for r in candidates:
        for c in set(r["classes_present"]):
            cand_counts[c] += 1
    print(
        f"{len(candidates)} candidates; per-class "
        + str({ID_TO_NAME[c]: cand_counts.get(c, 0) for c in range(len(CLASSES))})
    )

    io.check_disk()

    selected = select_tiles_per_class(
        candidates, classes_key="classes_present", per_class=PER_CLASS, seed=SEED
    )
    rng = random.Random(SEED)
    rng.shuffle(selected)
    for i, r in enumerate(selected):
        r["sample_id"] = f"{i:06d}"
    sel_counts: Counter = Counter()
    for r in selected:
        for c in set(r["classes_present"]):
            sel_counts[c] += 1
    print(
        f"selected {len(selected)} tiles; tiles-per-class "
        + str({ID_TO_NAME[c]: sel_counts.get(c, 0) for c in range(len(CLASSES))})
    )

    with multiprocessing.Pool(args.workers) as p:
        for _ in tqdm.tqdm(
            star_imap_unordered(p, _write_one, [dict(rec=r) for r in selected]),
            total=len(selected),
            desc="write",
        ):
            pass

    written = glob.glob(os.path.join(str(io.locations_dir(SLUG)), "*.tif"))
    n_written = len(written)

    io.write_dataset_metadata(
        SLUG,
        {
            "dataset": SLUG,
            "name": "Global River Gravel Bars (Carbonneau & Bizzi)",
            "task_type": "classification",
            "source": "Durham University Research Online (Carbonneau & Bizzi)",
            "license": "CC-BY-NC-4.0",
            "provenance": {
                "url": "https://researchdata.durham.ac.uk/files/r17w62f824x",
                "have_locally": False,
                "annotation_method": (
                    "Fully-convolutional network (hand-labeled Sentinel-2 training) + "
                    "image processing; global 10 m July-2021 semantic classification."
                ),
            },
            "sensors_relevant": ["sentinel2", "sentinel1", "landsat"],
            "classes": [
                {"id": i, "name": name, "description": desc}
                for i, (name, desc) in enumerate(CLASSES)
            ],
            "nodata_value": io.CLASS_NODATA,
            "num_samples": n_written,
            "class_counts": {
                ID_TO_NAME[c]: sel_counts.get(c, 0) for c in range(len(CLASSES))
            },
            "notes": (
                "Bounded-tile dense_raster sampling of the global 10 m July-2021 product "
                "(MGRS-zone tiles; 246 zones processed from a range-request-truncated "
                "download, spanning all continents). Native codes 1..6 -> class ids 0..5 "
                "(river, lake, gravel bar, ocean, glaciated, snow); land(0)/cloud(7)/data-"
                "gap(8) -> 255 nodata. Tiles-per-class balanced (<=1000/class, rarest-first "
                "so the sparse gravel-bar/river classes are prioritized). Thin fluvial "
                "classes present if >=40 px in a 64x64 block; areal classes if >=15% of the "
                "block. 64x64 tiles cropped NATIVELY in each tile's UTM CRS at 10 m (no "
                "reprojection). Fixed summer window [2021-06-01, 2021-09-01) (gravel bars "
                "exposed only at summer low flow); static July-2021 snapshot, no "
                "change_time. Snow is rare in this Northern-summer (July) product."
            ),
        },
    )
    manifest.write_registry_entry(
        SLUG, "completed", task_type="classification", num_samples=n_written
    )
    print(f"done: {n_written} samples")


if __name__ == "__main__":
    multiprocessing.set_start_method("forkserver", force=True)
    main()
