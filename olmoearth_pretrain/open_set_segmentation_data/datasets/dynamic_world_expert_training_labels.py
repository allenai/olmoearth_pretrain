"""Process the Dynamic World *expert* training labels into open-set-segmentation tiles.

Source: PANGAEA "Dynamic World training dataset for global land use and land cover
categorization of satellite imagery" (Tait et al. 2021, DOI 10.1594/PANGAEA.933475;
supplement to Brown et al. 2022, Nature Scientific Data 9:251). Human-labeled dense
land-use/land-cover markup on ~24k Sentinel-2 tiles worldwide (10 m, 510x510 px = 5.1 km
tiles). This dataset uses only the **Experts** folder -- tiles densely labeled by a team of
25 expert human labelers recruited by National Geographic Society (the highest-quality
reference subset; the Non_expert crowd folder and the validation holdout are not used here).

Each source GeoTIFF is single-band uint8 already in a local UTM projection at 10 m, north-up.
Tier 1 class values: 0 No data (left unmarked), 1 Water, 2 Trees, 3 Grass, 4 Flooded
Vegetation, 5 Crops, 6 Scrub, 7 Built Area, 8 Bare Ground, 9 Snow/Ice, 10 Cloud. We map
source 1..9 -> output ids 0..8 (the manifest's 9 land-cover classes) and both 0 (unmarked)
and 10 (cloud) -> nodata/ignore 255.

Recipe (spec 4, dense_raster; reference data -- no homogeneity filtering needed): because
each source tile is already local UTM at 10 m, no reprojection is required. Each 510x510
tile is cut into a grid of <=64x64 windows (8x8 grid; edge windows are 62 px, still <=64).
A window is kept only if >=25% of its pixels are labeled (value in 1..9; unmarked/cloud
excluded). Selection is tiles-per-class balanced (each window counts toward every class
present, rarest class first), up to 1000 windows/class and <=25k total. Time range = a
1-year window on the acquisition year parsed from the tile filename (dates span 2017-2019,
all Sentinel-era; ~89% are 2019, ~9% 2018, ~1% 2017).

Reproduce:
    python3 -m olmoearth_pretrain.open_set_segmentation_data.datasets.dynamic_world_expert_training_labels
"""

import argparse
import glob
import multiprocessing
import os
import re
from collections import Counter
from typing import Any

import numpy as np
import rasterio
import tqdm
from rasterio.crs import CRS
from rasterio.windows import Window
from rslearn.utils.geometry import Projection
from rslearn.utils.mp import star_imap_unordered

from olmoearth_pretrain.open_set_segmentation_data import (
    download,
    io,
    manifest,
    sampling,
)

SLUG = "dynamic_world_expert_training_labels"
NAME = "Dynamic World Expert Training Labels"
DOI = "https://doi.org/10.1594/PANGAEA.933475"
PANGAEA_ID = "933475"
ZIP_URL = f"https://download.pangaea.de/dataset/{PANGAEA_ID}/files/Experts_tiles.zip"
README_URL = f"https://download.pangaea.de/dataset/{PANGAEA_ID}/files/README.txt"

RAW = io.raw_dir(SLUG)
EXTRACT_DIR = RAW / "extracted"
EXPERTS_GLOB = os.path.join(EXTRACT_DIR.path, "Experts", "**", "*.tif")

TILE = io.MAX_TILE  # 64
RES = 10.0
SRC_TILE_PX = 510
LABELED_FRAC_MIN = 0.25  # keep a window only if >=25% of its pixels carry a real class
PER_CLASS = 1000
SEED = 42

# (source raster value, output name, description). Output id = index into this list.
# Descriptions are the Dynamic World Tier-1 class definitions (Brown et al. 2022).
CLASSES: list[tuple[int, str, str]] = [
    (
        1,
        "Water",
        "Permanent and seasonal water bodies: rivers, ponds, lakes, oceans, reservoirs, and "
        "open water without emergent vegetation.",
    ),
    (
        2,
        "Trees",
        "Significant clustering of tall (~15 m+) dense vegetation over a large area: forests, "
        "dense wooded areas, tree plantations (incl. oil palm), mangroves at tree height.",
    ),
    (
        3,
        "Grass",
        "Open areas of homogeneous grasses with little to no taller vegetation: natural "
        "meadows, savannas with low/no tree cover, parks, lawns, golf courses.",
    ),
    (
        4,
        "Flooded vegetation",
        "Vegetation with obvious intermixing of water for most of the year: seasonally "
        "flooded vegetation, emergent wetland vegetation, rice paddies and other heavily "
        "irrigated/inundated agriculture.",
    ),
    (
        5,
        "Crops",
        "Human-planted/cultivated cereals, grasses, and crops below tree height: row crops "
        "and planted grasslands grown for harvest or grazing.",
    ),
    (
        6,
        "Shrub & scrub",
        "Small clusters or single plants dispersed on a landscape showing exposed soil/rock: "
        "scrubland, bushland, moderate-to-sparse vegetation cover, scrub clearings in forest.",
    ),
    (
        7,
        "Built area",
        "Human-made structures and impervious surfaces: dense villages/towns/cities, major "
        "road and rail networks, paved surfaces, large homogeneous impervious areas.",
    ),
    (
        8,
        "Bare ground",
        "Rock or soil with very sparse to no vegetation across the year: sandy/rocky areas, "
        "dried lake beds, exposed rock, exposed soil, deserts.",
    ),
    (
        9,
        "Snow & ice",
        "Large homogeneous areas of thick snow or ice, typically in mountain regions or the "
        "highest latitudes.",
    ),
]
NUM_CLASSES = len(CLASSES)  # 9
# Remap lookup: source value -> output id (1..9 -> 0..8), everything else -> nodata 255.
_REMAP = np.full(256, io.CLASS_NODATA, dtype=np.uint8)
for _i, (_srcval, _n, _d) in enumerate(CLASSES):
    _REMAP[_srcval] = _i

_FNAME_RE = re.compile(r"dw_(-?[0-9.]+)_(-?[0-9.]+)-(\d{4})(\d{2})(\d{2})\.tif$")


def _ensure_raw() -> None:
    """Download + extract the Experts tiles zip (idempotent)."""
    io.check_disk()
    RAW.mkdir(parents=True, exist_ok=True)
    download.download_http(README_URL, RAW / "README.txt")
    zip_path = RAW / "Experts_tiles.zip"
    download.download_http(ZIP_URL, zip_path)
    if not glob.glob(EXPERTS_GLOB, recursive=True):
        download.extract_zip(zip_path, EXTRACT_DIR, skip_existing=False)


def _tile_year(path: str) -> int:
    m = _FNAME_RE.search(os.path.basename(path))
    if not m:
        raise ValueError(f"cannot parse date from {path}")
    return int(m.group(3))


def _scan_tile(path: str) -> list[dict[str, Any]]:
    """Cut one 510x510 source tile into <=64x64 windows; keep sufficiently-labeled ones.

    Returns lightweight metadata per kept window (no arrays); arrays are re-read for the
    selected subset in the write phase.
    """
    year = _tile_year(path)
    out: list[dict[str, Any]] = []
    with rasterio.open(path) as ds:
        crs_str = ds.crs.to_string()
        b = ds.bounds  # (left, bottom, right, top) in UTM metres
        # rslearn Projection pixel coords: col = x/RES, row = -y/RES (north-up, y decreasing
        # downward). Snap the tile origin (left, top) to integer pixel indices.
        col_base = int(round(b.left / RES))
        row_base = int(round(-b.top / RES))
        for gr in range(0, SRC_TILE_PX, TILE):
            h = min(TILE, SRC_TILE_PX - gr)
            for gc in range(0, SRC_TILE_PX, TILE):
                w = min(TILE, SRC_TILE_PX - gc)
                arr = ds.read(1, window=Window(gc, gr, w, h))
                out_arr = _REMAP[arr]
                labeled = int((out_arr != io.CLASS_NODATA).sum())
                if labeled < LABELED_FRAC_MIN * w * h:
                    continue
                present = sorted(
                    int(v) for v in np.unique(out_arr) if v != io.CLASS_NODATA
                )
                if not present:
                    continue
                col0 = col_base + gc
                row0 = row_base + gr
                out.append(
                    {
                        "path": path,
                        "gc": gc,
                        "gr": gr,
                        "w": w,
                        "h": h,
                        "crs": crs_str,
                        "bounds": (col0, row0, col0 + w, row0 + h),
                        "classes_present": present,
                        "year": year,
                        "source_id": f"{os.path.relpath(path, EXTRACT_DIR.path)}#r{gr}c{gc}",
                    }
                )
    return out


def _write_one(rec: dict[str, Any]) -> tuple[str, list[int]]:
    sid = rec["sample_id"]
    if (io.locations_dir(SLUG) / f"{sid}.tif").exists():
        return sid, rec["classes_present"]
    with rasterio.open(rec["path"]) as ds:
        arr = ds.read(1, window=Window(rec["gc"], rec["gr"], rec["w"], rec["h"]))
    out_arr = _REMAP[arr]
    proj = Projection(CRS.from_string(rec["crs"]), RES, -RES)
    bounds = tuple(rec["bounds"])
    io.write_label_geotiff(SLUG, sid, out_arr, proj, bounds, nodata=io.CLASS_NODATA)
    io.write_sample_json(
        SLUG,
        sid,
        proj,
        bounds,
        io.year_range(rec["year"]),
        source_id=rec["source_id"],
        classes_present=rec["classes_present"],
    )
    return sid, rec["classes_present"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=64)
    args = parser.parse_args()

    io.check_disk()
    manifest.write_registry_entry(SLUG, "in_progress")
    _ensure_raw()

    tiles = sorted(glob.glob(EXPERTS_GLOB, recursive=True))
    if not tiles:
        raise FileNotFoundError(f"No expert tiles under {EXTRACT_DIR}")
    print(f"scanning {len(tiles)} expert source tiles into <=64x64 windows")

    # ---- Phase 1: scan tiles into candidate windows (parallel) -----------------------
    cands: list[dict[str, Any]] = []
    with multiprocessing.Pool(args.workers) as pool:
        for recs in tqdm.tqdm(
            star_imap_unordered(pool, _scan_tile, [dict(path=p) for p in tiles]),
            total=len(tiles),
        ):
            cands.extend(recs)
    print(
        f"candidate windows (>= {int(LABELED_FRAC_MIN * 100)}% labeled): {len(cands)}"
    )
    avail = Counter()
    for r in cands:
        for cid in r["classes_present"]:
            avail[cid] += 1
    print("candidate windows per class:")
    for i, (_sv, name, _d) in enumerate(CLASSES):
        print(f"  {i:>2} {name:20} {avail.get(i, 0)}")

    # ---- Phase 2: tiles-per-class balanced selection ---------------------------------
    selected = sampling.select_tiles_per_class(
        cands,
        classes_key="classes_present",
        per_class=PER_CLASS,
        total_cap=sampling.MAX_SAMPLES_PER_DATASET,
        seed=SEED,
    )
    selected.sort(key=lambda r: r["source_id"])
    for i, r in enumerate(selected):
        r["sample_id"] = f"{i:06d}"
    print(f"selected {len(selected)} windows (<= {PER_CLASS}/class, 25k cap)")

    # ---- Phase 3: write patches (parallel) -------------------------------------------
    tile_counts = Counter()
    with multiprocessing.Pool(args.workers) as pool:
        done = 0
        for _sid, present in tqdm.tqdm(
            star_imap_unordered(pool, _write_one, [dict(rec=r) for r in selected]),
            total=len(selected),
        ):
            for cid in present:
                tile_counts[cid] += 1
            done += 1
            if done % 2000 == 0:
                io.check_disk()

    io.write_dataset_metadata(
        SLUG,
        {
            "dataset": SLUG,
            "name": NAME,
            "task_type": "classification",
            "source": "PANGAEA / Nature Sci Data (Dynamic World training data, Tait et al. 2021)",
            "license": "CC-BY-4.0",
            "provenance": {
                "url": DOI,
                "pangaea_id": PANGAEA_ID,
                "have_locally": False,
                "annotation_method": (
                    "manual dense markup by 25 expert human labelers (National Geographic "
                    "Society), visual interpretation of Sentinel-2 L2A true-color composites; "
                    "Experts subset only"
                ),
                "attribution": (
                    "Produced for the Dynamic World Project by National Geographic Society in "
                    "partnership with Google and the World Resources Institute; training-data "
                    "development funded in part by the Gordon and Betty Moore Foundation."
                ),
            },
            "sensors_relevant": ["sentinel2", "sentinel1", "landsat"],
            "classes": [
                {"id": i, "name": name, "description": desc, "source_value": srcval}
                for i, (srcval, name, desc) in enumerate(CLASSES)
            ],
            "nodata_value": io.CLASS_NODATA,
            "num_samples": len(selected),
            "class_tile_counts": {
                CLASSES[i][1]: int(tile_counts.get(i, 0)) for i in range(NUM_CLASSES)
            },
            "notes": (
                "Expert-labeled Dynamic World Tier-1 land cover. Source 510x510 single-band "
                "uint8 GeoTIFFs are already local UTM at 10 m; cut into <=64x64 windows (8x8 "
                "grid, edge windows 62 px). Source values 1..9 -> ids 0..8; 0 (unmarked) and 10 "
                "(cloud) -> nodata 255. Kept windows with >=25% labeled pixels. Tiles-per-class "
                "balanced (rarest first), <=1000 windows/class, 25k cap. Time range = 1-year "
                "window on the tile's acquisition year (2017-2019, all Sentinel-era). Only the "
                "Experts folder is used (Non_expert crowd tiles and the validation holdout are "
                "excluded)."
            ),
        },
    )
    print("written windows per class:")
    for i, (_sv, name, _d) in enumerate(CLASSES):
        print(f"  {i:>2} {name:20} {tile_counts.get(i, 0)}")

    manifest.write_registry_entry(
        SLUG, "completed", task_type="classification", num_samples=len(selected)
    )
    print("done")


if __name__ == "__main__":
    multiprocessing.set_start_method("forkserver", force=True)
    main()
