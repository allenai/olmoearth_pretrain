"""Process USGS USMIN Mine Features (POLYGON footprints) into segmentation label patches.

Source: USGS Mineral Resources "Prospect- and Mine-Related Features from U.S. Geological
Survey 7.5- and 15-Minute Topographic Quadrangle Maps" (USMIN), version 10.0 (May 2023),
public domain. Downloaded as the national File Geodatabase from ScienceBase:

  https://www.sciencebase.gov/catalog/file/get/5a1492c3e4b09fc93dcfd574?name=USGS_TopoMineSymbols_ver10_Geodatabase.zip
  (project page: https://mrdata.usgs.gov/usmin/)

The GDB holds point + polygon feature classes digitized from historical topographic maps,
at three source map scales: 1:24,000 (24k), 1:48,000 / 15-minute (48k), and 1:625,000
(625k). We use only the **24k and 48k** POLYGON layers (positional accuracy adequate for a
10 m grid); the **625k** layers are dropped (their ~hundreds-of-metres positional error
makes them unusable for 10 m label tiles). See the summary for the full rationale.

This dataset is POLYGON-ONLY: polygon features (real footprints) are RASTERIZED into a
<=64x64 UTM 10 m tile. The presence-only POINT markers (prospect pits, mine shafts, adits,
etc.) live in the sibling dataset ``usgs_usmin_mine_features_points`` and are NOT written
here.

Each feature carries a ``Ftr_Type`` (feature-type symbol). Only feature types that actually
occur as polygons are kept; point-only types (mine shaft, adit) are dropped from the class
map. Class ids are contiguous 0..N with 0 = background.

Class scheme (id 0 = background; 255 = nodata/ignore):
  0 background            5 tailings_pile
  1 prospect_pit          6 tailings_pond
  2 quarry_open_pit       7 mine_dump
  3 gravel_borrow_pit     8 disturbed_surface
  4 strip_mine

Time range: these are persistent, undated (map-digitized) features. Per spec §5 (static
labels), each sample gets a 1-year window at a representative Sentinel-era year, spread
pseudo-randomly across 2016-2022 for temporal diversity.

Run (idempotent; skips already-written tiles):
  python3 -m olmoearth_pretrain.open_set_segmentation_data.datasets.usgs_usmin_mine_features
"""

import argparse
import multiprocessing
import random
from collections import Counter
from typing import Any

import fiona
import numpy as np
import shapely
import tqdm
from rslearn.const import WGS84_PROJECTION
from rslearn.utils.mp import star_imap_unordered

from olmoearth_pretrain.open_set_segmentation_data import io, manifest
from olmoearth_pretrain.open_set_segmentation_data.rasterize import (
    geom_to_pixels,
    rasterize_shapes,
)
from olmoearth_pretrain.open_set_segmentation_data.sampling import balance_by_class

SLUG = "usgs_usmin_mine_features"
NAME = "USGS USMIN Mine Features"
SB_ITEM = "5a1492c3e4b09fc93dcfd574"
DOWNLOAD_URL = (
    "https://www.sciencebase.gov/catalog/file/get/"
    f"{SB_ITEM}?name=USGS_TopoMineSymbols_ver10_Geodatabase.zip"
)
GDB = "USGS_TopoMineSymbols_ver10_Geodatabase/USGS_TopoMineSymbols_ver10.gdb"
POLY_LAYERS = ["USGS_TopoMineSymbols_24k_Polygons", "USGS_TopoMineSymbols_48k_Polygons"]

# Class scheme. id 0 reserved for background. Only feature types that occur as polygons.
CID_BACKGROUND = 0
CLASSES = [
    {
        "id": 0,
        "name": "background",
        "description": "Negative / non-mine land: pixels outside any mapped mine feature.",
    },
    {
        "id": 1,
        "name": "prospect_pit",
        "description": "Small exploratory prospect pit or diggings (test excavation). "
        "Only the small subset with mapped polygon footprints is included here.",
    },
    {
        "id": 2,
        "name": "quarry_open_pit",
        "description": "Quarry or open-pit mine (rock/limestone/gypsum/pumice quarries, "
        "open-pit mines). Polygon footprints rasterized at 10 m.",
    },
    {
        "id": 3,
        "name": "gravel_borrow_pit",
        "description": "Gravel, sand, or borrow pit (surface aggregate extraction). "
        "Polygon footprints rasterized at 10 m.",
    },
    {
        "id": 4,
        "name": "strip_mine",
        "description": "Strip mine (surface/contour mining), large disturbed extraction area. "
        "Polygon footprints rasterized at 10 m.",
    },
    {
        "id": 5,
        "name": "tailings_pile",
        "description": "Tailings/waste pile (undifferentiated, placer, dredge, mill tailings) "
        "or slag pile. Polygon footprints rasterized at 10 m.",
    },
    {
        "id": 6,
        "name": "tailings_pond",
        "description": "Tailings pond, settling/leach/evaporation pond, or salt evaporator "
        "(impounded process water). Polygon footprints rasterized at 10 m.",
    },
    {
        "id": 7,
        "name": "mine_dump",
        "description": "Mine dump / ore stockpile (waste rock or ore storage). "
        "Polygon footprints rasterized at 10 m.",
    },
    {
        "id": 8,
        "name": "disturbed_surface",
        "description": "Mining-disturbed surface, disturbed-surface pit, or trench "
        "(bare disturbed ground). Polygon footprints rasterized at 10 m.",
    },
]
N_FEATURE_CLASSES = len(CLASSES) - 1  # excludes background

# Map raw Ftr_Type -> class id. Only polygon-bearing types; unmapped types are dropped
# (documented in summary). Point-only types (Mine Shaft/Air Shaft/Adit) are excluded.
FTR_TYPE_TO_CLASS = {
    # 1 prospect_pit
    "Prospect Pit": 1,
    "Diggings": 1,
    "Glory Hole": 1,
    # 2 quarry_open_pit
    "Quarry": 2,
    "Quarry - Rock": 2,
    "Quarry - Limestone": 2,
    "Quarry - Gypsum": 2,
    "Quarry - Pumice": 2,
    "Open Pit Mine": 2,
    "Open Pit Mine or Quarry": 2,
    # 3 gravel_borrow_pit
    "Gravel Pit": 3,
    "Borrow Pit": 3,
    "Sand Pit": 3,
    "Sand and Gravel Pit": 3,
    "Gravel/Borrow Pit - Undifferentiated": 3,
    # 4 strip_mine
    "Strip Mine": 4,
    # 5 tailings_pile
    "Tailings - Undifferentiated": 5,
    "Tailings - Placer": 5,
    "Tailings - Dredge": 5,
    "Tailings - Mill": 5,
    "Slag Pile": 5,
    # 6 tailings_pond
    "Tailings - Pond": 6,
    "Settling Pond": 6,
    "Leach Pond": 6,
    "Evaporation Pond": 6,
    "Salt Evaporator": 6,
    # 7 mine_dump
    "Mine Dump": 7,
    "Ore Stockpile/Storage": 7,
    # 8 disturbed_surface
    "Disturbed Surface": 8,
    "Disturbed Surface - Pit": 8,
    "Trench": 8,
}

# Sampling parameters.
PER_CLASS = 1000
YEARS = list(range(2016, 2023))  # representative Sentinel-era 1-year windows

MAX_POLY_TILE = io.MAX_TILE  # 64


def gdb_path() -> str:
    return str(io.raw_dir(SLUG) / GDB)


# --------------------------------------------------------------------------------------
# Reading source features.
# --------------------------------------------------------------------------------------
def read_polygons() -> list[dict[str, Any]]:
    """Read mapped polygon features into records with centroid lon/lat + geometry WKB."""
    recs: list[dict[str, Any]] = []
    for layer in POLY_LAYERS:
        with fiona.open(gdb_path(), layer=layer) as src:
            for i, feat in enumerate(src):
                cid = FTR_TYPE_TO_CLASS.get(feat["properties"].get("Ftr_Type"))
                if cid is None or feat["geometry"] is None:
                    continue
                try:
                    geom = shapely.geometry.shape(feat["geometry"])
                except Exception:
                    continue
                if geom.is_empty or not geom.is_valid:
                    geom = geom.buffer(0) if not geom.is_empty else geom
                    if geom.is_empty:
                        continue
                c = geom.centroid
                recs.append(
                    {
                        "kind": "polygon",
                        "class_id": cid,
                        "lon": float(c.x),
                        "lat": float(c.y),
                        "geom_wkb": shapely.to_wkb(geom),
                        "source_id": f"{layer}/{i}",
                    }
                )
    return recs


# --------------------------------------------------------------------------------------
# Writers (worker processes).
# --------------------------------------------------------------------------------------
def _write_polygon(rec: dict[str, Any]) -> str:
    sample_id = rec["sample_id"]
    tif = io.locations_dir(SLUG) / f"{sample_id}.tif"
    if tif.exists():
        return "skip"
    proj = io.utm_projection_for_lonlat(rec["lon"], rec["lat"])
    geom = shapely.from_wkb(rec["geom_wkb"])
    pix = geom_to_pixels(geom, WGS84_PROJECTION, proj)
    minx, miny, maxx, maxy = pix.bounds
    cx = int(round((minx + maxx) / 2))
    cy = int(round((miny + maxy) / 2))
    w = min(MAX_POLY_TILE, max(1, int(np.ceil(maxx - minx))))
    h = min(MAX_POLY_TILE, max(1, int(np.ceil(maxy - miny))))
    bounds = io.centered_bounds(cx, cy, w, h)
    arr = rasterize_shapes(
        [(pix, rec["class_id"])],
        bounds,
        fill=CID_BACKGROUND,
        dtype="uint8",
        all_touched=True,
    )
    io.write_label_geotiff(SLUG, sample_id, arr, proj, bounds, nodata=io.CLASS_NODATA)
    io.write_sample_json(
        SLUG,
        sample_id,
        proj,
        bounds,
        io.year_range(rec["year"]),
        source_id=rec["source_id"],
        classes_present=sorted(set(np.unique(arr).tolist()) - {io.CLASS_NODATA}),
    )
    return "polygon"


# --------------------------------------------------------------------------------------
# Selection.
# --------------------------------------------------------------------------------------
def select_records(polygons: list[dict[str, Any]], seed: int = 42) -> list[dict[str, Any]]:
    """Up to PER_CLASS polygon records per feature class (balanced, seeded)."""
    return balance_by_class(polygons, "class_id", per_class=PER_CLASS, seed=seed)


# --------------------------------------------------------------------------------------
# Main.
# --------------------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=64)
    args = parser.parse_args()

    io.check_disk()
    manifest.write_registry_entry(SLUG, "in_progress")

    raw = io.raw_dir(SLUG)
    raw.mkdir(parents=True, exist_ok=True)
    with (raw / "SOURCE.txt").open("w") as f:
        f.write(
            "USGS USMIN 'Prospect- and Mine-Related Features from USGS 7.5- and "
            "15-Minute Topographic Quadrangle Maps', version 10.0 (May 2023). "
            "Public domain.\n"
            f"ScienceBase item {SB_ITEM}\n{DOWNLOAD_URL}\n"
            f"National File Geodatabase; using 24k + 48k POLYGON layers "
            "(625k layers excluded: positional error too large for 10 m tiles).\n"
        )

    print("reading polygon features ...")
    polygons = read_polygons()
    print(f"  {len(polygons)} mapped polygon features")

    io.check_disk()

    selected = select_records(polygons)

    # Assign representative years (spread across Sentinel era).
    rng = random.Random(123)
    for r in selected:
        r["year"] = YEARS[rng.randrange(len(YEARS))]
    for i, r in enumerate(selected):
        r["sample_id"] = f"{i:06d}"

    # Report selection counts.
    sel_counts: Counter = Counter()
    for r in selected:
        sel_counts[r["class_id"]] += 1
    id_to_name = {c["id"]: c["name"] for c in CLASSES}
    print(f"selected {len(selected)} polygon tiles")
    for cid in sorted(sel_counts):
        print(f"  {sel_counts[cid]:5d}  {id_to_name[cid]:20s}")

    results: Counter = Counter()
    with multiprocessing.Pool(args.workers) as p:
        for res in tqdm.tqdm(
            star_imap_unordered(p, _write_polygon, [dict(rec=r) for r in selected]),
            total=len(selected),
        ):
            results[res] += 1
    print("write results:", dict(results))

    io.check_disk()

    class_counts = {
        id_to_name[cid]: sel_counts.get(cid, 0) for cid in range(1, len(CLASSES))
    }
    io.write_dataset_metadata(
        SLUG,
        {
            "dataset": SLUG,
            "name": NAME,
            "task_type": "classification",
            "source": "USGS (ScienceBase)",
            "license": "public domain",
            "provenance": {
                "url": "https://mrdata.usgs.gov/usmin/",
                "sciencebase_item": SB_ITEM,
                "download_url": DOWNLOAD_URL,
                "have_locally": False,
                "annotation_method": "manual digitizing of mine symbols from historical "
                "USGS topographic quadrangle maps",
                "version": "10.0 (May 2023)",
            },
            "sensors_relevant": ["sentinel2", "sentinel1", "landsat"],
            "classes": CLASSES,
            "nodata_value": io.CLASS_NODATA,
            "num_samples": len(selected),
            "class_counts": class_counts,
            "notes": (
                "Polygon-only dataset. Polygon mine footprints rasterized into <=64x64 "
                "UTM 10 m tiles (footprint centered; >640 m footprints keep the central "
                "64x64), background=0, nodata=255. Balanced to <=1000 polygons per class. "
                "Layers used: 24k + 48k polygon (625k dropped for poor positional "
                "accuracy). Only feature types that occur as polygons are kept; the "
                "point-only marker classes (mine_shaft, adit) and all point-encoded / "
                "negative tiles were moved to the sibling dataset "
                "usgs_usmin_mine_features_points. Unmapped minor Ftr_Type values "
                "(clay/cinder/shale/caliche/scoria/chert/marl/bentonite/shell/iron/"
                "lignite pits, generic Mine, coal/uranium/placer/hydraulic mines, mill "
                "site, tipple) dropped. Persistent features -> 1-year window at a "
                "representative Sentinel-era year (2016-2022)."
            ),
        },
    )
    manifest.write_registry_entry(
        SLUG, "completed", task_type="classification", num_samples=len(selected)
    )
    print("done:", len(selected), "samples")


if __name__ == "__main__":
    multiprocessing.set_start_method("forkserver", force=True)
    main()
