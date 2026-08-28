"""Process USGS USMIN Mine Features (POINT markers) into a presence-only point dataset.

Source: USGS Mineral Resources "Prospect- and Mine-Related Features from U.S. Geological
Survey 7.5- and 15-Minute Topographic Quadrangle Maps" (USMIN), version 10.0 (May 2023),
public domain. See the sibling polygon dataset ``usgs_usmin_mine_features`` for the full
provenance. This script reuses the SAME raw File Geodatabase (shared raw dir); it does not
re-download anything.

The GDB holds point + polygon feature classes digitized from historical topographic maps at
1:24,000 (24k), 1:48,000 / 15-minute (48k), and 1:625,000 (625k) source scales. We use only
the **24k and 48k** POINT layers; the 625k layers are dropped (positional error too large
for a 10 m grid).

This dataset is POINT-ONLY and PRESENCE-ONLY: each mapped mine-symbol point is a single
labeled location (no footprint). Rather than fabricated detection tiles, the points are
written to one dataset-wide GeoJSON point table (spec §2a), balanced to <=1000 per class.
There is NO background class -- every point is a positive presence of its feature type.

Class scheme (contiguous ids 0..8; the distinct feature types that occur as points):
  0 prospect_pit     3 quarry_open_pit    6 tailings_pile
  1 mine_shaft       4 gravel_borrow_pit  7 tailings_pond
  2 adit             5 strip_mine         8 mine_dump

Time range: these are persistent, undated (map-digitized) features. Per spec §5 (static
labels), each point gets a 1-year window at a representative Sentinel-era year, spread
pseudo-randomly across 2016-2022 for temporal diversity.

Run:
  python3 -m olmoearth_pretrain.open_set_segmentation_data.datasets.usgs_usmin_mine_features_points
"""

import argparse
import multiprocessing
import random
from collections import Counter
from typing import Any

import fiona

from olmoearth_pretrain.open_set_segmentation_data import io, manifest
from olmoearth_pretrain.open_set_segmentation_data.datasets.usgs_usmin_mine_features import (
    GDB,
    SB_ITEM,
    DOWNLOAD_URL,
)
from olmoearth_pretrain.open_set_segmentation_data.sampling import balance_by_class

SLUG = "usgs_usmin_mine_features_points"
NAME = "USGS USMIN Mine Features (points)"
# Raw source is shared with the polygon dataset (do NOT re-download).
RAW_SLUG = "usgs_usmin_mine_features"
POINT_LAYERS = ["USGS_TopoMineSymbols_24k_Points", "USGS_TopoMineSymbols_48k_Points"]

# Class scheme -- distinct feature-type classes that occur as POINT markers. NO background.
CLASSES = [
    {
        "id": 0,
        "name": "prospect_pit",
        "description": "Small exploratory prospect pit or diggings (test excavation), "
        "typically sub-10 m -- a presence marker only.",
    },
    {
        "id": 1,
        "name": "mine_shaft",
        "description": "Vertical mine shaft or air shaft; typically sub-10 m at the "
        "surface -- a presence marker only.",
    },
    {
        "id": 2,
        "name": "adit",
        "description": "Horizontal mine entrance (adit) driven into a hillside; typically "
        "sub-10 m -- a presence marker only.",
    },
    {
        "id": 3,
        "name": "quarry_open_pit",
        "description": "Quarry or open-pit mine (rock/limestone/gypsum/pumice quarries, "
        "open-pit mines) mapped as a point marker.",
    },
    {
        "id": 4,
        "name": "gravel_borrow_pit",
        "description": "Gravel, sand, or borrow pit (surface aggregate extraction) mapped "
        "as a point marker.",
    },
    {
        "id": 5,
        "name": "strip_mine",
        "description": "Strip mine (surface/contour mining) mapped as a point marker.",
    },
    {
        "id": 6,
        "name": "tailings_pile",
        "description": "Tailings/waste pile (undifferentiated, placer, dredge, mill "
        "tailings) or slag pile mapped as a point marker.",
    },
    {
        "id": 7,
        "name": "tailings_pond",
        "description": "Tailings pond, settling/leach/evaporation pond, or salt evaporator "
        "mapped as a point marker.",
    },
    {
        "id": 8,
        "name": "mine_dump",
        "description": "Mine dump / ore stockpile (waste rock or ore storage) mapped as a "
        "point marker.",
    },
]

# Map raw Ftr_Type -> point class id (contiguous 0..8, no background). Unmapped types are
# dropped (including disturbed-surface types, which do not occur as points).
FTR_TYPE_TO_CLASS = {
    # 0 prospect_pit
    "Prospect Pit": 0,
    "Diggings": 0,
    "Glory Hole": 0,
    # 1 mine_shaft
    "Mine Shaft": 1,
    "Air Shaft": 1,
    # 2 adit
    "Adit": 2,
    # 3 quarry_open_pit
    "Quarry": 3,
    "Quarry - Rock": 3,
    "Quarry - Limestone": 3,
    "Quarry - Gypsum": 3,
    "Quarry - Pumice": 3,
    "Open Pit Mine": 3,
    "Open Pit Mine or Quarry": 3,
    # 4 gravel_borrow_pit
    "Gravel Pit": 4,
    "Borrow Pit": 4,
    "Sand Pit": 4,
    "Sand and Gravel Pit": 4,
    "Gravel/Borrow Pit - Undifferentiated": 4,
    # 5 strip_mine
    "Strip Mine": 5,
    # 6 tailings_pile
    "Tailings - Undifferentiated": 6,
    "Tailings - Placer": 6,
    "Tailings - Dredge": 6,
    "Tailings - Mill": 6,
    "Slag Pile": 6,
    # 7 tailings_pond
    "Tailings - Pond": 7,
    "Settling Pond": 7,
    "Leach Pond": 7,
    "Evaporation Pond": 7,
    "Salt Evaporator": 7,
    # 8 mine_dump
    "Mine Dump": 8,
    "Ore Stockpile/Storage": 8,
}

PER_CLASS = 1000
YEARS = list(range(2016, 2023))  # representative Sentinel-era 1-year windows


def gdb_path() -> str:
    return str(io.raw_dir(RAW_SLUG) / GDB)


def read_points() -> list[dict[str, Any]]:
    """Read mapped point features into records with lon/lat + class id."""
    recs: list[dict[str, Any]] = []
    for layer in POINT_LAYERS:
        with fiona.open(gdb_path(), layer=layer) as src:
            for i, feat in enumerate(src):
                cid = FTR_TYPE_TO_CLASS.get(feat["properties"].get("Ftr_Type"))
                if cid is None or feat["geometry"] is None:
                    continue
                lon, lat = feat["geometry"]["coordinates"][:2]
                recs.append(
                    {
                        "class_id": cid,
                        "lon": float(lon),
                        "lat": float(lat),
                        "source_id": f"{layer}/{i}",
                    }
                )
    return recs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=64)
    parser.parse_args()

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
            "POINT markers only. Reuses the shared raw File Geodatabase downloaded for "
            f"dataset '{RAW_SLUG}' at {io.raw_dir(RAW_SLUG)} (not re-downloaded here); "
            "using 24k + 48k point layers (625k dropped for poor positional accuracy).\n"
        )

    print("reading point features ...")
    points = read_points()
    print(f"  {len(points)} mapped point features")

    io.check_disk()

    selected = balance_by_class(points, "class_id", per_class=PER_CLASS)

    # Assign representative years (spread across Sentinel era) for a static 1-year window.
    rng = random.Random(123)
    for r in selected:
        r["year"] = YEARS[rng.randrange(len(YEARS))]

    id_to_name = {c["id"]: c["name"] for c in CLASSES}
    sel_counts: Counter = Counter(r["class_id"] for r in selected)
    print(f"selected {len(selected)} points (<= {PER_CLASS}/class)")
    for cid in sorted(sel_counts):
        print(f"  {sel_counts[cid]:5d}  {id_to_name[cid]:20s}")

    # Presence-only sparse point dataset -> one dataset-wide point table (spec §2a).
    points_out = []
    for i, r in enumerate(selected):
        points_out.append(
            {
                "id": f"{i:06d}",
                "lon": r["lon"],
                "lat": r["lat"],
                "label": r["class_id"],
                "time_range": io.year_range(r["year"]),
                "change_time": None,
                "source_id": r["source_id"],
            }
        )
    io.write_points_table(SLUG, "classification", points_out)

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
                "shared_raw_dataset": RAW_SLUG,
            },
            "sensors_relevant": ["sentinel2", "sentinel1", "landsat"],
            "classes": CLASSES,
            "nodata_value": io.CLASS_NODATA,
            "num_samples": len(selected),
            "class_counts": {
                id_to_name[cid]: sel_counts.get(cid, 0) for cid in sorted(id_to_name)
            },
            "notes": (
                "Presence-only POINT-marker dataset (companion to the polygon dataset "
                "usgs_usmin_mine_features). Each mapped mine symbol is one labeled "
                "location written to a dataset-wide points.geojson table (spec 2a); there "
                "is NO background class -- every point is a positive presence of its "
                "feature type. Multi-class: the distinct feature types that occur as "
                "points, ids 0..8. Balanced to <=1000 points per class. Layers used: 24k "
                "+ 48k point (625k dropped for poor positional accuracy). Raw File "
                f"Geodatabase reused from the shared '{RAW_SLUG}' raw dir (not "
                "re-downloaded). Unmapped minor Ftr_Type values dropped; disturbed-surface "
                "types do not occur as points. Persistent features -> static 1-year window "
                "at a representative Sentinel-era year (2016-2022); change_time is null."
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
