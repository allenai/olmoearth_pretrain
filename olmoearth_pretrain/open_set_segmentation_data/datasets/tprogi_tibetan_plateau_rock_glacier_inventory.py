"""Process the TPRoGI (Tibetan Plateau Rock Glacier Inventory) into open-set segmentation labels.

Source: Zenodo record 10732042, Wang et al., "TPRoGI: a comprehensive rock glacier
inventory for the Tibetan Plateau using deep learning" (ESSD). Compiled by referring to
the IPA RGIK guidelines v1.0 (RGIK, 2023). Two shapefiles (EPSG:4326):
  * ``TPRoGI_Extended_Footprint.shp`` -- 44,273 extended-outline (footprint) polygons of
    each rock glacier; the landform footprints we rasterize.
  * ``TPRoGI_Primary_Marker.shp``     -- 44,273 primary-marker points (point location);
    not used here (footprint carries LAT/LON attributes and geometry).
Footprint attribute table carries morphometric/climate covariates (AREA, elevation,
slope, aspect, MAAT/MAGT/precip); there is NO activity sub-classification in TPRoGI, so
unlike the RGIK RoGI precedent (active/transitional/relict) this is a **single-class**
inventory: rock-glacier presence.

Task: positive-only classification of rock-glacier presence (one class). We rasterize each
extended-footprint polygon into a 64x64 UTM 10 m tile centered on the polygon's
representative point; pixels inside the outline carry class id 0 (rock_glacier), everything
outside is 255 (nodata/ignore). Per spec section 5 this is a positive-only landform: we do
NOT fabricate negatives; non-object pixels are left as nodata and the assembly step draws
negatives from other datasets. Mirrors the RGIK RoGI and debris-covered-glaciers recipes.

Sampling: single class, tiles-per-class balanced -> up to PER_CLASS (1000) footprints
selected (deterministic, seeded) out of 44,273. Median footprint is ~300 m across (fits a
640 m tile); ~large outlines (max ~2.1 km) are clipped to the 64x64 window, yielding a
homogeneous interior tile (same behaviour as the RGIK precedent).

Time range: rock glaciers are slow, persistent landforms; TPRoGI was mapped from 2021
(Q3/Q4) imagery (MAP_DATE). We assign every sample a uniform 1-year window (2021).

Run: python3 -m olmoearth_pretrain.open_set_segmentation_data.datasets.tprogi_tibetan_plateau_rock_glacier_inventory
"""

import argparse
import multiprocessing
import warnings
from collections import Counter
from typing import Any

import numpy as np
import tqdm
from rslearn.const import WGS84_PROJECTION
from rslearn.utils.mp import star_imap_unordered

from olmoearth_pretrain.open_set_segmentation_data import io
from olmoearth_pretrain.open_set_segmentation_data.rasterize import (
    geom_to_pixels,
    rasterize_shapes,
)
from olmoearth_pretrain.open_set_segmentation_data.sampling import balance_by_class

warnings.filterwarnings("ignore")

SLUG = "tprogi_tibetan_plateau_rock_glacier_inventory"
NAME = "TPRoGI (Tibetan Plateau Rock Glacier Inventory)"
RAW = io.raw_dir(SLUG)
FOOTPRINT = RAW / "TPRoGI_Extended_Footprint.shp"

TILE = 64
YEAR = 2021
PER_CLASS = 1000

# single positive class
CLASSES = [
    (
        0,
        "rock_glacier",
        "Rock glacier: ice-rich / debris-mantled creeping permafrost landform. Extended "
        "outline (full landform footprint) from the TPRoGI inventory, compiled per IPA "
        "RGIK guidelines v1.0 using deep learning (DeepLabv3+) on remote-sensing imagery.",
    ),
]
ID_TO_NAME = {cid: n for cid, n, _d in CLASSES}


# --------------------------------------------------------------------------- scan


def scan() -> list[dict[str, Any]]:
    """Load footprint polygons; return one record per rock-glacier outline."""
    import geopandas as gpd

    gdf = gpd.read_file(FOOTPRINT.path)
    reps = gdf.geometry.representative_point()  # EPSG:4326 -> lon/lat
    recs: list[dict[str, Any]] = []
    for i in range(len(gdf)):
        row = gdf.iloc[i]
        geom = row.geometry
        if geom is None or geom.is_empty:
            continue
        pt = reps.iloc[i]
        if pt is None or pt.is_empty:
            continue
        recs.append(
            {
                "cid": 0,
                "geom": geom,
                "lon": float(pt.x),
                "lat": float(pt.y),
                "rg_id": str(row["ID"]),
                "subregion": str(row["SUBREGION"])
                if row["SUBREGION"] is not None
                else "",
            }
        )
    return recs


# --------------------------------------------------------------------------- write


def _write_chunk(recs: list[dict[str, Any]]) -> list[tuple[str, int]]:
    """Rasterize + write a chunk of footprint records. Returns (sample_id, cid)."""
    src_proj = WGS84_PROJECTION
    written: list[tuple[str, int]] = []
    for rec in recs:
        sample_id = rec["sample_id"]
        cid = rec["cid"]
        tif = io.locations_dir(SLUG) / f"{sample_id}.tif"
        if tif.exists():
            written.append((sample_id, cid))
            continue
        proj, col, row = io.lonlat_to_utm_pixel(rec["lon"], rec["lat"])
        bounds = io.centered_bounds(col, row, TILE, TILE)
        geom_px = geom_to_pixels(rec["geom"], src_proj, proj)
        arr = rasterize_shapes(
            [(geom_px, cid)],
            bounds,
            fill=io.CLASS_NODATA,
            dtype="uint8",
            all_touched=True,
        )
        if not np.any(arr == cid):
            continue  # polygon missed the window (degenerate); skip
        io.write_label_geotiff(
            SLUG, sample_id, arr, proj, bounds, nodata=io.CLASS_NODATA
        )
        src_id = f"{rec['subregion']}/{rec['rg_id']}"
        io.write_sample_json(
            SLUG,
            sample_id,
            proj,
            bounds,
            io.year_range(YEAR),
            source_id=src_id,
            classes_present=[cid],
        )
        written.append((sample_id, cid))
    return written


# ---------------------------------------------------------------------------- main


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=64)
    args = parser.parse_args()

    io.check_disk()
    if not FOOTPRINT.exists():
        raise RuntimeError(
            f"Footprint shapefile not found at {FOOTPRINT}; run download first."
        )

    with (RAW / "SOURCE.txt").open("w") as f:
        f.write(
            "Zenodo record 10732042 (Wang et al., TPRoGI, ESSD).\n"
            "TPRoGI_Extended_Footprint.shp -> 44,273 rock-glacier footprint polygons "
            "(EPSG:4326). Primary marker shapefile not used.\n"
        )

    recs = scan()
    print(f"scanned {len(recs)} footprint records (single class rock_glacier)")

    # Single-class tiles-per-class balance (deterministic ordering -> stable ids).
    selected = balance_by_class(recs, "cid", per_class=PER_CLASS)
    selected.sort(key=lambda r: (r["cid"], r["rg_id"]))
    for i, r in enumerate(selected):
        r["sample_id"] = f"{i:06d}"
    print(f"selected {len(selected)} footprints (per_class={PER_CLASS})")

    io.check_disk()

    n_workers = max(1, min(args.workers, len(selected)))
    chunks: list[list[dict[str, Any]]] = [[] for _ in range(n_workers)]
    for i, r in enumerate(selected):
        chunks[i % n_workers].append(r)
    jobs = [dict(recs=c) for c in chunks if c]

    written: list[tuple[str, int]] = []
    with multiprocessing.Pool(len(jobs)) as p:
        for res in tqdm.tqdm(
            star_imap_unordered(p, _write_chunk, jobs), total=len(jobs), desc="write"
        ):
            written.extend(res)

    counts = Counter(cid for _sid, cid in written)
    print(f"wrote {len(written)} samples")
    for cid, name, _d in CLASSES:
        print(f"  class {cid} ({name}): {counts.get(cid, 0)}")

    io.write_dataset_metadata(
        SLUG,
        {
            "dataset": SLUG,
            "name": NAME,
            "task_type": "classification",
            "source": "Zenodo (record 10732042)",
            "license": "CC-BY-4.0",
            "provenance": {
                "url": "https://doi.org/10.5281/zenodo.10732042",
                "have_locally": False,
                "annotation_method": "deep learning (DeepLabv3+) on remote-sensing imagery "
                "+ manual verification, per IPA RGIK guidelines v1.0 (Wang et al., ESSD)",
            },
            "sensors_relevant": ["sentinel2", "sentinel1", "landsat"],
            "classes": [
                {"id": cid, "name": name, "description": desc}
                for cid, name, desc in CLASSES
            ],
            "nodata_value": io.CLASS_NODATA,
            "num_samples": len(written),
            "class_counts": {
                ID_TO_NAME[cid]: counts.get(cid, 0) for cid, *_ in CLASSES
            },
            "notes": (
                "Extended-footprint polygons rasterized to 64x64 UTM 10 m tiles; class 0 "
                "(rock_glacier) inside the outline, 255 nodata outside. Positive-only "
                "landform (single class): no negatives fabricated (spec 5); assembly step "
                "supplies negatives from other datasets. 44,273 footprints available; "
                f"{len(written)} sampled (per_class={PER_CLASS}, seeded). Median footprint "
                "~300 m across; large outlines (max ~2.1 km) clipped to the window. Mapped "
                "from 2021 imagery -> uniform 2021 1-year time range. Primary-marker "
                "shapefile not used."
            ),
        },
    )
    print("done")


if __name__ == "__main__":
    multiprocessing.set_start_method("forkserver", force=True)
    main()
