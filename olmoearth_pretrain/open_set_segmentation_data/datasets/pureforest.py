"""Process PureForest into open-set-segmentation label patches (tree species polygons).

Source: PureForest (IGN France), Hugging Face ``IGNF/PureForest``. 135,569 patches of
50 m x 50 m, each a **monospecific** forest area annotated with a single tree species,
derived from 449 curated forests across 40+ southern-French departments. Annotations were
selected from the BD Foret vector database and curated by IGN expert photointerpreters
(National Forest Inventory ground truth used to confirm purity). The proposed
classification has **13 semantic classes** hierarchically grouping 18 tree species.
Licensed under the French Etalab Open Licence 2.0.

We only need the label geometry + species, which live in the metadata GeoPackage
``metadata/PureForest-patches.gpkg`` (EPSG:2154 / Lambert-93; each feature is a 50 m
square with ``class_index`` 0-12). The multi-GB imagery / Lidar zips are NOT downloaded.

Task: per-pixel **classification** (tree species). Patches are tiny (50 m = 5 px at
10 m), so we **aggregate contiguous patches on a 320 m metric grid** into <=32x32 UTM
10 m tiles: every patch burns its ``class_index`` into its 5x5 px footprint, unlabeled
pixels are 255 (nodata/ignore) -- there is no true background class, so land outside the
monospecific patches is "ignore". Grid cells are almost always monospecific (a handful of
cells straddle two neighbouring forests and carry two classes, which is valid).

Sampling: **tiles-per-class balanced** with per_class=1000 and the 25k cap. Rare classes
(Fir, Douglas, Larch, Spruce) are kept in full -- they are inherently rare and the
downstream assembly step, not this script, filters classes that end up too small.

Time range: tree species is a **static** label (a monospecific stand does not change
species year to year); source acquisitions span 2018-2025 but per-patch acquisition years
are not in the released metadata. Per spec Section 5 (static labels) we anchor every sample on a
single representative 1-year window in the Sentinel era (2021).

Run (idempotent; skips already-written {sample_id}.tif):
  python3 -m olmoearth_pretrain.open_set_segmentation_data.datasets.pureforest
"""

import argparse
import csv
import multiprocessing
from collections import Counter, defaultdict
from typing import Any

import numpy as np
import shapely
import tqdm
from rasterio.crs import CRS
from rslearn.utils.geometry import Projection
from rslearn.utils.mp import star_imap_unordered

from olmoearth_pretrain.open_set_segmentation_data import io, manifest
from olmoearth_pretrain.open_set_segmentation_data.download import hf_download
from olmoearth_pretrain.open_set_segmentation_data.rasterize import (
    geom_to_pixels,
    rasterize_shapes,
)
from olmoearth_pretrain.open_set_segmentation_data.sampling import (
    select_tiles_per_class,
)

SLUG = "pureforest"
NAME = "PureForest"
REPO_ID = "IGNF/PureForest"

GPKG_REL = "metadata/PureForest-patches.gpkg"
DICT_REL = "metadata/PureForestID-dictionnary.csv"

GRID_M = 320  # metric aggregation grid (Lambert-93) -> 32 px @ 10 m
TILE_PX = GRID_M // io.RESOLUTION  # 32
PER_CLASS = 1000
ANCHOR_YEAR = 2021  # representative Sentinel-era year (species labels are static)

_WGS84_SRC = Projection(CRS.from_epsg(4326), 1, 1)


def load_class_descriptions(dict_path: str) -> dict[int, dict[str, Any]]:
    """Build {class_index: {name, description}} from the ID dictionary CSV.

    The dictionary maps each class to one or more species (18 species -> 13 classes) plus
    a broadleaf/needleleaf hierarchy; we compose a description listing the grouped species.
    """
    rows: list[dict[str, str]] = []
    with open(dict_path, newline="") as f:
        rows = list(csv.DictReader(f))
    by_class: dict[int, dict[str, Any]] = {}
    species: dict[int, list[str]] = defaultdict(list)
    for r in rows:
        cid = int(r["class_index"])
        by_class.setdefault(
            cid,
            {
                "name": r["class_name_en"].strip(),
                "hierarchy_1": r["hierarchy_1"].strip(),
                "hierarchy_2": r["hierarchy_2"].strip(),
            },
        )
        sp = f"{r['species_name_latin'].strip()} ({r['species_name_en'].strip()})"
        if sp not in species[cid]:
            species[cid].append(sp)
    out: dict[int, dict[str, Any]] = {}
    for cid, info in by_class.items():
        desc = (
            f"{info['hierarchy_1']} / genus {info['hierarchy_2']}. Monospecific stands of: "
            + "; ".join(species[cid])
            + "."
        )
        out[cid] = {"name": info["name"], "description": desc}
    return out


def build_tile_records(gpkg_path: str) -> list[dict[str, Any]]:
    """Read the patch GeoPackage and aggregate patches into 320 m grid-cell tile records.

    Returns one record per occupied grid cell: {cell center lon/lat, patch (wkb, class)
    list, classes_present, year, source_id}.
    """
    import geopandas as gpd

    gdf = gpd.read_file(gpkg_path)  # EPSG:2154 (Lambert-93), metres
    cls = gdf["class_index"].to_numpy().astype(int)
    cent = gdf.geometry.centroid
    gx = np.floor(cent.x.to_numpy() / GRID_M).astype(np.int64)
    gy = np.floor(cent.y.to_numpy() / GRID_M).astype(np.int64)

    # Reproject patch polygons to WGS84 once (proven eurocrops path).
    geom_wgs = gpd.GeoSeries(gdf.geometry, crs=2154).to_crs(4326)
    dept = gdf["french_department_id"].astype(str).to_numpy()

    # Group patch indices by grid cell.
    cell_idx: dict[tuple[int, int], list[int]] = defaultdict(list)
    for i in range(len(gdf)):
        cell_idx[(int(gx[i]), int(gy[i]))].append(i)

    # Cell centre lon/lat: build centre points in Lambert, reproject all at once.
    cells = list(cell_idx.keys())
    cx = np.array([c[0] * GRID_M + GRID_M / 2 for c in cells])
    cy = np.array([c[1] * GRID_M + GRID_M / 2 for c in cells])
    centres = gpd.GeoSeries(gpd.points_from_xy(cx, cy), crs=2154).to_crs(4326)

    records: list[dict[str, Any]] = []
    for k, cell in enumerate(cells):
        idxs = cell_idx[cell]
        patches = [(shapely.to_wkb(geom_wgs.iloc[i]), int(cls[i])) for i in idxs]
        classes_present = sorted({int(cls[i]) for i in idxs})
        pt = centres.iloc[k]
        records.append(
            {
                "lon": float(pt.x),
                "lat": float(pt.y),
                "patches": patches,
                "classes_present": classes_present,
                "year": ANCHOR_YEAR,
                "source_id": f"{dept[idxs[0]]}/cell_{cell[0]}_{cell[1]}",
            }
        )
    return records


def _write_tile(rec: dict[str, Any]) -> tuple[str, str, list[int]]:
    sample_id = rec["sample_id"]
    tif = io.locations_dir(SLUG) / f"{sample_id}.tif"
    if tif.exists():
        return sample_id, "skip", rec["classes_present"]
    try:
        proj = io.utm_projection_for_lonlat(rec["lon"], rec["lat"])
        _, col, row = io.lonlat_to_utm_pixel(rec["lon"], rec["lat"], proj)
        bounds = io.centered_bounds(col, row, TILE_PX, TILE_PX)
        shapes = [
            (geom_to_pixels(shapely.from_wkb(w), _WGS84_SRC, proj), cid)
            for w, cid in rec["patches"]
        ]
        arr = rasterize_shapes(
            shapes, bounds, fill=io.CLASS_NODATA, dtype="uint8", all_touched=True
        )
        present = sorted(set(np.unique(arr).tolist()) - {io.CLASS_NODATA})
        if not present:
            return sample_id, "empty", []
        io.write_label_geotiff(
            SLUG, sample_id, arr, proj, bounds, nodata=io.CLASS_NODATA
        )
        io.write_sample_json(
            SLUG,
            sample_id,
            proj,
            bounds,
            io.year_range(rec["year"]),
            source_id=rec["source_id"],
            classes_present=present,
        )
        return sample_id, "ok", present
    except Exception as e:  # noqa: BLE001
        print(f"error on {sample_id}: {e}")
        return sample_id, "error", []


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=64)
    args = parser.parse_args()

    io.check_disk()
    manifest.write_registry_entry(SLUG, "in_progress")

    raw = io.raw_dir(SLUG)
    gpkg = hf_download(REPO_ID, GPKG_REL, raw)
    dict_csv = hf_download(REPO_ID, DICT_REL, raw)
    with (raw / "SOURCE.txt").open("w") as f:
        f.write(
            "PureForest (IGN France), Hugging Face IGNF/PureForest, Etalab Open Licence 2.0.\n"
            "https://huggingface.co/datasets/IGNF/PureForest ; arXiv:2404.12064\n"
            "Only metadata/PureForest-patches.gpkg + PureForestID-dictionnary.csv used "
            "(label geometry + species); imagery/Lidar zips not downloaded.\n"
        )

    class_info = load_class_descriptions(str(dict_csv))
    records = build_tile_records(str(gpkg))
    print(f"occupied 320m grid cells (candidate tiles): {len(records)}")
    io.check_disk()

    selected = select_tiles_per_class(
        records, classes_key="classes_present", per_class=PER_CLASS, total_cap=25000
    )
    for i, r in enumerate(selected):
        r["sample_id"] = f"{i:06d}"
    print(f"selected {len(selected)} tiles (per_class={PER_CLASS}, 25k cap)")

    results: Counter = Counter()
    written_by_class: Counter = Counter()
    with multiprocessing.Pool(args.workers) as p:
        for _sid, res, present in tqdm.tqdm(
            star_imap_unordered(p, _write_tile, [dict(rec=r) for r in selected]),
            total=len(selected),
        ):
            results[res] += 1
            if res in ("ok", "skip"):
                for c in present:
                    written_by_class[c] += 1
    print("write results:", dict(results))
    io.check_disk()

    classes = [
        {
            "id": cid,
            "name": class_info[cid]["name"],
            "description": class_info[cid]["description"],
        }
        for cid in sorted(class_info)
    ]
    class_counts = {
        class_info[cid]["name"]: int(written_by_class.get(cid, 0))
        for cid in sorted(class_info)
    }
    num_written = int(results.get("ok", 0) + results.get("skip", 0))

    io.write_dataset_metadata(
        SLUG,
        {
            "dataset": SLUG,
            "name": NAME,
            "task_type": "classification",
            "source": "Hugging Face (IGNF/PureForest)",
            "license": "Etalab Open Licence 2.0",
            "provenance": {
                "url": "https://huggingface.co/datasets/IGNF/PureForest",
                "have_locally": False,
                "annotation_method": (
                    "BD Foret polygons curated by IGN photointerpreters; purity confirmed "
                    "with French National Forest Inventory ground truth"
                ),
                "paper": "arXiv:2404.12064",
            },
            "sensors_relevant": ["sentinel2", "sentinel1", "landsat"],
            "classes": classes,
            "nodata_value": io.CLASS_NODATA,
            "num_samples": num_written,
            "class_counts": class_counts,
            "notes": (
                "13 tree-species classes (grouping 18 species) over monospecific French "
                "forest patches. 50 m patches aggregated on a 320 m Lambert-93 grid into "
                "<=32x32 UTM 10 m tiles: class_index inside each patch footprint, 255 "
                "(nodata/ignore) outside (no background class). Tiles-per-class balanced, "
                f"per_class={PER_CLASS}, 25k cap. Static species label anchored on a "
                f"representative {ANCHOR_YEAR} 1-year window (per-patch acquisition years "
                "not in released metadata; acquisitions span 2018-2025)."
            ),
        },
    )

    manifest.write_registry_entry(
        SLUG, "completed", task_type="classification", num_samples=num_written
    )
    print(f"done: {num_written} tiles across {len(class_info)} classes")
    print("class counts:", class_counts)


if __name__ == "__main__":
    multiprocessing.set_start_method("forkserver", force=True)
    main()
