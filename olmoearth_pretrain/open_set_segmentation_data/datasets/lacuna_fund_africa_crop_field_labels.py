"""Process Lacuna Fund Africa Crop Field Labels into open-set-segmentation patches.

Source: "A region-wide, multi-year set of crop field boundary labels for Africa"
(Estes et al., 2024, arXiv:2412.18483), funded by the Lacuna Fund, led by Farmerline
with Spatial Collective and the Agricultural Impacts Research Group at Clark University.
GitHub: https://github.com/agroimpacts/lacunalabels . Data are hosted on the public
Registry of Open Data on AWS bucket ``s3://africa-field-boundary-labels`` (us-west-2,
unsigned/no credential) and on Zenodo (record 11060871). Labels: CC-BY-4.0; imagery is
Planet NICFI (not needed here). ~825k manually-digitized crop-field boundary polygons
across continental Africa, drawn on Planet NICFI basemaps for imagery months spanning
2017-2023.

We use ONLY the vector labels:
  * ``mapped_fields_final.parquet`` — 825,395 field-boundary polygons (WGS84), columns
    fid/name/assignment_id/completion_time/category. ``name`` is the Planet image-chip id
    and ``assignment_id`` the labelling assignment; together they identify one labelled
    ~1.2 km chip. ``category`` is the field type (annualcropland dominates; a handful of
    fallow / treecrop / unsure / cloudshadow).
  * ``label_catalog_allclasses.csv`` — per-assignment metadata: chip-center lon/lat (x,y)
    and the ``image_date`` (YYYY-MM-15) of the Planet basemap that was labelled. We join on
    (name, assignment_id) to recover each assignment's center and imagery year.
We do NOT download the Planet image chips or 3-class label rasters (pretraining supplies
its own imagery).

Class scheme (dense 3-class segmentation, all resolvable at 10 m; mirrors the sibling
``ai4boundaries`` field-boundary dataset for consistency):
  0 = non-field / background
  1 = crop field interior
  2 = crop field boundary   [priority over interior]
Field polygons (categories annualcropland/fallow/treecrop) are rasterized as interior (1);
their outlines (all_touched) as boundary (2), overlaid on top. Field boundaries at 10 m are
the core signal this dataset was built to expose (median field ~0.5 ha ~= 50 px at 10 m; a
field boundary is ~1-2 px). The few ``unsure1``/``unsure2``/``cloudshadow`` polygons (~59
total) are written as nodata/ignore (255) so they are neither field nor background.

Processing (task spec sec.4 polygons, sec.5 tiles-per-class balancing):
  * One 64x64, 10 m, local-UTM tile per labelling assignment, centered on the chip center
    (catalog x,y). 64 px = 640 m stays safely inside the ~1.2 km labelled chip footprint,
    so background pixels are genuinely-examined non-field land, not un-labelled area (field
    extent per chip: median ~610 m, 90th ~800 m). Polygons are reprojected to the tile's UTM
    pixel grid and rasterized.
  * Only assignments carrying at least one field polygon AND a valid catalog center +
    image_date are candidates. Tiles-per-class balanced selection: <=1000 tiles per class,
    rarest-class-first, capped at 25,000 total. Because most chips contain all three classes,
    the boundary/interior split makes the per-class balance meaningful and the selection
    typically settles near ~1000 tiles.

Time range: seasonal crop labels -> a 1-year window anchored on the labelled imagery year
(``image_date``); all imagery months fall in 2017-2023 (Sentinel era). Not a change dataset
(change_time=null).

Run: python3 -m olmoearth_pretrain.open_set_segmentation_data.datasets.lacuna_fund_africa_crop_field_labels
"""

import argparse
import multiprocessing
import pickle
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd
import tqdm
from rslearn.const import WGS84_PROJECTION
from rslearn.utils.mp import star_imap_unordered

from olmoearth_pretrain.open_set_segmentation_data import (
    io,
    manifest,
    rasterize,
    sampling,
)

SLUG = "lacuna_fund_africa_crop_field_labels"
NAME = "Lacuna Fund Africa Crop Field Labels"
TILE = 64
PER_CLASS = 1000
FIELD_CATEGORIES = {"annualcropland", "fallow", "treecrop"}
IGNORE_CATEGORIES = {"unsure1", "unsure2", "cloudshadow"}

CLASSES = [
    (
        "non-field",
        "Background / non-field: land within a labelled Planet chip that was examined by an "
        "annotator and NOT delineated as a crop field.",
    ),
    (
        "crop field interior",
        "Interior of a manually-digitized crop-field polygon (category annualcropland, fallow, "
        "or treecrop).",
    ),
    (
        "crop field boundary",
        "Boundary pixel of a crop-field polygon (the digitized parcel outline); the core signal "
        "this crop-field-boundary dataset was built to expose at 10 m.",
    ),
]
NUM_CLASSES = len(CLASSES)

PARQUET = "mapped_fields_final.parquet"
CATALOG = "label_catalog_allclasses.csv"


def _rasterize_assignment(
    polys: list[tuple[Any, str]], lon: float, lat: float
) -> tuple[np.ndarray, str, tuple[int, int, int, int]]:
    """Build the 3-class uint8 tile for one assignment centered on (lon, lat)."""
    proj, col, row = io.lonlat_to_utm_pixel(lon, lat)
    bounds = io.centered_bounds(col, row, TILE, TILE)

    field_geoms = []
    ignore_geoms = []
    for geom, category in polys:
        if not geom.is_valid:
            geom = geom.buffer(0)
        if geom.is_empty:
            continue
        px = rasterize.geom_to_pixels(geom, WGS84_PROJECTION, proj)
        if category in FIELD_CATEGORIES:
            field_geoms.append(px)
        elif category in IGNORE_CATEGORIES:
            ignore_geoms.append(px)

    arr = np.zeros((TILE, TILE), dtype=np.uint8)  # 0 = non-field
    if field_geoms:
        interior = rasterize.rasterize_shapes(
            [(g, 1) for g in field_geoms], bounds, fill=0
        )[0]
        boundary = rasterize.rasterize_shapes(
            [(g.boundary, 1) for g in field_geoms], bounds, fill=0, all_touched=True
        )[0]
        arr[interior > 0] = 1
        arr[boundary > 0] = 2
    if ignore_geoms:
        ign = rasterize.rasterize_shapes(
            [(g, 1) for g in ignore_geoms], bounds, fill=0, all_touched=True
        )[0]
        # only overwrite pixels not already assigned to a confident field class
        arr[(ign > 0) & (arr == 0)] = io.CLASS_NODATA
    return arr, proj.crs.to_string(), bounds


def _scan_one(task: dict[str, Any]) -> dict[str, Any] | None:
    """Rasterize one assignment; return a record with the array + classes present."""
    try:
        arr, crs_str, bounds = _rasterize_assignment(
            task["polys"], task["lon"], task["lat"]
        )
    except Exception as e:  # noqa: BLE001
        print(f"WARN scan failed {task['source_id']}: {e}")
        return None
    present = sorted(int(v) for v in np.unique(arr) if v != io.CLASS_NODATA)
    if not (1 in present or 2 in present):
        return None  # require at least one field pixel
    return {
        "arr": arr,
        "crs": crs_str,
        "bounds": bounds,
        "year": task["year"],
        "classes_present": present,
        "source_id": task["source_id"],
    }


def _build_tasks() -> list[dict[str, Any]]:
    raw = io.raw_dir(SLUG)
    print(f"loading {raw / PARQUET}")
    gdf = gpd.read_parquet(str(raw / PARQUET))
    gdf["assignment_id"] = gdf["assignment_id"].astype(float)
    cat = pd.read_csv(str(raw / CATALOG), low_memory=False)
    cat["assignment_id"] = pd.to_numeric(cat["assignment_id"], errors="coerce")
    cat["year"] = pd.to_datetime(cat["image_date"], errors="coerce").dt.year
    cat = cat.dropna(subset=["assignment_id", "x", "y", "year"])
    cat_idx = cat.set_index(["name", "assignment_id"])[["x", "y", "year"]]

    tasks: list[dict[str, Any]] = []
    n_no_meta = 0
    for (name, aid), sub in gdf.groupby(["name", "assignment_id"], sort=True):
        key = (name, aid)
        if key not in cat_idx.index:
            n_no_meta += 1
            continue
        row = cat_idx.loc[key]
        if hasattr(row, "iloc") and getattr(row, "ndim", 1) == 2:
            row = row.iloc[0]
        polys = [(g, c) for g, c in zip(sub.geometry.values, sub["category"].values)]
        tasks.append(
            {
                "polys": polys,
                "lon": float(row["x"]),
                "lat": float(row["y"]),
                "year": int(row["year"]),
                "source_id": f"{name}/{int(aid)}",
            }
        )
    print(
        f"built {len(tasks)} assignment tasks "
        f"({n_no_meta} field assignments dropped: no catalog center/date)"
    )
    return tasks


def _scan_all(workers: int) -> list[dict[str, Any]]:
    cache = io.raw_dir(SLUG) / "scan_cache.pkl"
    if cache.exists():
        print(f"loading cached scan from {cache}")
        with cache.open("rb") as f:
            return pickle.load(f)
    tasks = _build_tasks()
    print(f"scanning {len(tasks)} assignments (mp rasterize)")
    recs: list[dict[str, Any]] = []
    with multiprocessing.Pool(workers) as p:
        for rec in tqdm.tqdm(
            star_imap_unordered(p, _scan_one, [dict(task=t) for t in tasks]),
            total=len(tasks),
        ):
            if rec is not None:
                recs.append(rec)
    print(f"scanned {len(recs)} field-containing candidate tiles")
    io.raw_dir(SLUG).mkdir(parents=True, exist_ok=True)
    tmp = io.raw_dir(SLUG) / "scan_cache.pkl.tmp"
    with tmp.open("wb") as f:
        pickle.dump(recs, f)
    tmp.rename(cache)
    return recs


def _write_one(rec: dict[str, Any]) -> None:
    from rasterio.crs import CRS
    from rslearn.utils.geometry import Projection

    sample_id = rec["sample_id"]
    if (io.locations_dir(SLUG) / f"{sample_id}.tif").exists():
        return
    proj = Projection(CRS.from_string(rec["crs"]), io.RESOLUTION, -io.RESOLUTION)
    arr = rec["arr"]
    bounds = tuple(rec["bounds"])
    io.write_label_geotiff(SLUG, sample_id, arr, proj, bounds, nodata=io.CLASS_NODATA)
    io.write_sample_json(
        SLUG,
        sample_id,
        proj,
        bounds,
        io.year_range(rec["year"]),
        source_id=rec["source_id"],
        classes_present=sorted(int(v) for v in np.unique(arr) if v != io.CLASS_NODATA),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=64)
    parser.add_argument("--write-workers", type=int, default=64)
    args = parser.parse_args()

    io.check_disk()
    manifest.write_registry_entry(SLUG, "in_progress")

    raw = io.raw_dir(SLUG)
    raw.mkdir(parents=True, exist_ok=True)
    with (raw / "SOURCE.txt").open("w") as f:
        f.write(
            "Source: Lacuna Fund Africa Crop Field Labels (Estes et al. 2024, "
            "arXiv:2412.18483). GitHub: https://github.com/agroimpacts/lacunalabels\n"
            "Data (labels): public AWS Open Data bucket s3://africa-field-boundary-labels "
            "(us-west-2, unsigned) and Zenodo record 11060871. License CC-BY-4.0.\n"
            "Downloaded (labels only): mapped_fields_final.parquet (825,395 field polygons, "
            "WGS84) and label_catalog_allclasses.csv (per-assignment chip center + "
            "image_date). NOT downloaded: Planet NICFI image chips or 3-class label rasters "
            "(pretraining supplies imagery).\n"
        )

    records = _scan_all(args.workers)
    selected = sampling.select_tiles_per_class(
        records,
        classes_key="classes_present",
        per_class=PER_CLASS,
        total_cap=sampling.MAX_SAMPLES_PER_DATASET,
    )
    selected.sort(key=lambda r: r["source_id"])
    for i, r in enumerate(selected):
        r["sample_id"] = f"{i:06d}"
    print(f"selected {len(selected)} tiles (of {len(records)} candidates)")

    with multiprocessing.Pool(args.write_workers) as p:
        for _ in tqdm.tqdm(
            star_imap_unordered(p, _write_one, [dict(rec=r) for r in selected]),
            total=len(selected),
        ):
            pass

    tile_counts = {i: 0 for i in range(NUM_CLASSES)}
    year_counts: dict[int, int] = {}
    for r in selected:
        for c in r["classes_present"]:
            tile_counts[c] += 1
        year_counts[r["year"]] = year_counts.get(r["year"], 0) + 1

    io.write_dataset_metadata(
        SLUG,
        {
            "dataset": SLUG,
            "name": NAME,
            "task_type": "classification",
            "source": "GitHub (agroimpacts/lacunalabels) / AWS Open Data",
            "license": "CC-BY-4.0",
            "provenance": {
                "url": "https://github.com/agroimpacts/lacunalabels",
                "have_locally": False,
                "annotation_method": "manual visual interpretation of Planet NICFI basemaps",
            },
            "sensors_relevant": ["sentinel2", "sentinel1", "landsat"],
            "classes": [
                {"id": i, "name": name, "description": desc}
                for i, (name, desc) in enumerate(CLASSES)
            ],
            "nodata_value": io.CLASS_NODATA,
            "num_samples": len(selected),
            "class_tile_counts": {
                CLASSES[i][0]: tile_counts[i] for i in range(NUM_CLASSES)
            },
            "year_counts": {str(k): year_counts[k] for k in sorted(year_counts)},
            "notes": (
                "Continent-wide African crop-field-boundary polygons (~825k, manual visual "
                "interpretation on Planet NICFI basemaps, 2017-2023). 3-class dense "
                "segmentation: 0 non-field/background, 1 crop field interior, 2 crop field "
                "boundary (boundary wins over interior). One 64x64 10 m local-UTM tile per "
                "labelling assignment, centered on the chip center (640 m tile stays inside "
                "the ~1.2 km labelled chip so background is examined non-field, not "
                "un-labelled land). Field polygons (annualcropland/fallow/treecrop) "
                "rasterized as interior; polygon outlines (all_touched) as boundary; the ~59 "
                "unsure/cloudshadow polygons written as nodata/ignore (255). Time range = "
                "1-year window anchored on the labelled imagery year (image_date). "
                "Tiles-per-class balanced to <=1000/class, rarest-first, <=25k total. Field "
                "assignments lacking a catalog center/date were dropped."
            ),
        },
    )
    manifest.write_registry_entry(
        SLUG, "completed", task_type="classification", num_samples=len(selected)
    )
    print("class tile counts:")
    for i in range(NUM_CLASSES):
        print(f"  {i} {CLASSES[i][0]:22} {tile_counts[i]}")
    print("year counts:", {k: year_counts[k] for k in sorted(year_counts)})
    print("done")


if __name__ == "__main__":
    multiprocessing.set_start_method("forkserver", force=True)
    main()
