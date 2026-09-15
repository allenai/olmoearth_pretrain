"""Process Munich480 / MTLCC into open-set-segmentation label patches (dense crop-type raster).

Source: MTLCC -- Rußwurm & Körner (2018), "Multi-Temporal Land Cover Classification with
Sequential Recurrent Encoders", ISPRS IJGI 7(4):129. A multi-temporal Sentinel-2 crop-type
segmentation benchmark over a 102 x 42 km area north of Munich, Bavaria, Germany, for the
2016 and 2017 growing seasons. Ground-truth crop labels come from Bavarian farmer
declarations (IACS / STMELF). 17 crop classes.

Georeferencing: the classic MTLCC "ML-ready" release ships the training data as 24/48-px
TFRecord tensor tiles (with a separate geotransforms.csv); rather than pull the 42 GB
TFRecord archive to recover geolocation, we take the **georeferenced ground-truth crop
parcels** distributed as ESRI shapefiles inside the 1.4 GB ``showcase.zip`` on Zenodo
(record 5712933). Those shapefiles (``fields16.shp`` / ``fields17.shp``) are native
**EPSG:32632 (WGS84 / UTM zone 32N)** polygons in metres -- fully georeferenced, no CRS
recovery needed -- with attributes ``if`` (field id), ``labelid`` (1..26, non-sequential),
``label`` (crop name). We download only the ~120 MB of shapefile parts we need via HTTP
range requests into the zip (no full-archive download; imagery not needed -- pretraining
supplies its own).

Task: per-pixel **classification** (crop type), ``dense_raster``. This is a genuinely dense
multi-class benchmark (each 640 m window contains many adjacent crop fields), so rather than
one-tile-per-parcel we rasterize the whole labeled region per year into a single UTM 10 m
label array (unlabeled land = 255 nodata/ignore -- only declared fields carry ground truth),
then cut it into 64x64 (640 m) tiles and pick tiles with **tiles-per-class balanced**
sampling (rare crops prioritized) under the 25k per-dataset cap.

Classes (17): the source ``classes.csv`` order, 0-based (``class_id`` = row index):
0 sugar beet, 1 summer oat, 2 meadow, 3 rape, 4 hop, 5 winter spelt, 6 winter triticale,
7 beans, 8 peas, 9 potatoe, 10 soybeans, 11 asparagus, 12 winter wheat, 13 winter barley,
14 winter rye, 15 summer barley, 16 maize. (Source label ids 1,2,3,5,8,9,12,13,15,16,17,
19,22,23,24,25,26 are remapped to 0..16, matching the MTLCC labid->dimid lookup.)

Time range: 1-year window anchored on each tile's labeled year (2016 or 2017). Both are in
the Sentinel-2 era (post-2016 rule: 2016 is OK). Static seasonal crop labels -> no
change_time. Both years are included; a tile at the same location in 2016 and 2017 is two
independent samples (different crop, different year window).

Run (idempotent; skips already-written {sample_id}.tif):
  python3 -m olmoearth_pretrain.open_set_segmentation_data.datasets.munich480_mtlcc
"""

import argparse
import multiprocessing
import zipfile
from collections import Counter
from typing import Any

import geopandas as gpd
import numpy as np
import tqdm
from affine import Affine
from rasterio.crs import CRS
from rasterio.features import rasterize
from rslearn.utils.geometry import Projection
from rslearn.utils.mp import star_imap_unordered

from olmoearth_pretrain.open_set_segmentation_data import io, manifest
from olmoearth_pretrain.open_set_segmentation_data.download import HttpRangeFile
from olmoearth_pretrain.open_set_segmentation_data.sampling import (
    balance_tiles_by_class,
)

SLUG = "munich480_mtlcc"
NAME = "Munich480 / MTLCC"

ZENODO_URL = "https://zenodo.org/api/records/5712933/files/showcase.zip/content"
# shapefile part suffixes we need for geopandas (fields{16,17}) + the class table
SHP_PARTS = [".shp", ".shx", ".dbf", ".prj"]
YEARS = [2016, 2017]
YEAR_TO_FILE = {2016: "fields16", 2017: "fields17"}

EPSG = 32632  # WGS84 / UTM zone 32N (native CRS of the parcels)
TILE = 64  # 64 px * 10 m = 640 m windows (hard cap)
PER_CLASS = (
    1000  # spec target; lowered to 25000 // N by balance_tiles_by_class if needed.
)

# Source classes.csv order -> 0-based class id. (labelid, name)
CLASS_TABLE = [
    (1, "sugar beet"),
    (2, "summer oat"),
    (3, "meadow"),
    (5, "rape"),
    (8, "hop"),
    (9, "winter spelt"),
    (12, "winter triticale"),
    (13, "beans"),
    (15, "peas"),
    (16, "potatoe"),
    (17, "soybeans"),
    (19, "asparagus"),
    (22, "winter wheat"),
    (23, "winter barley"),
    (24, "winter rye"),
    (25, "summer barley"),
    (26, "maize"),
]
LABELID_TO_CLASS = {labid: i for i, (labid, _) in enumerate(CLASS_TABLE)}
CLASS_NAMES = [name for _, name in CLASS_TABLE]
CLASS_DESCRIPTIONS = {
    name: (
        f"Fields declared as '{name}' by Bavarian farmers (IACS/STMELF) in the MTLCC "
        "Munich480 crop-type benchmark; label burned inside declared parcels, unlabeled "
        "land is ignore (255)."
    )
    for name in CLASS_NAMES
}

_PROJ = Projection(CRS.from_epsg(EPSG), io.RESOLUTION, -io.RESOLUTION)


def ensure_data() -> None:
    """Extract only the fields16/17 shapefile parts + classes.csv from the remote
    showcase.zip via HTTP range requests (no full-archive download). Idempotent.
    """
    raw = io.raw_dir(SLUG)
    raw.mkdir(parents=True, exist_ok=True)
    needed = [f"{YEAR_TO_FILE[y]}{ext}" for y in YEARS for ext in SHP_PARTS] + [
        "classes.csv"
    ]
    if all((raw / n).exists() for n in needed):
        return
    rf = HttpRangeFile(ZENODO_URL)
    try:
        zf = zipfile.ZipFile(rf)
        members = {m.split("/")[-1]: m for m in zf.namelist() if not m.endswith("/")}
        for n in needed:
            dst = raw / n
            if dst.exists():
                continue
            # classes.csv lives under tif/convgru256/2016/; shapefiles under shp/.
            member = members.get(n)
            if member is None:
                raise RuntimeError(f"{n} not found in showcase.zip")
            data = zf.read(member)
            tmp = raw / (n + ".tmp")
            with tmp.open("wb") as f:
                f.write(data)
            tmp.rename(dst)
    finally:
        rf.close()
    with (raw / "SOURCE.txt").open("w") as f:
        f.write(
            "Munich480 / MTLCC crop-type ground-truth parcels (Bavaria, Germany), "
            "Rußwurm & Körner, ISPRS IJGI 2018.\n"
            f"Selectively extracted (HTTP range) from Zenodo record 5712933 showcase.zip:\n"
            f"  {ZENODO_URL}\n"
            "  - fields16.{shp,shx,dbf,prj} (2016 crop parcels, EPSG:32632)\n"
            "  - fields17.{shp,shx,dbf,prj} (2017 crop parcels, EPSG:32632)\n"
            "  - classes.csv (labelid -> crop name)\n"
            "License: CC-BY-4.0. Only label vector files downloaded; imagery not needed.\n"
        )


def _rasterize_year(year: int) -> tuple[np.ndarray, tuple[int, int]]:
    """Rasterize all crop parcels of a year into one big UTM 10 m label array.

    Returns (array[H, W] uint8, (gx0, gy0)) where (gx0, gy0) is the top-left pixel origin
    (integer pixel coords in _PROJ). Unlabeled pixels = 255.
    """
    raw = io.raw_dir(SLUG)
    gdf = gpd.read_file(str(raw / f"{YEAR_TO_FILE[year]}.shp"))
    assert gdf.crs is not None and gdf.crs.to_epsg() == EPSG, (
        f"unexpected CRS {gdf.crs}"
    )
    # Metres -> pixel coords of _PROJ: x_pix = x/10, y_pix = -y/10 (north-up, y_res<0).
    gdf = gdf.copy()
    gdf["geometry"] = gdf.geometry.affine_transform(
        [1.0 / io.RESOLUTION, 0, 0, -1.0 / io.RESOLUTION, 0, 0]
    )
    minx, miny, maxx, maxy = gdf.total_bounds
    gx0, gy0 = int(np.floor(minx)), int(np.floor(miny))
    gx1, gy1 = int(np.ceil(maxx)), int(np.ceil(maxy))
    W, H = gx1 - gx0, gy1 - gy0
    shapes = []
    for geom, labid in zip(gdf.geometry.values, gdf["labelid"].values):
        cid = LABELID_TO_CLASS.get(int(labid))
        if cid is None or geom is None or geom.is_empty:
            continue
        shapes.append((geom, int(cid)))
    transform = Affine(1, 0, gx0, 0, 1, gy0)
    arr = rasterize(
        shapes,
        out_shape=(H, W),
        transform=transform,
        fill=io.CLASS_NODATA,
        dtype="uint8",
        all_touched=False,
    )
    return arr, (gx0, gy0)


def _write_tile(rec: dict[str, Any]) -> tuple[str, str]:
    sample_id = rec["sample_id"]
    tif = io.locations_dir(SLUG) / f"{sample_id}.tif"
    if tif.exists():
        return sample_id, "skip"
    try:
        arr = rec["arr"]
        bounds = rec["bounds"]
        io.write_label_geotiff(
            SLUG, sample_id, arr, _PROJ, bounds, nodata=io.CLASS_NODATA
        )
        io.write_sample_json(
            SLUG,
            sample_id,
            _PROJ,
            bounds,
            io.year_range(rec["year"]),
            source_id=rec["source_id"],
            classes_present=sorted(
                int(c) for c in np.unique(arr) if c != io.CLASS_NODATA
            ),
        )
        return sample_id, "ok"
    except Exception as e:  # noqa: BLE001
        print(f"error on {sample_id}: {e}")
        return sample_id, "error"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=64)
    args = parser.parse_args()

    io.check_disk()
    manifest.write_registry_entry(SLUG, "in_progress")
    ensure_data()

    # ---- Rasterize each year, then enumerate candidate 64x64 tiles ------------------
    candidates: list[dict[str, Any]] = []
    for year in YEARS:
        arr, (gx0, gy0) = _rasterize_year(year)
        H, W = arr.shape
        n_year = 0
        for ti in range(0, H - TILE + 1, TILE):
            for tj in range(0, W - TILE + 1, TILE):
                sub = arr[ti : ti + TILE, tj : tj + TILE]
                present = [int(c) for c in np.unique(sub) if c != io.CLASS_NODATA]
                if not present:
                    continue
                x0 = gx0 + tj
                y0 = gy0 + ti
                candidates.append(
                    {
                        "arr": sub.copy(),
                        "bounds": (x0, y0, x0 + TILE, y0 + TILE),
                        "classes_present": present,
                        "year": year,
                        "source_id": f"{YEAR_TO_FILE[year]}/tile_{tj // TILE}_{ti // TILE}",
                    }
                )
                n_year += 1
        print(f"  {year}: {n_year} labeled 64x64 tiles from {W}x{H} px region")
    print(f"total candidate tiles: {len(candidates)}")

    # ---- Tiles-per-class balanced selection (25k cap) -------------------------------
    selected = balance_tiles_by_class(
        candidates, classes_key="classes_present", per_class=PER_CLASS, total_cap=25000
    )
    n_classes = len(CLASS_NAMES)
    eff = max(1, min(PER_CLASS, 25000 // n_classes))
    print(f"selected {len(selected)} tiles (eff per-class cap = {eff})")
    for i, r in enumerate(selected):
        r["sample_id"] = f"{i:06d}"

    # ---- Write tiles in parallel ----------------------------------------------------
    io.check_disk()
    results: Counter = Counter()
    written_by_class: Counter = Counter()
    id_to_rec = {r["sample_id"]: r for r in selected}
    with multiprocessing.Pool(args.workers) as p:
        for sample_id, res in tqdm.tqdm(
            star_imap_unordered(p, _write_tile, [dict(rec=r) for r in selected]),
            total=len(selected),
        ):
            results[res] += 1
            if res in ("ok", "skip"):
                for c in id_to_rec[sample_id]["classes_present"]:
                    written_by_class[c] += 1
    print("write results:", dict(results))
    io.check_disk()

    # ---- Metadata -------------------------------------------------------------------
    classes = [
        {"id": i, "name": name, "description": CLASS_DESCRIPTIONS[name]}
        for i, name in enumerate(CLASS_NAMES)
    ]
    class_counts = {
        CLASS_NAMES[c]: int(written_by_class.get(c, 0)) for c in range(n_classes)
    }
    num_written = int(results.get("ok", 0) + results.get("skip", 0))
    io.write_dataset_metadata(
        SLUG,
        {
            "dataset": SLUG,
            "name": NAME,
            "task_type": "classification",
            "source": "MTLCC (Rußwurm & Körner, ISPRS IJGI 2018) via Zenodo record 5712933",
            "license": "CC-BY-4.0",
            "provenance": {
                "url": "https://github.com/TUM-LMF/MTLCC",
                "data_url": "https://zenodo.org/record/5712933",
                "have_locally": False,
                "annotation_method": "farmer declaration (IACS / Bavarian STMELF)",
                "years": YEARS,
            },
            "sensors_relevant": ["sentinel2", "sentinel1", "landsat"],
            "classes": classes,
            "nodata_value": io.CLASS_NODATA,
            "num_samples": num_written,
            "class_counts": class_counts,
            "notes": (
                "Dense crop-type segmentation over a 102x42 km area north of Munich, "
                "Bavaria (2016 & 2017 seasons). Ground-truth crop parcels (EPSG:32632 UTM "
                "32N) rasterized per year into a UTM 10 m label array, cut into 64x64 "
                "(640 m) tiles, tiles-per-class balanced under the 25k cap. Only declared "
                "fields carry ground truth; unlabeled land = 255 (ignore), so there is no "
                "background class (no synthetic negatives). 17 source crop classes remapped "
                "to 0-based ids in classes.csv order. Time range = 1-year window anchored "
                "on each tile's labeled year (both post-2016). Only label shapefiles "
                "downloaded (selective HTTP-range extraction from showcase.zip)."
            ),
        },
    )

    manifest.write_registry_entry(
        SLUG, "completed", task_type="classification", num_samples=num_written
    )
    print(f"done: {num_written} samples across {n_classes} classes")


if __name__ == "__main__":
    multiprocessing.set_start_method("forkserver", force=True)
    main()
