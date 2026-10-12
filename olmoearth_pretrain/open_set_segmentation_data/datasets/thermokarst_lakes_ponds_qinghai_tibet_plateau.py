"""Process the Thermokarst Lake & Pond dataset of the Qinghai-Tibet Plateau into label patches.

Source: Zenodo record 5509325 -- "Thermokarst lake and pond dataset of the Qinghai-Tibet
Plateau (QTP)". Wei et al. (2021), "Sentinel-based inventory of thermokarst lakes and ponds
across permafrost landscapes on the Qinghai-Tibet Plateau", Earth and Space Science,
8(11), e2021EA001950 (https://doi.org/10.1029/2021EA001950). CC-BY-4.0.

The dataset is five ESRI polygon shapefiles (QTP_Perm_TL_2020_1..5.shp), each in its own
UTM projection (EPSG:32644 / 32645 / 32647 and equivalent custom UTM-44N WKT), together
covering the entire QTP permafrost landscape. 161,341 thermokarst water-body polygons
mapped from 2020 Sentinel-2 imagery via a random-forest model plus manual visual
vectorization, ranging from ~467 m2 to 3.09 x 10^6 m2. Attribute table carries Area (m2),
DMS Long/Lati strings, Perm_Type, Elevation, Basin, climate covariates, etc.; there is NO
lake/pond class field -- the split is by SIZE.

Class scheme (positive-only foreground; the product maps ONLY water bodies, so non-water
is NOT a mapped class -- it is left as nodata/ignore 255, per spec section 5, and the
assembly step supplies negatives from other datasets):

    0 = thermokarst lake   Water body with area >= 10,000 m2 (0.01 km2).
    1 = pond               Water body with area  < 10,000 m2 (0.01 km2).

The 10,000 m2 (0.01 km2) lake/pond threshold is the one adopted by the source paper
(ponds = standing water < 10,000 m2). By this split: 33,933 lakes (21%), 127,408 ponds
(79%). All polygons are >= ~467 m2 (>= ~5 px at 10 m), so every mapped water body is
resolvable at 10 m; the smallest ponds (~500 m2 ~ 5 px) are near the limit -- rasterized
with all_touched=True so they stay visible.

Sampling: pond polygons are extremely dense/clustered, so we snap every polygon centroid
to a 640 m grid (= a 64 px x 10 m output tile) in its file's UTM CRS, dedup to occupied
cells, tag each cell with the classes among its centroids, and run tiles-per-class
balanced selection (rarest class -- lakes -- filled first) to up to PER_CLASS (1000) tiles
per class. Each selected cell -> one 64x64 tile in local UTM at 10 m centered on the cell;
every water polygon intersecting the tile is rasterized with its area-derived class over a
255 (nodata) background. A tile counts toward every class actually present in it.

Time range: the product maps 2020, so each tile gets a 1-year window on 2020 (annual
presence classification of a persistent landform -- no change_time).

Run: python3 -m olmoearth_pretrain.open_set_segmentation_data.datasets.thermokarst_lakes_ponds_qinghai_tibet_plateau
Idempotent: existing locations/{id}.tif are skipped; the raw zips are downloaded+extracted
once into raw/{slug}/extracted/.
"""

import argparse
import glob
import multiprocessing
import os
import zipfile
from collections import Counter
from typing import Any

import numpy as np
import pyogrio
import tqdm
from pyproj import Transformer
from rasterio.crs import CRS
from rslearn.utils.geometry import Projection
from rslearn.utils.mp import star_imap_unordered
from shapely.geometry import box

from olmoearth_pretrain.open_set_segmentation_data import download, io, manifest
from olmoearth_pretrain.open_set_segmentation_data.rasterize import (
    geom_to_pixels,
    rasterize_shapes,
)
from olmoearth_pretrain.open_set_segmentation_data.sampling import (
    select_tiles_per_class,
)

SLUG = "thermokarst_lakes_ponds_qinghai_tibet_plateau"
NAME = "Thermokarst Lakes & Ponds, Qinghai-Tibet Plateau"
ZENODO_RECORD = "5509325"
ZIP_NAMES = [f"QTP_Perm_TL_2020_{i}.zip" for i in range(1, 6)]

TILE = 64  # 64 px * 10 m = 640 m output tile.
CELL_M = TILE * io.RESOLUTION  # 640 m grid cell in the source (metre) CRS.
QUERY_MARGIN_M = (
    500.0  # bbox half-margin (m) when fetching polygons around a tile center.
)
PER_CLASS = 1000  # up to 1000 tiles per class (spec section 5).
REP_YEAR = 2020  # the QTP product maps 2020.
LAKE_POND_THRESHOLD_M2 = 10000.0  # source paper: ponds < 0.01 km2, lakes >= 0.01 km2.

CLASS_LAKE = 0
CLASS_POND = 1
CLASSES = [
    (
        CLASS_LAKE,
        "thermokarst lake",
        "Thermokarst (thaw) lake: a standing water body >= 10,000 m2 (0.01 km2) formed by "
        "permafrost thaw and ground-ice melt on the Qinghai-Tibet Plateau, mapped from "
        "2020 Sentinel-2 imagery (Wei et al. 2021). Non-water pixels are nodata (255).",
    ),
    (
        CLASS_POND,
        "pond",
        "Thermokarst pond: a small standing water body < 10,000 m2 (0.01 km2) formed by "
        "permafrost thaw, mapped from 2020 Sentinel-2 imagery (Wei et al. 2021). Smallest "
        "ponds (~500 m2 ~ 5 px at 10 m) are near the resolution limit. Non-water pixels "
        "are nodata (255).",
    ),
]
ID_TO_NAME = {cid: n for cid, n, _d in CLASSES}


def _ext_dir() -> str:
    return os.path.join(str(io.raw_dir(SLUG)), "extracted")


def download_and_extract() -> None:
    """Download the five Zenodo zips and extract the shapefiles (idempotent)."""
    ext = _ext_dir()
    if os.path.isdir(ext) and len(glob.glob(os.path.join(ext, "*.shp"))) == 5:
        print(f"raw shapefiles already present in {ext}")
        return
    raw = io.raw_dir(SLUG)
    raw.mkdir(parents=True, exist_ok=True)
    print("downloading from Zenodo ...")
    download.download_zenodo(ZENODO_RECORD, raw, filenames=ZIP_NAMES)
    os.makedirs(ext, exist_ok=True)
    for zn in ZIP_NAMES:
        zp = os.path.join(str(raw), zn)
        with zipfile.ZipFile(zp) as zf:
            zf.extractall(ext)
    shps = glob.glob(os.path.join(ext, "*.shp"))
    if len(shps) != 5:
        raise RuntimeError(f"expected 5 shapefiles in {ext}, found {len(shps)}")


def _shps() -> list[str]:
    return sorted(glob.glob(os.path.join(_ext_dir(), "*.shp")))


def _class_for_area(area: float) -> int:
    return CLASS_LAKE if area >= LAKE_POND_THRESHOLD_M2 else CLASS_POND


def scan_file(path: str) -> list[dict[str, Any]]:
    """Read polygons; snap centroids to a CELL_M grid; return one record per occupied cell.

    Each record: {path, crs_wkt, cx, cy (cell center in file CRS), lon, lat, classes}.
    ``classes`` is the sorted set of area-derived class ids among the centroids in the cell
    (used only for tiles-per-class balancing; the written classes_present come from the
    actual rasterization).
    """
    gdf = pyogrio.read_dataframe(path, columns=["Area"], read_geometry=True)
    if len(gdf) == 0:
        return []
    cent = gdf.geometry.centroid
    cx = cent.x.values
    cy = cent.y.values
    cls = np.where(gdf["Area"].values >= LAKE_POND_THRESHOLD_M2, CLASS_LAKE, CLASS_POND)
    ix = np.floor(cx / CELL_M).astype(np.int64)
    iy = np.floor(cy / CELL_M).astype(np.int64)

    # Aggregate class set per occupied cell.
    cell_classes: dict[tuple[int, int], set[int]] = {}
    for a, b, c in zip(ix, iy, cls):
        cell_classes.setdefault((int(a), int(b)), set()).add(int(c))

    cells = np.array(sorted(cell_classes.keys()), dtype=np.int64)
    ccx = (cells[:, 0] + 0.5) * CELL_M
    ccy = (cells[:, 1] + 0.5) * CELL_M
    crs_wkt = gdf.crs.to_wkt()
    transformer = Transformer.from_crs(gdf.crs, 4326, always_xy=True)
    lon, lat = transformer.transform(ccx, ccy)
    lon = np.asarray(lon)
    lat = np.asarray(lat)

    recs: list[dict[str, Any]] = []
    for i in range(len(cells)):
        key = (int(cells[i, 0]), int(cells[i, 1]))
        recs.append(
            {
                "path": path,
                "crs_wkt": crs_wkt,
                "cx": float(ccx[i]),
                "cy": float(ccy[i]),
                "lon": float(lon[i]),
                "lat": float(lat[i]),
                "classes": sorted(cell_classes[key]),
            }
        )
    return recs


def _write_one(rec: dict[str, Any]) -> str | None:
    sample_id = rec["sample_id"]
    if (io.locations_dir(SLUG) / f"{sample_id}.tif").exists():
        return "skip"

    lon, lat = rec["lon"], rec["lat"]
    proj, col, row = io.lonlat_to_utm_pixel(lon, lat)
    bounds = io.centered_bounds(col, row, TILE, TILE)

    cx, cy = rec["cx"], rec["cy"]
    sub = pyogrio.read_dataframe(
        rec["path"],
        columns=["Area"],
        read_geometry=True,
        bbox=(
            cx - QUERY_MARGIN_M,
            cy - QUERY_MARGIN_M,
            cx + QUERY_MARGIN_M,
            cy + QUERY_MARGIN_M,
        ),
    )
    if len(sub) == 0:
        return None

    src_proj = Projection(CRS.from_wkt(rec["crs_wkt"]), 1, 1)
    tile_box = box(*bounds)
    shapes: list[tuple[Any, int]] = []
    for geom, area in zip(sub.geometry.values, sub["Area"].values):
        if geom is None or geom.is_empty:
            continue
        px = geom_to_pixels(geom, src_proj, proj)
        if px.is_empty:
            continue
        clip = px.intersection(tile_box)
        if clip.is_empty or clip.area <= 0:
            continue
        shapes.append((clip, _class_for_area(float(area))))
    if not shapes:
        return None

    # Positive-only: water polygons carry their class over a 255 (nodata/ignore) background.
    # all_touched=True so the smallest ponds (~5 px) stay visible.
    label = rasterize_shapes(
        shapes, bounds, fill=io.CLASS_NODATA, dtype="uint8", all_touched=True
    )[0]
    present = sorted(int(v) for v in np.unique(label) if int(v) != io.CLASS_NODATA)
    if not present:
        return None  # no resolvable water pixels landed in the tile

    io.write_label_geotiff(SLUG, sample_id, label, proj, bounds, nodata=io.CLASS_NODATA)
    io.write_sample_json(
        SLUG,
        sample_id,
        proj,
        bounds,
        io.year_range(REP_YEAR),
        source_id=f"{os.path.basename(rec['path'])}:{int(cx)}_{int(cy)}",
        classes_present=present,
    )
    return "+".join(str(c) for c in present)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=64)
    ap.add_argument("--per-class", type=int, default=PER_CLASS)
    args = ap.parse_args()

    io.check_disk()
    manifest.write_registry_entry(SLUG, "in_progress")

    download_and_extract()
    shps = _shps()
    print(f"{len(shps)} shapefiles")

    # Scan phase: occupied 640 m cells per file (parallel over the 5 files).
    with multiprocessing.Pool(min(args.workers, len(shps))) as p:
        results = list(
            tqdm.tqdm(
                star_imap_unordered(p, scan_file, [dict(path=s) for s in shps]),
                total=len(shps),
                desc="scan",
            )
        )
    candidates: list[dict[str, Any]] = []
    for r in results:
        candidates.extend(r)
    cell_class_counts = Counter()
    for c in candidates:
        for cid in c["classes"]:
            cell_class_counts[cid] += 1
    print(
        f"candidate cells: {len(candidates)} "
        + str({ID_TO_NAME[k]: v for k, v in cell_class_counts.items()})
    )

    io.check_disk()

    # Tiles-per-class balanced selection (rarest class -- lakes -- filled first).
    selected = select_tiles_per_class(
        candidates, classes_key="classes", per_class=args.per_class
    )
    # Deterministic id assignment.
    selected.sort(key=lambda r: (os.path.basename(r["path"]), r["cx"], r["cy"]))
    for j, rec in enumerate(selected):
        rec["sample_id"] = f"{j:06d}"
    print(f"selected {len(selected)} tiles of {len(candidates)} candidate cells")

    io.check_disk()

    # Write phase.
    tile_class_counter: Counter = Counter()  # counts tiles containing each class
    n_ok = 0
    with multiprocessing.Pool(args.workers) as p:
        for res in tqdm.tqdm(
            star_imap_unordered(p, _write_one, [dict(rec=r) for r in selected]),
            total=len(selected),
            desc="write tiles",
        ):
            if res is None:
                continue
            n_ok += 1
            if res != "skip":
                for c in res.split("+"):
                    tile_class_counter[int(c)] += 1

    n_written = len(list(io.locations_dir(SLUG).glob("*.tif")))
    print(f"tiles ok this run: {n_ok}; total tif on disk: {n_written}")
    for cid, name, _d in CLASSES:
        print(
            f"  tiles containing class {cid} ({name}): {tile_class_counter.get(cid, 0)}"
        )

    io.write_dataset_metadata(
        SLUG,
        {
            "dataset": SLUG,
            "name": NAME,
            "task_type": "classification",
            "source": "Zenodo 5509325 (Wei et al. 2021, Earth and Space Science)",
            "license": "CC-BY-4.0",
            "provenance": {
                "url": "https://zenodo.org/records/5509325",
                "have_locally": False,
                "annotation_method": (
                    "random-forest classification of 2020 Sentinel-2 imagery + manual "
                    "visual vectorization (Wei et al. 2021)"
                ),
            },
            "sensors_relevant": ["sentinel2", "sentinel1", "landsat"],
            "classes": [
                {"id": cid, "name": name, "description": desc}
                for cid, name, desc in CLASSES
            ],
            "nodata_value": io.CLASS_NODATA,
            "num_samples": n_written,
            "class_counts": {
                ID_TO_NAME[cid]: tile_class_counter.get(cid, 0) for cid, *_ in CLASSES
            },
            "notes": (
                "Bounded polygon sampling from five per-region shapefiles (161,341 "
                "thermokarst water-body polygons, mapped 2020 at 10 m, each file in its own "
                "UTM CRS). Lake/pond split is by SIZE (no class field in the source): the "
                "source paper's 10,000 m2 (0.01 km2) threshold -> class 0 thermokarst lake "
                "(>= 10,000 m2; 33,933 polygons) and class 1 pond (< 10,000 m2; 127,408 "
                "polygons). Polygon centroids snapped to a 640 m grid; cells tagged with "
                "their centroid classes; tiles-per-class balanced selection (rarest class "
                "-- lakes -- filled first) to up to per_class=1000 tiles per class. Each "
                "cell -> one 64x64 tile in local UTM at 10 m centered on the cell; every "
                "water polygon intersecting the tile rasterized with its area-derived class "
                "over a 255 (nodata/ignore) background. POSITIVE-ONLY foreground: the "
                "product maps only water bodies, so non-water is nodata (not a fabricated "
                "background class); the assembly step supplies negatives from other "
                "datasets (spec section 5). all_touched=True keeps the smallest ponds "
                "(~500 m2 ~ 5 px) visible; all polygons are >= ~467 m2 so none are "
                "sub-pixel at 10 m. Annual 2020 product -> 1-year time range on 2020 (no "
                "change_time)."
            ),
        },
    )
    manifest.write_registry_entry(
        SLUG, "completed", task_type="classification", num_samples=n_written
    )
    print("done")


if __name__ == "__main__":
    multiprocessing.set_start_method("forkserver", force=True)
    main()
