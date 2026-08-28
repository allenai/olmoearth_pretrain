"""Process the Coastal Aquaculture Pond dataset (China & SE Asia) into label patches.

Source: Zenodo record 10370830 -- "Fine-detailed coastal aquaculture pond dataset in
China and Southeast Asia in 2020 at a 30 m resolution" (CLAP_CSEA_2020). Duan et al.,
derived from the long time-series Landsat archive with an object-oriented hierarchical
classification method (MDPI / Remote Sensing). Distributed as a single .rar containing
per-country ESRI shapefiles of aquaculture-pond polygons in Albers equal-area projections
(ESRI:102025 for China, ESRI:102028 for the SE-Asia countries):

    Brunei, Cambodia, China, Indonesia (x2 tiles), Malaysia, Myanmar, Philippines,
    Singapore, Thailand, Timor-Leste, Vietnam

Total ~636k pond polygons. Ponds are small (typ. ~800-30000 m2, i.e. well under a 640 m
tile) but extremely dense in coastal deltas, so this is a LARGE polygon set and we do
BOUNDED sampling (spec 5).

Class scheme (binary; the source is a complete coastal map, so non-pond is a real
mapped class, not just an ignore region):

    0 = non-pond          Any mapped coastal pixel that is not an aquaculture pond.
    1 = aquaculture pond   Coastal fish / shrimp / crab aquaculture pond footprint.

Sampling: we snap every pond centroid to a 640 m grid (= a 64 px x 10 m output tile) in
its country's Albers CRS, take the set of occupied grid cells (dedups dense clustering:
~636k polygons -> ~144k cells), and uniformly sample TARGET cells across the pooled set
(so China / Vietnam / Thailand / Indonesia / Philippines -- where ponds actually are --
dominate, with the smaller countries represented proportionally). Each sampled cell
becomes one 64x64 tile centered on the cell, in local UTM at 10 m. Every pond polygon
intersecting the tile is rasterized as class 1; all other pixels are class 0. Both classes
appear in essentially every tile, so tiles-per-class balancing yields ~TARGET tiles for
the two-class scheme (spec: up to 1000 locations per class).

Time range: the product maps 2020, so each tile gets a 1-year window on 2020 (annual
presence classification -- persistent land use, no change_time).

Run: python3 -m olmoearth_pretrain.open_set_segmentation_data.datasets.coastal_aquaculture_ponds_china_se_asia
Idempotent: existing locations/{id}.tif are skipped; the raw .rar is downloaded+extracted
once into raw/{slug}/.
"""

import argparse
import glob
import multiprocessing
import os
import random
import subprocess
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

SLUG = "coastal_aquaculture_ponds_china_se_asia"
NAME = "Coastal Aquaculture Ponds (China & SE Asia)"
ZENODO_RECORD = "10370830"
RAR_NAME = "CLAP_CSEA_2020.rar"
SRC_SUBDIR = "CLAP_CSEA_2020"

TILE = 64  # 64 px * 10 m = 640 m output tile.
CELL_M = TILE * io.RESOLUTION  # 640 m grid cell in the source (metre) CRS.
QUERY_MARGIN_M = 500.0  # bbox half-margin (m) when fetching ponds around a tile center.
TARGET = 1000  # up to ~1000 tiles (spec: up to 1000 locations per class, 2 classes).
REP_YEAR = 2020  # the CLAP_CSEA product maps 2020.
SEED = 42

CLASS_NONPOND = 0
CLASS_POND = 1
CLASSES = [
    (
        "non-pond",
        "Any mapped coastal pixel that is not an aquaculture pond (other land cover, "
        "water, or built-up). The complement of the pond footprints within the coastal "
        "study area mapped by CLAP_CSEA_2020.",
    ),
    (
        "aquaculture pond",
        "Coastal aquaculture pond (fish / shrimp / crab culture pond) footprint, mapped "
        "from the long time-series Landsat archive by object-oriented hierarchical "
        "classification (Duan et al. 2020, CLAP_CSEA).",
    ),
]


def _src_dir() -> str:
    return os.path.join(str(io.raw_dir(SLUG)), SRC_SUBDIR)


def download_and_extract() -> None:
    """Download the Zenodo .rar and extract the shapefiles (idempotent)."""
    src = _src_dir()
    if os.path.isdir(src) and glob.glob(os.path.join(src, "*.shp")):
        print(f"raw shapefiles already present in {src}")
        return
    raw = io.raw_dir(SLUG)
    raw.mkdir(parents=True, exist_ok=True)
    print("downloading from Zenodo ...")
    download.download_zenodo(ZENODO_RECORD, raw, filenames=[RAR_NAME])
    rar_path = os.path.join(str(raw), RAR_NAME)
    print("extracting .rar with bsdtar ...")
    # bsdtar handles RAR read; extract into raw/{slug}/ (archive has its own top dir).
    subprocess.run(["bsdtar", "-xf", rar_path, "-C", str(raw)], check=True)
    if not glob.glob(os.path.join(src, "*.shp")):
        raise RuntimeError(f"extraction produced no shapefiles in {src}")


def _country_shps() -> list[str]:
    return sorted(glob.glob(os.path.join(_src_dir(), "*.shp")))


def scan_country(path: str) -> dict[str, Any]:
    """Read pond centroids, snap to a CELL_M grid, return occupied cell centers.

    Returns a dict with the country's CRS WKT and arrays of unique cell-center
    coordinates in both the source Albers CRS (cx/cy, metres) and WGS84 (lon/lat).
    """
    gdf = pyogrio.read_dataframe(path, columns=[], read_geometry=True)
    if len(gdf) == 0:
        return {"path": path, "crs_wkt": None, "cx": [], "cy": [], "lon": [], "lat": []}
    cent = gdf.geometry.centroid
    cx = cent.x.values
    cy = cent.y.values
    ix = np.floor(cx / CELL_M).astype(np.int64)
    iy = np.floor(cy / CELL_M).astype(np.int64)
    # Unique occupied cells; center each cell at (i + 0.5) * CELL_M.
    cells = np.unique(np.stack([ix, iy], axis=1), axis=0)
    ccx = (cells[:, 0] + 0.5) * CELL_M
    ccy = (cells[:, 1] + 0.5) * CELL_M
    crs_wkt = gdf.crs.to_wkt()
    transformer = Transformer.from_crs(gdf.crs, 4326, always_xy=True)
    lon, lat = transformer.transform(ccx, ccy)
    return {
        "path": path,
        "crs_wkt": crs_wkt,
        "cx": ccx.tolist(),
        "cy": ccy.tolist(),
        "lon": np.asarray(lon).tolist(),
        "lat": np.asarray(lat).tolist(),
    }


def _write_one(rec: dict[str, Any]) -> str | None:
    sample_id = rec["sample_id"]
    if (io.locations_dir(SLUG) / f"{sample_id}.tif").exists():
        return "skip"

    lon, lat = rec["lon"], rec["lat"]
    proj, col, row = io.lonlat_to_utm_pixel(lon, lat)
    bounds = io.centered_bounds(col, row, TILE, TILE)

    # Fetch pond polygons around the tile center from the source shapefile (bbox filter
    # in the country's Albers CRS -- uses the .sbn spatial index, ~20 ms).
    cx, cy = rec["cx"], rec["cy"]
    sub = pyogrio.read_dataframe(
        rec["path"],
        columns=[],
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
    for geom in sub.geometry.values:
        if geom is None or geom.is_empty:
            continue
        px = geom_to_pixels(geom, src_proj, proj)
        if px.is_empty:
            continue
        clip = px.intersection(tile_box)
        if clip.is_empty or clip.area <= 0:
            continue
        shapes.append((clip, CLASS_POND))
    if not shapes:
        return None

    # Rasterize ponds as class 1 over a class-0 (non-pond) background.
    label = rasterize_shapes(
        shapes, bounds, fill=CLASS_NONPOND, dtype="uint8", all_touched=False
    )[0]
    present = sorted(int(v) for v in np.unique(label))
    if CLASS_POND not in present:
        return None  # no resolvable pond pixels landed in the tile

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
    return "ok"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=64)
    ap.add_argument("--target", type=int, default=TARGET)
    args = ap.parse_args()

    io.check_disk()
    manifest.write_registry_entry(SLUG, "in_progress")

    download_and_extract()
    shps = _country_shps()
    print(f"{len(shps)} country shapefiles")

    # Scan phase: occupied 640 m cells per country (parallel over the ~12 files).
    with multiprocessing.Pool(min(args.workers, len(shps))) as p:
        results = list(
            tqdm.tqdm(
                star_imap_unordered(p, scan_country, [dict(path=s) for s in shps]),
                total=len(shps),
                desc="scan",
            )
        )

    candidates: list[dict[str, Any]] = []
    per_country: dict[str, int] = {}
    for r in results:
        n = len(r["cx"])
        per_country[os.path.basename(r["path"])] = n
        for i in range(n):
            candidates.append(
                {
                    "path": r["path"],
                    "crs_wkt": r["crs_wkt"],
                    "cx": r["cx"][i],
                    "cy": r["cy"][i],
                    "lon": r["lon"][i],
                    "lat": r["lat"][i],
                }
            )
    print(f"candidate cells: {len(candidates)} " + str(per_country))

    io.check_disk()

    rng = random.Random(SEED)
    rng.shuffle(candidates)
    selected = candidates[: args.target]
    for j, rec in enumerate(selected):
        rec["sample_id"] = f"{j:06d}"
    print(f"selected {len(selected)} tiles of {len(candidates)} candidate cells")

    io.check_disk()

    # Write phase.
    n_ok = 0
    with multiprocessing.Pool(args.workers) as p:
        for res in tqdm.tqdm(
            star_imap_unordered(p, _write_one, [dict(rec=r) for r in selected]),
            total=len(selected),
            desc="write tiles",
        ):
            if res in ("ok", "skip"):
                n_ok += 1

    n_written = len(list(io.locations_dir(SLUG).glob("*.tif")))
    print(f"tiles ok this run: {n_ok}; total tif on disk: {n_written}")

    io.write_dataset_metadata(
        SLUG,
        {
            "dataset": SLUG,
            "name": NAME,
            "task_type": "classification",
            "source": "Zenodo 10370830 (CLAP_CSEA_2020; Duan et al., MDPI Remote Sensing)",
            "license": "open (CC-BY; Zenodo open access)",
            "provenance": {
                "url": "https://zenodo.org/records/10370830",
                "have_locally": False,
                "annotation_method": (
                    "object-oriented hierarchical classification of the long time-series "
                    "Landsat archive with manual training samples (2020)"
                ),
            },
            "sensors_relevant": ["sentinel2", "sentinel1", "landsat"],
            "classes": [
                {"id": i, "name": name, "description": desc}
                for i, (name, desc) in enumerate(CLASSES)
            ],
            "nodata_value": io.CLASS_NODATA,
            "num_samples": n_written,
            "notes": (
                "Bounded polygon sampling from per-country pond shapefiles (~636k polygons "
                "in ESRI Albers projections, mapped 2020 at ~30 m). Pond centroids snapped "
                "to a 640 m grid (~144k occupied cells); TARGET cells uniformly sampled "
                "from the pooled set (China / Vietnam / Thailand / Indonesia / Philippines "
                "dominate, matching real pond density). Each cell -> one 64x64 tile in "
                "local UTM at 10 m; every pond intersecting the tile rasterized as class 1 "
                "(aquaculture pond) over a class-0 (non-pond) background. Both classes "
                "present per tile (binary segmentation). Source is a complete coastal map, "
                "so non-pond is a real mapped class (not an ignore region); 255 reserved "
                "for nodata only. Annual 2020 product -> 1-year time range on 2020 (no "
                "change_time). Ponds mapped at 30 m are rasterized at 10 m (nearest); very "
                "small/thin ponds may under-register at 10 m."
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
