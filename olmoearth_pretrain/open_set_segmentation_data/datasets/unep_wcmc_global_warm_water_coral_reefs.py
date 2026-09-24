"""Process the UNEP-WCMC Global Distribution of Warm-Water Coral Reefs into label tiles.

Source: UNEP-WCMC, WorldFish Centre, WRI, TNC (2021). "Global distribution of warm-water
coral reefs" v4.1 (WCMC-008), the most comprehensive global baseline map of tropical /
subtropical coral reefs, compiled from the Millennium Coral Reef Mapping Project
(IMaRS-USF and IRD 2005, IMaRS-USF 2005), the World Atlas of Coral Reefs (Spalding et al.
2001) and other sources. Data DOI 10.34892/t2wk-5t34. License: UNEP-WCMC General Data
License (excluding WDPA) -- free use with attribution, non-commercial, no redistribution
of the source data (we derive internal label rasters only, not a re-distributable copy).

ACCESS (no credential): the shapefiles are a single public S3 download
  https://datadownload-production.s3.us-east-1.amazonaws.com/WCMC008_CoralReefs2021_v4_1.zip
(~208 MB). It contains two EPSG:4326 layers:
  * WCMC008_CoralReef2021_Py_v4_1  -- 17,504 reef PRESENCE polygons (real footprints)
  * WCMC008_CoralReef2021_Pt_v4_1  -- 925 point-only reefs (no footprint, GIS_AREA_K = 0)

TASK: single foreground class (id 0 = "warm-water coral reef" presence). This is a
POSITIVE-ONLY / no-background dataset (spec section 5): non-reef pixels are left as
nodata/ignore (255); we do NOT fabricate negatives (pretraining assembly supplies them by
sampling other datasets). task_type = classification.

10 m SUITABILITY / min size: larger reef complexes are clearly discernible in shallow-water
optical at 10 m. We rasterize the PRESENCE POLYGONS only and keep polygons with
GIS_AREA_K >= MIN_POLY_AREA_KM2 (0.01 km2 ~= 100 pixels at 10 m), and additionally require
each written tile to carry >= MIN_REEF_PIXELS reef pixels, so every label patch shows a
resolvable reef footprint. The 925 point-only reefs (and sub-0.01 km2 polygon slivers) are
sub-pixel single locations with no footprint and are EXCLUDED (documented in the summary).

SAMPLING (global product -> bounded, spec section 5): one candidate 640 m (64 px @ 10 m)
UTM tile per reef polygon (snapped to a global per-UTM-zone tile grid, deduplicated ->
~11.2k candidates). To spread the <=1000-tile budget across the world's reef provinces
instead of over-representing dense regions (e.g. the Great Barrier Reef), we select
round-robin across 0.25-degree geographic cells (~3.2k occupied cells), so nearly every
selected tile is a distinct cell.

TIME RANGE: reef locations are persistent geological/biological structures (the source
compiles surveys mostly from 1989-2002, but the reefs remain in place). Per spec section 5
(static labels), we assign a representative 1-year window in the Sentinel era (2020),
change_time = null. This matches the sibling allen_coral_atlas handling.

Run: python3 -m olmoearth_pretrain.open_set_segmentation_data.datasets.unep_wcmc_global_warm_water_coral_reefs
"""

import argparse
import multiprocessing
import random
from collections import defaultdict
from typing import Any

import geopandas as gpd
import numpy as np
import shapely
from pyproj import Transformer
from rasterio.crs import CRS
from rslearn.utils.geometry import Projection
from rslearn.utils.mp import star_imap_unordered

from olmoearth_pretrain.open_set_segmentation_data import download, io, manifest
from olmoearth_pretrain.open_set_segmentation_data.rasterize import rasterize_shapes

SLUG = "unep_wcmc_global_warm_water_coral_reefs"
NAME = "UNEP-WCMC Global Warm-Water Coral Reefs"

DL_URL = (
    "https://datadownload-production.s3.us-east-1.amazonaws.com/"
    "WCMC008_CoralReefs2021_v4_1.zip"
)
ZIP_NAME = "WCMC008_CoralReefs2021_v4_1.zip"
PY_REL = (
    "extracted/14_001_WCMC008_CoralReefs2021_v4_1/01_Data/"
    "WCMC008_CoralReef2021_Py_v4_1.shp"
)

TILE = io.MAX_TILE  # 64 px @ 10 m = 640 m
NODATA = io.CLASS_NODATA  # 255; positive-only reef map -> outside reef = ignore
REEF_ID = 0  # single foreground class
PER_CLASS = 1000  # single-class classification target (spec section 5)
YEAR = 2020  # representative 1-year Sentinel-era window (reefs are persistent)

MIN_POLY_AREA_KM2 = (
    0.01  # skip sub-0.01 km2 slivers (~< 100 px @ 10 m; not confidently resolvable)
)
MIN_REEF_PIXELS = 16  # a written tile must contain >= this many reef pixels
CELL_DEG = 0.25  # geographic cell size for round-robin diversity sampling
BBOX_PAD_DEG = 0.02  # WGS84 pad when collecting polygons intersecting a tile
SEED = 42

CLASS_DESCRIPTION = (
    "Warm-water coral reef presence: tropical / subtropical shallow coral-reef area as "
    "mapped in the UNEP-WCMC Global Distribution of Warm-Water Coral Reefs v4.1 (WCMC-008), "
    "compiled from the Millennium Coral Reef Mapping Project, the World Atlas of Coral Reefs "
    "(Spalding et al. 2001) and other sources. Presence-only footprint (reef vs. unmapped)."
)


def _download_and_extract() -> gpd.GeoDataFrame:
    """Download the WCMC-008 zip (idempotent) and return the reef-polygon GeoDataFrame."""
    raw = io.raw_dir(SLUG)
    raw.mkdir(parents=True, exist_ok=True)
    zip_path = raw / ZIP_NAME
    if not zip_path.exists():
        print(f"downloading {ZIP_NAME} ...", flush=True)
        download.download_http(
            DL_URL,
            zip_path,
            headers={"User-Agent": "Mozilla/5.0 (olmoearth-pretrain)"},
            timeout=1800,
        )
    download.extract_zip(zip_path, raw / "extracted")
    with (raw / "SOURCE.txt").open("w") as f:
        f.write(
            "UNEP-WCMC, WorldFish Centre, WRI, TNC (2021). Global distribution of warm-water "
            "coral reefs, v4.1 (WCMC-008). Data DOI 10.34892/t2wk-5t34.\n"
            f"Public S3 download: {DL_URL}\n"
            "Layers: WCMC008_CoralReef2021_Py_v4_1 (17,504 presence polygons, used) + "
            "WCMC008_CoralReef2021_Pt_v4_1 (925 point-only reefs, excluded as sub-pixel).\n"
            "License: UNEP-WCMC General Data License (excluding WDPA); free use + "
            "attribution, non-commercial.\n"
        )
    return gpd.read_file((raw / PY_REL).path)


def _build_candidates(
    gdf: gpd.GeoDataFrame,
) -> tuple[list[dict[str, Any]], shapely.STRtree, list[Any]]:
    """One deduplicated 64x64 UTM tile per reef polygon (snapped to a per-zone tile grid).

    Returns (records, tree, geoms). Each record is lightweight {epsg, bounds(px), cell,
    source_id, lon, lat}; the WGS84 polygon geometries intersecting a tile are attached
    later (only for selected tiles) via ``_attach_shapes`` using the returned STRtree.
    """
    gdf = gdf[gdf["GIS_AREA_K"].astype(float) >= MIN_POLY_AREA_KM2].copy()
    geoms = list(gdf.geometry.values)
    reps = gdf.geometry.representative_point()
    lon = reps.x.values
    lat = reps.y.values
    zone = np.floor((lon + 180) / 6).astype(int) + 1
    epsg = np.where(lat >= 0, 32600 + zone, 32700 + zone).astype(int)

    tree = shapely.STRtree(geoms)
    seen: dict[tuple[int, int, int], dict[str, Any]] = {}
    for e in np.unique(epsg):
        m = epsg == e
        idxs = np.nonzero(m)[0]
        fwd = Transformer.from_crs("EPSG:4326", int(e), always_xy=True)
        xs, ys = fwd.transform(lon[m], lat[m])
        xs = np.asarray(xs, dtype=float)
        ys = np.asarray(ys, dtype=float)
        col = np.floor(xs / io.RESOLUTION).astype(int)
        row = np.floor(ys / -io.RESOLUTION).astype(int)  # north-up: py = utm_y / -10
        tc = col // TILE
        tr = row // TILE
        for j in range(len(idxs)):
            key = (int(e), int(tc[j]), int(tr[j]))
            if key in seen:
                continue
            x0, y0 = int(tc[j]) * TILE, int(tr[j]) * TILE
            lo = float(lon[idxs[j]])
            la = float(lat[idxs[j]])
            seen[key] = {
                "epsg": int(e),
                "bounds": [x0, y0, x0 + TILE, y0 + TILE],
                "cell": (round(lo / CELL_DEG), round(la / CELL_DEG)),
                "source_id": f"{int(e)}:{int(tc[j])}:{int(tr[j])}",
                "lon": lo,
                "lat": la,
            }
    return list(seen.values()), tree, geoms


def _attach_shapes(
    records: list[dict[str, Any]], tree: shapely.STRtree, geoms: list[Any]
) -> None:
    """Attach WGS84 polygon geometries (as WKB) intersecting each record's tile bbox."""
    inv_by_epsg: dict[int, Transformer] = {}
    for r in records:
        epsg = r["epsg"]
        inv = inv_by_epsg.get(epsg)
        if inv is None:
            inv = Transformer.from_crs(int(epsg), "EPSG:4326", always_xy=True)
            inv_by_epsg[epsg] = inv
        x0, y0, x1, y1 = r["bounds"]
        ux0, ux1 = x0 * io.RESOLUTION, x1 * io.RESOLUTION
        uy0, uy1 = y0 * -io.RESOLUTION, y1 * -io.RESOLUTION
        cx, cy = inv.transform([ux0, ux1, ux0, ux1], [uy0, uy0, uy1, uy1])
        minx, maxx = min(cx) - BBOX_PAD_DEG, max(cx) + BBOX_PAD_DEG
        miny, maxy = min(cy) - BBOX_PAD_DEG, max(cy) + BBOX_PAD_DEG
        cand = tree.query(shapely.box(minx, miny, maxx, maxy))
        r["shapes_wkb"] = [shapely.to_wkb(geoms[int(k)]) for k in cand]


def _select_round_robin(
    cands: list[dict[str, Any]], target: int
) -> list[dict[str, Any]]:
    """Round-robin across 0.25-degree cells for global diversity; up to ``target`` tiles."""
    by_cell: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    for r in sorted(cands, key=lambda r: r["source_id"]):  # deterministic input order
        by_cell[r["cell"]].append(r)
    rng = random.Random(SEED)
    cells = list(by_cell)
    rng.shuffle(cells)
    for c in cells:
        rng.shuffle(by_cell[c])
    out: list[dict[str, Any]] = []
    i = 0
    while len(out) < target and any(by_cell[c] for c in cells):
        c = cells[i % len(cells)]
        if by_cell[c]:
            out.append(by_cell[c].pop())
        i += 1
    return out


def _write_tile(rec: dict[str, Any]) -> int:
    """Rasterize the reef polygons into one tile (fill=nodata) and write tif + json.

    Returns the number of reef pixels written (0 => skipped).
    """
    sample_id = rec["sample_id"]
    tif = io.locations_dir(SLUG) / f"{sample_id}.tif"
    if tif.exists():
        # Idempotent skip: recover the real reef-pixel count so a re-run keeps stats correct.
        import rasterio

        with rasterio.open(tif.path) as ds:
            return int((ds.read(1) == REEF_ID).sum())
    epsg = rec["epsg"]
    proj = Projection(CRS.from_epsg(epsg), io.RESOLUTION, -io.RESOLUTION)
    bounds = tuple(rec["bounds"])
    fwd = Transformer.from_crs("EPSG:4326", int(epsg), always_xy=True)

    def to_px(coords: np.ndarray) -> np.ndarray:
        x, y = fwd.transform(coords[:, 0], coords[:, 1])
        return np.column_stack(
            [np.asarray(x) / io.RESOLUTION, np.asarray(y) / -io.RESOLUTION]
        )

    shapes = []
    for wkb in rec["shapes_wkb"]:
        g = shapely.from_wkb(wkb)
        if g.is_empty:
            continue
        if not g.is_valid:
            g = g.buffer(0)
            if g.is_empty:
                continue
        pg = shapely.transform(g, to_px)
        if not pg.is_empty:
            shapes.append((pg, REEF_ID))
    if not shapes:
        return 0
    arr = rasterize_shapes(
        shapes, bounds, fill=NODATA, dtype="uint8", all_touched=True
    )[0]
    n_reef = int((arr == REEF_ID).sum())
    if n_reef < MIN_REEF_PIXELS:
        return 0
    io.write_label_geotiff(SLUG, sample_id, arr, proj, bounds, nodata=NODATA)
    io.write_sample_json(
        SLUG,
        sample_id,
        proj,
        bounds,
        io.year_range(YEAR),
        source_id=rec["source_id"],
        classes_present=[REEF_ID],
    )
    return n_reef


def _write_metadata(num_samples: int, reef_px_stats: dict[str, float]) -> None:
    io.write_dataset_metadata(
        SLUG,
        {
            "dataset": SLUG,
            "name": NAME,
            "task_type": "classification",
            "source": "UNEP-WCMC, WorldFish Centre, WRI, TNC",
            "license": "UNEP-WCMC General Data License (excluding WDPA); free + attribution, non-commercial",
            "provenance": {
                "url": "https://data.unep-wcmc.org/datasets/1",
                "have_locally": False,
                "annotation_method": (
                    "Expert compilation of coral-reef presence from the Millennium Coral "
                    "Reef Mapping Project (IMaRS-USF/IRD 2005), the World Atlas of Coral "
                    "Reefs (Spalding et al. 2001) and other sources; UNEP-WCMC v4.1 (2021)"
                ),
                "data_doi": "10.34892/t2wk-5t34",
                "download": DL_URL,
            },
            "sensors_relevant": ["sentinel2", "sentinel1", "landsat"],
            "classes": [
                {
                    "id": REEF_ID,
                    "name": "warm-water coral reef",
                    "description": CLASS_DESCRIPTION,
                }
            ],
            "nodata_value": NODATA,
            "num_samples": num_samples,
            "tile_size": TILE,
            "min_poly_area_km2": MIN_POLY_AREA_KM2,
            "min_reef_pixels": MIN_REEF_PIXELS,
            "reef_pixels_per_tile": reef_px_stats,
            "time_range": [f"{YEAR}-01-01", f"{YEAR + 1}-01-01"],
            "notes": (
                "Positive-only single-class reef PRESENCE from UNEP-WCMC WCMC-008 v4.1 "
                "presence polygons (17,504). Non-reef pixels = nodata 255 (no background "
                "class; assembly supplies negatives). Rasterized reef polygons into 64x64 "
                "UTM tiles at 10 m (all_touched). Bounded global sample of a global product: "
                "one deduplicated tile per reef polygon (area >= 0.01 km2), selected "
                "round-robin across 0.25-deg cells for global reef-province diversity, "
                "capped at 1000. Each tile carries >= 16 reef pixels. The 925 point-only "
                "reefs and sub-0.01 km2 polygon slivers are excluded (sub-pixel / no "
                "resolvable footprint). Static 1-year window (2020, Sentinel era); reefs are "
                "persistent so source survey dates (mostly 1989-2002) are not used as the "
                "label time. change_time = null."
            ),
        },
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=64)
    args = ap.parse_args()

    io.check_disk()
    manifest.write_registry_entry(SLUG, "in_progress")

    gdf = _download_and_extract()
    io.check_disk()
    print(f"loaded {len(gdf)} reef polygons", flush=True)

    cands, tree, geoms = _build_candidates(gdf)
    print(
        f"candidate tiles (dedup, area>= {MIN_POLY_AREA_KM2} km2): {len(cands)}",
        flush=True,
    )
    print(
        f"distinct {CELL_DEG}-deg cells: {len({c['cell'] for c in cands})}", flush=True
    )

    selected = _select_round_robin(cands, PER_CLASS)
    _attach_shapes(selected, tree, geoms)
    selected.sort(key=lambda r: r["source_id"])
    for i, r in enumerate(selected):
        r["sample_id"] = f"{i:06d}"
    print(f"selected tiles: {len(selected)}", flush=True)

    io.check_disk()
    reef_px: list[int] = []
    with multiprocessing.Pool(args.workers) as p:
        for n in star_imap_unordered(p, _write_tile, [dict(rec=r) for r in selected]):
            if n:
                reef_px.append(int(n))
    written = len(reef_px)
    stats = {
        "min": int(min(reef_px)) if reef_px else 0,
        "median": float(np.median(reef_px)) if reef_px else 0.0,
        "max": int(max(reef_px)) if reef_px else 0,
        "mean": float(np.mean(reef_px)) if reef_px else 0.0,
    }
    print(f"written tiles: {written}; reef-pixels/tile stats: {stats}", flush=True)

    _write_metadata(written, stats)
    manifest.write_registry_entry(
        SLUG, "completed", task_type="classification", num_samples=written
    )
    print("done", flush=True)


if __name__ == "__main__":
    multiprocessing.set_start_method("forkserver", force=True)
    main()
