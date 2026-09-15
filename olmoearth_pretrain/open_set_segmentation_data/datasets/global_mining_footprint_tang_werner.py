"""Process the Global Mining Footprint (Tang & Werner) polygon product into binary
mine-footprint label tiles.

Source: Tang & Werner, "Global mining footprint mapped from high-resolution satellite
imagery", Communications Earth & Environment (2023). Zenodo record 6806817
(https://doi.org/10.5281/zenodo.6806817), CC-BY-4.0. The archive
``Supplementary 1：mine area polygons.rar`` (RAR5, 103 MB) contains three vector layers:

  * ``74548 mine polygons/74548_projected.shp`` — the headline product: 74,548 mine-area
    polygons finely contouring the surface footprint of mining across 135 countries,
    digitised by manual photointerpretation of high-resolution imagery (~2019). CRS is
    "WGS 84 / NSIDC EASE-Grid Global" (cylindrical equal area, metres). Polygon Z.
  * ``Artisanal and small-scale mine/...shp`` (4,058) and ``Larger scale mine/...gdb``
    (761) — separate ASM / large-scale subsets, not used here (see summary).

Task: **binary mine-footprint vs background segmentation** (label_type: polygons).

    0 = background (not a mapped mine)
    1 = mine (inside a Tang & Werner mine-area polygon)

Why binary: the manifest lists 6 fine mine-feature classes (waste-rock dumps, pits,
ponds, tailings dams, heap leach, processing), but the RELEASED polygons carry NO
per-polygon feature-type attribute. The ``Name`` field is just leftover KML placemark
text (mostly "多边形"/"未命名多边形" = "polygon"/"unnamed polygon", "Placemark", etc.),
not a class code. Per-feature-type classification is therefore not expressible from this
release, so we map to the well-supported binary mine/background signal (mines are large —
median polygon area ~0.12 km^2, i.e. tens of 10 m pixels — clearly observable at 10-30 m
from S2/S1/Landsat).

We rasterize mine polygons into 64x64 UTM 10 m tiles with the same BOUNDED, geographically
stratified strategy as the GLAKES script (round-robin over 1-degree lon/lat cells), capped
at 25,000 tiles total:

  * positive : centered on a stratified sample of mine-polygon centroids. All Tang & Werner
               polygons intersecting the 640 m tile are rasterized to class 1; the rest is
               background 0. Large mines fill most of the tile; small mines sit in
               background, so positive tiles usually contain both classes.
  * negative : background-only tiles, produced by offsetting a stratified sample of mine
               anchors by a random ~3-9 km vector so no mine falls in the tile. If the
               offset still clips a mapped mine, that mine is simply rasterized correctly
               (the tile then counts as a mine tile).

Per-tile intersecting polygons are read directly from the shapefile with a pyogrio bbox
spatial filter (in the source EASE-Grid CRS, using the .sbn spatial index), so no giant
in-memory STRtree is needed and both scan and write phases parallelize over a Pool.

Time range: mine footprints are quasi-static (imagery ~2019, manifest window 2016-2019).
Each tile gets a 1-year window with start year uniform in 2016-2019 (Sentinel era).

Run: ``python3 -m olmoearth_pretrain.open_set_segmentation_data.datasets.global_mining_footprint_tang_werner``
Idempotent: existing ``locations/{id}.tif`` are skipped.
"""

import argparse
import math
import multiprocessing
import os
import random
from collections import Counter, defaultdict
from typing import Any

import numpy as np
import tqdm
from rslearn.utils.mp import star_imap_unordered

from olmoearth_pretrain.open_set_segmentation_data import io

SLUG = "global_mining_footprint_tang_werner"
NAME = "Global Mining Footprint (Tang & Werner)"
RAW = str(io.raw_dir(SLUG))
_ARCHIVE_ROOT = os.path.join(RAW, "extract", "Supplementary 1：mine area polygons")
SHP = os.path.join(_ARCHIVE_ROOT, "74548 mine polygons", "74548_projected.shp")
PRJ = os.path.join(_ARCHIVE_ROOT, "74548 mine polygons", "74548_projected.prj")

# Source CRS WKT (WGS 84 / NSIDC EASE-Grid Global, metres) — read once at import.
with open(PRJ) as _f:
    SRC_WKT = _f.read().strip()

TILE = 64
POS_TARGET = 20000
NEG_TARGET = 5000
CELL = 1.0  # geographic stratification cell size (degrees)
YEARS = [2016, 2017, 2018, 2019]

CLASSES = [
    (
        "background",
        "Not a mapped mine: land or any surface outside a Tang & Werner mine-area polygon.",
    ),
    (
        "mine",
        "Inside a Tang & Werner mine-area polygon: the surface footprint of mining "
        "(open-pit/quarry excavations, waste-rock dumps, tailings/ponds, heap-leach and "
        "processing areas, etc., undifferentiated), manually contoured from high-resolution "
        "satellite imagery (~2019) across 135 countries.",
    ),
]
BG, MINE = 0, 1


# --------------------------------------------------------------------------- projections


def _src_proj():
    from rasterio.crs import CRS
    from rslearn.utils.geometry import Projection

    return Projection(CRS.from_wkt(SRC_WKT), 1, 1)


def _to_lonlat(xs: np.ndarray, ys: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Vectorized reprojection of EASE-Grid metre coords -> WGS84 lon/lat."""
    from pyproj import Transformer
    from rasterio.crs import CRS

    tr = Transformer.from_crs(CRS.from_wkt(SRC_WKT), "EPSG:4326", always_xy=True)
    lon, lat = tr.transform(xs, ys)
    return np.asarray(lon), np.asarray(lat)


# --------------------------------------------------------------------------- scan


def load_centroids() -> dict[str, np.ndarray]:
    """Read all 74548 polygons once; return EASE centroids + WGS84 lon/lat."""
    import pyogrio
    import shapely

    gdf = pyogrio.read_dataframe(SHP, columns=[], read_geometry=True)
    geoms = gdf.geometry.values
    cx = np.empty(len(geoms), dtype="float64")
    cy = np.empty(len(geoms), dtype="float64")
    for i, g in enumerate(geoms):
        if g is None or g.is_empty:
            cx[i] = np.nan
            cy[i] = np.nan
            continue
        c = shapely.centroid(g)
        cx[i] = c.x
        cy[i] = c.y
    lon, lat = _to_lonlat(cx, cy)
    return {"cx": cx, "cy": cy, "lon": lon, "lat": lat}


def stratified_indices(
    lons: np.ndarray, lats: np.ndarray, n: int, seed: int
) -> list[int]:
    """Round-robin over 1-degree lon/lat cells for geographic spread; up to n indices."""
    cells: dict[tuple, list] = defaultdict(list)
    for i in range(len(lons)):
        if math.isnan(lons[i]) or math.isnan(lats[i]):
            continue
        cells[
            (int(math.floor(lons[i] / CELL)), int(math.floor(lats[i] / CELL)))
        ].append(i)
    rng = random.Random(seed)
    order = list(cells.values())
    for lst in order:
        rng.shuffle(lst)
    rng.shuffle(order)
    out: list[int] = []
    i = 0
    while len(out) < n and order:
        lst = order[i % len(order)]
        if lst:
            out.append(lst.pop())
        i += 1
        if i % len(order) == 0:
            order = [l for l in order if l]
    return out[:n]


# --------------------------------------------------------------------------- write


def _read_mines_bbox(bbox_ease: tuple[float, float, float, float]) -> list[Any]:
    """Read mine geometries whose envelope intersects an EASE-metre bbox."""
    import pyogrio

    gdf = pyogrio.read_dataframe(SHP, bbox=bbox_ease, columns=[], read_geometry=True)
    return [g for g in gdf.geometry.values if g is not None and not g.is_empty]


def _write_one(rec: dict[str, Any]) -> str | None:
    import shapely
    from rslearn.const import WGS84_PROJECTION
    from shapely.geometry import box

    from olmoearth_pretrain.open_set_segmentation_data.rasterize import (
        geom_to_pixels,
        rasterize_shapes,
    )

    sample_id = rec["sample_id"]
    if (io.locations_dir(SLUG) / f"{sample_id}.tif").exists():
        return None

    lon, lat = rec["lon"], rec["lat"]
    proj, col, row = io.lonlat_to_utm_pixel(lon, lat)
    bounds = io.centered_bounds(col, row, TILE, TILE)
    src_proj = _src_proj()

    # Geographic query box (lon/lat) covering the tile with generous margin, reprojected
    # to EASE metres for the pyogrio bbox filter and for clipping huge polygons.
    mlat = (TILE * io.RESOLUTION) / 111320.0  # ~2x tile size in deg lat (safety)
    mlon = mlat / max(math.cos(math.radians(lat)), 0.1)
    ll_box = box(lon - mlon, lat - mlat, lon + mlon, lat + mlat)
    clip_ll_ease = geom_to_pixels(ll_box, WGS84_PROJECTION, src_proj)
    qminx, qminy, qmaxx, qmaxy = clip_ll_ease.bounds
    geoms = _read_mines_bbox((qminx, qminy, qmaxx, qmaxy))

    shapes = []
    for g in geoms:
        g = shapely.force_2d(g)
        try:
            gc = g.intersection(clip_ll_ease)
        except Exception:
            gc = g
        if gc.is_empty:
            continue
        px = geom_to_pixels(gc, src_proj, proj)
        if not px.is_empty:
            shapes.append((px, MINE))

    if shapes:
        label = rasterize_shapes(
            shapes, bounds, fill=BG, dtype="uint8", all_touched=True
        )[0]
    else:
        label = np.full((TILE, TILE), BG, dtype=np.uint8)

    present = sorted(int(v) for v in np.unique(label))
    io.write_label_geotiff(SLUG, sample_id, label, proj, bounds, nodata=io.CLASS_NODATA)
    io.write_sample_json(
        SLUG,
        sample_id,
        proj,
        bounds,
        io.year_range(rec["year"]),
        source_id=rec["source_id"],
        classes_present=present,
    )
    return "mine" if MINE in present else "background"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=64)
    args = ap.parse_args()

    io.check_disk()

    raw = io.raw_dir(SLUG)
    raw.mkdir(parents=True, exist_ok=True)
    with (raw / "SOURCE.txt").open("w") as f:
        f.write(
            "Global Mining Footprint, Tang & Werner, Commun. Earth Environ. 2023. "
            "Zenodo record 6806817 (CC-BY-4.0). "
            "https://doi.org/10.5281/zenodo.6806817\n"
            "File: 'Supplementary 1：mine area polygons.rar' (RAR5, 103 MB) -> extract/ "
            "with 74548 mine polygons/74548_projected.shp (74,548 mine-area polygons, "
            "WGS 84 / NSIDC EASE-Grid Global). Also ASM (.shp, 4058) and large-scale "
            "mine (.gdb, 761) subsets, unused. Download via "
            "'https://zenodo.org/records/6806817/files/Supplementary%201%EF%BC%9Amine%20"
            "area%20polygons.rar?download=1' (API /content path returns HTTP 403 behind "
            "Zenodo rate limiting); extract RAR5 with a modern 7z (conda-forge '7zip').\n"
        )

    # ---- Phase A: read all polygons once, compute centroids, stratified selection.
    print("loading centroids ...")
    cen = load_centroids()
    lon, lat = cen["lon"], cen["lat"]
    valid = np.sum(~np.isnan(lon))
    print(f"total polygons: {len(lon)} (valid centroids: {valid})")

    io.check_disk()

    pos_idx = stratified_indices(lon, lat, POS_TARGET, seed=1)
    neg_idx = stratified_indices(lon, lat, NEG_TARGET, seed=2)
    print(f"selected positives={len(pos_idx)} negative-anchors={len(neg_idx)}")

    rng = random.Random(7)
    records: list[dict[str, Any]] = []
    sid = 0

    for gi in pos_idx:
        records.append(
            {
                "sample_id": f"{sid:06d}",
                "kind": "pos",
                "lon": float(lon[gi]),
                "lat": float(lat[gi]),
                "year": rng.choice(YEARS),
                "source_id": f"poly:{gi}",
            }
        )
        sid += 1

    for gi in neg_idx:
        lat0 = float(lat[gi])
        dist_deg = rng.uniform(0.03, 0.08)  # ~3-9 km
        ang = rng.uniform(0, 2 * math.pi)
        dlat = dist_deg * math.sin(ang)
        dlon = dist_deg * math.cos(ang) / max(math.cos(math.radians(lat0)), 0.1)
        records.append(
            {
                "sample_id": f"{sid:06d}",
                "kind": "neg",
                "lon": float(lon[gi]) + dlon,
                "lat": max(min(lat0 + dlat, 83.0), -83.0),
                "year": rng.choice(YEARS),
                "source_id": f"poly:{gi}:neg",
            }
        )
        sid += 1

    print(f"total records to write: {len(records)}")
    io.check_disk()

    counts: Counter = Counter()
    with multiprocessing.Pool(args.workers) as p:
        for res in tqdm.tqdm(
            star_imap_unordered(p, _write_one, [dict(rec=r) for r in records]),
            total=len(records),
            desc="write tiles",
        ):
            if res is not None:
                counts[res] += 1

    n_written = len(list(io.locations_dir(SLUG).glob("*.tif")))
    io.write_dataset_metadata(
        SLUG,
        {
            "dataset": SLUG,
            "name": NAME,
            "task_type": "classification",
            "source": "Zenodo / Commun. Earth Environ. (Tang & Werner 2023)",
            "license": "CC-BY-4.0",
            "provenance": {
                "url": "https://doi.org/10.5281/zenodo.6806817",
                "have_locally": False,
                "annotation_method": "manual photointerpretation of high-resolution "
                "satellite imagery (~2019), 135 countries",
            },
            "sensors_relevant": ["sentinel2", "sentinel1", "landsat"],
            "classes": [
                {"id": i, "name": name, "description": desc}
                for i, (name, desc) in enumerate(CLASSES)
            ],
            "nodata_value": io.CLASS_NODATA,
            "num_samples": n_written,
            "tile_counts": {
                "mine_tiles": counts.get("mine", 0),
                "background_tiles": counts.get("background", 0),
            },
            "notes": (
                "Binary mine-footprint vs background segmentation from the Tang & Werner "
                "global mining footprint (74,548 mine-area polygons). 64x64 uint8 tiles "
                "in local UTM at 10 m; classes: 0 background, 1 mine (255 nodata, unused). "
                "Bounded geographically-stratified sampling (round-robin over 1-degree "
                "cells): ~20k positive tiles centered on mine-polygon centroids (all "
                "polygons intersecting a tile rasterized to class 1, all_touched=True) + "
                "~5k background-only negative tiles offset ~3-9 km from mine anchors. "
                "Capped at 25,000 tiles total. Manifest's 6 fine feature-type classes "
                "(pits/ponds/tailings/waste-rock/heap-leach/processing) are NOT present as "
                "per-polygon attributes in the released data, so only the undifferentiated "
                "mine footprint is expressible. Time range = 1-year window, start year "
                "uniform in 2016-2019."
            ),
        },
    )
    print("tile counts:", dict(counts))
    print("total tif on disk:", n_written)
    print("done")


if __name__ == "__main__":
    multiprocessing.set_start_method("forkserver", force=True)
    main()
