"""Process the Global Mining Polygons (Maus et al. v2) product into single-class
mining-area label tiles (positive-only segmentation).

Source: Maus, V. et al. (2022), "Global-scale mining polygons (Version 2)", PANGAEA,
https://doi.org/10.1594/PANGAEA.942325 (supplement to Maus et al., "An update on global
mining land use", Scientific Data 9, 433, 2022). License CC-BY-SA-4.0.

The main file ``global_mining_polygons_v2.gpkg`` (23.5 MB, downloaded directly, no account
needed) holds 44,929 hand-digitized mining land-use polygons covering 101,583 km^2 across
145 countries. Fields: ISO3_CODE, COUNTRY_NAME, AREA (km^2), geom (WGS84 EPSG:4326
polygons). Each polygon delineates ALL ground features related to a mine — open cuts,
tailings dams, waste-rock dumps, water ponds, processing infrastructure — as one
undifferentiated "mining area" footprint (there is no per-polygon feature-type attribute).
Digitized by visual interpretation of the 2019 Sentinel-2 cloudless mosaic (10 m), aided by
Google/Bing imagery, within a 10 km buffer of 34,820 S&P mining coordinates.

Task: **positive-only single-class polygon segmentation** (label_type: polygons).

    0 = mining area (inside a Maus et al. mining polygon)
    255 = nodata / ignore (everything outside a polygon)

Per spec section 5, this is a positive-only / no-background dataset: we do NOT fabricate
synthetic negatives. Non-polygon pixels are left as nodata (255); the pretraining assembly
step supplies negatives by sampling locations from other datasets.

Rasterization: each selected polygon -> one <=64x64 UTM 10 m tile centered on the polygon
(placement point = centroid, or a representative interior point if the centroid falls in a
concavity). All Maus polygons intersecting the tile bbox are rasterized to class 0
(all_touched=True so tiny mines survive); the rest of the tile is 255. Polygons larger than
a 640 m tile (~40% of the set, up to 2546 km^2) are captured as a central window of the
mine — still a valid all-mining positive patch. Polygon geometries intersecting a tile are
read on demand from the GeoPackage with a pyogrio bbox filter (uses the GPKG R-tree spatial
index), so both scan and write phases parallelize over a Pool with no giant in-memory tree.

Sampling: geographically stratified round-robin over 1-degree lon/lat cells (as in the
sibling global_mining_footprint_tang_werner script) so dense mining regions do not dominate;
one tile per selected polygon, capped at 25,000 tiles total (spec hard cap).

Time range: labels reflect the 2019 Sentinel-2 mosaic the mines were digitized from and
mining land use is persistent, so each tile gets the 1-year window 2019 (anchored on the
labeled year). See summary for why 2019 rather than the manifest's 2016-2019 span.

Run: ``python3 -m olmoearth_pretrain.open_set_segmentation_data.datasets.global_mining_polygons_maus_et_al_v2``
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

SLUG = "global_mining_polygons_maus_et_al_v2"
NAME = "Global Mining Polygons (Maus et al. v2)"
RAW = str(io.raw_dir(SLUG))
GPKG = os.path.join(RAW, "global_mining_polygons_v2.gpkg")

TILE = 64
TARGET = 25000  # spec hard cap (positive-only, single class)
CELL = 1.0  # geographic stratification cell size (degrees)
YEAR = 2019  # labeled year (2019 Sentinel-2 mosaic used for digitization)

CLASSES = [
    (
        "mining area",
        "Land used by the mining industry: the full surface footprint of a mine as one "
        "undifferentiated class, including open cuts/pits, tailings dams, waste-rock "
        "dumps, water ponds, processing infrastructure and other mining-related land "
        "cover. Hand-digitized by visual interpretation of the 2019 Sentinel-2 cloudless "
        "10 m mosaic (Maus et al. 2022), within a 10 km buffer of S&P mining coordinates.",
    ),
]
MINE = 0  # single foreground class id


# --------------------------------------------------------------------------- scan


def load_placement_points() -> dict[str, np.ndarray]:
    """Read all 44,929 polygons once; return a WGS84 lon/lat placement point per polygon.

    Placement point is the centroid, unless the centroid falls outside the polygon (a
    concave shape), in which case a guaranteed-interior representative point is used so the
    tile centered there always contains foreground.
    """
    import pyogrio
    import shapely

    gdf = pyogrio.read_dataframe(GPKG, columns=["ISO3_CODE"], read_geometry=True)
    geoms = gdf.geometry.values
    lon = np.full(len(geoms), np.nan, dtype="float64")
    lat = np.full(len(geoms), np.nan, dtype="float64")
    for i, g in enumerate(geoms):
        if g is None or g.is_empty:
            continue
        c = shapely.centroid(g)
        if not g.contains(c):
            c = shapely.force_2d(g).representative_point()
        lon[i] = c.x
        lat[i] = c.y
    return {"lon": lon, "lat": lat, "iso3": gdf["ISO3_CODE"].values}


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


def _read_mines_bbox(bbox_wgs84: tuple[float, float, float, float]) -> list[Any]:
    """Read mining geometries whose envelope intersects a WGS84 lon/lat bbox."""
    import pyogrio

    gdf = pyogrio.read_dataframe(GPKG, bbox=bbox_wgs84, columns=[], read_geometry=True)
    return [g for g in gdf.geometry.values if g is not None and not g.is_empty]


def _rasterize_tile(lon: float, lat: float) -> tuple[Any, Any, np.ndarray]:
    """Rasterize all mining polygons intersecting a tile centered on lon/lat."""
    import shapely
    from rslearn.const import WGS84_PROJECTION
    from shapely.geometry import box

    from olmoearth_pretrain.open_set_segmentation_data.rasterize import (
        geom_to_pixels,
        rasterize_shapes,
    )

    proj, col, row = io.lonlat_to_utm_pixel(lon, lat)
    bounds = io.centered_bounds(col, row, TILE, TILE)

    # Query box in lon/lat covering the tile with a generous margin (2x tile) so polygons
    # extending in from the sides are included.
    mlat = (2 * TILE * io.RESOLUTION) / 111320.0
    mlon = mlat / max(math.cos(math.radians(lat)), 0.1)
    geoms = _read_mines_bbox((lon - mlon, lat - mlat, lon + mlon, lat + mlat))
    ll_box = box(lon - mlon, lat - mlat, lon + mlon, lat + mlat)

    shapes = []
    for g in geoms:
        g = shapely.force_2d(g)
        try:
            gc = g.intersection(ll_box)
        except Exception:
            gc = g
        if gc.is_empty:
            continue
        px = geom_to_pixels(gc, WGS84_PROJECTION, proj)
        if not px.is_empty:
            shapes.append((px, MINE))

    if shapes:
        label = rasterize_shapes(
            shapes, bounds, fill=io.CLASS_NODATA, dtype="uint8", all_touched=True
        )[0]
    else:
        label = np.full((TILE, TILE), io.CLASS_NODATA, dtype=np.uint8)
    return proj, bounds, label


def _write_one(rec: dict[str, Any]) -> str | None:
    sample_id = rec["sample_id"]
    if (io.locations_dir(SLUG) / f"{sample_id}.tif").exists():
        return "skip"

    lon, lat = rec["lon"], rec["lat"]
    proj, bounds, label = _rasterize_tile(lon, lat)

    # Guard: if the centroid-centered window happened to miss the polygon entirely
    # (large concave mine), retry once centered on the polygon's representative point.
    if int((label == MINE).sum()) == 0 and rec.get("alt") is not None:
        alon, alat = rec["alt"]
        proj, bounds, label = _rasterize_tile(alon, alat)

    present = sorted(int(v) for v in np.unique(label) if v != io.CLASS_NODATA)
    io.write_label_geotiff(SLUG, sample_id, label, proj, bounds, nodata=io.CLASS_NODATA)
    io.write_sample_json(
        SLUG,
        sample_id,
        proj,
        bounds,
        io.year_range(YEAR),
        source_id=rec["source_id"],
        classes_present=present,
    )
    return "mine" if MINE in present else "empty"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=64)
    ap.add_argument("--limit", type=int, default=0, help="debug: cap number of tiles")
    args = ap.parse_args()

    io.check_disk()

    raw = io.raw_dir(SLUG)
    raw.mkdir(parents=True, exist_ok=True)
    with (raw / "SOURCE.txt").open("w") as f:
        f.write(
            "Global-scale mining polygons (Version 2), Maus et al. 2022. PANGAEA "
            "https://doi.org/10.1594/PANGAEA.942325 (CC-BY-SA-4.0). Supplement to Maus et "
            "al., Sci Data 9, 433 (2022), https://doi.org/10.1038/s41597-022-01547-4.\n"
            "Main file downloaded directly (no account needed): "
            "https://download.pangaea.de/dataset/942325/files/global_mining_polygons_v2.gpkg\n"
            "44,929 mining land-use polygons (WGS84 EPSG:4326); fields ISO3_CODE, "
            "COUNTRY_NAME, AREA (km^2). Grid/CSV/validation-point files not used.\n"
        )

    print("loading polygon placement points ...")
    cen = load_placement_points()
    lon, lat = cen["lon"], cen["lat"]
    valid = int(np.sum(~np.isnan(lon)))
    print(f"total polygons: {len(lon)} (valid placement points: {valid})")

    io.check_disk()

    n = TARGET if args.limit <= 0 else min(TARGET, args.limit)
    sel = stratified_indices(lon, lat, n, seed=1)
    print(f"selected {len(sel)} polygons (stratified over 1-degree cells)")

    records: list[dict[str, Any]] = []
    for sid, gi in enumerate(sel):
        records.append(
            {
                "sample_id": f"{sid:06d}",
                "lon": float(lon[gi]),
                "lat": float(lat[gi]),
                "alt": (float(lon[gi]), float(lat[gi])),
                "source_id": f"maus_v2_poly:{gi}",
            }
        )

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
            "source": "PANGAEA / Sci Data (Maus et al. 2022)",
            "license": "CC-BY-SA-4.0",
            "provenance": {
                "url": "https://doi.org/10.1594/PANGAEA.942325",
                "have_locally": False,
                "annotation_method": "manual photointerpretation of the 2019 Sentinel-2 "
                "10 m cloudless mosaic (aided by Google/Bing), 145 countries",
            },
            "sensors_relevant": ["sentinel2", "sentinel1", "landsat"],
            "classes": [
                {"id": i, "name": name, "description": desc}
                for i, (name, desc) in enumerate(CLASSES)
            ],
            "nodata_value": io.CLASS_NODATA,
            "num_samples": n_written,
            "notes": (
                "Positive-only single-class mining-area polygon segmentation from the Maus "
                "et al. v2 global mining polygons (44,929 hand-digitized polygons). 64x64 "
                "uint8 tiles in local UTM at 10 m; class 0 = mining area, 255 = nodata "
                "(all non-polygon pixels — NO synthetic negatives per spec section 5; "
                "assembly adds negatives from other datasets). One tile per polygon, "
                "centered on the polygon (centroid, or interior representative point for "
                "concave shapes); all Maus polygons intersecting a tile are burned in "
                "(all_touched=True). Geographically-stratified round-robin over 1-degree "
                "cells so dense mining regions do not dominate; capped at 25,000 tiles. "
                "Large polygons (~40% exceed a 640 m tile, up to 2546 km^2) are captured "
                "as a central all-mining window. The 6 fine feature types listed in the "
                "manifest (pits/tailings/waste-rock/ponds/processing) are NOT per-polygon "
                "attributes in the release, so only the undifferentiated mining footprint "
                "is expressible. Time range = 1-year window anchored on 2019 (the year of "
                "the Sentinel-2 mosaic used for digitization)."
            ),
        },
    )
    print("tile counts:", dict(counts))
    print("total tif on disk:", n_written)
    print("done")


if __name__ == "__main__":
    multiprocessing.set_start_method("forkserver", force=True)
    main()
