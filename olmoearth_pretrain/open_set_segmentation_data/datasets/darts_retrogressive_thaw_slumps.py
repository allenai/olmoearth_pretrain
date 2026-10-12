"""Process DARTS (Retrogressive Thaw Slumps) into positive-only polygon label tiles.

Source: Nitze, I., Heidler, K., Nesterova, N., Kuepper, J., Schuett, E., Hoelzer, T.,
Barth, S., Lara, M. J., Liljedahl, A., & Grosse, G. (2025). "DARTS: Multi-year database
of AI detected retrogressive thaw slumps (RTS) in hotspots of the circum-arctic permafrost
region - v1.2". Arctic Data Center. https://doi.org/10.18739/A22B8VD7C (open access;
CC-BY-4.0). Companion paper: Sci Data 12, 1512 (2025),
https://doi.org/10.1038/s41597-025-05810-2.

DARTS comprises footprints of active retrogressive thaw slumps (RTS) -- and, lumped into
the same detection target, active-layer detachment slides (ALD) -- automatically segmented
with a U-Net++ deep-learning model from ~3 m PlanetScope imagery (plus ArcticDEM slope /
relative elevation and Landsat trend layers), followed by review/validation. We use the
**Level 2** product: L2 features are the annual maximum RTS extent per calendar year
(L1 per-image footprints dissolved on the ``year`` attribute). 77,405 L2 polygons span
years 2018-2023 across circum-arctic hotspots (NW Canada, Siberia, ...), EPSG:4326.

Task: **positive-only single-class polygon segmentation** (label_type: polygons).

    0 = retrogressive thaw slump / active-layer detachment slide
    255 = nodata / ignore (everything outside a slump footprint)

Only ONE foreground class is expressible: the DARTS release carries no per-feature RTS-vs-
ALD attribute (the model detects RTS + ALD as a single "active slumping" target class),
so the manifest's two classes are represented by one unified class. Per spec section 5 this
is a positive-only / no-background dataset: we do NOT fabricate synthetic negatives; non-
polygon pixels are left 255 (assembly supplies negatives from other datasets).

Annually-resolved handling (spec section 5): each L2 feature is a footprint for one calendar
``year``. A thaw-slump scar is a *persistent* geomorphic feature (it stays visible long
after any single year's retreat), and the annual resolution here is coarser than the
~1-2 month change-timing bar for change labels. So this is treated as **presence/state
classification**: change_time = null, time_range = the static 1-year window anchored on the
feature's observation year (NOT a change label). A slump observed in multiple years yields
one separate annually-resolved sample per year -- that is the intended annual resolution.

Rasterization: each selected feature -> one 64x64 UTM 10 m tile centered on the polygon
(centroid, or an interior representative point for concave shapes). All L2 polygons **of
the same year** intersecting the tile bbox are burned to class 0 (all_touched=True so tiny
slumps -- median ~2188 m^2, ~22 px -- survive); the rest of the tile is 255. Same-year only,
so the mask stays temporally consistent with the tile's 1-year window. Polygons are read on
demand from the GeoPackage with a pyogrio bbox filter (GPKG R-tree spatial index), so both
scan and write phases parallelize over a Pool with no giant geometry tree in worker memory.

Sampling: 77,405 L2 features > the 25,000 per-dataset hard cap, so we take a geographically
stratified round-robin over 1-degree lon/lat cells (as in the sibling mining-polygons
script) to keep dense hotspots from dominating; one tile per selected feature, capped at
25,000 tiles. Year comes along with each feature (2021-2023 dominate, matching the data).

Run: ``python3 -m olmoearth_pretrain.open_set_segmentation_data.datasets.darts_retrogressive_thaw_slumps``
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

from olmoearth_pretrain.open_set_segmentation_data import io, manifest

SLUG = "darts_retrogressive_thaw_slumps"
NAME = "DARTS (Retrogressive Thaw Slumps)"
RAW = str(io.raw_dir(SLUG))
GPKG = os.path.join(RAW, "DARTS_NitzeEtAl_v1-2_features_2018-2023_level2.gpkg")

TILE = 64
TARGET = 25000  # spec hard cap (positive-only, single class)
CELL = 1.0  # geographic stratification cell size (degrees)

SLUMP = 0  # single foreground class id
CLASSES = [
    (
        "retrogressive thaw slump / active-layer detachment slide",
        "Footprint of an active retrogressive thaw slump (RTS) -- a hillslope thermokarst "
        "mass-wasting landform triggered by thawing ice-rich permafrost -- or the active "
        "area within a larger RTS landform. Active-layer detachment slides (ALD), a related "
        "shallow permafrost hillslope failure, are detected by the same model target and "
        "are included in this class (the DARTS v1.2 release provides no per-feature RTS-vs-"
        "ALD attribute). Segmented by a U-Net++ deep-learning model from ~3 m PlanetScope "
        "imagery plus ArcticDEM slope/relative-elevation and Landsat trend layers, then "
        "reviewed/validated (Nitze et al. 2025). This is the Level 2 product: annual "
        "maximum extent per calendar year.",
    ),
]


# --------------------------------------------------------------------------- scan


def load_placement_points() -> dict[str, np.ndarray]:
    """Read all L2 polygons once; return WGS84 placement point + year + id per feature.

    Placement point is the centroid, unless it falls outside the polygon (concave shape),
    in which case a guaranteed-interior representative point is used so the tile centered
    there always contains foreground.
    """
    import pyogrio
    import shapely

    gdf = pyogrio.read_dataframe(GPKG, columns=["year", "id"], read_geometry=True)
    geoms = gdf.geometry.values
    n = len(geoms)
    lon = np.full(n, np.nan, dtype="float64")
    lat = np.full(n, np.nan, dtype="float64")
    for i, g in enumerate(geoms):
        if g is None or g.is_empty:
            continue
        c = shapely.centroid(g)
        if not g.contains(c):
            c = shapely.force_2d(g).representative_point()
        lon[i] = c.x
        lat[i] = c.y
    return {
        "lon": lon,
        "lat": lat,
        "year": gdf["year"].values.astype("int64"),
        "id": gdf["id"].values.astype("int64"),
    }


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


def _read_slumps_bbox(
    bbox_wgs84: tuple[float, float, float, float], year: int
) -> list[Any]:
    """Read L2 slump geometries of ``year`` whose envelope intersects a WGS84 bbox."""
    import pyogrio

    gdf = pyogrio.read_dataframe(
        GPKG, bbox=bbox_wgs84, columns=["year"], read_geometry=True
    )
    return [
        g
        for g, y in zip(gdf.geometry.values, gdf["year"].values)
        if g is not None and not g.is_empty and int(y) == year
    ]


def _rasterize_tile(lon: float, lat: float, year: int) -> tuple[Any, Any, np.ndarray]:
    """Rasterize all same-year slump polygons intersecting a tile centered on lon/lat."""
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
    geoms = _read_slumps_bbox((lon - mlon, lat - mlat, lon + mlon, lat + mlat), year)
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
            shapes.append((px, SLUMP))

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

    lon, lat, year = rec["lon"], rec["lat"], rec["year"]
    proj, bounds, label = _rasterize_tile(lon, lat, year)

    present = sorted(int(v) for v in np.unique(label) if v != io.CLASS_NODATA)
    io.write_label_geotiff(SLUG, sample_id, label, proj, bounds, nodata=io.CLASS_NODATA)
    io.write_sample_json(
        SLUG,
        sample_id,
        proj,
        bounds,
        io.year_range(year),
        change_time=None,  # persistent state, annually resolved (see module docstring)
        source_id=rec["source_id"],
        classes_present=present,
    )
    return "slump" if SLUMP in present else "empty"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=64)
    ap.add_argument("--limit", type=int, default=0, help="debug: cap number of tiles")
    args = ap.parse_args()

    io.check_disk()
    manifest.write_registry_entry(SLUG, "in_progress")

    raw = io.raw_dir(SLUG)
    raw.mkdir(parents=True, exist_ok=True)
    with (raw / "SOURCE.txt").open("w") as f:
        f.write(
            "DARTS: Multi-year database of AI detected retrogressive thaw slumps (RTS) in "
            "hotspots of the circum-arctic permafrost region - v1.2. Nitze et al. 2025, "
            "Arctic Data Center, https://doi.org/10.18739/A22B8VD7C (CC-BY-4.0). Companion "
            "paper: Sci Data 12, 1512 (2025), https://doi.org/10.1038/s41597-025-05810-2.\n"
            "Level 2 features file downloaded directly (no account needed) from the "
            "Arctic Data Center / DataONE:\n"
            "https://arcticdata.io/metacat/d1/mn/v2/object/urn%3Auuid%3Af1169dfd-0b3e-405f-9c56-4ff3c4827316\n"
            "-> DARTS_NitzeEtAl_v1-2_features_2018-2023_level2.gpkg (77,405 annually "
            "aggregated RTS/ALD polygons, EPSG:4326, years 2018-2023). Level 1 (per-image) "
            "and coverage files not used.\n"
        )

    print("loading polygon placement points ...")
    cen = load_placement_points()
    lon, lat, year, fid = cen["lon"], cen["lat"], cen["year"], cen["id"]
    valid = int(np.sum(~np.isnan(lon)))
    print(f"total L2 polygons: {len(lon)} (valid placement points: {valid})")

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
                "year": int(year[gi]),
                "source_id": f"darts_l2:{int(fid[gi])}:{int(year[gi])}",
            }
        )

    # Report the year distribution of the selected subset.
    yr_counts = Counter(r["year"] for r in records)
    print("selected year distribution:", dict(sorted(yr_counts.items())))
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
            "source": "NSF Arctic Data Center / Sci Data (Nitze et al. 2025, DARTS v1.2)",
            "license": "CC-BY-4.0",
            "provenance": {
                "url": "https://doi.org/10.18739/A22B8VD7C",
                "have_locally": False,
                "annotation_method": "U-Net++ deep-learning segmentation of ~3 m "
                "PlanetScope imagery (+ ArcticDEM slope/relative elevation, Landsat "
                "trends), with review/validation; Level 2 = annual maximum extent per year",
            },
            "sensors_relevant": ["sentinel2", "sentinel1", "landsat"],
            "classes": [
                {"id": i, "name": name, "description": desc}
                for i, (name, desc) in enumerate(CLASSES)
            ],
            "nodata_value": io.CLASS_NODATA,
            "num_samples": n_written,
            "notes": (
                "Positive-only single-class permafrost thaw-slump polygon segmentation from "
                "DARTS v1.2 Level 2 (annually aggregated RTS/ALD footprints; 77,405 polygons, "
                "2018-2023). 64x64 uint8 tiles in local UTM at 10 m; class 0 = thaw slump / "
                "active-layer detachment slide, 255 = nodata (all non-polygon pixels -- NO "
                "synthetic negatives per spec section 5; assembly adds negatives from other "
                "datasets). One tile per selected feature, centered on the polygon (centroid, "
                "or interior representative point for concave shapes); all SAME-YEAR L2 "
                "polygons intersecting a tile are burned in (all_touched=True so tiny slumps "
                "survive). Only one foreground class is expressible: v1.2 carries no per-"
                "feature RTS-vs-ALD attribute, so the manifest's two classes are unified. "
                "Annually resolved => presence/state classification: change_time=null, "
                "time_range = static 1-year window on the feature's observation year (NOT a "
                "change label -- annual resolution is coarser than the ~1-2 month change bar; "
                "a slump scar is persistent). A slump seen in multiple years yields one "
                "sample per year. Geographically-stratified round-robin over 1-degree cells "
                "so dense hotspots (Banks Island, Peel Plateau, ...) do not dominate; capped "
                "at 25,000 tiles (of 77,405). All labels 2018-2023 (Sentinel era)."
            ),
        },
    )
    print("tile counts:", dict(counts))
    print("total tif on disk:", n_written)

    manifest.write_registry_entry(
        SLUG, "completed", task_type="classification", num_samples=n_written
    )
    print("done")


if __name__ == "__main__":
    multiprocessing.set_start_method("forkserver", force=True)
    main()
