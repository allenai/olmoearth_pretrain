"""Process the six-year circum-Antarctic iceberg outline product into iceberg-vs-ocean
binary segmentation label tiles.

Source: "A Six-year circum-Antarctic icebergs dataset (2018-2023)" (Zenodo record
17165466, ESSD, CC-BY-4.0). The ``Iceberg vector outline.zip`` archive contains six
GeoPackages, one per year, named ``{YYYY}10_distribution.gpkg`` -- each is the October
distribution of detected icebergs for that year (2018-2023). Icebergs were detected
from Sentinel-1 SAR via a semi-automated random-forest classifier + manual correction.
All layers are in EPSG:3031 (Antarctic Polar Stereographic). Each feature is a single
iceberg outline Polygon with attributes: lon, lat (centroid), area_km2,
area_uncertainty_km2, perimeter_km, long_axis_km, short_axis_km, mass_gt,
mass_uncertainty_gt. Per-year feature counts range ~35k-51k (~244k total).

Task: **binary iceberg vs ocean/sea-ice segmentation** (label_type: polygons). We
rasterize iceberg polygons into 64x64 UTM 10 m tiles:

    0 = background (open ocean / sea ice: not a mapped iceberg)
    1 = iceberg (inside a mapped iceberg outline polygon)

The source is a large circum-Antarctic vector (~244k polygons over 6 years), so we do
BOUNDED, geographically-stratified sampling (round-robin over 1-degree lon/lat cells,
pooled across all 6 years so the sample mixes years and regions), capped at 25,000 tiles
total. Every tile is centered on a sampled iceberg centroid; all iceberg polygons
intersecting the ~640 m tile are rasterized to class 1 (so nearby bergs are captured),
the rest is background 0. Most icebergs are small (median ~0.25 km2, ~500 m across) so
tiles usually contain both classes; the tail of giant tabular bergs (up to ~5700 km2)
yields all-iceberg tiles, which are still valid labels.

Following spec section 5 (positive-only datasets), we do NOT fabricate extra background-only
negative tiles -- the within-tile ocean around each berg is genuine, spatially-meaningful
background, and the assembly step supplies additional negatives from other datasets.

Time range: each GeoPackage is the October snapshot of its year. Icebergs drift, so a
full-year window would be ill-posed; we assign each tile the 1-MONTH window
[Oct 1, Nov 1) of its source year -- the tightest anchor the product supports. (Bergs
still drift within a month, a residual source of label noise; see summary.)

Run: ``python3 -m olmoearth_pretrain.open_set_segmentation_data.datasets.circum_antarctic_icebergs_sentinel_1``
Idempotent: existing ``locations/{id}.tif`` are skipped.
"""

import argparse
import math
import multiprocessing
import os
import random
from collections import Counter, defaultdict
from datetime import UTC, datetime
from typing import Any

import numpy as np
import tqdm
from rslearn.utils.mp import star_imap_unordered

from olmoearth_pretrain.open_set_segmentation_data import io

SLUG = "circum_antarctic_icebergs_sentinel_1"
NAME = "Circum-Antarctic Icebergs (Sentinel-1)"
RAW = str(io.raw_dir(SLUG))
GPKG_DIR = os.path.join(RAW, "extract", "Iceberg vector outline")

YEARS = [2018, 2019, 2020, 2021, 2022, 2023]
TILE = 64
TOTAL_TARGET = 25000
CELL = 1.0  # geographic stratification cell size (degrees)
QUERY_MARGIN_M = 800.0  # EPSG:3031 half-window (m) for the per-tile spatial filter

CLASSES = [
    (
        "background",
        "Open ocean or sea ice: any surface outside a mapped iceberg outline in the "
        "circum-Antarctic Sentinel-1 iceberg product.",
    ),
    (
        "iceberg",
        "Inside a mapped iceberg outline polygon (Sentinel-1 SAR-detected iceberg, "
        "semi-automated RF classification + manual correction). Icebergs carry "
        "geometric/mass attributes (area_km2, perimeter_km, long/short axis, mass_gt) "
        "in the source, collapsed here to a single per-pixel iceberg class.",
    ),
]
BG, ICEBERG = 0, 1


def gpkg_path(year: int) -> str:
    return os.path.join(GPKG_DIR, f"{year}10_distribution.gpkg")


def month_range(year: int) -> tuple[datetime, datetime]:
    """[Oct 1 year, Nov 1 year) UTC -- the source October snapshot window."""
    return (
        datetime(year, 10, 1, tzinfo=UTC),
        datetime(year, 11, 1, tzinfo=UTC),
    )


# --------------------------------------------------------------------------- scan


def load_centroids(year: int) -> dict[str, np.ndarray]:
    """Read only lon/lat/area_km2 (no geometry) for one year, plus EPSG:3031 x/y."""
    import pyogrio
    from pyproj import Transformer

    df = pyogrio.read_dataframe(
        gpkg_path(year), columns=["lon", "lat", "area_km2"], read_geometry=False
    )
    lon = df["lon"].to_numpy(dtype="float64")
    lat = df["lat"].to_numpy(dtype="float64")
    tr = Transformer.from_crs(4326, 3031, always_xy=True)
    x, y = tr.transform(lon, lat)
    return {
        "lon": lon,
        "lat": lat,
        "area": df["area_km2"].to_numpy(dtype="float64"),
        "x3031": np.asarray(x, dtype="float64"),
        "y3031": np.asarray(y, dtype="float64"),
    }


def stratified_indices(
    lons: np.ndarray, lats: np.ndarray, n: int, seed: int
) -> list[int]:
    """Round-robin over 1-degree lon/lat cells for geographic spread; up to n indices."""
    cells: dict[tuple, list] = defaultdict(list)
    for i in range(len(lons)):
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


# One rslearn Projection wrapping EPSG:3031 as pixel==metre, mirroring WGS84_PROJECTION
# (EPSG:4326_1_1). Source polygon coords are in EPSG:3031 metres.
def _proj_3031():
    from rasterio.crs import CRS
    from rslearn.utils.geometry import Projection

    return Projection(CRS.from_epsg(3031), 1, 1)


def _write_one(rec: dict[str, Any]) -> str | None:
    import pyogrio
    from shapely.geometry import box

    from olmoearth_pretrain.open_set_segmentation_data.rasterize import (
        geom_to_pixels,
        rasterize_shapes,
    )

    sample_id = rec["sample_id"]
    if (io.locations_dir(SLUG) / f"{sample_id}.tif").exists():
        return None

    lon, lat = rec["lon"], rec["lat"]
    x, y = rec["x3031"], rec["y3031"]
    proj, col, row = io.lonlat_to_utm_pixel(lon, lat)
    bounds = io.centered_bounds(col, row, TILE, TILE)

    qbox = (
        x - QUERY_MARGIN_M,
        y - QUERY_MARGIN_M,
        x + QUERY_MARGIN_M,
        y + QUERY_MARGIN_M,
    )
    clip = box(*qbox)
    gdf = pyogrio.read_dataframe(gpkg_path(rec["year"]), bbox=qbox, columns=[])
    geoms = [g for g in gdf.geometry.values if g is not None and not g.is_empty]

    src_proj = _proj_3031()
    shapes = []
    for g in geoms:
        try:
            gc = g.intersection(clip)
        except Exception:
            gc = g
        if gc.is_empty:
            continue
        px = geom_to_pixels(gc, src_proj, proj)
        if not px.is_empty:
            shapes.append((px, ICEBERG))

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
        month_range(rec["year"]),
        source_id=rec["source_id"],
        classes_present=present,
    )
    return "iceberg" if ICEBERG in present else "background"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=64)
    args = ap.parse_args()

    io.check_disk()

    raw = io.raw_dir(SLUG)
    raw.mkdir(parents=True, exist_ok=True)
    with (raw / "SOURCE.txt").open("w") as f:
        f.write(
            "A Six-year circum-Antarctic icebergs dataset (2018-2023). "
            "Zenodo record 17165466 (ESSD, CC-BY-4.0). "
            "https://doi.org/10.5281/zenodo.17165466\n"
            "File: 'Iceberg vector outline.zip' (346 MB) -> extract/Iceberg vector "
            "outline/{2018..2023}10_distribution.gpkg (six October iceberg-outline "
            "GeoPackages, EPSG:3031; Sentinel-1 SAR RF detection + manual correction).\n"
        )

    # ---- Phase A: attribute-only scan + pooled geographic stratified selection.
    per_year: dict[int, dict[str, np.ndarray]] = {}
    for yr in YEARS:
        per_year[yr] = load_centroids(yr)
        print(f"loaded {yr}: {len(per_year[yr]['lon'])} icebergs")

    all_lon = np.concatenate([per_year[y]["lon"] for y in YEARS])
    all_lat = np.concatenate([per_year[y]["lat"] for y in YEARS])
    all_x = np.concatenate([per_year[y]["x3031"] for y in YEARS])
    all_y = np.concatenate([per_year[y]["y3031"] for y in YEARS])
    year_of = np.concatenate(
        [np.full(len(per_year[y]["lon"]), y, dtype="int32") for y in YEARS]
    )
    local_of = np.concatenate(
        [np.arange(len(per_year[y]["lon"]), dtype="int64") for y in YEARS]
    )
    print(f"total icebergs (6 yr): {len(all_lon)}")

    io.check_disk()

    sel = stratified_indices(all_lon, all_lat, TOTAL_TARGET, seed=1)
    print(f"selected {len(sel)} tiles")

    records: list[dict[str, Any]] = []
    for sid, gi in enumerate(sel):
        yr = int(year_of[gi])
        records.append(
            {
                "sample_id": f"{sid:06d}",
                "lon": float(all_lon[gi]),
                "lat": float(all_lat[gi]),
                "x3031": float(all_x[gi]),
                "y3031": float(all_y[gi]),
                "year": yr,
                "source_id": f"{yr}10:{int(local_of[gi])}",
            }
        )

    year_hist = Counter(r["year"] for r in records)
    print("records per year:", dict(sorted(year_hist.items())))
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
            "source": "Zenodo / ESSD (Six-year circum-Antarctic icebergs)",
            "license": "CC-BY-4.0",
            "provenance": {
                "url": "https://doi.org/10.5281/zenodo.17165466",
                "have_locally": False,
                "annotation_method": "Sentinel-1 SAR semi-automated random-forest "
                "detection + manual correction",
            },
            "sensors_relevant": ["sentinel1", "sentinel2"],
            "classes": [
                {"id": i, "name": name, "description": desc}
                for i, (name, desc) in enumerate(CLASSES)
            ],
            "nodata_value": io.CLASS_NODATA,
            "num_samples": n_written,
            "tile_counts": {
                "iceberg_tiles": counts.get("iceberg", 0),
                "background_only_tiles": counts.get("background", 0),
            },
            "notes": (
                "Binary iceberg vs ocean/sea-ice segmentation from the six-year "
                "circum-Antarctic Sentinel-1 iceberg outline product (~244k polygons, "
                "Oct 2018-2023). 64x64 uint8 tiles in local UTM/UPS at 10 m; classes: "
                "0 background (ocean/sea ice), 1 iceberg (255 nodata, unused). Bounded "
                "geographically-stratified sampling (round-robin over 1-degree lon/lat "
                "cells, pooled across all 6 years), capped at 25,000 tiles. Each tile is "
                "centered on a sampled iceberg centroid; all iceberg polygons "
                "intersecting the tile are rasterized to class 1 (all_touched=True), the "
                "rest is background. Source polygons in EPSG:3031, reprojected per-tile "
                "to local UTM/UPS. Time range = 1-month window [Oct 1, Nov 1) of each "
                "tile's source year (the product is an October snapshot; a full year "
                "would be ill-posed because bergs drift). CAVEATS: (1) icebergs drift "
                "km/day so even a 1-month window carries positional label noise; "
                "(2) giant tabular bergs (area up to ~5700 km2) larger than the 640 m "
                "tile yield all-iceberg tiles with no background; (3) 'background' means "
                "'no mapped iceberg' -- sub-detection-threshold bergs may fall in "
                "background. Per spec 5, no synthetic background-only negative tiles are "
                "fabricated (positive-only dataset)."
            ),
        },
    )
    print("tile counts:", dict(counts))
    print("total tif on disk:", n_written)
    print("done")


if __name__ == "__main__":
    multiprocessing.set_start_method("forkserver", force=True)
    main()
