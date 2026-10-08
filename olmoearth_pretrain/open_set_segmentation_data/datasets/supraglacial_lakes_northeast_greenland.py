"""Process "Supraglacial Lakes, Northeast Greenland" polygons into label patches.

Source: Lutz, Bahrami, Braun (2024), "Supraglacial lake outlines over Northeast
Greenland from 2016 to 2022 using deep learning methods based on Sentinel-2 imagery",
PANGAEA (https://doi.org/10.1594/PANGAEA.973251). Supraglacial-lake polygon outlines over
the 79N Glacier and Zachariae Isstrom during the April-September melt seasons of 2016-2022,
segmented with a U-Net from Sentinel-2 (native 10 m). License: CC-BY-4.0 -> usable.

Data layout: seven annual zips (yyyy.zip, downloaded individually from PANGAEA -- no
account needed for single files). Each zip holds one shapefile PER Sentinel-2 acquisition
date, named ``yyyy-mm-dd_pred_vector.shp`` (437 dated scenes total). Each shapefile is a
FeatureCollection of lake Polygons in EPSG:3413 (NSIDC Polar Stereographic North) with
attributes ``raster_val`` (== 1.0 for every lake) and ``id``. 233,197 polygons total
(215,834 with area >= 100 m^2 = 1 S2 pixel).

Task: classification, positive-only foreground (spec 4 polygons + spec 5 positive-only).
There is a single foreground class; non-lake is NOT a class:
  0   = supraglacial_lake  (a mapped meltwater lake outline)
  255 = nodata / ignore    (everything else -- non-lake ice/rock/shadow). We do NOT
        fabricate negatives (spec 5); the assembly step supplies negatives from other
        datasets.

Encoding: for each SELECTED lake we cut ONE UTM tile (local UTM zone from the lake's
lon/lat, 10 m/pixel) centered on the lake's representative point, sized to the lake's
footprint + 8 px margin and capped at 64x64. ALL same-date polygons that fall inside the
tile (not just the selected one) are rasterized as class 0 (all_touched=True so tiny lakes
survive at 10 m); the rest is nodata. Reprojection EPSG:3413 -> UTM is done in pixel space
via rasterize.geom_to_pixels.

Sampling: single class, so up to 1000 tiles (spec 5). To spread coverage across space and
time we round-robin across the 437 dated scenes (a fresh random lake per scene each pass)
rather than sampling the pool uniformly (which 2019, ~27% of polygons, would dominate).
Only lakes with area >= 100 m^2 (>= 1 pixel) are eligible as tile centers; smaller slivers
(often model artifacts) are still drawn as class 0 when they fall inside a tile.

Time: each shapefile is a single dated S2 acquisition (2016-2022; all in the Sentinel
era). Supraglacial lakes are seasonal/transient meltwater features. We assign a 1-year
window = the calendar year of acquisition (which contains that year's April-September melt
season), per the orchestrator directive and spec 5's seasonal-label rule; the exact
acquisition date is preserved in ``source_id``. change_time is null (this is a
presence/state label, not a dated change event). CAVEAT: because lakes drain within
weeks, a given lake outline is only valid around its acquisition date; pretraining's
~360-day input window will include that year's melt season, but imagery elsewhere in the
window may not show the lake.

Run (idempotent -- skips already-written {id}.tif):
  python3 -m olmoearth_pretrain.open_set_segmentation_data.datasets.supraglacial_lakes_northeast_greenland
"""

import argparse
import multiprocessing
import random
import re
import zipfile
from collections import Counter, defaultdict
from typing import Any

import fiona
import numpy as np
import shapely
import tqdm
from rasterio.crs import CRS
from rslearn.const import WGS84_PROJECTION
from rslearn.utils.geometry import Projection, STGeometry
from rslearn.utils.mp import star_imap_unordered

from olmoearth_pretrain.open_set_segmentation_data import download, io, manifest
from olmoearth_pretrain.open_set_segmentation_data.rasterize import (
    geom_to_pixels,
    rasterize_shapes,
)

SLUG = "supraglacial_lakes_northeast_greenland"
NAME = "Supraglacial Lakes, Northeast Greenland"
URL = "https://doi.org/10.1594/PANGAEA.973251"
DATASET_ID = "973251"
YEARS = list(range(2016, 2023))

# Geometries are EPSG:3413 metres; treat metres as "pixels" at resolution 1 so
# geom_to_pixels reprojects them straight to the target UTM pixel grid.
SRC_PROJ = Projection(CRS.from_epsg(3413), 1, 1)

CID_LAKE = 0
MIN_AREA_M2 = 100.0  # >= 1 S2 pixel; eligible as a tile center.
PAD = 8  # px of margin around the lake footprint (before the 64 cap).
NEIGHBOR_RADIUS_M = 360.0  # 3413-metre halo for finding same-date neighbours in a tile.
PER_CLASS = 1000
DATE_RE = re.compile(r"(\d{4})-(\d{2})-(\d{2})_pred_vector")

CLASSES = [
    {
        "id": CID_LAKE,
        "name": "supraglacial_lake",
        "description": (
            "Supraglacial (surface) meltwater lake on the 79N Glacier / Zachariae "
            "Isstrom, Northeast Greenland, segmented from a Sentinel-2 acquisition with a "
            "U-Net (Lutz et al. 2023). Seasonal features that form in surface depressions "
            "during the April-September melt season. Outlines are direct model outputs "
            "(not manually corrected) and may include false positives from topographic or "
            "cloud shadows and slushy blue ice."
        ),
    }
]


def download_all() -> None:
    """Fetch the seven annual zips from PANGAEA (single-file endpoint, no account)."""
    raw = io.raw_dir(SLUG)
    raw.mkdir(parents=True, exist_ok=True)
    for y in YEARS:
        download.download_http(
            f"https://download.pangaea.de/dataset/{DATASET_ID}/files/{y}.zip",
            raw / f"{y}.zip",
        )
    with (raw / "SOURCE.txt").open("w") as fp:
        fp.write(
            "Supraglacial lake outlines over Northeast Greenland 2016-2022 (Lutz, "
            "Bahrami, Braun 2024).\n"
            f"{URL}\nLicense: CC-BY-4.0.\n"
            "Seven annual zips (yyyy.zip); each holds one shapefile per S2 acquisition "
            "date (yyyy-mm-dd_pred_vector.shp), EPSG:3413, polygons of supraglacial lakes.\n"
        )


def _read_one_shapefile(
    zip_path: str, member: str, year: int, date: str
) -> list[dict[str, Any]]:
    """Read all lake polygons from one dated shapefile inside a zip."""
    recs: list[dict[str, Any]] = []
    with fiona.open(f"zip://{zip_path}!{member}") as src:
        for feat in src:
            geom = shapely.geometry.shape(feat["geometry"])
            if geom.is_empty or geom.area <= 0:
                continue
            geom = shapely.force_2d(geom)
            c = geom.centroid
            recs.append(
                {
                    "year": year,
                    "date": date,
                    "poly_id": str(feat["properties"].get("id")),
                    "area": float(geom.area),
                    "cx": float(c.x),
                    "cy": float(c.y),
                    "geom_wkb": shapely.to_wkb(geom),
                }
            )
    return recs


def read_all_polygons(workers: int) -> list[dict[str, Any]]:
    """Read every polygon from every dated shapefile (parallel over shapefiles)."""
    raw = io.raw_dir(SLUG)
    tasks: list[dict[str, Any]] = []
    for y in YEARS:
        zp = str(raw / f"{y}.zip")
        with zipfile.ZipFile(zp) as zf:
            for member in zf.namelist():
                if not member.endswith(".shp"):
                    continue
                m = DATE_RE.search(member)
                if not m:
                    continue
                tasks.append(
                    dict(
                        zip_path=zp, member=member, year=y, date=f"{m[1]}-{m[2]}-{m[3]}"
                    )
                )
    recs: list[dict[str, Any]] = []
    with multiprocessing.Pool(workers) as p:
        for out in tqdm.tqdm(
            star_imap_unordered(p, _read_one_shapefile, tasks),
            total=len(tasks),
            desc="read shapefiles",
        ):
            recs.extend(out)
    return recs


def select_round_robin(
    recs: list[dict[str, Any]], n: int, seed: int = 42
) -> list[dict[str, Any]]:
    """Spread selection across dated scenes: one fresh random lake per scene per pass."""
    rng = random.Random(seed)
    by_scene: dict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)
    for r in recs:
        if r["area"] >= MIN_AREA_M2:
            by_scene[(r["year"], r["date"])].append(r)
    scenes = sorted(by_scene)
    for s in scenes:
        rng.shuffle(by_scene[s])
    order = list(scenes)
    rng.shuffle(order)
    selected: list[dict[str, Any]] = []
    idx = {s: 0 for s in scenes}
    progressed = True
    while len(selected) < n and progressed:
        progressed = False
        for s in order:
            if idx[s] < len(by_scene[s]):
                selected.append(by_scene[s][idx[s]])
                idx[s] += 1
                progressed = True
                if len(selected) >= n:
                    break
    return selected


def _write_tile(rec: dict[str, Any]) -> str:
    sample_id = rec["sample_id"]
    tif = io.locations_dir(SLUG) / f"{sample_id}.tif"
    if tif.exists():
        return "skip"
    sel = shapely.from_wkb(rec["geom_wkb"])
    # UTM zone from the lake's lon/lat.
    rp3413 = sel.representative_point()
    ll = STGeometry(SRC_PROJ, rp3413, None).to_projection(WGS84_PROJECTION).shp
    proj = io.utm_projection_for_lonlat(ll.x, ll.y)
    # Selected lake in UTM pixel space -> footprint + centre.
    sel_pix = geom_to_pixels(sel, SRC_PROJ, proj)
    minx, miny, maxx, maxy = sel_pix.bounds
    rp = sel_pix.representative_point()
    cx, cy = int(round(rp.x)), int(round(rp.y))
    w = min(io.MAX_TILE, max(1, int(np.ceil(maxx - minx)) + PAD))
    h = min(io.MAX_TILE, max(1, int(np.ceil(maxy - miny)) + PAD))
    bounds = io.centered_bounds(cx, cy, w, h)
    # Rasterize selected + all same-date neighbours falling in the tile.
    shapes = [(sel_pix, CID_LAKE)]
    for wkb in rec["neighbor_wkbs"]:
        gp = geom_to_pixels(shapely.from_wkb(wkb), SRC_PROJ, proj)
        gminx, gminy, gmaxx, gmaxy = gp.bounds
        if (
            gmaxx < bounds[0]
            or gminx > bounds[2]
            or gmaxy < bounds[1]
            or gminy > bounds[3]
        ):
            continue
        shapes.append((gp, CID_LAKE))
    arr = rasterize_shapes(
        shapes, bounds, fill=io.CLASS_NODATA, dtype="uint8", all_touched=True
    )
    io.write_label_geotiff(SLUG, sample_id, arr, proj, bounds, nodata=io.CLASS_NODATA)
    io.write_sample_json(
        SLUG,
        sample_id,
        proj,
        bounds,
        io.year_range(rec["year"]),
        source_id=f"{rec['year']}/{rec['date']}_pred_vector/{rec['poly_id']}",
        classes_present=[CID_LAKE],
    )
    return "ok"


def attach_neighbors(
    selected: list[dict[str, Any]], all_recs: list[dict[str, Any]]
) -> None:
    """For each selected lake, list same-date polygons whose 3413 bbox is near the tile."""
    by_scene: dict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)
    for r in all_recs:
        by_scene[(r["year"], r["date"])].append(r)
    for r in selected:
        cx, cy = r["cx"], r["cy"]
        rr = NEIGHBOR_RADIUS_M
        nbrs = []
        for o in by_scene[(r["year"], r["date"])]:
            if o is r:
                continue
            if abs(o["cx"] - cx) <= rr and abs(o["cy"] - cy) <= rr:
                nbrs.append(o["geom_wkb"])
        r["neighbor_wkbs"] = nbrs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=64)
    args = parser.parse_args()

    io.check_disk()
    manifest.write_registry_entry(SLUG, "in_progress")

    download_all()

    print("reading polygons ...", flush=True)
    recs = read_all_polygons(args.workers)
    print(
        f"  {len(recs)} polygons across "
        f"{len({(r['year'], r['date']) for r in recs})} dated scenes",
        flush=True,
    )
    year_counts = Counter(r["year"] for r in recs)
    print("polys/year:", dict(sorted(year_counts.items())), flush=True)

    selected = select_round_robin(recs, PER_CLASS)
    for i, r in enumerate(selected):
        r["sample_id"] = f"{i:06d}"
    sel_year_counts = Counter(r["year"] for r in selected)
    print(
        f"selected {len(selected)} tiles; per-year:",
        dict(sorted(sel_year_counts.items())),
        flush=True,
    )

    attach_neighbors(selected, recs)

    io.check_disk()
    results: Counter = Counter()
    with multiprocessing.Pool(args.workers) as p:
        for res in tqdm.tqdm(
            star_imap_unordered(p, _write_tile, [dict(rec=r) for r in selected]),
            total=len(selected),
            desc="write tiles",
        ):
            results[res] += 1
    print("write results:", dict(results), flush=True)

    io.check_disk()
    io.write_dataset_metadata(
        SLUG,
        {
            "dataset": SLUG,
            "name": NAME,
            "task_type": "classification",
            "source": "PANGAEA",
            "license": "CC-BY-4.0",
            "provenance": {
                "url": URL,
                "pangaea_dataset": DATASET_ID,
                "have_locally": False,
                "annotation_method": "derived (U-Net on Sentinel-2), Lutz et al. 2023",
            },
            "sensors_relevant": ["sentinel2", "sentinel1", "landsat"],
            "classes": CLASSES,
            "nodata_value": io.CLASS_NODATA,
            "num_samples": len(selected),
            "source_polygon_count": len(recs),
            "source_scene_count": len({(r["year"], r["date"]) for r in recs}),
            "selected_per_year": dict(sorted(sel_year_counts.items())),
            "tile_size_max": io.MAX_TILE,
            "region": "Northeast Greenland (79N Glacier and Zachariae Isstrom)",
            "notes": (
                "Supraglacial-lake polygons, PANGAEA 973251 (Lutz et al. 2024), U-Net on "
                "Sentinel-2, 2016-2022 April-September melt seasons. Single foreground "
                "class supraglacial_lake=0; non-lake=255 nodata (positive-only, no "
                "fabricated negatives per spec 5). Each selected lake -> ONE UTM tile "
                "@10 m centered on it, sized to footprint+8px capped 64x64, with all "
                "same-date polygons in the tile rasterized as lake (all_touched=True). "
                "Source geometries EPSG:3413 reprojected to local UTM. Up to 1000 tiles, "
                "round-robin across the 437 dated scenes for spatial/temporal spread; only "
                "lakes >=100 m^2 are tile centers. Time range = calendar year of "
                "acquisition (contains the melt season); exact date in source_id; "
                "change_time null. CAVEAT: lakes are transient (drain within weeks), so an "
                "outline is only strictly valid near its acquisition date; outlines are raw "
                "model output and may contain shadow/slush false positives."
            ),
        },
    )
    print(f"done: {len(selected)} tiles", flush=True)


if __name__ == "__main__":
    multiprocessing.set_start_method("forkserver", force=True)
    main()
