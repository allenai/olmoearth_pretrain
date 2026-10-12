"""Process Global Renewables Watch (GRW) solar PV polygons into label patches.

Source: Microsoft / Planet / TNC "Global Renewables Watch", a quarterly global inventory of
solar PV installations (polygons) and wind turbines (points) detected from PlanetScope
imagery with deep learning + human QC, with per-feature construction dates. v1.0 2024-Q2
GeoPackages from the project's GitHub releases:

  https://github.com/microsoft/global-renewables-watch/releases/download/v1.0/solar_all_2024q2_v1.gpkg
  https://github.com/microsoft/global-renewables-watch/releases/download/v1.0/wind_all_2024q2_v1.gpkg

This script handles ONLY the **solar PV polygons** (real rasterized footprints). The wind
turbines were previously encoded here with the fabricated detection encoding (positive
square + nodata buffer + synthetic background/negative tiles); they are now a separate
presence-only point dataset ``global_renewables_watch_points`` (spec 4/5: isolated point
inventories with fabricated negatives become presence-only points, negatives supplied by the
assembly step).

Class scheme (dense polygon segmentation):
  0 = background   (real non-solar pixels inside the tile, outside the PV footprint)
  1 = solar_pv     (PV installation polygon, rasterized into a <=64x64 UTM tile)
  255 = nodata / ignore

Per-feature ``construction_year`` sets a ~1-year time range. Bounded to <=1000 tiles,
stratified across construction years for temporal diversity.

Run (idempotent):
  python3 -m olmoearth_pretrain.open_set_segmentation_data.datasets.global_renewables_watch
"""

import argparse
import multiprocessing
from collections import Counter
from typing import Any

import fiona
import numpy as np
import shapely
import tqdm
from rasterio.crs import CRS
from rslearn.utils.geometry import Projection
from rslearn.utils.mp import star_imap_unordered

from olmoearth_pretrain.open_set_segmentation_data import io, manifest
from olmoearth_pretrain.open_set_segmentation_data.rasterize import (
    geom_to_pixels,
    rasterize_shapes,
)
from olmoearth_pretrain.open_set_segmentation_data.sampling import balance_by_class

SLUG = "global_renewables_watch"
NAME = "Global Renewables Watch (solar PV)"
RELEASE = "https://github.com/microsoft/global-renewables-watch/releases/download/v1.0"
SOLAR_FILE = "solar_all_2024q2_v1.gpkg"
WIND_FILE = "wind_all_2024q2_v1.gpkg"

CID_BACKGROUND = 0
CID_SOLAR = 1
CLASSES = [
    {
        "id": CID_BACKGROUND,
        "name": "background",
        "description": "Non-solar land inside the tile, outside the detected PV footprint.",
    },
    {
        "id": CID_SOLAR,
        "name": "solar_pv",
        "description": "Ground-mounted / large solar photovoltaic installation footprint "
        "(polygon rasterized at 10 m), from GRW PlanetScope deep-learning detections with "
        "human QC.",
    },
]

PER_CLASS = 1000
YEARS = list(range(2017, 2025))  # construction_year buckets present in the release.
PER_YEAR = PER_CLASS // len(YEARS)  # 125 -> 1000 total.
MAX_SOLAR_TILE = io.MAX_TILE  # 64

SRC_PROJ = Projection(CRS.from_epsg(3857), 1, 1)  # source geometries are EPSG:3857 metres.
_TO_WGS84 = None  # lazily-built pyproj transformer (per process).


def _lonlat(x: float, y: float) -> tuple[float, float]:
    """EPSG:3857 (x, y) metres -> (lon, lat) degrees."""
    global _TO_WGS84
    if _TO_WGS84 is None:
        from pyproj import Transformer

        _TO_WGS84 = Transformer.from_crs(3857, 4326, always_xy=True)
    lon, lat = _TO_WGS84.transform(x, y)
    return lon, lat


def read_solar() -> list[dict[str, Any]]:
    """Read solar PV polygons into records (year, centroid lon/lat, geometry WKB)."""
    path = io.raw_dir(SLUG) / SOLAR_FILE
    recs: list[dict[str, Any]] = []
    with fiona.open(path.path) as src:
        for i, feat in enumerate(src):
            props = feat["properties"]
            year = props.get("construction_year")
            if year is None:
                continue
            geom = shapely.geometry.shape(feat["geometry"])
            if geom.is_empty:
                continue
            c = geom.centroid
            lon, lat = _lonlat(c.x, c.y)
            recs.append(
                {
                    "year": int(year),
                    "lon": lon,
                    "lat": lat,
                    "geom_wkb": shapely.to_wkb(geom),
                    "source_id": f"solar/{i}",
                }
            )
    return recs


def _write_solar(rec: dict[str, Any]) -> str | None:
    sample_id = rec["sample_id"]
    tif = io.locations_dir(SLUG) / f"{sample_id}.tif"
    if tif.exists():
        return "skip"
    proj = io.utm_projection_for_lonlat(rec["lon"], rec["lat"])
    geom = shapely.from_wkb(rec["geom_wkb"])
    pix = geom_to_pixels(geom, SRC_PROJ, proj)
    minx, miny, maxx, maxy = pix.bounds
    cx = int(round((minx + maxx) / 2))
    cy = int(round((miny + maxy) / 2))
    w = min(MAX_SOLAR_TILE, max(1, int(np.ceil(maxx - minx))))
    h = min(MAX_SOLAR_TILE, max(1, int(np.ceil(maxy - miny))))
    bounds = io.centered_bounds(cx, cy, w, h)
    arr = rasterize_shapes(
        [(pix, CID_SOLAR)], bounds, fill=CID_BACKGROUND, dtype="uint8", all_touched=True
    )
    io.write_label_geotiff(SLUG, sample_id, arr, proj, bounds, nodata=io.CLASS_NODATA)
    io.write_sample_json(
        SLUG,
        sample_id,
        proj,
        bounds,
        io.year_range(rec["year"]),
        source_id=rec["source_id"],
        classes_present=sorted(set(np.unique(arr).tolist()) - {io.CLASS_NODATA}),
    )
    return "solar"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=64)
    args = parser.parse_args()

    io.check_disk()
    manifest.write_registry_entry(SLUG, "in_progress")

    raw = io.raw_dir(SLUG)
    raw.mkdir(parents=True, exist_ok=True)
    with (raw / "SOURCE.txt").open("w") as f:
        f.write(
            "Global Renewables Watch v1.0 (2024 Q2), Microsoft/Planet/TNC.\n"
            f"{RELEASE}/{SOLAR_FILE}\n{RELEASE}/{WIND_FILE}\n"
            "(wind turbines are a separate dataset: global_renewables_watch_points)\n"
        )

    print("reading solar polygons ...")
    solar = read_solar()
    print(f"  {len(solar)} solar PV polygons")

    io.check_disk()
    solar_sel = balance_by_class(solar, "year", per_class=PER_YEAR, seed=42)[:PER_CLASS]
    print(f"selected {len(solar_sel)} solar polygons")
    for i, r in enumerate(solar_sel):
        r["sample_id"] = f"{i:06d}"

    results: Counter = Counter()
    with multiprocessing.Pool(args.workers) as p:
        for res in tqdm.tqdm(
            star_imap_unordered(p, _write_solar, [dict(rec=r) for r in solar_sel]),
            total=len(solar_sel),
        ):
            results[res] += 1
    print("write results:", dict(results))

    io.check_disk()
    io.write_dataset_metadata(
        SLUG,
        {
            "dataset": SLUG,
            "name": NAME,
            "task_type": "classification",
            "source": "GitHub (Microsoft/Planet/TNC)",
            "license": "MIT",
            "provenance": {
                "url": "https://github.com/microsoft/global-renewables-watch",
                "have_locally": False,
                "annotation_method": "deep learning (PlanetScope) + human QC",
                "release": "v1.0 (2024 Q2)",
            },
            "sensors_relevant": ["sentinel2", "sentinel1", "landsat"],
            "classes": CLASSES,
            "nodata_value": io.CLASS_NODATA,
            "num_samples": len(solar_sel),
            "class_counts": {"solar_pv": len(solar_sel)},
            "notes": (
                "Solar PV installation footprints: polygons rasterized into variable "
                "<=64x64 UTM tiles (class solar_pv=1, background=0). 1-year time_range on "
                "each feature's construction_year (2017-2024). Wind turbines from the same "
                "product are a separate presence-only point dataset "
                "(global_renewables_watch_points). Derived product (DL+QC)."
            ),
        },
    )
    manifest.write_registry_entry(
        SLUG, "completed", task_type="classification", num_samples=len(solar_sel)
    )
    print("done")


if __name__ == "__main__":
    multiprocessing.set_start_method("forkserver", force=True)
    main()
