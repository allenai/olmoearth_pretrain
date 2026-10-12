"""Mesopotamian Archaeological Sites (tells) -> open-set-segmentation polygon labels.

Source: the FloodPlains Web GIS (University of Bologna / OrientLab,
https://floodplains.orientlab.net), a compilation of all published archaeological
surveys of the southern/central Mesopotamian floodplain (~66,000 km2). The core ground-
truth layer ``vw_site_survey_poly`` contains **4,934 georeferenced polygons** delineating
the contours of known archaeological mound sites ("tells"), each confirmed by ground
survey / surface-scatter study across 16 published survey projects. The polygons are
published (CC-BY) alongside the "human-AI collaboration for archaeological site detection"
paper (Sci. Rep. 2023) and mirrored as a shapefile in that paper's code repository
(github.com/mister-magpie/tell_segmentation, ``shapefiles/site_shape/``). The live
GeoServer WFS is credential-gated (blanket HTTP 401 on GetFeature), so we take the
published shapefile mirror instead.

Suitability at 10 m: tells are man-made occupation mounds, not sub-pixel points. Their
mapped footprints are substantial -- median footprint ~136 m across (~19 px at 10 m),
90th pct ~500 m; 98.8% span >=30 m and 98% cover >=9 pixels at 10 m. The persistent
topographic/soil/vegetation signature of a mound is observable in Sentinel-2/Landsat, so
we rasterize each polygon into a <=64x64 UTM 10 m tile.

Task: per-pixel **classification**, a single presence class:
  0  archaeological mound/tell   (rasterized polygon footprint)
This is a **presence-only** dataset (no background/negative class). Following spec 5,
outside-polygon pixels are left as nodata/ignore (255); the pretraining-assembly step
supplies negatives from other datasets. Do NOT fabricate synthetic background here.

Tile: centered on each polygon, sized to the polygon's pixel footprint, capped at 64x64.
Polygons larger than 640 m (326 of 4,934, e.g. the great tell-cities Uruk, Lagash, Girsu,
Adab) overflow a 64 px tile and are center-cropped to their interior -- still a valid
positive mask. ``all_touched=True`` so even the few sub-pixel polygons mark >=1 pixel.

Time range: the sites are persistent/static -> a fixed representative 1-year Sentinel-era
window (2020). ``source_id`` carries the site ``entry_id`` (e.g. QD001).

Sampling: single class, spec cap 1000 locations/class -> up to 1000 tiles drawn (seeded)
from the 4,934 polygons.

Run (idempotent; skips already-written {sample_id}.tif):
  python3 -m olmoearth_pretrain.open_set_segmentation_data.datasets.mesopotamian_archaeological_sites_tells
"""

import argparse
import multiprocessing
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import shapely
import tqdm
from rasterio.crs import CRS
from rslearn.utils.geometry import Projection
from rslearn.utils.mp import star_imap_unordered

from olmoearth_pretrain.open_set_segmentation_data import download, io, manifest
from olmoearth_pretrain.open_set_segmentation_data.rasterize import (
    geom_to_pixels,
    rasterize_shapes,
)
from olmoearth_pretrain.open_set_segmentation_data.sampling import balance_by_class

SLUG = "mesopotamian_archaeological_sites_tells"
NAME = "Mesopotamian Archaeological Sites (tells)"

# Published shapefile mirror of the FloodPlains ground-truth site polygons.
SHAPEFILES_ZIP_URL = "https://raw.githubusercontent.com/mister-magpie/tell_segmentation/main/shapefiles.zip"
SHP_RELPATH = "shapefiles/site_shape/vw_site_survey_poly.shp"

CLASS_ID = 0
CLASS_NAME = "archaeological mound/tell"
CLASS_DESC = (
    "Contour of a known archaeological occupation mound ('tell') in the southern/central "
    "Mesopotamian floodplain, compiled from published ground surveys and confirmed by "
    "surface-scatter study (FloodPlains Project, Univ. Bologna). Footprints range from "
    "sub-hectare mounds to multi-km tell-cities (Uruk, Lagash, Girsu, Adab)."
)
PER_CLASS = 1000
STATIC_YEAR = 2020  # representative Sentinel-era year for these persistent sites
MAX_TILE = io.MAX_TILE  # 64

_WGS84_SRC = Projection(CRS.from_epsg(4326), 1, 1)


def ensure_data() -> str:
    """Download + unzip the shapefile mirror; return the local path to the site .shp."""
    raw = io.raw_dir(SLUG)
    raw.mkdir(parents=True, exist_ok=True)
    zip_path = raw / "shapefiles.zip"
    download.download_http(SHAPEFILES_ZIP_URL, zip_path)
    unzip_root = Path(raw.path) / "unzip"
    shp = unzip_root / SHP_RELPATH
    if not shp.exists():
        unzip_root.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(Path(zip_path.path)) as zf:
            zf.extractall(unzip_root)
    if not shp.exists():
        raise RuntimeError(f"expected {SHP_RELPATH} not found after unzip")
    with (raw / "SOURCE.txt").open("w") as f:
        f.write(
            "Mesopotamian Archaeological Sites (tells).\n"
            "FloodPlains Web GIS, Univ. of Bologna / OrientLab "
            "(https://floodplains.orientlab.net), layer vw_site_survey_poly "
            "(4,934 ground-truth mound-site polygons). License: CC-BY (paper).\n"
            "Live GeoServer WFS is credential-gated (HTTP 401 on GetFeature); used the "
            "published shapefile mirror from the Sci. Rep. 2023 paper repo:\n"
            f"  {SHAPEFILES_ZIP_URL}\n"
            "  (github.com/mister-magpie/tell_segmentation, resolved via bit.ly/NSR_floodplains)\n"
            "Papers: Sci. Rep. 2023 (s41598-023-36015-5); PLOS One 2025 (pone.0330419).\n"
        )
    return str(shp)


def load_records(shp_path: str) -> list[dict[str, Any]]:
    """Read all site polygons -> list of records (geom_wkb, lon/lat, source_id)."""
    import geopandas as gpd

    gdf = gpd.read_file(shp_path)
    if gdf.crs is None or gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs(4326)
    records: list[dict[str, Any]] = []
    for _, row in gdf.iterrows():
        geom = row.geometry
        if geom is None or geom.is_empty:
            continue
        if not geom.is_valid:
            geom = geom.buffer(0)
            if geom.is_empty:
                continue
        rp = geom.representative_point()  # guaranteed inside; used to pick UTM zone
        eid = row.get("entry_id")
        records.append(
            {
                "geom_wkb": shapely.to_wkb(geom),
                "lon": float(rp.x),
                "lat": float(rp.y),
                "label": CLASS_ID,
                "source_id": str(eid) if eid is not None else None,
            }
        )
    return records


def _write_tile(rec: dict[str, Any]) -> tuple[str, str]:
    sample_id = rec["sample_id"]
    tif = io.locations_dir(SLUG) / f"{sample_id}.tif"
    if tif.exists():
        return sample_id, "skip"
    try:
        geom = shapely.from_wkb(rec["geom_wkb"])  # WGS84 lon/lat
        proj = io.utm_projection_for_lonlat(rec["lon"], rec["lat"])
        pix = geom_to_pixels(geom, _WGS84_SRC, proj)
        minx, miny, maxx, maxy = pix.bounds
        cx = int(round((minx + maxx) / 2))
        cy = int(round((miny + maxy) / 2))
        w = min(MAX_TILE, max(1, int(np.ceil(maxx - minx))))
        h = min(MAX_TILE, max(1, int(np.ceil(maxy - miny))))
        bounds = io.centered_bounds(cx, cy, w, h)
        arr = rasterize_shapes(
            [(pix, CLASS_ID)],
            bounds,
            fill=io.CLASS_NODATA,
            dtype="uint8",
            all_touched=True,
        )
        if not (arr != io.CLASS_NODATA).any():
            # sub-pixel/degenerate polygon: stamp the center pixel as positive.
            arr[0, arr.shape[1] // 2, arr.shape[2] // 2] = CLASS_ID
        io.write_label_geotiff(
            SLUG, sample_id, arr, proj, bounds, nodata=io.CLASS_NODATA
        )
        io.write_sample_json(
            SLUG,
            sample_id,
            proj,
            bounds,
            io.year_range(STATIC_YEAR),
            source_id=rec["source_id"],
            classes_present=[CLASS_ID],
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

    shp_path = ensure_data()
    records = load_records(shp_path)
    print(f"loaded {len(records)} site polygons")

    selected = balance_by_class(records, "label", per_class=PER_CLASS)
    for i, r in enumerate(selected):
        r["sample_id"] = f"{i:06d}"
    print(f"selected {len(selected)} tiles (<= {PER_CLASS}/class)")

    io.check_disk()
    stats: Counter = Counter()
    with multiprocessing.Pool(args.workers) as pool:
        for _sid, status in tqdm.tqdm(
            star_imap_unordered(pool, _write_tile, [{"rec": r} for r in selected]),
            total=len(selected),
        ):
            stats[status] += 1
    print("write stats:", dict(stats))

    n_ok = stats["ok"] + stats["skip"]
    io.write_dataset_metadata(
        SLUG,
        {
            "dataset": SLUG,
            "name": NAME,
            "task_type": "classification",
            "source": "FloodPlains Project (Univ. Bologna / OrientLab)",
            "license": "CC-BY",
            "provenance": {
                "url": "https://floodplains.orientlab.net",
                "shapefile_mirror": SHAPEFILES_ZIP_URL,
                "layer": "vw_site_survey_poly",
                "have_locally": False,
                "annotation_method": (
                    "manual compilation of 16 published ground surveys; site contours "
                    "digitized over satellite imagery, confirmed by surface scatter"
                ),
            },
            "sensors_relevant": ["sentinel2", "sentinel1", "landsat"],
            "classes": [
                {"id": CLASS_ID, "name": CLASS_NAME, "description": CLASS_DESC}
            ],
            "nodata_value": io.CLASS_NODATA,
            "num_samples": n_ok,
            "notes": (
                "Presence-only single-class polygon dataset (archaeological tells). "
                "Rasterized polygon footprints into <=64x64 UTM 10 m tiles; outside-polygon "
                "= nodata (255), NO fabricated background (assembly supplies negatives). "
                "Tile sized to each polygon footprint, capped at 64; 326 polygons >640 m "
                "(great tell-cities) are center-cropped. Static persistent sites -> fixed "
                f"{STATIC_YEAR} 1-year window. Full source has 4,934 polygons; sampled "
                f"<= {PER_CLASS} per spec per-class cap."
            ),
        },
    )
    manifest.write_registry_entry(
        SLUG, "completed", task_type="classification", num_samples=n_ok
    )
    print("done:", dict(stats))


if __name__ == "__main__":
    main()
