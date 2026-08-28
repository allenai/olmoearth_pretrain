"""Process "ChinaPV" solar photovoltaic polygons into label patches.

Source: ChinaPV, "the spatial distribution of solar photovoltaic installation dataset
across China in 2015 and 2020" (Sci Data / Zenodo record 14292571,
https://zenodo.org/records/14292571). PV installations across China were mapped from
Landsat-8 imagery (30 m) with manual adjustment/refinement, vectorized as polygons for
two epochs (2015 and 2020). License: CC-BY-4.0 -> usable.

Files (shapefiles, EPSG:4326, 3D Polygon):
  * ChinaPV_2020_v1.1.shp  -- 10,985 PV polygons for 2020  (USED)
  * ChinaPV_2015_v1.1.shp  -- 1,645  PV polygons for 2015  (DROPPED: pre-2016, spec 2)
  * PV_test_samples.shp    -- author train/test sample points (not needed)
Each polygon has: Lat, Lon (centroid), Area (km2), Perimeter, Province (str),
``urban`` (int: 0 = rural/ground-mounted, 1 = urban/distributed PV).

Time: labels are annual state maps. Only the 2020 epoch is in the Sentinel era; the 2015
epoch is entirely pre-2016 and is dropped per spec 2 (a PV panel visible only in 2015
cannot be confidently placed post-2016). Time range = 1-year window on 2020; PV
installations are persistent, so a static-label year window is appropriate.

Encoding (polygons, spec section 4). The source ``urban`` attribute is a genuine
appearance/context split (rural utility-scale ground-mount vs urban rooftop/distributed
PV), and the manifest names the class "PV installation (urban/rural)", so we keep TWO
foreground classes plus background:
  0 = background (non-PV land within the tile -- real surroundings, spatially meaningful,
      NOT a fabricated negative)
  1 = pv_rural  (urban == 0: rural / ground-mounted PV)
  2 = pv_urban  (urban == 1: urban / distributed PV)
Each polygon is rasterized into ONE <=64x64 UTM tile at 10 m/pixel, centered on the
geometry's representative point (guaranteed inside the polygon), sized to the footprint
plus an 8 px background margin and capped at 64x64. Polygons larger than 640 m are
represented by a 64x64 crop around the representative point (local footprint + boundary).
all_touched rasterization so thin/small installations remain visible at 10 m.

Sampling: classification, up to 1000 tiles per foreground class (balance_by_class over the
urban/rural class), well under the 25k hard cap. Background is a normal class id but is not
a sampling target (it appears in essentially every tile).

Run (idempotent -- skips already-written {id}.tif):
  python3 -m olmoearth_pretrain.open_set_segmentation_data.datasets.chinapv
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

from olmoearth_pretrain.open_set_segmentation_data import download, io, manifest
from olmoearth_pretrain.open_set_segmentation_data.rasterize import (
    geom_to_pixels,
    rasterize_shapes,
)
from olmoearth_pretrain.open_set_segmentation_data.sampling import balance_by_class

SLUG = "chinapv"
NAME = "ChinaPV"
URL = "https://zenodo.org/records/14292571"
ZENODO_ID = "14292571"

SHP_2020 = "ChinaPV_2020_v1.1.shp"
SHP_2015 = "ChinaPV_2015_v1.1.shp"
# All sidecar parts needed to open each shapefile.
DL_FILES = [
    "ChinaPV_2020_v1.1.shp",
    "ChinaPV_2020_v1.1.shx",
    "ChinaPV_2020_v1.1.dbf",
    "ChinaPV_2020_v1.1.prj",
    "ChinaPV_2015_v1.1.shp",
    "ChinaPV_2015_v1.1.shx",
    "ChinaPV_2015_v1.1.dbf",
    "ChinaPV_2015_v1.1.prj",
]

CID_BACKGROUND = 0
CID_RURAL = 1
CID_URBAN = 2
CLASSES = [
    {
        "id": CID_BACKGROUND,
        "name": "background",
        "description": "Non-PV land surface surrounding / between the PV panels within the "
        "tile. Real observed surroundings, not a fabricated negative.",
    },
    {
        "id": CID_RURAL,
        "name": "pv_rural",
        "description": "Rural / ground-mounted utility-scale photovoltaic installation "
        "footprint (source ``urban`` == 0). ChinaPV 2020, mapped from Landsat-8 with manual "
        "refinement.",
    },
    {
        "id": CID_URBAN,
        "name": "pv_urban",
        "description": "Urban / distributed photovoltaic installation footprint (source "
        "``urban`` == 1; e.g. rooftop / built-up-area PV). ChinaPV 2020, mapped from "
        "Landsat-8 with manual refinement.",
    },
]

YEAR = 2020
PAD = 8  # px of background margin added around the footprint (before the 64 cap).
PER_CLASS = 1000
SRC_PROJ = Projection(CRS.from_epsg(4326), 1, 1)  # geometries are WGS84 lon/lat.


def read_polygons() -> list[dict[str, Any]]:
    """Read the 2020 PV polygons into records (lon/lat, geometry WKB, urban class)."""
    path = io.raw_dir(SLUG) / SHP_2020
    recs: list[dict[str, Any]] = []
    with fiona.open(path.path) as src:
        for idx, feat in enumerate(src):
            props = feat["properties"]
            geom = shapely.geometry.shape(feat["geometry"])
            if geom.is_empty:
                continue
            geom = shapely.force_2d(geom)
            urban = int(props["urban"])
            recs.append(
                {
                    "lon": float(props["Lon"]),
                    "lat": float(props["Lat"]),
                    "geom_wkb": shapely.to_wkb(geom),
                    "fg_class": CID_URBAN if urban == 1 else CID_RURAL,
                    "urban": urban,
                    "province": props.get("Province"),
                    "area_km2": float(props["Area"]),
                    "source_id": f"ChinaPV_2020/{idx}",
                }
            )
    return recs


def _write_poly(rec: dict[str, Any]) -> str:
    sample_id = rec["sample_id"]
    tif = io.locations_dir(SLUG) / f"{sample_id}.tif"
    if tif.exists():
        return "skip"
    proj = io.utm_projection_for_lonlat(rec["lon"], rec["lat"])
    geom = shapely.from_wkb(rec["geom_wkb"])
    pix = geom_to_pixels(geom, SRC_PROJ, proj)
    minx, miny, maxx, maxy = pix.bounds
    rp = pix.representative_point()
    cx, cy = int(round(rp.x)), int(round(rp.y))
    w = min(io.MAX_TILE, max(1, int(np.ceil(maxx - minx)) + PAD))
    h = min(io.MAX_TILE, max(1, int(np.ceil(maxy - miny)) + PAD))
    bounds = io.centered_bounds(cx, cy, w, h)
    arr = rasterize_shapes(
        [(pix, rec["fg_class"])],
        bounds,
        fill=CID_BACKGROUND,
        dtype="uint8",
        all_touched=True,
    )
    present = sorted(set(np.unique(arr).tolist()))
    io.write_label_geotiff(SLUG, sample_id, arr, proj, bounds, nodata=io.CLASS_NODATA)
    io.write_sample_json(
        SLUG,
        sample_id,
        proj,
        bounds,
        io.year_range(YEAR),
        source_id=rec["source_id"],
        classes_present=present,
    )
    return "urban" if rec["fg_class"] == CID_URBAN else "rural"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=64)
    args = parser.parse_args()

    io.check_disk()
    manifest.write_registry_entry(SLUG, "in_progress")

    raw = io.raw_dir(SLUG)
    raw.mkdir(parents=True, exist_ok=True)
    download.download_zenodo(ZENODO_ID, raw, filenames=DL_FILES)
    with (raw / "SOURCE.txt").open("w") as fp:
        fp.write(
            "ChinaPV: spatial distribution of solar photovoltaic installations across "
            "China in 2015 and 2020 (Sci Data; Zenodo record 14292571).\n"
            f"{URL}\n"
            "License: CC-BY-4.0.\n"
            f"{SHP_2020}  (10,985 PV polygons, 2020; USED)\n"
            f"{SHP_2015}  (1,645 PV polygons, 2015; DROPPED -- pre-2016)\n"
        )

    print("reading 2020 polygons ...", flush=True)
    recs = read_polygons()
    print(f"  {len(recs)} polygons", flush=True)

    src_counts = Counter(r["fg_class"] for r in recs)
    print("source fg counts:", dict(src_counts), flush=True)

    # Up to PER_CLASS tiles per foreground (urban/rural) class.
    selected = balance_by_class(recs, key="fg_class", per_class=PER_CLASS)
    selected.sort(key=lambda r: (r["fg_class"], r["source_id"]))
    for i, r in enumerate(selected):
        r["sample_id"] = f"{i:06d}"
    print(f"selected {len(selected)} polygons", flush=True)

    io.check_disk()
    results: Counter = Counter()
    provinces: Counter = Counter()
    with multiprocessing.Pool(args.workers) as p:
        for res in tqdm.tqdm(
            star_imap_unordered(p, _write_poly, [dict(rec=r) for r in selected]),
            total=len(selected),
        ):
            results[res] += 1
    for r in selected:
        provinces[r["province"]] += 1
    print("write results:", dict(results), flush=True)

    io.check_disk()
    io.write_dataset_metadata(
        SLUG,
        {
            "dataset": SLUG,
            "name": NAME,
            "task_type": "classification",
            "source": "Zenodo / Sci Data",
            "license": "CC-BY-4.0",
            "provenance": {
                "url": URL,
                "zenodo_record": ZENODO_ID,
                "have_locally": False,
                "annotation_method": "Landsat-8 derived-product + manual refinement",
                "file": SHP_2020,
            },
            "sensors_relevant": ["sentinel2", "sentinel1", "landsat"],
            "classes": CLASSES,
            "nodata_value": io.CLASS_NODATA,
            "num_samples": len(selected),
            "class_tile_counts": {
                "pv_rural": results.get("rural", 0),
                "pv_urban": results.get("urban", 0),
            },
            "source_class_counts": {
                "pv_rural": src_counts.get(CID_RURAL, 0),
                "pv_urban": src_counts.get(CID_URBAN, 0),
            },
            "province_counts": dict(sorted(provinces.items(), key=lambda kv: -kv[1])),
            "tile_size": io.MAX_TILE,
            "time_range_year": YEAR,
            "notes": (
                "ChinaPV PV-installation polygons, 2020 epoch only (10,985 polygons; the "
                "2015 epoch of 1,645 polygons is dropped as entirely pre-2016 per spec 2). "
                "Each polygon rasterized into ONE <=64x64 UTM tile @10 m: 1=pv_rural "
                "(source urban==0, ground-mounted utility-scale), 2=pv_urban (source "
                "urban==1, distributed/rooftop), 0=background (real surroundings). Tile "
                "centered on the geometry's representative point, sized to footprint + 8 px "
                "margin, capped at 64x64; the ~33% of polygons spanning >640 m -> a 64x64 "
                "crop around that point (local footprint + boundary). all_touched "
                "rasterization. Sampled up to 1000 tiles per foreground class "
                "(balance_by_class). Positive-only foreground classes -> no fabricated "
                "negatives (spec 5); background pixels are genuine surroundings. Time range "
                "= 1-year window on 2020; PV installations are persistent. Caveat: the "
                "urban/rural split is the source ``urban`` attribute; at 10 m the panels "
                "themselves may not always be visually separable from surroundings alone."
            ),
        },
    )
    print(f"done: {len(selected)} tiles", flush=True)


if __name__ == "__main__":
    multiprocessing.set_start_method("forkserver", force=True)
    main()
