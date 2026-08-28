"""Process "AI Dataset for Solar Energy Locations in India" into label patches.

Source: Microsoft / The Nature Conservancy, "An Artificial Intelligence Dataset for
Solar Energy Locations in India" (Ortiz et al. 2022, arXiv:2202.01340), published on
GitHub: https://github.com/microsoft/solar-farms-mapping . A spatially-explicit ML model
mapped utility-scale solar PV across India from Sentinel-2 imagery; predictions were
validated by human experts to yield **1363 solar PV farms** for 2021.

Files (data/):
  * solar_farms_india_2021.geojson         -- raw individual polygons (4158 parts)
  * solar_farms_india_2021_merged.geojson  -- parts clustered into 1363 farms by
                                              proximity (shared ``fid``); ONE MultiPolygon
                                              per farm. We use THIS file (one farm = one
                                              sample, human-validated count).
  * ..._merged_simplified.geojson          -- simplified geometry (not used).
Each feature (EPSG:4326) has: State, Area (m2), Latitude, Longitude (center point), fid.

License: dataset is CDLA-Permissive-2.0 (open; attribution-friendly permissive) -> usable.

Encoding (polygons, spec section 4). Positive-only single foreground class. For each farm
we rasterize the merged MultiPolygon into ONE <=64x64 UTM tile at 10 m/pixel:
  0   = background (non-solar land within the tile -- the real surroundings of the farm,
        spatially meaningful, NOT a fabricated negative)
  1   = utility_scale_pv_farm (solar panel footprint)
The tile is centered on the geometry's representative point (guaranteed inside a panel
polygon, robust to farms whose merged parts are scattered) and sized to the farm's
footprint plus a small background margin, capped at 64x64. Farms larger than 640 m are
represented by a 64x64 crop around the representative point (local footprint + boundary).
all_touched rasterization so thin/small farms remain visible at 10 m.

Sampling: one tile per farm -> all 1363 human-validated farms kept (single positive class,
far under the 25k hard cap; see spec section 5 -- do NOT fabricate extra negatives).
Time range: 1-year window anchored on 2021 (the dataset year); solar farms are persistent.

Run (idempotent -- skips already-written {id}.tif):
  python3 -m olmoearth_pretrain.open_set_segmentation_data.datasets.\
ai_dataset_for_solar_energy_locations_in_india
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

SLUG = "ai_dataset_for_solar_energy_locations_in_india"
NAME = "AI Dataset for Solar Energy Locations in India"
URL = "https://github.com/microsoft/solar-farms-mapping"
RAW_BASE = "https://raw.githubusercontent.com/microsoft/solar-farms-mapping/main/data/"
MERGED_FILE = "solar_farms_india_2021_merged.geojson"
RAW_FILE = "solar_farms_india_2021.geojson"

CID_BACKGROUND = 0
CID_SOLAR = 1
CLASSES = [
    {
        "id": CID_BACKGROUND,
        "name": "background",
        "description": "Non-solar land surface surrounding / between the solar-farm panels "
        "within the tile. Real observed surroundings, not a fabricated negative.",
    },
    {
        "id": CID_SOLAR,
        "name": "utility_scale_pv_farm",
        "description": "Utility-scale ground-mounted photovoltaic solar-farm footprint "
        "(panel arrays) in India, 2021. Mapped from Sentinel-2 by an ML model and "
        "validated by human experts (Ortiz et al. 2022).",
    },
]

YEAR = 2021
PAD = 8  # px of background margin added around the footprint (before the 64 cap).
SRC_PROJ = Projection(CRS.from_epsg(4326), 1, 1)  # geometries are WGS84 lon/lat.


def read_farms() -> list[dict[str, Any]]:
    """Read the 1363 merged farms into records (lon/lat, geometry WKB, props)."""
    path = io.raw_dir(SLUG) / MERGED_FILE
    recs: list[dict[str, Any]] = []
    with fiona.open(path.path) as src:
        for feat in src:
            props = feat["properties"]
            geom = shapely.geometry.shape(feat["geometry"])
            if geom.is_empty:
                continue
            recs.append(
                {
                    "lon": float(props["Longitude"]),
                    "lat": float(props["Latitude"]),
                    "geom_wkb": shapely.to_wkb(geom),
                    "fid": int(props["fid"]),
                    "state": props.get("State"),
                    "area_m2": float(props["Area"]),
                    "source_id": f"fid/{int(props['fid'])}",
                }
            )
    return recs


def _write_farm(rec: dict[str, Any]) -> str:
    sample_id = rec["sample_id"]
    tif = io.locations_dir(SLUG) / f"{sample_id}.tif"
    if tif.exists():
        return "skip"
    proj = io.utm_projection_for_lonlat(rec["lon"], rec["lat"])
    geom = shapely.from_wkb(rec["geom_wkb"])
    pix = geom_to_pixels(geom, SRC_PROJ, proj)
    minx, miny, maxx, maxy = pix.bounds
    # Center on a point guaranteed to be inside a panel polygon (robust for farms whose
    # merged parts are scattered, where the bbox center could fall on empty land).
    rp = pix.representative_point()
    cx, cy = int(round(rp.x)), int(round(rp.y))
    w = min(io.MAX_TILE, max(1, int(np.ceil(maxx - minx)) + PAD))
    h = min(io.MAX_TILE, max(1, int(np.ceil(maxy - miny)) + PAD))
    bounds = io.centered_bounds(cx, cy, w, h)
    arr = rasterize_shapes(
        [(pix, CID_SOLAR)], bounds, fill=CID_BACKGROUND, dtype="uint8", all_touched=True
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
    return "solar" if CID_SOLAR in present else "empty"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=64)
    args = parser.parse_args()

    io.check_disk()
    manifest.write_registry_entry(SLUG, "in_progress")

    raw = io.raw_dir(SLUG)
    raw.mkdir(parents=True, exist_ok=True)
    from olmoearth_pretrain.open_set_segmentation_data import download

    for f in (MERGED_FILE, RAW_FILE):
        download.download_http(RAW_BASE + f, raw / f)
    with (raw / "SOURCE.txt").open("w") as fp:
        fp.write(
            "AI Dataset for Solar Energy Locations in India "
            "(Microsoft/TNC; Ortiz et al. 2022, arXiv:2202.01340).\n"
            f"{URL}\n"
            f"{RAW_BASE}{MERGED_FILE}  (1363 human-validated farms; used)\n"
            f"{RAW_BASE}{RAW_FILE}  (4158 raw polygon parts; reference)\n"
            "License: CDLA-Permissive-2.0.\n"
        )

    print("reading merged farms ...", flush=True)
    farms = read_farms()
    print(f"  {len(farms)} farms", flush=True)

    farms.sort(key=lambda r: r["fid"])
    for i, r in enumerate(farms):
        r["sample_id"] = f"{i:06d}"

    io.check_disk()
    results: Counter = Counter()
    states: Counter = Counter()
    with multiprocessing.Pool(args.workers) as p:
        for res in tqdm.tqdm(
            star_imap_unordered(p, _write_farm, [dict(rec=r) for r in farms]),
            total=len(farms),
        ):
            results[res] += 1
    for r in farms:
        states[r["state"]] += 1
    print("write results:", dict(results), flush=True)

    io.check_disk()
    io.write_dataset_metadata(
        SLUG,
        {
            "dataset": SLUG,
            "name": NAME,
            "task_type": "classification",
            "source": "GitHub (Microsoft/TNC)",
            "license": "CDLA-Permissive-2.0",
            "provenance": {
                "url": URL,
                "paper": "arXiv:2202.01340",
                "have_locally": False,
                "annotation_method": "ML segmentation (Sentinel-2) + full human expert validation",
                "file": MERGED_FILE,
            },
            "sensors_relevant": ["sentinel2", "sentinel1", "landsat"],
            "classes": CLASSES,
            "nodata_value": io.CLASS_NODATA,
            "num_samples": len(farms),
            "class_tile_counts": {
                "utility_scale_pv_farm": results.get("solar", 0),
                "with_background_pixels": None,
            },
            "state_counts": dict(sorted(states.items(), key=lambda kv: -kv[1])),
            "tile_size": io.MAX_TILE,
            "time_range_year": YEAR,
            "notes": (
                "1363 human-validated utility-scale solar PV farms across India (2021), "
                "from the merged geojson (one MultiPolygon per farm). Each farm rasterized "
                "into ONE <=64x64 UTM tile @10 m: 1=utility_scale_pv_farm, 0=background "
                "(real surroundings). Tile centered on the geometry's representative point "
                "(robust for scattered merged farms), sized to footprint + 8 px margin, "
                "capped at 64x64; farms larger than 640 m -> a 64x64 crop around that point. "
                "all_touched rasterization. Positive-only single class -> no fabricated "
                "negatives (spec section 5); background pixels are genuine surroundings. All "
                "1363 farms kept (single class, well under the 25k cap; slightly above the "
                "1000/class soft guidance -- kept every validated farm intentionally). Time "
                "range = 1-year window on 2021; solar farms are persistent."
            ),
        },
    )
    print(f"done: {len(farms)} tiles", flush=True)


if __name__ == "__main__":
    multiprocessing.set_start_method("forkserver", force=True)
    main()
