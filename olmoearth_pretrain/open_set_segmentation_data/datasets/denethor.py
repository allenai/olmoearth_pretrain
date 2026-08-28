"""Process DENETHOR into open-set-segmentation label patches (rasterized crop polygons).

Source: DENETHOR (Kondmann et al., NeurIPS 2021 Datasets & Benchmarks) --
"The DynamicEarthNET dataset for Harmonized, inter-Operable, analysis-Ready, daily crop
monitoring from space". Crop-type field parcels for two spatially-separated 24 km x 24 km
tiles in Brandenburg, Germany, taken from different years (train tile 2018, test tile
2019). Field boundaries + crop ids come from the German state of Brandenburg cadastral /
CAP farmer-declaration data (GeoBasis-DE/LGB), harmonized into 9 high-level crop classes.

Only the **label vector files** are needed here (pretraining supplies its own imagery), so
we download just the two crop-parcel GeoJSONs (not the Planet Fusion / Sentinel time
series). They are hosted, unauthenticated, on Source Cooperative under the ESA "Fusion
Competition" project (the successor to the retired Radiant MLHub
``dlr_fusion_competition_germany`` collection):
  https://data.source.coop/esa/fusion-competition/
  - br-18E-242N-crop-labels-train-2018.geojson  (train tile, 2018)
  - br-17E-243N-crop-labels-test-2019.geojson   (test tile, 2019)

Each feature is a MultiPolygon in EPSG:25833 (ETRS89 / UTM 33N) with properties
``fid``, ``crop_id`` (1-9), ``crop_name``, ``SHAPE_AREA``, ``SHAPE_LEN``.

Task: per-pixel **classification** (crop type). Each parcel is rasterized into a <=64x64
local-UTM 10 m tile (tile sized to the parcel footprint, centered on it, capped at 64):
the parcel's crop class id is burned inside the polygon, everything outside is nodata
(255) -- we only have a ground-truth crop label inside declared parcels, so outside is
"ignore", not a background class (matches the eurocrops recipe).

Classes (9): the source's 9 high-level crop classes. We keep the source ``crop_id`` order
but shift to 0-based ids (class id = crop_id - 1, ids 0-8). Names/descriptions from the
dataset documentation (Crops_GT_Brandenburg_Doc.pdf).

Sampling: tiles-per-class balanced with the 25k per-dataset cap (``balance_by_class``).
The full dataset is only ~4.6k parcels (max 954 in one class), well under both the
1000/class and 25k caps, so no truncation occurs in practice.

Time range: 1-year window anchored on each tile's labeled year (2018 / 2019). Both are
post-2016 (Sentinel era). Static seasonal crop labels -> no change_time.

Run (idempotent; skips already-written {sample_id}.tif):
  python3 -m olmoearth_pretrain.open_set_segmentation_data.datasets.denethor
"""

import argparse
import multiprocessing
from collections import Counter
from typing import Any

import geopandas as gpd
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

SLUG = "denethor"
NAME = "DENETHOR"

SOURCE_BASE = "https://data.source.coop/esa/fusion-competition"
# (filename on Source Cooperative, short tile code, labeled year).
TILES = [
    {
        "file": "br-18E-242N-crop-labels-train-2018.geojson",
        "code": "train2018",
        "year": 2018,
    },
    {
        "file": "br-17E-243N-crop-labels-test-2019.geojson",
        "code": "test2019",
        "year": 2019,
    },
]
DOC_FILE = "Crops_GT_Brandenburg_Doc.pdf"

# Source high-level crop classes, indexed by crop_id (1-9). class_id = crop_id - 1.
CROP_NAMES = {
    1: "Wheat",
    2: "Rye",
    3: "Barley",
    4: "Oats",
    5: "Corn",
    6: "Oil Seeds",
    7: "Root Crops",
    8: "Meadows",
    9: "Forage Crops",
}
CROP_DESCRIPTIONS = {
    1: "Wheat fields (CAP-declared, Brandenburg; merged from the 1-999 German crop code system).",
    2: "Rye fields (CAP-declared, Brandenburg).",
    3: "Barley fields (CAP-declared, Brandenburg).",
    4: "Oats fields (CAP-declared, Brandenburg).",
    5: "Corn / maize fields (CAP-declared, Brandenburg).",
    6: "Oil-seed crops, e.g. rapeseed/canola and sunflower (CAP-declared, Brandenburg).",
    7: "Root crops, e.g. sugar beet and potato; rare in this region but retained to reflect real crop imbalance.",
    8: "Meadows / permanent grassland (CAP-declared, Brandenburg).",
    9: "Forage crops, e.g. legumes and other fodder crops (CAP-declared, Brandenburg).",
}

PER_CLASS = (
    1000  # spec target; lowered automatically to 25000 // N by balance_by_class.
)
MAX_TILE = io.MAX_TILE  # 64
_WGS84_SRC = Projection(CRS.from_epsg(4326), 1, 1)


def ensure_data() -> None:
    """Download the two crop-label GeoJSONs + the documentation PDF into raw_dir."""
    raw = io.raw_dir(SLUG)
    raw.mkdir(parents=True, exist_ok=True)
    # Source Cooperative sits behind Cloudflare, which 403s the default urllib agent.
    hdrs = {"User-Agent": "Mozilla/5.0 (open-set-segmentation data fetch)"}
    for t in TILES:
        download.download_http(
            f"{SOURCE_BASE}/{t['file']}", raw / t["file"], headers=hdrs
        )
    download.download_http(f"{SOURCE_BASE}/{DOC_FILE}", raw / DOC_FILE, headers=hdrs)
    with (raw / "SOURCE.txt").open("w") as f:
        f.write(
            "DENETHOR crop-type label parcels (Brandenburg, Germany), Kondmann et al., "
            "NeurIPS 2021.\n"
            "Downloaded from Source Cooperative (ESA Fusion Competition), unauthenticated:\n"
            f"  {SOURCE_BASE}/\n"
            + "".join(
                f"  - {t['file']} (tile {t['code']}, year {t['year']})\n" for t in TILES
            )
            + "License: DL-DE/BY-2.0 (c) GeoBasis-DE/LGB (2018/19); original data altered.\n"
            "Only the label vector files are downloaded; pretraining supplies its own imagery.\n"
        )


def _write_tile(rec: dict[str, Any]) -> tuple[str, str]:
    sample_id = rec["sample_id"]
    tif = io.locations_dir(SLUG) / f"{sample_id}.tif"
    if tif.exists():
        return sample_id, "skip"
    try:
        geom = shapely.from_wkb(rec["geom_wkb"])  # WGS84 (lon/lat) geometry
        proj = io.utm_projection_for_lonlat(rec["lon"], rec["lat"])
        pix = geom_to_pixels(geom, _WGS84_SRC, proj)
        minx, miny, maxx, maxy = pix.bounds
        cx = int(round((minx + maxx) / 2))
        cy = int(round((miny + maxy) / 2))
        w = min(MAX_TILE, max(1, int(np.ceil(maxx - minx))))
        h = min(MAX_TILE, max(1, int(np.ceil(maxy - miny))))
        bounds = io.centered_bounds(cx, cy, w, h)
        arr = rasterize_shapes(
            [(pix, int(rec["class_id"]))],
            bounds,
            fill=io.CLASS_NODATA,
            dtype="uint8",
            all_touched=True,
        )
        if not (arr != io.CLASS_NODATA).any():
            return sample_id, "empty"
        io.write_label_geotiff(
            SLUG, sample_id, arr, proj, bounds, nodata=io.CLASS_NODATA
        )
        io.write_sample_json(
            SLUG,
            sample_id,
            proj,
            bounds,
            io.year_range(rec["year"]),
            source_id=rec["source_id"],
            classes_present=sorted(set(np.unique(arr).tolist()) - {io.CLASS_NODATA}),
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

    ensure_data()
    raw = io.raw_dir(SLUG)

    # ---- Read parcels from both tiles, reproject to WGS84, build candidate records ----
    records: list[dict[str, Any]] = []
    for t in TILES:
        gdf = gpd.read_file(str(raw / t["file"]))
        gdf = gdf.to_crs(4326)
        n_valid = 0
        for _, row in gdf.iterrows():
            geom = row.geometry
            if geom is None or geom.is_empty:
                continue
            crop_id = row.get("crop_id")
            try:
                crop_id = int(crop_id)
            except (TypeError, ValueError):
                continue
            if crop_id not in CROP_NAMES:
                continue
            cent = geom.centroid
            if not (np.isfinite(cent.x) and np.isfinite(cent.y)):
                continue
            records.append(
                {
                    "class_id": crop_id - 1,
                    "lon": float(cent.x),
                    "lat": float(cent.y),
                    "geom_wkb": shapely.to_wkb(geom),
                    "year": t["year"],
                    "source_id": f"{t['code']}/{int(row['fid'])}",
                }
            )
            n_valid += 1
        print(f"  {t['code']}: {n_valid} valid parcels")

    print(f"total candidate parcels: {len(records)}")

    selected = balance_by_class(
        records, key="class_id", per_class=PER_CLASS, total_cap=25000
    )
    n_classes = len(CROP_NAMES)
    eff_per_class = max(1, min(PER_CLASS, 25000 // n_classes))
    print(f"selected {len(selected)} parcels (eff per-class cap = {eff_per_class})")

    for i, r in enumerate(selected):
        r["sample_id"] = f"{i:06d}"

    # ---- Write tiles in parallel --------------------------------------------------
    io.check_disk()
    results: Counter = Counter()
    written_by_class: Counter = Counter()
    id_to_rec = {r["sample_id"]: r for r in selected}
    with multiprocessing.Pool(args.workers) as p:
        for sample_id, res in tqdm.tqdm(
            star_imap_unordered(p, _write_tile, [dict(rec=r) for r in selected]),
            total=len(selected),
        ):
            results[res] += 1
            if res in ("ok", "skip"):
                written_by_class[id_to_rec[sample_id]["class_id"]] += 1
    print("write results:", dict(results))

    io.check_disk()

    # ---- Metadata -----------------------------------------------------------------
    classes = [
        {
            "id": crop_id - 1,
            "name": CROP_NAMES[crop_id],
            "description": CROP_DESCRIPTIONS[crop_id],
        }
        for crop_id in sorted(CROP_NAMES)
    ]
    class_counts = {
        CROP_NAMES[crop_id]: int(written_by_class.get(crop_id - 1, 0))
        for crop_id in sorted(CROP_NAMES)
    }
    num_written = int(results.get("ok", 0) + results.get("skip", 0))
    io.write_dataset_metadata(
        SLUG,
        {
            "dataset": SLUG,
            "name": NAME,
            "task_type": "classification",
            "source": "Source Cooperative (ESA Fusion Competition) / DENETHOR (NeurIPS 2021)",
            "license": "DL-DE/BY-2.0",
            "provenance": {
                "url": "https://github.com/lukaskondmann/DENETHOR",
                "data_url": f"{SOURCE_BASE}/",
                "have_locally": False,
                "annotation_method": "farmer declaration (CAP) / Brandenburg cadastral data (GeoBasis-DE/LGB)",
                "tiles": [
                    {"file": t["file"], "code": t["code"], "year": t["year"]}
                    for t in TILES
                ],
            },
            "sensors_relevant": ["sentinel2", "sentinel1", "landsat"],
            "classes": classes,
            "nodata_value": io.CLASS_NODATA,
            "num_samples": num_written,
            "class_counts": class_counts,
            "notes": (
                "Crop-type field parcels for two 24x24 km tiles in Brandenburg, Germany "
                "(train tile 2018, test tile 2019). Each parcel rasterized into a <=64x64 "
                "local-UTM 10 m tile: crop class id (crop_id-1, ids 0-8) inside the polygon, "
                "255 (nodata/ignore) outside (no true background class; unlabeled land is "
                "ignore). Parcels larger than 640 m are centered and cropped to a 64x64 "
                "window. Tiles-per-class balanced with the 25k cap (no truncation: dataset "
                "is only ~4.6k parcels). Time range = 1-year window anchored on each tile's "
                "labeled year. Only the label GeoJSONs were downloaded (imagery not needed)."
            ),
        },
    )

    manifest.write_registry_entry(
        SLUG, "completed", task_type="classification", num_samples=num_written
    )
    print(f"done: {num_written} samples across {n_classes} classes")


if __name__ == "__main__":
    multiprocessing.set_start_method("forkserver", force=True)
    main()
