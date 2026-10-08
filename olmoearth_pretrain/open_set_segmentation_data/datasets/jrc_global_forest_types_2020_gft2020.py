"""Process JRC Global Forest Types 2020 (GFT2020 V1) into open-set-segmentation labels.

Source: EC JRC "Global map of forest types 2020 - version 1" (Bourgoin et al. 2026,
doi:10.2905/JRC.C760PNG; https://forobs.jrc.ec.europa.eu/GFT). A global 10 m
derived-product map (EPSG:4326, ~8.333e-5 deg/px) that classifies the forest area of the
JRC Global Forest Cover 2020 v3 mask into the main forest types defined by the EU
Deforestation Regulation (EUDR, Reg. (EU) 2023/1115). Distributed as 10x10-deg GeoTIFF
tiles and as one ~50 GB global COG on the JRC Big Data Platform (JEODPP):

    https://jeodpp.jrc.ec.europa.eu/ftp/jrc-opendata/FOREST/GFT2020/LATEST/single-cog/JRC_GFT2020_V1_cog.tif

License: CC BY 4.0 (free with attribution).

Raster value legend (verified by reading the COG; the manifest lists four EUDR types but
the V1 raster merges planted + plantation into a single "planted/plantation" value 20):

    0  = non-forest / outside the GFC2020 forest mask  -> nodata (255)
    1  = naturally regenerating forest
    10 = primary forest
    20 = planted / plantation forest

This is a forest-type-only product defined *inside* a forest mask, so per spec §2 we set
non-forest / no-data (source 0) to nodata=255 rather than inventing a background class.

Mapping to output class ids (uint8):
    id 0 = primary forest                 (source 10)
    id 1 = naturally regenerating forest  (source 1)
    id 2 = planted / plantation forest    (source 20)

task_type=classification, dense_raster. 2020 is a static per-year state, so change_time is
null and the time range is a 1-year window on 2020 (§5).

This is a GLOBAL 10 m product, so per spec §5 we do BOUNDED-TILE sampling: we range-read a
spatially-distributed set of small windows (REGIONS) across all continents and forest
biomes from the global COG (never the whole 50 GB mosaic), cache each locally, scan them
for spatially-homogeneous >=64x64 forest windows (a dominant forest class covering >=50%
of the block with <=20% non-forest -- §4 guidance to prefer homogeneous/high-confidence
windows for derived-product maps), then balance to <=1000 tiles/class. Selected native
windows (EPSG:4326) are reprojected to a local UTM projection at 10 m with NEAREST
resampling (categorical labels).

Run:  python3 -m olmoearth_pretrain.open_set_segmentation_data.datasets.jrc_global_forest_types_2020_gft2020
"""

import argparse
import multiprocessing
import os
from collections import Counter
from typing import Any

import numpy as np
import rasterio
import tqdm
from rasterio.warp import Resampling, reproject, transform_bounds
from rasterio.windows import from_bounds
from rslearn.utils.mp import star_imap_unordered
from rslearn.utils.raster_format import get_transform_from_projection_and_bounds

from olmoearth_pretrain.open_set_segmentation_data import io

SLUG = "jrc_global_forest_types_2020_gft2020"

COG_URL = (
    "/vsicurl/https://jeodpp.jrc.ec.europa.eu/ftp/jrc-opendata/FOREST/GFT2020/"
    "LATEST/single-cog/JRC_GFT2020_V1_cog.tif"
)
HTTP_COG_URL = COG_URL.replace("/vsicurl/", "")

# GDAL settings for efficient windowed reads over HTTP from the global COG.
_GDAL_ENV = {
    "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR",
    "CPL_VSIL_CURL_ALLOWED_EXTENSIONS": ".tif",
    "GDAL_HTTP_MULTIRANGE": "YES",
    "GDAL_HTTP_MERGE_CONSECUTIVE_RANGES": "YES",
    "VSI_CACHE": "TRUE",
    "CPL_VSIL_CURL_CACHE_SIZE": "200000000",
}

YEAR = 2020
PER_CLASS = 1000
TILE = 64  # output UTM tile is 64x64 @ 10 m
BLOCK = 76  # native (~9.26 m) block ~= 700 m ~= a 64 px @ 10 m UTM tile
REGION_DEG = 0.4  # each sampled region window is 0.4 x 0.4 deg (~4800 px)
DOMINANCE_FLOOR = 0.5  # a candidate block must be >=50% one forest class
FOREST_FLOOR = 0.8  # ... and >=80% forest overall (<=20% non-forest/nodata)
PAD_DEG = 0.003  # ~330 m geographic pad so the UTM tile is fully covered

# source raster value -> output class id ; source 0 -> nodata (non-forest)
SRC_TO_ID = {10: 0, 1: 1, 20: 2}
FOREST_VALUES = tuple(SRC_TO_ID.keys())

# Output classes (id order). Descriptions follow the EUDR / FAO FRA definitions used by
# the JRC GFT2020 product.
CLASSES = [
    (
        "primary forest",
        "Naturally regenerated forest of native tree species where there are no clearly "
        "visible indications of human activity and ecological processes are not "
        "significantly disturbed (EUDR / FAO FRA definition).",
    ),
    (
        "naturally regenerating forest",
        "Forest predominantly composed of trees established through natural regeneration, "
        "excluding forests that are clearly primary and excluding planted/plantation "
        "forest (EUDR definition).",
    ),
    (
        "planted / plantation forest",
        "Forest predominantly composed of trees established through planting and/or "
        "deliberate seeding, including intensively managed plantation forest on short "
        "rotations. GFT2020 V1 merges the EUDR 'planted forest' and 'plantation forest' "
        "types into this single mapped class (raster value 20).",
    ),
]

# Spatially-distributed forest regions across continents/biomes. (lon, lat) centres of a
# REGION_DEG box, chosen to collectively cover all three forest types. Bounded-tile
# sampling only -- we never read the whole global mosaic.
REGIONS: dict[str, tuple[float, float]] = {
    # --- Amazon / neotropics (primary + regenerating) ---
    "amazon_central_br": (-63.0, -3.0),
    "amazon_south_br": (-60.0, -6.0),
    "amazon_peru": (-73.0, -8.0),
    "amazon_colombia": (-71.0, 1.0),
    "amazon_guyana": (-59.0, 4.0),
    "atlantic_forest_br": (-40.5, -19.5),
    "costa_rica": (-84.0, 10.0),
    # --- Congo basin (primary + regenerating) ---
    "congo_drc": (24.0, 1.0),
    "congo_gabon": (11.5, 0.0),
    "congo_cameroon": (13.5, 3.5),
    "congo_east": (27.5, -1.0),
    # --- SE Asia / Oceania tropics ---
    "borneo_kalimantan": (114.0, 0.5),
    "sumatra": (102.0, -1.0),
    "new_guinea": (140.0, -5.0),
    "new_guinea_w": (137.0, -4.0),
    "western_ghats_in": (75.5, 12.5),
    # --- Boreal (primary + regenerating) ---
    "siberia_c": (100.0, 58.0),
    "siberia_e": (110.0, 62.0),
    "siberia_w": (88.0, 60.0),
    "canada_bc": (-122.0, 55.0),
    "canada_quebec": (-74.0, 49.0),
    "canada_ontario": (-88.0, 50.0),
    "sweden_north": (18.0, 65.0),
    "finland": (26.0, 63.0),
    "russia_west": (40.0, 58.0),
    "alaska_interior": (-150.0, 64.0),
    # --- Temperate naturally regenerating ---
    "appalachia_us": (-81.0, 37.5),
    "great_lakes_us": (-85.0, 45.5),
    "germany_alps": (11.0, 47.8),
    "carpathians": (24.5, 47.5),
    "pnw_us": (-122.5, 44.0),
    "ne_china": (128.0, 45.0),
    "japan_honshu": (138.0, 36.5),
    "eastern_australia": (152.0, -30.0),
    "tasmania": (146.5, -42.0),
    # --- Planted / plantation forest ---
    "se_us_pine_ga": (-82.0, 33.0),
    "se_us_pine_ms": (-89.5, 31.5),
    "se_us_pine_nc": (-79.0, 35.0),
    "brazil_eucalyptus": (-49.5, -20.5),
    "brazil_eucalyptus2": (-51.0, -23.0),
    "chile_plantation": (-72.5, -38.0),
    "chile_plantation2": (-73.0, -40.0),
    "new_zealand_ni": (176.0, -38.5),
    "new_zealand_si": (172.5, -42.5),
    "portugal": (-8.2, 40.0),
    "spain_nw": (-7.0, 42.5),
    "france_landes": (-0.8, 44.3),
    "sweden_south": (14.5, 58.0),
    "germany_north": (9.5, 52.5),
    "china_south": (114.0, 26.0),
    "china_southwest": (110.0, 25.0),
    "japan_kyushu": (131.0, 32.5),
    "south_africa_mpu": (30.5, -25.5),
    "india_plantation": (77.0, 13.5),
    "vietnam": (106.0, 20.0),
    "sw_australia": (116.0, -34.0),
    "victoria_australia": (147.5, -37.5),
    "uruguay": (-56.0, -32.5),
    "indonesia_java": (110.5, -7.5),
}


def _apply_gdal_env() -> None:
    for k, v in _GDAL_ENV.items():
        os.environ[k] = v


def region_path(name: str):
    return io.raw_dir(SLUG) / "regions" / f"{name}.tif"


def download_region(name: str) -> None:
    """Range-read a REGION_DEG window from the global COG and cache it locally (idempotent)."""
    dst = region_path(name)
    if dst.exists():
        return
    _apply_gdal_env()
    lon, lat = REGIONS[name]
    half = REGION_DEG / 2.0
    with rasterio.open(COG_URL) as ds:
        win = from_bounds(lon - half, lat - half, lon + half, lat + half, ds.transform)
        arr = ds.read(1, window=win)
        win_transform = ds.window_transform(win)
        profile = {
            "driver": "GTiff",
            "dtype": "uint8",
            "count": 1,
            "height": arr.shape[0],
            "width": arr.shape[1],
            "crs": ds.crs,
            "transform": win_transform,
            "compress": "deflate",
            "tiled": True,
        }
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.parent / (dst.name + ".tmp")
    with rasterio.open(str(tmp), "w", **profile) as out:
        out.write(arr, 1)
    tmp.rename(dst)


def _scan_region(name: str) -> list[dict[str, Any]]:
    """Find homogeneous BLOCKxBLOCK forest windows in one cached region tile."""
    path = str(region_path(name))
    with rasterio.open(path) as ds:
        arr = ds.read(1)
        st = ds.transform
    h, w = arr.shape
    nby, nbx = h // BLOCK, w // BLOCK
    if nby == 0 or nbx == 0:
        return []
    a = arr[: nby * BLOCK, : nbx * BLOCK].reshape(nby, BLOCK, nbx, BLOCK)
    denom = float(BLOCK * BLOCK)
    forest = np.zeros((nby, nbx), np.float32)
    best = np.zeros((nby, nbx), np.float32)
    best_src = np.zeros((nby, nbx), np.uint8)
    for v in FOREST_VALUES:
        cnt = (a == v).sum(axis=(1, 3)).astype(np.float32) / denom
        forest += cnt
        m = cnt > best
        best[m] = cnt[m]
        best_src[m] = v
    qual = (forest >= FOREST_FLOOR) & (best >= DOMINANCE_FLOOR) & (best_src > 0)
    brs, bcs = np.nonzero(qual)
    cx = bcs * BLOCK + BLOCK / 2.0
    cy = brs * BLOCK + BLOCK / 2.0
    lons = st.c + cx * st.a
    lats = st.f + cy * st.e
    recs = []
    for br, bc, lon, lat in zip(
        brs.tolist(), bcs.tolist(), lons.tolist(), lats.tolist()
    ):
        src_v = int(best_src[br, bc])
        recs.append(
            {
                "region": name,
                "lon": float(lon),
                "lat": float(lat),
                "label": SRC_TO_ID[src_v],
                "frac": float(best[br, bc]),
                "source_id": f"{name}_r{br}_c{bc}",
            }
        )
    return recs


def _write_one(rec: dict[str, Any]) -> None:
    sample_id = rec["sample_id"]
    if (io.locations_dir(SLUG) / f"{sample_id}.tif").exists():
        return
    lon, lat = rec["lon"], rec["lat"]
    dst_proj, col, row = io.lonlat_to_utm_pixel(lon, lat)
    bounds = io.centered_bounds(col, row, TILE, TILE)
    dst_transform = get_transform_from_projection_and_bounds(dst_proj, bounds)

    # Geographic bbox of the UTM tile so we can window the cached source read.
    xs = [bounds[0] * io.RESOLUTION, bounds[2] * io.RESOLUTION]
    ys = [bounds[1] * -io.RESOLUTION, bounds[3] * -io.RESOLUTION]
    left, right = min(xs), max(xs)
    bottom, top = min(ys), max(ys)
    l2, b2, r2, t2 = transform_bounds(
        dst_proj.crs, "EPSG:4326", left, bottom, right, top
    )

    with rasterio.open(str(region_path(rec["region"]))) as ds:
        win = from_bounds(
            l2 - PAD_DEG, b2 - PAD_DEG, r2 + PAD_DEG, t2 + PAD_DEG, ds.transform
        )
        src = ds.read(1, window=win, boundless=True, fill_value=0)
        win_transform = ds.window_transform(win)
        src_crs = ds.crs

    src_state = np.zeros((TILE, TILE), np.uint8)
    reproject(
        source=src,
        destination=src_state,
        src_transform=win_transform,
        src_crs=src_crs,
        dst_transform=dst_transform,
        dst_crs=dst_proj.crs,
        resampling=Resampling.nearest,
        src_nodata=0,
        dst_nodata=0,
    )
    out = np.full((TILE, TILE), io.CLASS_NODATA, np.uint8)
    for v, cid in SRC_TO_ID.items():
        out[src_state == v] = cid

    io.write_label_geotiff(
        SLUG, sample_id, out, dst_proj, bounds, nodata=io.CLASS_NODATA
    )
    present = sorted(int(x) for x in np.unique(out) if x != io.CLASS_NODATA)
    io.write_sample_json(
        SLUG,
        sample_id,
        dst_proj,
        bounds,
        io.year_range(YEAR),
        change_time=None,
        source_id=rec["source_id"],
        classes_present=present,
    )


def _write_source_txt() -> None:
    d = io.raw_dir(SLUG)
    d.mkdir(parents=True, exist_ok=True)
    (d / "SOURCE.txt").write_text(
        "JRC Global Forest Types 2020 (GFT2020) version 1.\n"
        f"Global 10 m COG (EPSG:4326), read via windowed HTTP range requests from:\n  {HTTP_COG_URL}\n"
        "Landing page: https://forobs.jrc.ec.europa.eu/GFT\n"
        "DOI: 10.2905/JRC.C760PNG  License: CC BY 4.0.\n"
        "Bounded-tile sampling only: the ~50 GB global mosaic is NOT downloaded; small\n"
        "0.4-deg windows over selected forest regions are cached under regions/.\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=64)
    args = parser.parse_args()

    _apply_gdal_env()
    io.check_disk()
    _write_source_txt()

    print(f"Range-reading {len(REGIONS)} region windows from the global COG...")
    with multiprocessing.Pool(min(len(REGIONS), 24)) as p:
        for _ in tqdm.tqdm(
            star_imap_unordered(p, download_region, [dict(name=n) for n in REGIONS]),
            total=len(REGIONS),
        ):
            pass
    io.check_disk()

    print("Scanning regions for homogeneous forest windows...")
    with multiprocessing.Pool(min(len(REGIONS), 32)) as p:
        all_recs: list[dict[str, Any]] = []
        for recs in tqdm.tqdm(
            star_imap_unordered(p, _scan_region, [dict(name=n) for n in REGIONS]),
            total=len(REGIONS),
        ):
            all_recs.extend(recs)
    print(f"scanned {len(all_recs)} candidate homogeneous windows")
    print("candidate class counts:", dict(Counter(r["label"] for r in all_recs)))

    from olmoearth_pretrain.open_set_segmentation_data.sampling import balance_by_class

    selected = balance_by_class(all_recs, "label", per_class=PER_CLASS)
    for i, r in enumerate(selected):
        r["sample_id"] = f"{i:06d}"
    print(f"selected {len(selected)} windows (<= {PER_CLASS}/class)")

    io.check_disk()
    with multiprocessing.Pool(args.workers) as p:
        for _ in tqdm.tqdm(
            star_imap_unordered(p, _write_one, [dict(rec=r) for r in selected]),
            total=len(selected),
        ):
            pass

    counts = Counter(r["label"] for r in selected)
    class_counts = {name: counts.get(i, 0) for i, (name, _d) in enumerate(CLASSES)}
    print("class counts:", class_counts)

    io.write_dataset_metadata(
        SLUG,
        {
            "dataset": SLUG,
            "name": "JRC Global Forest Types 2020 (GFT2020)",
            "task_type": "classification",
            "source": "EC JRC",
            "license": "CC BY 4.0 (free with attribution)",
            "provenance": {
                "url": "https://forobs.jrc.ec.europa.eu/GFT",
                "have_locally": False,
                "annotation_method": "derived-product (JRC GFT2020 V1, 10 m, EPSG:4326)",
                "citation": (
                    "Bourgoin, Ameztoy, Verhegghen, Carboni, Achard, Colditz (2026): "
                    "Global map of forest types 2020 - version 1. EC JRC. "
                    "doi:10.2905/JRC.C760PNG"
                ),
                "cog": HTTP_COG_URL,
                "year": YEAR,
                "source_value_legend": {
                    "0": "non-forest / outside forest mask (nodata)",
                    "1": "naturally regenerating forest",
                    "10": "primary forest",
                    "20": "planted / plantation forest",
                },
            },
            "sensors_relevant": ["sentinel2", "sentinel1", "landsat"],
            "classes": [
                {"id": i, "name": name, "description": desc}
                for i, (name, desc) in enumerate(CLASSES)
            ],
            "nodata_value": io.CLASS_NODATA,
            "num_samples": len(selected),
            "class_counts": class_counts,
            "notes": (
                "Bounded-tile sampling of the global JRC GFT2020 V1 10 m forest-type "
                f"product: {len(REGIONS)} spatially-distributed 0.4-deg windows across the "
                "Amazon, Congo, SE-Asia, boreal, temperate and plantation forest biomes on "
                "all continents, range-read from the global COG (the ~50 GB mosaic is never "
                "fully downloaded). Homogeneous >=64x64 windows (a dominant forest class "
                ">=50%, <=20% non-forest) reprojected from native ~10 m EPSG:4326 to local "
                "UTM at 10 m with nearest resampling. Non-forest/outside-mask (source 0) is "
                "nodata=255. The V1 raster merges EUDR 'planted' and 'plantation' into one "
                "class (value 20), so 3 classes are emitted rather than the manifest's 4. "
                "Static 2020 label: change_time=null, 1-year time range on 2020."
            ),
        },
    )
    print("done")


if __name__ == "__main__":
    multiprocessing.set_start_method("forkserver", force=True)
    main()
