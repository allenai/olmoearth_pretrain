"""Process MapBiomas Brasil annual LULC (Collection 9) into open-set-segmentation labels.

Source: MapBiomas Project — Brazil annual land-use/land-cover, Collection 9 (1985-2023),
30 m, Landsat-based. Public: the per-year national coverage mosaics are distributed as
single-band uint8 COGs (EPSG:4326, ~0.00027 deg/px ~= 30 m) on Google Cloud Storage:

    https://storage.googleapis.com/mapbiomas-public/initiatives/brasil/collection_9/
        lclu/coverage/brasil_coverage_{YEAR}.tif

Landing page: https://brasil.mapbiomas.org/en/downloads/  License: CC BY-SA 4.0.

We read the {YEAR} national COG (~158828 x 155241 px, ~24.6 Gpx) via windowed HTTP range
requests only (the full mosaic is never downloaded). We chose YEAR=2022 (post-2016).

NATIVE RESOLUTION IS 30 m (Landsat), NOT 10 m. Per spec we resample the categorical label
to the pretraining 10 m grid with NEAREST resampling (each ~30 m native pixel becomes a
~3x3 block of 10 m pixels). A 64x64 @ 10 m UTM output tile (640 m) is drawn from a ~22x22
native block. This is documented in the summary.

LEGEND COLLAPSE. MapBiomas Collection 9 has a deep hierarchical legend (~30 codes incl.
per-crop subclasses: soybean, sugar cane, rice, cotton, coffee, citrus, palm oil, ...).
Most crop / natural subclasses are NOT reliably separable at 30 m, so we collapse the
legend to a coherent set of level-1/level-2 classes and keep the distinctive Brazilian
ecosystem classes (savanna/Cerrado, floodable forest, mangrove, wetland, grassland). The
16-class output map (`SRC_TO_ID` below) is uint8; source 0 / 27 (no-data / not observed)
and any unmapped code -> nodata (255).

task_type=classification, dense_raster. 2022 is a static per-year state, so change_time is
null and the time range is a 1-year window on 2022 (spec 5).

Bounded-tile sampling (spec 5 + 4): we range-read spatially-distributed windows across the
six Brazilian biomes (Amazon, Cerrado, Atlantic Forest, Caatinga, Pantanal, Pampa) plus
targeted regions for rare classes (mangrove/aquaculture coasts, mining districts, coffee /
citrus / silviculture belts, rice in the Pampa), cache each locally, scan them for
mostly-observed >=22x22 native blocks (tiles-per-class balanced, rarest class first), then
reproject each selected block to a local UTM projection at 10 m (nearest) as a 64x64 tile.

Run:  python3 -m olmoearth_pretrain.open_set_segmentation_data.datasets.mapbiomas_brasil_annual_lulc
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
from olmoearth_pretrain.open_set_segmentation_data.sampling import (
    select_tiles_per_class,
)

SLUG = "mapbiomas_brasil_annual_lulc"

YEAR = 2022
COG_URL = (
    "/vsicurl/https://storage.googleapis.com/mapbiomas-public/initiatives/brasil/"
    f"collection_9/lclu/coverage/brasil_coverage_{YEAR}.tif"
)
HTTP_COG_URL = COG_URL.replace("/vsicurl/", "")

_GDAL_ENV = {
    "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR",
    "CPL_VSIL_CURL_ALLOWED_EXTENSIONS": ".tif",
    "GDAL_HTTP_MULTIRANGE": "YES",
    "GDAL_HTTP_MERGE_CONSECUTIVE_RANGES": "YES",
    "VSI_CACHE": "TRUE",
    "CPL_VSIL_CURL_CACHE_SIZE": "200000000",
}

PER_CLASS = 1000
TILE = 64  # output UTM tile 64x64 @ 10 m (= 640 m)
BLOCK = 22  # native block ~= 640 m (native ~30 m px) -> one output tile
REGION_DEG = 0.6  # each sampled region window is 0.6 x 0.6 deg
VALID_FLOOR = 0.90  # a candidate block must be >=90% observed (mapped) pixels
MIN_PRESENT_FRAC = 0.05  # a class counts as "present" in a block if it covers >=5%
PAD_DEG = 0.006  # ~660 m geographic pad so the reprojected UTM tile is covered

# --- Full MapBiomas Collection 9 (Brazil) legend: source code -> canonical name. Used for
# the metadata source_value_legend (documentation of the raw codes). ---
CODE_TO_NAME: dict[int, str] = {
    3: "Forest Formation",
    4: "Savanna Formation",
    5: "Mangrove",
    6: "Floodable Forest",
    49: "Wooded Sandbank Vegetation (Restinga Arborea)",
    11: "Wetland",
    12: "Grassland",
    32: "Hypersaline Tidal Flat",
    29: "Rocky Outcrop",
    50: "Herbaceous Sandbank Vegetation",
    13: "Other Non Forest Formations",
    15: "Pasture",
    18: "Agriculture",
    19: "Temporary Crop",
    39: "Soybean",
    20: "Sugar cane",
    40: "Rice",
    62: "Cotton",
    41: "Other Temporary Crops",
    36: "Perennial Crop",
    46: "Coffee",
    47: "Citrus",
    35: "Palm Oil",
    48: "Other Perennial Crops",
    9: "Forest Plantation (Silviculture)",
    21: "Mosaic of Uses",
    22: "Non vegetated area",
    23: "Beach, Dune and Sand Spot",
    24: "Urban Area",
    30: "Mining",
    25: "Other non Vegetated Areas",
    26: "Water",
    33: "River, Lake and Ocean",
    31: "Aquaculture",
    27: "Not Observed",
}

# --- Legend collapse: source code -> output class id (uint8). Codes absent here (0, 27,
# and any unmapped) -> nodata (255). ---
SRC_TO_ID: dict[int, int] = {
    3: 0,  # Forest Formation
    4: 1,  # Savanna Formation (Cerrado)
    5: 2,  # Mangrove
    6: 3,  # Floodable Forest
    9: 4,  # Forest Plantation (Silviculture)
    11: 5,  # Wetland
    12: 6,  # Grassland
    13: 7,
    29: 7,
    32: 7,
    49: 7,
    50: 7,  # Other Non-Forest Natural Formation
    15: 8,  # Pasture
    18: 9,
    19: 9,
    39: 9,
    20: 9,
    40: 9,
    62: 9,
    41: 9,  # Temporary Crop
    36: 10,
    46: 10,
    47: 10,
    35: 10,
    48: 10,  # Perennial Crop
    21: 11,  # Mosaic of Uses (agri/pasture mix)
    24: 12,  # Urban Area
    30: 13,  # Mining
    22: 14,
    23: 14,
    25: 14,  # Other Non-Vegetated Area
    26: 15,
    33: 15,
    31: 15,  # Water (incl. aquaculture)
}

# Output classes in id order (name, description).
CLASSES: list[tuple[str, str]] = [
    (
        "Forest Formation",
        "Dense natural forest (Amazon terra-firme rainforest, Atlantic Forest, seasonal "
        "and deciduous forests). MapBiomas code 3.",
    ),
    (
        "Savanna Formation",
        "Cerrado savanna woodland/shrubland with a continuous grass layer and scattered "
        "trees. MapBiomas code 4.",
    ),
    (
        "Mangrove",
        "Coastal salt-tolerant forest in the intertidal zone. MapBiomas code 5.",
    ),
    (
        "Floodable Forest",
        "Seasonally/permanently flooded forest (varzea, igapo, floodplain woodland). "
        "New in Collection 9. MapBiomas code 6.",
    ),
    (
        "Forest Plantation (Silviculture)",
        "Planted-tree forestry (eucalyptus, pine, etc.) established by planting/seeding. "
        "MapBiomas code 9.",
    ),
    ("Wetland", "Non-forest flooded/marshy natural vegetation. MapBiomas code 11."),
    (
        "Grassland",
        "Natural grassland / campo (open herbaceous natural vegetation). MapBiomas code 12.",
    ),
    (
        "Other Non-Forest Natural Formation",
        "Other natural non-forest cover: rocky outcrops, hypersaline tidal flats, wooded "
        "and herbaceous sandbank (restinga) vegetation, and other non-forest formations. "
        "MapBiomas codes 13, 29, 32, 49, 50.",
    ),
    ("Pasture", "Managed/cultivated pasture for livestock. MapBiomas code 15."),
    (
        "Temporary Crop",
        "Annual/seasonal cropland: soybean, sugar cane, rice, cotton and other temporary "
        "crops (per-crop subclasses collapsed; not reliably separable at 30 m). MapBiomas "
        "codes 18/19/39/20/40/62/41.",
    ),
    (
        "Perennial Crop",
        "Perennial cropland: coffee, citrus, oil palm and other perennial crops (subclasses "
        "collapsed). MapBiomas codes 36/46/47/35/48.",
    ),
    (
        "Mosaic of Uses",
        "Mixed agriculture and pasture that cannot be separated at the mapping scale. "
        "MapBiomas code 21.",
    ),
    ("Urban Area", "Urban built-up areas. MapBiomas code 24."),
    (
        "Mining",
        "Open-pit / surface mining (industrial and garimpo). MapBiomas code 30.",
    ),
    (
        "Other Non-Vegetated Area",
        "Other non-vegetated surfaces: beaches, dunes, sand spots and other non-vegetated "
        "areas. MapBiomas codes 22/23/25.",
    ),
    (
        "Water",
        "Rivers, lakes, ocean, reservoirs and aquaculture ponds. MapBiomas codes 26/33/31.",
    ),
]
N_CLASSES = len(CLASSES)

# Spatially-distributed (lon, lat) region centres across the six Brazilian biomes, plus
# targeted regions for rare classes. Bounded-tile sampling only.
REGIONS: dict[str, tuple[float, float]] = {
    # --- Amazon (forest, floodable forest, pasture frontier, mining, palm oil) ---
    "amazon_central": (-63.0, -3.5),
    "amazon_para_east": (-52.0, -3.3),
    "amazon_rondonia_arc": (-62.0, -10.2),
    "amazon_mt_soy_frontier": (-55.0, -12.2),
    "amazon_amazonas_varzea": (-64.5, -3.8),
    "amazon_acre": (-70.0, -9.5),
    "amazon_roraima": (-61.0, 2.2),
    "amazon_maranhao": (-46.5, -4.5),
    "amazon_carajas_mining": (-50.2, -6.05),
    "amazon_tapajos_garimpo": (-56.4, -6.4),
    "amazon_para_palm": (-48.3, -2.4),
    # --- Cerrado (savanna, soy, pasture, grassland, urban) ---
    "cerrado_goias": (-49.3, -16.2),
    "cerrado_matopiba_bahia": (-45.6, -12.2),
    "cerrado_tocantins": (-48.2, -10.3),
    "cerrado_minas": (-45.4, -17.2),
    "cerrado_ms": (-54.2, -19.4),
    "cerrado_brasilia": (-47.9, -15.8),
    "cerrado_piaui_matopiba": (-45.2, -9.2),
    # --- Atlantic Forest (forest, urban, sugarcane, coffee, citrus, silviculture) ---
    "atlantic_sp_metro": (-46.6, -23.5),
    "atlantic_sp_sugarcane": (-48.5, -21.4),
    "atlantic_sp_citrus": (-48.8, -21.0),
    "atlantic_sp_eucalyptus": (-48.0, -22.6),
    "atlantic_rio": (-43.2, -22.9),
    "atlantic_sul_minas_coffee": (-46.0, -21.4),
    "atlantic_es_eucalyptus": (-40.4, -18.6),
    "atlantic_parana": (-51.2, -25.2),
    "atlantic_santa_catarina": (-49.6, -27.1),
    "atlantic_bahia_coast": (-39.4, -15.2),
    # --- Caatinga (dry savanna/shrub, grassland, rocky outcrop, other natural) ---
    "caatinga_bahia": (-41.2, -10.3),
    "caatinga_pernambuco": (-38.6, -8.5),
    "caatinga_piaui": (-42.2, -8.2),
    "caatinga_ceara": (-39.6, -5.6),
    # --- Pantanal (wetland, floodable forest, grassland) ---
    "pantanal_north": (-56.6, -16.6),
    "pantanal_south": (-57.1, -18.6),
    # --- Pampa (grassland, rice, pasture) ---
    "pampa_rs_central": (-54.0, -30.5),
    "pampa_rs_rice": (-52.5, -31.4),
    "pampa_rs_west": (-56.2, -30.1),
    # --- Coastal (mangrove, aquaculture, beach, restinga) ---
    "coast_para_maranhao_mangrove": (-44.6, -1.6),
    "coast_ne_aquaculture": (-37.0, -5.1),
    "coast_se_cananeia_mangrove": (-47.9, -25.0),
    "coast_bahia_restinga": (-38.6, -12.6),
    # --- Iron mining (Minas quadrilatero ferrifero) ---
    "minas_iron_mining": (-43.9, -20.2),
}


def _apply_gdal_env() -> None:
    for k, v in _GDAL_ENV.items():
        os.environ[k] = v


def region_path(name: str):
    return io.raw_dir(SLUG) / "regions" / f"{name}.tif"


def download_region(name: str) -> None:
    """Range-read a REGION_DEG window from the national COG; cache locally (idempotent)."""
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


# Vectorized source-code -> output-id lookup (256 entries; 255 = nodata).
_LUT = np.full(256, io.CLASS_NODATA, dtype=np.uint8)
for _src, _cid in SRC_TO_ID.items():
    _LUT[_src] = _cid


def _scan_region(name: str) -> list[dict[str, Any]]:
    """Find mostly-observed BLOCKxBLOCK native windows in one cached region tile."""
    path = str(region_path(name))
    with rasterio.open(path) as ds:
        arr = ds.read(1)
        st = ds.transform
    ids = _LUT[arr]
    h, w = ids.shape
    nby, nbx = h // BLOCK, w // BLOCK
    if nby == 0 or nbx == 0:
        return []
    a = ids[: nby * BLOCK, : nbx * BLOCK].reshape(nby, BLOCK, nbx, BLOCK)
    denom = float(BLOCK * BLOCK)
    # valid (mapped, non-nodata) fraction per block
    valid = (a != io.CLASS_NODATA).sum(axis=(1, 3)).astype(np.float32) / denom
    # per-class fraction per block
    fracs = np.zeros((nby, nbx, N_CLASSES), np.float32)
    for cid in range(N_CLASSES):
        fracs[:, :, cid] = (a == cid).sum(axis=(1, 3)).astype(np.float32) / denom
    qual = valid >= VALID_FLOOR
    brs, bcs = np.nonzero(qual)
    recs = []
    for br, bc in zip(brs.tolist(), bcs.tolist()):
        present = [
            cid for cid in range(N_CLASSES) if fracs[br, bc, cid] >= MIN_PRESENT_FRAC
        ]
        if not present:
            continue
        cx = bc * BLOCK + BLOCK / 2.0
        cy = br * BLOCK + BLOCK / 2.0
        lon = st.c + cx * st.a
        lat = st.f + cy * st.e
        recs.append(
            {
                "region": name,
                "lon": float(lon),
                "lat": float(lat),
                "classes_present": present,
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

    # Geographic bbox of the UTM tile to window the cached source read.
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

    src_codes = np.zeros((TILE, TILE), np.uint8)
    reproject(
        source=src,
        destination=src_codes,
        src_transform=win_transform,
        src_crs=src_crs,
        dst_transform=dst_transform,
        dst_crs=dst_proj.crs,
        resampling=Resampling.nearest,
        src_nodata=0,
        dst_nodata=0,
    )
    out = _LUT[src_codes]

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
        "MapBiomas Brasil annual LULC, Collection 9 (30 m, Landsat-based).\n"
        f"National per-year coverage COG (EPSG:4326, ~30 m), year {YEAR}, read via windowed\n"
        f"HTTP range requests from:\n  {HTTP_COG_URL}\n"
        "Landing page: https://brasil.mapbiomas.org/en/downloads/\n"
        "License: CC BY-SA 4.0.\n"
        "Bounded-tile sampling only: the ~24.6 Gpx national mosaic is NOT downloaded; small\n"
        "0.6-deg windows over selected biome/rare-class regions are cached under regions/.\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=64)
    args = parser.parse_args()

    _apply_gdal_env()
    io.check_disk()
    _write_source_txt()

    print(
        f"Range-reading {len(REGIONS)} region windows from the {YEAR} national COG..."
    )
    with multiprocessing.Pool(min(len(REGIONS), 24)) as p:
        for _ in tqdm.tqdm(
            star_imap_unordered(p, download_region, [dict(name=n) for n in REGIONS]),
            total=len(REGIONS),
        ):
            pass
    io.check_disk()

    print("Scanning regions for mostly-observed candidate windows...")
    with multiprocessing.Pool(min(len(REGIONS), 32)) as p:
        all_recs: list[dict[str, Any]] = []
        for recs in tqdm.tqdm(
            star_imap_unordered(p, _scan_region, [dict(name=n) for n in REGIONS]),
            total=len(REGIONS),
        ):
            all_recs.extend(recs)
    print(f"scanned {len(all_recs)} candidate windows")
    cand_counts: Counter = Counter()
    for r in all_recs:
        for c in r["classes_present"]:
            cand_counts[c] += 1
    print(
        "candidate per-class tile counts:",
        {CLASSES[i][0]: cand_counts.get(i, 0) for i in range(N_CLASSES)},
    )

    selected = select_tiles_per_class(
        all_recs, classes_key="classes_present", per_class=PER_CLASS
    )
    for i, r in enumerate(selected):
        r["sample_id"] = f"{i:06d}"
    print(
        f"selected {len(selected)} tiles (tiles-per-class balanced, <= {PER_CLASS}/class)"
    )

    io.check_disk()
    with multiprocessing.Pool(args.workers) as p:
        for _ in tqdm.tqdm(
            star_imap_unordered(p, _write_one, [dict(rec=r) for r in selected]),
            total=len(selected),
        ):
            pass

    sel_counts: Counter = Counter()
    for r in selected:
        for c in r["classes_present"]:
            sel_counts[c] += 1
    class_counts = {CLASSES[i][0]: sel_counts.get(i, 0) for i in range(N_CLASSES)}
    print("selected per-class tile counts:", class_counts)

    io.write_dataset_metadata(
        SLUG,
        {
            "dataset": SLUG,
            "name": "MapBiomas Brasil (annual LULC)",
            "task_type": "classification",
            "source": "MapBiomas",
            "license": "CC-BY-SA-4.0",
            "provenance": {
                "url": "https://brasil.mapbiomas.org/en/downloads/",
                "have_locally": False,
                "annotation_method": "derived-product (MapBiomas Collection 9, 30 m, Landsat)",
                "cog": HTTP_COG_URL,
                "year": YEAR,
                "native_resolution_m": 30,
                "source_value_legend": {
                    str(k): v for k, v in sorted(CODE_TO_NAME.items())
                },
                "class_collapse": {str(k): v for k, v in sorted(SRC_TO_ID.items())},
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
                "Bounded-tile sampling of the MapBiomas Brasil Collection 9 30 m annual "
                f"LULC national COG for {YEAR}: {len(REGIONS)} spatially-distributed 0.6-deg "
                "windows across the Amazon, Cerrado, Atlantic Forest, Caatinga, Pantanal and "
                "Pampa biomes plus targeted mangrove/aquaculture coasts, mining districts and "
                "coffee/citrus/silviculture belts, range-read from the national COG (the "
                "~24.6 Gpx mosaic is never fully downloaded). Mostly-observed (>=90% mapped) "
                "22x22 native blocks were selected tiles-per-class balanced (rarest first) "
                "and reprojected from native ~30 m EPSG:4326 to local UTM at 10 m with NEAREST "
                "resampling (categorical). The deep MapBiomas legend (per-crop subclasses) is "
                f"collapsed to {N_CLASSES} level-1/level-2 classes separable at 30 m; source "
                "0/27 (no-data/not observed) and unmapped codes are nodata=255. Static "
                f"{YEAR} label: change_time=null, 1-year time range on {YEAR}."
            ),
        },
    )
    print("done")


if __name__ == "__main__":
    multiprocessing.set_start_method("forkserver", force=True)
    main()
