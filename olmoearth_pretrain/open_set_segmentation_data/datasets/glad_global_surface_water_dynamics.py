"""Process GLAD Global Surface Water Dynamics into open-set-segmentation label patches.

Source: UMD GLAD "Global surface water dynamics 1999-2021/2025" (Pickens et al. 2020,
Remote Sensing of Environment 243, 111792; https://glad.umd.edu/dataset/global-surface-
water-dynamics). A 30 m derived-product mapping of inland surface-water dynamics from the
full Landsat archive, distributed as 10x10-degree uint8 GeoTIFF tiles in EPSG:4326 on a
public Google Cloud Storage bucket (Collection 2, Provisional v2.0). License: CC-BY 4.0.

Tile URL scheme (from the product download page's JS):
    https://storage.googleapis.com/earthenginepartners-hansen/waterC2/<LAT>_<LON>/<FILE>.tif
where <LAT> is the tile's top-left latitude zero-padded to 3 chars (e.g. 00N, 40N, 20S)
and <LON> the top-left longitude padded to 4 (e.g. 020E, 070W). Layers (<FILE>) include
per-year annual water percent (``<YYYY>_percent``, 1999-2025), the multi-year interannual
``dynamic_classes_99_25``, monthly means, etc.

WHY THE ANNUAL WATER PERCENT LAYER (and how classes map)
--------------------------------------------------------
The manifest lists five classes: stable water, seasonal water, water gain, water loss,
land. "Water gain" and "water loss" are **multi-year change classes** defined over the
whole 1999-2021 period with NO precise event date, so they FAIL the ~1-2 month change-
timing rule (spec §5) and cannot be encoded as dated change labels. We therefore drop
gain/loss and build a per-pixel **static classification** from the per-year ``annual
water percent`` layer, which gives exactly the within-year (intra-annual) water state for
a single Sentinel-era reference year:

    percent == 0     -> land            (no surface water in any observation that year)
    percent 1..99    -> seasonal water  (intra-annual variation: present some observations)
    percent == 100   -> stable water    (water in every valid observation that year)
    255              -> nodata

Seasonal (intra-annual) water is a valid within-year class; "stable water" here is the
per-year permanent-water state (water all year), consistent with the manifest's stable-
water notion. task_type = classification, 1-year time range anchored on the reference year.

We evaluated the MANUAL time-series reference sample
(https://glad.geog.umd.edu/timeSeriesReference/timeSeriesSample.zip): it is georeferenced
(600 ~30 m single-pixel polygons, EPSG:4326) but tiny and its per-point ``Stratum`` field
is the map class the point was drawn from (a multi-year stratum incl. gain/loss), not a
clean per-year interpretation. It cannot supply balanced static per-year classes, so we
use the derived annual-percent raster directly (an expert-validated product), sampling only
high-confidence windows -- the same choice made for jrc_global_surface_water.

Because this is a HUGE global product (each tile is 40000x40000 px), we do BOUNDED-TILE
sampling (spec §5): a handful of representative INTERIOR continental 10x10-deg tiles across
diverse biomes/hemispheres, scanned for non-overlapping ~64px-footprint windows and
selected tiles-per-class (rarest first) up to 1000 tiles/class. Interior tiles are used so
value 0 corresponds to genuine dry land (GLAD maps inland water; ocean is not a target).
Native 30 m EPSG:4326 windows are reprojected to local UTM at 10 m with nearest resampling
(categorical labels).

Run:  python3 -m olmoearth_pretrain.open_set_segmentation_data.datasets.glad_global_surface_water_dynamics
"""

import argparse
import multiprocessing
from collections import Counter
from typing import Any

import numpy as np
import rasterio
import tqdm
from rasterio.warp import Resampling, reproject, transform_bounds
from rasterio.windows import from_bounds
from rslearn.utils.mp import star_imap_unordered
from rslearn.utils.raster_format import get_transform_from_projection_and_bounds

from olmoearth_pretrain.open_set_segmentation_data import download, io
from olmoearth_pretrain.open_set_segmentation_data.sampling import (
    select_tiles_per_class,
)

SLUG = "glad_global_surface_water_dynamics"

# Reference year (annual water percent layer), within manifest range 2016-2021.
YEAR = 2020

PER_CLASS = 1000
BLOCK = 22  # native (30 m) block ~= 610 m ~= a 64 px @ 10 m UTM tile footprint
TILE = 64
PRESENT_FRAC = 0.05  # a class is "present" in a window if it covers >= 5% of the block
WATER_MIN = 0.10  # a water window must be >= 10% water (high-confidence water signal)

BASE_URL = "https://storage.googleapis.com/earthenginepartners-hansen/waterC2"

# Representative INTERIOR 10x10-deg tiles (top-left corner label: <LAT>_<LON>).
# Chosen for diverse biomes/hemispheres and abundant inland water, and to avoid the ocean
# (GLAD maps INLAND water; interior tiles make value 0 correspond to genuine dry land).
TILES = {
    "00N_020E": "Congo Basin interior (20-30E, 0-10S) - rivers, wetlands",
    "00N_070W": "Central Amazon (70-60W, 0-10S) - Solimoes floodplain, rivers",
    "70N_060E": "W Siberia (60-70E, 60-70N) - Ob wetlands, thermokarst lakes",
    "60N_110W": "Canadian prairies/shield (110-100W, 50-60N) - lakes",
    "30N_080E": "Ganges/Himalaya foreland (80-90E, 20-30N) - seasonal floodplain",
    "20S_130E": "Australia interior (130-140E, 20-30S) - Lake Eyre basin, ephemeral lakes",
    "20N_010W": "Niger inland delta / Sahel (10W-0, 10-20N) - seasonal water",
    "50N_020E": "E Europe (20-30E, 40-50N) - lakes, rivers, reservoirs",
}

# Class order -> id. Descriptions from the GLAD product definition (Pickens et al. 2020).
CLASSES = [
    (
        "land",
        "Land or other non-water surface: annual water percent 0 (no surface water detected in "
        "any valid land/water observation of the reference year). NB: GLAD maps INLAND water; "
        "only interior continental tiles are sampled so value 0 corresponds to genuine dry land.",
    ),
    (
        "seasonal water",
        "Seasonal (intra-annual) surface water: annual water percent 1-99 (water present in some "
        "but not all valid observations of the reference year) - floodplains, ephemeral/seasonal "
        "lakes, and rivers with within-year variation in extent.",
    ),
    (
        "stable water",
        "Stable surface water: annual water percent 100 (water in every valid observation of the "
        "reference year) - perennial lakes, reservoirs and large rivers.",
    ),
]

# Sentinel used for out-of-source fill (raw percent is only 0..100 with nodata 255, so 255
# is safe as both source-nodata and out-of-window fill).
SRC_FILL = 255


def tile_url(tile: str) -> str:
    return f"{BASE_URL}/{tile}/{YEAR}_percent.tif"


def raw_tile_path(tile: str):
    return io.raw_dir(SLUG) / f"GLAD_waterC2_{YEAR}_percent_{tile}.tif"


def _map_percent(v: np.ndarray) -> np.ndarray:
    """Map raw annual water percent (0..100, 255 nodata) -> class id
    (0 land, 1 seasonal, 2 stable, CLASS_NODATA outside-source/nodata).
    """
    out = np.full(v.shape, io.CLASS_NODATA, np.uint8)
    out[v == 0] = 0
    out[(v >= 1) & (v <= 99)] = 1
    out[v == 100] = 2
    return out


def download_tiles() -> None:
    """Download the chosen annual-percent tiles (idempotent, disk-guarded)."""
    io.raw_dir(SLUG).mkdir(parents=True, exist_ok=True)
    for tile in TILES:
        io.check_disk()
        dst = raw_tile_path(tile)
        if dst.exists():
            print(f"  [skip] {dst.name} already present")
            continue
        print(f"  downloading {tile} -> {dst.name}")
        download.download_http(tile_url(tile), dst)


def _scan_tile(tile: str) -> list[dict[str, Any]]:
    """Scan non-overlapping BLOCKxBLOCK native windows; return candidate records.

    A window is a candidate if it is either pure land (0% water -> class [0]) or has a
    strong water signal (>= WATER_MIN water -> classes present at >= PRESENT_FRAC). Windows
    with weak/ambiguous water (0 < frac_water < WATER_MIN) are skipped to keep labels
    high-confidence. Windows containing any nodata are skipped. Each record lists
    classes_present so tiles-per-class selection can prioritize rare classes.
    """
    path = str(raw_tile_path(tile))
    with rasterio.open(path) as ds:
        arr = ds.read(1)
        st = ds.transform
    h, w = arr.shape
    nby, nbx = h // BLOCK, w // BLOCK
    a = arr[: nby * BLOCK, : nbx * BLOCK].reshape(nby, BLOCK, nbx, BLOCK)
    denom = float(BLOCK * BLOCK)
    f_nodata = (a == 255).sum(axis=(1, 3)).astype(np.float32) / denom
    f_land = (a == 0).sum(axis=(1, 3)).astype(np.float32) / denom
    f_seas = ((a >= 1) & (a <= 99)).sum(axis=(1, 3)).astype(np.float32) / denom
    f_stab = (a == 100).sum(axis=(1, 3)).astype(np.float32) / denom
    f_water = f_seas + f_stab

    clean = f_nodata == 0.0
    land = (f_water == 0.0) & clean
    water = (f_water >= WATER_MIN) & clean
    qual = land | water
    brs, bcs = np.nonzero(qual)

    cx = bcs * BLOCK + BLOCK / 2.0
    cy = brs * BLOCK + BLOCK / 2.0
    lons = st.c + cx * st.a
    lats = st.f + cy * st.e

    recs = []
    for br, bc, lon, lat in zip(
        brs.tolist(), bcs.tolist(), lons.tolist(), lats.tolist()
    ):
        present = []
        if f_land[br, bc] >= PRESENT_FRAC:
            present.append(0)
        if f_seas[br, bc] >= PRESENT_FRAC:
            present.append(1)
        if f_stab[br, bc] >= PRESENT_FRAC:
            present.append(2)
        if not present:
            continue
        recs.append(
            {
                "tile": tile,
                "lon": float(lon),
                "lat": float(lat),
                "classes_present": present,
                "source_id": f"{tile}_r{br}_c{bc}",
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

    xs = [bounds[0] * io.RESOLUTION, bounds[2] * io.RESOLUTION]
    ys = [bounds[1] * -io.RESOLUTION, bounds[3] * -io.RESOLUTION]
    left, right = min(xs), max(xs)
    bottom, top = min(ys), max(ys)
    l2, b2, r2, t2 = transform_bounds(
        dst_proj.crs, "EPSG:4326", left, bottom, right, top
    )
    pad = 0.003  # ~330 m margin so the tile is fully covered before nearest-resampling

    with rasterio.open(str(raw_tile_path(rec["tile"]))) as ds:
        win = from_bounds(l2 - pad, b2 - pad, r2 + pad, t2 + pad, ds.transform)
        src = ds.read(1, window=win, boundless=True, fill_value=SRC_FILL)
        win_transform = ds.window_transform(win)
        src_crs = ds.crs

    dst = np.full((TILE, TILE), SRC_FILL, np.uint8)
    reproject(
        source=src,
        destination=dst,
        src_transform=win_transform,
        src_crs=src_crs,
        dst_transform=dst_transform,
        dst_crs=dst_proj.crs,
        resampling=Resampling.nearest,
        src_nodata=SRC_FILL,
        dst_nodata=SRC_FILL,
    )
    out = _map_percent(dst)

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
        source_id=rec["source_id"],
        classes_present=present,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=64)
    args = parser.parse_args()

    io.check_disk()
    from olmoearth_pretrain.open_set_segmentation_data import manifest

    manifest.write_registry_entry(SLUG, "in_progress")

    print("Downloading GLAD annual water percent tiles...")
    download_tiles()
    io.check_disk()

    print("Scanning tiles for candidate windows...")
    with multiprocessing.Pool(min(len(TILES), 8)) as p:
        all_recs: list[dict[str, Any]] = []
        for recs in tqdm.tqdm(
            star_imap_unordered(p, _scan_tile, [dict(tile=t) for t in TILES]),
            total=len(TILES),
        ):
            all_recs.extend(recs)
    cand_counts = Counter()
    for r in all_recs:
        for c in r["classes_present"]:
            cand_counts[c] += 1
    print(
        f"scanned {len(all_recs)} candidate windows; per-class candidates: {dict(cand_counts)}"
    )

    # Tiles-per-class balanced: rarest class first, <= PER_CLASS/class.
    selected = select_tiles_per_class(
        all_recs, classes_key="classes_present", per_class=PER_CLASS
    )
    for i, r in enumerate(selected):
        r["sample_id"] = f"{i:06d}"
    print(f"selected {len(selected)} windows (tiles-per-class, <= {PER_CLASS}/class)")

    io.check_disk()
    with multiprocessing.Pool(args.workers) as p:
        for _ in tqdm.tqdm(
            star_imap_unordered(p, _write_one, [dict(rec=r) for r in selected]),
            total=len(selected),
        ):
            pass

    # Report tiles-per-class counts (a tile counts toward every class it contains).
    class_counts = {name: 0 for name, _ in CLASSES}
    for r in selected:
        for c in r["classes_present"]:
            class_counts[CLASSES[c][0]] += 1
    print("tiles-per-class counts:", class_counts)

    io.write_dataset_metadata(
        SLUG,
        {
            "dataset": SLUG,
            "name": "GLAD Global Surface Water Dynamics",
            "task_type": "classification",
            "source": "UMD GLAD",
            "license": "CC-BY 4.0 (free/public, attribution required)",
            "provenance": {
                "url": "https://glad.umd.edu/dataset/global-surface-water-dynamics",
                "have_locally": False,
                "annotation_method": (
                    "derived-product (GLAD annual water percent, Landsat 1999-2025 "
                    "Collection 2 v2.0); manual time-series reference sample evaluated but "
                    "not used (600 tiny stratum-labelled points, no clean per-year classes)"
                ),
                "citation": (
                    "Pickens, A.H., Hansen, M.C., Hancher, M., Stehman, S.V., Tyukavina, A., "
                    "Potapov, P., Marroquin, B., Sherani, Z., 2020. Mapping and sampling to "
                    "characterize global inland water dynamics from 1999 to 2018 with full "
                    "Landsat time-series. Remote Sensing of Environment 243, 111792. "
                    "https://doi.org/10.1016/j.rse.2020.111792"
                ),
                "product": "annual water percent (waterC2, Provisional v2.0)",
                "reference_year": YEAR,
                "tiles": TILES,
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
                "Bounded-tile sampling of the global GLAD surface-water-dynamics product: "
                f"{len(TILES)} representative INTERIOR 10x10-deg tiles across diverse biomes "
                f"and hemispheres, reference year {YEAR} (within manifest 2016-2021). Annual "
                "water percent -> class: 0 -> land, 1-99 -> seasonal water, 100 -> stable "
                "water. Multi-year 'water gain'/'water loss' change classes were DROPPED: "
                "they span the full 1999-2021 period with no precise event date and fail the "
                "~1-2 month change-timing rule (spec 5), so they are not encoded as dated "
                "change labels. Non-overlapping ~64px-footprint windows selected tiles-per-"
                "class (rarest first); water windows require >=10% water pixels (high-"
                "confidence), land windows are pure land; windows with any nodata skipped. "
                "Reprojected from native 30 m EPSG:4326 to local UTM at 10 m with nearest "
                "resampling. Interior tiles chosen because GLAD maps INLAND water, making "
                "value 0 correspond to genuine dry land. The manual time-series reference "
                "sample (timeSeriesSample.zip, 600 stratum-labelled ~30 m points) was "
                "evaluated but not used: too small and its class field is the multi-year map "
                "stratum, not a clean per-year interpretation."
            ),
        },
    )
    manifest.write_registry_entry(
        SLUG, "completed", task_type="classification", num_samples=len(selected)
    )
    print("done")


if __name__ == "__main__":
    multiprocessing.set_start_method("forkserver", force=True)
    main()
