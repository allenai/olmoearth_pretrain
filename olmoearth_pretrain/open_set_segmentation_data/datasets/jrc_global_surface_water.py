"""Process JRC Global Surface Water into open-set-segmentation label patches.

Source: EC JRC Global Surface Water Explorer (Pekel et al. 2016, Nature;
https://global-surface-water.appspot.com/). A 30 m derived-product mapping of surface
water dynamics over 1984-2021 (v1.4), distributed as 10x10-degree GeoTIFF tiles in
EPSG:4326 on Google Cloud Storage. We use the **Seasonality** product, whose per-pixel
value is the number of months surface water was present in the reference year:

    0        = no surface water (land)
    1 .. 11  = seasonal water (present 1-11 months)
    12       = permanent water (present all 12 months)

These map exactly to the manifest's three classes (permanent water / seasonal water /
no water). We treat the seasonality state as a per-pixel **classification** label
(task_type=classification) with a 1-year time range anchored on the product reference
year 2021 (within the manifest range 2016-2021; permanent water is temporally stable).

This is a HUGE global product, so per the spec (§5, large global derived-product) we do
BOUNDED-TILE sampling: download a handful of representative **interior continental** tiles
across diverse biomes/hemispheres (Amazon, Congo, W Siberia, Canada, Ganges, Australia,
Sahel, E Europe), scan non-overlapping ~64px-footprint windows, and use tiles-per-class
balanced selection (rarest class first) to reach up to 1000 tiles/class. Interior tiles
are chosen deliberately: JRC GSW masks the ocean to value 0 (== "no water"), so coastal
tiles would mislabel open ocean as land; interior tiles make value 0 == genuine dry land.

Native 30 m EPSG:4326 windows are reprojected to a local UTM projection at 10 m with
nearest resampling (categorical labels).

Tile URL (GCS public bucket):
    https://storage.googleapis.com/global-surface-water/downloads2021/seasonality/
        seasonality_<lonLabel>_<latLabel>v1_4_2021.tif
where <lonLabel>_<latLabel> is the tile's NW-corner label (e.g. 20E_0N, 70W_0N, 130E_20S).

Run:  python3 -m olmoearth_pretrain.open_set_segmentation_data.datasets.jrc_global_surface_water
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

SLUG = "jrc_global_surface_water"

# Product reference year (the seasonality tiles are named ..._2021); within manifest 2016-2021.
YEAR = 2021

PER_CLASS = 1000
BLOCK = 22  # native (30 m) block ~= 610 m ~= a 64 px @ 10 m UTM tile footprint
TILE = 64
PRESENT_FRAC = 0.05  # a class is "present" in a window if it covers >= 5% of the block
WATER_MIN = 0.10  # a water window must be >= 10% water (high-confidence water signal)

BASE_URL = (
    "https://storage.googleapis.com/global-surface-water/downloads2021/seasonality"
)

# Representative INTERIOR 10x10-deg tiles (NW-corner label: <westLon><E|W>_<northLat><N|S>).
# Chosen for diverse biomes/hemispheres and abundant inland water, and to avoid ocean
# (which JRC GSW masks to value 0 == "no water").
TILES = {
    "20E_0N": "Congo Basin interior (20-30E, 0-10S) - rivers, wetlands",
    "70W_0N": "Central Amazon (70-60W, 0-10S) - Solimoes floodplain, rivers",
    "60E_70N": "W Siberia (60-70E, 60-70N) - Ob wetlands, thermokarst lakes",
    "110W_60N": "Canadian prairies/shield (110-100W, 50-60N) - lakes",
    "80E_30N": "Ganges/Himalaya foreland (80-90E, 20-30N) - seasonal floodplain",
    "130E_20S": "Australia interior (130-140E, 20-30S) - Lake Eyre basin, ephemeral lakes",
    "10W_20N": "Niger inland delta / Sahel (10W-0, 10-20N) - seasonal water",
    "20E_50N": "E Europe (20-30E, 40-50N) - lakes, rivers, reservoirs",
}

# Manifest class order -> id. Descriptions from the JRC GSW product definition (Pekel 2016).
CLASSES = [
    (
        "no water",
        "Land or other non-water surface: no surface water detected in any month of the "
        "reference year (JRC GSW Seasonality value 0). NB: JRC GSW masks the ocean to 0; only "
        "interior continental tiles are sampled here so value 0 corresponds to genuine dry land.",
    ),
    (
        "seasonal water",
        "Seasonal surface water: water present 1-11 months of the year (JRC GSW Seasonality "
        "values 1-11) - floodplains, ephemeral/seasonal lakes, and rivers with intra-annual "
        "variation in extent.",
    ),
    (
        "permanent water",
        "Permanent surface water: water present all 12 months of the year (JRC GSW Seasonality "
        "value 12) - perennial lakes, reservoirs and large rivers.",
    ),
]

# Sentinel used for out-of-source fill (raw seasonality is only 0..12, so 255 is safe).
SRC_FILL = 255


def tile_url(tile: str) -> str:
    return f"{BASE_URL}/seasonality_{tile}v1_4_{YEAR}.tif"


def raw_tile_path(tile: str):
    return io.raw_dir(SLUG) / f"JRC_GSW_seasonality_{YEAR}_{tile}.tif"


def _map_seasonality(v: np.ndarray) -> np.ndarray:
    """Map raw seasonality (0..12, SRC_FILL) -> class id (0 no-water, 1 seasonal, 2 permanent,
    CLASS_NODATA outside-source).
    """
    out = np.full(v.shape, io.CLASS_NODATA, np.uint8)
    out[v == 0] = 0
    out[(v >= 1) & (v <= 11)] = 1
    out[v == 12] = 2
    return out


def download_tiles() -> None:
    """Download the chosen seasonality tiles (idempotent, disk-guarded)."""
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
    high-confidence. Each record lists classes_present so tiles-per-class selection can
    prioritize rare classes.
    """
    path = str(raw_tile_path(tile))
    with rasterio.open(path) as ds:
        arr = ds.read(1)
        st = ds.transform
    h, w = arr.shape
    nby, nbx = h // BLOCK, w // BLOCK
    a = arr[: nby * BLOCK, : nbx * BLOCK].reshape(nby, BLOCK, nbx, BLOCK)
    denom = float(BLOCK * BLOCK)
    # per-class pixel fractions in each block
    f_no = (a == 0).sum(axis=(1, 3)).astype(np.float32) / denom
    f_seas = ((a >= 1) & (a <= 11)).sum(axis=(1, 3)).astype(np.float32) / denom
    f_perm = (a == 12).sum(axis=(1, 3)).astype(np.float32) / denom
    f_water = f_seas + f_perm

    land = f_water == 0.0
    water = f_water >= WATER_MIN
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
        if f_no[br, bc] >= PRESENT_FRAC:
            present.append(0)
        if f_seas[br, bc] >= PRESENT_FRAC:
            present.append(1)
        if f_perm[br, bc] >= PRESENT_FRAC:
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
    out = _map_seasonality(dst)

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

    print("Downloading JRC GSW seasonality tiles...")
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

    # Tiles-per-class balanced: rarest class (seasonal water) first, <= PER_CLASS/class.
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
            "name": "JRC Global Surface Water",
            "task_type": "classification",
            "source": "EC JRC",
            "license": "open (free with attribution; Copernicus / JRC open data)",
            "provenance": {
                "url": "https://global-surface-water.appspot.com/",
                "have_locally": False,
                "annotation_method": "derived-product (JRC GSW Seasonality, Landsat 1984-2021)",
                "citation": "Pekel et al. 2016, Nature, doi:10.1038/nature20584",
                "product": "Seasonality v1.4",
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
                "Bounded-tile sampling of the global JRC GSW Seasonality product: "
                f"{len(TILES)} representative INTERIOR 10x10-deg tiles across diverse biomes "
                "and hemispheres, reference year 2021. Seasonality value -> class: 0 -> no "
                "water, 1-11 -> seasonal water, 12 -> permanent water. Non-overlapping "
                "~64px-footprint windows selected tiles-per-class (rarest first); water "
                "windows require >=10% water pixels (high-confidence), land windows are pure "
                "no-water. Reprojected from native 30 m EPSG:4326 to local UTM at 10 m with "
                "nearest resampling. Interior tiles chosen because JRC GSW masks the ocean to "
                "value 0; this makes value 0 correspond to genuine dry land. The manifest also "
                "references validation/reference points; those are not published as a "
                "downloadable tile set on the GSW portal and were not used - the derived "
                "Seasonality raster (an expert-validated product) is used directly."
            ),
        },
    )
    print("done")


if __name__ == "__main__":
    multiprocessing.set_start_method("forkserver", force=True)
    main()
