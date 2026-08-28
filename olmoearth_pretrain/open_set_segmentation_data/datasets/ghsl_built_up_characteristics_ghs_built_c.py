"""Process GHSL Built-up Characteristics (GHS-BUILT-C) into open-set-segmentation labels.

Source: EC JRC / GHSL GHS-BUILT-C R2023A, the **Morphological Settlement Zone (MSZ)**
product. This is a GLOBAL derived-product raster at **10 m** native resolution in Mollweide
(ESRI:54009) that classifies each 10 m cell inside/around settlements into a morphological
"built-up characteristics" typology combining (a) open-space types, (b) residential vs
non-residential built spaces, and (c) building-height density bins:

    0        outside settlement zone / no data      -> mapped to nodata (255)
    1        open space, low vegetation (NDVI<=0.3)
    2        open space, medium vegetation (0.3<NDVI<=0.5)
    3        open space, high vegetation (NDVI>0.5)
    4        open space, water surfaces
    5        open space, road surfaces
    11..15   built, RESIDENTIAL, by height (<=3 / 3-6 / 6-15 / 15-30 / >30 m)
    21..25   built, NON-RESIDENTIAL, by height (<=3 / 3-6 / 6-15 / 15-30 / >30 m)
    255      no data

The 15 meaningful morphological codes (1-5, 11-15, 21-25) become classification class ids
0..14 (see CLASSES). Source codes 0 and 255 (outside the settlement zone / no data) are
mapped to nodata/ignore (255) -- they are not a defined "built-up characteristic".

This is a GLOBAL derived-product map distributed as 375 land tiles (1000 km x 1000 km,
100000x100000 px, ~4-430 MB LZW each). Per the spec (5) we do BOUNDED-TILE sampling: we
download only 29 tiles covering a representative global spread of settlement types
(megacities with high-rise cores on every continent, plus rural/natural areas), then draw a
tiles-per-class balanced set of 64x64 @10 m windows (<=1000 tiles/class, prioritizing rare
classes). Unlike the sibling GHS-SMOD (1 km upsampled), GHS-BUILT-C is natively 10 m, so
each 640 m window carries genuine per-pixel morphological structure (buildings, roads,
vegetation) -- a real dense-segmentation label. Windows are cut in local UTM and reprojected
from Mollweide with NEAREST resampling (categorical). Time range is a 1-year window anchored
on the product epoch (2018).

Run:  python3 -m olmoearth_pretrain.open_set_segmentation_data.datasets.ghsl_built_up_characteristics_ghs_built_c
"""

import argparse
import multiprocessing
from collections import Counter
from typing import Any

import numpy as np
import rasterio
import tqdm
from pyproj import Transformer
from rasterio.warp import Resampling, reproject, transform_bounds
from rasterio.windows import from_bounds
from rslearn.utils.mp import star_imap_unordered
from rslearn.utils.raster_format import get_transform_from_projection_and_bounds

from olmoearth_pretrain.open_set_segmentation_data import download, io, sampling

SLUG = "ghsl_built_up_characteristics_ghs_built_c"

# GHS-BUILT-C R2023A is computed for the 2018 epoch (the only epoch of this product).
YEAR = 2018

PER_CLASS = 1000
TILE = 64
SRC_CRS = "ESRI:54009"  # World Mollweide
SRC_NODATA = 255
# candidate window centres to draw per class per tile before balancing
CAND_PER_CLASS_PER_TILE = 1200
# keep candidate centres this far (px) from tile edges so a 64x64 window stays in-tile
EDGE_MARGIN = 96

# GHSL tiling schema (10 m Mollweide): tile R{row}_C{col}, each 1,000,000 m wide.
# R1_C1 top-left corner = (-18041000, 9000000). Verified against tile headers.
TILE_W_M = 1_000_000.0
ORIGIN_X = -18041000.0
ORIGIN_Y = 9_000_000.0

BASE_URL = (
    "https://jeodpp.jrc.ec.europa.eu/ftp/jrc-opendata/GHSL/GHS_BUILT_C_GLOBE_R2023A/"
    "GHS_BUILT_C_MSZ_E2018_GLOBE_R2023A_54009_10/V1-0/tiles/"
    "GHS_BUILT_C_MSZ_E2018_GLOBE_R2023A_54009_10_V1_0_R{r}_C{c}.zip"
)
INNER_TIF = "GHS_BUILT_C_MSZ_E2018_GLOBE_R2023A_54009_10_V1_0_R{r}_C{c}.tif"

# Representative global spread of GHSL tiles (R, C) -> the regions they were chosen to cover.
# Chosen for diversity of settlement morphology (high-rise megacities on every continent,
# plus rural / natural / arid areas so open-space classes and short/low-rise built classes
# are all represented). Verified to exist on the JRC FTP.
TILES: dict[tuple[int, int], str] = {
    (3, 19): "London / N Europe",
    (3, 21): "Moscow",
    (3, 24): "Siberia (rural)",
    (4, 19): "Paris / W Europe",
    (5, 8): "Los Angeles",
    (5, 11): "US Midwest (rural)",
    (5, 12): "New York City",
    (5, 21): "Istanbul",
    (5, 31): "Tokyo",
    (6, 21): "Cairo",
    (6, 24): "Dubai / Gulf",
    (6, 26): "Delhi",
    (6, 30): "Shanghai",
    (7, 9): "Mexico City",
    (7, 26): "Mumbai / India",
    (7, 29): "Hong Kong / Shenzhen",
    (8, 19): "Sahel edge (arid)",
    (9, 11): "Bogota",
    (9, 19): "Lagos",
    (9, 29): "Singapore",
    (10, 13): "Amazon (rural)",
    (10, 20): "Kinshasa",
    (10, 22): "Nairobi",
    (10, 29): "Jakarta",
    (11, 11): "Lima",
    (12, 14): "Sao Paulo",
    (13, 21): "Johannesburg",
    (13, 31): "Australia outback (rural)",
    (14, 32): "Sydney",
}

# Class id (position) -> (name, description, [source codes]).
CLASSES: list[tuple[str, str, list[int]]] = [
    (
        "open_space_low_vegetation",
        "Open space within the settlement zone with low vegetation cover (NDVI <= 0.3): bare "
        "soil, sparse vegetation, or hard surfaces that are not roads or water.",
        [1],
    ),
    (
        "open_space_medium_vegetation",
        "Open space within the settlement zone with medium vegetation cover (0.3 < NDVI <= 0.5).",
        [2],
    ),
    (
        "open_space_high_vegetation",
        "Open space within the settlement zone with high/dense vegetation cover (NDVI > 0.5): "
        "parks, dense green space, tree cover.",
        [3],
    ),
    (
        "open_space_water",
        "Open space within the settlement zone classified as permanent water surface.",
        [4],
    ),
    (
        "open_space_road",
        "Open space within the settlement zone classified as road / paved transport surface.",
        [5],
    ),
    (
        "residential_h_le_3m",
        "Built-up residential space with average building height <= 3 m (single-storey).",
        [11],
    ),
    (
        "residential_h_3_6m",
        "Built-up residential space with average building height 3-6 m (~1-2 storeys).",
        [12],
    ),
    (
        "residential_h_6_15m",
        "Built-up residential space with average building height 6-15 m (~2-5 storeys).",
        [13],
    ),
    (
        "residential_h_15_30m",
        "Built-up residential space with average building height 15-30 m (mid-rise).",
        [14],
    ),
    (
        "residential_h_gt_30m",
        "Built-up residential space with average building height > 30 m (high-rise).",
        [15],
    ),
    (
        "nonresidential_h_le_3m",
        "Built-up non-residential space (commercial/industrial/institutional) with average "
        "building height <= 3 m.",
        [21],
    ),
    (
        "nonresidential_h_3_6m",
        "Built-up non-residential space with average building height 3-6 m.",
        [22],
    ),
    (
        "nonresidential_h_6_15m",
        "Built-up non-residential space with average building height 6-15 m.",
        [23],
    ),
    (
        "nonresidential_h_15_30m",
        "Built-up non-residential space with average building height 15-30 m (mid-rise).",
        [24],
    ),
    (
        "nonresidential_h_gt_30m",
        "Built-up non-residential space with average building height > 30 m (high-rise).",
        [25],
    ),
]

# LUT: source code (0..255) -> class id, default nodata (255).
LUT = np.full(256, io.CLASS_NODATA, dtype=np.uint8)
for _cid, (_n, _d, _codes) in enumerate(CLASSES):
    for _code in _codes:
        LUT[_code] = _cid


def tile_zip_path(r: int, c: int):
    return io.raw_dir(SLUG) / f"GHS_BUILT_C_MSZ_R2023A_R{r}_C{c}.zip"


def tile_vsi(r: int, c: int) -> str:
    zp = tile_zip_path(r, c)
    return f"/vsizip/{zp.path}/{INNER_TIF.format(r=r, c=c)}"


def tile_origin(r: int, c: int) -> tuple[float, float]:
    """Mollweide (left, top) of tile R{r}_C{c}."""
    return (ORIGIN_X + (c - 1) * TILE_W_M, ORIGIN_Y - (r - 1) * TILE_W_M)


def _download_one(rc: tuple[int, int]) -> None:
    r, c = rc
    io.check_disk()
    download.download_http(BASE_URL.format(r=r, c=c), tile_zip_path(r, c))


def download_source(workers: int) -> None:
    """Download all representative tile zips (idempotent, disk-guarded, parallel)."""
    args = [dict(rc=rc) for rc in TILES]
    with multiprocessing.Pool(min(workers, len(args))) as p:
        for _ in tqdm.tqdm(
            star_imap_unordered(p, _download_one, args),
            total=len(args),
            desc="download",
        ):
            pass


def _scan_tile(rc: tuple[int, int]) -> list[dict[str, Any]]:
    """Load a full tile, draw candidate window centres per class, tag classes_present.

    Returns lightweight candidate dicts (no arrays). classes_present is computed from the
    native (Mollweide) 64x64 window around each centre; it closely matches the reprojected
    UTM tile (same 10 m resolution, nearest resampling).
    """
    r, c = rc
    with rasterio.open(tile_vsi(r, c)) as ds:
        raw = ds.read(1)  # (H, W) uint8, full tile
    ids = LUT[raw]  # class-id array, 255 = nodata
    del raw
    h, w = ids.shape
    left, top = tile_origin(r, c)
    rng = np.random.default_rng(1000 * r + c)
    tf = Transformer.from_crs(SRC_CRS, "EPSG:4326", always_xy=True)

    cands: list[dict[str, Any]] = []
    for cid in range(len(CLASSES)):
        idx = np.flatnonzero(ids.reshape(-1) == cid)
        if idx.size == 0:
            continue
        if idx.size > CAND_PER_CLASS_PER_TILE:
            idx = rng.choice(idx, CAND_PER_CLASS_PER_TILE, replace=False)
        rows = (idx // w).astype(np.int64)
        cols = (idx % w).astype(np.int64)
        keep = (
            (rows >= EDGE_MARGIN)
            & (rows < h - EDGE_MARGIN)
            & (cols >= EDGE_MARGIN)
            & (cols < w - EDGE_MARGIN)
        )
        rows, cols = rows[keep], cols[keep]
        if rows.size == 0:
            continue
        mx = left + (cols + 0.5) * io.RESOLUTION
        my = top - (rows + 0.5) * io.RESOLUTION
        lons, lats = tf.transform(mx.tolist(), my.tolist())
        half = TILE // 2
        for row, col, lon, lat in zip(rows.tolist(), cols.tolist(), lons, lats):
            win = ids[row - half : row - half + TILE, col - half : col - half + TILE]
            present = [int(v) for v in np.unique(win) if v != io.CLASS_NODATA]
            cands.append(
                {
                    "r": r,
                    "c": c,
                    "row": int(row),
                    "col": int(col),
                    "lon": float(lon),
                    "lat": float(lat),
                    "classes_present": present,
                    "source_id": f"R{r}_C{c}_r{row}_c{col}",
                }
            )
    del ids
    return cands


# ---- writer: cache open source datasets per process ----
_DS: dict[tuple[int, int], Any] = {}


def _src(r: int, c: int):
    key = (r, c)
    if key not in _DS:
        _DS[key] = rasterio.open(tile_vsi(r, c))
    return _DS[key]


def _write_one(rec: dict[str, Any]) -> None:
    sample_id = rec["sample_id"]
    if (io.locations_dir(SLUG) / f"{sample_id}.tif").exists():
        return
    lon, lat = rec["lon"], rec["lat"]
    dst_proj, col, row = io.lonlat_to_utm_pixel(lon, lat)
    bounds = io.centered_bounds(col, row, TILE, TILE)
    dst_transform = get_transform_from_projection_and_bounds(dst_proj, bounds)

    # Geographic extent (metres) of the UTM tile -> Mollweide read window.
    xs = [bounds[0] * io.RESOLUTION, bounds[2] * io.RESOLUTION]
    ys = [bounds[1] * -io.RESOLUTION, bounds[3] * -io.RESOLUTION]
    left, right = min(xs), max(xs)
    bottom, top = min(ys), max(ys)
    l2, b2, r2, t2 = transform_bounds(dst_proj.crs, SRC_CRS, left, bottom, right, top)
    pad = 300.0  # ~30 native cells of margin

    ds = _src(rec["r"], rec["c"])
    win = from_bounds(l2 - pad, b2 - pad, r2 + pad, t2 + pad, ds.transform)
    src = ds.read(1, window=win, boundless=True, fill_value=SRC_NODATA)
    win_transform = ds.window_transform(win)

    dst_raw = np.full((TILE, TILE), SRC_NODATA, dtype=np.uint8)
    reproject(
        source=src,
        destination=dst_raw,
        src_transform=win_transform,
        src_crs=ds.crs,
        dst_transform=dst_transform,
        dst_crs=dst_proj.crs,
        resampling=Resampling.nearest,
        src_nodata=SRC_NODATA,
        dst_nodata=SRC_NODATA,
    )
    out = LUT[dst_raw]  # class ids, 255 nodata (also maps source 0 -> 255)

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
    parser.add_argument("--scan-workers", type=int, default=20)
    args = parser.parse_args()

    io.check_disk()
    from olmoearth_pretrain.open_set_segmentation_data import manifest

    manifest.write_registry_entry(SLUG, "in_progress")

    print(f"Downloading {len(TILES)} representative GHS-BUILT-C tiles...")
    download_source(args.workers)
    io.check_disk()

    print("Scanning tiles for class-tagged candidate windows...")
    cands: list[dict[str, Any]] = []
    with multiprocessing.Pool(args.scan_workers) as p:
        for res in tqdm.tqdm(
            star_imap_unordered(p, _scan_tile, [dict(rc=rc) for rc in TILES]),
            total=len(TILES),
            desc="scan",
        ):
            cands.extend(res)
    print(f"  {len(cands)} candidate windows across {len(TILES)} tiles")
    cand_class_counts = Counter()
    for rec in cands:
        for cc in set(rec["classes_present"]):
            cand_class_counts[cc] += 1
    print(
        "  candidate windows containing each class:",
        {i: cand_class_counts.get(i, 0) for i in range(len(CLASSES))},
    )

    print("Tiles-per-class balanced selection (rarest first)...")
    selected = sampling.select_tiles_per_class(
        cands, classes_key="classes_present", per_class=PER_CLASS
    )
    for i, rec in enumerate(selected):
        rec["sample_id"] = f"{i:06d}"
    print(f"  selected {len(selected)} windows")

    io.check_disk()
    with multiprocessing.Pool(args.workers) as p:
        for _ in tqdm.tqdm(
            star_imap_unordered(p, _write_one, [dict(rec=r) for r in selected]),
            total=len(selected),
            desc="write",
        ):
            pass

    # Class-balance report over the SELECTED tiles (a tile counts for every class in it).
    sel_class_counts: Counter = Counter()
    for rec in selected:
        for cc in set(rec["classes_present"]):
            sel_class_counts[cc] += 1
    class_counts = {
        name: sel_class_counts.get(i, 0) for i, (name, _d, _c) in enumerate(CLASSES)
    }
    print("selected tiles per class:", class_counts)

    io.write_dataset_metadata(
        SLUG,
        {
            "dataset": SLUG,
            "name": "GHSL Built-up Characteristics (GHS-BUILT-C)",
            "task_type": "classification",
            "source": "EC JRC / GHSL",
            "license": "open + attribution (CC BY 4.0)",
            "provenance": {
                "url": "https://human-settlement.emergency.copernicus.eu/datasets.php",
                "have_locally": False,
                "annotation_method": "derived-product (GHSL morphological settlement zone) + reference validation",
                "product": "GHS_BUILT_C_MSZ_E2018_GLOBE_R2023A_54009_10",
                "epoch": YEAR,
                "native_resolution_m": 10,
                "native_crs": SRC_CRS,
            },
            "sensors_relevant": ["sentinel2", "sentinel1", "landsat"],
            "classes": [
                {"id": i, "name": name, "description": desc}
                for i, (name, desc, _c) in enumerate(CLASSES)
            ],
            "nodata_value": io.CLASS_NODATA,
            "num_samples": len(selected),
            "tiles_per_class": class_counts,
            "regions_sampled": sorted(set(TILES.values())),
            "notes": (
                "Bounded-tile sampling of the GLOBAL JRC GHSL GHS-BUILT-C R2023A Morphological "
                f"Settlement Zone (MSZ) product, 10 m native, epoch {YEAR}. 29 of the 375 land "
                "tiles were downloaded, covering a representative global spread of settlement "
                "morphologies (high-rise megacity cores on every continent + rural/arid/natural "
                "areas). Tiles-per-class balanced selection of up to 1000 tiles/class (rarest "
                "class first) over 64x64 @10 m windows cut in local UTM and reprojected from "
                "Mollweide with NEAREST resampling (categorical). Source codes 1-5 (open-space "
                "types), 11-15 (residential by height) and 21-25 (non-residential by height) map "
                "to class ids 0-14; source 0 (outside settlement zone) and 255 map to nodata "
                "(255). Because the product is natively 10 m, each 640 m window carries genuine "
                "per-pixel morphological structure (a real dense-segmentation label), unlike the "
                "sibling GHS-SMOD (1 km upsampled). Rare non-residential height classes are "
                "sparse; kept per spec 5 (downstream assembly filters too-small classes)."
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
