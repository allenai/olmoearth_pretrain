"""Process Blue Ice Areas of Antarctica (Tollenaar et al., 2024) into label patches.

Source: Tollenaar, V. et al. "Where the White Continent is blue: deep learning locates
bare ice in Antarctica", Geophysical Research Letters (2024). Data on Zenodo record
10539933 (concept DOI 10.5281/zenodo.8333864), license CC-BY-4.0:

  smoothed_BIAs.zip        Antarctic-wide blue/bare-ice-area polygons (6644 polygons,
                           EPSG:3031). The smoothed, published product (CNN output on a
                           Landsat-8 / Sentinel-2 austral-summer composite, vectorized).
  handlabels_sq*.zip       Hand-outlined blue-ice polygons in 5 training squares (manual
                           reference; used to train/validate the CNN).
  BIA_map.nc (9.2 GB)      Per-pixel presence raster (NOT downloaded; polygons suffice).
  merged_bands_*.tif (5.8 GB) Input imagery composite (NOT needed; pretraining supplies
                           its own imagery).

This is a **binary dense segmentation** (label_type: polygons + dense_raster):
  0 = background       non-blue-ice Antarctic surface within the tile (snow / firn / rock /
                       other ice) — genuine, spatially-meaningful negatives adjacent to the
                       ice, not fabricated.
  1 = blue/bare ice    perennially wind-scoured bare/blue glacial ice, spectrally distinct.

We use the continent-wide ``smoothed_BIAs`` polygons (the actual published product) rather
than only the 5 hand-labelled squares, to get pan-Antarctic geographic diversity. Blue ice
is spectrally distinct and the product was validated against manual test squares, so polygon
interiors are high-confidence (spec section 4/5: derived-product maps -> prefer
high-confidence/homogeneous windows).

Tiling: blue-ice fields range from < 0.04 km2 to > 8000 km2 (most exceed one 640 m tile).
We sample candidate tile centres from within each polygon (roughly one per 640 m tile of
polygon area, capped per polygon), snap them to a 64-px grid in a local UTM/UPS projection,
and rasterize **all** blue-ice polygons intersecting each 64x64 (640 m) tile at 10 m. This
yields homogeneous interior tiles (all blue ice), boundary tiles (blue ice + background) and
background-dominant edge tiles. We stratify the selection across blue-ice-fraction buckets
(interior / edge / sliver) so both classes and both interior + boundary geometry are well
represented.

Time range: blue ice areas are persistent geomorphological features (perennial wind
ablation keeps them snow-free for decades), so this is a static label. We assign a single
representative 1-year Sentinel-era window (2019); the underlying composite spans ~2016-2024
and blue ice is stable across it (spec section 5, static labels). ``change_time`` is null.

Run (idempotent; skips already-written tiles):
  python3 -m olmoearth_pretrain.open_set_segmentation_data.datasets.blue_ice_areas_of_antarctica_tollenaar_et_al
"""

import argparse
import math
import multiprocessing
import random
from collections import Counter
from typing import Any

import numpy as np
import shapely
import tqdm
from pyproj import Transformer
from rasterio.crs import CRS
from rslearn.utils.geometry import Projection, STGeometry
from rslearn.utils.get_utm_ups_crs import get_utm_ups_projection
from rslearn.utils.mp import star_imap_unordered

from olmoearth_pretrain.open_set_segmentation_data import io
from olmoearth_pretrain.open_set_segmentation_data.download import download_zenodo
from olmoearth_pretrain.open_set_segmentation_data.rasterize import (
    geom_to_pixels,
    rasterize_shapes,
)
from olmoearth_pretrain.open_set_segmentation_data.sampling import balance_by_class

SLUG = "blue_ice_areas_of_antarctica_tollenaar_et_al"
NAME = "Blue Ice Areas of Antarctica (Tollenaar et al.)"
ZENODO_RECORD = "10539933"
ZENODO_DOI = "https://doi.org/10.5281/zenodo.8333864"

# Files to fetch from Zenodo: the continent-wide polygon product + the (small) hand-label
# and square shapefiles (kept for provenance / possible future higher-quality use).
ZENODO_FILES = [
    "smoothed_BIAs.zip",
    "handlabels_sq246.zip",
    "handlabels_sq264.zip",
    "handlabels_sq265.zip",
    "handlabels_sq278.zip",
    "handlabels_sq409.zip",
    "train_squares.zip",
    "validation_squares.zip",
    "test_squares.zip",
]

CID_BACKGROUND = 0
CID_BLUE_ICE = 1
CLASSES = [
    {
        "id": CID_BACKGROUND,
        "name": "background",
        "description": "Non-blue-ice Antarctic surface within the tile (snow, firn, "
        "exposed rock, or other/covered glacial ice) surrounding a blue-ice area. Genuine "
        "negatives adjacent to the ice, not fabricated.",
    },
    {
        "id": CID_BLUE_ICE,
        "name": "blue_bare_ice",
        "description": "Blue / bare glacial ice: perennially wind-scoured, snow-free ice "
        "exposed at the surface (spectrally distinct, bluish). From Tollenaar et al. 2024 "
        "CNN detections on a Landsat-8 / Sentinel-2 austral-summer composite, smoothed and "
        "vectorized (Antarctic-wide 'smoothed_BIAs' product).",
    },
]

SRC_CRS = CRS.from_epsg(3031)  # Antarctic Polar Stereographic (product's native CRS)
# Identity projection (1 px == 1 metre, no y-flip) so STGeometry treats the polygons'
# native 3031 metre coordinates directly (mirrors rslearn's WGS84_PROJECTION = 1,1).
P3031 = Projection(SRC_CRS, 1, 1)

TILE = io.MAX_TILE  # 64 px @ 10 m = 640 m
TILE_M = TILE * io.RESOLUTION  # 640 m
TILE_KM2 = (TILE_M / 1000.0) ** 2  # 0.4096 km^2 per tile
CAP_PER_POLY = 6  # max candidate tiles sampled from any single polygon
REP_YEAR = 2019  # representative Sentinel-era year (blue ice is persistent)

PER_BUCKET = 500  # -> up to 1500 tiles across {interior, edge, sliver}
SEED = 42

# ---- worker globals (loaded lazily; forkserver workers don't inherit parent memory) ----
_G: dict[str, Any] = {}


def _ensure_loaded() -> dict[str, Any]:
    if _G:
        return _G
    import geopandas as gpd
    from shapely import STRtree

    shp = io.raw_dir(SLUG) / "extracted" / "smoothed_BIAs.shp"
    gdf = gpd.read_file(shp.path)
    geoms = list(gdf.geometry.values)
    _G["geoms"] = geoms
    _G["tree"] = STRtree(geoms)
    _G["to_wgs84"] = Transformer.from_crs(3031, 4326, always_xy=True)
    return _G


def _sample_points(geom: Any, n: int, rng: random.Random) -> list[tuple[float, float]]:
    """Sample up to ``n`` points inside a polygon (rejection sampling in its bbox)."""
    minx, miny, maxx, maxy = geom.bounds
    pts: list[tuple[float, float]] = []
    tries = 0
    limit = n * 60
    while len(pts) < n and tries < limit:
        x = rng.uniform(minx, maxx)
        y = rng.uniform(miny, maxy)
        if geom.contains(shapely.Point(x, y)):
            pts.append((x, y))
        tries += 1
    if not pts:
        rp = geom.representative_point()
        pts = [(rp.x, rp.y)]
    return pts


def _candidate_keys(poly_idx: int) -> list[tuple[str, int, int]]:
    """Sample candidate 64-px tile keys (crs, x0, y0) from within one polygon."""
    g = _ensure_loaded()
    geom = g["geoms"][poly_idx]
    area_km2 = geom.area / 1e6
    n = min(CAP_PER_POLY, max(1, int(math.ceil(area_km2 / TILE_KM2))))
    rng = random.Random(SEED + poly_idx)
    keys: set[tuple[str, int, int]] = set()
    for x, y in _sample_points(geom, n, rng):
        lon, lat = g["to_wgs84"].transform(x, y)
        proj = get_utm_ups_projection(lon, lat, io.RESOLUTION, -io.RESOLUTION)
        p = STGeometry(P3031, shapely.Point(x, y), None).to_projection(proj).shp
        x0 = int(math.floor(p.x / TILE)) * TILE
        y0 = int(math.floor(p.y / TILE)) * TILE
        keys.add((proj.crs.to_string(), x0, y0))
    return list(keys)


def _rasterize_tile(crs_str: str, x0: int, y0: int) -> np.ndarray | None:
    """Rasterize all blue-ice polygons intersecting a tile into a (1,64,64) uint8 array."""
    g = _ensure_loaded()
    proj = Projection(CRS.from_string(crs_str), io.RESOLUTION, -io.RESOLUTION)
    bounds = (x0, y0, x0 + TILE, y0 + TILE)
    tile_box_px = shapely.box(*bounds)
    # Tile footprint in 3031 metres (for spatial-index query + geometry clipping).
    box_3031 = STGeometry(proj, tile_box_px, None).to_projection(P3031).shp
    clip_3031 = box_3031.buffer(30.0)  # small pad so edge geometry isn't lost
    idxs = g["tree"].query(box_3031)
    shapes: list[tuple[Any, int]] = []
    for i in idxs:
        geom = g["geoms"][int(i)]
        if not geom.intersects(box_3031):
            continue
        clipped = geom.intersection(clip_3031)
        if clipped.is_empty:
            continue
        pix = geom_to_pixels(clipped, P3031, proj)
        if pix.is_empty:
            continue
        shapes.append((pix, CID_BLUE_ICE))
    if not shapes:
        return None
    return rasterize_shapes(
        shapes, bounds, fill=CID_BACKGROUND, dtype="uint8", all_touched=False
    )


def _scan_tile(crs_str: str, x0: int, y0: int) -> dict[str, Any] | None:
    arr = _rasterize_tile(crs_str, x0, y0)
    if arr is None:
        return None
    blue_frac = float((arr == CID_BLUE_ICE).mean())
    if blue_frac <= 0.0:
        return None
    classes_present = sorted(int(v) for v in np.unique(arr))
    if blue_frac >= 0.85:
        bucket = "interior"
    elif blue_frac <= 0.15:
        bucket = "sliver"
    else:
        bucket = "edge"
    return {
        "crs": crs_str,
        "x0": x0,
        "y0": y0,
        "blue_frac": blue_frac,
        "frac_bucket": bucket,
        "classes_present": classes_present,
    }


def _write_tile(rec: dict[str, Any]) -> str:
    sample_id = rec["sample_id"]
    tif = io.locations_dir(SLUG) / f"{sample_id}.tif"
    if tif.exists():
        return "skip"
    arr = _rasterize_tile(rec["crs"], rec["x0"], rec["y0"])
    if arr is None:
        return "empty"
    proj = Projection(CRS.from_string(rec["crs"]), io.RESOLUTION, -io.RESOLUTION)
    bounds = (rec["x0"], rec["y0"], rec["x0"] + TILE, rec["y0"] + TILE)
    io.write_label_geotiff(SLUG, sample_id, arr, proj, bounds, nodata=io.CLASS_NODATA)
    io.write_sample_json(
        SLUG,
        sample_id,
        proj,
        bounds,
        io.year_range(REP_YEAR),
        change_time=None,
        source_id=f"smoothed_BIAs/tile_{rec['crs'].replace(':', '')}_{rec['x0']}_{rec['y0']}",
        classes_present=sorted(int(v) for v in np.unique(arr)),
    )
    return "written"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=64)
    args = parser.parse_args()

    io.check_disk()

    # --- download + extract the (small) polygon shapefiles ---
    raw = io.raw_dir(SLUG)
    raw.mkdir(parents=True, exist_ok=True)
    extracted = raw / "extracted"
    shp = extracted / "smoothed_BIAs.shp"
    if not shp.exists():
        print("downloading Zenodo shapefiles ...", flush=True)
        download_zenodo(ZENODO_RECORD, raw, filenames=ZENODO_FILES)
        import zipfile

        extracted.mkdir(parents=True, exist_ok=True)
        for z in raw.glob("*.zip"):
            with zipfile.ZipFile(z.path) as zf:
                for member in zf.namelist():
                    if member.startswith("__MACOSX") or member.endswith("/"):
                        continue
                    target = extracted / member.split("/")[-1]
                    with zf.open(member) as src, target.open("wb") as dst:
                        dst.write(src.read())
    with (raw / "SOURCE.txt").open("w") as f:
        f.write(
            "Blue Ice Areas of Antarctica - Tollenaar et al., GRL (2024).\n"
            f"Zenodo record {ZENODO_RECORD} ({ZENODO_DOI}), license CC-BY-4.0.\n"
            "Used: smoothed_BIAs.shp (continent-wide blue/bare-ice polygons, EPSG:3031).\n"
            "NOT downloaded: BIA_map.nc (9.2 GB per-pixel raster), "
            "merged_bands_composite*.tif (5.8 GB imagery) - polygons suffice and "
            "pretraining supplies its own imagery.\n"
        )

    io.check_disk()

    # --- scan phase 1: sample candidate tile keys from every polygon (parallel) ---
    g = _ensure_loaded()
    n_polys = len(g["geoms"])
    print(
        f"loaded {n_polys} smoothed blue-ice polygons; sampling candidate tiles ...",
        flush=True,
    )
    keys: set[tuple[str, int, int]] = set()
    with multiprocessing.Pool(args.workers) as p:
        for res in tqdm.tqdm(
            star_imap_unordered(
                p, _candidate_keys, [dict(poly_idx=i) for i in range(n_polys)]
            ),
            total=n_polys,
        ):
            keys.update(res)
    print(f"  {len(keys)} unique candidate tiles", flush=True)

    # --- scan phase 2: rasterize each unique tile to get class content (parallel) ---
    key_list = sorted(keys)
    records: list[dict[str, Any]] = []
    with multiprocessing.Pool(args.workers) as p:
        for res in tqdm.tqdm(
            star_imap_unordered(
                p,
                _scan_tile,
                [dict(crs_str=c, x0=x, y0=y) for (c, x, y) in key_list],
            ),
            total=len(key_list),
        ):
            if res is not None:
                records.append(res)
    bkt = Counter(r["frac_bucket"] for r in records)
    print(f"  {len(records)} blue-ice tiles; frac buckets: {dict(bkt)}", flush=True)

    # --- select: stratify across blue-ice-fraction buckets (interior/edge/sliver) ---
    selected = balance_by_class(
        records, "frac_bucket", per_class=PER_BUCKET, seed=SEED, total_cap=None
    )
    sel_bkt = Counter(r["frac_bucket"] for r in selected)
    print(f"selected {len(selected)} tiles; buckets: {dict(sel_bkt)}", flush=True)

    selected.sort(key=lambda r: (r["crs"], r["x0"], r["y0"]))
    for i, r in enumerate(selected):
        r["sample_id"] = f"{i:06d}"

    io.check_disk()

    # --- write phase (parallel) ---
    results: Counter = Counter()
    class_tile_counts: Counter = Counter()
    with multiprocessing.Pool(args.workers) as p:
        for res in tqdm.tqdm(
            star_imap_unordered(p, _write_tile, [dict(rec=r) for r in selected]),
            total=len(selected),
        ):
            results[res] += 1
    for r in selected:
        for c in r["classes_present"]:
            class_tile_counts[c] += 1
    print("write results:", dict(results), flush=True)
    print("class tile-appearance counts:", dict(class_tile_counts), flush=True)

    io.check_disk()
    io.write_dataset_metadata(
        SLUG,
        {
            "dataset": SLUG,
            "name": NAME,
            "task_type": "classification",
            "source": "Zenodo (Tollenaar et al., GRL 2024)",
            "license": "CC-BY-4.0",
            "provenance": {
                "url": ZENODO_DOI,
                "have_locally": False,
                "annotation_method": "derived (CNN on Landsat-8/Sentinel-2 composite), "
                "smoothed + vectorized; manual hand-labelled training squares available",
                "file_used": "smoothed_BIAs.shp",
            },
            "sensors_relevant": ["sentinel2", "sentinel1", "landsat"],
            "classes": CLASSES,
            "nodata_value": io.CLASS_NODATA,
            "num_samples": len(selected),
            "class_tile_counts": {
                str(k): v for k, v in sorted(class_tile_counts.items())
            },
            "sampling": {
                "per_frac_bucket": PER_BUCKET,
                "frac_bucket_counts": dict(sel_bkt),
                "cap_per_polygon": CAP_PER_POLY,
                "tile_size_px": TILE,
                "n_source_polygons": n_polys,
            },
            "time_range_rule": f"static persistent feature -> representative 1-year window {REP_YEAR}",
            "notes": (
                "Binary blue/bare-ice dense segmentation from Tollenaar et al. 2024 "
                "continent-wide 'smoothed_BIAs' polygons (6644 polygons, EPSG:3031). "
                "0=background (non-blue-ice Antarctic surface within tile), 1=blue/bare ice. "
                "Candidate 64x64 (640 m) tiles sampled from within polygons (<=6/polygon), "
                "snapped to a 64-px grid in local UTM/UPS at 10 m; all intersecting blue-ice "
                "polygons rasterized per tile. Selection stratified across blue-ice-fraction "
                "buckets {interior>=0.85, edge, sliver<=0.15} (<=500 each) for interior + "
                "boundary + background diversity. Blue ice is a persistent feature; static "
                "1-year window 2019 (composite spans ~2016-2024). Hand-labelled squares "
                "(handlabels_sq*.shp) are a higher-quality manual reference but cover only 5 "
                "regions; we use the validated continent-wide product for pan-Antarctic "
                "diversity (derived-product high-confidence interiors, spec section 4/5). "
                "Background is spatially meaningful within-tile terrain (not fabricated), so "
                "no separate far negatives were generated."
            ),
        },
    )
    print(f"done: {len(selected)} tiles")


if __name__ == "__main__":
    multiprocessing.set_start_method("forkserver", force=True)
    main()
