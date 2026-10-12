"""Process the Hi-MAG glacial-lake inventory (High Mountain Asia) into label patches.

Source: Chen, F. et al. "Annual 30 m dataset for glacial lakes in High Mountain Asia
from 2008 to 2017", Earth System Science Data (ESSD, 2021). Zenodo record 4275164
(DOI 10.5281/zenodo.4275164), license CC-BY-4.0.

  Hi-MAG database.zip  Annual glacial-lake POLYGON shapefiles, one per year 2008-2017
                       (Hi_MAG_database_YYYY.shp), in Asia North Albers Equal Area Conic
                       (ESRI:102025, metres). Per-lake attributes: GL_Type (proglacial /
                       supraglacial / unconnected glacial / ice-marginal), GL_Area (m^2),
                       GL_Elev, GL_SubR (HMA sub-region), GL_Peri, GL_ID (lon/lat code),
                       Distance (to nearest glacier). The source already applies a minimum
                       mapped-lake area (smallest lake in 2017 is ~0.0081 km^2 ~= 81 pixels
                       at 10 m), so every lake is well observable at 10 m.

Encoding — **binary presence / water-extent dense segmentation** (label_type polygons +
dense_raster). We rasterize the glacial-lake polygons into 64x64 (640 m) tiles at 10 m in a
local UTM projection:
  0 = background     surrounding High-Mountain-Asia terrain within the tile (glaciers,
                     moraine, rock, snow, vegetated valley floor). Genuine, spatially
                     meaningful negatives adjacent to the lakes, not fabricated.
  1 = glacial_lake   glacial-lake water surface (any of the four Hi-MAG lake types).

We collapse the four Hi-MAG lake *types* into a single ``glacial_lake`` water class: the
proglacial / supraglacial / unconnected / ice-marginal distinction is defined by a lake's
position/connectivity to its parent glacier (distance, contact), which is not spectrally
separable from a single S2/S1/Landsat tile — so binary water-extent is the well-posed target
(the type attribute is retained in each sample's source_id / summary for provenance).

Year: we use the **2017** inventory (the most recent Hi-MAG year, post-2016). Glacial lakes
are persistent surface-water bodies, so this is a static label: a representative 1-year
Sentinel-era window (2017), ``change_time=null`` (spec section 5, static labels). The
optional multi-year "change/expansion" variant is deliberately NOT used: Hi-MAG snapshots are
annual, so lake growth is only resolvable to ~a year — too coarse for the <=1-2-month
change-timing rule — whereas the presence/extent state is genuinely persistent.

Tiling (mirrors the blue-ice-areas dataset): each lake centroid is projected to its local
UTM zone at 10 m and snapped to a 64-px grid; the unique grid cells become candidate tiles,
and every lake polygon intersecting a tile is rasterized into it. Lakes are small relative to
a 640 m tile (median ~0.03 km^2), so most tiles are one lake surrounded by real terrain, with
larger lakes giving lake-dominant interior tiles. We keep one tile per unique grid cell (each
tile contains lake pixels by construction), capped at the 25k per-dataset limit.

Run (idempotent; skips already-written tiles):
  python3 -m olmoearth_pretrain.open_set_segmentation_data.datasets.hi_mag_glacial_lakes_high_mountain_asia
"""

import argparse
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

SLUG = "hi_mag_glacial_lakes_high_mountain_asia"
NAME = "Hi-MAG Glacial Lakes (High Mountain Asia)"
ZENODO_RECORD = "4275164"
ZENODO_DOI = "https://doi.org/10.5281/zenodo.4275164"
ZENODO_FILES = ["Hi-MAG database.zip", "Metadata for Hi-MAG database.docx"]

# The 2017 inventory is the most recent Hi-MAG year (post-2016). Lakes persist, so a static
# representative 1-year Sentinel-era window is used.
YEAR = 2017
SHP_NAME = f"Hi_MAG_database_{YEAR}.shp"

CID_BACKGROUND = 0
CID_GLACIAL_LAKE = 1
CLASSES = [
    {
        "id": CID_BACKGROUND,
        "name": "background",
        "description": "Surrounding High-Mountain-Asia terrain within the tile (glacier "
        "ice, moraine/debris, exposed rock, snow, or vegetated valley floor) around a "
        "glacial lake. Genuine spatially-meaningful negatives adjacent to the lake, not "
        "fabricated.",
    },
    {
        "id": CID_GLACIAL_LAKE,
        "name": "glacial_lake",
        "description": "Glacial-lake water surface from the Hi-MAG inventory (Chen et al., "
        "ESSD 2021): proglacial, supraglacial, unconnected-glacial, or ice-marginal lakes "
        "in High Mountain Asia, semi-automatically delineated on ~30 m Landsat and manually "
        "refined. All four lake types collapsed to a single water class (the type is a "
        "positional/connectivity attribute, not spectrally separable at 10 m).",
    },
]

TILE = io.MAX_TILE  # 64 px @ 10 m = 640 m
SEED = 42

# ---- worker globals (loaded lazily; forkserver workers don't inherit parent memory) ----
_G: dict[str, Any] = {}


def _shp_path() -> str:
    return (io.raw_dir(SLUG) / "extracted" / "Hi-MAG database" / SHP_NAME).path


def _ensure_loaded() -> dict[str, Any]:
    if _G:
        return _G
    import geopandas as gpd
    from shapely import STRtree

    gdf = gpd.read_file(_shp_path())
    # Fix any invalid polygons up front so intersection/rasterize never chokes.
    geoms = [g if g.is_valid else g.buffer(0) for g in gdf.geometry.values]
    _G["geoms"] = geoms
    _G["types"] = list(gdf["GL_Type"].values)
    _G["ids"] = list(gdf["GL_ID"].values)
    _G["tree"] = STRtree(geoms)
    src_crs = CRS.from_wkt(gdf.crs.to_wkt())
    # Identity projection (1 unit == 1 metre, no y-flip): keeps the polygons' native Albers
    # metre coordinates so to_projection into a UTM (10, -10) proj does the real reprojection.
    _G["p_src"] = Projection(src_crs, 1, 1)
    _G["to_wgs84"] = Transformer.from_crs(src_crs, CRS.from_epsg(4326), always_xy=True)
    return _G


def _tile_key(lake_idx: int) -> tuple[str, int, int] | None:
    """Local-UTM 64-px grid cell (crs, x0, y0) containing a lake's centroid."""
    g = _ensure_loaded()
    geom = g["geoms"][lake_idx]
    c = geom.centroid
    lon, lat = g["to_wgs84"].transform(c.x, c.y)
    proj = get_utm_ups_projection(lon, lat, io.RESOLUTION, -io.RESOLUTION)
    p = STGeometry(g["p_src"], shapely.Point(c.x, c.y), None).to_projection(proj).shp
    x0 = int(np.floor(p.x / TILE)) * TILE
    y0 = int(np.floor(p.y / TILE)) * TILE
    return (proj.crs.to_string(), x0, y0)


def _rasterize_tile(crs_str: str, x0: int, y0: int) -> np.ndarray | None:
    """Rasterize all lake polygons intersecting a tile into a (1, 64, 64) uint8 array."""
    g = _ensure_loaded()
    proj = Projection(CRS.from_string(crs_str), io.RESOLUTION, -io.RESOLUTION)
    bounds = (x0, y0, x0 + TILE, y0 + TILE)
    tile_box_px = shapely.box(*bounds)
    box_src = STGeometry(proj, tile_box_px, None).to_projection(g["p_src"]).shp
    clip_src = box_src.buffer(30.0)  # small pad so edge geometry isn't clipped away
    shapes: list[tuple[Any, int]] = []
    for i in g["tree"].query(box_src):
        geom = g["geoms"][int(i)]
        if not geom.intersects(box_src):
            continue
        clipped = geom.intersection(clip_src)
        if clipped.is_empty:
            continue
        pix = geom_to_pixels(clipped, g["p_src"], proj)
        if pix.is_empty:
            continue
        shapes.append((pix, CID_GLACIAL_LAKE))
    if not shapes:
        return None
    return rasterize_shapes(
        shapes, bounds, fill=CID_BACKGROUND, dtype="uint8", all_touched=False
    )


def _scan_tile(crs_str: str, x0: int, y0: int) -> dict[str, Any] | None:
    arr = _rasterize_tile(crs_str, x0, y0)
    if arr is None:
        return None
    lake_frac = float((arr == CID_GLACIAL_LAKE).mean())
    if lake_frac <= 0.0:
        return None
    return {
        "crs": crs_str,
        "x0": x0,
        "y0": y0,
        "lake_frac": lake_frac,
        "classes_present": sorted(int(v) for v in np.unique(arr)),
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
        io.year_range(YEAR),
        change_time=None,
        source_id=f"Hi_MAG_{YEAR}/tile_{rec['crs'].replace(':', '')}_{rec['x0']}_{rec['y0']}",
        classes_present=sorted(int(v) for v in np.unique(arr)),
    )
    return "written"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=64)
    args = parser.parse_args()

    io.check_disk()

    # --- download + extract the (small) annual polygon shapefiles ---
    raw = io.raw_dir(SLUG)
    raw.mkdir(parents=True, exist_ok=True)
    extracted = raw / "extracted"
    shp = extracted / "Hi-MAG database" / SHP_NAME
    if not shp.exists():
        print("downloading Hi-MAG database from Zenodo ...", flush=True)
        download_zenodo(ZENODO_RECORD, raw, filenames=ZENODO_FILES)
        import zipfile

        extracted.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile((raw / "Hi-MAG database.zip").path) as zf:
            zf.extractall(extracted.path)
    with (raw / "SOURCE.txt").open("w") as f:
        f.write(
            "Hi-MAG Glacial Lakes (High Mountain Asia) - Chen et al., ESSD (2021).\n"
            f"Zenodo record {ZENODO_RECORD} ({ZENODO_DOI}), license CC-BY-4.0.\n"
            "Annual 30 m glacial-lake polygon shapefiles 2008-2017 (Asia North Albers "
            "Equal Area Conic, ESRI:102025). This dataset uses the 2017 inventory "
            f"({SHP_NAME}) for a post-2016 static water-extent label. No imagery pulled "
            "(pretraining supplies its own).\n"
        )

    io.check_disk()

    # --- scan phase 1: local-UTM 64-px grid cell for each lake centroid (parallel) ---
    g = _ensure_loaded()
    n_lakes = len(g["geoms"])
    type_counts = Counter(g["types"])
    print(
        f"loaded {n_lakes} glacial-lake polygons ({YEAR}); types: {dict(type_counts)}",
        flush=True,
    )
    keys: set[tuple[str, int, int]] = set()
    with multiprocessing.Pool(args.workers) as p:
        for res in tqdm.tqdm(
            star_imap_unordered(
                p, _tile_key, [dict(lake_idx=i) for i in range(n_lakes)]
            ),
            total=n_lakes,
        ):
            if res is not None:
                keys.add(res)
    print(f"  {len(keys)} unique candidate tiles", flush=True)

    # --- scan phase 2: rasterize each unique tile to confirm lake content (parallel) ---
    key_list = sorted(keys)
    records: list[dict[str, Any]] = []
    with multiprocessing.Pool(args.workers) as p:
        for res in tqdm.tqdm(
            star_imap_unordered(
                p, _scan_tile, [dict(crs_str=c, x0=x, y0=y) for (c, x, y) in key_list]
            ),
            total=len(key_list),
        ):
            if res is not None:
                records.append(res)
    print(f"  {len(records)} tiles contain lake pixels", flush=True)

    # --- select: keep all lake tiles up to the 25k per-dataset cap (random if over) ---
    from olmoearth_pretrain.open_set_segmentation_data.sampling import (
        MAX_SAMPLES_PER_DATASET,
    )

    records.sort(key=lambda r: (r["crs"], r["x0"], r["y0"]))
    if len(records) > MAX_SAMPLES_PER_DATASET:
        rng = random.Random(SEED)
        records = sorted(
            rng.sample(records, MAX_SAMPLES_PER_DATASET),
            key=lambda r: (r["crs"], r["x0"], r["y0"]),
        )
        print(f"  capped to {len(records)} tiles (25k limit)", flush=True)
    for i, r in enumerate(records):
        r["sample_id"] = f"{i:06d}"

    io.check_disk()

    # --- write phase (parallel) ---
    results: Counter = Counter()
    class_tile_counts: Counter = Counter()
    with multiprocessing.Pool(args.workers) as p:
        for res in tqdm.tqdm(
            star_imap_unordered(p, _write_tile, [dict(rec=r) for r in records]),
            total=len(records),
        ):
            results[res] += 1
    for r in records:
        for c in r["classes_present"]:
            class_tile_counts[c] += 1
    fracs = np.array([r["lake_frac"] for r in records])
    print("write results:", dict(results), flush=True)
    print("class tile-appearance counts:", dict(class_tile_counts), flush=True)
    print(
        f"lake_frac: min={fracs.min():.3f} median={np.median(fracs):.3f} "
        f"max={fracs.max():.3f}",
        flush=True,
    )

    io.check_disk()
    io.write_dataset_metadata(
        SLUG,
        {
            "dataset": SLUG,
            "name": NAME,
            "task_type": "classification",
            "source": "Zenodo / ESSD (Chen et al., 2021)",
            "license": "CC-BY-4.0",
            "provenance": {
                "url": ZENODO_DOI,
                "have_locally": False,
                "annotation_method": "semi-automated delineation on ~30 m Landsat + manual "
                "expert refinement",
                "file_used": SHP_NAME,
            },
            "sensors_relevant": ["sentinel2", "sentinel1", "landsat"],
            "classes": CLASSES,
            "nodata_value": io.CLASS_NODATA,
            "num_samples": len(records),
            "class_tile_counts": {
                str(k): v for k, v in sorted(class_tile_counts.items())
            },
            "source_lake_type_counts": {
                str(k): v for k, v in sorted(type_counts.items())
            },
            "sampling": {
                "year": YEAR,
                "tile_size_px": TILE,
                "n_source_lakes": n_lakes,
                "grid_snap_px": TILE,
                "min_source_lake_area_km2": 0.0081,
                "cap": MAX_SAMPLES_PER_DATASET,
            },
            "time_range_rule": (
                f"persistent surface-water body -> static representative 1-year window {YEAR}"
            ),
            "notes": (
                "Binary glacial-lake water-extent dense segmentation from the Hi-MAG "
                "inventory (Chen et al., ESSD 2021; annual 30 m glacial-lake polygons for "
                "High Mountain Asia). 0=background (surrounding HMA terrain within tile), "
                "1=glacial_lake (water surface). Uses the 2017 inventory (15,348 lakes; most "
                "recent, post-2016). The four Hi-MAG lake types (proglacial/supraglacial/"
                "unconnected-glacial/ice-marginal) are collapsed to one water class because "
                "the type is a positional/connectivity attribute not spectrally separable at "
                "10 m. Each lake centroid -> local UTM 10 m, snapped to a 64-px (640 m) grid; "
                "every lake polygon intersecting a tile is rasterized; one tile per unique "
                "grid cell (all contain lake pixels). Source already applies a min mapped area "
                "(~0.0081 km^2 ~= 81 px @10 m) so all lakes are observable at 10 m; no extra "
                "size filter needed. Static persistent label, 1-year window 2017, "
                "change_time=null (annual snapshots make lake expansion only year-resolvable, "
                "too coarse for the change-timing rule, and the water-extent state is "
                "persistent). Background is real within-tile terrain, not fabricated."
            ),
        },
    )
    print(f"done: {len(records)} tiles")


if __name__ == "__main__":
    multiprocessing.set_start_method("forkserver", force=True)
    main()
