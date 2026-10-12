"""Process the West-Antarctic supraglacial lake & channel inventory into label patches.

Source: "Supraglacial lakes and channels in West Antarctica and Antarctic Peninsula
during January 2017" (Corr, Leeson, McMillan, Zhang & Barnes, Earth System Science
Data, 2022). Zenodo record 5642755 (DOI 10.5281/zenodo.5642755), license CC-BY-4.0. A
continent-scale inventory of ~10,500 supraglacial lakes and channels delineated from
January-2017 Landsat 8 and Sentinel-2 imagery (semi-automated classification + manual
post-processing) across the West Antarctic Ice Sheet and Antarctic Peninsula.

The Zenodo record is dominated by ~190 raw Landsat-8 / Sentinel-2 scene archives
(~130 TB total) that pretraining does NOT need — it supplies its own imagery. All the
label geometry we need lives in a single 17 MB file, ``WAIS_Max_Extent.zip``, which holds
``WAIS_Jan_2017_Polygons.shp`` (10,478 features; also as GeoJSON/KMZ). We download only
that file. Per-feature attributes: ``Feature_Cl`` (Lake / Channel), ``POLY_AREA``,
``Location`` (Ice Shelf / Grounded Ice / Crosses GL), REMA elevation, ice speed, shape
metrics, etc. CRS is Antarctic Polar Stereographic (PS_WGS84, lat_of_origin -71).

Encoding — **positive-only two-class polygon segmentation** (label_type polygons; spec
sections 4 polygon-rasterize and 5 positive-only). We rasterize the polygons into 64x64
(640 m) tiles at 10 m in a local UTM/UPS projection:
  0   = supraglacial_lake      lake water surface (Feature_Cl == "Lake").
  1   = supraglacial_channel   supraglacial meltwater channel (Feature_Cl == "Channel").
  255 = nodata / ignore        every non-feature pixel (surrounding ice, firn, snow, rock,
                               unmapped area).

This is a **positive-only foreground** dataset (two water-feature classes, no clean
background class): per spec section 5 we do NOT fabricate negatives — non-feature pixels are
left as nodata/ignore (255), and the pretraining-assembly step supplies negatives by
sampling other datasets. (This differs deliberately from the sibling Hi-MAG glacial-lake
dataset, which used a background=0 class; the orchestrator's dataset-specific directive for
this inventory is positive-only, and the surrounding Antarctic ice/firn/cloud is a less
clean negative than High-Mountain-Asia terrain.) Rasterization uses ``all_touched=True`` so
the smallest lakes (min ~96 m^2 ~= 1 px @10 m) and the thin channels stay visible at 10 m.

Year: the inventory is a January-2017 (austral-summer melt-peak) snapshot. Supraglacial
lakes/channels are seasonal, so this is treated as a seasonal/annual label: a 1-year window
anchored on 2017 (spec section 5), ``change_time=null``. It is NOT a change label (a single
dated inventory, no pre/post pair).

Tiling (mirrors the Hi-MAG glacial-lake dataset): each feature centroid is projected to its
local UTM/UPS zone at 10 m and snapped to a 64-px grid; the unique grid cells become
candidate tiles, and every lake/channel polygon intersecting a tile is rasterized into it.
Most features are small relative to a 640 m tile (median lake max-dim ~60 m; only ~3% of
lakes and ~14% of channels exceed 640 m), so a centroid tile captures the feature (large
features are captured as a representative central window). Selection is tiles-per-class
balanced (spec section 5), rarest class first, up to 1000 tiles per class, well under the
25k per-dataset cap. Channels (255 features) are rare and are all retained.

Run (idempotent; skips already-written tiles):
  python3 -m olmoearth_pretrain.open_set_segmentation_data.datasets.supraglacial_lakes_channels_west_antarctica
"""

import argparse
import multiprocessing
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
from olmoearth_pretrain.open_set_segmentation_data.download import (
    download_zenodo,
    extract_zip,
)
from olmoearth_pretrain.open_set_segmentation_data.rasterize import (
    geom_to_pixels,
    rasterize_shapes,
)
from olmoearth_pretrain.open_set_segmentation_data.sampling import (
    MAX_SAMPLES_PER_DATASET,
    select_tiles_per_class,
)

SLUG = "supraglacial_lakes_channels_west_antarctica"
NAME = "Supraglacial Lakes & Channels, West Antarctica"
ZENODO_RECORD = "5642755"
ZENODO_DOI = "https://doi.org/10.5281/zenodo.5642755"
# Only the small vector inventory is needed; the ~190 imagery scene archives (~130 TB) are
# not downloaded (pretraining supplies its own imagery).
ZENODO_FILE = "WAIS_Max_Extent.zip"
SHP_RELPATH = "WAIS_Max_Extent/WAIS_Jan_2017_Polygons.shp"

# January-2017 snapshot; supraglacial hydrology is seasonal -> 1-year window anchored on 2017.
YEAR = 2017

CID_LAKE = 0
CID_CHANNEL = 1
FEATURE_CLASS_TO_CID = {"Lake": CID_LAKE, "Channel": CID_CHANNEL}
CLASSES = [
    {
        "id": CID_LAKE,
        "name": "supraglacial_lake",
        "description": "Supraglacial lake water surface on the West Antarctic Ice Sheet / "
        "Antarctic Peninsula during January 2017 (Corr et al., ESSD 2022): ponded surface "
        "meltwater delineated from Landsat 8 / Sentinel-2 imagery (semi-automated "
        "classification + manual refinement). Includes lakes on ice shelves, grounded ice, "
        "and across the grounding line.",
    },
    {
        "id": CID_CHANNEL,
        "name": "supraglacial_channel",
        "description": "Supraglacial meltwater channel (surface stream/river routing "
        "meltwater across the ice) mapped in the same January-2017 inventory. Elongated "
        "features; rasterized with all_touched so thin channels remain visible at 10 m.",
    },
]

TILE = io.MAX_TILE  # 64 px @ 10 m = 640 m
PER_CLASS = 1000

# ---- worker globals (loaded lazily; forkserver workers don't inherit parent memory) ----
_G: dict[str, Any] = {}


def _shp_path() -> str:
    return (io.raw_dir(SLUG) / "extracted" / SHP_RELPATH).path


def _ensure_loaded() -> dict[str, Any]:
    if _G:
        return _G
    import geopandas as gpd
    from shapely import STRtree

    gdf = gpd.read_file(_shp_path())
    # Fix any invalid polygons up front so intersection/rasterize never chokes.
    geoms = [g if g.is_valid else g.buffer(0) for g in gdf.geometry.values]
    _G["geoms"] = geoms
    _G["cids"] = [FEATURE_CLASS_TO_CID[c] for c in gdf["Feature_Cl"].values]
    _G["feature_cls"] = list(gdf["Feature_Cl"].values)
    _G["tree"] = STRtree(geoms)
    src_crs = CRS.from_wkt(gdf.crs.to_wkt())
    # Identity projection (1 unit == 1 metre, no y-flip): keeps the polygons' native Polar
    # Stereographic metre coordinates so to_projection into a UTM/UPS (10, -10) proj does the
    # real reprojection.
    _G["p_src"] = Projection(src_crs, 1, 1)
    _G["to_wgs84"] = Transformer.from_crs(src_crs, CRS.from_epsg(4326), always_xy=True)
    return _G


def _tile_key(feat_idx: int) -> tuple[str, int, int] | None:
    """Local-UTM/UPS 64-px grid cell (crs, x0, y0) containing a feature's centroid."""
    g = _ensure_loaded()
    geom = g["geoms"][feat_idx]
    c = geom.centroid
    lon, lat = g["to_wgs84"].transform(c.x, c.y)
    proj = get_utm_ups_projection(lon, lat, io.RESOLUTION, -io.RESOLUTION)
    p = STGeometry(g["p_src"], shapely.Point(c.x, c.y), None).to_projection(proj).shp
    x0 = int(np.floor(p.x / TILE)) * TILE
    y0 = int(np.floor(p.y / TILE)) * TILE
    return (proj.crs.to_string(), x0, y0)


def _rasterize_tile(crs_str: str, x0: int, y0: int) -> np.ndarray | None:
    """Rasterize all lake/channel polygons intersecting a tile into a (1, 64, 64) uint8 array.

    Lake pixels = 0, channel pixels = 1, everything else = 255 (nodata/ignore).
    """
    g = _ensure_loaded()
    proj = Projection(CRS.from_string(crs_str), io.RESOLUTION, -io.RESOLUTION)
    bounds = (x0, y0, x0 + TILE, y0 + TILE)
    tile_box_px = shapely.box(*bounds)
    box_src = STGeometry(proj, tile_box_px, None).to_projection(g["p_src"]).shp
    clip_src = box_src.buffer(30.0)  # small pad so edge geometry isn't clipped away
    lake_shapes: list[tuple[Any, int]] = []
    chan_shapes: list[tuple[Any, int]] = []
    for i in g["tree"].query(box_src):
        i = int(i)
        geom = g["geoms"][i]
        if not geom.intersects(box_src):
            continue
        clipped = geom.intersection(clip_src)
        if clipped.is_empty:
            continue
        pix = geom_to_pixels(clipped, g["p_src"], proj)
        if pix.is_empty:
            continue
        (chan_shapes if g["cids"][i] == CID_CHANNEL else lake_shapes).append(
            (pix, g["cids"][i])
        )
    if not lake_shapes and not chan_shapes:
        return None
    # Paint lakes first, then channels on top, so at the rare lake/channel adjacency the
    # channel (rarer, valued) class wins. all_touched keeps tiny lakes and thin channels.
    return rasterize_shapes(
        lake_shapes + chan_shapes,
        bounds,
        fill=io.CLASS_NODATA,
        dtype="uint8",
        all_touched=True,
    )


def _classes_present(arr: np.ndarray) -> list[int]:
    return sorted(int(v) for v in np.unique(arr) if v != io.CLASS_NODATA)


def _scan_tile(crs_str: str, x0: int, y0: int) -> dict[str, Any] | None:
    arr = _rasterize_tile(crs_str, x0, y0)
    if arr is None:
        return None
    present = _classes_present(arr)
    if not present:
        return None
    return {"crs": crs_str, "x0": x0, "y0": y0, "classes_present": present}


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
        source_id=f"WAIS_Jan2017/tile_{rec['crs'].replace(':', '')}_{rec['x0']}_{rec['y0']}",
        classes_present=_classes_present(arr),
    )
    return "written"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=64)
    args = parser.parse_args()

    io.check_disk()

    # --- download + extract only the (17 MB) vector inventory; no imagery scenes ---
    raw = io.raw_dir(SLUG)
    raw.mkdir(parents=True, exist_ok=True)
    extracted = raw / "extracted"
    if not (extracted / SHP_RELPATH).exists():
        print(
            f"downloading {ZENODO_FILE} from Zenodo record {ZENODO_RECORD} ...",
            flush=True,
        )
        download_zenodo(ZENODO_RECORD, raw, filenames=[ZENODO_FILE])
        extract_zip(raw / ZENODO_FILE, extracted, skip_existing=False)
    with (raw / "SOURCE.txt").open("w") as f:
        f.write(
            "Supraglacial Lakes & Channels, West Antarctica - Corr et al., ESSD (2022).\n"
            f"Zenodo record {ZENODO_RECORD} ({ZENODO_DOI}), license CC-BY-4.0.\n"
            "Continent-scale inventory of ~10,500 supraglacial lakes and channels for "
            "January 2017 (Landsat 8 + Sentinel-2, semi-automated + manual). Only the "
            f"vector inventory ({ZENODO_FILE} -> {SHP_RELPATH}) is used; the ~190 raw "
            "imagery scene archives (~130 TB) are NOT downloaded (pretraining supplies its "
            "own imagery).\n"
        )

    io.check_disk()

    # --- load inventory ---
    g = _ensure_loaded()
    n_feat = len(g["geoms"])
    feat_counts = Counter(g["feature_cls"])
    print(f"loaded {n_feat} features; Feature_Cl: {dict(feat_counts)}", flush=True)

    # --- scan phase 1: local-UTM/UPS 64-px grid cell for each feature centroid (parallel) ---
    keys: set[tuple[str, int, int]] = set()
    with multiprocessing.Pool(args.workers) as p:
        for res in tqdm.tqdm(
            star_imap_unordered(
                p, _tile_key, [dict(feat_idx=i) for i in range(n_feat)]
            ),
            total=n_feat,
        ):
            if res is not None:
                keys.add(res)
    print(f"  {len(keys)} unique candidate tiles", flush=True)

    # --- scan phase 2: rasterize each unique tile to confirm feature content (parallel) ---
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
    print(f"  {len(records)} tiles contain lake/channel pixels", flush=True)

    # --- select: tiles-per-class balanced, rarest (channel) first, <=1000/class, <=25k ---
    selected = select_tiles_per_class(
        records,
        classes_key="classes_present",
        per_class=PER_CLASS,
        total_cap=MAX_SAMPLES_PER_DATASET,
    )
    selected.sort(key=lambda r: (r["crs"], r["x0"], r["y0"]))
    for i, r in enumerate(selected):
        r["sample_id"] = f"{i:06d}"
    print(f"  selected {len(selected)} tiles", flush=True)

    io.check_disk()

    # --- write phase (parallel) ---
    results: Counter = Counter()
    class_tile_counts: Counter = Counter()
    for r in selected:
        for c in r["classes_present"]:
            class_tile_counts[c] += 1
    with multiprocessing.Pool(args.workers) as p:
        for res in tqdm.tqdm(
            star_imap_unordered(p, _write_tile, [dict(rec=r) for r in selected]),
            total=len(selected),
        ):
            results[res] += 1
    print("write results:", dict(results), flush=True)
    print("class tile-appearance counts:", dict(class_tile_counts), flush=True)

    io.check_disk()
    io.write_dataset_metadata(
        SLUG,
        {
            "dataset": SLUG,
            "name": NAME,
            "task_type": "classification",
            "source": "Zenodo / ESSD (Corr et al., 2022)",
            "license": "CC-BY-4.0",
            "provenance": {
                "url": ZENODO_DOI,
                "have_locally": False,
                "annotation_method": "semi-automated classification of Landsat 8 / "
                "Sentinel-2 (Jan 2017) + manual post-processing",
                "file_used": f"{ZENODO_FILE} -> {SHP_RELPATH}",
            },
            "sensors_relevant": ["sentinel2", "sentinel1", "landsat"],
            "classes": CLASSES,
            "nodata_value": io.CLASS_NODATA,
            "num_samples": len(selected),
            "class_tile_counts": {
                str(k): v for k, v in sorted(class_tile_counts.items())
            },
            "source_feature_class_counts": {str(k): v for k, v in feat_counts.items()},
            "sampling": {
                "year": YEAR,
                "tile_size_px": TILE,
                "n_source_features": n_feat,
                "grid_snap_px": TILE,
                "per_class": PER_CLASS,
                "cap": MAX_SAMPLES_PER_DATASET,
                "all_touched": True,
                "positive_only": True,
            },
            "time_range_rule": (
                "seasonal (austral-summer) supraglacial hydrology snapshot -> 1-year window "
                f"anchored on {YEAR}; change_time=null (single dated inventory, not a "
                "pre/post change label)"
            ),
            "notes": (
                "Positive-only two-class supraglacial-hydrology segmentation from the "
                "January-2017 West-Antarctic inventory (Corr et al., ESSD 2022). "
                "0=supraglacial_lake, 1=supraglacial_channel, 255=nodata/ignore (all "
                "non-feature pixels; no background/negative class fabricated -- assembly "
                "supplies negatives, spec section 5). Source: 10,478 polygons (10,223 lakes, "
                "255 channels) in WAIS_Jan_2017_Polygons.shp (Antarctic Polar Stereographic); "
                "only the 17 MB vector file is downloaded, not the ~130 TB of imagery scenes. "
                "Each feature centroid -> local UTM/UPS 10 m, snapped to a 64-px (640 m) grid; "
                "every polygon intersecting a tile is rasterized (all_touched=True so tiny "
                "lakes ~1 px and thin channels stay visible); one tile per unique grid cell. "
                "Tiles-per-class balanced (rarest class first), <=1000 tiles/class. Channels "
                "are sparse (255 source features) but all retained per spec section 5 (rare "
                "classes kept; downstream filtering drops too-small classes). Seasonal label, "
                "1-year window 2017, change_time=null."
            ),
        },
    )
    print(f"done: {len(selected)} tiles")


if __name__ == "__main__":
    multiprocessing.set_start_method("forkserver", force=True)
    main()
