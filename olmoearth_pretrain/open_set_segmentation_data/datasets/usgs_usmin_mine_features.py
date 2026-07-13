"""Process USGS USMIN Mine Features into open-set-segmentation label patches.

Source: USGS Mineral Resources "Prospect- and Mine-Related Features from U.S. Geological
Survey 7.5- and 15-Minute Topographic Quadrangle Maps" (USMIN), version 10.0 (May 2023),
public domain. Downloaded as the national File Geodatabase from ScienceBase:

  https://www.sciencebase.gov/catalog/file/get/5a1492c3e4b09fc93dcfd574?name=USGS_TopoMineSymbols_ver10_Geodatabase.zip
  (project page: https://mrdata.usgs.gov/usmin/)

The GDB holds point + polygon feature classes digitized from historical topographic maps,
at three source map scales: 1:24,000 (24k), 1:48,000 / 15-minute (48k), and 1:625,000
(625k). We use only the **24k and 48k** layers (positional accuracy adequate for a 10 m
grid); the **625k** layers are dropped (their ~hundreds-of-metres positional error makes
them unusable for 10 m label tiles). See the summary for the full rationale.

Each feature carries a ``Ftr_Type`` (feature-type symbol). We build ONE unified
classification dataset combining both geometry kinds (spec §5 multi-modality rule):
  - polygon features (real footprints) are RASTERIZED into a <=64x64 UTM 10 m tile;
  - point features (presence markers, no footprint) use the tunable DETECTION encoding
    (1 px positive + 10 px nodata buffer ring + background fill in a 32x32 context tile).
"Prefer polygons where available" (task): within each class, polygon records are selected
before point records.

Class scheme (id 0 = background; 255 = nodata/ignore = detection buffer rings):
  0 background        6 strip_mine
  1 prospect_pit      7 tailings_pile
  2 mine_shaft        8 tailings_pond
  3 adit              9 mine_dump
  4 quarry_open_pit  10 disturbed_surface
  5 gravel_borrow_pit

Observability caveat (documented in the summary): quarry/open-pit, strip mine,
gravel/borrow pit, tailings ponds/piles, mine dumps and disturbed-surface polygons have
footprints resolvable at 10 m; the point-only classes prospect_pit / mine_shaft / adit are
often sub-10 m and mostly serve as weak presence-detection targets.

Time range: these are persistent, undated (map-digitized) features. Per spec §5 (static
labels), each sample gets a 1-year window at a representative Sentinel-era year, spread
pseudo-randomly across 2016-2022 for temporal diversity.

Run (idempotent; skips already-written tiles):
  python3 -m olmoearth_pretrain.open_set_segmentation_data.datasets.usgs_usmin_mine_features
"""

import argparse
import multiprocessing
import random
from collections import Counter, defaultdict
from typing import Any

import fiona
import numpy as np
import shapely
import tqdm
from rslearn.const import WGS84_PROJECTION
from rslearn.utils.mp import star_imap_unordered
from scipy.spatial import cKDTree

from olmoearth_pretrain.open_set_segmentation_data import io, manifest
from olmoearth_pretrain.open_set_segmentation_data.rasterize import (
    geom_to_pixels,
    rasterize_shapes,
)

SLUG = "usgs_usmin_mine_features"
NAME = "USGS USMIN Mine Features"
SB_ITEM = "5a1492c3e4b09fc93dcfd574"
DOWNLOAD_URL = (
    "https://www.sciencebase.gov/catalog/file/get/"
    f"{SB_ITEM}?name=USGS_TopoMineSymbols_ver10_Geodatabase.zip"
)
GDB = "USGS_TopoMineSymbols_ver10_Geodatabase/USGS_TopoMineSymbols_ver10.gdb"
POINT_LAYERS = ["USGS_TopoMineSymbols_24k_Points", "USGS_TopoMineSymbols_48k_Points"]
POLY_LAYERS = ["USGS_TopoMineSymbols_24k_Polygons", "USGS_TopoMineSymbols_48k_Polygons"]

# Class scheme. id 0 reserved for background.
CID_BACKGROUND = 0
CLASSES = [
    {
        "id": 0,
        "name": "background",
        "description": "Negative / non-mine land: pixels outside any mapped mine feature.",
    },
    {
        "id": 1,
        "name": "prospect_pit",
        "description": "Small exploratory prospect pit or diggings (test excavation). "
        "Point-only; typically sub-10 m, so mainly a weak presence-detection target.",
    },
    {
        "id": 2,
        "name": "mine_shaft",
        "description": "Vertical mine shaft or air shaft. Point-only; typically sub-10 m at "
        "the surface, so mainly a weak presence-detection target.",
    },
    {
        "id": 3,
        "name": "adit",
        "description": "Horizontal mine entrance (adit) driven into a hillside. Point-only; "
        "typically sub-10 m, so mainly a weak presence-detection target.",
    },
    {
        "id": 4,
        "name": "quarry_open_pit",
        "description": "Quarry or open-pit mine (rock/limestone/gypsum/pumice quarries, "
        "open-pit mines). Polygons rasterized; points detection-encoded.",
    },
    {
        "id": 5,
        "name": "gravel_borrow_pit",
        "description": "Gravel, sand, or borrow pit (surface aggregate extraction). "
        "Polygons rasterized; points detection-encoded.",
    },
    {
        "id": 6,
        "name": "strip_mine",
        "description": "Strip mine (surface/contour mining), large disturbed extraction area. "
        "Predominantly polygons rasterized at 10 m.",
    },
    {
        "id": 7,
        "name": "tailings_pile",
        "description": "Tailings/waste pile (undifferentiated, placer, dredge, mill tailings) "
        "or slag pile. Polygons rasterized; points detection-encoded.",
    },
    {
        "id": 8,
        "name": "tailings_pond",
        "description": "Tailings pond, settling/leach/evaporation pond, or salt evaporator "
        "(impounded process water). Polygons rasterized; points detection-encoded.",
    },
    {
        "id": 9,
        "name": "mine_dump",
        "description": "Mine dump / ore stockpile (waste rock or ore storage). "
        "Polygons rasterized; points detection-encoded.",
    },
    {
        "id": 10,
        "name": "disturbed_surface",
        "description": "Mining-disturbed surface, disturbed-surface pit, or trench "
        "(bare disturbed ground). Predominantly polygons rasterized at 10 m.",
    },
]
N_FEATURE_CLASSES = len(CLASSES) - 1  # excludes background

# Map raw Ftr_Type -> class id. Unmapped types are dropped (documented in summary).
FTR_TYPE_TO_CLASS = {
    # 1 prospect_pit
    "Prospect Pit": 1,
    "Diggings": 1,
    "Glory Hole": 1,
    # 2 mine_shaft
    "Mine Shaft": 2,
    "Air Shaft": 2,
    # 3 adit
    "Adit": 3,
    # 4 quarry_open_pit
    "Quarry": 4,
    "Quarry - Rock": 4,
    "Quarry - Limestone": 4,
    "Quarry - Gypsum": 4,
    "Quarry - Pumice": 4,
    "Open Pit Mine": 4,
    "Open Pit Mine or Quarry": 4,
    # 5 gravel_borrow_pit
    "Gravel Pit": 5,
    "Borrow Pit": 5,
    "Sand Pit": 5,
    "Sand and Gravel Pit": 5,
    "Gravel/Borrow Pit - Undifferentiated": 5,
    # 6 strip_mine
    "Strip Mine": 6,
    # 7 tailings_pile
    "Tailings - Undifferentiated": 7,
    "Tailings - Placer": 7,
    "Tailings - Dredge": 7,
    "Tailings - Mill": 7,
    "Slag Pile": 7,
    # 8 tailings_pond
    "Tailings - Pond": 8,
    "Settling Pond": 8,
    "Leach Pond": 8,
    "Evaporation Pond": 8,
    "Salt Evaporator": 8,
    # 9 mine_dump
    "Mine Dump": 9,
    "Ore Stockpile/Storage": 9,
    # 10 disturbed_surface
    "Disturbed Surface": 10,
    "Disturbed Surface - Pit": 10,
    "Trench": 10,
}

# Sampling / encoding parameters.
PER_CLASS = 1000
N_NEGATIVES = 500  # background-only tiles for the detection classes
YEARS = list(range(2016, 2023))  # representative Sentinel-era 1-year windows

DET_TILE = 32
DET_POS_SIZE = 1
DET_BUFFER = 10
MAX_POLY_TILE = io.MAX_TILE  # 64

_TO_3857 = None
_TO_4326 = None


def _to_3857(lon: float, lat: float) -> tuple[float, float]:
    global _TO_3857
    if _TO_3857 is None:
        from pyproj import Transformer

        _TO_3857 = Transformer.from_crs(4326, 3857, always_xy=True)
    return _TO_3857.transform(lon, lat)


def _to_4326(x: float, y: float) -> tuple[float, float]:
    global _TO_4326
    if _TO_4326 is None:
        from pyproj import Transformer

        _TO_4326 = Transformer.from_crs(3857, 4326, always_xy=True)
    return _TO_4326.transform(x, y)


def gdb_path() -> str:
    return str(io.raw_dir(SLUG) / GDB)


# --------------------------------------------------------------------------------------
# Reading source features.
# --------------------------------------------------------------------------------------
def read_points() -> list[dict[str, Any]]:
    """Read mapped point features into records with lon/lat + class id."""
    recs: list[dict[str, Any]] = []
    for layer in POINT_LAYERS:
        with fiona.open(gdb_path(), layer=layer) as src:
            for i, feat in enumerate(src):
                cid = FTR_TYPE_TO_CLASS.get(feat["properties"].get("Ftr_Type"))
                if cid is None or feat["geometry"] is None:
                    continue
                lon, lat = feat["geometry"]["coordinates"][:2]
                recs.append(
                    {
                        "kind": "point",
                        "class_id": cid,
                        "lon": float(lon),
                        "lat": float(lat),
                        "source_id": f"{layer}/{i}",
                    }
                )
    return recs


def read_polygons() -> list[dict[str, Any]]:
    """Read mapped polygon features into records with centroid lon/lat + geometry WKB."""
    recs: list[dict[str, Any]] = []
    for layer in POLY_LAYERS:
        with fiona.open(gdb_path(), layer=layer) as src:
            for i, feat in enumerate(src):
                cid = FTR_TYPE_TO_CLASS.get(feat["properties"].get("Ftr_Type"))
                if cid is None or feat["geometry"] is None:
                    continue
                try:
                    geom = shapely.geometry.shape(feat["geometry"])
                except Exception:
                    continue
                if geom.is_empty or not geom.is_valid:
                    geom = geom.buffer(0) if not geom.is_empty else geom
                    if geom.is_empty:
                        continue
                c = geom.centroid
                recs.append(
                    {
                        "kind": "polygon",
                        "class_id": cid,
                        "lon": float(c.x),
                        "lat": float(c.y),
                        "geom_wkb": shapely.to_wkb(geom),
                        "source_id": f"{layer}/{i}",
                    }
                )
    return recs


# --------------------------------------------------------------------------------------
# Writers (worker processes).
# --------------------------------------------------------------------------------------
def _write_polygon(rec: dict[str, Any]) -> str:
    sample_id = rec["sample_id"]
    tif = io.locations_dir(SLUG) / f"{sample_id}.tif"
    if tif.exists():
        return "skip"
    proj = io.utm_projection_for_lonlat(rec["lon"], rec["lat"])
    geom = shapely.from_wkb(rec["geom_wkb"])
    pix = geom_to_pixels(geom, WGS84_PROJECTION, proj)
    minx, miny, maxx, maxy = pix.bounds
    cx = int(round((minx + maxx) / 2))
    cy = int(round((miny + maxy) / 2))
    w = min(MAX_POLY_TILE, max(1, int(np.ceil(maxx - minx))))
    h = min(MAX_POLY_TILE, max(1, int(np.ceil(maxy - miny))))
    bounds = io.centered_bounds(cx, cy, w, h)
    arr = rasterize_shapes(
        [(pix, rec["class_id"])],
        bounds,
        fill=CID_BACKGROUND,
        dtype="uint8",
        all_touched=True,
    )
    io.write_label_geotiff(SLUG, sample_id, arr, proj, bounds, nodata=io.CLASS_NODATA)
    io.write_sample_json(
        SLUG,
        sample_id,
        proj,
        bounds,
        io.year_range(rec["year"]),
        source_id=rec["source_id"],
        classes_present=sorted(set(np.unique(arr).tolist()) - {io.CLASS_NODATA}),
    )
    return "polygon"


def _write_point(rec: dict[str, Any]) -> str:
    sample_id = rec["sample_id"]
    tif = io.locations_dir(SLUG) / f"{sample_id}.tif"
    if tif.exists():
        return "skip"
    proj = io.utm_projection_for_lonlat(rec["lon"], rec["lat"])
    _, col, row = io.lonlat_to_utm_pixel(rec["lon"], rec["lat"], proj)
    bounds = io.centered_bounds(col, row, DET_TILE, DET_TILE)
    x_min, y_min, _, _ = bounds
    # Self + any neighbor point features (any class) falling inside this tile.
    positives: list[tuple[int, int, int]] = []
    cands = [(rec["lon"], rec["lat"], rec["class_id"])] + rec.get("neighbors", [])
    for lon, lat, cid in cands:
        _, c, r = io.lonlat_to_utm_pixel(lon, lat, proj)
        lc, lr = c - x_min, r - y_min
        if 0 <= lc < DET_TILE and 0 <= lr < DET_TILE:
            positives.append((lr, lc, cid))
    arr = _encode_det(positives)
    io.write_label_geotiff(SLUG, sample_id, arr, proj, bounds, nodata=io.CLASS_NODATA)
    io.write_sample_json(
        SLUG,
        sample_id,
        proj,
        bounds,
        io.year_range(rec["year"]),
        source_id=rec["source_id"],
        classes_present=sorted(set(np.unique(arr).tolist()) - {io.CLASS_NODATA}),
    )
    return "point"


def _write_negative(rec: dict[str, Any]) -> str:
    sample_id = rec["sample_id"]
    tif = io.locations_dir(SLUG) / f"{sample_id}.tif"
    if tif.exists():
        return "skip"
    proj = io.utm_projection_for_lonlat(rec["lon"], rec["lat"])
    _, col, row = io.lonlat_to_utm_pixel(rec["lon"], rec["lat"], proj)
    bounds = io.centered_bounds(col, row, DET_TILE, DET_TILE)
    arr = _encode_det([])
    io.write_label_geotiff(SLUG, sample_id, arr, proj, bounds, nodata=io.CLASS_NODATA)
    io.write_sample_json(
        SLUG,
        sample_id,
        proj,
        bounds,
        io.year_range(rec["year"]),
        source_id=rec["source_id"],
        classes_present=[CID_BACKGROUND],
    )
    return "negative"


def _encode_det(positives: list[tuple[int, int, int]]) -> np.ndarray:
    from olmoearth_pretrain.open_set_segmentation_data.sampling import (
        encode_detection_tile,
    )

    return encode_detection_tile(
        positives,
        tile_size=DET_TILE,
        positive_size=DET_POS_SIZE,
        buffer_size=DET_BUFFER,
        nodata=io.CLASS_NODATA,
        background=CID_BACKGROUND,
    )[np.newaxis]


def _dispatch(rec: dict[str, Any]) -> str:
    if rec["kind"] == "polygon":
        return _write_polygon(rec)
    if rec["kind"] == "point":
        return _write_point(rec)
    return _write_negative(rec)


# --------------------------------------------------------------------------------------
# Selection.
# --------------------------------------------------------------------------------------
def select_records(
    points: list[dict[str, Any]], polygons: list[dict[str, Any]], seed: int = 42
) -> list[dict[str, Any]]:
    """Up to PER_CLASS per feature class, preferring polygons over points."""
    rng = random.Random(seed)
    by_class: dict[int, dict[str, list]] = defaultdict(
        lambda: {"polygon": [], "point": []}
    )
    for r in polygons:
        by_class[r["class_id"]]["polygon"].append(r)
    for r in points:
        by_class[r["class_id"]]["point"].append(r)
    selected: list[dict[str, Any]] = []
    for cid in sorted(by_class):
        polys = by_class[cid]["polygon"][:]
        pts = by_class[cid]["point"][:]
        rng.shuffle(polys)
        rng.shuffle(pts)
        chosen = polys[:PER_CLASS]
        if len(chosen) < PER_CLASS:
            chosen += pts[: PER_CLASS - len(chosen)]
        selected.extend(chosen)
    return selected


def make_negatives(
    pts_xy: np.ndarray, tree: cKDTree, pts: list[dict[str, Any]], n: int, seed: int = 7
) -> list[dict[str, Any]]:
    """Background-only tile centers offset from point features, guaranteed feature-free."""
    rng = random.Random(seed)
    out: list[dict[str, Any]] = []
    attempts = 0
    while len(out) < n and attempts < n * 50:
        attempts += 1
        base = pts[rng.randrange(len(pts))]
        ang = rng.uniform(0, 2 * np.pi)
        dist = rng.uniform(3000, 15000)  # metres (EPSG:3857)
        bx, by = _to_3857(base["lon"], base["lat"])
        x, y = bx + dist * np.cos(ang), by + dist * np.sin(ang)
        if tree.query_ball_point([x, y], r=1000.0):
            continue
        lon, lat = _to_4326(x, y)
        if not (18 <= lat <= 72):
            continue
        out.append(
            {
                "kind": "negative",
                "lon": float(lon),
                "lat": float(lat),
                "source_id": f"negative/{len(out)}",
            }
        )
    return out


# --------------------------------------------------------------------------------------
# Main.
# --------------------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=64)
    args = parser.parse_args()

    io.check_disk()
    manifest.write_registry_entry(SLUG, "in_progress")

    raw = io.raw_dir(SLUG)
    raw.mkdir(parents=True, exist_ok=True)
    with (raw / "SOURCE.txt").open("w") as f:
        f.write(
            "USGS USMIN 'Prospect- and Mine-Related Features from USGS 7.5- and "
            "15-Minute Topographic Quadrangle Maps', version 10.0 (May 2023). "
            "Public domain.\n"
            f"ScienceBase item {SB_ITEM}\n{DOWNLOAD_URL}\n"
            f"National File Geodatabase; using 24k + 48k point/polygon layers "
            "(625k layers excluded: positional error too large for 10 m tiles).\n"
        )

    print("reading polygon features ...")
    polygons = read_polygons()
    print(f"  {len(polygons)} mapped polygon features")
    print("reading point features ...")
    points = read_points()
    print(f"  {len(points)} mapped point features")

    io.check_disk()

    # Global KDTree over ALL point features (EPSG:3857) for negatives + neighbor marking.
    pts_xy = np.array([_to_3857(p["lon"], p["lat"]) for p in points], dtype=float)
    tree = cKDTree(pts_xy)

    selected = select_records(points, polygons)
    negatives = make_negatives(pts_xy, tree, points, N_NEGATIVES)

    # Assign representative years (spread across Sentinel era) + neighbor points for tiles.
    rng = random.Random(123)
    for r in selected:
        r["year"] = YEARS[rng.randrange(len(YEARS))]
        if r["kind"] == "point":
            x, y = _to_3857(r["lon"], r["lat"])
            idxs = tree.query_ball_point([x, y], r=1000.0)
            r["neighbors"] = [
                (points[i]["lon"], points[i]["lat"], points[i]["class_id"])
                for i in idxs
                if points[i]["source_id"] != r["source_id"]
            ][:200]
    for r in negatives:
        r["year"] = YEARS[rng.randrange(len(YEARS))]

    all_recs = selected + negatives
    for i, r in enumerate(all_recs):
        r["sample_id"] = f"{i:06d}"

    # Report selection counts.
    sel_counts: Counter = Counter()
    kind_counts: Counter = Counter()
    for r in selected:
        sel_counts[r["class_id"]] += 1
        kind_counts[(r["class_id"], r["kind"])] += 1
    id_to_name = {c["id"]: c["name"] for c in CLASSES}
    print(f"selected {len(selected)} feature tiles + {len(negatives)} negatives")
    for cid in sorted(sel_counts):
        print(
            f"  {sel_counts[cid]:5d}  {id_to_name[cid]:20s} "
            f"(poly={kind_counts[(cid, 'polygon')]}, point={kind_counts[(cid, 'point')]})"
        )

    results: Counter = Counter()
    with multiprocessing.Pool(args.workers) as p:
        for res in tqdm.tqdm(
            star_imap_unordered(p, _dispatch, [dict(rec=r) for r in all_recs]),
            total=len(all_recs),
        ):
            results[res] += 1
    print("write results:", dict(results))

    io.check_disk()

    class_counts = {
        id_to_name[cid]: sel_counts.get(cid, 0) for cid in range(1, len(CLASSES))
    }
    class_counts["background_negative_tiles"] = len(negatives)
    io.write_dataset_metadata(
        SLUG,
        {
            "dataset": SLUG,
            "name": NAME,
            "task_type": "classification",
            "source": "USGS (ScienceBase)",
            "license": "public domain",
            "provenance": {
                "url": "https://mrdata.usgs.gov/usmin/",
                "sciencebase_item": SB_ITEM,
                "download_url": DOWNLOAD_URL,
                "have_locally": False,
                "annotation_method": "manual digitizing of mine symbols from historical "
                "USGS topographic quadrangle maps",
                "version": "10.0 (May 2023)",
            },
            "sensors_relevant": ["sentinel2", "sentinel1", "landsat"],
            "classes": CLASSES,
            "nodata_value": io.CLASS_NODATA,
            "detection_encoding": {
                "applies_to": "point features (all feature classes)",
                "tile_size": DET_TILE,
                "positive_size": DET_POS_SIZE,
                "buffer_size": DET_BUFFER,
            },
            "num_samples": len(all_recs),
            "class_counts": class_counts,
            "notes": (
                "Unified point+polygon dataset. Polygon features rasterized into "
                "<=64x64 UTM 10 m tiles (footprint centered; >640 m footprints keep the "
                "central 64x64); point features detection-encoded (1 px positive + 10 px "
                "nodata buffer ring, 32x32 context tile, all nearby point features marked). "
                "Within each class, polygons preferred over points. Layers used: 24k + 48k "
                "point/polygon (625k dropped for poor positional accuracy). Unmapped minor "
                "Ftr_Type values (clay/cinder/shale/caliche/scoria/chert/marl/bentonite/"
                "shell/iron/lignite pits, generic Mine, coal/uranium/placer/hydraulic mines, "
                "mill site, tipple) dropped. Point-only classes prospect_pit/mine_shaft/adit "
                "are often sub-10 m -> weak presence-detection only; quarry/open-pit, strip "
                "mine, gravel/borrow pit, tailings pond/pile, mine dump, disturbed surface "
                "polygons are resolvable at 10 m. Persistent features -> 1-year window at a "
                "representative Sentinel-era year (2016-2022)."
            ),
        },
    )
    manifest.write_registry_entry(
        SLUG, "completed", task_type="classification", num_samples=len(all_recs)
    )
    print("done:", len(all_recs), "samples")


if __name__ == "__main__":
    multiprocessing.set_start_method("forkserver", force=True)
    main()
