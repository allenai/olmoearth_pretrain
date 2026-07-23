"""GEM Global Active Faults Database -> active-fault slip-type line masks.

Source: GEM Global Active Faults Database (GAF-DB) -- Styron, R. & Pagani, M. (2020),
"The GEM Global Active Faults Database", Earthquake Spectra 36(1_suppl):160-180
(doi 10.1177/8755293020944182). A global, homogenized compilation of ACTIVE FAULT
TRACES (surface traces of Quaternary-active faults) with kinematics and slip-rate
attributes, compiled from ~hundreds of national/regional sources.
GitHub: https://github.com/GEMScienceTools/gem-global-active-faults (CC-BY-SA-4.0).
We use ``geojson/gem_active_faults_harmonized.geojson`` (13,696 WGS84 LineStrings, the
harmonized-attribute release), whose ``slip_type`` field carries fault kinematics.

===================================================================================
SUITABILITY JUDGMENT (spec S4 "lines": reject if the feature is not observable at
10-30 m). Fault traces are geological LINEAMENTS, and only SOME are surface-expressed
at 10-30 m. We ACCEPT the dataset but apply strict observability filtering and document
heavy caveats:

  WHY ACCEPT (partial observability): a large fraction of continental active faults are
  mapped precisely because they have geomorphic surface expression -- fault scarps,
  offset ridges/drainages, range-front escarpments, fault-line valleys, sag ponds --
  which is exactly the linear/geomorphic signal Sentinel-2 / Landsat / Sentinel-1
  pretraining can learn. These are visible at 10-30 m when the trace is buffered to a
  short zone.

  WHY FILTER (much is NOT observable): (1) many faults are blind/buried or have subtle
  sub-pixel scarps; (2) OFFSHORE / SUBMARINE fault classes -- oceanic spreading ridges,
  subduction megathrust traces at trenches, oceanic transform faults -- are not visible
  in land imagery at all; (3) the trace positional accuracy is highly variable: where
  recorded, the ``accuracy`` field holds source mapping scales as coarse as 1:100k-1:1M,
  i.e. hundreds of metres to kilometres of positional uncertainty, so a rasterized line
  may not sit exactly on the imaged feature.

  FILTERS APPLIED:
   - LAND MASK: every tile centre must fall on land (Natural Earth 50 m land polygons).
     This removes offshore spreading ridges, subduction-trench traces and oceanic
     transforms directly -- the strongest observability guard.
   - SLIP-TYPE DROP: Spreading_Ridge and Subduction_Thrust (plate-boundary features, not
     mapped surface fault traces; overwhelmingly submarine), Blind Thrust (blind by
     definition), and the fold axes Anticline / Syncline (folds, not faults, and not in
     the slip-type class scheme), plus NULL slip_type, are dropped.
   - A wider dilation (2 px, ~40-50 m wide zone) partially absorbs positional error.

  RESIDUAL CAVEATS (see summary): even after filtering, alignment between a rasterized
  trace and the imaged geomorphic feature is imperfect (coarse source accuracy), and some
  retained continental faults have weak/no 10 m surface expression. Labels are therefore
  noisier than field-mapped reference data; downstream assembly filtering + the
  positive-only scheme mitigate this.
===================================================================================

Class scheme (id = slip-type family). Only surface-observable continental fault
kinematics are kept; the manifest "thrust" class is NOT represented because its only
GEM members are offshore Subduction_Thrust and Blind Thrust (both dropped) -- compressional
faulting is represented by "reverse" (thrust is kinematically a low-angle reverse fault):

  0 = normal        (extensional dip-slip: Normal)
  1 = reverse       (compressional dip-slip: Reverse)
  2 = strike-slip   (Dextral, Sinistral, Strike-Slip, Dextral/Sinistral Transform)
  3 = oblique       (mixed dip-/strike-slip: *-Reverse, *-Normal, *-Strike-Slip combos)

Non-fault pixels are nodata (255): POSITIVE-ONLY mask (spec S5) -- no fabricated
background class or negatives; the assembly step supplies negatives from other datasets.

Recipe (spec S4 lines + S5 bounded global product): fault LineStrings are partitioned
onto a ~640 m latitude-aware geographic grid; each occupied on-land cell becomes one
64x64 local-UTM 10 m tile centred on the cell, into which the fault segments in that cell
are rasterized (centreline buffered ~2 px, all_touched). A tile counts toward every
slip-type it contains -> tiles-per-class balanced sampling (rarest class first) to
<=1000 tiles/class, bounding the otherwise global product.

Time (spec S5, static labels): faults are persistent static features; each tile gets a
static 1-year window (change_time=null) spread deterministically over 2016-2019 for
imagery diversity.

Run (idempotent; skips already-written tiles):
  python3 -m olmoearth_pretrain.open_set_segmentation_data.datasets.gem_global_active_faults_database
"""

import argparse
import math
import multiprocessing
import random
from collections import Counter, defaultdict
from typing import Any

import numpy as np
import shapely
import shapely.prepared
from rasterio.crs import CRS
from rslearn.utils.geometry import Projection
from rslearn.utils.mp import star_imap_unordered

from olmoearth_pretrain.open_set_segmentation_data import (
    download,
    io,
    manifest,
    sampling,
)
from olmoearth_pretrain.open_set_segmentation_data.rasterize import (
    geom_to_pixels,
    rasterize_shapes,
)

SLUG = "gem_global_active_faults_database"
NAME = "GEM Global Active Faults Database"

GEOJSON_URL = (
    "https://raw.githubusercontent.com/GEMScienceTools/gem-global-active-faults/"
    "master/geojson/gem_active_faults_harmonized.geojson"
)
GEOJSON_NAME = "gem_active_faults_harmonized.geojson"
LAND_SHP = "ne_50m_land.shp"

# Source CRS is WGS84 lon/lat degrees. Projection resolution (1, 1) treats geometry
# coordinates (degrees) as pixel==CRS-unit so geom_to_pixels reprojects degrees -> local
# UTM pixels correctly (same convention as the grip4 line script).
SRC_PROJ = Projection(CRS.from_epsg(4326), 1, 1)

TILE = 64  # 640 m tiles.
CELL_M = TILE * io.RESOLUTION  # 640 m cell footprint.
DEG_PER_M_LAT = 1.0 / 111_320.0
DLAT = CELL_M * DEG_PER_M_LAT  # ~0.005749 deg latitude per 640 m.

DILATE_RADIUS_PX = 2.0  # buffer centreline ~2 px -> ~40-50 m wide zone at 10 m.
MIN_FAULT_PIXELS = 4  # drop tiles whose fault mask is a trivial sliver / empty.

PER_CLASS = 1000  # up to 1000 tiles per slip-type family (spec S5).
CAND_PER_CLASS = 6000  # candidate cells per class fed to the balancer (headroom).

YEARS = [2016, 2017, 2018, 2019]  # static-label windows spread for imagery diversity.

# Cell-id packing: cell_id = (iy + OFF) * MULT + (ix + OFF).
_OFF = 200_000
_MULT = 1_000_000

# slip_type (harmonized) -> class id. Everything not listed here is DROPPED (offshore
# plate-boundary features, blind thrusts, fold axes, and NULL); see module docstring.
NORMAL, REVERSE, STRIKE_SLIP, OBLIQUE = 0, 1, 2, 3
SLIP_TYPE_MAP: dict[str, int] = {
    "Normal": NORMAL,
    "Reverse": REVERSE,
    "Dextral": STRIKE_SLIP,
    "Sinistral": STRIKE_SLIP,
    "Strike-Slip": STRIKE_SLIP,
    "Dextral_Transform": STRIKE_SLIP,
    "Sinistral_Transform": STRIKE_SLIP,
    # oblique-slip combinations
    "Reverse-Strike-Slip": OBLIQUE,
    "Dextral-Reverse": OBLIQUE,
    "Sinistral-Reverse": OBLIQUE,
    "Dextral-Normal": OBLIQUE,
    "Sinistral-Normal": OBLIQUE,
    "Normal-Dextral": OBLIQUE,
    "Normal-Sinistral": OBLIQUE,
    "Reverse-Dextral": OBLIQUE,
    "Reverse-Sinistral": OBLIQUE,
    "Normal-Strike-Slip": OBLIQUE,
    "Dextral-Oblique": OBLIQUE,
}
# Explicitly-dropped slip types (documented; anything not in SLIP_TYPE_MAP is dropped):
#   Spreading_Ridge, Subduction_Thrust (offshore plate boundaries), Blind Thrust (blind),
#   Anticline, Syncline (folds), NULL/empty.

CLASSES = [
    {
        "id": NORMAL,
        "name": "normal",
        "description": (
            "Extensional (normal) dip-slip active fault trace (GEM slip_type 'Normal'). "
            "Hanging-wall drops relative to footwall; commonly forms range-front / basin "
            "scarps observable at 10-30 m."
        ),
    },
    {
        "id": REVERSE,
        "name": "reverse",
        "description": (
            "Compressional (reverse/thrust) dip-slip active fault trace (GEM slip_type "
            "'Reverse'). Hanging-wall rides up over footwall; forms fault scarps / uplifted "
            "range fronts. Represents thrust-sense faulting (thrust = low-angle reverse); "
            "the offshore Subduction_Thrust and Blind Thrust slip types are dropped."
        ),
    },
    {
        "id": STRIKE_SLIP,
        "name": "strike-slip",
        "description": (
            "Horizontal-slip active fault trace (GEM slip_type Dextral / Sinistral / "
            "Strike-Slip / Dextral_Transform / Sinistral_Transform). Forms linear "
            "fault-line valleys, offset drainages/ridges and sag ponds. Oceanic transform "
            "segments are removed by the land mask."
        ),
    },
    {
        "id": OBLIQUE,
        "name": "oblique",
        "description": (
            "Oblique-slip active fault trace: mixed dip- and strike-slip kinematics (GEM "
            "combined slip_types, e.g. Dextral-Reverse, Sinistral-Normal, "
            "Reverse-Strike-Slip)."
        ),
    },
]
CLASS_NAMES = {c["id"]: c["name"] for c in CLASSES}


# --------------------------------------------------------------------------- download


def ensure_raw() -> tuple[str, str]:
    """Download the GEM harmonized fault geojson; return (geojson_path, land_shp_path)."""
    raw = io.raw_dir(SLUG)
    raw.mkdir(parents=True, exist_ok=True)
    gj = raw / GEOJSON_NAME
    if not gj.exists():
        io.check_disk()
        print("downloading GEM active faults geojson ...", flush=True)
        download.download_http(GEOJSON_URL, gj)

    land = raw / LAND_SHP
    if not land.exists():
        # Natural Earth 50 m land polygons via cartopy's cached copy (small; used only as
        # an observability land mask). Copy into raw/ for provenance.
        import shutil

        import cartopy.io.shapereader as shpreader

        fn = shpreader.natural_earth(resolution="50m", category="physical", name="land")
        import os

        base = os.path.splitext(fn)[0]
        for ext in (".shp", ".shx", ".dbf", ".prj", ".cpg"):
            src = base + ext
            if os.path.exists(src):
                shutil.copy(src, (raw / (LAND_SHP[:-4] + ext)).path)

    with (raw / "SOURCE.txt").open("w") as f:
        f.write(
            "GEM Global Active Faults Database (GAF-DB) -- Styron & Pagani (2020), "
            "Earthquake Spectra 36(1_suppl):160-180, doi 10.1177/8755293020944182.\n"
            "GitHub: https://github.com/GEMScienceTools/gem-global-active-faults "
            "(CC-BY-SA-4.0).\n"
            f"File: geojson/{GEOJSON_NAME} (13,696 WGS84 LineStrings; slip_type = fault "
            "kinematics).\n"
            "Land mask: Natural Earth 50 m physical land polygons (public domain), used to "
            "keep only on-land (observable) fault traces.\n"
        )
    return gj.path, land.path


# --------------------------------------------------------------------------- grid math


def _grid_indices(lon: np.ndarray, lat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Vectorized ~640 m latitude-aware grid indices for arrays of lon/lat (degrees)."""
    iy = np.floor(lat / DLAT).astype(np.int64)
    latc = (iy + 0.5) * DLAT
    coslat = np.clip(np.cos(np.radians(latc)), 0.02, 1.0)
    dlon = DLAT / coslat
    ix = np.floor(lon / dlon).astype(np.int64)
    return ix, iy


def _cell_center(ix: int, iy: int) -> tuple[float, float]:
    """(ix, iy) -> (lon, lat) of the cell center (inverse of _grid_indices)."""
    latc = (iy + 0.5) * DLAT
    coslat = max(math.cos(math.radians(latc)), 0.02)
    dlon = DLAT / coslat
    lonc = (ix + 0.5) * dlon
    return lonc, latc


def _pack(ix: np.ndarray, iy: np.ndarray) -> np.ndarray:
    return (iy + _OFF) * _MULT + (ix + _OFF)


def _unpack(cell_id: int) -> tuple[int, int]:
    q, r = divmod(int(cell_id), _MULT)
    return int(r - _OFF), int(q - _OFF)


# --------------------------------------------------------------------------- load


def load_segments() -> tuple[np.ndarray, np.ndarray]:
    """Read the GEM geojson -> (geoms, class_ids) for mapped-slip-type faults.

    Filters slip_type to SLIP_TYPE_MAP (drops offshore/blind/fold/NULL).
    """
    import geopandas as gpd

    raw = io.raw_dir(SLUG)
    gdf = gpd.read_file((raw / GEOJSON_NAME).path, columns=["slip_type"])
    st = gdf["slip_type"].fillna("").to_numpy()
    cls = np.array([SLIP_TYPE_MAP.get(s, -1) for s in st], dtype=np.int64)
    keep = cls >= 0
    gdf = gdf[keep]
    cls = cls[keep]
    return gdf.geometry.to_numpy(), cls.astype(np.int8)


def build_cell_masks(geoms: np.ndarray, cls: np.ndarray) -> dict[int, int]:
    """Map every ~640 m grid cell that a fault line CROSSES to its class bitmask.

    Fault LineStrings are long and span many cells, so we densify each line (a vertex
    every ~half-cell) and record the grid cell of every sampled point -- assigning the
    line to ALL cells it passes through, not just its bbox center.
    """
    step = DLAT * 0.5  # ~320 m densification so consecutive samples stay within a cell.
    id_to_mask: dict[int, int] = defaultdict(int)
    for geom, c in zip(geoms.tolist(), cls.tolist()):
        dense = shapely.segmentize(geom, max_segment_length=step)
        xy = shapely.get_coordinates(dense)
        if xy.size == 0:
            continue
        ix, iy = _grid_indices(xy[:, 0], xy[:, 1])
        bit = 1 << int(c)
        for cid in np.unique(_pack(ix, iy)).tolist():
            id_to_mask[int(cid)] |= bit
    return dict(id_to_mask)


def select_cells(
    id_to_mask_all: dict[int, int], land_prep: Any
) -> list[dict[str, Any]]:
    """On-land, class-balanced (tiles-per-class) selection of grid cells."""
    # Land filter: keep only cells whose center is on land (observability guard).
    id_to_mask: dict[int, int] = {}
    for cid, m in id_to_mask_all.items():
        lon, lat = _cell_center(*_unpack(cid))
        if land_prep.contains(shapely.Point(lon, lat)):
            id_to_mask[cid] = m
    print(
        f"  {len(id_to_mask_all):,} crossed cells; {len(id_to_mask):,} on land",
        flush=True,
    )

    rng = random.Random(42)
    cand_ids: set[int] = set()
    for c in range(len(CLASSES)):
        has = [cid for cid, m in id_to_mask.items() if m & (1 << c)]
        rng.shuffle(has)
        cand_ids.update(has[:CAND_PER_CLASS])

    records: list[dict[str, Any]] = []
    for cid in cand_ids:
        m = id_to_mask[cid]
        present = [c for c in range(len(CLASSES)) if m & (1 << c)]
        records.append({"cell_id": int(cid), "classes_present": present})

    return sampling.balance_tiles_by_class(
        records,
        "classes_present",
        per_class=PER_CLASS,
        total_cap=sampling.MAX_SAMPLES_PER_DATASET,
    )


# --------------------------------------------------------------------------- write


def _write_one(rec: dict[str, Any]) -> str:
    sample_id = rec["sample_id"]
    tif = io.locations_dir(SLUG) / f"{sample_id}.tif"
    if tif.exists():
        return "skip"

    ix, iy = _unpack(rec["cell_id"])
    lon, lat = _cell_center(ix, iy)
    proj = io.utm_projection_for_lonlat(lon, lat)
    _, col, row = io.lonlat_to_utm_pixel(lon, lat, proj)
    bounds = io.centered_bounds(col, row, TILE, TILE)

    shapes: list[tuple[Any, int]] = []
    for wkb, cid in zip(rec["wkbs"], rec["clses"]):
        geom = shapely.from_wkb(wkb)
        try:
            pix = geom_to_pixels(geom, SRC_PROJ, proj)
        except Exception:
            continue
        if pix.is_empty:
            continue
        dil = pix.buffer(DILATE_RADIUS_PX)
        if dil.is_empty:
            continue
        shapes.append((dil, int(cid)))
    if not shapes:
        return "empty"

    arr = rasterize_shapes(
        shapes, bounds, fill=io.CLASS_NODATA, dtype="uint8", all_touched=True
    )[0]
    present = [int(v) for v in np.unique(arr) if v != io.CLASS_NODATA]
    if not present or int((arr != io.CLASS_NODATA).sum()) < MIN_FAULT_PIXELS:
        return "empty"

    year = YEARS[rec["cell_id"] % len(YEARS)]
    io.write_label_geotiff(SLUG, sample_id, arr, proj, bounds, nodata=io.CLASS_NODATA)
    io.write_sample_json(
        SLUG,
        sample_id,
        proj,
        bounds,
        io.year_range(year),
        change_time=None,
        source_id=f"gem_cell_{ix}_{iy}",
        classes_present=present,
    )
    return "written"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=64)
    parser.add_argument(
        "--probe", action="store_true", help="scan/report only, no writes"
    )
    args = parser.parse_args()

    io.check_disk()
    manifest.write_registry_entry(SLUG, "in_progress")

    gj_path, land_path = ensure_raw()

    print("loading Natural Earth land mask ...", flush=True)
    import geopandas as gpd

    land = gpd.read_file(land_path).geometry.union_all()
    land_prep = shapely.prepared.prep(land)

    print("reading GEM active faults ...", flush=True)
    geoms, cls = load_segments()
    print(f"  {len(cls):,} fault segments with a mapped slip-type family", flush=True)
    print(
        "class distribution (all mapped segments):",
        {CLASS_NAMES[k]: int(v) for k, v in sorted(Counter(cls.tolist()).items())},
        flush=True,
    )

    io.check_disk()

    print("indexing grid cells crossed by fault traces ...", flush=True)
    id_to_mask_all = build_cell_masks(geoms, cls)
    print("selecting on-land class-balanced cells ...", flush=True)
    selected = select_cells(id_to_mask_all, land_prep)
    sel_ids = {r["cell_id"] for r in selected}
    print(f"  {len(sel_ids):,} cells selected", flush=True)

    # Gather the actual fault lines intersecting each selected tile via a spatial index
    # (a line contributes to a tile if its geometry crosses the ~640 m tile footprint).
    from shapely.strtree import STRtree

    tree = STRtree(list(geoms))
    half = (
        DLAT * 0.6
    )  # tile half-extent in degrees (slightly > half-cell for edge lines).
    by_cell: dict[int, dict[str, list]] = {}
    for cid in sel_ids:
        lon, lat = _cell_center(*_unpack(cid))
        coslat = max(math.cos(math.radians(lat)), 0.02)
        tbox = shapely.box(
            lon - half / coslat, lat - half, lon + half / coslat, lat + half
        )
        hits = tree.query(tbox, predicate="intersects").tolist()
        if not hits:
            continue
        by_cell[cid] = {
            "wkbs": [shapely.to_wkb(geoms[i]) for i in hits],
            "clses": [int(cls[i]) for i in hits],
        }

    records = []
    for cid in sorted(by_cell):
        records.append(
            {
                "cell_id": cid,
                "wkbs": by_cell[cid]["wkbs"],
                "clses": by_cell[cid]["clses"],
            }
        )
    for i, r in enumerate(records):
        r["sample_id"] = f"{i:06d}"
    print(f"  {len(records):,} candidate tiles to rasterize", flush=True)

    if args.probe:
        print("probe only; exiting before writes")
        return

    results: Counter = Counter()
    with multiprocessing.Pool(args.workers) as p:
        for res in star_imap_unordered(p, _write_one, [dict(rec=r) for r in records]):
            results[res] += 1
    print("write results:", dict(results), flush=True)

    io.check_disk()

    # Recompute per-class tile counts from the tiles actually on disk (idempotent-stable).
    import json as _json

    class_tile_counts: Counter = Counter()
    year_counts: Counter = Counter()
    num_samples = 0
    for jp in io.locations_dir(SLUG).glob("*.json"):
        with jp.open() as f:
            meta = _json.load(f)
        num_samples += 1
        for c in meta.get("classes_present", []):
            class_tile_counts[int(c)] += 1
        year_counts[meta["time_range"][0][:4]] += 1
    print(
        "per-class tile counts:",
        {CLASS_NAMES[k]: v for k, v in sorted(class_tile_counts.items())},
        flush=True,
    )

    io.write_dataset_metadata(
        SLUG,
        {
            "dataset": SLUG,
            "name": NAME,
            "task_type": "classification",
            "source": "GEM Foundation (Global Active Faults Database)",
            "license": "CC-BY-SA-4.0",
            "provenance": {
                "url": "https://github.com/GEMScienceTools/gem-global-active-faults",
                "have_locally": False,
                "annotation_method": (
                    "expert-compiled / harmonized global active-fault trace compilation "
                    "(Styron & Pagani 2020)"
                ),
                "citation": (
                    "Styron, R. & Pagani, M. (2020). The GEM Global Active Faults "
                    "Database. Earthquake Spectra 36(1_suppl):160-180. "
                    "doi:10.1177/8755293020944182"
                ),
            },
            "sensors_relevant": ["sentinel2", "sentinel1", "landsat"],
            "classes": CLASSES,
            "nodata_value": io.CLASS_NODATA,
            "num_samples": num_samples,
            "class_tile_counts": {
                CLASS_NAMES[k]: v for k, v in sorted(class_tile_counts.items())
            },
            "anchor_year_counts": {k: v for k, v in sorted(year_counts.items())},
            "dilate_radius_px": DILATE_RADIUS_PX,
            "notes": (
                "Positive-only active-fault slip-type line segmentation. GEM Global Active "
                "Faults harmonized LineStrings (EPSG:4326) rasterized (centreline buffered "
                f"~{int(DILATE_RADIUS_PX)} px -> ~40-50 m wide, all_touched) into 64x64 "
                "local-UTM 10 m tiles. Classes: 0 normal, 1 reverse, 2 strike-slip, "
                "3 oblique; non-fault pixels = nodata (255). SUITABILITY: fault traces are "
                "geological lineaments only PARTIALLY observable at 10-30 m; accepted with "
                "strict filtering (spec S4 lines). OBSERVABILITY FILTERS: (a) Natural Earth "
                "50 m LAND MASK -- only on-land tile centres kept, removing offshore "
                "spreading ridges, subduction-trench traces and oceanic transforms; "
                "(b) slip_type DROP of Spreading_Ridge & Subduction_Thrust (offshore plate "
                "boundaries), Blind Thrust (blind), Anticline/Syncline (folds), and NULL. "
                "The manifest 'thrust' class is therefore NOT represented (its only members "
                "are offshore/blind); compressional faulting is under 'reverse'. Bounded "
                "sampling (spec S5): faults partitioned onto a ~640 m latitude-aware grid; "
                "on-land candidate cells class-balanced (tiles-per-class, rarest first) to "
                f"<= {PER_CLASS} tiles/class; a tile counts toward every slip-type it "
                f"contains; tiles with < {MIN_FAULT_PIXELS} fault px dropped. Faults are "
                "persistent static features; each tile gets a static 1-year window "
                "(change_time=null) spread over 2016-2019. CAVEATS: GEM trace positional "
                "accuracy is highly variable (source mapping scales as coarse as 1:100k-"
                "1:1M where recorded), so a rasterized line may not sit exactly on the "
                "imaged geomorphic feature; some retained continental faults have weak/no "
                "10 m surface expression. Labels are noisier than field reference data; the "
                "wider dilation partially absorbs positional error and downstream assembly "
                "filtering + the positive-only scheme mitigate residual noise."
            ),
        },
    )

    manifest.write_registry_entry(
        SLUG, "completed", task_type="classification", num_samples=num_samples
    )
    print(f"done: {num_samples} tiles", flush=True)


if __name__ == "__main__":
    multiprocessing.set_start_method("forkserver", force=True)
    main()
