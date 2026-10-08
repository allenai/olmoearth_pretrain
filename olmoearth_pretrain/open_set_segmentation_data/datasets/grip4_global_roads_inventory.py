"""GRIP4 Global Roads Inventory -> open-set-segmentation road-type line masks.

Source: GRIP4 (Global Roads Inventory Project, version 4) -- Meijer, Huijbregts,
Schotten & Schipper (2018), "Global patterns of current and future road
infrastructure", Environmental Research Letters 13:064006
(doi 10.1088/1748-9326/aabd42). A harmonized global vector roads database (~21.7 M km,
~21 M LineString segments) compiled by PBL Netherlands Environmental Assessment Agency
from ~60 national/regional/global sources (incl. OpenStreetMap) and finalized in 2018.
Distributed per-region as shapefiles at https://www.globio.info/download-grip-dataset
(PBL data portal). CRS EPSG:4326 (WGS84 lon/lat degrees).

Each segment carries a road type GP_RTP (the 5 manifest classes):
  1 Highways, 2 Primary roads, 3 Secondary roads, 4 Tertiary roads, 5 Local roads
  (0 = Unspecified -- dropped). Other attributes: GP_REX (existence: 1 open, 2
  restricted, 3 closed, 4 under-construction, 0 unspec), GP_RSE (surface), GP_RSY
  (year the source describes, all <= 2015), GP_RCY (country), gp_gripreg (region 1-7).

Recipe (spec S4 "lines"): rasterize road centerlines into thin dilated masks visible at
10 m/pixel. Five foreground classes (road type). A tile counts toward every road type it
contains -> tiles-per-class balanced sampling (spec S5), rarest class first (highways are
rarest), up to 1000 tiles/class.

Class map (id = GP_RTP - 1):
  0 = highways, 1 = primary, 2 = secondary, 3 = tertiary, 4 = local.
Non-road pixels are nodata (255): this is a POSITIVE-ONLY mask (spec S5) -- we do NOT
fabricate a background class or negative tiles; the assembly step supplies negatives from
other datasets. (Roads of a type not present in a tile are simply absent from that tile;
across tiles every type is represented.)

Suitability at 10 m: major roads (highways/primary/secondary) are 10-30 m wide and are
clearly resolvable at 10 m in Sentinel-2; tertiary/local roads are narrower (often
< 10 m) but are still visible as linear features and are exactly the kind of linear
infrastructure S2/S1 pretraining should learn. A centerline dilated ~1 px (-> ~20-30 m,
2-3 px) is a meaningful 10 m label. ACCEPTED.

Bounded sampling (spec S5, "large global derived-product rasters"): GRIP4 is a global
derived product; we do NOT process the whole planet. We download a representative subset
of REGIONS spanning multiple continents and development levels -- Region 1 (North
America, dense/developed), Region 3 (Africa, sparse/developing, much unpaved), Region 5
(Middle East & Central Asia, arid/mixed), Region 6 (South & East Asia, very dense),
Region 7 (Oceania, island/sparse) -- and draw a class-balanced bounded set of tiles from
them. Region 2 (Central & South America) and Region 4 (Europe) are omitted to keep the
download/processing bounded; the retained regions already span dense and sparse networks
on both hemispheres and contain all 5 road types.

Time (spec S5, static labels): roads are persistent static features (source years all
<= 2015, so every mapped road exists by the Sentinel era). We assign each tile a static
1-year window with change_time=None, spread deterministically over 2016-2018 (the manifest
range) for imagery diversity.

Tiling: road segments are partitioned onto a ~640 m latitude-aware geographic grid; each
occupied cell becomes one 64x64 (640 m) local-UTM 10 m tile centered on the cell center,
into which every segment assigned to the cell is rasterized (clipped). Candidate cells are
class-balanced (tiles-per-class) down to <=1000 tiles/class; tiles whose rasterized road
mask has < MIN_ROAD_PIXELS pixels are dropped.

Run (idempotent; skips already-written tiles):
  python3 -m olmoearth_pretrain.open_set_segmentation_data.datasets.grip4_global_roads_inventory
"""

import argparse
import math
import multiprocessing
import random
from collections import Counter, defaultdict
from typing import Any

import numpy as np
import shapely
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

SLUG = "grip4_global_roads_inventory"
NAME = "GRIP4 Global Roads Inventory"

# Representative subset of GRIP4 regions (see module docstring); global coverage is NOT
# attempted (spec S5). region -> (shp zip name, shp base name).
REGIONS = [1, 3, 5, 6, 7]
_ZIP = "GRIP4_Region{r}_vector_shp.zip"
_SHP = "GRIP4_region{r}.shp"
_BASE_URL = "https://dataportaal.pbl.nl/downloads/GRIP4/"

# Source CRS is WGS84 lon/lat degrees. With Projection resolution (1, 1) the geometry
# coordinates (degrees) are treated as pixel==CRS-unit, so geom_to_pixels reprojects
# degrees -> local UTM pixels correctly (same convention the congo/termpicks scripts use
# for their projected source CRS, here applied to the geographic CRS).
SRC_PROJ = Projection(CRS.from_epsg(4326), 1, 1)

TILE = 64  # 640 m tiles.
CELL_M = TILE * io.RESOLUTION  # 640 m cell footprint.
DEG_PER_M_LAT = 1.0 / 111_320.0
DLAT = CELL_M * DEG_PER_M_LAT  # ~0.005749 deg latitude per 640 m.

DILATE_RADIUS_PX = 1.0  # buffer the centerline ~1 px -> ~2-3 px (20-30 m) wide at 10 m.
MIN_ROAD_PIXELS = 3  # drop tiles whose road mask is a trivial sliver / empty.

PER_CLASS = 1000  # up to 1000 tiles per road type (spec S5).
CAND_PER_CLASS = 4000  # candidate cells per class fed to the balancer (headroom).

# Static-label 1-year windows spread over the manifest range for imagery diversity.
YEARS = [2016, 2017, 2018]

# Cell-id packing: cell_id = (iy + OFF) * MULT + (ix + OFF).
_OFF = 200_000
_MULT = 1_000_000

CLASSES = [
    {
        "id": 0,
        "name": "highways",
        "description": (
            "GRIP4 road type 1 (Highways): controlled-access / major trunk roads. "
            "Widest class (typ. 20-40 m incl. carriageways), clearly resolvable at 10 m."
        ),
    },
    {
        "id": 1,
        "name": "primary",
        "description": "GRIP4 road type 2 (Primary roads): major inter-city / arterial roads.",
    },
    {
        "id": 2,
        "name": "secondary",
        "description": "GRIP4 road type 3 (Secondary roads): important regional connector roads.",
    },
    {
        "id": 3,
        "name": "tertiary",
        "description": "GRIP4 road type 4 (Tertiary roads): local connector / minor through roads.",
    },
    {
        "id": 4,
        "name": "local",
        "description": (
            "GRIP4 road type 5 (Local roads): minor / residential / access roads. "
            "Narrowest class (often < 10 m), near the 10 m resolution limit."
        ),
    },
]
CLASS_NAMES = {c["id"]: c["name"] for c in CLASSES}


def _download_and_extract() -> None:
    import zipfile

    raw = io.raw_dir(SLUG)
    raw.mkdir(parents=True, exist_ok=True)
    for r in REGIONS:
        shp = raw / _SHP.format(r=r)
        if shp.exists():
            continue
        zp = raw / _ZIP.format(r=r)
        if not zp.exists():
            print(f"downloading region {r} ...", flush=True)
            download.download_http(_BASE_URL + _ZIP.format(r=r), zp)
        with zipfile.ZipFile(zp.path) as z:
            z.extractall(raw.path)
    with (raw / "SOURCE.txt").open("w") as f:
        f.write(
            "GRIP4 Global Roads Inventory Project v4 -- Meijer et al. 2018, "
            "Environmental Research Letters 13:064006 (doi 10.1088/1748-9326/aabd42). "
            "PBL Netherlands Environmental Assessment Agency.\n"
            "https://www.globio.info/download-grip-dataset  (per-region vector "
            "shapefiles from https://dataportaal.pbl.nl/downloads/GRIP4/)\n"
            f"Regions downloaded (representative subset, NOT global): {REGIONS}.\n"
            "CRS EPSG:4326. Road type field GP_RTP: 1 highways, 2 primary, 3 secondary, "
            "4 tertiary, 5 local, 0 unspecified. License CC0 (regional data ODbL where "
            "sourced from OpenStreetMap).\n"
        )


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


def load_segments() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Read the region shapefiles -> (geoms, class_ids, cell_ids) arrays.

    Filters to GP_RTP in 1..5 and drops GP_REX == 4 (under construction). class_id =
    GP_RTP - 1. cell_id is the ~640 m grid cell of each segment's bounding-box center.
    """
    import geopandas as gpd

    raw = io.raw_dir(SLUG)
    geoms_all: list[np.ndarray] = []
    cls_all: list[np.ndarray] = []
    cell_all: list[np.ndarray] = []
    for r in REGIONS:
        shp = (raw / _SHP.format(r=r)).path
        gdf = gpd.read_file(shp, columns=["GP_RTP", "GP_REX"], engine="pyogrio")
        rtp = gdf["GP_RTP"].to_numpy()
        rex = gdf["GP_REX"].to_numpy()
        keep = (rtp >= 1) & (rtp <= 5) & (rex != 4)
        gdf = gdf[keep]
        rtp = rtp[keep]
        geom = gdf.geometry.to_numpy()
        b = gdf.geometry.bounds.to_numpy()  # minx, miny, maxx, maxy
        lon = (b[:, 0] + b[:, 2]) * 0.5
        lat = (b[:, 1] + b[:, 3]) * 0.5
        ix, iy = _grid_indices(lon, lat)
        geoms_all.append(geom)
        cls_all.append((rtp - 1).astype(np.int8))
        cell_all.append(_pack(ix, iy))
        print(f"  region {r}: {len(rtp):,} segments kept", flush=True)
    geoms = np.concatenate(geoms_all)
    cls = np.concatenate(cls_all)
    cells = np.concatenate(cell_all)
    return geoms, cls, cells


def select_cells(cls: np.ndarray, cells: np.ndarray) -> list[dict[str, Any]]:
    """Class-balanced (tiles-per-class) selection of grid cells.

    Compute per-cell class bitmask, build a bounded candidate set (up to CAND_PER_CLASS
    cells per class, seeded), then apply the shared tiles-per-class balancer.
    """
    order = np.argsort(cells, kind="stable")
    sc = cells[order]
    sbit = 1 << cls[order].astype(np.int64)
    uniq, first = np.unique(sc, return_index=True)
    mask = np.bitwise_or.reduceat(sbit, first)  # per-cell OR of class bits

    rng = random.Random(42)
    cand_ids: set[int] = set()
    for c in range(len(CLASSES)):
        has = uniq[(mask & (1 << c)) != 0]
        has = has.tolist()
        rng.shuffle(has)
        cand_ids.update(has[:CAND_PER_CLASS])

    id_to_mask = {int(u): int(m) for u, m in zip(uniq, mask)}
    records: list[dict[str, Any]] = []
    for cid in cand_ids:
        m = id_to_mask[cid]
        present = [c for c in range(len(CLASSES)) if m & (1 << c)]
        records.append({"cell_id": int(cid), "classes_present": present})

    selected = sampling.balance_tiles_by_class(
        records,
        "classes_present",
        per_class=PER_CLASS,
        total_cap=sampling.MAX_SAMPLES_PER_DATASET,
    )
    return selected


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

    # Rasterize wider roads LAST so they win pixel conflicts (draw local first, highway
    # last): sort by class id descending.
    pairs = sorted(zip(rec["wkbs"], rec["clses"]), key=lambda p: -p[1])
    shapes: list[tuple[Any, int]] = []
    for wkb, cid in pairs:
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
    if not present or int((arr != io.CLASS_NODATA).sum()) < MIN_ROAD_PIXELS:
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
        source_id=f"grip4_cell_{ix}_{iy}",
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

    _download_and_extract()

    print("reading GRIP4 region shapefiles ...", flush=True)
    geoms, cls, cells = load_segments()
    print(f"  {len(cls):,} total segments across regions {REGIONS}", flush=True)
    print(
        "class distribution (all segments):",
        {CLASS_NAMES[k]: int(v) for k, v in sorted(Counter(cls.tolist()).items())},
        flush=True,
    )

    io.check_disk()

    print("selecting class-balanced cells ...", flush=True)
    selected = select_cells(cls, cells)
    sel_ids = {r["cell_id"] for r in selected}
    print(f"  {len(sel_ids):,} cells selected", flush=True)

    # Collect segment geometries for the selected cells only.
    keep = np.isin(cells, np.fromiter(sel_ids, dtype=np.int64))
    idxs = np.nonzero(keep)[0]
    by_cell: dict[int, dict[str, list]] = defaultdict(lambda: {"wkbs": [], "clses": []})
    for i in idxs:
        cid = int(cells[i])
        by_cell[cid]["wkbs"].append(shapely.to_wkb(geoms[i]))
        by_cell[cid]["clses"].append(int(cls[i]))

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
            "source": "PBL / GLOBIO (GRIP4)",
            "license": "CC0 (regional data ODbL where sourced from OpenStreetMap)",
            "provenance": {
                "url": "https://www.globio.info/download-grip-dataset",
                "have_locally": False,
                "annotation_method": (
                    "harmonized derived vector product compiled by PBL from ~60 "
                    "national/regional/global road sources incl. OpenStreetMap "
                    "(Meijer et al. 2018)"
                ),
                "citation": (
                    "Meijer J.R., Huijbregts M.A.J., Schotten K.C.G.J., Schipper A.M. "
                    "(2018). Global patterns of current and future road infrastructure. "
                    "Environmental Research Letters 13:064006. "
                    "doi:10.1088/1748-9326/aabd42"
                ),
            },
            "sensors_relevant": ["sentinel2", "sentinel1", "landsat"],
            "classes": CLASSES,
            "nodata_value": io.CLASS_NODATA,
            "num_samples": num_samples,
            "regions_sampled": REGIONS,
            "class_tile_counts": {
                CLASS_NAMES[k]: v for k, v in sorted(class_tile_counts.items())
            },
            "anchor_year_counts": {k: v for k, v in sorted(year_counts.items())},
            "notes": (
                "Positive-only road-type line segmentation. GRIP4 global road "
                "LineStrings (EPSG:4326, Meijer et al. 2018) rasterized (centerline "
                "buffered ~1 px -> ~20-30 m wide, all_touched) into 64x64 local-UTM 10 m "
                "tiles. Classes (id = GP_RTP-1): 0 highways, 1 primary, 2 secondary, "
                "3 tertiary, 4 local; non-road pixels = nodata (255). Wider road types "
                "are drawn last so they win pixel conflicts. Bounded sampling (spec S5): "
                f"a representative REGION subset {REGIONS} (North America, Africa, Middle "
                "East & Central Asia, South & East Asia, Oceania) was downloaded -- global "
                "coverage was NOT attempted; Region 2 (Central & South America) and Region "
                "4 (Europe) omitted. Segments partitioned onto a ~640 m latitude-aware "
                "geographic grid; candidate cells class-balanced (tiles-per-class, rarest "
                f"first) to <= {PER_CLASS} tiles/class; a tile counts toward every road "
                "type it contains. GP_REX==4 (under construction) segments dropped; "
                "GP_RTP==0 (unspecified) dropped. Tiles with < "
                f"{MIN_ROAD_PIXELS} road px dropped. Roads are persistent static features "
                "(source years <= 2015); each tile gets a static 1-year window "
                "(change_time=null) spread over 2016-2018 for imagery diversity. Caveat: "
                "GRIP4 is a harmonized derived product (incl. crowdsourced OSM), so "
                "omitted/misclassified roads and positional error of a few pixels are "
                "possible; local/tertiary roads are near the 10 m resolution limit."
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
