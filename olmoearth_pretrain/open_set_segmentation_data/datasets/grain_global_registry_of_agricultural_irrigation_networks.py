"""GRAIN (Global Registry of Agricultural Irrigation Networks) -> canal line masks.

Source: GRAIN v.1.0 -- a global OpenStreetMap-derived dataset of the world's
irrigation/canal centerlines, refined by an ML classifier that separates *agricultural*
canals from urban / navigational / natural waterways (Zenodo record 16786488,
doi:10.5281/zenodo.16786488; codebase https://github.com/SarathUW/GRAIN; ESSD). ~3.8 M km
of canal LineStrings across 95 countries, distributed as one 1.9 GB zip containing
per-country/region GeoParquet + ESRI shapefiles. Geometry CRS EPSG:4326 (WGS84 degrees).

Per-segment attributes include:
  - ``canal_use``: the semantic use class -- one of {Agricultural, Urban Waterway,
    Navigational Waterway, Other}. THIS is our class label (see mapping below).
  - ``predicted_class``: ML output {canal, Canal_natural}; we keep only ``canal``
    (Canal_natural = natural channel, not a built canal -> dropped).
  - ``osm_label`` (canal/ditch/drain/stream/river), ``length_KM``, ``elev_diff_M``,
    ``slope_MKM``, ``confidence`` (ML confidence), ``koppen_class_code``, ``osm_name``,
    ``grain_id``, ``osm_id``.

Class map (the 3 manifest classes = the three built-canal ``canal_use`` values;
``Other`` is dropped as semantically ambiguous):
  0 = irrigation_canal        (canal_use == "Agricultural")
  1 = urban_canal             (canal_use == "Urban Waterway")
  2 = navigational_waterway   (canal_use == "Navigational Waterway")
Non-canal pixels are nodata (255): this is a POSITIVE-ONLY mask (spec S5) -- we do NOT
fabricate a background class or negative tiles; the assembly step supplies negatives from
other datasets.

Recipe (spec S4 "lines"): rasterize canal centerlines into thin dilated masks visible at
10 m/pixel. A tile counts toward every canal class it contains -> tiles-per-class balanced
sampling (spec S5), rarest class first (navigational_waterway is rarest), up to 1000
tiles/class.

Suitability at 10 m (spec S4 caveat, "only large canals/aqueducts discernible"): GRAIN has
no width attribute. Major irrigation/navigational canals and aqueducts are 10-50 m wide and
clearly resolvable at 10 m in Sentinel-2; many minor field ditches are sub-pixel (< 10 m
wide). We rasterize each centerline dilated ~1 px (-> ~2-3 px, 20-30 m wide, all_touched)
so every canal becomes a nominal linear label at 10 m, and we drop tiles whose rasterized
canal mask is a trivial sliver (< MIN_CANAL_PIXELS px). This keeps the label as
"where a mapped canal runs" -- a meaningful linear-infrastructure target for S2/S1/Landsat
-- while acknowledging that the narrowest ditches are near/below the resolution limit
(noted in the summary). ACCEPTED.

Bounded sampling (spec S5, "large global derived-product rasters"): GRAIN is a global
product; we do NOT process all 95 countries / 3.8 M km. We SELECTIVELY extract (via HTTP
Range requests into the remote zip -- no 1.9 GB bulk download) the GeoParquet for a
representative COUNTRY subset spanning every inhabited continent and a range of climates
(arid, temperate, tropical) and including the navigational-canal-rich basins of Europe /
China / the US, then draw a class-balanced bounded set of tiles from them.

Time (spec S5, static labels): canals are persistent static infrastructure (OSM, update
date 2025; manifest range 2016-2025). Each tile gets a static 1-year window
(change_time=None) spread deterministically over 2019-2024 for imagery diversity.

Tiling: canal segments are partitioned onto a ~640 m latitude-aware geographic grid; each
occupied cell becomes one 64x64 (640 m) local-UTM 10 m tile centered on the cell center,
into which every segment assigned to the cell is rasterized (clipped). Candidate cells are
class-balanced (tiles-per-class) down to <=1000 tiles/class; tiles whose rasterized canal
mask has < MIN_CANAL_PIXELS pixels are dropped.

Run (idempotent; skips already-written tiles):
  python3 -m olmoearth_pretrain.open_set_segmentation_data.datasets.grain_global_registry_of_agricultural_irrigation_networks
"""

import argparse
import math
import multiprocessing
import random
import zipfile
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

SLUG = "grain_global_registry_of_agricultural_irrigation_networks"
NAME = "GRAIN (Global Registry of Agricultural Irrigation Networks)"

ZENODO_URL = "https://zenodo.org/api/records/16786488/files/GRAIN_v.1.0.zip/content"
_ZIP_MEMBER = "GRAIN_v.1.0/GeoParquet/{country}_GRAIN_v.1.0.parquet"

# Representative bounded COUNTRY subset (see module docstring); global coverage is NOT
# attempted (spec S5). Chosen to span every inhabited continent and arid/temperate/tropical
# climates, and to include the navigational-canal-rich networks of Europe, China and the US
# (so the rare navigational_waterway class reaches its target).
COUNTRIES = [
    # Europe (navigational-canal rich + temperate agriculture)
    "netherlands",
    "france",
    "poland",
    "spain",
    "sweden",
    "england",
    # Asia (monsoon / arid irrigation, dense canal networks)
    "china",
    "india_northern-zone",
    "india_southern-zone",
    "pakistan",
    "bangladesh",
    "indonesia",
    "vietnam",
    "japan",
    # Americas
    "us-midwest",
    "us-south",
    "us-west",
    "mexico",
    "brazil",
    "argentina",
    # Africa (arid Nile / semi-arid)
    "egypt",
    "south-africa",
    # Oceania
    "australia",
    # Middle East (arid, major irrigation)
    "iran",
    "iraq",
]

# GRAIN geometry CRS is WGS84 lon/lat degrees. With Projection resolution (1, 1) the
# geometry coordinates (degrees) are treated as pixel==CRS-unit, so geom_to_pixels
# reprojects degrees -> local UTM pixels correctly (same convention grip4 uses).
SRC_PROJ = Projection(CRS.from_epsg(4326), 1, 1)

TILE = 64  # 640 m tiles.
CELL_M = TILE * io.RESOLUTION  # 640 m cell footprint.
DEG_PER_M_LAT = 1.0 / 111_320.0
DLAT = CELL_M * DEG_PER_M_LAT  # ~0.005749 deg latitude per 640 m.

DILATE_RADIUS_PX = 1.0  # buffer the centerline ~1 px -> ~2-3 px (20-30 m) wide at 10 m.
MIN_CANAL_PIXELS = 3  # drop tiles whose canal mask is a trivial sliver / empty.

PER_CLASS = 1000  # up to 1000 tiles per canal class (spec S5).
CAND_PER_CLASS = 4000  # candidate cells per class fed to the balancer (headroom).

# Static-label 1-year windows spread over recent Sentinel-era years for imagery diversity.
YEARS = [2019, 2020, 2021, 2022, 2023, 2024]

# Cell-id packing: cell_id = (iy + OFF) * MULT + (ix + OFF).
_OFF = 200_000
_MULT = 1_000_000

# canal_use value -> class id.
USE_TO_CLASS = {
    "Agricultural": 0,
    "Urban Waterway": 1,
    "Navigational Waterway": 2,
}

CLASSES = [
    {
        "id": 0,
        "name": "irrigation_canal",
        "description": (
            "GRAIN canal_use == 'Agricultural': built canals/ditches/drains whose ML- and "
            "context-inferred use is agricultural irrigation or drainage. The dominant "
            "GRAIN class and the dataset's namesake (agricultural irrigation networks)."
        ),
    },
    {
        "id": 1,
        "name": "urban_canal",
        "description": (
            "GRAIN canal_use == 'Urban Waterway': built canals within/serving urban areas "
            "(stormwater, ornamental, conveyance) rather than agricultural fields."
        ),
    },
    {
        "id": 2,
        "name": "navigational_waterway",
        "description": (
            "GRAIN canal_use == 'Navigational Waterway': canals/canalized waterways used "
            "for navigation/shipping (e.g. barge canals). Rarest class; typically the "
            "widest and most clearly resolvable at 10 m."
        ),
    },
]
CLASS_NAMES = {c["id"]: c["name"] for c in CLASSES}


def _extract_country_parquets() -> None:
    """Selectively extract the chosen countries' GeoParquet from the remote zip.

    Uses HTTP Range requests into the Zenodo zip (download.HttpRangeFile + zipfile) so we
    fetch ONLY the per-country GeoParquet members we need (a few hundred MB) rather than
    bulk-downloading the whole 1.9 GB archive (spec S8 selective extraction). Idempotent:
    skips countries already on disk under raw/{slug}/GeoParquet/.
    """
    raw = io.raw_dir(SLUG)
    pq_dir = raw / "GeoParquet"
    pq_dir.mkdir(parents=True, exist_ok=True)

    missing = [
        c for c in COUNTRIES if not (pq_dir / f"{c}_GRAIN_v.1.0.parquet").exists()
    ]
    if missing:
        print(
            f"selectively extracting {len(missing)} country parquet(s) from zenodo zip "
            "via range requests ...",
            flush=True,
        )
        rf = download.HttpRangeFile(ZENODO_URL)
        try:
            zf = zipfile.ZipFile(rf)
            for c in missing:
                member = _ZIP_MEMBER.format(country=c)
                dst = pq_dir / f"{c}_GRAIN_v.1.0.parquet"
                tmp = pq_dir / f"{c}_GRAIN_v.1.0.parquet.tmp"
                data = zf.read(member)
                with tmp.open("wb") as f:
                    f.write(data)
                tmp.rename(dst)
                print(f"  {c}: {len(data) / 1e6:.1f} MB", flush=True)
            print(
                f"  fetched ~{rf.n_bytes / 1e6:.0f} MB total via "
                f"{rf.n_requests} range requests",
                flush=True,
            )
        finally:
            rf.close()

    with (raw / "SOURCE.txt").open("w") as f:
        f.write(
            "GRAIN v.1.0 -- Global Registry of Agricultural Irrigation Networks.\n"
            "Zenodo record 16786488 (doi:10.5281/zenodo.16786488), CC-BY-4.0. "
            "Codebase: https://github.com/SarathUW/GRAIN\n"
            "Global OSM-derived canal centerlines refined by ML to separate agricultural "
            "canals; ~3.8 M km / 95 countries. Distributed as one 1.9 GB zip of per-country "
            "GeoParquet + ESRI shapefiles (geometry EPSG:4326).\n"
            "We SELECTIVELY extracted (HTTP Range into the zip) only the GeoParquet for a "
            f"representative bounded COUNTRY subset (NOT global): {COUNTRIES}.\n"
            "Class field canal_use: Agricultural -> irrigation_canal, Urban Waterway -> "
            "urban_canal, Navigational Waterway -> navigational_waterway, Other -> dropped. "
            "predicted_class == 'canal' kept (Canal_natural dropped).\n"
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
    """Read the country GeoParquet files -> (geoms, class_ids, cell_ids) arrays.

    Filters to predicted_class == 'canal' and canal_use in USE_TO_CLASS; class_id =
    USE_TO_CLASS[canal_use]. cell_id is the ~640 m grid cell of each segment's
    bounding-box center. Drops empty/degenerate geometries.
    """
    import geopandas as gpd

    pq_dir = io.raw_dir(SLUG) / "GeoParquet"
    geoms_all: list[np.ndarray] = []
    cls_all: list[np.ndarray] = []
    cell_all: list[np.ndarray] = []
    for c in COUNTRIES:
        path = (pq_dir / f"{c}_GRAIN_v.1.0.parquet").path
        gdf = gpd.read_parquet(
            path, columns=["predicted_class", "canal_use", "geometry"]
        )
        use = gdf["canal_use"].to_numpy()
        pred = gdf["predicted_class"].to_numpy()
        cls = np.array([USE_TO_CLASS.get(u, -1) for u in use], dtype=np.int64)
        keep = (cls >= 0) & (pred == "canal")
        gdf = gdf[keep]
        cls = cls[keep]
        if len(gdf) == 0:
            print(f"  {c}: 0 kept", flush=True)
            continue
        geom = gdf.geometry.to_numpy()
        b = gdf.geometry.bounds.to_numpy()  # minx, miny, maxx, maxy
        lon = (b[:, 0] + b[:, 2]) * 0.5
        lat = (b[:, 1] + b[:, 3]) * 0.5
        ix, iy = _grid_indices(lon, lat)
        geoms_all.append(geom)
        cls_all.append(cls.astype(np.int8))
        cell_all.append(_pack(ix, iy))
        print(
            f"  {c}: {len(cls):,} segments kept {dict(Counter(cls.tolist()))}",
            flush=True,
        )
    geoms = np.concatenate(geoms_all)
    cls = np.concatenate(cls_all)
    cells = np.concatenate(cell_all)
    return geoms, cls, cells


def select_cells(cls: np.ndarray, cells: np.ndarray) -> list[dict[str, Any]]:
    """Class-balanced (tiles-per-class) selection of grid cells.

    Compute per-cell class bitmask, build a bounded candidate set (up to CAND_PER_CLASS
    cells per class, seeded), then apply the shared tiles-per-class balancer (rarest
    class -- navigational_waterway -- filled first).
    """
    order = np.argsort(cells, kind="stable")
    sc = cells[order]
    sbit = 1 << cls[order].astype(np.int64)
    uniq, first = np.unique(sc, return_index=True)
    mask = np.bitwise_or.reduceat(sbit, first)  # per-cell OR of class bits

    rng = random.Random(42)
    cand_ids: set[int] = set()
    for c in range(len(CLASSES)):
        has = uniq[(mask & (1 << c)) != 0].tolist()
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

    # Draw rarer classes LAST so they win pixel conflicts: sort by class id ascending
    # (irrigation first, navigational last).
    pairs = sorted(zip(rec["wkbs"], rec["clses"]), key=lambda p: p[1])
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
    if not present or int((arr != io.CLASS_NODATA).sum()) < MIN_CANAL_PIXELS:
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
        source_id=f"grain_cell_{ix}_{iy}",
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

    _extract_country_parquets()

    print("reading GRAIN country GeoParquet files ...", flush=True)
    geoms, cls, cells = load_segments()
    print(
        f"  {len(cls):,} total canal segments across {len(COUNTRIES)} countries",
        flush=True,
    )
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
            "source": "Zenodo / ESSD (GRAIN v.1.0)",
            "license": "CC-BY-4.0",
            "provenance": {
                "url": "https://doi.org/10.5281/zenodo.16786488",
                "have_locally": False,
                "annotation_method": (
                    "OpenStreetMap canal centerlines refined by an ML classifier that "
                    "separates agricultural canals from urban/navigational/natural "
                    "waterways; validated (GRAIN v.1.0)."
                ),
                "codebase": "https://github.com/SarathUW/GRAIN",
            },
            "sensors_relevant": ["sentinel2", "sentinel1", "landsat"],
            "classes": CLASSES,
            "nodata_value": io.CLASS_NODATA,
            "num_samples": num_samples,
            "countries_sampled": COUNTRIES,
            "class_tile_counts": {
                CLASS_NAMES[k]: v for k, v in sorted(class_tile_counts.items())
            },
            "anchor_year_counts": {k: v for k, v in sorted(year_counts.items())},
            "notes": (
                "Positive-only canal-type line segmentation. GRAIN v.1.0 global canal "
                "LineStrings (EPSG:4326, OSM refined by ML) rasterized (centerline buffered "
                "~1 px -> ~20-30 m wide, all_touched) into 64x64 local-UTM 10 m tiles. "
                "Classes from canal_use: 0 irrigation_canal (Agricultural), 1 urban_canal "
                "(Urban Waterway), 2 navigational_waterway (Navigational Waterway); "
                "canal_use=='Other' and predicted_class=='Canal_natural' dropped. Non-canal "
                "pixels = nodata (255); rarer classes drawn last so they win pixel conflicts. "
                "Bounded sampling (spec S5): a representative bounded COUNTRY subset "
                f"{COUNTRIES} was SELECTIVELY extracted (HTTP Range into the Zenodo zip -- no "
                "1.9 GB bulk download); global coverage was NOT attempted. Segments "
                "partitioned onto a ~640 m latitude-aware geographic grid; candidate cells "
                "class-balanced (tiles-per-class, rarest first) to <= "
                f"{PER_CLASS} tiles/class; a tile counts toward every canal class it "
                f"contains. Tiles with < {MIN_CANAL_PIXELS} canal px dropped. Canals are "
                "persistent static features; each tile gets a static 1-year window "
                "(change_time=null) spread over 2019-2024 for imagery diversity. Caveat: "
                "GRAIN has no width attribute and is OSM-derived; major irrigation / "
                "navigational canals and aqueducts are resolvable at 10 m, but the narrowest "
                "field ditches are near/below the 10 m limit and the ~20-30 m dilated label "
                "is nominal for those; positional error of a few pixels and OSM "
                "omissions/misclassifications are possible."
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
