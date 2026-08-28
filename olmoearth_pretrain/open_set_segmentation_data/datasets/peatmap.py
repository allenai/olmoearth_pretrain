"""Process PEATMAP (global peatland extent) into open-set-segmentation label patches.

Source: PEATMAP (Xu, Morris, Liu, Holden 2017/2018), University of Leeds research data
archive record 251 (https://archive.researchdata.leeds.ac.uk/251/, DOI 10.5518/252,
CC-BY-4.0). PEATMAP is a meta-analysis harmonizing the best available global / regional /
national peat maps into a single global set of **peatland extent polygons**, delivered as
per-continent ESRI shapefiles in the World Cylindrical Equal Area projection (ESRI:54034,
metres). We use every continent shapefile:

  Africa, Asia (E/NE/SE/Siberia + Histosols), Europe (British Isles, Finland, Norway,
  Sweden, Other), North America (Canada, USA, Other), Oceania, South America.

Binary peatland-vs-background segmentation. Class scheme (uint8):
  0 = background  (non-peat land / water)
  1 = peatland    (inside a PEATMAP polygon)
  255 = nodata/ignore (unused here; declared for consistency)

Each label is a 64x64 (640 m) tile at 10 m/pixel in the local UTM zone. Positive tiles are
centered on interior points of peatland polygons (peat polygons intersecting the tile are
rasterized as class 1, everything else 0). Background-only negatives are drawn away from
any peatland (all class 0), so the background class has genuine negatives.

PEATMAP is a large global derived product (~316k polygons); per spec we draw a **bounded,
regionally-diverse** sample rather than global coverage: an even per-continent quota of
peatland-positive tiles (area-weighted interior points within each continent) plus an equal
number of negatives. Static baseline -> a representative 1-year window in the Sentinel era.

Run (idempotent):
  python3 -m olmoearth_pretrain.open_set_segmentation_data.datasets.peatmap
"""

import argparse
import glob
import multiprocessing
import random
from collections import Counter, defaultdict
from typing import Any

import numpy as np
import shapely
import tqdm
from pyogrio import read_dataframe
from rasterio.crs import CRS
from rslearn.const import WGS84_PROJECTION
from rslearn.utils.geometry import Projection, STGeometry
from rslearn.utils.mp import star_imap_unordered
from shapely import STRtree

from olmoearth_pretrain.open_set_segmentation_data import io
from olmoearth_pretrain.open_set_segmentation_data.rasterize import (
    geom_to_pixels,
    rasterize_shapes,
)

SLUG = "peatmap"
NAME = "PEATMAP"

# PEATMAP source geometries are in ESRI:54034 (World Cylindrical Equal Area), metres.
SRC_CRS = CRS.from_user_input("ESRI:54034")
SRC_PROJ = Projection(SRC_CRS, 1, 1)

# Class scheme.
CID_BACKGROUND = 0
CID_PEATLAND = 1
CLASSES = [
    {
        "id": CID_BACKGROUND,
        "name": "background",
        "description": "Non-peatland: any 10 m pixel outside a PEATMAP peatland polygon "
        "(other land cover or water).",
    },
    {
        "id": CID_PEATLAND,
        "name": "peatland",
        "description": "Peatland extent per PEATMAP: land with a peat (organic) soil layer, "
        "harmonized from the best available global/regional/national peat maps "
        "(Xu et al. 2018 meta-analysis). Includes bogs, fens, mires and tropical peat "
        "swamp forest wherever the source maps classify the substrate as peat.",
    },
]

# Sampling parameters (binary -> per-class balanced, following the rubber precedent).
PER_CLASS = 1000  # peatland-positive tiles
N_NEGATIVES = 1000  # background-only negative tiles
TILE = 64  # 64x64 @ 10 m = 640 m tiles
REPRESENTATIVE_YEAR = 2020  # static baseline -> representative Sentinel-era year
QUERY_PAD_M = (
    2000.0  # padding (metres, in ESRI:54034) when clipping candidate peat geoms
)
SEED = 42

# Continent -> shapefile glob (relative to raw_dir). All continents used.
CONTINENT_GLOBS = {
    "Africa": "Africa/*/*.shp",
    "Asia": "Asia/*/*.shp",
    "Europe": "Europe/*/*.shp",
    "North_America": "North_America/*/*.shp",
    "Oceania": "Oceania/*/*.shp",
    "South_America": "South_America/*/*.shp",
}


# --------------------------------------------------------------------------------------
# Loading source geometries.
# --------------------------------------------------------------------------------------
def load_continent(pattern: str) -> np.ndarray:
    """Load all (2D, valid) polygon geometries from a continent's shapefiles.

    Uses pyogrio for fast vectorized reads (fiona's per-feature Python loop is ~50x
    slower on the 145k-polygon Finland layer). Returns a numpy object array of shapely
    geometries.
    """
    raw = io.raw_dir(SLUG)
    chunks: list[np.ndarray] = []
    for shp in sorted(glob.glob(str(raw / pattern))):
        gdf = read_dataframe(shp, read_geometry=True, columns=[])
        if len(gdf):
            chunks.append(np.asarray(gdf.geometry.values, dtype=object))
    if not chunks:
        return np.empty(0, dtype=object)
    geoms = np.concatenate(chunks)
    geoms = shapely.force_2d(geoms)
    geoms = geoms[np.array([g is not None for g in geoms])]
    geoms = geoms[~shapely.is_empty(geoms)]
    # NOTE: we do NOT make_valid the full polygons here -- make_valid on the few hundred
    # self-intersecting source polygons is pathologically slow (minutes). GEOS
    # clip_by_rect / area / STRtree tolerate invalid input; the only place validity
    # matters is the small clipped candidate geometry, which is repaired lazily in
    # _clip_candidates (cheap, since it is already bounded to a ~4 km box).
    return geoms


def _polygon_parts(geoms: np.ndarray) -> np.ndarray:
    """Explode an array of geometries into constituent single Polygons."""
    parts = shapely.get_parts(geoms)
    return parts[shapely.get_type_id(parts) == 3]  # 3 == Polygon


def _sample_interior_point(poly: Any, rng: random.Random) -> tuple[float, float]:
    """Sample a random interior point of a single polygon (fallback: representative point)."""
    minx, miny, maxx, maxy = poly.bounds
    for _ in range(40):
        x = rng.uniform(minx, maxx)
        y = rng.uniform(miny, maxy)
        if poly.contains(shapely.Point(x, y)):
            return x, y
    p = poly.representative_point()
    return p.x, p.y


def _to_wgs84(x54: float, y54: float) -> tuple[float, float]:
    """ESRI:54034 (x, y) metres -> (lon, lat) degrees."""
    st = STGeometry(SRC_PROJ, shapely.Point(x54, y54), None).to_projection(
        WGS84_PROJECTION
    )
    return float(st.shp.x), float(st.shp.y)


# --------------------------------------------------------------------------------------
# Building sample records (main process; attaches clipped peat geometry WKB).
# --------------------------------------------------------------------------------------
def _clip_candidates(
    tree: STRtree, geoms: list[Any], x54: float, y54: float
) -> list[bytes]:
    """Return WKB of peat geometry within a padded box around (x54, y54), clipped to it."""
    box = shapely.box(
        x54 - QUERY_PAD_M, y54 - QUERY_PAD_M, x54 + QUERY_PAD_M, y54 + QUERY_PAD_M
    )
    out: list[bytes] = []
    for idx in tree.query(box):
        g = geoms[idx]
        try:
            clipped = shapely.clip_by_rect(
                g,
                x54 - QUERY_PAD_M,
                y54 - QUERY_PAD_M,
                x54 + QUERY_PAD_M,
                y54 + QUERY_PAD_M,
            )
        except shapely.errors.GEOSException:
            clipped = shapely.clip_by_rect(
                shapely.make_valid(g),
                x54 - QUERY_PAD_M,
                y54 - QUERY_PAD_M,
                x54 + QUERY_PAD_M,
                y54 + QUERY_PAD_M,
            )
        if clipped.is_empty:
            continue
        if not clipped.is_valid:
            clipped = shapely.make_valid(clipped)
            if clipped.is_empty:
                continue
        out.append(shapely.to_wkb(clipped))
    return out


def _has_peat(tree: STRtree, geoms: list[Any], x54: float, y54: float) -> bool:
    """True if any peat polygon intersects a padded box around (x54, y54)."""
    box = shapely.box(
        x54 - QUERY_PAD_M, y54 - QUERY_PAD_M, x54 + QUERY_PAD_M, y54 + QUERY_PAD_M
    )
    for idx in tree.query(box):
        if geoms[idx].intersects(box):
            return True
    return False


def build_positive_records(
    continent: str,
    geoms: np.ndarray,
    tree: STRtree,
    n: int,
    rng: random.Random,
) -> list[dict[str, Any]]:
    """Area-weighted interior-point positives within a continent."""
    # Explode to single polygons; area-weight (cumulative distribution computed once).
    polys = _polygon_parts(geoms)
    if len(polys) == 0:
        return []
    areas = shapely.area(polys)
    total = areas.sum()
    if total <= 0:
        return []
    cum = np.cumsum(areas)
    cum = cum / cum[-1]
    recs: list[dict[str, Any]] = []
    attempts = 0
    seen: set[tuple[int, int]] = set()
    while len(recs) < n and attempts < n * 20:
        attempts += 1
        pi = min(int(np.searchsorted(cum, rng.random())), len(polys) - 1)
        x54, y54 = _sample_interior_point(polys[pi], rng)
        # Dedup near-identical centers (200 m grid in 54034).
        key = (int(x54 // 200), int(y54 // 200))
        if key in seen:
            continue
        seen.add(key)
        lon, lat = _to_wgs84(x54, y54)
        recs.append(
            {
                "kind": "positive",
                "continent": continent,
                "lon": lon,
                "lat": lat,
                "peat_wkb": _clip_candidates(tree, geoms, x54, y54),
                "source_id": f"{continent}/pos/{len(recs)}",
            }
        )
    return recs


def build_negative_records(
    positives_by_continent: dict[str, list[dict[str, Any]]],
    trees: dict[str, STRtree],
    geoms_by_continent: dict[str, list[Any]],
    n: int,
    rng: random.Random,
) -> list[dict[str, Any]]:
    """Background-only negatives: offset from a peat point, rejected if peat is nearby."""
    recs: list[dict[str, Any]] = []
    continents = [c for c, ps in positives_by_continent.items() if ps]
    attempts = 0
    while len(recs) < n and attempts < n * 60:
        attempts += 1
        cont = continents[rng.randrange(len(continents))]
        base = positives_by_continent[cont][
            rng.randrange(len(positives_by_continent[cont]))
        ]
        # Offset the base point in ESRI:54034 metres.
        st = STGeometry(WGS84_PROJECTION, shapely.Point(base["lon"], base["lat"]), None)
        p54 = st.to_projection(SRC_PROJ).shp
        ang = rng.uniform(0, 2 * np.pi)
        dist = rng.uniform(30000, 120000)  # 30-120 km
        x54 = p54.x + dist * np.cos(ang)
        y54 = p54.y + dist * np.sin(ang)
        if _has_peat(trees[cont], geoms_by_continent[cont], x54, y54):
            continue
        lon, lat = _to_wgs84(x54, y54)
        if not (-60 <= lat <= 78):
            continue
        recs.append(
            {
                "kind": "negative",
                "continent": cont,
                "lon": lon,
                "lat": lat,
                "peat_wkb": [],
                "source_id": f"{cont}/neg/{len(recs)}",
            }
        )
    return recs


# --------------------------------------------------------------------------------------
# Writer (runs in worker processes).
# --------------------------------------------------------------------------------------
def _write_tile(rec: dict[str, Any]) -> str:
    sample_id = rec["sample_id"]
    tif = io.locations_dir(SLUG) / f"{sample_id}.tif"
    if tif.exists():
        return "skip"
    proj = io.utm_projection_for_lonlat(rec["lon"], rec["lat"])
    _, col, row = io.lonlat_to_utm_pixel(rec["lon"], rec["lat"], proj)
    bounds = io.centered_bounds(col, row, TILE, TILE)
    shapes: list[tuple[Any, int]] = []
    for wkb in rec["peat_wkb"]:
        geom = shapely.from_wkb(wkb)
        if geom.is_empty:
            continue
        pix = geom_to_pixels(geom, SRC_PROJ, proj)
        if pix.is_empty:
            continue
        # make_valid/clip can yield collections; rasterize only the polygonal parts.
        for part in shapely.get_parts(pix):
            if part.geom_type == "Polygon" and not part.is_empty:
                shapes.append((part, CID_PEATLAND))
    if shapes:
        arr = rasterize_shapes(
            shapes, bounds, fill=CID_BACKGROUND, dtype="uint8", all_touched=True
        )
    else:
        arr = np.full((1, TILE, TILE), CID_BACKGROUND, dtype=np.uint8)
    io.write_label_geotiff(SLUG, sample_id, arr, proj, bounds, nodata=io.CLASS_NODATA)
    io.write_sample_json(
        SLUG,
        sample_id,
        proj,
        bounds,
        io.year_range(REPRESENTATIVE_YEAR),
        source_id=rec["source_id"],
        classes_present=sorted(set(np.unique(arr).tolist()) - {io.CLASS_NODATA}),
    )
    return "peatland" if shapes and CID_PEATLAND in np.unique(arr) else "background"


# --------------------------------------------------------------------------------------
# Main.
# --------------------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=64)
    args = parser.parse_args()

    io.check_disk()

    raw = io.raw_dir(SLUG)
    raw.mkdir(parents=True, exist_ok=True)
    with (raw / "SOURCE.txt").open("w") as f:
        f.write(
            "PEATMAP: global peatland extent polygons (Xu, Morris, Liu, Holden 2018).\n"
            "University of Leeds research data archive record 251, DOI 10.5518/252, CC-BY-4.0.\n"
            "https://archive.researchdata.leeds.ac.uk/251/\n"
            "Per-continent ESRI shapefiles in ESRI:54034 (World Cylindrical Equal Area).\n"
        )

    rng = random.Random(SEED)

    # Load all continents and build per-continent spatial indexes.
    geoms_by_continent: dict[str, list[Any]] = {}
    trees: dict[str, STRtree] = {}
    for cont, pattern in CONTINENT_GLOBS.items():
        g = load_continent(pattern)
        geoms_by_continent[cont] = g
        trees[cont] = STRtree(g)
        print(f"  {cont}: {len(g)} polygons")
    io.check_disk()

    # Even per-continent positive quota for regional diversity.
    continents = list(CONTINENT_GLOBS.keys())
    per_cont = -(-PER_CLASS // len(continents))  # ceil
    positives_by_continent: dict[str, list[dict[str, Any]]] = {}
    for cont in continents:
        recs = build_positive_records(
            cont, geoms_by_continent[cont], trees[cont], per_cont, rng
        )
        positives_by_continent[cont] = recs
        print(f"  {cont}: {len(recs)} positive tiles")

    positives = [r for recs in positives_by_continent.values() for r in recs]
    rng.shuffle(positives)
    positives = positives[:PER_CLASS]

    negatives = build_negative_records(
        positives_by_continent, trees, geoms_by_continent, N_NEGATIVES, rng
    )
    print(f"selected {len(positives)} positives, {len(negatives)} negatives")

    selected = positives + negatives
    for i, r in enumerate(selected):
        r["sample_id"] = f"{i:06d}"

    io.check_disk()
    results: Counter = Counter()
    with multiprocessing.Pool(args.workers) as p:
        for res in tqdm.tqdm(
            star_imap_unordered(p, _write_tile, [dict(rec=r) for r in selected]),
            total=len(selected),
        ):
            results[res] += 1
    print("write results:", dict(results))
    io.check_disk()

    # Per-continent positive counts for the summary.
    per_cont_counts: dict[str, int] = defaultdict(int)
    for r in positives:
        per_cont_counts[r["continent"]] += 1

    io.write_dataset_metadata(
        SLUG,
        {
            "dataset": SLUG,
            "name": NAME,
            "task_type": "classification",
            "source": "University of Leeds (PEATMAP)",
            "license": "CC-BY-4.0 (open, Leeds)",
            "provenance": {
                "url": "https://archive.researchdata.leeds.ac.uk/251/",
                "doi": "10.5518/252",
                "have_locally": False,
                "annotation_method": "derived-product (meta-analysis of peat maps)",
                "citation": "Xu, Morris, Liu, Holden (2018) PEATMAP, Catena; DOI 10.1016/j.catena.2017.09.010",
            },
            "sensors_relevant": ["sentinel2", "sentinel1", "landsat"],
            "classes": CLASSES,
            "nodata_value": io.CLASS_NODATA,
            "num_samples": len(selected),
            "class_counts": {
                "peatland": len(positives),
                "background_negative_tiles": len(negatives),
                "positives_by_continent": dict(per_cont_counts),
            },
            "notes": (
                "Binary peatland-vs-background segmentation from PEATMAP global peatland "
                "extent polygons. Each label is a 64x64 UTM tile at 10 m; peat polygons "
                "intersecting a tile are rasterized as class 1 (all_touched), else class 0. "
                "Positive tiles are area-weighted interior points of peatland polygons with "
                "an even per-continent quota (bounded, regionally-diverse sample of a large "
                "global derived product, not global coverage); equal-count background-only "
                "negatives are drawn 30-120 km from peat and verified peat-free. Static "
                f"baseline -> 1-year window anchored on {REPRESENTATIVE_YEAR} (Sentinel era). "
                "Source is a derived product (meta-analysis of national/regional peat maps)."
            ),
        },
    )
    print("done")


if __name__ == "__main__":
    multiprocessing.set_start_method("forkserver", force=True)
    main()
