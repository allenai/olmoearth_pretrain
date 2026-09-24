"""Process Intact Forest Landscapes (IFL) into open-set-segmentation label patches.

Source: intactforests.org / GLAD (The IFL Mapping Team; Potapov, Turubanova, et al.;
World Resources Institute / University of Maryland / Greenpeace). IFLs are "forest
wildlands": roadless forest landscapes (>= 500 km2, min width 10 km) within the forest
zone that show no signs of significant human transformation (no conversion, roads, or
industrial resource extraction), mapped globally by manual photointerpretation of
Landsat/high-resolution imagery. Delivered as per-epoch GeoPackages (2000, 2013, 2016,
2020, 2025) of large MultiPolygons in EPSG:4326, with fields IFL_ID and Area{year}
(hectares). https://intactforests.org/data.ifl.html, license CC-BY-4.0.

We use the **2020** epoch (a representative Sentinel-era layer; 2016/2020/2025 are all
post-2015). IFL change between epochs is a multi-year reduction, not a dated event, so per
the task spec we treat presence as a **static** label (change_time=null), NOT a dated
change label.

Binary presence segmentation. Class scheme (uint8):
  0 = background                 (non-IFL: land/water outside an IFL polygon)
  1 = intact_forest_landscape    (inside an IFL 2020 polygon)
  255 = nodata/ignore            (declared for consistency; not used here)

Each label is a 64x64 (640 m) tile at 10 m/pixel in the local UTM zone. IFL polygons are
enormous (>= 500 km2), so most positive tiles fall entirely inside one polygon (all class
1); tiles near a boundary show a mix. IFL is a large global derived product; per spec we
draw a **bounded, regionally-diverse** sample rather than global coverage: an even
per-region quota (the 6 IFL_ID region codes: SAM, AFR, NAM, AUS, SEA, NEA) of
area-weighted interior-point positive tiles, plus an equal number of background-only
negatives drawn away from any IFL and verified IFL-free.

Run (idempotent):
  python3 -m olmoearth_pretrain.open_set_segmentation_data.datasets.intact_forest_landscapes_ifl
"""

import argparse
import multiprocessing
import random
import re
from collections import Counter, defaultdict
from typing import Any

import numpy as np
import shapely
import tqdm
from pyogrio import read_dataframe
from rasterio.crs import CRS
from rslearn.utils.geometry import Projection
from rslearn.utils.mp import star_imap_unordered
from shapely import STRtree

from olmoearth_pretrain.open_set_segmentation_data import io
from olmoearth_pretrain.open_set_segmentation_data.rasterize import (
    geom_to_pixels,
    rasterize_shapes,
)

SLUG = "intact_forest_landscapes_ifl"
NAME = "Intact Forest Landscapes (IFL)"

EPOCH = 2020
GPKG_NAME = f"IFL_{EPOCH}.gpkg"
AREA_FIELD = f"Area{EPOCH}"
GPKG_URL = f"https://intactforests.org/shp/{GPKG_NAME}"
PDF_URL = "https://intactforests.org/shp/IFL_2000-2025.pdf"

# Source geometries are lon/lat WGS84. Projection(res=1,1) keeps native degree units so
# geom_to_pixels can reproject to a local UTM tile grid (mirrors the peatmap pattern).
SRC_CRS = CRS.from_epsg(4326)
SRC_PROJ = Projection(SRC_CRS, 1, 1)

# Class scheme.
CID_BACKGROUND = 0
CID_IFL = 1
CLASSES = [
    {
        "id": CID_BACKGROUND,
        "name": "background",
        "description": "Non-IFL: any 10 m pixel outside an Intact Forest Landscape "
        "polygon (converted, fragmented, roaded, or otherwise human-transformed land, or "
        "water, or forest zone that does not meet the IFL criteria).",
    },
    {
        "id": CID_IFL,
        "name": "intact_forest_landscape",
        "description": "Intact Forest Landscape (IFL 2020): a territory within the current "
        "forest extent, >= 500 km2 and >= 10 km wide, containing forest and non-forest "
        "ecosystems minimally influenced by human economic activity -- no conversion, "
        "roads, settlements, or industrial resource extraction (Potapov et al. 2008, 2017; "
        "The IFL Mapping Team 2025). Mapped globally by manual photointerpretation.",
    },
]

# Sampling parameters (binary presence -> per-class balanced, peatmap precedent).
PER_CLASS = 1000  # IFL-positive tiles
N_NEGATIVES = 1000  # background-only negative tiles
TILE = 64  # 64x64 @ 10 m = 640 m tiles
REPRESENTATIVE_YEAR = (
    EPOCH  # static presence -> 1-year window anchored on the 2020 epoch
)
QUERY_PAD_DEG = (
    0.02  # ~2 km padding (deg) when clipping candidate IFL geoms to a tile box
)
SEED = 42

# IFL_ID region prefixes -> readable region names (used for the even per-region quota).
REGION_NAMES = {
    "SAM": "South_America",
    "AFR": "Africa",
    "NAM": "North_America",
    "AUS": "Australia_Oceania",
    "SEA": "Southeast_Asia",
    "NEA": "North_Eurasia",
}


def _region_of(ifl_id: str) -> str:
    m = re.match(r"([A-Za-z]+)_", ifl_id)
    return m.group(1) if m else "OTHER"


# --------------------------------------------------------------------------------------
# Loading source geometries.
# --------------------------------------------------------------------------------------
def load_features() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (geoms, region_codes, areas_ha) arrays for all IFL 2020 polygons."""
    gpkg = io.raw_dir(SLUG) / GPKG_NAME
    gdf = read_dataframe(str(gpkg))
    geoms = shapely.force_2d(np.asarray(gdf.geometry.values, dtype=object))
    regions = np.array([_region_of(i) for i in gdf["IFL_ID"].values], dtype=object)
    areas = gdf[AREA_FIELD].values.astype(float)  # hectares
    keep = np.array([g is not None and not g.is_empty for g in geoms])
    return geoms[keep], regions[keep], areas[keep]


def _sample_interior_point(poly: Any, rng: random.Random) -> tuple[float, float]:
    """Sample a random interior lon/lat of a single polygon (fallback: representative pt)."""
    minx, miny, maxx, maxy = poly.bounds
    for _ in range(40):
        x = rng.uniform(minx, maxx)
        y = rng.uniform(miny, maxy)
        if poly.contains(shapely.Point(x, y)):
            return x, y
    p = poly.representative_point()
    return p.x, p.y


# --------------------------------------------------------------------------------------
# Building sample records (main process; attaches clipped IFL geometry WKB).
# --------------------------------------------------------------------------------------
def _clip_candidates(
    tree: STRtree, geoms: np.ndarray, lon: float, lat: float
) -> list[bytes]:
    """WKB of IFL geometry within a padded lon/lat box around the point, clipped to it."""
    x0, y0, x1, y1 = (
        lon - QUERY_PAD_DEG,
        lat - QUERY_PAD_DEG,
        lon + QUERY_PAD_DEG,
        lat + QUERY_PAD_DEG,
    )
    box = shapely.box(x0, y0, x1, y1)
    out: list[bytes] = []
    for idx in tree.query(box):
        g = geoms[idx]
        try:
            clipped = shapely.clip_by_rect(g, x0, y0, x1, y1)
        except shapely.errors.GEOSException:
            clipped = shapely.clip_by_rect(shapely.make_valid(g), x0, y0, x1, y1)
        if clipped.is_empty:
            continue
        if not clipped.is_valid:
            clipped = shapely.make_valid(clipped)
            if clipped.is_empty:
                continue
        out.append(shapely.to_wkb(clipped))
    return out


def _has_ifl(tree: STRtree, geoms: np.ndarray, lon: float, lat: float) -> bool:
    """True if any IFL polygon intersects a padded box around (lon, lat)."""
    box = shapely.box(
        lon - QUERY_PAD_DEG,
        lat - QUERY_PAD_DEG,
        lon + QUERY_PAD_DEG,
        lat + QUERY_PAD_DEG,
    )
    for idx in tree.query(box):
        if geoms[idx].intersects(box):
            return True
    return False


def build_positive_records(
    region: str,
    idxs: np.ndarray,
    geoms: np.ndarray,
    areas_ha: np.ndarray,
    tree: STRtree,
    n: int,
    rng: random.Random,
) -> list[dict[str, Any]]:
    """Area-weighted interior-point positives within a region.

    Polygon selection is weighted by the reported IFL area (hectares); within the chosen
    multipolygon a constituent part is picked weighted by its geometric area (parts share a
    latitude, so the deg^2 distortion cancels), then an interior point is sampled.
    """
    if len(idxs) == 0:
        return []
    w = areas_ha[idxs]
    w = w / w.sum()
    cum = np.cumsum(w)
    recs: list[dict[str, Any]] = []
    attempts = 0
    seen: set[tuple[int, int]] = set()
    while len(recs) < n and attempts < n * 20:
        attempts += 1
        fi = idxs[min(int(np.searchsorted(cum, rng.random())), len(idxs) - 1)]
        parts = shapely.get_parts(geoms[fi])
        parts = parts[shapely.get_type_id(parts) == 3]  # Polygon
        if len(parts) == 0:
            continue
        pareas = shapely.area(parts)
        if pareas.sum() <= 0:
            continue
        pcum = np.cumsum(pareas / pareas.sum())
        poly = parts[min(int(np.searchsorted(pcum, rng.random())), len(parts) - 1)]
        lon, lat = _sample_interior_point(poly, rng)
        # Dedup near-identical centers on a ~0.05 deg (~5 km) grid.
        key = (int(lon / 0.05), int(lat / 0.05))
        if key in seen:
            continue
        seen.add(key)
        recs.append(
            {
                "region": region,
                "lon": lon,
                "lat": lat,
                "ifl_wkb": _clip_candidates(tree, geoms, lon, lat),
                "source_id": f"{region}/pos/{len(recs)}",
            }
        )
    return recs


def build_negative_records(
    positives_by_region: dict[str, list[dict[str, Any]]],
    global_tree: STRtree,
    geoms: np.ndarray,
    n: int,
    rng: random.Random,
) -> list[dict[str, Any]]:
    """Background-only negatives: offset from a positive point, rejected if IFL is nearby."""
    recs: list[dict[str, Any]] = []
    regions = [r for r, ps in positives_by_region.items() if ps]
    attempts = 0
    while len(recs) < n and attempts < n * 80:
        attempts += 1
        reg = regions[rng.randrange(len(regions))]
        base = positives_by_region[reg][rng.randrange(len(positives_by_region[reg]))]
        # Offset in metres, converted to degrees at the base latitude.
        dist_km = rng.uniform(30, 150)
        ang = rng.uniform(0, 2 * np.pi)
        dlat = (dist_km * np.sin(ang)) / 111.0
        dlon = (dist_km * np.cos(ang)) / (
            111.0 * max(0.05, np.cos(np.radians(base["lat"])))
        )
        lon = base["lon"] + dlon
        lat = base["lat"] + dlat
        if not (-180 <= lon <= 180 and -60 <= lat <= 78):
            continue
        if _has_ifl(global_tree, geoms, lon, lat):
            continue
        recs.append(
            {
                "region": reg,
                "lon": lon,
                "lat": lat,
                "ifl_wkb": [],
                "source_id": f"{reg}/neg/{len(recs)}",
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
    for wkb in rec["ifl_wkb"]:
        geom = shapely.from_wkb(wkb)
        if geom.is_empty:
            continue
        pix = geom_to_pixels(geom, SRC_PROJ, proj)
        if pix.is_empty:
            continue
        for part in shapely.get_parts(pix):
            if part.geom_type == "Polygon" and not part.is_empty:
                shapes.append((part, CID_IFL))
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
    return "ifl" if shapes and CID_IFL in np.unique(arr) else "background"


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
    from olmoearth_pretrain.open_set_segmentation_data import download

    download.download_http(GPKG_URL, raw / GPKG_NAME, timeout=900)
    download.download_http(PDF_URL, raw / "IFL_2000-2025.pdf", timeout=300)
    with (raw / "SOURCE.txt").open("w") as f:
        f.write(
            "Intact Forest Landscapes (IFL) 2020, intactforests.org / GLAD.\n"
            f"{GPKG_URL}\n{PDF_URL}\n"
            "The IFL Mapping Team (Potapov, Turubanova, et al.); WRI / U. Maryland / "
            "Greenpeace. CC-BY-4.0. Per-epoch GeoPackages of large MultiPolygons "
            "(EPSG:4326); fields IFL_ID, Area{year} (hectares). 2020 epoch used.\n"
        )
    io.check_disk()

    geoms, regions, areas_ha = load_features()
    print(f"loaded {len(geoms)} IFL 2020 polygons")
    global_tree = STRtree(geoms)

    region_codes = sorted(REGION_NAMES.keys())
    region_idxs = {rc: np.where(regions == rc)[0] for rc in region_codes}
    for rc in region_codes:
        print(f"  {REGION_NAMES[rc]} ({rc}): {len(region_idxs[rc])} polygons")

    rng = random.Random(SEED)

    # Even per-region positive quota for regional diversity.
    per_region = -(-PER_CLASS // len(region_codes))  # ceil
    positives_by_region: dict[str, list[dict[str, Any]]] = {}
    for rc in region_codes:
        idxs = region_idxs[rc]
        recs = build_positive_records(
            REGION_NAMES[rc], idxs, geoms, areas_ha, global_tree, per_region, rng
        )
        positives_by_region[REGION_NAMES[rc]] = recs
        print(f"  {REGION_NAMES[rc]}: {len(recs)} positive tiles")

    positives = [r for recs in positives_by_region.values() for r in recs]
    rng.shuffle(positives)
    positives = positives[:PER_CLASS]

    negatives = build_negative_records(
        positives_by_region, global_tree, geoms, N_NEGATIVES, rng
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

    per_region_counts: dict[str, int] = defaultdict(int)
    for r in positives:
        per_region_counts[r["region"]] += 1

    io.write_dataset_metadata(
        SLUG,
        {
            "dataset": SLUG,
            "name": NAME,
            "task_type": "classification",
            "source": "intactforests.org / GLAD (The IFL Mapping Team, WRI/UMD/Greenpeace)",
            "license": "CC-BY-4.0",
            "provenance": {
                "url": "https://intactforests.org/data.ifl.html",
                "have_locally": False,
                "annotation_method": "manual photointerpretation",
                "epoch": EPOCH,
                "citation": "Potapov et al. 2008, 2017; The IFL Mapping Team 2025.",
            },
            "sensors_relevant": ["sentinel2", "sentinel1", "landsat"],
            "classes": CLASSES,
            "nodata_value": io.CLASS_NODATA,
            "num_samples": len(selected),
            "class_counts": {
                "intact_forest_landscape": len(positives),
                "background_negative_tiles": len(negatives),
                "positives_by_region": dict(per_region_counts),
            },
            "notes": (
                "Binary IFL-presence segmentation from Intact Forest Landscapes 2020 "
                "polygons. Each label is a 64x64 UTM tile at 10 m; IFL polygons "
                "intersecting a tile are rasterized as class 1 (all_touched), else class 0. "
                "IFL polygons are very large (>= 500 km2) so most positive tiles are wholly "
                "class 1; boundary tiles are mixed. Positive tiles are area-weighted "
                "interior points with an even per-region quota over the 6 IFL_ID regions "
                "(SAM/AFR/NAM/AUS/SEA/NEA) -- a bounded, regionally-diverse sample of a "
                "large global derived product, not global coverage. Equal-count "
                "background-only negatives are drawn 30-150 km from any positive and "
                "verified IFL-free. IFL change between epochs is a multi-year reduction, "
                "not a dated event, so presence is treated as STATIC (change_time=null) "
                f"with a 1-year window anchored on {REPRESENTATIVE_YEAR} (Sentinel era). "
                "Source is a derived product from manual photointerpretation."
            ),
        },
    )
    print("done")


if __name__ == "__main__":
    multiprocessing.set_start_method("forkserver", force=True)
    main()
