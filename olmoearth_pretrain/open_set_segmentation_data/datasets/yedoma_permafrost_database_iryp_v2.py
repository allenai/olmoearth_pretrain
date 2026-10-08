"""Process the Yedoma Permafrost Database (IRYP v2) into open-set-segmentation labels.

Source: Strauss, J. et al. (2022): Database of Ice-Rich Yedoma Permafrost Version 2
(IRYP v2). PANGAEA, https://doi.org/10.1594/PANGAEA.940078 (CC-BY-4.0); supplement to the
Circum-Arctic Map of the Yedoma Permafrost Domain (Strauss et al. 2021, Frontiers in Earth
Science, https://doi.org/10.3389/feart.2021.758360).

Yedoma is a late-Pleistocene ice-rich (syngenetic) permafrost deposit. Its EXTENT is a
mapped geomorphic region compiled/harmonized from geological & stratigraphic maps -- like a
lithology/geology map, coarse but a valid per-pixel region class. We treat it as a
polygon -> dense region classification of Yedoma extent presence. It is NOT a dated change
event -> change_time=null, static 1-year Sentinel-era window.

We use the ``IRYP_v2_yedoma_confidence`` shapefile: 13,833 Yedoma-deposit polygons in
EPSG:3571 (WGS84 North Pole LAEA Bering Sea, metres) carrying a ``confidence`` attribute
with three levels of Yedoma-presence certainty (confirmed / likely / uncertain), exactly
the class scheme named in the manifest. The confidence levels reflect mapping certainty of
the SAME phenomenon (Yedoma presence), not visually distinct land classes; a downstream
consumer training binary Yedoma presence should merge classes 1-3. We keep them separate
per spec 5 (do not drop classes; downstream can merge/filter).

Class scheme (uint8):
  0 = background          (non-Yedoma terrain inside a tile)
  1 = yedoma_confirmed
  2 = yedoma_likely
  3 = yedoma_uncertain
  255 = nodata/ignore     (declared for consistency; unused)

Each label is a 64x64 (640 m) tile at 10 m/pixel in the local UTM zone. Positive tiles are
area-weighted interior points of Yedoma polygons (per confidence class, so rare classes
reach their quota); every Yedoma polygon intersecting a tile is rasterized (all_touched) at
its confidence class value, the rest is background. Background-only negatives are drawn away
from any Yedoma so the background class has genuine negatives (the polygon boundaries are a
real land/region delineation, cf. peatmap). Tiles-per-class balanced to <=1000/class,
<=25,000 total.

This is a large regional derived-product map; per spec 5 we draw a bounded, area-weighted
sample across the circum-Arctic distribution rather than exhaustive coverage.

Run (idempotent):
  python3 -m olmoearth_pretrain.open_set_segmentation_data.datasets.yedoma_permafrost_database_iryp_v2
"""

import argparse
import multiprocessing
import random
from collections import Counter
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

from olmoearth_pretrain.open_set_segmentation_data import io, sampling
from olmoearth_pretrain.open_set_segmentation_data.rasterize import (
    geom_to_pixels,
    rasterize_shapes,
)

SLUG = "yedoma_permafrost_database_iryp_v2"
NAME = "Yedoma Permafrost Database (IRYP v2)"

# Source shapefile (extracted under raw_dir/extract) is in EPSG:3571, metres.
SHP = "extract/IRYP_v2_yedoma_confidence.shp"
SRC_CRS = CRS.from_epsg(3571)
SRC_PROJ = Projection(SRC_CRS, 1, 1)

# Class scheme.
CID_BACKGROUND = 0
CONF_TO_CID = {"confirmed": 1, "likely": 2, "uncertain": 3}
CLASSES = [
    {
        "id": CID_BACKGROUND,
        "name": "background",
        "description": "Non-Yedoma terrain: any 10 m pixel outside a mapped Yedoma-deposit "
        "polygon (other permafrost/land/water).",
    },
    {
        "id": 1,
        "name": "yedoma_confirmed",
        "description": "Late-Pleistocene ice-rich Yedoma permafrost deposit, presence "
        "CONFIRMED from lithological/stratigraphic source-map information and field/expert "
        "knowledge (IRYP v2 confidence class 'confirmed', conf_id 11-14).",
    },
    {
        "id": 2,
        "name": "yedoma_likely",
        "description": "Yedoma deposit, presence LIKELY per source maps (IRYP v2 confidence "
        "class 'likely', conf_id 21-22).",
    },
    {
        "id": 3,
        "name": "yedoma_uncertain",
        "description": "Yedoma deposit, presence UNCERTAIN per source maps (IRYP v2 "
        "confidence class 'uncertain', conf_id 31).",
    },
]

PER_CLASS = 1000  # positive tiles per confidence class
N_NEGATIVES = 1000  # background-only negative tiles
TILE = 64  # 64x64 @ 10 m = 640 m tiles
REPRESENTATIVE_YEAR = (
    2020  # static geomorphic label -> representative Sentinel-era year
)
QUERY_PAD_M = (
    700.0  # padding (m, EPSG:3571) when clipping candidate geoms to a tile box
)
DEDUP_GRID_M = 1000.0  # spread positive centers on a ~1 km grid
SEED = 42


# --------------------------------------------------------------------------------------
# Loading source geometries.
# --------------------------------------------------------------------------------------
def load_polys() -> tuple[np.ndarray, np.ndarray]:
    """Load Yedoma polygons and their class ids. Returns (geoms, class_ids)."""
    raw = io.raw_dir(SLUG)
    gdf = read_dataframe(str(raw / SHP), columns=["confidence"])
    geoms = shapely.force_2d(np.asarray(gdf.geometry.values, dtype=object))
    cids = np.array([CONF_TO_CID[c] for c in gdf["confidence"].values], dtype=np.int64)
    keep = np.array([g is not None for g in geoms]) & ~shapely.is_empty(geoms)
    return geoms[keep], cids[keep]


def _polygon_parts_with_class(
    geoms: np.ndarray, cids: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Explode (multi)polygons into single Polygons, carrying their class id."""
    parts_list: list[Any] = []
    cls_list: list[int] = []
    for g, c in zip(geoms, cids):
        for part in shapely.get_parts(g):
            if shapely.get_type_id(part) == 3 and not part.is_empty:  # 3 == Polygon
                parts_list.append(part)
                cls_list.append(int(c))
    return np.array(parts_list, dtype=object), np.array(cls_list, dtype=np.int64)


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


def _to_wgs84(x: float, y: float) -> tuple[float, float]:
    st = STGeometry(SRC_PROJ, shapely.Point(x, y), None).to_projection(WGS84_PROJECTION)
    return float(st.shp.x), float(st.shp.y)


def _clip_candidates(
    tree: STRtree, geoms: np.ndarray, cids: np.ndarray, x: float, y: float
) -> list[tuple[bytes, int]]:
    """Return (wkb, class_id) of Yedoma geometry clipped to a padded box around (x, y)."""
    x0, y0, x1, y1 = x - QUERY_PAD_M, y - QUERY_PAD_M, x + QUERY_PAD_M, y + QUERY_PAD_M
    box = shapely.box(x0, y0, x1, y1)
    out: list[tuple[bytes, int]] = []
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
        out.append((shapely.to_wkb(clipped), int(cids[idx])))
    return out


def _has_yedoma(tree: STRtree, geoms: np.ndarray, x: float, y: float) -> bool:
    """True if any Yedoma polygon intersects a padded box around (x, y)."""
    box = shapely.box(
        x - QUERY_PAD_M, y - QUERY_PAD_M, x + QUERY_PAD_M, y + QUERY_PAD_M
    )
    return any(geoms[idx].intersects(box) for idx in tree.query(box))


# --------------------------------------------------------------------------------------
# Candidate records.
# --------------------------------------------------------------------------------------
def build_positive_candidates(
    polys: np.ndarray,
    poly_cids: np.ndarray,
    tree: STRtree,
    geoms: np.ndarray,
    cids: np.ndarray,
    n_per_class: int,
    rng: random.Random,
) -> list[dict[str, Any]]:
    """Area-weighted interior-point positives, per confidence class."""
    recs: list[dict[str, Any]] = []
    seen: set[tuple[int, int]] = set()
    for cid in sorted(set(poly_cids.tolist())):
        mask = poly_cids == cid
        cpolys = polys[mask]
        areas = shapely.area(cpolys)
        total = areas.sum()
        if total <= 0 or len(cpolys) == 0:
            continue
        cum = np.cumsum(areas) / total
        got = 0
        attempts = 0
        while got < n_per_class and attempts < n_per_class * 40:
            attempts += 1
            pi = min(int(np.searchsorted(cum, rng.random())), len(cpolys) - 1)
            x, y = _sample_interior_point(cpolys[pi], rng)
            key = (int(x // DEDUP_GRID_M), int(y // DEDUP_GRID_M))
            if key in seen:
                continue
            seen.add(key)
            lon, lat = _to_wgs84(x, y)
            recs.append(
                {
                    "kind": "positive",
                    "seed_cid": cid,
                    "lon": lon,
                    "lat": lat,
                    "geom": _clip_candidates(tree, geoms, cids, x, y),
                    "source_id": f"conf{cid}/pos/{got}",
                }
            )
            got += 1
    return recs


def build_negative_candidates(
    positives: list[dict[str, Any]],
    tree: STRtree,
    geoms: np.ndarray,
    n: int,
    rng: random.Random,
) -> list[dict[str, Any]]:
    """Background-only negatives: offset from a Yedoma point, rejected if Yedoma is nearby."""
    recs: list[dict[str, Any]] = []
    if not positives:
        return recs
    attempts = 0
    while len(recs) < n and attempts < n * 60:
        attempts += 1
        base = positives[rng.randrange(len(positives))]
        p = (
            STGeometry(WGS84_PROJECTION, shapely.Point(base["lon"], base["lat"]), None)
            .to_projection(SRC_PROJ)
            .shp
        )
        ang = rng.uniform(0, 2 * np.pi)
        dist = rng.uniform(20000, 150000)  # 20-150 km away
        x = p.x + dist * np.cos(ang)
        y = p.y + dist * np.sin(ang)
        if _has_yedoma(tree, geoms, x, y):
            continue
        lon, lat = _to_wgs84(x, y)
        if not (50 <= lat <= 80):
            continue
        recs.append(
            {
                "kind": "negative",
                "seed_cid": CID_BACKGROUND,
                "lon": lon,
                "lat": lat,
                "geom": [],
                "source_id": f"neg/{len(recs)}",
            }
        )
    return recs


# --------------------------------------------------------------------------------------
# Classes-present computation (worker) and tile writer (worker).
# --------------------------------------------------------------------------------------
def _tile_shapes(
    rec: dict[str, Any],
) -> tuple[Projection, tuple, list[tuple[Any, int]]]:
    """Reproject clipped Yedoma geometry into the tile's UTM pixel grid."""
    proj = io.utm_projection_for_lonlat(rec["lon"], rec["lat"])
    _, col, row = io.lonlat_to_utm_pixel(rec["lon"], rec["lat"], proj)
    bounds = io.centered_bounds(col, row, TILE, TILE)
    shapes: list[tuple[Any, int]] = []
    for wkb, cid in rec["geom"]:
        geom = shapely.from_wkb(wkb)
        if geom.is_empty:
            continue
        pix = geom_to_pixels(geom, SRC_PROJ, proj)
        if pix.is_empty:
            continue
        for part in shapely.get_parts(pix):
            if part.geom_type == "Polygon" and not part.is_empty:
                shapes.append((part, cid))
    return proj, bounds, shapes


def compute_classes_present(rec: dict[str, Any]) -> dict[str, Any]:
    """Rasterize into the tile grid just to record which class ids actually appear."""
    _, bounds, shapes = _tile_shapes(rec)
    if shapes:
        arr = rasterize_shapes(
            shapes, bounds, fill=CID_BACKGROUND, dtype="uint8", all_touched=True
        )
        present = sorted(set(np.unique(arr).tolist()) - {io.CLASS_NODATA})
    else:
        present = [CID_BACKGROUND]
    rec = dict(rec)
    rec["classes_present"] = present
    return rec


def _write_tile(rec: dict[str, Any]) -> str:
    sample_id = rec["sample_id"]
    tif = io.locations_dir(SLUG) / f"{sample_id}.tif"
    if tif.exists():
        return "skip"
    proj, bounds, shapes = _tile_shapes(rec)
    if shapes:
        arr = rasterize_shapes(
            shapes, bounds, fill=CID_BACKGROUND, dtype="uint8", all_touched=True
        )
    else:
        arr = np.full((1, TILE, TILE), CID_BACKGROUND, dtype=np.uint8)
    io.write_label_geotiff(SLUG, sample_id, arr, proj, bounds, nodata=io.CLASS_NODATA)
    present = sorted(set(np.unique(arr).tolist()) - {io.CLASS_NODATA})
    io.write_sample_json(
        SLUG,
        sample_id,
        proj,
        bounds,
        io.year_range(REPRESENTATIVE_YEAR),
        source_id=rec["source_id"],
        classes_present=present,
    )
    return rec["kind"]


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
            "Yedoma Permafrost Database (IRYP v2).\n"
            "Strauss, J. et al. (2022), PANGAEA, https://doi.org/10.1594/PANGAEA.940078 "
            "(CC-BY-4.0).\n"
            "Files: https://download.pangaea.de/dataset/940078/files/\n"
            "  IRYP_v2_yedoma_confidence_Shapefile.zip  <- USED (Yedoma deposit polygons "
            "with confidence classes, EPSG:3571)\n"
            "  IRYP_v2_yedoma_domain_Shapefile.zip      (20 km-buffered domain envelope; "
            "downloaded, not used as label -- see summary)\n"
            "Supplement to Strauss et al. 2021, Frontiers in Earth Science, "
            "https://doi.org/10.3389/feart.2021.758360\n"
        )

    rng = random.Random(SEED)

    geoms, cids = load_polys()
    print(f"loaded {len(geoms)} Yedoma polygons")
    tree = STRtree(geoms)
    polys, poly_cids = _polygon_parts_with_class(geoms, cids)
    print(f"exploded to {len(polys)} single polygons")
    io.check_disk()

    positives = build_positive_candidates(
        polys, poly_cids, tree, geoms, cids, PER_CLASS * 2, rng
    )
    print(f"built {len(positives)} positive candidates")
    negatives = build_negative_candidates(positives, tree, geoms, N_NEGATIVES, rng)
    print(f"built {len(negatives)} negative candidates")

    # Compute classes_present for positives (parallel; involves rasterization).
    with multiprocessing.Pool(args.workers) as p:
        positives = list(
            tqdm.tqdm(
                star_imap_unordered(
                    p, compute_classes_present, [dict(rec=r) for r in positives]
                ),
                total=len(positives),
                desc="classes_present",
            )
        )
    for r in negatives:
        r["classes_present"] = [CID_BACKGROUND]

    # Tiles-per-class balanced over the Yedoma classes (1/2/3): prioritizes rare classes.
    selected_pos = sampling.balance_tiles_by_class(
        positives, classes_key="classes_present", per_class=PER_CLASS, seed=SEED
    )
    # Keep negatives up to N_NEGATIVES, within the overall cap.
    room = sampling.MAX_SAMPLES_PER_DATASET - len(selected_pos)
    selected = selected_pos + negatives[: max(0, min(N_NEGATIVES, room))]
    for i, r in enumerate(selected):
        r["sample_id"] = f"{i:06d}"
    print(
        f"selected {len(selected_pos)} positives + "
        f"{len(selected) - len(selected_pos)} negatives = {len(selected)}"
    )

    io.check_disk()
    results: Counter = Counter()
    with multiprocessing.Pool(args.workers) as p:
        for res in tqdm.tqdm(
            star_imap_unordered(p, _write_tile, [dict(rec=r) for r in selected]),
            total=len(selected),
            desc="write",
        ):
            results[res] += 1
    print("write results:", dict(results))
    io.check_disk()

    # Class pixel/tile presence counts for the summary.
    class_tile_counts: Counter = Counter()
    for r in selected:
        for c in r.get("classes_present", []):
            class_tile_counts[c] += 1

    io.write_dataset_metadata(
        SLUG,
        {
            "dataset": SLUG,
            "name": NAME,
            "task_type": "classification",
            "source": "PANGAEA (IRYP v2, Strauss et al. 2022)",
            "license": "CC-BY-4.0",
            "provenance": {
                "url": "https://doi.org/10.1594/PANGAEA.940078",
                "have_locally": False,
                "annotation_method": "manual/photointerpretation; harmonized digitization "
                "of geological & stratigraphic source maps + field/expert knowledge",
                "layer_used": "IRYP_v2_yedoma_confidence (13,833 polygons, EPSG:3571)",
                "citation": "Strauss et al. (2021) Circum-Arctic Map of the Yedoma "
                "Permafrost Domain, Frontiers in Earth Science 9, "
                "https://doi.org/10.3389/feart.2021.758360",
            },
            "sensors_relevant": ["sentinel2", "sentinel1", "landsat"],
            "classes": CLASSES,
            "nodata_value": io.CLASS_NODATA,
            "num_samples": len(selected),
            "class_tile_counts": {
                str(k): v for k, v in sorted(class_tile_counts.items())
            },
            "notes": (
                "Yedoma late-Pleistocene ice-rich permafrost DEPOSIT extent, treated as a "
                "polygon -> dense region classification (static geomorphic region, like a "
                "lithology map). 64x64 UTM tiles at 10 m; Yedoma polygons intersecting a tile "
                "are rasterized (all_touched) at their confidence class value, rest is "
                "background(0). Confidence classes 1/2/3 (confirmed/likely/uncertain) are "
                "mapping-certainty tiers of the SAME phenomenon (Yedoma presence), NOT "
                "visually distinct land classes -- a consumer training binary Yedoma presence "
                "should merge classes 1-3. Positive tiles are area-weighted interior points "
                "per confidence class (rare classes reach quota); equal-count background-only "
                "negatives drawn 20-150 km from Yedoma and verified Yedoma-free. Static "
                f"baseline -> 1-year window anchored on {REPRESENTATIVE_YEAR}; change_time=null. "
                "CAVEAT: Yedoma is a subsurface deposit compiled/harmonized from geological "
                "maps at heterogeneous scales -- boundaries are COARSE relative to 10 m, so "
                "expect boundary label noise; surface expression (thermokarst uplands, "
                "ice-wedge polygons) is only partly resolvable at 10-30 m. Bounded regional "
                "sample of a large circum-Arctic derived product, not exhaustive coverage."
            ),
        },
    )
    print("done")


if __name__ == "__main__":
    multiprocessing.set_start_method("forkserver", force=True)
    main()
