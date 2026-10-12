"""Process Amazon Mining Watch into open-set-segmentation label patches.

Source: Source Cooperative, Earth Genome ("Amazon Mining Watch"),
https://source.coop/earthgenome/amazon-mining-watch (public, unsigned S3 bucket
``us-west-2.opendata.source.coop``, prefix ``earthgenome/amazon-mining-watch``).

The product is annual, ML-detected polygons of artisanal/illegal gold-mine scars in
Sentinel-2 across the Amazon basin. Each annual GeoJSON (2018..2024) is *cumulative* from
2018 (i.e. the 2020 file contains every scar detected 2018-2020). CRS is CRS84 (lon/lat).
Detections come from overlapping 480 m x 480 m patches merged into polygons; individual
scar polygons are large areal features (median footprint ~830 m, many multi-km), so this
is a presence/segmentation label, not positive-only point detection.

Encoding: a single positive class. We lay a 640 m (64 px @ 10 m) grid over each UTM zone
and emit, per grid cell that a mine-scar polygon covers, a 64x64 uint8 tile with
``mine_scar`` (id 1) where polygons fall and ``background`` (id 0) elsewhere. Because
scars span many cells, tiles naturally mix full-interior and edge coverage. Each grid
cell is assigned to the *earliest* year it becomes positive (files are cumulative), and
its time range is that ~1-year window. We also emit background-only negative tiles a
short distance away from positives (in the same UTM zone, verified free of any polygon)
so the class has genuine negatives. Bounded to <=1000 tiles/class.
"""

import argparse
import multiprocessing
import random
from collections import Counter, defaultdict
from typing import Any

import numpy as np
import shapely
import tqdm
from rasterio.crs import CRS
from rslearn.const import WGS84_PROJECTION
from rslearn.utils.geometry import Projection, STGeometry
from rslearn.utils.get_utm_ups_crs import get_utm_ups_projection
from rslearn.utils.mp import star_imap_unordered
from shapely.geometry import shape

from olmoearth_pretrain.open_set_segmentation_data import io
from olmoearth_pretrain.open_set_segmentation_data.rasterize import (
    geom_to_pixels,
    rasterize_shapes,
)

SLUG = "amazon_mining_watch"
NAME = "Amazon Mining Watch"
URL = "https://source.coop/earthgenome/amazon-mining-watch"
BUCKET = "us-west-2.opendata.source.coop"
PREFIX = "earthgenome/amazon-mining-watch"

# Per-year cumulative GeoJSON keys (2018..2024). 2025 is quarterly / out of the
# requested 2018-2024 range and uses the newer model, so it is not used.
YEAR_KEYS = {
    2018: f"{PREFIX}/2018/amazon_basin_48px_v3.2-3.7ensemble_dissolved-0.6_2018-2018cumulative.geojson",
    2019: f"{PREFIX}/2019/amazon_basin_48px_v3.2-3.7ensemble_dissolved-0.6_2018-2019cumulative.geojson",
    2020: f"{PREFIX}/2020/amazon_basin_48px_v3.2-3.7ensemble_dissolved-0.6_2018-2020cumulative.geojson",
    2021: f"{PREFIX}/2021/amazon_basin_48px_v3.2-3.7ensemble_dissolved-0.6_2018-2021cumulative.geojson",
    2022: f"{PREFIX}/2022/amazon_basin_48px_v3.2-3.7ensemble_dissolved-0.6_2018-2022cumulative.geojson",
    2023: f"{PREFIX}/2023/amazon_basin_48px_v3.2-3.7ensemble_dissolved-0.6_2018-2023cumulative.geojson",
    2024: f"{PREFIX}/2024/amazon_basin-48px_v0.X-SSL4EO-MLPensemble_2018-2024cumulative-clean.geojson",
}
YEARS = sorted(YEAR_KEYS)

TILE = 64  # 64 px @ 10 m = 640 m
BACKGROUND_ID = 0
MINE_ID = 1
PER_YEAR = 160  # positives sampled per year (7 yr * 160 -> trimmed to <=1000)
MAX_POSITIVES = 1000
# Cap grid cells drawn from a single (possibly multi-km) polygon so one giant scar does
# not dominate; we only need a bounded sample.
MAX_CELLS_PER_POLY = 64
# Negative tile placement: offset (in tiles) from a seed positive.
NEG_MIN_OFFSET = 6  # >= 3.8 km away
NEG_MAX_OFFSET = 40  # <= 25.6 km away
SEED = 42


def _raw_path(year: int):
    return io.raw_dir(SLUG) / f"{year}_{YEAR_KEYS[year].split('/')[-1]}"


def download_all() -> None:
    """Download the 7 annual GeoJSONs to raw/ (atomic, idempotent)."""
    import boto3
    import botocore

    io.raw_dir(SLUG).mkdir(parents=True, exist_ok=True)
    s3 = boto3.client(
        "s3", config=botocore.config.Config(signature_version=botocore.UNSIGNED)
    )
    for year in YEARS:
        dst = _raw_path(year)
        if dst.exists():
            continue
        tmp = dst.parent / (dst.name + ".tmp")
        s3.download_file(Bucket=BUCKET, Key=YEAR_KEYS[year], Filename=tmp.path)
        tmp.rename(dst)
    with (io.raw_dir(SLUG) / "SOURCE.txt").open("w") as f:
        f.write(
            f"Source Cooperative (Earth Genome) Amazon Mining Watch\n{URL}\n"
            f"s3://{BUCKET}/{PREFIX}/ (public, unsigned)\n"
        )


def _load_year_geoms(year: int) -> list[Any]:
    """Load a year's GeoJSON into a list of shapely geometries (lon/lat)."""
    import json

    with _raw_path(year).open() as f:
        data = json.load(f)
    return [shape(feat["geometry"]) for feat in data["features"]]


# Per-process caches for the fast scan path (populated lazily inside workers).
_ZONE_CACHE: dict[tuple[int, int], Projection] = {}
_TF_CACHE: dict[int, Any] = {}


def _utm_proj_for(lon: float, lat: float) -> Projection:
    key = (round(lon), round(lat))
    p = _ZONE_CACHE.get(key)
    if p is None:
        p = get_utm_ups_projection(lon, lat, io.RESOLUTION, -io.RESOLUTION)
        _ZONE_CACHE[key] = p
    return p


def _transformer(epsg: int) -> Any:
    tf = _TF_CACHE.get(epsg)
    if tf is None:
        from pyproj import Transformer

        tf = Transformer.from_crs("EPSG:4326", f"EPSG:{epsg}", always_xy=True)
        _TF_CACHE[epsg] = tf
    return tf


def _geom_to_pixel_space(geom: Any, tf: Any) -> Any:
    """Reproject a lon/lat geometry into UTM 10 m pixel space (px=easting/10,
    py=northing/-10) using a cached pyproj transformer, preserving holes/multipart.
    """

    def _fwd(coords: np.ndarray) -> np.ndarray:
        x, y = tf.transform(coords[:, 0], coords[:, 1])
        return np.column_stack(
            [np.asarray(x) / io.RESOLUTION, np.asarray(y) / -io.RESOLUTION]
        )

    return shapely.transform(geom, _fwd)


def _cells_for_geom(geom: Any) -> tuple[int, list[tuple[int, int]]] | None:
    """Return (epsg, [(cx, cy), ...]) grid cells a polygon actually intersects.

    Derive the local UTM zone from the centroid, reproject the polygon into UTM 10 m
    pixel space (cached transformer), enumerate the 64 px (640 m) grid cells its bbox
    spans, and keep only cells whose 64x64 box the polygon truly intersects (not just
    bbox overlap). Cells are (col//TILE, row//TILE); pixel_y = northing / -10. Bounded to
    MAX_CELLS_PER_POLY intersecting cells for very large polygons.
    """
    c = geom.centroid
    proj = _utm_proj_for(c.x, c.y)
    epsg = proj.crs.to_epsg()
    pgeom = _geom_to_pixel_space(geom, _transformer(epsg))
    if pgeom.is_empty:
        return None
    minx, miny, maxx, maxy = pgeom.bounds
    cx0, cx1 = int(np.floor(minx / TILE)), int(np.floor((maxx - 1e-6) / TILE))
    cy0, cy1 = int(np.floor(miny / TILE)), int(np.floor((maxy - 1e-6) / TILE))
    candidates = [(cx, cy) for cx in range(cx0, cx1 + 1) for cy in range(cy0, cy1 + 1)]
    if not candidates:
        return None
    # Keep only cells the polygon genuinely intersects.
    cells = [
        (cx, cy)
        for cx, cy in candidates
        if pgeom.intersects(
            shapely.box(cx * TILE, cy * TILE, cx * TILE + TILE, cy * TILE + TILE)
        )
    ]
    if not cells:
        return None
    if len(cells) > MAX_CELLS_PER_POLY:
        rng = random.Random(f"{epsg}:{cx0}:{cy0}")
        cells = rng.sample(cells, MAX_CELLS_PER_POLY)
    return epsg, cells


def _scan_chunk(job: dict[str, Any]) -> tuple[int, list[tuple[int, int, int]]]:
    """Return (year, distinct (epsg, cx, cy) positive cells) for a chunk of geometries."""
    year = job["year"]
    geoms = shapely.from_wkb(job["wkb"])
    seen: set[tuple[int, int, int]] = set()
    for g in geoms:
        try:
            res = _cells_for_geom(g)
        except Exception:
            continue
        if res is None:
            continue
        epsg, cells = res
        for cx, cy in cells:
            seen.add((epsg, cx, cy))
    return year, sorted(seen)


def _proj_for_epsg(epsg: int) -> Projection:
    return Projection(CRS.from_epsg(epsg), io.RESOLUTION, -io.RESOLUTION)


def _tile_bounds(cx: int, cy: int) -> tuple[int, int, int, int]:
    return (cx * TILE, cy * TILE, cx * TILE + TILE, cy * TILE + TILE)


def _tile_lonlat_box(proj: Projection, bounds: tuple[int, int, int, int]) -> Any:
    x0, y0, x1, y1 = bounds
    box = shapely.box(x0, y0, x1, y1)
    return STGeometry(proj, box, None).to_projection(WGS84_PROJECTION).shp


# Per-worker cache of (year -> (geoms, STRtree)) so grouped writes reuse the index.
_YEAR_CACHE: dict[int, tuple[list[Any], Any]] = {}


def _get_year_index(year: int) -> tuple[list[Any], Any]:
    if year not in _YEAR_CACHE:
        geoms = _load_year_geoms(year)
        tree = shapely.STRtree(geoms)
        _YEAR_CACHE[year] = (geoms, tree)
    return _YEAR_CACHE[year]


def _write_positive(rec: dict[str, Any]) -> int:
    sample_id = rec["sample_id"]
    tif = io.locations_dir(SLUG) / f"{sample_id}.tif"
    if tif.exists():
        return MINE_ID
    epsg, cx, cy, year = rec["epsg"], rec["cx"], rec["cy"], rec["year"]
    proj = _proj_for_epsg(epsg)
    bounds = _tile_bounds(cx, cy)
    geoms, tree = _get_year_index(year)
    lonlat_box = _tile_lonlat_box(proj, bounds)
    idxs = tree.query(lonlat_box)
    shapes = []
    for i in idxs:
        g = geoms[int(i)]
        if not g.intersects(lonlat_box):
            continue
        px = geom_to_pixels(g, WGS84_PROJECTION, proj)
        if px.is_valid and not px.is_empty:
            shapes.append((px, MINE_ID))
    if not shapes:
        return -1  # degenerate; nothing to draw
    arr = rasterize_shapes(shapes, bounds, fill=BACKGROUND_ID, dtype="uint8")
    if int(arr.max()) != MINE_ID:
        return -1  # polygon(s) missed all pixel centers -> not a positive tile
    io.write_label_geotiff(SLUG, sample_id, arr, proj, bounds, nodata=io.CLASS_NODATA)
    io.write_sample_json(
        SLUG,
        sample_id,
        proj,
        bounds,
        io.year_range(year),
        source_id=f"{year}:{epsg}:{cx}:{cy}",
        classes_present=[BACKGROUND_ID, MINE_ID],
    )
    return MINE_ID


def _write_negative(rec: dict[str, Any]) -> int:
    sample_id = rec["sample_id"]
    tif = io.locations_dir(SLUG) / f"{sample_id}.tif"
    if tif.exists():
        return BACKGROUND_ID
    epsg, cx, cy, year = rec["epsg"], rec["cx"], rec["cy"], rec["year"]
    proj = _proj_for_epsg(epsg)
    bounds = _tile_bounds(cx, cy)
    arr = np.full((1, TILE, TILE), BACKGROUND_ID, dtype=np.uint8)
    io.write_label_geotiff(SLUG, sample_id, arr, proj, bounds, nodata=io.CLASS_NODATA)
    io.write_sample_json(
        SLUG,
        sample_id,
        proj,
        bounds,
        io.year_range(year),
        source_id=f"neg:{year}:{epsg}:{cx}:{cy}",
        classes_present=[BACKGROUND_ID],
    )
    return BACKGROUND_ID


def _pick_negatives(
    positives: list[dict[str, Any]],
    positive_cells: set[tuple[int, int, int]],
    n: int,
) -> list[dict[str, Any]]:
    """Offset from seed positives into verified-empty cells (same UTM zone)."""
    rng = random.Random(SEED)
    by_year_geoms: dict[int, tuple[list[Any], Any]] = {}
    negs: list[dict[str, Any]] = []
    chosen: set[tuple[int, int, int]] = set()
    attempts = 0
    while len(negs) < n and attempts < n * 50:
        attempts += 1
        seed = rng.choice(positives)
        epsg, year = seed["epsg"], seed["year"]
        dx = rng.randint(NEG_MIN_OFFSET, NEG_MAX_OFFSET) * rng.choice([-1, 1])
        dy = rng.randint(NEG_MIN_OFFSET, NEG_MAX_OFFSET) * rng.choice([-1, 1])
        cx, cy = seed["cx"] + dx, seed["cy"] + dy
        key = (epsg, cx, cy)
        if key in positive_cells or key in chosen:
            continue
        # Verify the candidate tile touches no polygon in any year.
        proj = _proj_for_epsg(epsg)
        bounds = _tile_bounds(cx, cy)
        lonlat_box = _tile_lonlat_box(proj, bounds)
        clash = False
        for y in YEARS:
            if y not in by_year_geoms:
                geoms = _load_year_geoms(y)
                by_year_geoms[y] = (geoms, shapely.STRtree(geoms))
            geoms, tree = by_year_geoms[y]
            for i in tree.query(lonlat_box):
                if geoms[int(i)].intersects(lonlat_box):
                    clash = True
                    break
            if clash:
                break
        if clash:
            continue
        chosen.add(key)
        negs.append({"epsg": epsg, "cx": cx, "cy": cy, "year": year})
    return negs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=64)
    args = parser.parse_args()

    io.check_disk()
    download_all()
    io.check_disk()

    # Scan (parallel over feature chunks) -> earliest-year per positive grid cell.
    jobs: list[dict[str, Any]] = []
    for year in YEARS:
        geoms = _load_year_geoms(year)
        for i in range(0, len(geoms), 200):
            chunk = geoms[i : i + 200]
            jobs.append({"year": year, "wkb": shapely.to_wkb(chunk)})
    earliest: dict[tuple[int, int, int], int] = {}
    with multiprocessing.Pool(args.workers) as p:
        for year, cells in tqdm.tqdm(
            star_imap_unordered(p, _scan_chunk, [dict(job=j) for j in jobs]),
            total=len(jobs),
            desc="scan",
        ):
            for cell in cells:
                if cell not in earliest or year < earliest[cell]:
                    earliest[cell] = year
    print(f"positive grid cells (distinct locations): {len(earliest)}", flush=True)

    records = [
        {"epsg": e, "cx": cx, "cy": cy, "year": yr}
        for (e, cx, cy), yr in earliest.items()
    ]
    # Balance across years, then trim to MAX_POSITIVES.
    rng = random.Random(SEED)
    by_year: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for r in records:
        by_year[r["year"]].append(r)
    positives: list[dict[str, Any]] = []
    for yr in YEARS:
        items = by_year[yr]
        rng.shuffle(items)
        positives.extend(items[:PER_YEAR])
    rng.shuffle(positives)
    positives = positives[:MAX_POSITIVES]
    print(f"selected {len(positives)} positive tiles across years")

    positive_cells = {(r["epsg"], r["cx"], r["cy"]) for r in records}
    n_neg = min(len(positives), MAX_POSITIVES)
    negatives = _pick_negatives(positives, positive_cells, n_neg)
    print(f"selected {len(negatives)} background-only negative tiles")

    # Assign sample ids: positives first, then negatives.
    for i, r in enumerate(positives):
        r["sample_id"] = f"{i:06d}"
    for j, r in enumerate(negatives):
        r["sample_id"] = f"{len(positives) + j:06d}"

    io.check_disk()
    with multiprocessing.Pool(args.workers) as p:
        pos_res = list(
            tqdm.tqdm(
                star_imap_unordered(
                    p, _write_positive, [dict(rec=r) for r in positives]
                ),
                total=len(positives),
                desc="positives",
            )
        )
        neg_res = list(
            tqdm.tqdm(
                star_imap_unordered(
                    p, _write_negative, [dict(rec=r) for r in negatives]
                ),
                total=len(negatives),
                desc="negatives",
            )
        )

    n_pos_written = sum(1 for r in pos_res if r == MINE_ID)
    n_neg_written = sum(1 for r in neg_res if r == BACKGROUND_ID)
    n_degenerate = sum(1 for r in pos_res if r == -1)
    total = n_pos_written + n_neg_written
    year_counts = Counter(r["year"] for r in positives)

    io.write_dataset_metadata(
        SLUG,
        {
            "dataset": SLUG,
            "name": NAME,
            "task_type": "classification",
            "source": "Source Cooperative (Earth Genome)",
            "license": "CC-BY-4.0",
            "provenance": {
                "url": URL,
                "have_locally": False,
                "annotation_method": "ML + manual review (Sentinel-2)",
            },
            "sensors_relevant": ["sentinel2", "sentinel1", "landsat"],
            "classes": [
                {
                    "id": BACKGROUND_ID,
                    "name": "background",
                    "description": "Non-mine surface (forest, water, other land cover) with no detected artisanal gold-mine scar.",
                },
                {
                    "id": MINE_ID,
                    "name": "mine_scar",
                    "description": "Artisanal / illegal gold-mine scar detected in Sentinel-2, per Amazon Mining Watch (Earth Genome). Areal bare-earth/tailings-pond footprint of alluvial gold mining.",
                },
            ],
            "nodata_value": io.CLASS_NODATA,
            "num_samples": total,
            "num_positive_tiles": n_pos_written,
            "num_negative_tiles": n_neg_written,
            "class_counts": {"background": total, "mine_scar": n_pos_written},
            "tile_size": TILE,
            "year_counts": {str(y): year_counts.get(y, 0) for y in YEARS},
            "notes": (
                "64x64 UTM 10 m segmentation tiles; mine_scar=1 rasterized from cumulative "
                "annual GeoJSON polygons, background=0 elsewhere; grid-aligned 640 m cells. "
                "Each cell assigned to earliest year it appears (files are cumulative from "
                "2018); ~1-year time range. Background-only negative tiles placed 3.8-25.6 km "
                "from positives (verified polygon-free across all years). Years 2018-2024 "
                "(2025 quarterly files excluded). Note: 2024 uses a newer, more sensitive "
                "model than 2018-2023 (per source README), so cross-year trends are not "
                f"reliable. {n_degenerate} candidate cells dropped as degenerate."
            ),
        },
    )
    print(
        f"done: {n_pos_written} mine_scar tiles, {n_neg_written} background tiles, "
        f"total {total} (dropped {n_degenerate} degenerate)"
    )


if __name__ == "__main__":
    multiprocessing.set_start_method("forkserver", force=True)
    main()
