"""Process the Colorado Geological Survey statewide alluvial-fan / debris-fan inventory
into landform segmentation label tiles (label_type: polygons).

Source: Colorado Geological Survey (CGS) Online Series ON-006 "Alluvial Fan Mapping of
Colorado", a statewide effort of county-specific LiDAR-derived polygon inventories of
alluvial fans and high-angle / debris fans (areas at risk of debris flows / mudflows,
especially post-wildfire). Free public data. The manifest url points at the Teller County
publication (ON-006-29D), but CGS publishes the whole statewide inventory as a single
queryable ArcGIS REST MapServer:

    https://cgsarcimage.mines.edu/arcgis/rest/services/Hazards/
        ON_006_All_Current_Alluvial_Fan_Mapping_Colorado/MapServer

which we use instead of the individual per-county ZIPs (each county ZIP is gated behind a
Gravity Form email-capture download on the CGS website; the REST service delivers the same
polygons directly, label-only, with no credential). The service exposes, per county, an
"Alluvial Fans" polygon layer and a "High Angle (Debris) Fans" polygon layer (plus point /
county-outline layers we ignore). We pull every county's two fan-polygon layers via the
REST ``query`` endpoint (paged, GeoJSON, outSR=4326) into raw/.

Counties covered (as of 2026-07): Boulder, Chaffee, Clear Creek, Fremont, Garfield,
Gilpin, Lake, Pitkin, Summit, Teller -> 8,577 fan polygons total (7,208 alluvial fans +
1,369 high-angle/debris fans).

Task: **alluvial-fan landform segmentation** (polygons -> classification):

    0 = background            (mapped terrain that is not a delineated fan -- a genuine
                               observed negative: CGS comprehensively mapped fans within
                               each county study area, so non-fan pixels around a fan are
                               real "not-a-fan" terrain, not fabricated negatives)
    1 = alluvial_fan          (a gently-sloping alluvial fan landform)
    2 = high_angle_debris_fan (a high-angle fan / debris fan: steeper landform near the
                               apex / feeder channel, downslope of the source area)

The manifest lists two classes ("alluvial fan", "high-angle/debris fan"); we add a
``background`` class (id 0) exactly as the analogous USGS karst closed-depression dataset
does, because the surrounding terrain is genuinely observed non-fan context (documented
judgment call).

Tiling: each fan polygon seeds one 64x64 (640 m) tile in a local UTM projection at 10
m/pixel, centered on the fan. **All** fan polygons (of either class) that intersect the
tile are rasterized (so an alluvial fan and an adjacent high-angle fan at its apex are both
labeled, and non-fan pixels stay background), using ``all_touched=True`` so even a
sub-640 m fan contributes >=1 positive pixel. A fan larger than 64 px on an axis (rare) is
gridded into non-overlapping 64x64 windows (up to MAX_TILES_PER_FEATURE). High-angle fans
are burned after alluvial fans so the steeper class wins any overlap.

Selection: class-balanced by the seed fan's class (spec 5), up to PER_CLASS=1000 tiles per
fan class (alluvial fans are subsampled from 7,208; all/most high-angle fans kept), well
under the 25,000 cap.

Time: fans are static topographic landforms, persistent across the Sentinel era. There is
no per-fan date (LiDAR compiled across ~2016-2026), so a representative 1-year window
(REP_YEAR) is assigned and ``change_time`` is null.

Observability: alluvial / debris fans are 10^3-10^6 m^2 landforms, readily resolved at
10-30 m. A small MIN_AREA_M2 floor drops tiny slivers only.

Run: python3 -m olmoearth_pretrain.open_set_segmentation_data.datasets.colorado_alluvial_debris_fan_mapping
Idempotent: existing locations/{id}.tif are skipped; raw GeoJSON layers are skipped if present.
"""

import argparse
import json
import math
import multiprocessing
import random
from collections import Counter
from typing import Any

import numpy as np
import shapely
import shapely.geometry
import shapely.wkb
import tqdm
from pyproj import Geod
from rslearn.const import WGS84_PROJECTION
from rslearn.utils.geometry import Projection
from rslearn.utils.mp import star_imap_unordered

from olmoearth_pretrain.open_set_segmentation_data import (
    download,
    io,
    manifest,
    sampling,
)

SLUG = "colorado_alluvial_debris_fan_mapping"
NAME = "Colorado Alluvial & Debris Fan Mapping"

BASE_URL = (
    "https://cgsarcimage.mines.edu/arcgis/rest/services/Hazards/"
    "ON_006_All_Current_Alluvial_Fan_Mapping_Colorado/MapServer"
)
PUB_URL = "https://coloradogeologicalsurvey.org/publications/alluvial-fan-map-data-teller-colorado/"
DOI = "https://doi.org/10.58783/cgs.on00629d.bfqt5710"
UA = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/122 Safari/537.36"
)

# (layer_id, county, class_id) for the fan-polygon layers of the statewide MapServer.
# Alluvial-fan layers -> ALLUVIAL(1); high-angle/debris-fan layers -> HIGH_ANGLE(2).
BG, ALLUVIAL, HIGH_ANGLE = 0, 1, 2
FAN_LAYERS: list[tuple[int, str, int]] = [
    (4, "Boulder", ALLUVIAL),
    (5, "Boulder", HIGH_ANGLE),
    (9, "Chaffee", ALLUVIAL),
    (10, "Chaffee", HIGH_ANGLE),
    (14, "ClearCreek", ALLUVIAL),
    (15, "ClearCreek", HIGH_ANGLE),
    (19, "Fremont", ALLUVIAL),
    (20, "Fremont", HIGH_ANGLE),
    (24, "Garfield", ALLUVIAL),
    (25, "Garfield", HIGH_ANGLE),
    (30, "Gilpin", ALLUVIAL),
    (31, "Gilpin", HIGH_ANGLE),
    (35, "Lake", ALLUVIAL),
    (36, "Lake", HIGH_ANGLE),
    (40, "Pitkin", ALLUVIAL),
    (41, "Pitkin", HIGH_ANGLE),
    (45, "Summit", ALLUVIAL),
    (46, "Summit", HIGH_ANGLE),
    (50, "Teller", ALLUVIAL),
    (51, "Teller", HIGH_ANGLE),
]

CLASSES = [
    (
        "background",
        "Mapped terrain that is not a delineated fan: genuine observed non-fan context "
        "within a county study area. CGS comprehensively maps fans per county, so "
        "out-of-polygon pixels around a fan are real negatives (no synthetic negatives).",
    ),
    (
        "alluvial_fan",
        "A gently-sloping alluvial fan: a fan-shaped deposit of sediment built where a "
        "channel emerges from a confined valley/mountain front onto lower-gradient ground. "
        "Mapped from Colorado LiDAR (2-5 ft contours + terrain metrics); at risk of "
        "sediment-laden flooding / debris flows, especially after wildfire.",
    ),
    (
        "high_angle_debris_fan",
        "A high-angle fan / debris fan: a steeper (mean slope typically >20 deg) fan "
        "landform located downslope of the fan apex, feeder channel and source area. "
        "Mapped from LiDAR-derived contours; higher debris-flow / mudflow hazard.",
    ),
]

TILE = 64
REP_YEAR = 2020  # representative Sentinel-era year (fans are static; LiDAR ~2016-2026)
MIN_AREA_M2 = 900.0  # observability floor (~1 Landsat 30 m pixel); fans are much larger
MAX_TILES_PER_FEATURE = 16
PER_CLASS = 1000
MAX_SAMPLES = sampling.MAX_SAMPLES_PER_DATASET  # 25000
GEOD = Geod(ellps="WGS84")

# Per-worker globals (populated by _init_worker): spatial index over ALL fan polygons so a
# tile can be labeled with every fan (of either class) that overlaps it.
_GEOMS: list[Any] = []
_GCLASS: list[int] = []
_TREE: Any = None


# --------------------------------------------------------------------------- download


def raw_geojson_path(layer_id: int, county: str) -> Any:
    return io.raw_dir(SLUG) / f"layer_{layer_id:02d}_{county}.geojson"


def download_layers() -> None:
    """Download each county's two fan-polygon layers as GeoJSON into raw/ (idempotent)."""
    io.raw_dir(SLUG).mkdir(parents=True, exist_ok=True)
    with (io.raw_dir(SLUG) / "SOURCE.txt").open("w") as f:
        f.write(
            "Colorado Geological Survey -- statewide Alluvial Fan Mapping (ON-006).\n"
            "Free public data. Manifest publication (Teller Co): "
            f"{PUB_URL}\nDOI (Teller ON-006-29D): {DOI}\n"
            f"ArcGIS REST MapServer (all counties): {BASE_URL}\n"
            "Downloaded fan-polygon layers (per county: Alluvial Fans + High Angle/Debris "
            "Fans) via the REST query endpoint (f=geojson, outSR=4326, paged). Point and "
            "county-outline layers ignored. County ZIPs on the CGS site are gated behind a "
            "Gravity Form email-capture download; the REST service delivers the same "
            "polygons directly with no credential.\n"
        )
    for layer_id, county, _cls in FAN_LAYERS:
        dst = raw_geojson_path(layer_id, county)
        if dst.exists():
            print(f"  [skip] {dst.name}")
            continue
        print(f"  downloading layer {layer_id} ({county}) -> {dst.name}")
        download.download_arcgis_layer(
            BASE_URL, layer_id, dst, out_sr=4326, page=2000, headers={"User-Agent": UA}
        )


def load_all_fans() -> list[dict[str, Any]]:
    """Read every raw GeoJSON layer into flat fan records (WGS84 shapely geom + class)."""
    fans: list[dict[str, Any]] = []
    for layer_id, county, cls in FAN_LAYERS:
        path = raw_geojson_path(layer_id, county)
        with path.open() as f:
            fc = json.load(f)
        for i, feat in enumerate(fc.get("features", [])):
            geom_json = feat.get("geometry")
            if not geom_json:
                continue
            try:
                geom = shapely.geometry.shape(geom_json)
            except Exception:  # noqa: BLE001
                continue
            if geom.is_empty or not geom.is_valid:
                geom = geom.buffer(0)
            if geom.is_empty:
                continue
            oid = feat.get("id", i)
            fans.append(
                {
                    "geom": geom,
                    "class_id": cls,
                    "source_id": f"{county}/L{layer_id}/OID{oid}",
                }
            )
    return fans


# --------------------------------------------------------------------------- worker init


def _init_worker(fan_wkbs: list[bytes], fan_classes: list[int]) -> None:
    from shapely.strtree import STRtree

    global _GEOMS, _GCLASS, _TREE
    _GEOMS = [shapely.wkb.loads(w) for w in fan_wkbs]
    _GCLASS = list(fan_classes)
    _TREE = STRtree(_GEOMS)


# --------------------------------------------------------------------------- tiling


def _candidate_task(rec: dict[str, Any]) -> list[dict[str, Any]]:
    """Return candidate tile records (crs + pixel bounds + wgs84 bbox) seeding on one fan.

    Runs inside the worker pool (per-fan reprojection is the expensive step -- LiDAR fan
    polygons have many vertices and each needs a pyproj transform, so this is parallelized).
    """
    from rslearn.utils.geometry import STGeometry

    from olmoearth_pretrain.open_set_segmentation_data.rasterize import geom_to_pixels

    geom = shapely.wkb.loads(rec["wkb"])
    idx = rec["idx"]
    c = geom.centroid
    proj = io.utm_projection_for_lonlat(float(c.x), float(c.y))
    px = geom_to_pixels(geom, WGS84_PROJECTION, proj)
    if px.is_empty:
        return []
    minx, miny, maxx, maxy = px.bounds
    w, h = maxx - minx, maxy - miny
    crs = proj.crs.to_string()
    out: list[dict[str, Any]] = []

    def make(bounds: tuple[int, int, int, int]) -> dict[str, Any]:
        # WGS84 bbox of the tile (for the neighbor spatial query in the worker).
        box = shapely.geometry.box(*bounds)
        ll = STGeometry(proj, box, None).to_projection(WGS84_PROJECTION).shp.bounds
        return {
            "crs": crs,
            "bounds": list(bounds),
            "wgs84_bbox": list(ll),
            "seed_class": rec["class_id"],
            "source_id": rec["source_id"],
        }

    if w <= TILE and h <= TILE:
        col = round((minx + maxx) / 2.0)
        row = round((miny + maxy) / 2.0)
        out.append(make(io.centered_bounds(col, row, TILE, TILE)))
        return out

    # Large fan (rare): sample up to MAX_TILES_PER_FEATURE non-overlapping 64x64 windows
    # that intersect the fan. Bounded random sampling of grid positions keeps this cheap
    # even if a (possibly malformed) polygon has an enormous pixel bbox.
    from shapely.prepared import prep

    x0, y0 = math.floor(minx), math.floor(miny)
    nx = max(1, math.ceil(w / TILE))
    ny = max(1, math.ceil(h / TILE))
    prepared = prep(px)
    rng = random.Random(idx)
    seen: set[tuple[int, int]] = set()
    max_tries = min(nx * ny, 3000)
    tries = 0
    while len(out) < MAX_TILES_PER_FEATURE and tries < max_tries:
        i, j = rng.randrange(nx), rng.randrange(ny)
        tries += 1
        if (i, j) in seen:
            continue
        seen.add((i, j))
        b = (x0 + i * TILE, y0 + j * TILE, x0 + i * TILE + TILE, y0 + j * TILE + TILE)
        if prepared.intersects(shapely.geometry.box(*b)):
            out.append(make(b))
    return out


def _write_one(rec: dict[str, Any]) -> str | None:
    from rasterio.crs import CRS

    from olmoearth_pretrain.open_set_segmentation_data.rasterize import (
        geom_to_pixels,
        rasterize_shapes,
    )

    sample_id = rec["sample_id"]
    if (io.locations_dir(SLUG) / f"{sample_id}.tif").exists():
        return None

    proj = Projection(CRS.from_string(rec["crs"]), io.RESOLUTION, -io.RESOLUTION)
    bounds = tuple(rec["bounds"])
    tile_box_px = shapely.geometry.box(*bounds)

    # Find every fan overlapping the tile (WGS84 bbox query), reproject + clip to the tile.
    query_box = shapely.geometry.box(*rec["wgs84_bbox"])
    hits = _TREE.query(query_box) if _TREE is not None else []
    shapes: list[tuple[Any, int]] = []
    for j in np.atleast_1d(hits):
        j = int(j)
        gclass = _GCLASS[j]
        px = geom_to_pixels(_GEOMS[j], WGS84_PROJECTION, proj)
        if px.is_empty:
            continue
        clip = px.intersection(tile_box_px)
        if clip.is_empty:
            continue
        shapes.append((clip, gclass))
    # Burn alluvial (1) first, high-angle (2) last so the steeper class wins overlaps.
    shapes.sort(key=lambda s: s[1])

    label = rasterize_shapes(shapes, bounds, fill=BG, dtype="uint8", all_touched=True)[
        0
    ]
    present = sorted(int(v) for v in np.unique(label))
    time_range = io.year_range(REP_YEAR)
    io.write_label_geotiff(SLUG, sample_id, label, proj, bounds, nodata=io.CLASS_NODATA)
    io.write_sample_json(
        SLUG,
        sample_id,
        proj,
        bounds,
        time_range,
        change_time=None,
        source_id=rec["source_id"],
        classes_present=present,
    )
    tag = "+".join(str(p) for p in present)
    return tag


# --------------------------------------------------------------------------- main


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=64)
    ap.add_argument("--per_class", type=int, default=PER_CLASS)
    ap.add_argument("--min_area_m2", type=float, default=MIN_AREA_M2)
    args = ap.parse_args()

    io.check_disk()
    manifest.write_registry_entry(SLUG, "in_progress")

    download_layers()
    io.check_disk()

    fans = load_all_fans()
    print(f"loaded {len(fans)} fan polygons from {len(FAN_LAYERS)} layers")

    # Areas (geodesic, m^2) + observability filter + class counts.
    kept: list[dict[str, Any]] = []
    areas = []
    dropped = 0
    for fan in fans:
        area = abs(GEOD.geometry_area_perimeter(fan["geom"])[0])
        if area < args.min_area_m2:
            dropped += 1
            continue
        fan["area_m2"] = area
        areas.append(area)
        kept.append(fan)
    a = np.array(areas)
    pct = {
        f"p{q}": round(float(np.percentile(a, q)), 1) for q in (0, 25, 50, 75, 95, 100)
    }
    src_counts = Counter(f["class_id"] for f in kept)
    print(f"kept {len(kept)} fans (dropped {dropped} < {args.min_area_m2} m^2)")
    print("area(m^2) distribution:", pct)
    print(
        "source class counts:",
        {ALLUVIAL: src_counts[ALLUVIAL], HIGH_ANGLE: src_counts[HIGH_ANGLE]},
    )

    # Candidate tiles (one+ per fan). Per-fan reprojection is the expensive step (many
    # vertices per LiDAR polygon), so run it across the worker pool.
    cand_recs = [
        dict(
            rec={
                "wkb": shapely.wkb.dumps(f["geom"]),
                "class_id": f["class_id"],
                "source_id": f["source_id"],
                "idx": i,
            }
        )
        for i, f in enumerate(kept)
    ]
    candidates: list[dict[str, Any]] = []
    with multiprocessing.Pool(args.workers) as p:
        for cands in tqdm.tqdm(
            star_imap_unordered(p, _candidate_task, cand_recs),
            total=len(cand_recs),
            desc="candidates",
        ):
            candidates.extend(cands)
    print(f"{len(candidates)} candidate tiles")

    # Class-balanced selection by the seed fan's class (spec 5): up to per_class per class.
    selected = sampling.balance_by_class(
        candidates, "seed_class", per_class=args.per_class, total_cap=MAX_SAMPLES
    )
    for j, r in enumerate(selected):
        r["sample_id"] = f"{j:06d}"
    print(
        f"selected {len(selected)} tiles (<= {args.per_class}/fan class, cap {MAX_SAMPLES})"
    )

    # Write tiles in parallel; each worker holds an STRtree over all fans for neighbor labels.
    fan_wkbs = [shapely.wkb.dumps(f["geom"]) for f in kept]
    fan_classes = [f["class_id"] for f in kept]
    io.check_disk()
    present_counts: Counter = Counter()
    with multiprocessing.Pool(
        args.workers, initializer=_init_worker, initargs=(fan_wkbs, fan_classes)
    ) as p:
        for res in tqdm.tqdm(
            star_imap_unordered(p, _write_one, [dict(rec=r) for r in selected]),
            total=len(selected),
            desc="write tiles",
        ):
            if res is not None:
                present_counts[res] += 1

    # Count how many written tiles contain each class (from classes_present tags).
    tiles_with_class = Counter()
    for tag, n in present_counts.items():
        for p_ in tag.split("+"):
            tiles_with_class[int(p_)] += n
    n_written = len(list(io.locations_dir(SLUG).glob("*.tif")))
    seed_counts = Counter(r["seed_class"] for r in selected)

    io.write_dataset_metadata(
        SLUG,
        {
            "dataset": SLUG,
            "name": NAME,
            "task_type": "classification",
            "source": "Colorado Geological Survey",
            "license": "free public",
            "provenance": {
                "url": PUB_URL,
                "doi": DOI,
                "arcgis_service": BASE_URL,
                "have_locally": False,
                "annotation_method": "manual mapping from Colorado LiDAR (2-5 ft contours + terrain metrics)",
            },
            "sensors_relevant": ["sentinel2", "sentinel1", "landsat"],
            "classes": [
                {"id": i, "name": name, "description": desc}
                for i, (name, desc) in enumerate(CLASSES)
            ],
            "nodata_value": io.CLASS_NODATA,
            "num_samples": n_written,
            "counties": sorted({c for _, c, _ in FAN_LAYERS}),
            "n_source_polygons": len(fans),
            "n_kept_after_area_filter": len(kept),
            "area_filter_m2": args.min_area_m2,
            "area_distribution_m2": pct,
            "source_class_counts": {
                "alluvial_fan": src_counts[ALLUVIAL],
                "high_angle_debris_fan": src_counts[HIGH_ANGLE],
            },
            "seed_tile_counts": {
                "alluvial_fan": seed_counts[ALLUVIAL],
                "high_angle_debris_fan": seed_counts[HIGH_ANGLE],
            },
            "tiles_containing_class": {
                CLASSES[c][0]: tiles_with_class.get(c, 0) for c in (0, 1, 2)
            },
            "rep_year": REP_YEAR,
            "notes": (
                "Statewide CGS alluvial-fan / high-angle-debris-fan landform segmentation "
                "from the ON-006 ArcGIS REST service (10 counties). 64x64 uint8 tiles in "
                "local UTM at 10 m; 0=background (genuine observed non-fan terrain), "
                "1=alluvial_fan, 2=high_angle_debris_fan (255 nodata declared, unused). "
                "Each tile seeds on one fan and rasterizes ALL fans overlapping it "
                "(all_touched=True; high-angle burned last). Static landforms -> "
                f"representative 1-year window (REP_YEAR={REP_YEAR}); change_time null. "
                "Class-balanced by seed fan class up to 1000/class; alluvial fans "
                "subsampled from the larger pool. background added as a class (not in the "
                "2-class manifest) matching the USGS karst-depression precedent. Access: "
                "county ZIPs are behind an email-capture Gravity Form, so labels pulled "
                "from the public REST query endpoint (no credential)."
            ),
        },
    )
    print("tiles containing each class:", dict(tiles_with_class))
    print("total tif on disk:", n_written)
    manifest.write_registry_entry(
        SLUG, "completed", task_type="classification", num_samples=n_written
    )
    print("done")


if __name__ == "__main__":
    multiprocessing.set_start_method("forkserver", force=True)
    main()
