"""Process the USPVDB (US Large-Scale Solar PV Database) into open-set-segmentation tiles.

Source (external, USGS / LBNL, public domain): https://energy.usgs.gov/uspvdb/
The U.S. Large-Scale Solar Photovoltaic Database provides the **array-boundary polygons**
(and centroids) of every U.S. front-of-the-meter ground-mounted PV facility with capacity
>= 1 MW, digitized and position-verified from aerial imagery and quality-checked. Each
facility record carries an installation/commissioning year ``p_year`` (year the facility's
installation was completed).

We download only the LABEL polygons (no imagery -- pretraining supplies imagery) from the
public ArcGIS FeatureServer as one GeoJSON FeatureCollection (6,611 facility polygons; a
Cloudflare edge rejects the default urllib User-Agent, so a browser UA header is sent):
  https://energy.usgs.gov/arcgis/rest/services/Hosted/uspvdbDyn/FeatureServer/0
Each feature is one facility (unique ``case_id``) with a Polygon/MultiPolygon footprint in
WGS84 plus ``p_year``, ``xlong``/``ylat`` (centroid), ``p_area`` (m^2), ``p_cap_ac`` (MW).

Decisions (spec sections 2-5):
  * label_type polygons -> POLYGON rasterization recipe (spec section 4). Large-scale solar
    farms are large, high-contrast footprints (median ~25 px = ~250 m across at 10 m; many
    exceed a 640 m tile) -> clearly observable at 10 m from Sentinel-2/Landsat.
  * Task = presence/state classification (spec section 2 option a). Classes: 0 background,
    1 solar_pv. USPVDB is a COMPLETE inventory of U.S. >=1 MW PV, so within any tile every
    PV facility is in the database -> non-panel pixels are TRUE negatives (like the Stanford
    well-pad dataset). Each tile rasterizes ALL facility polygons intersecting it as class 1;
    the rest is background 0. We emit positive (panel) tiles + background-only NEGATIVE tiles
    sampled inside CONUS away from any known PV (complete-coverage negatives, spec section 5).
  * CHANGE vs PRESENCE (spec section 5 timing rule): ``p_year`` is year-granular only, so the
    commissioning event is NOT resolvable to ~1-2 months and CANNOT be used as a change label.
    Instead we use the persistent post-construction STATE (a built solar farm stays visible
    for years) as presence classification with change_time=null, anchoring each tile's 1-year
    window in the Sentinel-2 era AFTER commissioning so the panels are guaranteed present:
    window_year = clamp(p_year + 1, 2017, 2024). A farm built in 2010 is still visible in
    2017; a farm built in 2022 gets a 2023 window. This keeps every facility (incl. pre-2016
    ones) usable while honoring the post-2016 rule. Negatives get a static 2022 window.
  * Tile = 64x64 (640 m @ 10 m) centered on the facility centroid. Large farms fill the tile
    (mostly class 1, still a valid presence label); small farms appear as a class-1 blob with
    background context. all_touched rasterization so small facilities are not lost.
  * Sampling (spec section 5): single foreground class -> up to PER_CLASS (1000) positive
    solar tiles + N_NEGATIVES (1000) background tiles (well under the 25k cap), matching the
    well-pad/turbine detection precedent.

Classes: 0 background, 1 solar_pv.

Run (idempotent; skips already-written tiles):
  python3 -m olmoearth_pretrain.open_set_segmentation_data.datasets.uspvdb_us_large_scale_solar_pv_database
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
import tqdm
from rasterio.crs import CRS
from rslearn.const import WGS84_PROJECTION
from rslearn.utils.geometry import Projection, STGeometry
from rslearn.utils.mp import star_imap_unordered

from olmoearth_pretrain.open_set_segmentation_data import (
    download,
    io,
    manifest,
    rasterize,
)

SLUG = "uspvdb_us_large_scale_solar_pv_database"
NAME = "USPVDB (US Large-Scale Solar PV Database)"
URL = "https://energy.usgs.gov/uspvdb/"
FEATURESERVER = (
    "https://energy.usgs.gov/arcgis/rest/services/Hosted/uspvdbDyn/FeatureServer"
)
GEOJSON = "uspvdb.geojson"
UA = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0 Safari/537.36"
    )
}

BACKGROUND_ID = 0
SOLAR_ID = 1
CLASS_NAMES = {BACKGROUND_ID: "background", SOLAR_ID: "solar_pv"}

TILE = io.MAX_TILE  # 64 px @ 10 m = 640 m
PER_CLASS = 1000  # positive solar tiles (single foreground class, spec section 5)
N_NEGATIVES = 1000  # background-only tiles sampled inside CONUS away from any PV
NEG_YEAR = 2022  # static representative window for background negatives (post-2016)
WINDOW_MIN, WINDOW_MAX = 2017, 2024  # S2-era post-commissioning window clamp
NEG_MIN_KM = 3.0  # negatives must be >= this from any facility centroid
NEG_OFF_KM = (
    15.0,
    60.0,
)  # negative offset distance from a random facility (guarantees land)
SEED = 42


def _download() -> str:
    raw = io.raw_dir(SLUG)
    raw.mkdir(parents=True, exist_ok=True)
    out = raw / GEOJSON
    download.download_arcgis_layer(
        FEATURESERVER,
        0,
        out,
        order_field="objectid",
        page=2000,
        out_sr=4326,
        headers=UA,
    )
    with (raw / "SOURCE.txt").open("w") as f:
        f.write(
            "USPVDB (US Large-Scale Solar PV Database), USGS / LBNL, public domain.\n"
            f"{URL}\n"
            f"Label-only download of facility array-boundary polygons from the ArcGIS "
            f"FeatureServer (browser UA header required):\n{FEATURESERVER}/0\n"
            "6,611 facilities (unique case_id); Polygon/MultiPolygon footprints in WGS84 "
            "with p_year (commissioning year), xlong/ylat (centroid), p_area, p_cap_ac. "
            "Imagery is supplied by pretraining, not downloaded here.\n"
        )
    return str(out)


def _load_facilities() -> list[dict[str, Any]]:
    """Parse the GeoJSON into per-facility records (one feature = one facility)."""
    path = io.raw_dir(SLUG) / GEOJSON
    fc = json.load(path.open())
    facs: list[dict[str, Any]] = []
    for f in fc["features"]:
        p = f["properties"]
        try:
            geom = shapely.geometry.shape(f["geometry"])
        except Exception:
            continue
        if geom.is_empty:
            continue
        if not geom.is_valid:
            geom = geom.buffer(0)
            if geom.is_empty:
                continue
        lon, lat = p.get("xlong"), p.get("ylat")
        if lon is None or lat is None:
            c = geom.centroid
            lon, lat = float(c.x), float(c.y)
        facs.append(
            {
                "case_id": p.get("case_id"),
                "lon": float(lon),
                "lat": float(lat),
                "p_year": int(p["p_year"]) if p.get("p_year") else None,
                "geom": geom,
                "src": f"uspvdb/case_id={p.get('case_id')}/{p.get('p_name')}",
            }
        )
    return facs


def _window_year(p_year: int | None) -> int:
    if not p_year:
        return NEG_YEAR
    return max(WINDOW_MIN, min(WINDOW_MAX, p_year + 1))


def _tile_geoms(
    tree: shapely.STRtree, geoms: list[Any], lon: float, lat: float
) -> dict[str, Any]:
    """Build a 64x64 tile centered on (lon, lat) and gather intersecting facility geoms."""
    proj, col, row = io.lonlat_to_utm_pixel(lon, lat)
    bounds = io.centered_bounds(col, row, TILE, TILE)
    tile_box = shapely.box(bounds[0], bounds[1], bounds[2], bounds[3])
    tile_wgs84 = STGeometry(proj, tile_box, None).to_projection(WGS84_PROJECTION).shp
    hits = tree.query(tile_wgs84)
    wkbs: list[bytes] = []
    for i in np.atleast_1d(hits).tolist():
        g = geoms[i]
        if g.intersects(tile_wgs84):
            wkbs.append(shapely.to_wkb(g))
    return {
        "crs": proj.crs.to_string(),
        "bounds": list(bounds),
        "geoms_wkb": wkbs,
    }


def _write_sample(rec: dict[str, Any]) -> str:
    sample_id = rec["sample_id"]
    if (io.locations_dir(SLUG) / f"{sample_id}.tif").exists():
        return "skip"
    proj = Projection(CRS.from_string(rec["crs"]), io.RESOLUTION, -io.RESOLUTION)
    bounds = tuple(rec["bounds"])
    shapes: list[tuple[Any, int]] = []
    for wkb in rec["geoms_wkb"]:
        g = shapely.from_wkb(wkb)
        gp = rasterize.geom_to_pixels(g, WGS84_PROJECTION, proj)
        if not gp.is_empty:
            shapes.append((gp, SOLAR_ID))
    if shapes:
        arr = rasterize.rasterize_shapes(
            shapes, bounds, fill=BACKGROUND_ID, dtype="uint8", all_touched=True
        )
    else:
        w, h = bounds[2] - bounds[0], bounds[3] - bounds[1]
        arr = np.full((1, h, w), BACKGROUND_ID, dtype=np.uint8)
    io.write_label_geotiff(SLUG, sample_id, arr, proj, bounds, nodata=io.CLASS_NODATA)
    io.write_sample_json(
        SLUG,
        sample_id,
        proj,
        bounds,
        io.year_range(rec["window_year"]),
        change_time=None,
        source_id=rec["src"],
        classes_present=sorted(set(np.unique(arr).tolist())),
    )
    return "pos" if shapes else "neg"


def _make_negatives(
    facs: list[dict[str, Any]], n: int, rng: random.Random
) -> list[tuple[float, float]]:
    """Sample n background points inside CONUS, >= NEG_MIN_KM from any facility centroid.

    Each candidate is a random facility centroid offset by a random 15-60 km vector; since
    facilities sit on land within the US, an offset of that size almost always stays on land
    within the country. We then reject any candidate within NEG_MIN_KM of a facility.
    """
    flon = np.array([f["lon"] for f in facs])
    flat = np.array([f["lat"] for f in facs])
    flon_r = np.radians(flon)
    flat_r = np.radians(flat)
    lon_lo, lon_hi = float(flon.min()), float(flon.max())
    lat_lo, lat_hi = float(flat.min()), float(flat.max())
    out: list[tuple[float, float]] = []
    attempts = 0
    while len(out) < n and attempts < n * 200:
        attempts += 1
        base = facs[rng.randrange(len(facs))]
        d_km = rng.uniform(*NEG_OFF_KM)
        bearing = rng.uniform(0, 2 * math.pi)
        dlat = (d_km * math.cos(bearing)) / 111.0
        dlon = (d_km * math.sin(bearing)) / (
            111.0 * math.cos(math.radians(base["lat"]))
        )
        lon = base["lon"] + dlon
        lat = base["lat"] + dlat
        if not (lon_lo <= lon <= lon_hi and lat_lo <= lat <= lat_hi):
            continue
        # Vectorized haversine to nearest facility.
        lo_r, la_r = math.radians(lon), math.radians(lat)
        dphi = flat_r - la_r
        dlmb = flon_r - lo_r
        a = (
            np.sin(dphi / 2) ** 2
            + np.cos(la_r) * np.cos(flat_r) * np.sin(dlmb / 2) ** 2
        )
        dist_km = 6371.0 * 2 * np.arcsin(np.sqrt(a))
        if dist_km.min() >= NEG_MIN_KM:
            out.append((lon, lat))
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=64)
    args = parser.parse_args()

    io.check_disk()
    manifest.write_registry_entry(SLUG, "in_progress")

    _download()
    io.check_disk()

    facs = _load_facilities()
    print(f"loaded {len(facs)} facilities", flush=True)
    geoms = [f["geom"] for f in facs]
    tree = shapely.STRtree(geoms)

    rng = random.Random(SEED)

    # Positives: up to PER_CLASS facilities, one 64x64 tile each.
    order = list(range(len(facs)))
    rng.shuffle(order)
    sel_pos = order[:PER_CLASS]
    pos_recs: list[dict[str, Any]] = []
    for i in sel_pos:
        f = facs[i]
        t = _tile_geoms(tree, geoms, f["lon"], f["lat"])
        t.update(window_year=_window_year(f["p_year"]), src=f["src"], kind="pos")
        pos_recs.append(t)
    print(f"prepared {len(pos_recs)} positive tiles", flush=True)

    # Negatives: background-only tiles inside CONUS away from any PV.
    neg_pts = _make_negatives(facs, N_NEGATIVES, rng)
    neg_recs: list[dict[str, Any]] = []
    for lon, lat in neg_pts:
        t = _tile_geoms(
            tree, geoms, lon, lat
        )  # normally empty; robust if a farm clips in
        t.update(
            window_year=NEG_YEAR, src=f"background/{lon:.4f},{lat:.4f}", kind="neg"
        )
        neg_recs.append(t)
    print(f"prepared {len(neg_recs)} background negative tiles", flush=True)

    all_recs = pos_recs + neg_recs
    all_recs.sort(key=lambda r: (r["crs"], r["bounds"][0], r["bounds"][1]))
    for idx, r in enumerate(all_recs):
        r["sample_id"] = f"{idx:06d}"

    io.check_disk()
    results: Counter = Counter()
    class_tile_counts: Counter = Counter()
    with multiprocessing.Pool(args.workers) as p:
        for res in tqdm.tqdm(
            star_imap_unordered(p, _write_sample, [dict(rec=r) for r in all_recs]),
            total=len(all_recs),
        ):
            results[res] += 1
    print("write results:", dict(results), flush=True)

    # Class tile counts (a tile counts toward every class present in it).
    n_pos = len(pos_recs)
    class_tile_counts[SOLAR_ID] = n_pos
    class_tile_counts[BACKGROUND_ID] = len(
        all_recs
    )  # every tile contains background pixels
    io.check_disk()

    io.write_dataset_metadata(
        SLUG,
        {
            "dataset": SLUG,
            "name": NAME,
            "task_type": "classification",
            "source": "USGS / LBNL (USPVDB)",
            "license": "public domain (U.S. Government work)",
            "provenance": {
                "url": URL,
                "have_locally": False,
                "annotation_method": "manual digitization / position-verification from aerial imagery",
            },
            "sensors_relevant": ["sentinel2", "sentinel1", "landsat"],
            "classes": [
                {
                    "id": BACKGROUND_ID,
                    "name": "background",
                    "description": "Land containing no large-scale (>=1 MW) ground-mounted "
                    "solar PV. True negative: USPVDB is a complete U.S. inventory, so any PV "
                    "in the tile would be in the database.",
                },
                {
                    "id": SOLAR_ID,
                    "name": "solar_pv",
                    "description": "Ground-mounted large-scale (>=1 MW) photovoltaic solar "
                    "facility array-boundary footprint, manually digitized and "
                    "position-verified from aerial imagery (USPVDB).",
                },
            ],
            "nodata_value": io.CLASS_NODATA,
            "num_samples": len(all_recs),
            "class_tile_counts": {
                CLASS_NAMES[SOLAR_ID]: class_tile_counts[SOLAR_ID],
                CLASS_NAMES[BACKGROUND_ID]: class_tile_counts[BACKGROUND_ID],
                "solar_positive_tiles": n_pos,
                "background_negative_tiles": len(neg_recs),
            },
            "available_facilities": len(facs),
            "tile_size": TILE,
            "window_rule": f"clamp(p_year+1, {WINDOW_MIN}, {WINDOW_MAX}); negatives={NEG_YEAR}",
            "notes": (
                "Large-scale solar PV facility array-boundary polygons rasterized to class 1 "
                "(solar_pv) in each tile's own UTM 10 m grid; background=0. USPVDB is a "
                "complete U.S. >=1 MW inventory so within-tile background is a true negative "
                "(complete-coverage negatives, like the well-pad dataset): all facility "
                "polygons intersecting a tile are rasterized. Presence/state classification, "
                "NOT change: p_year is year-granular only (not resolvable to ~1-2 months per "
                "the spec change-timing rule), so the persistent post-construction state is "
                "used with change_time=null and a 1-year window anchored AFTER commissioning "
                f"in the S2 era (window_year=clamp(p_year+1,{WINDOW_MIN},{WINDOW_MAX})); this "
                "keeps pre-2016 facilities (still visible post-2016) while honoring the "
                "post-2016 rule. Tile=64x64 (640 m) centered on the facility centroid; large "
                "farms fill the tile, small farms are a class-1 blob with context; "
                "all_touched rasterization so small facilities are retained. Sampling: up to "
                f"{PER_CLASS} positive solar tiles (of {len(facs)} facilities) + "
                f"{N_NEGATIVES} background-only negative tiles sampled inside the U.S. "
                f"(>= {NEG_MIN_KM} km from any facility). change_time=null. All years used."
            ),
        },
    )
    manifest.write_registry_entry(
        SLUG, "completed", task_type="classification", num_samples=len(all_recs)
    )
    print(f"done: {len(all_recs)} samples", flush=True)


if __name__ == "__main__":
    multiprocessing.set_start_method("forkserver", force=True)
    main()
