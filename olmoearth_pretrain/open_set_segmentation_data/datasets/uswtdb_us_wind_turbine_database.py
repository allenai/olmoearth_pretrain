"""Process the USWTDB (US Wind Turbine Database) into open-set-segmentation tiles.

Source (external, USGS / LBNL / AWEA, public domain -- a U.S. Government work):
  https://energy.usgs.gov/uswtdb/
The U.S. Wind Turbine Database is the authoritative national inventory of onshore and
offshore wind turbines in the United States and its territories, each turbine
position-verified against high-resolution aerial/satellite imagery and updated quarterly.
It is a COMPLETE inventory (every known U.S. turbine), which makes within-tile non-turbine
pixels TRUE negatives -- the same property the USPVDB solar and Stanford well-pad datasets
rely on.

We download only the LABEL points (no imagery -- pretraining supplies imagery) from the
public USGS EERSC PostgREST API as one JSON array (75,727 turbines):
  https://energy.usgs.gov/api/uswtdb/v1/turbines
Each record is one turbine (unique ``case_id``) with WGS84 ``xlong``/``ylat``, project
online year ``p_year`` (year-granular), and turbine attributes: ``t_cap`` (nameplate kW),
``t_hh`` (hub height m), ``t_rd`` (rotor diameter m), ``t_model``/``t_manu``, ``t_offshore``
(0/1), and location/attribute confidence ``t_conf_loc``/``t_conf_atr`` (1-3).

Decisions (spec sections 2-5):
  * label_type points -> OBJECT-DETECTION, positive-only recipe (spec section 4): a turbine
    marks presence; absence is everywhere else. We use the tunable detection encoding
    (``sampling.encode_detection_tile``), identical to the local ``olmoearth_wind_turbine``
    detection dataset: a 64x64 (640 m @ 10 m) context tile centered on the turbine pixel,
    the turbine a 1x1 positive (class 1 = turbine) ringed by a 10 px nodata (255) buffer
    (turbine coordinates are position-verified but not pixel-exact at 10 m), all other pixels
    background (class 0). A single turbine tower/pad is ~1 px at 10 m but is a strong,
    detectable signature (tower shadow, gravel pad, access roads) -> observable at 10-30 m
    from Sentinel-2/Sentinel-1/Landsat.
  * COMPLETE inventory -> every other turbine that falls inside a tile is also marked
    positive (STRtree query over all turbines), so background pixels are TRUE negatives.
    Dense wind farms put several turbines in one 64x64 tile.
  * CHANGE vs PRESENCE (spec section 5 timing rule): ``p_year`` is year-granular only, so the
    installation event is NOT resolvable to ~1-2 months and CANNOT be a change label. We use
    the persistent post-construction STATE (a turbine stays visible for years) as presence
    with change_time=null, anchoring each tile's 1-year window in the Sentinel-2 era AFTER
    commissioning so the turbine is present: window_year = clamp(p_year+1, 2017, 2024). This
    keeps pre-2016 turbines (still standing post-2016) while honoring the post-2016 rule.
  * NEGATIVES (detection exception, spec section 5): background-only tiles sampled inside the
    U.S. (offset from random turbines) at least NEG_MIN_KM from any turbine.

Classes: 0 background, 1 turbine.

Sampling (spec section 5): single foreground class -> up to PER_CLASS (1000) positive turbine
tiles + N_NEGATIVES (1000) background tiles (well under the 25k cap), matching the
turbine/well-pad/solar detection precedent.

Run (idempotent; skips already-written tiles):
  python3 -m olmoearth_pretrain.open_set_segmentation_data.datasets.uswtdb_us_wind_turbine_database
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

from olmoearth_pretrain.open_set_segmentation_data import download, io, manifest
from olmoearth_pretrain.open_set_segmentation_data.sampling import encode_detection_tile

SLUG = "uswtdb_us_wind_turbine_database"
NAME = "USWTDB (US Wind Turbine Database)"
URL = "https://energy.usgs.gov/uswtdb/"
API = "https://energy.usgs.gov/api/uswtdb/v1/turbines"
JSON_NAME = "uswtdb_turbines.json"
UA = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0 Safari/537.36"
    )
}

BACKGROUND_ID = 0
TURBINE_ID = 1
CLASS_NAMES = {BACKGROUND_ID: "background", TURBINE_ID: "turbine"}

# Detection encoding parameters (spec section 4), matching olmoearth_wind_turbine.
DET_TILE = io.MAX_TILE  # 64 px @ 10 m = 640 m
DET_POS_SIZE = 1
DET_BUFFER = 10

PER_CLASS = 1000  # positive turbine tiles (single foreground class, spec section 5)
N_NEGATIVES = 1000  # background-only tiles inside the US away from any turbine
NEG_YEAR = 2022  # static representative window for background negatives (post-2016)
WINDOW_MIN, WINDOW_MAX = 2017, 2024  # S2-era post-commissioning window clamp
NEG_MIN_KM = 3.0  # negatives must be >= this from any turbine
NEG_OFF_KM = (
    15.0,
    60.0,
)  # negative offset distance from a random turbine (stays in-country)
SEED = 42


def _download() -> str:
    raw = io.raw_dir(SLUG)
    raw.mkdir(parents=True, exist_ok=True)
    out = raw / JSON_NAME
    download.download_postgrest_json(
        API,
        out,
        select="case_id,p_name,t_state,p_year,t_cap,t_hh,t_rd,t_model,t_manu,"
        "t_conf_loc,t_conf_atr,t_img_date,t_img_src,t_offshore,xlong,ylat",
        order="case_id",
        page=20000,
        headers=UA,
    )
    with (raw / "SOURCE.txt").open("w") as f:
        f.write(
            "USWTDB (US Wind Turbine Database), USGS / LBNL / AWEA, public domain "
            "(U.S. Government work).\n"
            f"{URL}\n"
            "Label-only download of turbine POINTS from the public USGS EERSC PostgREST "
            f"API:\n{API}\n"
            "One JSON array of turbines (unique case_id) with WGS84 xlong/ylat, p_year "
            "(year-granular commissioning year), t_cap/t_hh/t_rd/t_model/t_manu, t_offshore, "
            "and t_conf_loc/t_conf_atr confidence. Imagery is supplied by pretraining, not "
            "downloaded here.\n"
        )
    return str(out)


def _load_turbines() -> list[dict[str, Any]]:
    """Parse the JSON array into per-turbine records (one row = one turbine)."""
    path = io.raw_dir(SLUG) / JSON_NAME
    rows = json.load(path.open())
    out: list[dict[str, Any]] = []
    for r in rows:
        lon, lat = r.get("xlong"), r.get("ylat")
        if lon is None or lat is None:
            continue
        try:
            lon, lat = float(lon), float(lat)
        except (TypeError, ValueError):
            continue
        if not (-180 <= lon <= 180 and -90 <= lat <= 90):
            continue
        out.append(
            {
                "case_id": r.get("case_id"),
                "lon": lon,
                "lat": lat,
                "p_year": int(r["p_year"]) if r.get("p_year") else None,
                "offshore": bool(r.get("t_offshore")),
                "src": f"uswtdb/case_id={r.get('case_id')}/{r.get('p_name')}",
            }
        )
    return out


def _window_year(p_year: int | None) -> int:
    if not p_year:
        return NEG_YEAR
    return max(WINDOW_MIN, min(WINDOW_MAX, p_year + 1))


def _tile_record(
    tree: shapely.STRtree,
    pts_wgs84: list[Any],
    lon: float,
    lat: float,
) -> dict[str, Any]:
    """Build a 64x64 tile centered on (lon, lat); gather all turbines inside it."""
    proj, col, row = io.lonlat_to_utm_pixel(lon, lat)
    bounds = io.centered_bounds(col, row, DET_TILE, DET_TILE)
    tile_box = shapely.box(bounds[0], bounds[1], bounds[2], bounds[3])
    tile_wgs84 = STGeometry(proj, tile_box, None).to_projection(WGS84_PROJECTION).shp
    hits = tree.query(tile_wgs84)
    turb_lonlat: list[tuple[float, float]] = []
    for i in np.atleast_1d(hits).tolist():
        p = pts_wgs84[i]
        if tile_wgs84.contains(p):
            turb_lonlat.append((float(p.x), float(p.y)))
    return {
        "crs": proj.crs.to_string(),
        "bounds": list(bounds),
        "turbines_lonlat": turb_lonlat,
    }


def _write_sample(rec: dict[str, Any]) -> str:
    sample_id = rec["sample_id"]
    if (io.locations_dir(SLUG) / f"{sample_id}.tif").exists():
        return "skip"
    proj = Projection(CRS.from_string(rec["crs"]), io.RESOLUTION, -io.RESOLUTION)
    bounds = tuple(rec["bounds"])
    x_min, y_min = bounds[0], bounds[1]
    positives: list[tuple[int, int, int]] = []
    for lon, lat in rec["turbines_lonlat"]:
        g = STGeometry(WGS84_PROJECTION, shapely.Point(lon, lat), None).to_projection(
            proj
        )
        lc = int(math.floor(g.shp.x)) - x_min
        lr = int(math.floor(g.shp.y)) - y_min
        if 0 <= lc < DET_TILE and 0 <= lr < DET_TILE:
            positives.append((lr, lc, TURBINE_ID))
    arr = encode_detection_tile(
        positives,
        tile_size=DET_TILE,
        positive_size=DET_POS_SIZE,
        buffer_size=DET_BUFFER,
        nodata=io.CLASS_NODATA,
        background=BACKGROUND_ID,
    )
    io.write_label_geotiff(SLUG, sample_id, arr, proj, bounds, nodata=io.CLASS_NODATA)
    io.write_sample_json(
        SLUG,
        sample_id,
        proj,
        bounds,
        io.year_range(rec["window_year"]),
        change_time=None,
        source_id=rec["src"],
        classes_present=sorted(set(np.unique(arr).tolist()) - {io.CLASS_NODATA}),
    )
    return "pos" if positives else "neg"


def _make_negatives(
    turbs: list[dict[str, Any]], n: int, rng: random.Random
) -> list[tuple[float, float]]:
    """Sample n background points inside the US, >= NEG_MIN_KM from any turbine.

    Each candidate is a random turbine offset by a random 15-60 km vector; since turbines
    sit within the US, an offset that size almost always stays on U.S. land/waters. We then
    reject any candidate within NEG_MIN_KM of a turbine (vectorized haversine).
    """
    tlon = np.array([t["lon"] for t in turbs])
    tlat = np.array([t["lat"] for t in turbs])
    tlon_r = np.radians(tlon)
    tlat_r = np.radians(tlat)
    lon_lo, lon_hi = float(tlon.min()), float(tlon.max())
    lat_lo, lat_hi = float(tlat.min()), float(tlat.max())
    out: list[tuple[float, float]] = []
    attempts = 0
    while len(out) < n and attempts < n * 200:
        attempts += 1
        base = turbs[rng.randrange(len(turbs))]
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
        lo_r, la_r = math.radians(lon), math.radians(lat)
        dphi = tlat_r - la_r
        dlmb = tlon_r - lo_r
        a = (
            np.sin(dphi / 2) ** 2
            + np.cos(la_r) * np.cos(tlat_r) * np.sin(dlmb / 2) ** 2
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

    turbs = _load_turbines()
    print(f"loaded {len(turbs)} turbines", flush=True)
    pts_wgs84 = [shapely.Point(t["lon"], t["lat"]) for t in turbs]
    tree = shapely.STRtree(pts_wgs84)

    rng = random.Random(SEED)

    # Positives: up to PER_CLASS turbines as tile centers; each tile also marks every
    # other turbine falling inside it (complete inventory -> known true negatives).
    order = list(range(len(turbs)))
    rng.shuffle(order)
    sel_pos = order[:PER_CLASS]
    pos_recs: list[dict[str, Any]] = []
    for i in sel_pos:
        t = turbs[i]
        rec = _tile_record(tree, pts_wgs84, t["lon"], t["lat"])
        rec.update(window_year=_window_year(t["p_year"]), src=t["src"], kind="pos")
        pos_recs.append(rec)
    print(f"prepared {len(pos_recs)} positive turbine tiles", flush=True)

    # Negatives: background-only tiles inside the US away from any turbine.
    neg_pts = _make_negatives(turbs, N_NEGATIVES, rng)
    neg_recs: list[dict[str, Any]] = []
    for lon, lat in neg_pts:
        rec = _tile_record(
            tree, pts_wgs84, lon, lat
        )  # normally empty; robust if one clips in
        rec.update(
            window_year=NEG_YEAR, src=f"background/{lon:.4f},{lat:.4f}", kind="neg"
        )
        neg_recs.append(rec)
    print(f"prepared {len(neg_recs)} background negative tiles", flush=True)

    all_recs = pos_recs + neg_recs
    all_recs.sort(key=lambda r: (r["crs"], r["bounds"][0], r["bounds"][1]))
    for idx, r in enumerate(all_recs):
        r["sample_id"] = f"{idx:06d}"

    io.check_disk()
    results: Counter = Counter()
    with multiprocessing.Pool(args.workers) as p:
        for res in tqdm.tqdm(
            star_imap_unordered(p, _write_sample, [dict(rec=r) for r in all_recs]),
            total=len(all_recs),
        ):
            results[res] += 1
    print("write results:", dict(results), flush=True)
    io.check_disk()

    n_turbine_points = sum(len(r["turbines_lonlat"]) for r in pos_recs)
    n_offshore = sum(1 for t in turbs if t["offshore"])
    io.write_dataset_metadata(
        SLUG,
        {
            "dataset": SLUG,
            "name": NAME,
            "task_type": "classification",  # detection encoded as per-pixel classes
            "source": "USGS / LBNL / AWEA (USWTDB)",
            "license": "public domain (U.S. Government work)",
            "provenance": {
                "url": URL,
                "have_locally": False,
                "annotation_method": "manual position verification against aerial imagery",
            },
            "sensors_relevant": ["sentinel2", "sentinel1", "landsat"],
            "classes": [
                {
                    "id": BACKGROUND_ID,
                    "name": "background",
                    "description": "Land/water containing no wind turbine. True negative: "
                    "USWTDB is a complete U.S. inventory, so any turbine in the tile would "
                    "be in the database.",
                },
                {
                    "id": TURBINE_ID,
                    "name": "turbine",
                    "description": "Utility-scale wind turbine (onshore or offshore), "
                    "position-verified against high-resolution aerial imagery (USWTDB). "
                    "Tower/pad footprint is ~1 px at 10 m but a strong signature (shadow, "
                    "gravel pad, access roads).",
                },
            ],
            "nodata_value": io.CLASS_NODATA,
            "detection_encoding": {
                "tile_size": DET_TILE,
                "positive_size": DET_POS_SIZE,
                "buffer_size": DET_BUFFER,
            },
            "num_samples": len(all_recs),
            "class_tile_counts": {
                "turbine_positive_tiles": len(pos_recs),
                "background_negative_tiles": len(neg_recs),
                "turbine_points_in_positives": n_turbine_points,
            },
            "available": {
                "total_turbines": len(turbs),
                "offshore_turbines": n_offshore,
            },
            "tile_size": DET_TILE,
            "window_rule": f"clamp(p_year+1, {WINDOW_MIN}, {WINDOW_MAX}); negatives={NEG_YEAR}",
            "notes": (
                "USWTDB national wind-turbine POINT inventory (75,727 turbines) encoded via "
                "the object-detection recipe (spec section 4): 64x64 UTM 10 m context tile "
                "per turbine, 1 px positive (id 1 = turbine) + 10 px nodata (255) buffer ring "
                "(coords position-verified but not pixel-exact at 10 m), rest background "
                "(id 0). USWTDB is a complete U.S. inventory so within-tile background is a "
                "TRUE negative and every turbine falling inside a tile is marked (dense wind "
                "farms give several turbines per tile). Presence/state, NOT change: p_year is "
                "year-granular only (not resolvable to ~1-2 months per the spec change-timing "
                "rule), so the persistent post-construction state is used with change_time=null "
                "and a 1-year window anchored AFTER commissioning in the S2 era "
                f"(window_year=clamp(p_year+1,{WINDOW_MIN},{WINDOW_MAX})); this keeps pre-2016 "
                "turbines (still standing post-2016) while honoring the post-2016 rule. "
                "Negatives: background-only tiles sampled inside the U.S. "
                f">= {NEG_MIN_KM} km from any turbine (window_year={NEG_YEAR}). Onshore and "
                f"offshore ({n_offshore}) turbines both used. Single foreground class -> up to "
                f"{PER_CLASS} positive turbine tiles + {N_NEGATIVES} background negative tiles."
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
