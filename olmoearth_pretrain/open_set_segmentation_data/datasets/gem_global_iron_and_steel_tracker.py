"""GEM Global Iron and Steel Tracker (GIST) -> steel/iron-plant presence detection tiles.

Source (external, Global Energy Monitor, CC-BY-4.0):
  https://globalenergymonitor.org/projects/global-iron-and-steel-tracker/download-data/
The Global Iron and Steel Tracker (GIST) is the authoritative asset-level inventory of the
world's crude iron and steel plants. It includes *every* plant currently operating with a
capacity of >= 500,000 tonnes/yr (ttpa) of crude iron or steel, plus plants proposed / under
construction since 2017 or retired / mothballed since 2020. Each plant carries WGS84
coordinates (with an exact/approximate accuracy flag), a commissioning "Start date" (year),
the furnace mix ("Main production equipment": BF, BOF, EAF, DRI, IF, ...), capacities, and a
per-unit operating status. It is expert-compiled from company filings, government data,
satellite imagery and news, and linked to a GEM.wiki page per plant.

ACCESS (spec section 8 triage): GIST is distributed behind a lightweight web *download form*
(name/email/use-case), NOT an authenticated credential gate. The download page ships a
**public** Supabase "publishable" key and an unverified mint_submission -> presign -> object
flow (see ``download.download_gem_tracker``), so the CC-BY-4.0 file is automatable without any
credential from ``.env``. ACCEPTED. We pull only the plant-level LABEL table (no imagery --
pretraining supplies imagery); we request the ``iron-steel-plant-tracker`` slug (one row per
plant with coordinates) and the ``Plant capacities and status`` sheet for status.

Decisions (spec sections 2-5):
  * label_type points -> OBJECT-DETECTION, positive-only recipe (spec section 4): a plant
    marks presence; absence is everywhere else. Tunable detection encoding
    (``sampling.encode_detection_tile``): a 64x64 (640 m @ 10 m) context tile centered on the
    plant pixel, the plant a POSITIVE_SIZE (21 px ~= 210 m) square of class 1 (steel/iron
    plant) sized to the discernible core of a large complex, ringed by a BUFFER (15 px ~=
    150 m) nodata (255) band, all other pixels background (class 0). Buffer >= 10 px per spec
    (the coordinate is a point, not a footprint), and generous because these are very large,
    clearly-discernible complexes.
  * INCLUSION: only plants with at least one unit that is operating / operating
    pre-retirement / mothballed / mothballed pre-retirement AND an 'exact' (satellite-
    confirmed) coordinate -> physically standing, visible structures precisely located
    (986 of 1293 plants). Dropped: announced / cancelled (not built), construction-only
    (not yet a plant), retired-only (may be demolished and cannot be time-anchored reliably),
    and 'approximate' coordinates (city/subnational estimates, potentially km off -> beyond
    the detection buffer). Noted in the summary.
  * COMPLETENESS -> within-tile non-plant pixels are approximately TRUE negatives: GIST is a
    complete inventory of >= 500 ktpa plants, so any *large* steel plant in a tile is in the
    database and is marked positive (STRtree query over all included plants; industrial
    clusters put several plants in one tile). Sub-500 ktpa mini-mills are not tracked (minor
    false-negative risk; noted).
  * CHANGE vs PRESENCE (spec section 5 timing rule): "Start date" is year-granular only, so the
    build event is NOT resolvable to ~1-2 months -> NOT a change label. We use the persistent
    post-construction STATE (a steel mill stays visible for decades) as presence with
    change_time=null, anchoring each tile's 1-year window in the Sentinel era at/after the
    start year, spread deterministically over [lo, 2025] for imagery diversity where
    lo = clamp(start, 2017, 2025) (start<2017 or unknown -> lo=2017). This keeps pre-2016
    plants (still standing post-2016) while honoring the post-2016 rule.
  * NEGATIVES (detection exception, spec section 5): background-only tiles sampled globally as
    15-80 km offsets from random plants (staying near industrial/populated land) and kept
    >= NEG_MIN_KM from any plant.

Classes: 0 background, 1 steel/iron plant.

Sampling (spec section 5): single foreground class -> all 986 present exact-coordinate plants
as positive tiles (~= the "up to 1000/class" target; the complete operating inventory) +
N_NEGATIVES background tiles. Well under the 25k cap.

Run (idempotent; skips already-written tiles):
  python3 -m olmoearth_pretrain.open_set_segmentation_data.datasets.gem_global_iron_and_steel_tracker
"""

import argparse
import hashlib
import math
import multiprocessing
import random
from collections import Counter
from typing import Any

import numpy as np
import shapely
from rasterio.crs import CRS
from rslearn.const import WGS84_PROJECTION
from rslearn.utils.geometry import Projection, STGeometry
from rslearn.utils.mp import star_imap_unordered

from olmoearth_pretrain.open_set_segmentation_data import download, io, manifest
from olmoearth_pretrain.open_set_segmentation_data.sampling import encode_detection_tile

SLUG = "gem_global_iron_and_steel_tracker"
NAME = "GEM Global Iron and Steel Tracker"
URL = "https://globalenergymonitor.org/projects/global-iron-and-steel-tracker/download-data/"
GEM_SLUG = "iron-steel-plant-tracker"
XLSX_NAME = "Plant-level_data_Global_Iron_and_Steel_Tracker_June_2026_V1.xlsx"
PLANT_SHEET = "Plant data"
STATUS_SHEET = "Plant capacities and status"

BACKGROUND_ID = 0
PLANT_ID = 1
CLASS_NAMES = {BACKGROUND_ID: "background", PLANT_ID: "steel/iron plant"}

# Statuses that mean the plant physically stands and is visible in imagery.
PRESENT_STATUSES = {
    "operating",
    "operating pre-retirement",
    "mothballed",
    "mothballed pre-retirement",
}

# Detection encoding parameters (spec section 4). Large complexes -> a sizeable positive
# core (~210 m) with a generous nodata buffer (~150 m).
DET_TILE = io.MAX_TILE  # 64 px @ 10 m = 640 m
DET_POS_SIZE = 21  # ~210 m positive square (discernible core of a large plant)
DET_BUFFER = 15  # ~150 m nodata ring (>= 10 px; absorbs point/footprint mismatch)

N_NEGATIVES = 1000  # background-only tiles away from any plant
NEG_YEAR = 2021  # representative static window for negatives (post-2016)
WINDOW_MIN, WINDOW_MAX = 2017, 2025  # S2-era window clamp
NEG_MIN_KM = 5.0  # negatives must be >= this from any plant
NEG_OFF_KM = (15.0, 80.0)  # negative offset distance from a random plant
SEED = 42


def _download(contact: dict[str, str]) -> str:
    raw = io.raw_dir(SLUG)
    raw.mkdir(parents=True, exist_ok=True)
    out = raw / XLSX_NAME
    if not out.exists():
        io.check_disk()
        print("downloading GIST plant-level data via GEM download form ...", flush=True)
        paths = download.download_gem_tracker([GEM_SLUG], raw, contact)
        # The presign may return a newer-dated filename; normalize to the one we read.
        if not out.exists() and paths:
            paths[0].rename(out)
    with (raw / "SOURCE.txt").open("w") as f:
        f.write(
            "GEM Global Iron and Steel Tracker (GIST), Global Energy Monitor, "
            "CC-BY-4.0.\n"
            f"{URL}\n"
            "Recommended citation: 'Global Iron and Steel Tracker, Global Energy Monitor, "
            "June 2026 (V1) release.'\n"
            f"Label-only download of the plant-level table (GEM download slug "
            f"'{GEM_SLUG}') via the public Supabase mint_submission->presign flow "
            "(no credential; unverified web download form). One row per plant with WGS84 "
            "'Coordinates' (lat, lon), 'Coordinate accuracy', 'Start date' (year), "
            "'Main production equipment' (BF/BOF/EAF/DRI/IF furnaces), and a per-unit "
            "operating status ('Plant capacities and status' sheet). Imagery is supplied by "
            "pretraining, not downloaded here.\n"
        )
    return out.path


def _parse_coord(val: Any) -> tuple[float, float] | None:
    if not isinstance(val, str):
        return None
    try:
        a, b = val.split(",")
        lat, lon = float(a), float(b)
    except (ValueError, TypeError):
        return None
    if not (-90 <= lat <= 90 and -180 <= lon <= 180):
        return None
    return lon, lat


def _load_plants(contact: dict[str, str]) -> list[dict[str, Any]]:
    """Read the GIST plant table -> per-plant records for physically-present plants."""
    import pandas as pd

    path = _download(contact)
    df = pd.read_excel(path, sheet_name=PLANT_SHEET)
    cs = pd.read_excel(path, sheet_name=STATUS_SHEET)

    status_sets = cs.groupby("GEM plant ID")["Status"].apply(
        lambda s: set(str(x) for x in s.dropna())
    )

    out: list[dict[str, Any]] = []
    for _, r in df.iterrows():
        pid = r["GEM plant ID"]
        sset = status_sets.get(pid, set())
        if not (sset & PRESENT_STATUSES):
            continue
        # Keep only 'exact' coordinates: GEM 'approximate' points are city/subnational/
        # country-level estimates that can be many km off the plant -- beyond the detection
        # buffer, they would place the positive box on empty land. 'exact' means GEM
        # confirmed the centroid against satellite imagery, so it lands on the complex.
        if str(r.get("Coordinate accuracy")).strip().lower() != "exact":
            continue
        lonlat = _parse_coord(r.get("Coordinates"))
        if lonlat is None:
            continue
        lon, lat = lonlat
        start = pd.to_numeric(r.get("Start date"), errors="coerce")
        start_year = int(start) if not (start is None or math.isnan(start)) else None
        equip = r.get("Main production equipment")
        equip = str(equip) if isinstance(equip, str) else ""
        out.append(
            {
                "pid": str(pid),
                "lon": lon,
                "lat": lat,
                "start_year": start_year,
                "accuracy": str(r.get("Coordinate accuracy")),
                "equipment": equip,
                "country": str(r.get("Country/area")),
                "name": str(r.get("Plant name (English)")),
            }
        )
    return out


def _window_year(start_year: int | None, pid: str) -> int:
    """Presence-window year: >= start (if known) and in the Sentinel era, spread for
    imagery diversity via a deterministic per-plant hash.
    """
    lo = WINDOW_MIN
    if start_year is not None:
        lo = max(WINDOW_MIN, min(WINDOW_MAX, start_year))
    span = WINDOW_MAX - lo + 1
    h = int(hashlib.md5(pid.encode()).hexdigest(), 16)
    return lo + (h % span)


def _tile_record(
    tree: shapely.STRtree,
    pts_wgs84: list[Any],
    plants: list[dict[str, Any]],
    lon: float,
    lat: float,
) -> dict[str, Any]:
    """Build a 64x64 tile centered on (lon, lat); gather all present plants inside it."""
    proj, col, row = io.lonlat_to_utm_pixel(lon, lat)
    bounds = io.centered_bounds(col, row, DET_TILE, DET_TILE)
    tile_box = shapely.box(bounds[0], bounds[1], bounds[2], bounds[3])
    tile_wgs84 = STGeometry(proj, tile_box, None).to_projection(WGS84_PROJECTION).shp
    hits = tree.query(tile_wgs84)
    plant_lonlat: list[tuple[float, float]] = []
    for i in np.atleast_1d(hits).tolist():
        p = pts_wgs84[i]
        if tile_wgs84.contains(p):
            plant_lonlat.append((float(p.x), float(p.y)))
    return {
        "crs": proj.crs.to_string(),
        "bounds": list(bounds),
        "plants_lonlat": plant_lonlat,
    }


def _write_sample(rec: dict[str, Any]) -> str:
    sample_id = rec["sample_id"]
    if (io.locations_dir(SLUG) / f"{sample_id}.tif").exists():
        return "skip"
    proj = Projection(CRS.from_string(rec["crs"]), io.RESOLUTION, -io.RESOLUTION)
    bounds = tuple(rec["bounds"])
    x_min, y_min = bounds[0], bounds[1]
    positives: list[tuple[int, int, int]] = []
    for lon, lat in rec["plants_lonlat"]:
        g = STGeometry(WGS84_PROJECTION, shapely.Point(lon, lat), None).to_projection(
            proj
        )
        lc = int(math.floor(g.shp.x)) - x_min
        lr = int(math.floor(g.shp.y)) - y_min
        if 0 <= lc < DET_TILE and 0 <= lr < DET_TILE:
            positives.append((lr, lc, PLANT_ID))
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
    plants: list[dict[str, Any]], n: int, rng: random.Random
) -> list[tuple[float, float]]:
    """Sample n background points as 15-80 km offsets from random plants, >= NEG_MIN_KM
    from any plant (vectorized haversine). Offsets that size stay near industrial/populated
    land, giving realistic hard negatives.
    """
    plon = np.array([p["lon"] for p in plants])
    plat = np.array([p["lat"] for p in plants])
    plon_r = np.radians(plon)
    plat_r = np.radians(plat)
    out: list[tuple[float, float]] = []
    attempts = 0
    while len(out) < n and attempts < n * 300:
        attempts += 1
        base = plants[rng.randrange(len(plants))]
        d_km = rng.uniform(*NEG_OFF_KM)
        bearing = rng.uniform(0, 2 * math.pi)
        dlat = (d_km * math.cos(bearing)) / 111.0
        dlon = (d_km * math.sin(bearing)) / (
            111.0 * math.cos(math.radians(base["lat"]))
        )
        lon = base["lon"] + dlon
        lat = base["lat"] + dlat
        if not (-180 <= lon <= 180 and -85 <= lat <= 85):
            continue
        lo_r, la_r = math.radians(lon), math.radians(lat)
        dphi = plat_r - la_r
        dlmb = plon_r - lo_r
        a = (
            np.sin(dphi / 2) ** 2
            + np.cos(la_r) * np.cos(plat_r) * np.sin(dlmb / 2) ** 2
        )
        dist_km = 6371.0 * 2 * np.arcsin(np.sqrt(a))
        if dist_km.min() >= NEG_MIN_KM:
            out.append((lon, lat))
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=64)
    parser.add_argument(
        "--contact-name",
        required=True,
        help="Your name, submitted to the GEM download form.",
    )
    parser.add_argument(
        "--contact-email",
        required=True,
        help="Your email, submitted to the GEM download form.",
    )
    parser.add_argument(
        "--contact-organization",
        default="",
        help="Your organization, submitted to the GEM download form.",
    )
    args = parser.parse_args()

    contact = {
        "name": args.contact_name,
        "email": args.contact_email,
        "organization": args.contact_organization,
    }

    io.check_disk()
    manifest.write_registry_entry(SLUG, "in_progress")

    plants = _load_plants(contact)
    print(f"loaded {len(plants)} present (operating/mothballed) plants", flush=True)
    io.check_disk()

    pts_wgs84 = [shapely.Point(p["lon"], p["lat"]) for p in plants]
    tree = shapely.STRtree(pts_wgs84)
    rng = random.Random(SEED)

    # Positives: one tile per present plant; each tile also marks every other present plant
    # falling inside it (complete >= 500 ktpa inventory -> approximately true negatives).
    pos_recs: list[dict[str, Any]] = []
    for p in plants:
        rec = _tile_record(tree, pts_wgs84, plants, p["lon"], p["lat"])
        rec.update(
            window_year=_window_year(p["start_year"], p["pid"]),
            src=f"gem_plant/{p['pid']}/{p['name']}/{p['equipment']}/start={p['start_year']}",
            kind="pos",
        )
        pos_recs.append(rec)
    print(f"prepared {len(pos_recs)} positive plant tiles", flush=True)

    # Negatives: background-only tiles away from any plant.
    neg_pts = _make_negatives(plants, N_NEGATIVES, rng)
    neg_recs: list[dict[str, Any]] = []
    for lon, lat in neg_pts:
        rec = _tile_record(tree, pts_wgs84, plants, lon, lat)
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
    import tqdm

    with multiprocessing.Pool(args.workers) as pool:
        for res in tqdm.tqdm(
            star_imap_unordered(pool, _write_sample, [dict(rec=r) for r in all_recs]),
            total=len(all_recs),
        ):
            results[res] += 1
    print("write results:", dict(results), flush=True)
    io.check_disk()

    n_plant_points = sum(len(r["plants_lonlat"]) for r in pos_recs)
    n_exact = sum(1 for p in plants if p["accuracy"] == "exact")
    io.write_dataset_metadata(
        SLUG,
        {
            "dataset": SLUG,
            "name": NAME,
            "task_type": "classification",  # detection encoded as per-pixel classes
            "source": "Global Energy Monitor (Global Iron and Steel Tracker)",
            "license": "CC-BY-4.0",
            "provenance": {
                "url": URL,
                "have_locally": False,
                "annotation_method": (
                    "authoritative/expert asset-level inventory (company filings, "
                    "government data, satellite imagery, news), positions with an "
                    "exact/approximate accuracy flag"
                ),
                "citation": (
                    "Global Iron and Steel Tracker, Global Energy Monitor, June 2026 "
                    "(V1) release."
                ),
            },
            "sensors_relevant": ["sentinel2", "sentinel1", "landsat"],
            "classes": [
                {
                    "id": BACKGROUND_ID,
                    "name": "background",
                    "description": (
                        "Land/water containing no large (>= 500 ktpa) iron/steel plant. "
                        "Approximately true negative: GIST is a complete inventory of "
                        ">= 500 ktpa plants; sub-threshold mini-mills are not tracked."
                    ),
                },
                {
                    "id": PLANT_ID,
                    "name": "steel/iron plant",
                    "description": (
                        "Crude iron/steel plant with >= 500,000 t/yr capacity (BF, BOF, "
                        "EAF, DRI, IF and related furnaces) -- a very large, clearly "
                        "discernible industrial complex (blast furnaces, stoves, sinter/"
                        "coking plants, rolling mills, stockyards). Position marks the "
                        "plant; the labeled square is the discernible core."
                    ),
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
                "plant_positive_tiles": len(pos_recs),
                "background_negative_tiles": len(neg_recs),
                "plant_points_in_positives": n_plant_points,
            },
            "available": {
                "plants_total_in_source": 1293,
                "plants_present_exact_included": len(plants),
                "note": (
                    "included = operating/mothballed unit AND 'exact' coordinate; "
                    f"all {n_exact} included are exact by construction"
                ),
            },
            "tile_size": DET_TILE,
            "window_rule": (
                f"presence/state; change_time=null; window in [clamp(start,{WINDOW_MIN},"
                f"{WINDOW_MAX}), {WINDOW_MAX}] spread by plant hash; negatives={NEG_YEAR}"
            ),
            "notes": (
                "GEM Global Iron and Steel Tracker plant POINTS encoded via the "
                "object-detection recipe (spec section 4): 64x64 UTM 10 m context tile per "
                f"plant, {DET_POS_SIZE} px positive square (id 1 = steel/iron plant, ~210 m "
                f"discernible core) + {DET_BUFFER} px nodata (255) buffer ring (~150 m; "
                ">= 10 px per spec, generous because the coordinate is a point not a "
                "footprint), rest background (id 0). INCLUSION: only plants with "
                "a unit that is operating / operating pre-retirement / mothballed / mothballed "
                "pre-retirement AND an 'exact' (satellite-confirmed) coordinate (986 of 1293 "
                "physically-standing, precisely-located plants); dropped "
                "announced/cancelled (not built), construction-only (not yet a plant), "
                "retired-only (may be demolished / cannot time-anchor), and 'approximate' "
                "coordinates (city/subnational estimates, potentially km off). COMPLETENESS: GIST is "
                "a complete inventory of >= 500 ktpa plants so within-tile background is "
                "approximately a TRUE negative and every included plant falling inside a tile "
                "is marked (industrial clusters give several plants per tile); sub-500 ktpa "
                "mini-mills are untracked (minor false-negative risk). Presence/state, NOT "
                "change: 'Start date' is year-granular only (not resolvable to ~1-2 months per "
                "the spec change-timing rule), so the persistent post-construction state is "
                "used with change_time=null and a 1-year window at/after the start year in the "
                f"S2 era (window in [clamp(start,{WINDOW_MIN},{WINDOW_MAX}),{WINDOW_MAX}] spread "
                "deterministically per plant for imagery diversity); pre-2016 plants still "
                "standing post-2016 are kept while honoring the post-2016 rule. Negatives: "
                f"background-only tiles sampled globally as {int(NEG_OFF_KM[0])}-"
                f"{int(NEG_OFF_KM[1])} km offsets from random plants, >= {NEG_MIN_KM} km from "
                f"any plant (window_year={NEG_YEAR}). CAVEAT: many steel complexes are larger "
                "than the labeled positive core, so pixels in the background ring of a "
                "positive tile can still be plant; the nodata buffer and dedicated negatives "
                "mitigate this. Single foreground class -> all present plants as positives + "
                f"{N_NEGATIVES} negatives."
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
