"""Process the Global Wind Power Tracker (GWPT) into detection label tiles.

Source: Global Wind Power Tracker, Global Energy Monitor (GEM), February 2026 release
(https://globalenergymonitor.org/projects/global-wind-power-tracker, CC-BY-4.0). A
researcher-curated, facility-level inventory of utility-scale (>=10 MW) onshore and
offshore wind power project *phases* worldwide (33,248 phases in the Feb-2026 release),
each with a point Latitude/Longitude, an operating Status, an Installation Type
(Onshore / Offshore ...), a commissioning Start year, and a Location accuracy flag
(exact / approximate). Distributed as a single .xlsx (sheet "Data"). The download sits
behind an email form on the GEM site that mints a short-lived capability token and returns
a presigned DigitalOcean Spaces URL (see raw/SOURCE.txt for the exact recipe).

Task type: positive-only object DETECTION, encoded as per-pixel classes (spec section 4).
A GWPT record is a single point marking a wind farm; there is no dataset-provided
background/negative class, so we use the tunable detection encoding: a 1 px positive at the
point, a 10 px nodata buffer ring (location coordinates are only project-level and often
"approximate", so a thick ignore ring avoids penalizing near-misses), and background (0)
filling the rest of a 32x32 UTM 10 m context tile, plus dedicated background-only negative
tiles drawn far from any tracked wind farm.

Class scheme (spec section 5 "multi-target / unified scheme"): the manifest lists one class
"wind farm (onshore/offshore)", but onshore and offshore farms look very different at
10-30 m (turbines + pads/access roads on land vs. turbine monopiles standing in open water),
so we split into two observable positive classes and keep background as class 0:
    0 = background, 1 = onshore_wind_farm, 2 = offshore_wind_farm, 255 = nodata/ignore.
Utility-scale wind farms (many large turbines with cleared pads and access roads spread over
hundreds of metres) are resolvable at 10 m; the DeepOWT precedent detects even individual
offshore turbines at Sentinel-1 10 m.

Only *operating* phases are used as positives (they are physically built and visible);
construction / pre-construction / announced / cancelled / shelved / retired / mothballed
phases are excluded. Phases with Installation Type "Unknown" (cannot assign onshore/offshore)
are excluded as positives.

Time / change handling (spec section 5). A built wind farm is a PERSISTENT structure, not a
dated change event: once operating it stays visible for years, and GWPT only resolves the
Start year to a calendar YEAR (coarser than the ~1-2 month change-timing requirement), so we
do NOT emit dated change labels (change_time = null). Each positive is given a 1-year time
window sampled (seeded) from the years the farm is both operating and inside the Sentinel era:
[max(start_year, 2016), min(2025, retired_year - 1)] (start_year missing -> assume a
pre-existing persistent farm, [2016, 2025]). Phases whose first operating year is after 2025
(no full Sentinel year yet) are skipped. Other operating farms that fall inside a tile are
also encoded by their class in that same year (present -> onshore/offshore positive;
not-yet-built -> left background; unknown installation -> nodata).

Sampling: up to 1000 tiles for onshore (stratified across years for temporal diversity), all
operating offshore phases kept (rare class; spec section 5 says do not drop rare classes),
plus up to 1000 background-only negative tiles. Well under the 25k per-dataset cap.

Run (idempotent; skips already-written tiles):
  python3 -m olmoearth_pretrain.open_set_segmentation_data.datasets.global_wind_power_tracker
"""

import argparse
import math
import multiprocessing
import random
from collections import Counter
from typing import Any

import numpy as np
import tqdm
from rslearn.utils.mp import star_imap_unordered
from scipy.spatial import cKDTree

from olmoearth_pretrain.open_set_segmentation_data import io, manifest
from olmoearth_pretrain.open_set_segmentation_data.sampling import (
    balance_by_class,
    encode_detection_tile,
)

SLUG = "global_wind_power_tracker"
NAME = "Global Wind Power Tracker"
URL = "https://globalenergymonitor.org/projects/global-wind-power-tracker"
XLSX_FILE = "Global-Wind-Power-Tracker-February-2026.xlsx"
DATA_SHEET = "Data"

# Class scheme (background + two observable positive classes).
CID_BACKGROUND = 0
CID_ONSHORE = 1
CID_OFFSHORE = 2
POSITIVE_CIDS = (CID_ONSHORE, CID_OFFSHORE)
CLASSES = [
    {
        "id": CID_BACKGROUND,
        "name": "background",
        "description": "No tracked utility-scale wind farm present (land or open water "
        "away from any GWPT phase).",
    },
    {
        "id": CID_ONSHORE,
        "name": "onshore_wind_farm",
        "description": "Operating utility-scale (>=10 MW) onshore wind power facility: "
        "multiple large turbines with cleared pads and access roads on land "
        "(GWPT Installation Type 'Onshore').",
    },
    {
        "id": CID_OFFSHORE,
        "name": "offshore_wind_farm",
        "description": "Operating utility-scale (>=10 MW) offshore wind power facility: "
        "turbines mounted (fixed-bottom or floating) in the sea "
        "(GWPT Installation Type 'Offshore ...').",
    },
]

# Sentinel-era window bounds. min_year: first year of usable S2/S1 imagery. max_year: last
# full calendar year we assume imagery is available and the farm still standing.
MIN_YEAR = 2016
MAX_YEAR = 2025

PER_CLASS = 1000
N_NEGATIVES = 1000
SEED = 42

# Detection encoding parameters (spec section 4). 32x32 = 320 m context tile at 10 m; a
# single-pixel positive at the (project-level, often approximate) point ringed by a 10 px
# nodata buffer (21x21 ignore) so near-misses are not penalized; rest background.
DET_TILE = 32
DET_POS_SIZE = 1
DET_BUFFER = 10

# Negative tiles must sit far from any tracked farm. 0.02 deg (~2.2 km) >> 320 m tile.
NEG_MIN_DIST_DEG = 0.02
# Neighbor search radius for in-tile farms (deg); precise filter is by tile pixel bounds.
# 320 m tile; ~0.01 deg (~1.1 km) covers it at all latitudes.
NEIGHBOR_RADIUS_DEG = 0.01


def _inst_class(installation: Any) -> int | None:
    """Map GWPT Installation Type to a positive class id, or None if unusable."""
    if installation is None:
        return None
    s = str(installation).strip().lower()
    if s.startswith("offshore"):
        return CID_OFFSHORE
    if s == "onshore":
        return CID_ONSHORE
    return None  # "Unknown"


def _to_year(value: Any) -> int | None:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _load_farms() -> list[dict[str, Any]]:
    """Read the GWPT xlsx Data sheet into flat farm records."""
    import openpyxl

    path = io.raw_dir(SLUG) / XLSX_FILE
    wb = openpyxl.load_workbook(path.path, read_only=True, data_only=True)
    ws = wb[DATA_SHEET]
    rows = ws.iter_rows(values_only=True)
    hdr = list(next(rows))
    idx = {h: i for i, h in enumerate(hdr)}
    farms: list[dict[str, Any]] = []
    for r in rows:
        if all(x is None for x in r):
            continue
        try:
            lon = float(r[idx["Longitude"]])
            lat = float(r[idx["Latitude"]])
        except (TypeError, ValueError):
            continue
        if not (-180 <= lon <= 180 and -80 <= lat <= 84):
            continue  # outside UTM validity / bad coord
        farms.append(
            {
                "lon": lon,
                "lat": lat,
                "status": str(r[idx["Status"]]) if r[idx["Status"]] else "",
                "cls": _inst_class(r[idx["Installation Type"]]),
                "start_year": _to_year(r[idx["Start year"]]),
                "retired_year": _to_year(r[idx["Retired year"]]),
                "accuracy": r[idx["Location accuracy"]],
                "phase_id": r[idx["GEM phase ID"]],
            }
        )
    wb.close()
    return farms


def _presence_range(farm: dict[str, Any]) -> tuple[int, int] | None:
    """Inclusive [lo, hi] of full calendar years the farm is operating & in Sentinel era."""
    sy = farm["start_year"]
    lo = MIN_YEAR if sy is None else max(sy, MIN_YEAR)
    hi = MAX_YEAR
    if farm["retired_year"] is not None:
        hi = min(hi, farm["retired_year"] - 1)
    if lo > hi:
        return None
    return (lo, hi)


def _build_candidates(
    farms: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], cKDTree, cKDTree, list[dict[str, Any]]]:
    """Return (positive_candidates, operating_tree, all_tree, operating_farms).

    One positive candidate per operating on/offshore farm with a valid presence window; the
    label year is sampled (seeded) from that window for temporal diversity. Also returns
    KD-trees over operating-farm coords (in-tile neighbor lookup) and all-farm coords
    (negative-distance filtering).
    """
    rng = random.Random(SEED)
    pos: list[dict[str, Any]] = []
    operating = [f for f in farms if f["status"] == "operating"]
    for i, f in enumerate(farms):
        if f["status"] != "operating" or f["cls"] is None:
            continue
        rng_range = _presence_range(f)
        if rng_range is None:
            continue
        lo, hi = rng_range
        year = rng.randint(lo, hi)
        pos.append(
            {
                "kind": "pos",
                "class": f["cls"],
                "year": year,
                "lon": f["lon"],
                "lat": f["lat"],
                "source_id": f"gwpt/{f['phase_id']}",
            }
        )
    op_tree = cKDTree(np.array([[f["lon"], f["lat"]] for f in operating], dtype=float))
    all_tree = cKDTree(np.array([[f["lon"], f["lat"]] for f in farms], dtype=float))
    return pos, op_tree, all_tree, operating


def _build_negatives(
    farms: list[dict[str, Any]], all_tree: cKDTree, n: int
) -> list[dict[str, Any]]:
    """Generate background-only tiles far from any tracked farm.

    Candidates are made by offsetting random operating farms by a random large bearing so
    they land in comparable terrain/regions, then kept only if the nearest tracked farm is
    > NEG_MIN_DIST_DEG away (guarantees an all-background tile). A seeded year in the
    Sentinel era is assigned to each.
    """
    rng = random.Random(SEED + 1)
    operating = [f for f in farms if f["status"] == "operating"]
    out: list[dict[str, Any]] = []
    attempts = 0
    while len(out) < n and attempts < n * 200:
        attempts += 1
        base = rng.choice(operating)
        dist = rng.uniform(0.15, 0.8) * rng.choice([-1, 1])
        dist2 = rng.uniform(0.15, 0.8) * rng.choice([-1, 1])
        lon = base["lon"] + dist
        lat = base["lat"] + dist2
        if not (-180 <= lon <= 180 and -80 <= lat <= 84):
            continue
        d, _ = all_tree.query([lon, lat], k=1)
        if d <= NEG_MIN_DIST_DEG:
            continue
        out.append(
            {
                "kind": "neg",
                "class": CID_BACKGROUND,
                "year": rng.randint(MIN_YEAR, MAX_YEAR),
                "lon": lon,
                "lat": lat,
                "source_id": "gwpt/negative",
            }
        )
    return out


def _resolve_neighbors(
    rec: dict[str, Any], operating: list[dict[str, Any]], op_tree: cKDTree
) -> None:
    """Attach in-tile operating-farm neighbors present in rec['year'] to rec['neighbors'].

    Each neighbor -> (lon, lat, cid): its class (1/2) if present that year and installation
    known; 255 (ignore) if present but installation unknown. Not-yet-built / retired
    neighbors are dropped (correctly left as background). The center point is excluded.
    """
    year = rec["year"]
    idxs = op_tree.query_ball_point([rec["lon"], rec["lat"]], r=NEIGHBOR_RADIUS_DEG)
    out: list[tuple[float, float, int]] = []
    for j in idxs:
        q = operating[j]
        if q["lon"] == rec["lon"] and q["lat"] == rec["lat"]:
            continue
        pr = _presence_range(q)
        if pr is None or not (pr[0] <= year <= pr[1]):
            continue  # not built yet / retired in this year -> background
        cid = q["cls"] if q["cls"] is not None else io.CLASS_NODATA
        out.append((q["lon"], q["lat"], cid))
    rec["neighbors"] = out


def _write_tile(rec: dict[str, Any]) -> str:
    sample_id = rec["sample_id"]
    if (io.locations_dir(SLUG) / f"{sample_id}.tif").exists():
        return "skip"
    proj = io.utm_projection_for_lonlat(rec["lon"], rec["lat"])
    _, col, row = io.lonlat_to_utm_pixel(rec["lon"], rec["lat"], proj)
    bounds = io.centered_bounds(col, row, DET_TILE, DET_TILE)
    x_min, y_min = bounds[0], bounds[1]

    positives: list[tuple[int, int, int]] = []
    if rec["kind"] == "pos":
        positives.append((row - y_min, col - x_min, rec["class"]))
    for lon, lat, cid in rec.get("neighbors", []):
        _, c, r = io.lonlat_to_utm_pixel(lon, lat, proj)
        lc, lr = c - x_min, r - y_min
        if 0 <= lc < DET_TILE and 0 <= lr < DET_TILE:
            positives.append((lr, lc, cid))

    arr = encode_detection_tile(
        positives,
        tile_size=DET_TILE,
        positive_size=DET_POS_SIZE,
        buffer_size=DET_BUFFER,
        nodata=io.CLASS_NODATA,
        background=CID_BACKGROUND,
    )
    io.write_label_geotiff(SLUG, sample_id, arr, proj, bounds, nodata=io.CLASS_NODATA)
    io.write_sample_json(
        SLUG,
        sample_id,
        proj,
        bounds,
        io.year_range(rec["year"]),
        change_time=None,
        source_id=rec["source_id"],
        classes_present=sorted(set(np.unique(arr).tolist()) - {io.CLASS_NODATA}),
    )
    return "pos" if rec["kind"] == "pos" else "neg"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=64)
    args = parser.parse_args()

    io.check_disk()
    manifest.write_registry_entry(SLUG, "in_progress")

    raw = io.raw_dir(SLUG)
    raw.mkdir(parents=True, exist_ok=True)
    with (raw / "SOURCE.txt").open("w") as f:
        f.write(
            "Global Wind Power Tracker, Global Energy Monitor (GEM), February 2026 release.\n"
            f"{URL}\nLicense: CC-BY-4.0.\n"
            f"File: {XLSX_FILE} (sheet 'Data', 33,248 wind farm phases).\n\n"
            "Download recipe (GEM email-gated presign flow, no account required):\n"
            "  KEY=sb_publishable_8mQAV8B2HhveNc5T8VGqPQ_1lgsFAvz\n"
            "  1) POST https://auxunjnrktkmeqyoyngm.supabase.co/rest/v1/rpc/mint_submission\n"
            "     headers: apikey:$KEY, authorization:Bearer $KEY, content-type:application/json\n"
            "     body: {name,email,organization,sector,country,use_case,license_text,\n"
            "            requested_slugs:['wind-power-tracker'],request_mode:'slugs',\n"
            "            custom_fields:{},dynamic_params:null,email_optin:false,\n"
            "            form_key:'wind-power-tracker',page_url:<project url>,useragent:'...'}\n"
            "     -> returns {capability_token}\n"
            "  2) POST https://auxunjnrktkmeqyoyngm.supabase.co/functions/v1/presign\n"
            "     header: authorization:Bearer <capability_token>\n"
            "     -> returns {urls:[{url,filename}]}; GET url -> the .xlsx\n"
        )

    farms = _load_farms()
    print(f"loaded {len(farms)} wind farm phases", flush=True)

    pos_cands, op_tree, all_tree, operating = _build_candidates(farms)
    by_class: dict[int, list[dict[str, Any]]] = {c: [] for c in POSITIVE_CIDS}
    for r in pos_cands:
        by_class[r["class"]].append(r)
    print(
        "positive candidates: "
        + ", ".join(f"{c}={len(v)}" for c, v in by_class.items()),
        flush=True,
    )

    selected: list[dict[str, Any]] = []
    # Onshore: balance across years for temporal diversity, cap at PER_CLASS.
    onshore = balance_by_class(
        by_class[CID_ONSHORE],
        "year",
        per_class=math.ceil(PER_CLASS / 10) * 3,
        seed=SEED,
    )[:PER_CLASS]
    selected.extend(onshore)
    print(f"  onshore: selected {len(onshore)}", flush=True)
    # Offshore is rare -> keep all (spec section 5: don't drop rare classes).
    offshore = by_class[CID_OFFSHORE][:PER_CLASS]
    selected.extend(offshore)
    print(f"  offshore: selected {len(offshore)}", flush=True)

    negatives = _build_negatives(farms, all_tree, N_NEGATIVES)
    selected.extend(negatives)
    print(f"  negatives: selected {len(negatives)}", flush=True)

    for r in selected:
        _resolve_neighbors(r, operating, op_tree)
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
    print("write results:", dict(results), flush=True)

    io.check_disk()

    class_counts = {
        "onshore_wind_farm": len(onshore),
        "offshore_wind_farm": len(offshore),
        "background_negative_tiles": len(negatives),
    }
    io.write_dataset_metadata(
        SLUG,
        {
            "dataset": SLUG,
            "name": NAME,
            "task_type": "classification",  # detection encoded as per-pixel classes
            "source": "Global Energy Monitor",
            "license": "CC-BY-4.0",
            "provenance": {
                "url": URL,
                "have_locally": False,
                "annotation_method": "manual/expert curation",
                "file": XLSX_FILE,
                "release": "February 2026",
            },
            "sensors_relevant": ["sentinel2", "sentinel1", "landsat"],
            "classes": CLASSES,
            "nodata_value": io.CLASS_NODATA,
            "detection_encoding": {
                "tile_size": DET_TILE,
                "positive_size": DET_POS_SIZE,
                "buffer_size": DET_BUFFER,
            },
            "num_samples": len(selected),
            "class_counts": class_counts,
            "notes": (
                "Utility-scale (>=10 MW) wind farm DETECTION from Global Energy Monitor's "
                "GWPT point inventory (Feb 2026 release, 33,248 phases). Only 'operating' "
                "phases used as positives; onshore vs offshore kept as two observable classes "
                "(0=background, 1=onshore_wind_farm, 2=offshore_wind_farm, 255=nodata). "
                "Detection encoding: 32x32 UTM 10 m context tile per farm, 1 px positive + "
                "10 px nodata buffer (21x21 ignore), rest background; other in-tile operating "
                "farms encoded by their class in the same year (unknown installation -> 255). "
                "Persistent-structure time model (change_time=null): each farm gets a 1-year "
                "window sampled from [max(start_year,2016), min(2025, retired-1)] "
                "(missing start_year -> [2016,2025]); phases first operating after 2025 skipped. "
                "GWPT resolves commissioning only to a calendar year (coarser than the ~1-2 "
                "month change-timing rule), so NO dated change labels. Location accuracy is a "
                "mix of 'exact'/'approximate' (project-level points); the 10 px buffer absorbs "
                "positional imprecision. Sampling: onshore capped at 1000 (stratified across "
                "years); all operating offshore kept (rare class, ~360; spec section 5 keeps "
                "rare classes); plus 1000 background-only negatives far (>~2 km) from any "
                "tracked farm. Well under the 25k per-dataset cap."
            ),
        },
    )
    manifest.write_registry_entry(
        SLUG, "completed", task_type="classification", num_samples=len(selected)
    )
    print(f"done: {len(selected)} samples", flush=True)


if __name__ == "__main__":
    multiprocessing.set_start_method("forkserver", force=True)
    main()
