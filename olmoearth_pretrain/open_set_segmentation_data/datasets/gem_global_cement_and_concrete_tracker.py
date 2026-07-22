"""GEM Global Cement and Concrete Tracker (GCCT) -> open-set-segmentation presence points.

Source: Global Cement and Concrete Tracker (GCCT), Global Energy Monitor (GEM), July 2025
(V1) release (https://globalenergymonitor.org/projects/global-cement-and-concrete-tracker,
CC-BY-4.0). An asset-level, researcher-curated inventory of the world's cement/clinker
plants (3,515 plants in the July-2025 release), each with a geocoded point ("Coordinates",
"lat, lon"), a "Coordinate accuracy" flag (exact = plant location confirmed via satellite
imagery / approximate = city-or-coarser estimate), an "Operating status"
(operating / operating pre-retirement / construction / announced / mothballed / retired /
cancelled / unknown), a "Plant type" (integrated / grinding / clinker only / unknown), a
"Start date" (commissioning year or "unknown"), and capacity / ownership attributes.
Distributed as a single .xlsx ("Plant Data" sheet). The download sits behind an email form
on the GEM site that mints a short-lived capability token and returns a presigned
DigitalOcean Spaces URL (see raw/SOURCE.txt and _download_xlsx below; no account required).

TRIAGE / suitability (spec sections 2, 4, 5, 8) -- ACCEPTED as a single-phenomenon
PRESENCE classification emitted as a point table (spec 2a), NOT a change dataset:

  * Observability at 10 m: cement plants are large industrial complexes -- integrated
    plants have rotary kilns, tall preheater towers, clinker/silo storage and an adjacent
    limestone quarry; grinding plants have grinding mills + silos. Clearly discernible at
    Sentinel-2/Landsat 10-30 m. GEM's own "exact" coordinate accuracy means the plant was
    confirmed against satellite imagery, so exact-accuracy centroids land on the plant.
  * Encoding: the source gives POINTS (plant centroids), not footprints, so we emit a
    single-foreground-class presence dataset (0 = cement_plant) as points.geojson (spec 2a),
    NOT per-point GeoTIFFs. Positive-only: negatives are added downstream at assembly time
    (spec 5); we do not fabricate synthetic negatives.
  * Class scheme decision (spec 5): the manifest lists "cement plant (integrated/grinding)".
    Plant type is well-populated (integrated 2421, grinding 947) and integrated plants are
    generally larger, but reliably separating integrated vs grinding from a single 10 m
    POINT (no footprint) is only weakly observable, and 118 "unknown" + 29 "clinker only"
    do not map cleanly, so we take the spec's DEFAULT single presence class. Plant type,
    capacity, production type are kept only as documented auxiliary attributes, not classes
    (none is reliably observable at 10 m from a point).

Status / year filter (spec 2, 5, 8) -- observable-in-the-Sentinel-era plants only:
  * Keep operating + "operating pre-retirement" plants (physically built and standing).
  * Drop announced / construction / mothballed / retired / cancelled / unknown status
    (not confirmable as an active, standing plant in a Sentinel-era window; retirements
    have no dated retired-year in the release so cannot be time-bounded).
  * Keep only "exact" Coordinate accuracy (satellite-confirmed). "approximate" points are
    city/subnational/country-level estimates that can be many km off the plant -- unusable
    as a 1x1 point label -- so they are dropped.

Time / change handling (spec 5): a built cement plant is a PERSISTENT structure, not a
dated change event, and Start date is only a calendar YEAR (coarser than the ~1-2 month
change-timing rule), so change_time = null. Each plant gets a 1-year window sampled
(seeded) from [max(start_year, 2016), 2025] (start "unknown" -> assume pre-existing,
[2016, 2025]); operating plants whose start year is after 2025 are skipped.

Run (idempotent; re-downloads the xlsx only if missing, then overwrites outputs):
  python3 -m olmoearth_pretrain.open_set_segmentation_data.datasets.gem_global_cement_and_concrete_tracker
"""

import argparse
import json
import random
import urllib.request
from collections import Counter
from typing import Any

from olmoearth_pretrain.open_set_segmentation_data import io, manifest

SLUG = "gem_global_cement_and_concrete_tracker"
NAME = "GEM Global Cement and Concrete Tracker"
URL = "https://globalenergymonitor.org/projects/global-cement-and-concrete-tracker"
XLSX_FILE = "Global-Cement-and-Concrete-Tracker_July-2025.xlsx"
DATA_SHEET = "Plant Data"

# GEM email-gated presign download (public supabase key; no account required).
GEM_KEY = "sb_publishable_8mQAV8B2HhveNc5T8VGqPQ_1lgsFAvz"
GEM_BASE = "https://auxunjnrktkmeqyoyngm.supabase.co"
GEM_TRACKER_SLUG = "cement-concrete-tracker"

CID_CEMENT = 0
CLASSES = [
    {
        "id": CID_CEMENT,
        "name": "cement_plant",
        "description": (
            "An operating cement/clinker plant from Global Energy Monitor's Global Cement "
            "and Concrete Tracker. Integrated plants have rotary kilns, preheater towers, "
            "clinker/silo storage and usually an adjacent limestone quarry; grinding plants "
            "have grinding mills and silos. A large industrial facility observable at 10 m "
            "in Sentinel-2/Landsat imagery. Point marks the plant centroid (GEM 'exact' "
            "coordinate accuracy = satellite-confirmed location)."
        ),
    },
]

# Sentinel-era window bounds.
MIN_YEAR = 2016
MAX_YEAR = 2025
SEED = 42
COORD_DECIMALS = 5  # ~1 m; collapses exact-duplicate records onto one 10 m pixel

# Statuses that denote a physically-built, standing plant.
KEEP_STATUSES = {"operating", "operating pre-retirement"}


def _download_xlsx() -> None:
    """Fetch the GCCT xlsx via GEM's presign flow into raw/ (idempotent, atomic)."""
    from olmoearth_pretrain.open_set_segmentation_data import download

    raw = io.raw_dir(SLUG)
    raw.mkdir(parents=True, exist_ok=True)
    dst = raw / XLSX_FILE
    if dst.exists():
        return
    body = json.dumps(
        {
            "name": "Research",
            "email": "research@example.org",
            "organization": "AI2",
            "sector": "Research",
            "country": "United States",
            "use_case": "research",
            "license_text": "CC-BY-4.0",
            "requested_slugs": [GEM_TRACKER_SLUG],
            "request_mode": "slugs",
            "custom_fields": {},
            "dynamic_params": None,
            "email_optin": False,
            "form_key": GEM_TRACKER_SLUG,
            "page_url": f"{URL}/",
            "useragent": "Mozilla/5.0",
        }
    ).encode()
    req = urllib.request.Request(
        f"{GEM_BASE}/rest/v1/rpc/mint_submission",
        data=body,
        headers={
            "apikey": GEM_KEY,
            "authorization": f"Bearer {GEM_KEY}",
            "content-type": "application/json",
        },
    )
    with urllib.request.urlopen(req, timeout=120) as r:
        token = json.loads(r.read())["capability_token"]
    req2 = urllib.request.Request(
        f"{GEM_BASE}/functions/v1/presign",
        data=b"{}",
        headers={
            "authorization": f"Bearer {token}",
            "content-type": "application/json",
        },
    )
    with urllib.request.urlopen(req2, timeout=120) as r2:
        presign = json.loads(r2.read())
    file_url = presign["urls"][0]["url"]
    download.download_http(file_url, dst)


def _to_year(value: Any) -> int | None:
    try:
        return int(str(value).strip()[:4])
    except (TypeError, ValueError):
        return None


def _parse_coords(value: Any) -> tuple[float, float] | None:
    """Parse a 'lat, lon' string into (lon, lat) with range validation."""
    if value is None:
        return None
    parts = str(value).split(",")
    if len(parts) != 2:
        return None
    try:
        lat = float(parts[0].strip())
        lon = float(parts[1].strip())
    except ValueError:
        return None
    if not (-90 <= lat <= 90 and -180 <= lon <= 180):
        return None
    if lat == 0.0 and lon == 0.0:
        return None
    return lon, lat


def _load_plants() -> tuple[list[dict[str, Any]], Counter]:
    """Read the GCCT 'Plant Data' sheet and apply the status/accuracy/coord filters."""
    import openpyxl

    path = (io.raw_dir(SLUG) / XLSX_FILE).path
    wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
    ws = wb[DATA_SHEET]
    rows = ws.iter_rows(values_only=True)
    hdr = list(next(rows))
    idx = {h: i for i, h in enumerate(hdr)}

    dropped: Counter = Counter()
    seen: set[tuple[float, float]] = set()
    plants: list[dict[str, Any]] = []
    for r in rows:
        if all(x is None for x in r):
            continue
        status = str(r[idx["Operating status"]] or "").strip().lower()
        if status not in KEEP_STATUSES:
            dropped[f"status:{status or 'blank'}"] += 1
            continue
        accuracy = str(r[idx["Coordinate accuracy"]] or "").strip().lower()
        if accuracy != "exact":
            dropped[f"accuracy:{accuracy or 'blank'}"] += 1
            continue
        coords = _parse_coords(r[idx["Coordinates"]])
        if coords is None:
            dropped["bad_coords"] += 1
            continue
        lon, lat = coords
        start_year = _to_year(r[idx["Start date"]])
        lo = MIN_YEAR if start_year is None else max(start_year, MIN_YEAR)
        if lo > MAX_YEAR:
            dropped["start_after_2025"] += 1
            continue
        key = (round(lon, COORD_DECIMALS), round(lat, COORD_DECIMALS))
        if key in seen:
            dropped["dup_coords"] += 1
            continue
        seen.add(key)
        plants.append(
            {
                "lon": lon,
                "lat": lat,
                "lo_year": lo,
                "start_year": start_year,
                "plant_type": str(r[idx["Plant type"]] or "").strip().lower(),
                "source_id": f"gcct/{r[idx['GEM Plant ID']]}",
            }
        )
    wb.close()
    return plants, dropped


def main() -> None:
    argparse.ArgumentParser().parse_args()

    io.check_disk()
    manifest.write_registry_entry(SLUG, "in_progress")

    _download_xlsx()
    raw = io.raw_dir(SLUG)
    with (raw / "SOURCE.txt").open("w") as f:
        f.write(
            "Global Cement and Concrete Tracker (GCCT), Global Energy Monitor (GEM), "
            "July 2025 (V1) release.\n"
            f"{URL}\nLicense: CC-BY-4.0.\n"
            f"File: {XLSX_FILE} (sheet 'Plant Data', 3,515 cement/clinker plants).\n\n"
            "Download recipe (GEM email-gated presign flow, no account required):\n"
            f"  KEY={GEM_KEY}\n"
            f"  SLUG={GEM_TRACKER_SLUG}\n"
            f"  1) POST {GEM_BASE}/rest/v1/rpc/mint_submission\n"
            "     headers: apikey:$KEY, authorization:Bearer $KEY, content-type:application/json\n"
            "     body: {name,email,organization,sector,country,use_case,license_text,\n"
            "            requested_slugs:[$SLUG],request_mode:'slugs',custom_fields:{},\n"
            "            dynamic_params:null,email_optin:false,form_key:$SLUG,\n"
            "            page_url:<project url>,useragent:'...'}\n"
            "     -> returns {capability_token}\n"
            f"  2) POST {GEM_BASE}/functions/v1/presign\n"
            "     header: authorization:Bearer <capability_token>, body {}\n"
            "     -> returns {urls:[{url,filename}]}; GET url -> the .xlsx\n"
        )

    plants, dropped = _load_plants()
    print(
        f"kept {len(plants)} operating exact-coord plants; dropped {dict(dropped)}",
        flush=True,
    )

    rng = random.Random(SEED)
    points: list[dict[str, Any]] = []
    year_counts: Counter = Counter()
    type_counts: Counter = Counter()
    for i, p in enumerate(plants):
        year = rng.randint(p["lo_year"], MAX_YEAR)
        year_counts[year] += 1
        type_counts[p["plant_type"]] += 1
        points.append(
            {
                "id": f"{i:06d}",
                "lon": p["lon"],
                "lat": p["lat"],
                "label": CID_CEMENT,
                "time_range": io.year_range(year),
                "change_time": None,
                "source_id": p["source_id"],
            }
        )
    io.write_points_table(SLUG, "classification", points)
    io.check_disk()

    io.write_dataset_metadata(
        SLUG,
        {
            "dataset": SLUG,
            "name": NAME,
            "task_type": "classification",
            "source": "Global Energy Monitor",
            "license": "CC-BY-4.0",
            "provenance": {
                "url": URL,
                "have_locally": False,
                "annotation_method": "authoritative expert curation, satellite-confirmed",
                "file": XLSX_FILE,
                "release": "July 2025 (V1)",
            },
            "sensors_relevant": ["sentinel2", "sentinel1", "landsat"],
            "classes": CLASSES,
            "nodata_value": io.CLASS_NODATA,
            "num_samples": len(points),
            "class_counts": {"cement_plant": len(points)},
            "auxiliary_plant_type_counts": {
                k: type_counts[k] for k in sorted(type_counts)
            },
            "year_counts": {str(y): year_counts[y] for y in sorted(year_counts)},
            "notes": (
                "Presence point dataset: single foreground class 0=cement_plant, emitted as "
                "points.geojson (spec 2a). From GEM's GCCT plant inventory (July 2025 V1, "
                "3,515 plants). Kept only 'operating'/'operating pre-retirement' plants with "
                "'exact' (satellite-confirmed) Coordinate accuracy; dropped announced/"
                "construction/mothballed/retired/cancelled/unknown status and approximate "
                "(city-level) coordinates. Plant type (integrated/grinding/clinker-only), "
                "capacity and production type are NOT used as class targets (weak/unobservable "
                "from a 10 m point) -- retained only as documented auxiliary attributes. "
                "Positive-only (spec 5): negatives supplied downstream from other datasets. "
                "Persistent-structure time model (change_time=null): each plant gets a 1-year "
                "window sampled from [max(start_year,2016), 2025] (start 'unknown' -> "
                "[2016,2025]); Start date resolves only to a calendar year (coarser than the "
                "~1-2 month change-timing rule) so NO dated change labels. Coordinates "
                f"de-duplicated at {COORD_DECIMALS} dp. Well under the 25k per-dataset cap."
            ),
        },
    )
    manifest.write_registry_entry(
        SLUG, "completed", task_type="classification", num_samples=len(points)
    )
    print(f"done: {len(points)} presence points", flush=True)
    print("plant types:", dict(type_counts), flush=True)
    print("year counts:", {y: year_counts[y] for y in sorted(year_counts)}, flush=True)


if __name__ == "__main__":
    main()
