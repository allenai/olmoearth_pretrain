"""Process the Global Wind Power Tracker (GWPT) into presence-only points.

Source: Global Wind Power Tracker, Global Energy Monitor (GEM), February 2026 release
(https://globalenergymonitor.org/projects/global-wind-power-tracker, CC-BY-4.0). A
researcher-curated, facility-level inventory of utility-scale (>=10 MW) onshore and
offshore wind power project *phases* worldwide (33,248 phases in the Feb-2026 release),
each with a point Latitude/Longitude, an operating Status, an Installation Type
(Onshore / Offshore ...), a commissioning Start year, and a Location accuracy flag
(exact / approximate). Distributed as a single .xlsx (sheet "Data"). The download sits
behind an email form on the GEM site that mints a short-lived capability token and returns
a presigned DigitalOcean Spaces URL (see raw/SOURCE.txt for the exact recipe).

Task type: presence-only POINTS (spec section 2a). Each selected operating wind farm is
emitted as one presence point in a dataset-wide ``points.geojson``; negatives are supplied
by the downstream assembly (no fabricated background tiles here).

Class scheme: onshore and offshore farms look very different at 10-30 m (turbines +
pads/access roads on land vs. turbine monopiles standing in open water), so we keep two
observable positive classes:
    0 = onshore_wind_farm, 1 = offshore_wind_farm.
Utility-scale wind farms (many large turbines with cleared pads and access roads spread
over hundreds of metres) are resolvable at 10 m; the DeepOWT precedent detects even
individual offshore turbines at Sentinel-1 10 m.

Only *operating* phases are used as positives (they are physically built and visible);
construction / pre-construction / announced / cancelled / shelved / retired / mothballed
phases are excluded. Phases with Installation Type "Unknown" (cannot assign onshore/offshore)
are excluded.

Time / change handling. A built wind farm is a PERSISTENT structure, not a dated change
event: once operating it stays visible for years, and GWPT only resolves the Start year to
a calendar YEAR (coarser than the ~1-2 month change-timing requirement), so we do NOT emit
dated change labels (change_time = null). Each positive is given a 1-year time window
sampled (seeded) from the years the farm is both operating and inside the Sentinel era:
[max(start_year, 2016), min(2025, retired_year - 1)] (start_year missing -> assume a
pre-existing persistent farm, [2016, 2025]). Phases whose first operating year is after 2025
(no full Sentinel year yet) are skipped.

Sampling: up to 1000 points per class (sampling.balance_by_class, default 25k total cap).
Offshore is a rare class (~360) and is kept in full (spec section 5 keeps rare classes).

Run (reuses cached raw xlsx):
  python3 -m olmoearth_pretrain.open_set_segmentation_data.datasets.global_wind_power_tracker
"""

import argparse
import multiprocessing
import random
from collections import Counter
from typing import Any

from olmoearth_pretrain.open_set_segmentation_data import io, manifest
from olmoearth_pretrain.open_set_segmentation_data.sampling import balance_by_class

SLUG = "global_wind_power_tracker"
NAME = "Global Wind Power Tracker"
URL = "https://globalenergymonitor.org/projects/global-wind-power-tracker"
XLSX_FILE = "Global-Wind-Power-Tracker-February-2026.xlsx"
DATA_SHEET = "Data"

# Class scheme (two observable positive classes; no background).
CID_ONSHORE = 0
CID_OFFSHORE = 1
CLASSES = [
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
CID_TO_NAME = {c["id"]: c["name"] for c in CLASSES}

# Sentinel-era window bounds. min_year: first year of usable S2/S1 imagery. max_year: last
# full calendar year we assume imagery is available and the farm still standing.
MIN_YEAR = 2016
MAX_YEAR = 2025

PER_CLASS = 1000
SEED = 42


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


def _build_records(farms: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """One presence record per operating on/offshore farm with a valid presence window.

    The label year is sampled (seeded) from the presence window for temporal diversity.
    """
    rng = random.Random(SEED)
    recs: list[dict[str, Any]] = []
    for f in farms:
        if f["status"] != "operating" or f["cls"] is None:
            continue
        pr = _presence_range(f)
        if pr is None:
            continue
        lo, hi = pr
        recs.append(
            {
                "label": f["cls"],
                "year": rng.randint(lo, hi),
                "lon": f["lon"],
                "lat": f["lat"],
                "source_id": f"gwpt/{f['phase_id']}",
            }
        )
    return recs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=64)
    parser.parse_args()

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

    recs = _build_records(farms)
    cand_counts = Counter(r["label"] for r in recs)
    print(
        "presence candidates: "
        + ", ".join(f"{CID_TO_NAME[c]}={cand_counts[c]}" for c in sorted(cand_counts)),
        flush=True,
    )

    selected = balance_by_class(recs, "label", per_class=PER_CLASS, seed=SEED)
    print(f"selected {len(selected)} points (<= {PER_CLASS}/class)", flush=True)

    points = []
    for i, r in enumerate(selected):
        points.append(
            {
                "id": f"{i:06d}",
                "lon": r["lon"],
                "lat": r["lat"],
                "label": r["label"],
                "time_range": io.year_range(r["year"]),
                "change_time": None,
                "source_id": r["source_id"],
            }
        )
    io.write_points_table(SLUG, "classification", points)

    counts = Counter(r["label"] for r in selected)
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
                "annotation_method": "manual/expert curation",
                "file": XLSX_FILE,
                "release": "February 2026",
            },
            "sensors_relevant": ["sentinel2", "sentinel1", "landsat"],
            "classes": CLASSES,
            "num_samples": len(selected),
            "class_counts": {
                CID_TO_NAME[c]: counts.get(c, 0) for c in sorted(CID_TO_NAME)
            },
            "notes": (
                "Presence-only POINTS converted from the former detection-tile encoding; "
                "negatives are supplied by the downstream assembly. Utility-scale (>=10 MW) "
                "wind farms from Global Energy Monitor's GWPT point inventory (Feb 2026 "
                "release, 33,248 phases). Only 'operating' phases used; onshore vs offshore "
                "kept as two observable classes (0=onshore_wind_farm, 1=offshore_wind_farm). "
                "Persistent-structure time model (change_time=null): each farm gets a 1-year "
                "window sampled from [max(start_year,2016), min(2025, retired-1)] "
                "(missing start_year -> [2016,2025]); phases first operating after 2025 "
                "skipped. GWPT resolves commissioning only to a calendar year (coarser than "
                "the ~1-2 month change-timing rule), so NO dated change labels. Sampling: up "
                "to 1000 points/class (balance_by_class); offshore is a rare class (~360) "
                "kept in full (spec section 5 keeps rare classes)."
            ),
        },
    )
    manifest.write_registry_entry(
        SLUG, "completed", task_type="classification", num_samples=len(selected)
    )
    print(f"done: {len(selected)} points", flush=True)


if __name__ == "__main__":
    multiprocessing.set_start_method("forkserver", force=True)
    main()
