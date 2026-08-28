"""GEM Global Iron and Steel Tracker (GIST) -> steel/iron-plant presence points.

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

Task type: presence-only classification POINTS (spec section 2a). Each included plant is
emitted as a single presence point in one dataset-wide ``points.geojson``. Single foreground
class (steel/iron plant, id 0). Negatives are supplied downstream by the assembly step from
other datasets.

Decisions (spec sections 2-5):
  * INCLUSION: only plants with at least one unit that is operating / operating
    pre-retirement / mothballed / mothballed pre-retirement AND an 'exact' (satellite-
    confirmed) coordinate -> physically standing, visible structures precisely located
    (986 of 1293 plants). Dropped: announced / cancelled (not built), construction-only
    (not yet a plant), retired-only (may be demolished and cannot be time-anchored reliably),
    and 'approximate' coordinates (city/subnational estimates, potentially km off).
  * CHANGE vs PRESENCE (spec section 5 timing rule): "Start date" is year-granular only, so the
    build event is NOT resolvable to ~1-2 months -> NOT a change label. We use the persistent
    post-construction STATE (a steel mill stays visible for decades) as presence with
    change_time=null, anchoring each point's 1-year window in the Sentinel era at/after the
    start year, spread deterministically over [lo, 2025] for imagery diversity where
    lo = clamp(start, 2017, 2025) (start<2017 or unknown -> lo=2017).

Run (reuses cached raw):
  python3 -m olmoearth_pretrain.open_set_segmentation_data.datasets.gem_global_iron_and_steel_tracker
"""

import argparse
import hashlib
import math
import multiprocessing
from collections import Counter
from typing import Any

from olmoearth_pretrain.open_set_segmentation_data import download, io, manifest
from olmoearth_pretrain.open_set_segmentation_data.sampling import balance_by_class

SLUG = "gem_global_iron_and_steel_tracker"
NAME = "GEM Global Iron and Steel Tracker"
URL = "https://globalenergymonitor.org/projects/global-iron-and-steel-tracker/download-data/"
GEM_SLUG = "iron-steel-plant-tracker"
XLSX_NAME = "Plant-level_data_Global_Iron_and_Steel_Tracker_June_2026_V1.xlsx"
PLANT_SHEET = "Plant data"
STATUS_SHEET = "Plant capacities and status"

# Single foreground class: steel/iron plant (id 0). No background class.
PLANT_ID = 0

# Statuses that mean the plant physically stands and is visible in imagery.
PRESENT_STATUSES = {
    "operating",
    "operating pre-retirement",
    "mothballed",
    "mothballed pre-retirement",
}

WINDOW_MIN, WINDOW_MAX = 2017, 2025  # S2-era window clamp
PER_CLASS = 1000
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
        # country-level estimates that can be many km off the plant. 'exact' means GEM
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=64)
    parser.add_argument(
        "--contact-name",
        default="",
        help="Your name, submitted to the GEM download form (only needed on first download).",
    )
    parser.add_argument(
        "--contact-email",
        default="",
        help="Your email, submitted to the GEM download form (only needed on first download).",
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

    records: list[dict[str, Any]] = []
    for p in plants:
        records.append(
            {
                "label": PLANT_ID,
                "lon": p["lon"],
                "lat": p["lat"],
                "window_year": _window_year(p["start_year"], p["pid"]),
                "source_id": (
                    f"gem_plant/{p['pid']}/{p['name']}/{p['equipment']}/"
                    f"start={p['start_year']}"
                ),
                "equipment": p["equipment"],
                "country": p["country"],
            }
        )
    print(f"built {len(records)} presence records", flush=True)
    selected = balance_by_class(records, "label", per_class=PER_CLASS)
    print(f"selected {len(selected)} (<= {PER_CLASS}/class)", flush=True)

    points = []
    for i, r in enumerate(selected):
        points.append(
            {
                "id": f"{i:06d}",
                "lon": r["lon"],
                "lat": r["lat"],
                "label": r["label"],
                "time_range": io.year_range(r["window_year"]),
                "change_time": None,
                "source_id": r["source_id"],
                "equipment": r["equipment"],
                "country": r["country"],
            }
        )
    io.write_points_table(SLUG, "classification", points)

    counts = Counter(r["label"] for r in selected)
    n_exact = sum(1 for p in plants if p["accuracy"] == "exact")
    io.write_dataset_metadata(
        SLUG,
        {
            "dataset": SLUG,
            "name": NAME,
            "task_type": "classification",
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
                    "id": PLANT_ID,
                    "name": "steel/iron plant",
                    "description": (
                        "Crude iron/steel plant with >= 500,000 t/yr capacity (BF, BOF, "
                        "EAF, DRI, IF and related furnaces) -- a very large, clearly "
                        "discernible industrial complex (blast furnaces, stoves, sinter/"
                        "coking plants, rolling mills, stockyards)."
                    ),
                },
            ],
            "nodata_value": io.CLASS_NODATA,
            "num_samples": len(points),
            "class_counts": {"steel/iron plant": counts.get(PLANT_ID, 0)},
            "available": {
                "plants_total_in_source": 1293,
                "plants_present_exact_included": len(plants),
                "note": (
                    "included = operating/mothballed unit AND 'exact' coordinate; "
                    f"all {n_exact} included are exact by construction"
                ),
            },
            "window_rule": (
                f"presence/state; change_time=null; window in [clamp(start,{WINDOW_MIN},"
                f"{WINDOW_MAX}), {WINDOW_MAX}] spread by plant hash"
            ),
            "notes": (
                "Presence-only classification POINTS converted from the old detection-tile "
                "encoding. Each included iron/steel plant is emitted as a single presence "
                "point (no fabricated GeoTIFF context, no background/negative tiles); single "
                "foreground class (id 0 = steel/iron plant). INCLUSION: only plants with "
                "a unit that is operating / operating pre-retirement / mothballed / mothballed "
                "pre-retirement AND an 'exact' (satellite-confirmed) coordinate (986 of 1293 "
                "physically-standing, precisely-located plants); dropped announced/cancelled "
                "(not built), construction-only (not yet a plant), retired-only (may be "
                "demolished / cannot time-anchor), and 'approximate' coordinates. Presence/"
                "state, NOT change: 'Start date' is year-granular only (not resolvable to "
                "~1-2 months per the spec change-timing rule), so the persistent post-"
                "construction state is used with change_time=null and a 1-year window at/after "
                f"the start year in the S2 era (window in [clamp(start,{WINDOW_MIN},"
                f"{WINDOW_MAX}),{WINDOW_MAX}] spread deterministically per plant for imagery "
                "diversity). up to 1000 points/class (balance_by_class). Negatives are supplied "
                "downstream by the assembly step from other datasets."
            ),
        },
    )
    manifest.write_registry_entry(
        SLUG, "completed", task_type="classification", num_samples=len(points)
    )
    print(f"done: {len(points)} points", flush=True)


if __name__ == "__main__":
    multiprocessing.set_start_method("forkserver", force=True)
    main()
