"""Process the Global Tailings Portal into an open-set-segmentation presence point table.

Source: Global Tailings Portal (GRID-Arendal), https://tailing.grida.no/. A free, public
disclosure database of mine **tailings storage facilities (TSFs)** launched Jan 2020,
built from disclosures by 100+ of the world's largest mining companies (originally
collected by the Church of England Investor Mining & Tailings Safety Initiative). Each TSF
is a company-disclosed record with a geocoded POINT (lat/lon centroid) plus attributes
(facility name, owner company, country, hazard/consequence classification).

Access (no credentials needed): the portal's Leaflet dashboard populates its markers from
a public JSON endpoint used here as the label source:
  https://tailing.grida.no/api/taillingLoc?format=json
-> list of {pk, tsf, latitude, longitude, country, hazard_categorization, owner_company}
(~2113 records). No imagery is pulled (pretraining supplies its own).

Encoding decision (spec section 2a / section 4 "points marking presence"):
  * TSFs are large industrial features (hundreds of m to km) -> observable at 10 m.
  * But the portal gives POINTS (disclosed centroids), NOT footprints, and coordinate
    precision is uneven (see caveat below). We therefore encode a single-foreground-class
    PRESENCE dataset: one point per facility, class 0 = "tailings_facility".
  * We DO NOT use the disclosed attributes (dam type, hazard/consequence class,
    construction year, active/inactive) as the class target: none of these is reliably
    observable from S2/S1/Landsat at 10 m. Simple presence is the only defensible target.
  * Emitted as a dataset-wide points.geojson (spec section 2a), NOT per-point GeoTIFFs.
    This is a positive-only presence dataset (spec section 5): non-facility negatives are
    supplied downstream at pretraining-assembly time by sampling other datasets; we do not
    fabricate synthetic negatives.

Coordinate-precision caveat (documented, not disqualifying): coordinates are
company-disclosed centroids. ~93% carry >=3 decimal places, but real positional accuracy
is unknown and some points may sit off the exact facility by tens of metres. Because a TSF
footprint is typically hundreds of metres across, a centroid with that level of error still
lands on or immediately beside the facility, so these remain useful (if weak) presence
labels. Records with missing / out-of-range / (0,0) coordinates are dropped; exact-duplicate
coordinates (same 10 m pixel) are de-duplicated.

Time range: TSFs are persistent structures and the disclosures describe existence as of the
~2019-2020 reporting round. Per spec section 5 (static labels) each point gets a 1-year
Sentinel-era window, spread across 2019-2023 for temporal diversity (all post-2016).
change_time = null (presence, not a dated change/event).

Run (idempotent; re-downloads only if the raw file is missing, then overwrites outputs):
  python3 -m olmoearth_pretrain.open_set_segmentation_data.datasets.global_tailings_portal
"""

import argparse
import json
import random
from collections import Counter
from typing import Any

from olmoearth_pretrain.open_set_segmentation_data import download, io, manifest

SLUG = "global_tailings_portal"
NAME = "Global Tailings Portal"
API_URL = "https://tailing.grida.no/api/taillingLoc?format=json"
RAW_FILE = "taillingLoc.json"

FACILITY_ID = 0
CLASSES = [
    {
        "id": FACILITY_ID,
        "name": "tailings_facility",
        "description": (
            "Mine tailings storage facility (TSF) / tailings dam: an engineered "
            "impoundment storing mine-waste slurry, disclosed by mining companies to the "
            "Global Tailings Portal (GRID-Arendal). Point marks the company-disclosed "
            "facility centroid; footprints are typically hundreds of metres to kilometres "
            "across and are observable at 10 m in Sentinel-2/Landsat imagery."
        ),
    },
]

# Persistent structures -> spread a 1-year window across Sentinel-era years for diversity.
YEARS = [2019, 2020, 2021, 2022, 2023]
SEED = 42
COORD_DECIMALS = 5  # ~1 m; collapses exact-duplicate disclosures onto one 10 m pixel


def _valid(rec: dict[str, Any]) -> bool:
    try:
        lat = float(rec["latitude"])
        lon = float(rec["longitude"])
    except (TypeError, ValueError, KeyError):
        return False
    if not (-90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0):
        return False
    if lat == 0.0 and lon == 0.0:
        return False
    return True


def load_records() -> list[dict[str, Any]]:
    """Download (idempotent) and parse the TSF location table into deduped valid records."""
    raw = io.raw_dir(SLUG)
    raw.mkdir(parents=True, exist_ok=True)
    dst = raw / RAW_FILE
    download.download_http(API_URL, dst)
    with dst.open() as f:
        data = json.load(f)

    seen: set[tuple[float, float]] = set()
    out: list[dict[str, Any]] = []
    n_invalid = 0
    n_dup = 0
    for rec in data:
        if not _valid(rec):
            n_invalid += 1
            continue
        lat = round(float(rec["latitude"]), COORD_DECIMALS)
        lon = round(float(rec["longitude"]), COORD_DECIMALS)
        key = (lat, lon)
        if key in seen:
            n_dup += 1
            continue
        seen.add(key)
        out.append(
            {
                "lon": float(rec["longitude"]),
                "lat": float(rec["latitude"]),
                "source_id": f"pk/{rec.get('pk')}",
            }
        )
    print(
        f"loaded {len(data)} raw records -> {len(out)} usable "
        f"({n_invalid} invalid coords, {n_dup} duplicate coords dropped)",
        flush=True,
    )
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.parse_args()

    io.check_disk()
    manifest.write_registry_entry(SLUG, "in_progress")

    raw = io.raw_dir(SLUG)
    raw.mkdir(parents=True, exist_ok=True)
    with (raw / "SOURCE.txt").open("w") as f:
        f.write(
            "Global Tailings Portal (GRID-Arendal), https://tailing.grida.no/ .\n"
            "Public disclosure database of mine tailings storage facilities (TSFs), "
            "launched Jan 2020 from 100+ mining companies' disclosures.\n"
            f"Label source (public JSON, no auth): {API_URL}\n"
            "-> list of {pk, tsf, latitude, longitude, country, hazard_categorization, "
            "owner_company}. Saved as taillingLoc.json. No imagery downloaded.\n"
            "License: free/public.\n"
        )

    records = load_records()
    io.check_disk()

    rng = random.Random(SEED)
    points: list[dict[str, Any]] = []
    for i, r in enumerate(records):
        year = YEARS[rng.randrange(len(YEARS))]
        points.append(
            {
                "id": f"{i:06d}",
                "lon": r["lon"],
                "lat": r["lat"],
                "label": FACILITY_ID,
                "time_range": io.year_range(year),
                "change_time": None,
                "source_id": r["source_id"],
            }
        )
    io.write_points_table(SLUG, "classification", points)

    year_counts = Counter(int(p["time_range"][0].year) for p in points)
    io.write_dataset_metadata(
        SLUG,
        {
            "dataset": SLUG,
            "name": NAME,
            "task_type": "classification",
            "source": "GRID-Arendal (Global Tailings Portal)",
            "license": "free/public",
            "provenance": {
                "url": "https://tailing.grida.no/",
                "label_endpoint": API_URL,
                "have_locally": False,
                "annotation_method": "company disclosure, geocoded (portal beta)",
            },
            "sensors_relevant": ["sentinel2", "sentinel1", "landsat"],
            "classes": CLASSES,
            "nodata_value": io.CLASS_NODATA,
            "num_samples": len(points),
            "class_counts": {"tailings_facility": len(points)},
            "year_counts": {str(y): year_counts[y] for y in sorted(year_counts)},
            "notes": (
                "Presence point dataset: single foreground class 0=tailings_facility, "
                "emitted as points.geojson (spec 2a). Disclosed attributes (dam type, "
                "hazard/consequence class, construction year, active/inactive) are NOT used "
                "as class targets - none is reliably observable at 10 m; simple presence is "
                "the target. Positive-only (spec 5): negatives supplied downstream from other "
                "datasets. Coordinates are company-disclosed centroids (uneven precision, "
                "some tens of metres off); TSF footprints are hundreds of m to km so "
                "centroids still land on/near the facility. Invalid/(0,0)/duplicate "
                "coordinates dropped. Persistent structures -> 1-year Sentinel-era window "
                "spread over 2019-2023, change_time=null."
            ),
        },
    )
    manifest.write_registry_entry(
        SLUG, "completed", task_type="classification", num_samples=len(points)
    )
    print(f"done: {len(points)} presence points", flush=True)


if __name__ == "__main__":
    main()
