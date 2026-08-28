"""Process the OurAirports point database into open-set-segmentation labels.

Source: OurAirports (https://ourairports.com/data/), a community-maintained global
database of ~85k airfields distributed as simple CSV downloads (public domain). Each
record has a lon/lat and a ``type`` classifying the airfield (large/medium/small
airport, heliport, seaplane base, plus non-target ``closed`` / ``balloonport``). This is
a pure sparse-point dataset (each label is one 10 m pixel with an airport-type class),
so we write ONE dataset-wide point table (points.geojson, spec 2a), balanced to
<=1000 per class.

Airports are static features, so per spec 5 we anchor each point to a single
representative 1-year window in the Sentinel era (2020). Airfield type is inferred by
OurAirports largely from runway length/infrastructure; large/medium airports and long
runways are clearly visible at 10-30 m, while many small airports, heliports, and
seaplane bases are near or below the resolution limit (a single helipad or a float dock
is sub-pixel). We keep every target class regardless (assembly-time filtering handles
rare/too-small classes, spec 5) and record the observability caveat in the summary.
"""

import argparse
import csv
import multiprocessing
from collections import Counter
from typing import Any

from olmoearth_pretrain.open_set_segmentation_data import download, io, manifest
from olmoearth_pretrain.open_set_segmentation_data.sampling import balance_by_class

SLUG = "ourairports"
AIRPORTS_URL = "https://ourairports.com/data/airports.csv"
RUNWAYS_URL = "https://ourairports.com/data/runways.csv"
PER_CLASS = 1000
# Static features -> representative 1-year window in the Sentinel era (spec 5).
REPRESENTATIVE_YEAR = 2020

# OurAirports ``type`` value -> (class name, description). Order defines the class id
# (0..4), matching the manifest class ordering. ``closed`` and ``balloonport`` are not
# target classes and are dropped.
CLASSES: list[tuple[str, str, str]] = [
    (
        "large_airport",
        "large airport",
        "Major airport with long paved runways and extensive terminal/apron infrastructure; clearly resolvable at 10-30 m.",
    ),
    (
        "medium_airport",
        "medium airport",
        "Regional airport with one or more substantial (often paved) runways; runway footprint visible at 10-30 m.",
    ),
    (
        "small_airport",
        "small airport",
        "Small airfield / general-aviation strip, frequently a single short paved or unpaved runway; may be near the resolution limit.",
    ),
    (
        "heliport",
        "heliport",
        "Helicopter landing facility (helipad/pads); the pad itself is typically sub-pixel at 10 m though the surrounding site may be visible.",
    ),
    (
        "seaplane_base",
        "seaplane base",
        "Water-based facility for seaplane operations (a marked water area plus docks); the operating area is largely sub-resolution at 10-30 m.",
    ),
]
TYPE_TO_ID = {src: i for i, (src, _name, _desc) in enumerate(CLASSES)}
DROPPED_TYPES = {"closed", "balloonport"}


def scan_records(csv_path: str) -> tuple[list[dict[str, Any]], Counter]:
    """Read airports.csv into flat records for target classes with valid coordinates.

    Returns (records, raw_type_counts). ``raw_type_counts`` covers every ``type`` seen
    (including dropped/non-target ones) for reporting.
    """
    records: list[dict[str, Any]] = []
    raw_counts: Counter = Counter()
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            t = row["type"]
            raw_counts[t] += 1
            if t not in TYPE_TO_ID:
                continue
            lat_s, lon_s = row["latitude_deg"], row["longitude_deg"]
            if not lat_s or not lon_s:
                continue
            try:
                lon, lat = float(lon_s), float(lat_s)
            except ValueError:
                continue
            if not (-180.0 <= lon <= 180.0 and -90.0 <= lat <= 90.0):
                continue
            records.append(
                {
                    "lon": lon,
                    "lat": lat,
                    "label": t,
                    "source_id": row.get("ident") or str(row.get("id")),
                }
            )
    return records, raw_counts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=64)
    args = parser.parse_args()
    _ = args  # single-file CSV parse is fast; no pool needed for the scan itself.

    io.check_disk()
    manifest.write_registry_entry(SLUG, "in_progress")

    raw = io.raw_dir(SLUG)
    raw.mkdir(parents=True, exist_ok=True)
    airports_csv = download.download_http(AIRPORTS_URL, raw / "airports.csv")
    # runways.csv is downloaded for provenance/completeness; not required for typing.
    download.download_http(RUNWAYS_URL, raw / "runways.csv")

    recs, raw_counts = scan_records(str(airports_csv))
    print(
        f"scanned {len(recs)} target-class airfields; raw type counts: {dict(raw_counts)}"
    )

    selected = balance_by_class(recs, "label", per_class=PER_CLASS)
    print(f"selected {len(selected)} (<= {PER_CLASS}/class, 25k total cap)")

    points = []
    for i, r in enumerate(selected):
        points.append(
            {
                "id": f"{i:06d}",
                "lon": r["lon"],
                "lat": r["lat"],
                "label": TYPE_TO_ID[r["label"]],
                "time_range": io.year_range(REPRESENTATIVE_YEAR),
                "source_id": r["source_id"],
            }
        )
    io.write_points_table(SLUG, "classification", points)

    counts = Counter(r["label"] for r in selected)
    io.write_dataset_metadata(
        SLUG,
        {
            "dataset": SLUG,
            "name": "OurAirports",
            "task_type": "classification",
            "source": "OurAirports",
            "license": "public domain",
            "provenance": {
                "url": "https://ourairports.com/data/",
                "have_locally": False,
                "annotation_method": "manual (crowdsourced)",
            },
            "sensors_relevant": ["sentinel2", "sentinel1", "landsat"],
            "classes": [
                {"id": i, "name": name, "description": desc}
                for i, (_src, name, desc) in enumerate(CLASSES)
            ],
            "nodata_value": io.CLASS_NODATA,
            "num_samples": len(selected),
            "class_counts": {name: counts.get(src, 0) for src, name, _ in CLASSES},
            "notes": (
                "1x1 point-segmentation labels; airport type from OurAirports 'type' field. "
                "'closed' and 'balloonport' dropped (non-target). Static features -> a single "
                f"representative 1-year window ({REPRESENTATIVE_YEAR}) in the Sentinel era. "
                "Small airports/heliports/seaplane bases may be sub-resolution at 10-30 m; "
                "kept regardless per spec 5 (assembly filters rare/too-small classes)."
            ),
        },
    )
    manifest.write_registry_entry(
        SLUG, "completed", task_type="classification", num_samples=len(selected)
    )
    print("done")


if __name__ == "__main__":
    multiprocessing.set_start_method("forkserver", force=True)
    main()
