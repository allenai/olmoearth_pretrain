"""Collapse Caldera Database (CCDB v4.0) -> open-set-segmentation point labels.

Source: CCDB, "The collapse caldera worldwide database" (Geyer & Marti; GVB-CSIC),
version 4.0 (2019), published on Zenodo (record 10636011, DOI 10.5281/zenodo.10636011,
CC-BY-NC-4.0). One Excel workbook (``CCDB4_zenodo.xls``); the ``Calderas`` sheet holds
477 collapse-caldera records with WGS84 Latitude/Longitude, caldera Max/Min diameter and
area, geological Age (+ epoch), and many structural/petrological attributes.

TRIAGE / suitability (spec 2, 5, 8) -- ACCEPTED as a WEAK single-phenomenon PRESENCE
classification, NOT a change dataset:

  * Time validity: every record's Age/epoch is geological (Pleistocene, Holocene,
    Miocene, ... -- thousands to millions of years). The caldera *collapse event* is NOT
    an observable Sentinel-era change, so a CHANGE encoding is invalid and is rejected.
    However a collapse caldera is a PERSISTENT LANDFORM still visible today, so we treat
    it as a static present-day observation: change_time=null, a representative recent
    1-year window in the Sentinel era.
  * Observability at 10 m: calderas are large (median Max_diameter 10 km; 343/386 are
    >= 5 km across, only 2 < 1 km) -- clearly discernible topographic depressions at
    10-30 m. Coordinate precision is mixed (~129 records rounded to <= 2 dp, i.e. ~1 km)
    but that error is small relative to the km-scale landform, so the point still lands on
    the caldera. The database also carries diameter/extent, confirming a real footprint.
  * Label meaningfulness: only "a collapse caldera is present here" is a coherent,
    imagery-observable per-location label. Subsurface/geological attributes (magma
    composition, collapse type, chamber depth, rock suite) are NOT inferable from optical/
    SAR imagery at 10 m, and preservation/state is recorded for only 35/477 records, so we
    deliberately do NOT use any of them as the class. Single presence class.

This is analogous to the accepted atlas_of_hillforts presence dataset. Points carry a
class, so we write the dataset-wide point table (spec 2a), not per-point GeoTIFFs.

Classes (presence-only; no background/negative class -- assembly adds negatives, spec 5):
  0 collapse_caldera   <- any CCDB caldera record with valid WGS84 coordinates
"""

import argparse
from collections import Counter
from typing import Any

import pandas as pd

from olmoearth_pretrain.open_set_segmentation_data import download, io, manifest
from olmoearth_pretrain.open_set_segmentation_data.sampling import balance_by_class

SLUG = "collapse_caldera_database_ccdb"
NAME = "Collapse Caldera Database (CCDB)"
ZENODO_RECORD = "10636011"
XLS_NAME = "CCDB4_zenodo.xls"
SHEET = "Calderas"
PER_CLASS = 1000
STATIC_YEAR = 2020  # representative Sentinel-era year for these persistent landforms
COORD_DP = 6  # decimal places for de-duplicating coincident caldera records

CLASSES = [
    (
        "collapse_caldera",
        "Location of a collapse caldera (a large volcanic collapse depression, typically "
        "1-20+ km across) from the CCDB worldwide database. A persistent topographic "
        "landform observable at 10-30 m. The collapse event itself is geological (Age "
        "ranges Precambrian..Holocene) and is NOT treated as an observable change; only "
        "present-day landform presence is labeled.",
    ),
]


def download_raw() -> pd.DataFrame:
    """Download the CCDB workbook from Zenodo (atomic) and return the Calderas sheet."""
    raw = io.raw_dir(SLUG)
    download.download_zenodo(ZENODO_RECORD, raw, filenames=[XLS_NAME])
    return pd.read_excel(str(raw / XLS_NAME), sheet_name=SHEET)


def main() -> None:
    argparse.ArgumentParser().parse_args()

    io.check_disk()
    manifest.write_registry_entry(SLUG, "in_progress")

    df = download_raw()
    print(f"loaded {len(df)} CCDB caldera records")

    lat = pd.to_numeric(df["Latitude"], errors="coerce")
    lon = pd.to_numeric(df["Longitude"], errors="coerce")

    records: list[dict[str, Any]] = []
    seen: set[tuple[float, float]] = set()
    dropped = Counter()
    for i in range(len(df)):
        la, lo = lat.iloc[i], lon.iloc[i]
        if pd.isna(la) or pd.isna(lo):
            dropped["no_coords"] += 1
            continue
        la, lo = float(la), float(lo)
        if not (-90 <= la <= 90 and -180 <= lo <= 180):
            dropped["bad_coords"] += 1
            continue
        key = (round(lo, COORD_DP), round(la, COORD_DP))
        if key in seen:
            dropped["dup_coords"] += 1
            continue
        seen.add(key)
        idc = df["IDCaldera"].iloc[i]
        name = df["CALDERA"].iloc[i]
        sid = (
            str(idc) if not pd.isna(idc) else (str(name) if not pd.isna(name) else None)
        )
        records.append({"lon": lo, "lat": la, "label": 0, "source_id": sid})
    print(f"usable {len(records)} unique-coordinate calderas; dropped {dict(dropped)}")

    # Single presence class + well under caps => no truncation, but balance for determinism.
    selected = balance_by_class(records, "label", per_class=PER_CLASS)
    print(f"selected {len(selected)} points (<= {PER_CLASS}/class)")

    time_range = io.year_range(STATIC_YEAR)
    points = [
        {
            "id": f"{i:06d}",
            "lon": r["lon"],
            "lat": r["lat"],
            "label": r["label"],
            "time_range": time_range,
            "change_time": None,
            "source_id": r["source_id"],
        }
        for i, r in enumerate(selected)
    ]
    io.write_points_table(SLUG, "classification", points)

    counts = Counter(r["label"] for r in selected)
    io.write_dataset_metadata(
        SLUG,
        {
            "dataset": SLUG,
            "name": NAME,
            "task_type": "classification",
            "source": "GVB-CSIC / Zenodo (CCDB v4.0, 2019)",
            "license": "CC-BY-NC-4.0",
            "provenance": {
                "url": "https://zenodo.org/doi/10.5281/zenodo.10636010",
                "record": f"https://zenodo.org/records/{ZENODO_RECORD}",
                "file": XLS_NAME,
                "have_locally": False,
                "annotation_method": "manual expert compilation (field studies)",
            },
            "sensors_relevant": ["sentinel2", "sentinel1", "landsat"],
            "classes": [
                {"id": i, "name": name, "description": desc}
                for i, (name, desc) in enumerate(CLASSES)
            ],
            "nodata_value": io.CLASS_NODATA,
            "num_samples": len(selected),
            "class_counts": {
                name: counts.get(i, 0) for i, (name, _d) in enumerate(CLASSES)
            },
            "notes": (
                "Weak single-phenomenon presence label at collapse-caldera center points "
                "(km-scale volcanic collapse depressions; median Max_diameter ~10 km). 1x1 "
                "point segmentation via points.geojson (spec 2a). Presence-only: no "
                "background/negative class (assembly adds negatives). Collapse ages are "
                "geological (Precambrian..Holocene), so this is NOT a change dataset: "
                f"change_time=null, static {STATIC_YEAR} 1-year window; the landform "
                "persists to the present. Subsurface/geological attributes (magma "
                "composition, collapse type, preservation) are not observable at 10 m and "
                "are intentionally not used as classes. Coordinates de-duplicated at "
                f"{COORD_DP} dp; records without valid WGS84 lon/lat dropped."
            ),
        },
    )
    manifest.write_registry_entry(
        SLUG, "completed", task_type="classification", num_samples=len(selected)
    )
    print("done:", dict(counts))


if __name__ == "__main__":
    main()
