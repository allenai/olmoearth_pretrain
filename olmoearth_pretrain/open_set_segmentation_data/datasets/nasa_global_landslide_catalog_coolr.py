"""Process the NASA Global Landslide Catalog (GLC / COOLR) into a sparse-point table.

Source: NASA GSFC "Global Landslide Catalog Export" (public domain), NASA Open Data
Portal, https://landslides.nasa.gov. Distributed as a single CSV of 11,033 documented
landslide events, one row per event, with (among ~31 attributes):

  * ``longitude`` / ``latitude`` -- point location (decimal degrees; all rows populated).
  * ``event_date`` -- event date+time, e.g. ``05/19/2017 08:14:00 PM`` (precise to the day).
  * ``location_accuracy`` -- how well the point is located: ``exact``, ``1km``, ``5km``,
    ``10km``, ``25km``, ``50km``, ``100km``, ``250km``, ``unknown``.
  * ``landslide_category`` -- type (landslide, mudslide, rock_fall, debris_flow, complex, ...).
  * ``landslide_trigger`` / ``landslide_size`` / ``fatality_count`` / country fields, etc.

Task: sparse-point CLASSIFICATION of each landslide location by ``landslide_category``
(spec 2a -> one ``points.geojson``, not per-sample GeoTIFFs). A landslide is a dated EVENT,
so each point is a CHANGE label: ``change_time`` = the event date, kept as the reference for
building the windows. Instead of a single centered window, we emit two adjacent six-month
windows split at ``change_time``: ``pre_time_range`` (the <=183 days immediately before) and
``post_time_range`` (the <=183 days immediately after), with ``time_range`` set to null
(built via ``io.pre_post_time_ranges(change_time, ...)``), so pretraining pairs a "before"
image stack with an "after" stack -- seeing the slope before and after the failure -- and
probes on their difference.

FILTERS APPLIED (documented in the summary):
  * Timing precision (spec 5, hard): ``event_date`` is precise to the day for every row, so
    every kept event has a confident change_time (<< 1-2 month requirement).
  * Location accuracy (hard): only ``exact`` and ``1km`` points are kept. Coarser accuracies
    (5km/10km/25km/50km/100km/250km/unknown) place the point tens to hundreds of 10 m pixels
    from the true failure and are NOT usable on the S2 grid, so they are dropped.
  * Sentinel era (spec, hard): only events with year >= 2016 are kept (the GLC export ends
    2017-09; pre-2016 events are dropped).
  Net: 1,169 of 11,033 events pass all three filters.

Positive-only presence points (there is no "no-landslide" class); per spec 5 we do NOT
fabricate negatives -- the assembly step supplies them from other datasets.

Run: python3 -m olmoearth_pretrain.open_set_segmentation_data.datasets.nasa_global_landslide_catalog_coolr
Idempotent: rewrites points.geojson + metadata.json from the (cached) raw CSV download.
"""

import argparse
from collections import Counter
from datetime import UTC

import pandas as pd

from olmoearth_pretrain.open_set_segmentation_data import download, io, manifest

SLUG = "nasa_global_landslide_catalog_coolr"
NAME = "NASA Global Landslide Catalog / COOLR"
CSV_URL = (
    "https://data.nasa.gov/docs/legacy/Global_Landslide_Catalog_Export/"
    "Global_Landslide_Catalog_Export_rows.csv"
)
CSV_NAME = "glc_export.csv"
MIN_YEAR = 2016  # Sentinel era
# Location accuracies precise enough to place a point on the 10 m S2 grid. Coarser codes
# put the point up to tens/hundreds of km (thousands of px) from the failure -> rejected.
KEEP_ACCURACY = {"exact", "1km"}
PER_CLASS = 1000

# Canonical landslide-category classes: stable ids ordered by full-catalog frequency, with
# definitions. Covers every ``landslide_category`` value in the GLC (incl. ones that survive
# no filter) so ids never shift; unused-in-selection categories simply do not appear.
CLASSES = [
    (
        "landslide",
        "Generic landslide / slope failure not assigned a more specific type.",
    ),
    (
        "mudslide",
        "Rapid flow of water-saturated fine debris and mud (mudflow / mudslide).",
    ),
    (
        "rock_fall",
        "Detachment and free-fall / bounce of rock from a steep slope or cliff.",
    ),
    ("complex", "Compound failure combining two or more movement types."),
    ("debris_flow", "Fast channelized flow of saturated coarse debris (debris flow)."),
    ("other", "Documented mass movement not matching the standard category set."),
    ("unknown", "Landslide of unreported / undetermined movement type."),
    ("riverbank_collapse", "Failure/collapse of a river or stream bank."),
    ("snow_avalanche", "Rapid downslope flow of snow (snow avalanche)."),
    ("translational_slide", "Slide moving along a roughly planar rupture surface."),
    ("lahar", "Volcanic mudflow/debris flow of pyroclastic material and water."),
    ("earth_flow", "Slow-to-rapid flow of fine-grained soil/earth (earthflow)."),
    ("creep", "Slow, continuous downslope deformation of soil/regolith."),
    ("topple", "Forward rotation/overturning of a rock or soil mass about a pivot."),
]
NAME_TO_ID = {name: i for i, (name, _d) in enumerate(CLASSES)}
VALID_NAMES = set(NAME_TO_ID)


def normalize_category(raw) -> str:
    """Map a raw landslide_category value to a canonical class name."""
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return "unknown"
    v = str(raw).strip().lower().replace(" ", "_").replace("-", "_")
    if v in ("", "nan"):
        return "unknown"
    if v in VALID_NAMES:
        return v
    # Fold a few spelling variants.
    if v in ("rockfall", "rock_slide", "rockslide"):
        return "rock_fall"
    if v in ("mudflow",):
        return "mudslide"
    return "other"


def load_records(csv_path: str) -> tuple[list[dict], dict]:
    """Read the GLC CSV; return (kept records, drop-counter stats)."""
    df = pd.read_csv(csv_path)
    stats = {"total": len(df)}

    d = pd.to_datetime(df["event_date"], errors="coerce")
    df["_dt"] = d
    df["_lon"] = pd.to_numeric(df["longitude"], errors="coerce")
    df["_lat"] = pd.to_numeric(df["latitude"], errors="coerce")
    df["_acc"] = df["location_accuracy"].astype("string").str.strip().str.lower()

    coords_ok = (
        df["_lon"].between(-180, 180)
        & df["_lat"].between(-90, 90)
        & df["_lon"].notna()
        & df["_lat"].notna()
    )
    date_ok = df["_dt"].notna()
    year_ok = df["_dt"].dt.year >= MIN_YEAR
    acc_ok = df["_acc"].isin(KEEP_ACCURACY)

    stats["no_coords"] = int((~coords_ok).sum())
    stats["no_date"] = int((~date_ok).sum())
    stats["pre_2016"] = int((date_ok & ~year_ok).sum())
    stats["coarse_accuracy"] = int((~acc_ok).sum())

    keep = coords_ok & date_ok & year_ok & acc_ok
    stats["kept"] = int(keep.sum())

    recs: list[dict] = []
    for _, row in df[keep].iterrows():
        dt = row["_dt"].to_pydatetime().replace(tzinfo=UTC)
        recs.append(
            {
                "lon": float(row["_lon"]),
                "lat": float(row["_lat"]),
                "date": dt,
                "year": int(dt.year),
                "category": normalize_category(row["landslide_category"]),
                "accuracy": str(row["_acc"]),
                "src_id": str(row.get("event_id", "")),
            }
        )
    return recs, stats


def main() -> None:
    argparse.ArgumentParser().parse_args()

    io.check_disk()
    manifest.write_registry_entry(SLUG, "in_progress")

    raw = io.raw_dir(SLUG)
    raw.mkdir(parents=True, exist_ok=True)
    download.download_http(CSV_URL, raw / CSV_NAME)
    with (raw / "SOURCE.txt").open("w") as f:
        f.write(
            "NASA Global Landslide Catalog (GLC / COOLR), NASA GSFC. Public domain.\n"
            "https://landslides.nasa.gov  |  https://data.nasa.gov (Global Landslide "
            "Catalog Export)\n"
            f"{CSV_NAME}: {CSV_URL}\n"
            "11,033 documented landslide events (through 2017-09). Processed as sparse "
            f"points; kept events with year>={MIN_YEAR}, location_accuracy in "
            f"{sorted(KEEP_ACCURACY)}, and valid coordinates.\n"
        )

    recs, stats = load_records(str((raw / CSV_NAME).path))
    print(f"filter stats: {stats}")

    from olmoearth_pretrain.open_set_segmentation_data.sampling import balance_by_class

    selected = balance_by_class(recs, "category", per_class=PER_CLASS)
    print(f"selected {len(selected)} points (<= {PER_CLASS}/class)")

    points = []
    for i, r in enumerate(sorted(selected, key=lambda x: (x["date"], x["src_id"]))):
        change_time = r["date"]
        pre_range, post_range = io.pre_post_time_ranges(change_time)
        points.append(
            {
                "id": f"{i:06d}",
                "lon": r["lon"],
                "lat": r["lat"],
                "label": NAME_TO_ID[r["category"]],
                "time_range": (pre_range[0], post_range[1]),
                "pre_time_range": pre_range,
                "post_time_range": post_range,
                "change_time": change_time,
                "source_id": r["src_id"],
                "location_accuracy": r["accuracy"],
            }
        )
    io.write_points_table(SLUG, "classification", points)

    counts = Counter(r["category"] for r in selected)
    acc_counts = dict(sorted(Counter(r["accuracy"] for r in selected).items()))
    year_counts = dict(sorted(Counter(r["year"] for r in selected).items()))
    io.write_dataset_metadata(
        SLUG,
        {
            "dataset": SLUG,
            "name": NAME,
            "task_type": "classification",
            "source": "NASA GSFC (Global Landslide Catalog / COOLR)",
            "license": "public domain",
            "provenance": {
                "url": "https://landslides.nasa.gov",
                "have_locally": False,
                "annotation_method": (
                    "manual report compilation from media/scientific/government reports "
                    "+ citizen science (COOLR)"
                ),
            },
            "sensors_relevant": ["sentinel2", "sentinel1", "landsat"],
            "classes": [
                {"id": i, "name": name, "description": desc}
                for i, (name, desc) in enumerate(CLASSES)
            ],
            "nodata_value": io.CLASS_NODATA,
            "num_samples": len(points),
            "class_counts": {name: counts.get(name, 0) for name, _ in CLASSES},
            "accuracy_counts": acc_counts,
            "year_counts": year_counts,
            "filter_stats": stats,
            "notes": (
                "Sparse-point classification of documented landslide events by "
                "landslide_category. Each landslide is a dated EVENT -> change label: "
                "change_time = event_date (precise to the day for every row) and time_range "
                "= a 1-year window centered on it. FILTERS: kept only location_accuracy in "
                f"{sorted(KEEP_ACCURACY)} (coarser 5-250 km / unknown points place the point "
                "tens-hundreds of 10 m px off and were dropped: "
                f"{stats['coarse_accuracy']} rows); kept only year>={MIN_YEAR} "
                f"({stats['pre_2016']} pre-2016 rows dropped, and the export ends 2017-09). "
                f"All {stats['total']} rows have valid coordinates. Net kept "
                f"{stats['kept']}. Positive-only presence points -- no negative class is "
                "fabricated (assembly supplies negatives). Sparse categories (e.g. topple, "
                "earth_flow) are retained per spec; downstream assembly drops too-small ones."
            ),
        },
    )
    print(f"class counts: {dict(counts)}")
    print(f"accuracy: {acc_counts}")
    print(f"years: {year_counts}")
    manifest.write_registry_entry(
        SLUG, "completed", task_type="classification", num_samples=len(points)
    )
    print("done")


if __name__ == "__main__":
    main()
