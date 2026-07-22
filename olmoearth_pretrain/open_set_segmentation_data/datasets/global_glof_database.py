"""Process the Global GLOF Database V3.0 into a sparse-point label table.

Source: "Glacier Lake Outburst Flood Database V3.0" (Lutzmann/Veh et al., ESSD),
Zenodo record 7330345 (CC-BY-4.0), https://doi.org/10.5281/zenodo.7330345. The record
is a single OpenDocument spreadsheet ``glofdatabase_V3.ods`` (plus a parameter readme)
with one sheet per glacier region (Andes, European Alps, NW North America, High Mountain
Asia, Scandinavia, Other, Iceland, Greenland). Each row is one documented glacier lake
outburst flood (GLOF) event with, among ~57 attributes:

  * ``Longitude`` / ``Latitude`` -- point location of the source glacier lake (decimal deg).
  * ``Date`` / ``Date_Min`` / ``Date_Max`` -- event date(s) (YYYY-MM-DD, or year-only).
  * ``Lake_type`` -- the impounding dam type (ice, moraine, bedrock, water pocket, ...).

IMPORTANT provenance / scope notes (judgment calls, see summary):
  * The manifest bills this as "points + polygons" with "manually mapped lake polygons".
    The V3.0 Zenodo release distributes NO polygon geometry -- only scalar
    ``Lake_area_before/after`` (m2) and ``Perimeter`` values derived from polygons that
    are not published. So there is nothing larger-than-a-pixel to rasterize: this is a
    PURE SPARSE-POINT dataset (spec 2a) -> one ``points.geojson``, not per-sample GeoTIFFs.
  * The event dates span 1100-2022. Per the Sentinel-era rule we keep ONLY events with a
    usable year >= 2016 (249 events with coordinates); the ~2900 pre-2016 events are dropped.
  * Positive-only presence points (there is no "no-GLOF" class); per spec 5 we do NOT
    fabricate negatives -- the assembly step supplies them from other datasets.

Task: sparse-point CLASSIFICATION of each GLOF location by dam type. A GLOF is a dated
EVENT (sudden lake drainage), so each point is a change label. When a full YYYY-MM-DD is
known, ``change_time`` = the event date and the sample gets two adjacent six-month windows
split exactly at ``change_time`` (via ``io.pre_post_time_ranges``): ``pre_time_range`` is the
~6 months (<=183 days) immediately before the event and ``post_time_range`` is the ~6 months
(<=183 days) immediately after, with ``time_range`` = null. Year-only events instead keep a
single 1-year ``time_range`` for the calendar year with ``change_time`` = null and no
pre/post windows. Pretraining pairs the "before" image stack with the "after" stack and
probes on their difference, letting the model see the lake before/after drainage.

Run: python3 -m olmoearth_pretrain.open_set_segmentation_data.datasets.global_glof_database
Idempotent: rewrites points.geojson + metadata.json from the (cached) raw download.
"""

import argparse
import re
from collections import Counter
from datetime import UTC, datetime

import pandas as pd

from olmoearth_pretrain.open_set_segmentation_data import download, io, manifest

SLUG = "global_glof_database"
NAME = "Global GLOF Database"
ZENODO_RECORD = "7330345"
ODS_NAME = "glofdatabase_V3.ods"
README_NAME = "Parameter_Readme.ods"
MIN_YEAR = 2016  # Sentinel era
PER_CLASS = 1000

# Canonical dam-type classes (ids stable, ordered by post-2016 frequency), with definitions.
CLASSES = [
    (
        "ice_dammed",
        "Lake impounded by a glacier ice dam (marginal/proglacial ice-dammed lake).",
    ),
    (
        "ice_volcanic",
        "Ice-dammed lake in a volcanic setting draining as a joekulhlaup (ice - volc).",
    ),
    (
        "moraine_dammed",
        "Lake impounded by a moraine dam (proglacial moraine-dammed lake).",
    ),
    (
        "water_pocket",
        "Englacial/subglacial water pocket releasing a flood (no distinct surface lake dam).",
    ),
    (
        "combined",
        "Mixed/combined impounding materials (e.g. ice/moraine, moraine/bedrock).",
    ),
    ("supraglacial", "Supraglacial lake on the glacier surface."),
    ("bedrock_dammed", "Lake impounded by a bedrock dam / bedrock threshold."),
    ("subglacial", "Subglacial lake / reservoir draining beneath the glacier."),
    ("volcanic", "Volcanically-driven drainage not otherwise ice/moraine classified."),
    ("snow", "Snow-dammed lake."),
    ("other", "Other impounding material (e.g. colluvial material)."),
    ("unknown", "Impounding dam type not reported / unknown."),
]
NAME_TO_ID = {name: i for i, (name, _d) in enumerate(CLASSES)}

SHEETS = [
    "Andes",
    "European Alps",
    "NW North America",
    "High Mountain Asia",
    "Scandinavia",
    "Other",
    "Iceland",
    "Greenland",
]


def normalize_dam_type(raw) -> str:
    """Map a raw Lake_type string to one of the canonical dam-type class names."""
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return "unknown"
    v = str(raw).strip().lower().replace("–", "-")  # en-dash -> hyphen
    if v in ("", "unknown"):
        return "unknown"
    if "/" in v:  # mixed materials (ice/moraine, moraine/bedrock, bedrock/ice, ...)
        return "combined"
    if "water pocket" in v:
        return "water_pocket"
    if v.startswith("ice") and "volc" in v:
        return "ice_volcanic"
    if v.startswith("ice"):
        return "ice_dammed"
    if "moraine" in v:
        return "moraine_dammed"
    if "bedrock" in v:
        return "bedrock_dammed"
    if "supraglacial" in v:
        return "supraglacial"
    if "subglacial" in v:
        return "subglacial"
    if "combined" in v:
        return "combined"
    if v.startswith("volc"):
        return "volcanic"
    if "snow" in v:
        return "snow"
    if v.startswith("other"):
        return "other"
    return "unknown"


def _is_number(x) -> bool:
    try:
        float(x)
        return True
    except (TypeError, ValueError):
        return False


def _parse_full_date(v) -> datetime | None:
    """Return a UTC datetime if v is a full YYYY-MM-DD (or Timestamp), else None."""
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return None
    if isinstance(v, (pd.Timestamp, datetime)):
        return datetime(v.year, v.month, v.day, tzinfo=UTC)
    m = re.match(r"^\s*(\d{4})-(\d{2})-(\d{2})", str(v))
    if not m:
        return None
    y, mo, d = int(m.group(1)), int(m.group(2)), int(m.group(3))
    try:
        return datetime(y, mo, d, tzinfo=UTC)
    except ValueError:
        return None


def _best_year(row) -> int | None:
    for col in ("Date", "Date_Min", "Date_Max"):
        v = row[col]
        if v is None or (isinstance(v, float) and pd.isna(v)):
            continue
        if isinstance(v, (pd.Timestamp, datetime)):
            return int(v.year)
        m = re.match(r"^\s*(\d{4})", str(v))
        if m:
            return int(m.group(1))
    return None


def load_records(ods_path: str) -> list[dict]:
    """Read all region sheets, filter to post-2016 events with valid coordinates."""
    frames = []
    for sheet in SHEETS:
        df = pd.read_excel(ods_path, sheet_name=sheet, engine="odf")
        df["__sheet"] = sheet
        frames.append(df)
    alld = pd.concat(frames, ignore_index=True)
    # Drop the two secondary header rows embedded per sheet (non-numeric ID).
    alld = alld[alld["ID"].apply(_is_number)].copy()
    alld["Longitude"] = pd.to_numeric(alld["Longitude"], errors="coerce")
    alld["Latitude"] = pd.to_numeric(alld["Latitude"], errors="coerce")

    recs: list[dict] = []
    for _, row in alld.iterrows():
        lon, lat = row["Longitude"], row["Latitude"]
        if pd.isna(lon) or pd.isna(lat):
            continue
        if not (-180 <= lon <= 180 and -90 <= lat <= 90):
            continue
        year = _best_year(row)
        if year is None or year < MIN_YEAR:
            continue
        recs.append(
            {
                "lon": float(lon),
                "lat": float(lat),
                "year": year,
                "date": _parse_full_date(row["Date"]),
                "dam_type": normalize_dam_type(row["Lake_type"]),
                "sheet": row["__sheet"],
                "src_id": str(row["ID"]),
            }
        )
    return recs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.parse_args()

    io.check_disk()
    manifest.write_registry_entry(SLUG, "in_progress")

    raw = io.raw_dir(SLUG)
    raw.mkdir(parents=True, exist_ok=True)
    for fname in (ODS_NAME, README_NAME):
        url = f"https://zenodo.org/api/records/{ZENODO_RECORD}/files/{fname}/content"
        download.download_http(url, raw / fname)
    with (raw / "SOURCE.txt").open("w") as f:
        f.write(
            "Glacier Lake Outburst Flood Database V3.0 (ESSD), Zenodo record 7330345 "
            "(CC-BY-4.0). https://doi.org/10.5281/zenodo.7330345\n"
            f"{ODS_NAME}: 8 regional sheets, ~3150 GLOF events (~57 attributes each).\n"
            f"{README_NAME}: parameter definitions.\n"
            "No polygon geometry is published (only scalar Lake_area/Perimeter values); "
            f"processed as sparse points, events with year>={MIN_YEAR} and coordinates only.\n"
        )

    recs = load_records(str((raw / ODS_NAME).path))
    print(f"post-2016 GLOF events with coordinates: {len(recs)}")

    # Balance per class (per_class=1000; all classes are well under, so this keeps all).
    from olmoearth_pretrain.open_set_segmentation_data.sampling import balance_by_class

    selected = balance_by_class(recs, "dam_type", per_class=PER_CLASS)
    print(f"selected {len(selected)} points (<= {PER_CLASS}/class)")

    points = []
    n_change = 0
    for i, r in enumerate(sorted(selected, key=lambda x: (x["year"], x["src_id"]))):
        cid = NAME_TO_ID[r["dam_type"]]
        if r["date"] is not None:
            change_time = r["date"]
            pre_range, post_range = io.pre_post_time_ranges(change_time)
            time_range = (pre_range[0], post_range[1])
            n_change += 1
            point = {
                "id": f"{i:06d}",
                "lon": r["lon"],
                "lat": r["lat"],
                "label": cid,
                "time_range": time_range,
                "pre_time_range": pre_range,
                "post_time_range": post_range,
                "change_time": change_time,
                "source_id": f"{r['sheet']}/{r['src_id']}",
            }
        else:
            change_time = None
            time_range = io.year_range(r["year"])
            point = {
                "id": f"{i:06d}",
                "lon": r["lon"],
                "lat": r["lat"],
                "label": cid,
                "time_range": time_range,
                "change_time": change_time,
                "source_id": f"{r['sheet']}/{r['src_id']}",
            }
        points.append(point)
    io.write_points_table(SLUG, "classification", points)

    counts = Counter(r["dam_type"] for r in selected)
    year_counts = dict(sorted(Counter(r["year"] for r in selected).items()))
    io.write_dataset_metadata(
        SLUG,
        {
            "dataset": SLUG,
            "name": NAME,
            "task_type": "classification",
            "source": "Zenodo / ESSD (Glacier Lake Outburst Flood Database V3.0)",
            "license": "CC-BY-4.0",
            "provenance": {
                "url": "https://doi.org/10.5281/zenodo.7330345",
                "have_locally": False,
                "annotation_method": (
                    "manual literature compilation + satellite photointerpretation of "
                    "source glacier lakes"
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
            "year_counts": year_counts,
            "n_with_change_time": n_change,
            "notes": (
                "Sparse-point classification of documented glacier lake outburst floods "
                "(GLOFs) by impounding dam type (Lake_type). Source is the V3.0 ODS "
                "spreadsheet; NO lake polygons are published in this release (only scalar "
                "Lake_area/Perimeter), so despite the manifest's 'points + polygons' label "
                "this is processed as points only. Kept events with year>=2016 and valid "
                "coordinates (249 of ~3150; the rest are pre-Sentinel or lack coordinates). "
                "Each GLOF is a dated event -> change label: change_time = event date when a "
                f"full YYYY-MM-DD is known ({n_change} of {len(points)}), else null; "
                "time_range = 1-year window centered on the event (or the calendar year for "
                "year-only events). Positive-only presence points -- no negative class is "
                "fabricated (assembly supplies negatives). Dam-type variants normalized: "
                "'ice - volc' -> ice_volcanic, slashed mixes (ice/moraine, ...) -> combined."
            ),
        },
    )
    print(f"class counts: {dict(counts)}")
    print(f"years: {year_counts}")
    print(f"points with precise change_time: {n_change}/{len(points)}")
    manifest.write_registry_entry(
        SLUG, "completed", task_type="classification", num_samples=len(points)
    )
    print("done")


if __name__ == "__main__":
    main()
