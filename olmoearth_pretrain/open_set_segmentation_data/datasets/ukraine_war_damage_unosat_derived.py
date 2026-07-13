"""Ukraine War Damage (UNOSAT-derived) -> open-set-segmentation change point labels.

Source: ETH Zurich / prs-eth "ukraine-damage-mapping-tool" repository
(https://github.com/prs-eth/ukraine-damage-mapping-tool, MIT-licensed tool code; labels
derived from UNOSAT VHR damage assessments). The repo ships pre-processed UNOSAT
Comprehensive Damage Assessment (CDA) points for 18 Ukrainian AOIs in
``data/unosat_labels.geojson``: 18,686 Point features, each with a UNOSAT analyst damage
grade, the AOI, the city, and the DATE of the post-event VHR image the assessment was made
from (day-precise, all in 2022). These are the labels the paper pairs with Sentinel-1 time
series to detect the moment of destruction.

Why this is a CHANGE dataset (unlike the sibling ``unosat_conflict_damage_assessments``):
each point carries a specific day-precise post-event image date in 2022, and the damage it
records occurred during the war (after 2022-02-24) within weeks of that dated image. So the
change date is known to well within ~1-2 months (spec S5 timing rule): we set
``change_time`` = the assessment/image date and center a 1-year window on it. (The HDX-sourced
sibling could not, because those comprehensive products compare against baselines 1-3 years
earlier, so it recast to static presence/state instead.)

Encoding: sparse point segmentation -> one dataset-wide GeoJSON point table (spec S2a), one
Point feature per building-damage location, ``properties.label`` = damage-grade class id,
per-feature ``change_time`` + centered 1-year ``time_range``. Balanced to <=1000 per class
(spec S5).

10 m observability caveat: an individual building is ~1 pixel at 10 m, so a single
destroyed/damaged structure is near the resolution limit. Destroyed / severe damage of
larger structures, and the dense clusters of points in besieged cities (Mariupol etc.), are
the observable signal; finer grades (moderate/possible) of isolated buildings likely are
not. Grades are kept as a unified 4-class scheme so downstream can select; the limitation is
noted in the summary.

Run: python3 -m olmoearth_pretrain.open_set_segmentation_data.datasets.ukraine_war_damage_unosat_derived
"""

import argparse
import json
from collections import Counter
from datetime import UTC, datetime, timedelta

from olmoearth_pretrain.open_set_segmentation_data import download, io, manifest
from olmoearth_pretrain.open_set_segmentation_data.sampling import balance_by_class

SLUG = "ukraine_war_damage_unosat_derived"

BASE_URL = (
    "https://raw.githubusercontent.com/prs-eth/ukraine-damage-mapping-tool/main/data"
)
LABELS_FILE = "unosat_labels.geojson"
AOIS_FILE = "unosat_aois.geojson"

PER_CLASS = 1000
HALF_WINDOW_DAYS = 180  # +/-180d -> 360-day window centered on change_time (<= cap)
MIN_YEAR = 2016  # Sentinel era (all labels are 2022, so nothing is filtered).

# Unified UNOSAT building-damage grades -> class ids (most severe first). Numeric domain of
# ``damage`` in the source: 1=Destroyed, 2=Severe, 3=Moderate, 4=Possible are the building
# damage grades we keep. Codes 5/6/7/15 (no-visible-damage / other non-building-damage
# categories; 477 pts total) are ambiguous and dropped.
CLASSES = [
    (
        "destroyed",
        "Structure collapsed / largely reduced to rubble; footprint no longer intact.",
    ),
    (
        "severe_damage",
        "Major structural damage (partial collapse, roof/walls gone) but footprint partly standing.",
    ),
    (
        "moderate_damage",
        "Visible partial damage (roof holes, blast damage) with the structure largely standing.",
    ),
    (
        "possible_damage",
        "Possible / uncertain damage flagged by the analyst (lower confidence).",
    ),
]
GRADE_TO_ID = {1: 0, 2: 1, 3: 2, 4: 3}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.parse_args()

    io.check_disk()
    manifest.write_registry_entry(SLUG, "in_progress")

    # 1. Download the label-only GeoJSONs (small; ~7 MB + ~0.2 MB) to raw/.
    raw = io.raw_dir(SLUG)
    raw.mkdir(parents=True, exist_ok=True)
    labels_path = download.download_http(f"{BASE_URL}/{LABELS_FILE}", raw / LABELS_FILE)
    download.download_http(f"{BASE_URL}/{AOIS_FILE}", raw / AOIS_FILE)

    with labels_path.open("r") as f:
        feats = json.load(f)["features"]
    print(f"loaded {len(feats)} UNOSAT damage points")

    # 2. Build flat records: one per point with a known building-damage grade.
    records = []
    grade_all = Counter()
    for feat in feats:
        p = feat["properties"]
        grade = p["damage"]
        grade_all[grade] += 1
        if grade not in GRADE_TO_ID:
            continue
        date = datetime.strptime(p["date"][:10], "%Y-%m-%d").replace(tzinfo=UTC)
        if date.year < MIN_YEAR:
            continue
        lon, lat = feat["geometry"]["coordinates"]
        records.append(
            {
                "lon": float(lon),
                "lat": float(lat),
                "label": GRADE_TO_ID[grade],
                "change_time": date,
                "source_id": f"{p['aoi']}/{p['unosat_id']}",
            }
        )
    print(f"grade distribution (all): {dict(sorted(grade_all.items()))}")
    print(f"kept {len(records)} building-damage points (grades 1-4, {MIN_YEAR}+)")

    # 3. Balance to <=1000 per class (spec S5 classification cap).
    selected = balance_by_class(records, "label", per_class=PER_CLASS)
    print(f"selected {len(selected)} (<= {PER_CLASS}/class)")

    # 4. Point table: per-feature change_time + centered 1-year window (spec S5 change).
    points = []
    for i, r in enumerate(selected):
        ct = r["change_time"]
        points.append(
            {
                "id": f"{i:06d}",
                "lon": r["lon"],
                "lat": r["lat"],
                "label": r["label"],
                "time_range": (
                    ct - timedelta(days=HALF_WINDOW_DAYS),
                    ct + timedelta(days=HALF_WINDOW_DAYS),
                ),
                "change_time": ct,
                "source_id": r["source_id"],
            }
        )
    io.write_points_table(SLUG, "classification", points)

    counts = Counter(r["label"] for r in selected)
    io.write_dataset_metadata(
        SLUG,
        {
            "dataset": SLUG,
            "name": "Ukraine War Damage (UNOSAT-derived)",
            "task_type": "classification",
            "source": "GitHub (prs-eth) / UNOSAT",
            "license": "MIT (tool code); UNOSAT terms (labels)",
            "provenance": {
                "url": "https://github.com/prs-eth/ukraine-damage-mapping-tool",
                "have_locally": False,
                "annotation_method": "manual (UNOSAT analyst) VHR damage assessment",
            },
            "sensors_relevant": ["sentinel1", "sentinel2", "landsat"],
            "classes": [
                {"id": i, "name": name, "description": desc}
                for i, (name, desc) in enumerate(CLASSES)
            ],
            "nodata_value": io.CLASS_NODATA,
            "num_samples": len(selected),
            "class_counts": {
                name: counts.get(i, 0) for i, (name, _) in enumerate(CLASSES)
            },
            "notes": (
                "Sparse 1x1 change point labels for building war-damage in Ukraine (18 AOIs, "
                "all 2022). change_time = UNOSAT post-event VHR image date (day-precise); "
                "time_range = 360-day window centered on change_time. Positive-only (no intact "
                "class); assembly supplies negatives. 10 m caveat: individual buildings are "
                "~1 px, so destroyed/severe + dense clusters are the observable signal, finer "
                "grades of isolated buildings may not be. Grades 5/6/7/15 (477 pts) dropped as "
                "non-building-damage / ambiguous."
            ),
        },
    )
    manifest.write_registry_entry(
        SLUG, "completed", task_type="classification", num_samples=len(selected)
    )
    print("done")


if __name__ == "__main__":
    main()
