"""Process the Geo-Wiki Global Cropland Reference database (See et al. 2017).

Source: PANGAEA doi 10.1594/PANGAEA.873912 (open direct download, no credentials). A
crowdsourcing campaign (Geo-Wiki platform, Sept 2016, ~36k systematically-sampled global
locations) in which volunteers marked cropland grid cells within each frame via visual
interpretation of VHR (Google) imagery. We use ``loc_all_2.txt`` (the updated per
location-and-user table) which gives, per (location, user), the *mean cropland percentage*
``sumcrop`` (0-100) and the location centroid lon/lat.

Label recipe (sparse points -> point table, spec 2a):
- Aggregate ``sumcrop`` across all users per location -> mean cropland fraction (%).
  Rows with an empty ``sumcrop`` (users who skipped the location; the ``_2`` file marks
  these blank on purpose) are excluded from the mean.
- Manifest classes are binary (cropland / non-cropland), so we do BINARY CLASSIFICATION
  with a documented majority threshold: mean cropland % >= 50 -> ``cropland`` (id 0), else
  ``non-cropland`` (id 1). The raw continuous mean fraction is preserved per point as an
  auxiliary ``cropland_fraction`` property so downstream can re-threshold or treat it as a
  regression target.
- Time range: the campaign ran in Sept 2016 (submission timestamps), which is in the
  Sentinel era. Interpreted VHR imagery is of unknown/various years, so per spec 5 we assign
  a representative 1-year window anchored on the 2016 campaign year for every point. Caveat
  noted in the summary.

Balanced to <=1000/class (spec 5); ~2000 points total, well under the 25k cap.
"""

import argparse
import collections
import csv
import statistics
from typing import Any

from olmoearth_pretrain.open_set_segmentation_data import download, io, manifest
from olmoearth_pretrain.open_set_segmentation_data.sampling import balance_by_class

SLUG = "geo_wiki_global_cropland_reference_see_et_al_2017"
NAME = "Geo-Wiki Global Cropland Reference (See et al. 2017)"
PER_CLASS = 1000
CROP_THRESHOLD = 50.0  # mean cropland % >= this -> cropland (majority)
CAMPAIGN_YEAR = 2016  # Geo-Wiki campaign ran Sept 2016 (Sentinel era)

ZIP_URL = "https://store.pangaea.de/Publications/See_2017/loc_all_2.zip"
TABLE_NAME = "loc_all_2.txt"

# Manifest class order -> id. id 0 = cropland, id 1 = non-cropland.
CLASSES = [
    (
        "cropland",
        "Location where the majority (mean crowd-marked cropland fraction >= 50%) of the "
        "frame was interpreted as cropland (cultivated/managed agricultural land) from VHR "
        "imagery in the Geo-Wiki campaign.",
    ),
    (
        "non-cropland",
        "Location where less than 50% of the frame (on average across crowd participants) "
        "was interpreted as cropland; predominantly non-cropland land cover.",
    ),
]


def load_records() -> list[dict[str, Any]]:
    """Download+extract the source table, aggregate per location, return flat records."""
    raw = io.raw_dir(SLUG)
    raw.mkdir(parents=True, exist_ok=True)
    zip_path = raw / "loc_all_2.zip"
    download.download_http(ZIP_URL, zip_path)
    extracted = download.extract_zip(zip_path, raw / "extracted")
    table = extracted / TABLE_NAME

    # mean sumcrop across users per location; keep centroid lon/lat.
    crop_vals: dict[str, list[float]] = collections.defaultdict(list)
    locxy: dict[str, tuple[float, float]] = {}
    with table.open() as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            lid = row["location_id"]
            locxy[lid] = (float(row["loc_cent_X"]), float(row["loc_cent_Y"]))
            sc = row["sumcrop"].strip()
            if sc == "":
                continue
            crop_vals[lid].append(float(sc))

    recs: list[dict[str, Any]] = []
    for lid, (lon, lat) in locxy.items():
        vals = crop_vals.get(lid)
        if not vals:
            # Every user skipped this location -> no cropland judgement; drop it.
            continue
        frac = statistics.mean(vals)
        label = 0 if frac >= CROP_THRESHOLD else 1
        recs.append(
            {
                "location_id": lid,
                "lon": lon,
                "lat": lat,
                "label": label,
                "cropland_fraction": round(frac, 3),
                "n_users": len(vals),
            }
        )
    return recs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=64)
    parser.parse_args()

    io.check_disk()
    manifest.write_registry_entry(SLUG, "in_progress")

    recs = load_records()
    print(f"aggregated {len(recs)} locations with a cropland judgement")
    counts_all = collections.Counter(r["label"] for r in recs)
    print(f"pre-balance class counts (0=cropland,1=non-cropland): {dict(counts_all)}")

    selected = balance_by_class(recs, "label", per_class=PER_CLASS)
    print(f"selected {len(selected)} (<= {PER_CLASS}/class)")

    points = []
    for i, r in enumerate(selected):
        points.append(
            {
                "id": f"{i:06d}",
                "lon": r["lon"],
                "lat": r["lat"],
                "label": r["label"],
                "time_range": io.year_range(CAMPAIGN_YEAR),
                "source_id": f"location_id/{r['location_id']}",
                "cropland_fraction": r["cropland_fraction"],
            }
        )
    io.write_points_table(SLUG, "classification", points)

    counts = collections.Counter(r["label"] for r in selected)
    io.write_dataset_metadata(
        SLUG,
        {
            "dataset": SLUG,
            "name": NAME,
            "task_type": "classification",
            "source": "PANGAEA / Sci Data",
            "license": "CC-BY-3.0",
            "provenance": {
                "url": "https://doi.pangaea.de/10.1594/PANGAEA.873912",
                "have_locally": False,
                "annotation_method": "manual (crowdsourced expert visual interpretation, Geo-Wiki)",
            },
            "sensors_relevant": ["sentinel2", "sentinel1", "landsat"],
            "classes": [
                {"id": i, "name": name, "description": desc}
                for i, (name, desc) in enumerate(CLASSES)
            ],
            "nodata_value": io.CLASS_NODATA,
            "num_samples": len(selected),
            "class_counts": {
                name: counts.get(i, 0) for i, (name, _desc) in enumerate(CLASSES)
            },
            "notes": (
                "1x1 point-segmentation labels (points.geojson). Binary cropland classification "
                f"via majority threshold: mean crowd cropland pct >= {CROP_THRESHOLD:g} -> cropland. "
                "Raw mean fraction kept per point as 'cropland_fraction' (0-100). "
                f"1-year time range anchored on the {CAMPAIGN_YEAR} campaign year for all points; "
                "interpreted VHR imagery is of unknown/various years (caveat). Balanced to "
                f"<= {PER_CLASS}/class."
            ),
        },
    )
    manifest.write_registry_entry(
        SLUG, "completed", task_type="classification", num_samples=len(selected)
    )
    print("done")


if __name__ == "__main__":
    main()
