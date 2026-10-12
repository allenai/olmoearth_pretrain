"""Atlas of Hillforts of Britain and Ireland -> open-set-segmentation point labels.

Source: the Univ. Edinburgh / Univ. Oxford "Atlas of Hillforts of Britain and Ireland"
(4,147 Iron/Bronze Age hillfort sites), published as a public Esri Feature Service
(hillforts-oxforduni.hub.arcgis.com). Each record is a single point with WGS84
lon/lat and an expert "Reliability of Interpretation" attribute.

Suitability: hillforts are large enclosed earthwork ramparts (typically 1-20+ ha, i.e.
~100 m to >450 m across). At Sentinel-2/Landsat 10-30 m the individual rampart lines are
subtle, but the overall enclosure footprint and its persistent topographic / vegetation
signature are plausibly detectable, so we keep this as a WEAK single-phenomenon PRESENCE
label at the site point. Points carry a class, so we write the dataset-wide point table
(spec 2a), not per-point GeoTIFFs.

Classes (from Reliability of Interpretation):
  0 hillfort            <- "Confirmed"   (site confirmed as a hillfort)
  1 possible hillfort   <- "Unconfirmed" (interpretation as a hillfort not confirmed)
"Irreconciled issues" records (conflicting source data) are dropped as ambiguous.

Persistent/static heritage sites -> a representative 1-year window in the Sentinel era.
"""

import argparse
import json
import urllib.request
from collections import Counter
from typing import Any

from olmoearth_pretrain.open_set_segmentation_data import io, manifest
from olmoearth_pretrain.open_set_segmentation_data.sampling import balance_by_class

SLUG = "atlas_of_hillforts_of_britain_and_ireland"
NAME = "Atlas of Hillforts of Britain and Ireland"
FEATURESERVER = (
    "https://services1.arcgis.com/PTDJItLzolyiewT6/arcgis/rest/services/"
    "Atlas_of_Hillforts/FeatureServer/0"
)
PER_CLASS = 1000
STATIC_YEAR = 2020  # representative Sentinel-era year for these persistent sites

# Reliability of Interpretation -> (class_id, class_name)
REL_TO_CLASS = {
    "Confirmed": 0,
    "Unconfirmed": 1,
}
CLASSES = [
    (
        "hillfort",
        "Site whose interpretation as a hillfort is Confirmed in the Atlas of Hillforts "
        "(large enclosed earthwork/rampart, typically 1-20+ ha).",
    ),
    (
        "possible hillfort",
        "Site recorded in the Atlas of Hillforts but whose interpretation as a hillfort "
        "is Unconfirmed (possible hillfort).",
    ),
]


def download_raw() -> Any:
    """Download all Atlas records as one GeoJSON FeatureCollection to raw_dir (atomic)."""
    raw = io.raw_dir(SLUG)
    raw.mkdir(parents=True, exist_ok=True)
    dst = raw / "hillforts.geojson"
    if not dst.exists():
        feats: list[dict[str, Any]] = []
        offset = 0
        while True:
            url = (
                f"{FEATURESERVER}/query?where=1%3D1&outFields=*&returnGeometry=true"
                f"&outSR=4326&f=geojson&resultOffset={offset}&resultRecordCount=2000"
            )
            with urllib.request.urlopen(url) as r:
                batch = json.load(r)
            fs = batch.get("features", [])
            feats += fs
            if len(fs) < 2000:
                break
            offset += 2000
        tmp = raw / "hillforts.geojson.tmp"
        with tmp.open("w") as f:
            json.dump({"type": "FeatureCollection", "features": feats}, f)
        tmp.rename(dst)
    with dst.open() as f:
        return json.load(f)["features"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.parse_args()

    io.check_disk()
    manifest.write_registry_entry(SLUG, "in_progress")

    feats = download_raw()
    print(f"downloaded {len(feats)} hillfort records")

    records: list[dict[str, Any]] = []
    dropped = Counter()
    for feat in feats:
        p = feat["properties"]
        lon, lat = p.get("Longitude"), p.get("Latitude")
        rel = p.get("Reliability_of_Interpretation")
        if lon is None or lat is None:
            dropped["no_coords"] += 1
            continue
        if rel not in REL_TO_CLASS:
            dropped[f"rel:{rel}"] += 1
            continue
        records.append(
            {
                "lon": float(lon),
                "lat": float(lat),
                "label": REL_TO_CLASS[rel],
                "source_id": p.get("Atlas_ID") or str(p.get("Atlas_Number")),
            }
        )
    print(f"usable {len(records)}; dropped {dict(dropped)}")

    selected = balance_by_class(records, "label", per_class=PER_CLASS)
    print(f"selected {len(selected)} (<= {PER_CLASS}/class)")

    time_range = io.year_range(STATIC_YEAR)
    points = []
    for i, r in enumerate(selected):
        points.append(
            {
                "id": f"{i:06d}",
                "lon": r["lon"],
                "lat": r["lat"],
                "label": r["label"],
                "time_range": time_range,
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
            "source": "Univ. Edinburgh/Oxford (Atlas of Hillforts of Britain and Ireland)",
            "license": "open for research",
            "provenance": {
                "url": "https://hillforts.arch.ox.ac.uk/",
                "service": FEATURESERVER,
                "have_locally": False,
                "annotation_method": "expert compilation",
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
                "Weak single-phenomenon presence label at heritage site points "
                "(hillfort enclosures ~1-20+ ha). 1x1 point segmentation via points.json "
                f"(spec 2a). Persistent static sites -> fixed {STATIC_YEAR} 1-year window. "
                "Classes from Reliability of Interpretation (Confirmed=hillfort, "
                "Unconfirmed=possible hillfort); 'Irreconciled issues' dropped. "
                "Presence-only: no explicit negative/background class."
            ),
        },
    )
    manifest.write_registry_entry(
        SLUG, "completed", task_type="classification", num_samples=len(selected)
    )
    print("done:", dict(counts))


if __name__ == "__main__":
    main()
