"""Process HABSOS (Harmful Algal BloomS Observing System) into open-set-segmentation labels.

Source: NOAA NCEI HABSOS, an in-situ georeferenced marine HAB observation database
(primarily Karenia brevis red tide) with per-sample cell counts and a NOAA bloom-abundance
category. Public domain (NOAA). We pull the "Cell Counts" layer from the public NCEI ArcGIS
MapServer (no credential), filtered to the Sentinel era (SAMPLE_DATE >= 2016-01-01).

Each record is a dated, instantaneous in-situ measurement at a lon/lat point -> a sparse
POINT dataset, written as one dataset-wide GeoJSON point table (spec 2a), NOT per-point
GeoTIFFs.

Task: CLASSIFICATION into the HABSOS/NOAA K. brevis cell-abundance categories
(not_present / very_low / low / medium / high), which are precomputed from cell counts
(cells/L) by NCEI. This is the natural, well-defined labeling; thresholds are recorded in
metadata. Balanced to <=1000 samples/class (5 classes -> <=5000 points).

Time range: a red-tide bloom is a specific-date phenomenon (match-up truth), so each point
gets a SHORT window CENTERED on the observation date (+/-7 days = 14 days, well under the
360-day cap), not a static year. change_time is null (this is a state/condition label, not
a dated change event).
"""

import argparse
import json
import multiprocessing
from collections import Counter
from datetime import UTC, datetime

from olmoearth_pretrain.open_set_segmentation_data import download, io, manifest

SLUG = "habsos_harmful_algal_blooms_observing_system"
MAPSERVER = (
    "https://gis.ncdc.noaa.gov/arcgis/rest/services/ms/HABSOS_CellCounts/MapServer"
)
LAYER_ID = 0
PER_CLASS = 1000
HALF_WINDOW_DAYS = 7  # +/-7 days -> 14-day window centered on the observation date

# HABSOS K. brevis (Karenia brevis) cell-abundance categories, ordered by increasing
# abundance. NCEI assigns CATEGORY from the in-situ cell count (cells/L). Observed HABSOS
# thresholds (post-2016 export): not observed = 0; very low = 1..<10,000; low
# 10,000..<100,000; medium 100,000..<1,000,000; high >=1,000,000 cells/L. These match the
# standard NOAA red-tide bloom bins (background/very-low/low/medium/high) used for K. brevis.
CATEGORY_TO_ID = {
    "not observed": 0,
    "very low": 1,
    "low": 2,
    "medium": 3,
    "high": 4,
}
CLASSES = [
    ("not_present", "K. brevis not detected in the sample (0 cells/L)."),
    ("very_low", "Very low K. brevis abundance: 1 to <10,000 cells/L."),
    ("low", "Low K. brevis abundance: 10,000 to <100,000 cells/L."),
    ("medium", "Medium K. brevis abundance: 100,000 to <1,000,000 cells/L."),
    ("high", "High K. brevis abundance / bloom: >=1,000,000 cells/L."),
]


def download_raw() -> str:
    """Download the post-2016 HABSOS Cell Counts layer as GeoJSON; return the path."""
    raw = io.raw_dir(SLUG)
    raw.mkdir(parents=True, exist_ok=True)
    dst = raw / "habsos_cellcounts_2016plus.geojson"
    download.download_arcgis_layer(
        MAPSERVER,
        LAYER_ID,
        dst,
        where="SAMPLE_DATE >= date '2016-01-01'",
        out_sr=4326,
        page=5000,
    )
    with (raw / "SOURCE.txt").open("w") as f:
        f.write(
            "NOAA NCEI HABSOS 'Cell Counts' layer, public ArcGIS MapServer (no credential):\n"
            f"{MAPSERVER}/{LAYER_ID}\n"
            "Filter: SAMPLE_DATE >= 2016-01-01 (Sentinel era). Public domain (NOAA).\n"
        )
    return str(dst)


def load_records(path: str) -> list[dict]:
    """Parse GeoJSON into flat records; filter to post-2016 K. brevis with a valid category."""
    with open(path) as f:
        fc = json.load(f)
    recs = []
    for feat in fc["features"]:
        p = feat.get("properties", {})
        geom = feat.get("geometry")
        if not geom or not geom.get("coordinates"):
            continue
        lon, lat = geom["coordinates"][0], geom["coordinates"][1]
        if lon is None or lat is None:
            continue
        cat = p.get("CATEGORY")
        cid = CATEGORY_TO_ID.get(cat)
        if cid is None:  # drop uncategorized (None) records
            continue
        d = p.get("SAMPLE_DATE")
        if d is None:
            continue
        dt = datetime.fromtimestamp(d / 1000, tz=UTC)
        if dt.year < 2016:  # redundant with server filter, but be safe
            continue
        recs.append(
            {
                "lon": float(lon),
                "lat": float(lat),
                "label": cid,
                "category": cat,
                "cellcount": p.get("CELLCOUNT"),
                "date": dt,
                "state": p.get("STATE_ID"),
                "source_id": f"OBJECTID_{p.get('OBJECTID')}",
            }
        )
    return recs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.parse_args()

    io.check_disk()
    manifest.write_registry_entry(SLUG, "in_progress")

    path = download_raw()
    recs = load_records(path)
    print(f"loaded {len(recs)} post-2016 K. brevis records with a category")

    # balance_by_class enforces the 25k cap by default; 5 classes * 1000 = 5000 << 25k.
    from olmoearth_pretrain.open_set_segmentation_data.sampling import balance_by_class

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
                "time_range": io.centered_time_range(r["date"], HALF_WINDOW_DAYS),
                "change_time": None,
                "source_id": r["source_id"],
                # auxiliary fields copied verbatim into feature properties
                "category": r["category"],
                "cellcount_cells_per_l": r["cellcount"],
                "observation_date": r["date"].isoformat(),
                "state": r["state"],
            }
        )
    io.write_points_table(SLUG, "classification", points)

    counts = Counter(r["label"] for r in selected)
    io.write_dataset_metadata(
        SLUG,
        {
            "dataset": SLUG,
            "name": "HABSOS (Harmful Algal BloomS Observing System)",
            "task_type": "classification",
            "source": "NOAA NCEI",
            "license": "public domain (NOAA)",
            "provenance": {
                "url": "https://www.ncei.noaa.gov/products/harmful-algal-blooms-observing-system",
                "have_locally": False,
                "annotation_method": "in-situ cell counts (K. brevis, cells/L); NCEI-assigned bloom category",
                "access": f"{MAPSERVER}/{LAYER_ID} (public ArcGIS MapServer, no credential)",
            },
            "sensors_relevant": ["sentinel2", "sentinel1", "landsat"],
            "classes": [
                {"id": i, "name": name, "description": desc}
                for i, (name, desc) in enumerate(CLASSES)
            ],
            "class_thresholds_cells_per_l": {
                "not_present": "0",
                "very_low": "1 - <10,000",
                "low": "10,000 - <100,000",
                "medium": "100,000 - <1,000,000",
                "high": ">=1,000,000",
            },
            "nodata_value": io.CLASS_NODATA,
            "num_samples": len(selected),
            "class_counts": {
                name: counts.get(i, 0) for i, (name, _) in enumerate(CLASSES)
            },
            "notes": (
                "Sparse in-situ point table (spec 2a). Filtered to Sentinel era "
                "(SAMPLE_DATE >= 2016-01-01); pre-2016 dropped. All post-2016 records are "
                "Karenia brevis (Gulf of Mexico + SE US coast red tide). Per-point time_range "
                "is a 14-day window centered on the observation date (bloom is a specific-date "
                "match-up phenomenon), change_time=null. Uncategorized (None) records dropped. "
                "Balanced to <=1000/class."
            ),
        },
    )
    manifest.write_registry_entry(
        SLUG, "completed", task_type="classification", num_samples=len(selected)
    )
    print(
        "class counts:", {name: counts.get(i, 0) for i, (name, _) in enumerate(CLASSES)}
    )
    print("done")


if __name__ == "__main__":
    multiprocessing.set_start_method("forkserver", force=True)
    main()
