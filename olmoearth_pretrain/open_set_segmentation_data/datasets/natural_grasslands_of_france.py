"""Process Natural Grasslands of France into open-set-segmentation label points.

Source: Zenodo record 7895449 (Panhelleux, Rapinel & Hubert-Moy 2023, Data in Brief).
The release ships ``grassland_ground_points.geojson``: 1,770 field/aerial-verified ground
reference points across mainland France, each tagged ``type`` = "natural grassland"
(compilation of hundreds of field-based vegetation maps) or "artificial grassland"
(European Union Land Parcel Identification System, LPIS). Points are in Lambert-93
(EPSG:2154); we reproject to WGS84. The companion 10 m raster is a *derived-product map*
built from annual land-cover maps -- we prefer the in-situ reference points and ignore the
raster (spec: prefer manual reference over derived maps).

Sparse point classification (natural vs artificial grassland) -> one dataset-wide point
table (points.json, spec 2a), balanced to <=1000 per class. Labels are seasonal/annual;
the source is a 2016-2020 compilation with no per-point year, so each point is assigned a
representative 1-year window (2018) within that period.
"""

import argparse
import json
import multiprocessing

from pyproj import Transformer

from olmoearth_pretrain.open_set_segmentation_data import download, io, manifest
from olmoearth_pretrain.open_set_segmentation_data.sampling import balance_by_class

SLUG = "natural_grasslands_of_france"
ZENODO_RECORD = "7895449"
POINTS_FILE = "grassland_ground_points.geojson"
PER_CLASS = 1000
REP_YEAR = 2018  # representative year within the 2016-2020 source period

# Manifest class order -> id, with source-derived definitions.
CLASSES = [
    (
        "natural grassland",
        "Semi-natural / natural grassland with spontaneous vegetation, from a compilation "
        "of hundreds of field-based vegetation maps.",
    ),
    (
        "artificial grassland",
        "Sown / improved (artificial) grassland used for forage/pasture, from the European "
        "Union Land Parcel Identification System (LPIS).",
    ),
]
NAME_TO_ID = {name: i for i, (name, _desc) in enumerate(CLASSES)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.parse_args()

    io.check_disk()
    manifest.write_registry_entry(SLUG, "in_progress")

    raw = io.raw_dir(SLUG)
    raw.mkdir(parents=True, exist_ok=True)
    download.download_zenodo(ZENODO_RECORD, raw, filenames=[POINTS_FILE])

    with (raw / POINTS_FILE).open() as f:
        gj = json.load(f)

    # Source CRS is EPSG:2154 (Lambert-93); reproject each point to WGS84 lon/lat.
    transformer = Transformer.from_crs("EPSG:2154", "EPSG:4326", always_xy=True)

    recs = []
    for feat in gj["features"]:
        props = feat.get("properties", {})
        label_name = props.get("type")
        geom = feat.get("geometry")
        if label_name not in NAME_TO_ID or not geom:
            continue
        x, y = geom["coordinates"]
        lon, lat = transformer.transform(x, y)
        recs.append(
            {
                "lon": lon,
                "lat": lat,
                "label": label_name,
                "source_id": f"ID={props.get('ID')}",
            }
        )
    print(f"parsed {len(recs)} ground reference points")

    selected = balance_by_class(recs, "label", per_class=PER_CLASS)
    print(f"selected {len(selected)} (<= {PER_CLASS}/class)")

    points = []
    for i, r in enumerate(selected):
        points.append(
            {
                "id": f"{i:06d}",
                "lon": r["lon"],
                "lat": r["lat"],
                "label": NAME_TO_ID[r["label"]],
                "time_range": io.year_range(REP_YEAR),
                "source_id": r["source_id"],
            }
        )
    io.write_points_table(SLUG, "classification", points)

    from collections import Counter

    counts = Counter(r["label"] for r in selected)
    io.write_dataset_metadata(
        SLUG,
        {
            "dataset": SLUG,
            "name": "Natural Grasslands of France",
            "task_type": "classification",
            "source": "Zenodo / Data in Brief",
            "license": "CC-BY-4.0",
            "provenance": {
                "url": "https://doi.org/10.5281/zenodo.7895449",
                "have_locally": False,
                "annotation_method": "field maps + LPIS + aerial interpretation",
            },
            "sensors_relevant": ["sentinel2", "sentinel1", "landsat"],
            "classes": [
                {"id": i, "name": name, "description": desc}
                for i, (name, desc) in enumerate(CLASSES)
            ],
            "nodata_value": io.CLASS_NODATA,
            "num_samples": len(selected),
            "class_counts": {name: counts.get(name, 0) for name, _ in CLASSES},
            "notes": (
                "1x1 point-segmentation from in-situ ground reference points "
                "(grassland_ground_points.geojson); companion 10 m derived-product raster "
                "not used (reference preferred). Points reprojected EPSG:2154 -> WGS84. "
                "No per-point year in source; ~1-year window anchored on 2018 (within the "
                "2016-2020 source period)."
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
