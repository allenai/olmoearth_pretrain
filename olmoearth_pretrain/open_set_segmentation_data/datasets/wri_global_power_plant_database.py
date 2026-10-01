"""Process the WRI Global Power Plant Database into open-set-segmentation labels.

Source: WRI Global Power Plant Database v1.3.0 (CC-BY-4.0), a curated global
compilation of ~35k power plants in 167 countries. Each record is a plant with an
explicit ``latitude``/``longitude``, a ``primary_fuel`` (the class we label), a
``capacity_mw`` and (about half the time) a ``commissioning_year``.

This is a sparse-point classification dataset: the class is the plant's primary fuel
type. Per spec 2a we therefore write ONE dataset-wide ``points.geojson`` (via
``io.write_points_table``), one Point feature per plant, NOT per-sample GeoTIFFs.

Why a single point is an adequate label here: GPPD ships one representative coordinate
per plant whose spatial precision varies (``geolocation_source`` ranges from exact plant
locations to locality/centroid geocodes), and plant footprints span orders of magnitude
(a rooftop-scale gas peaker vs a multi-km solar or hydro complex). We do not have a
footprint polygon and the coordinate accuracy does not justify fabricating one, so a 1x1
10 m point marking plant presence at the reported location is the honest representation
(§2a); pretraining projects it onto the S2 grid.

Run: python3 -m olmoearth_pretrain.open_set_segmentation_data.datasets.wri_global_power_plant_database
"""

import argparse
import multiprocessing
from collections import Counter
from typing import Any

import pandas as pd

from olmoearth_pretrain.open_set_segmentation_data import io, manifest
from olmoearth_pretrain.open_set_segmentation_data.sampling import balance_by_class

SLUG = "wri_global_power_plant_database"
SOURCE_URL = "https://datasets.wri.org/datasets/global-power-plant-database"
CSV_REL = "extracted/global_power_plant_database.csv"
PER_CLASS = 1000

# Manifest class scheme (order == id). GPPD primary_fuel is mapped case-insensitively
# onto these 10 fuel classes; plants with a primary_fuel outside this set (Storage,
# Other, Cogeneration, Petcoke, "Wave and Tidal") are dropped and reported in the summary.
CLASSES = [
    ("coal", "Plants burning coal as the primary fuel for electricity generation."),
    ("gas", "Plants using natural gas / fossil gas as the primary fuel."),
    ("oil", "Plants burning oil / petroleum products as the primary fuel."),
    (
        "hydro",
        "Hydroelectric plants (run-of-river, reservoir, pumped storage) driven by water.",
    ),
    ("nuclear", "Nuclear fission power stations."),
    ("solar", "Solar power plants (utility-scale photovoltaic or concentrated solar)."),
    ("wind", "Wind power plants (onshore or offshore turbine arrays)."),
    ("biomass", "Plants burning biomass / bioenergy feedstock as the primary fuel."),
    ("geothermal", "Geothermal power plants using subsurface heat."),
    ("waste", "Waste-to-energy plants burning municipal or industrial waste."),
]
NAME_TO_ID = {name: i for i, (name, _d) in enumerate(CLASSES)}

# GPPD primary_fuel string (lowercased) -> our class name. Values not present here are
# outside the manifest's 10-class scheme and are intentionally dropped.
FUEL_MAP = {name: name for name, _ in CLASSES}

# Default representative year for a static persistent structure with no known
# commissioning year (§5: static labels -> a representative Sentinel-era 1-year window).
DEFAULT_YEAR = 2019
MIN_YEAR = 2016
MAX_YEAR = 2021


def _year_for(commissioning_year: float) -> int:
    """A 1-year Sentinel-era window in which the plant is expected to exist.

    Power plants are persistent structures, so any Sentinel-era year in which the plant
    already exists is valid. If a commissioning year is known and later than the default
    window, anchor on it (so we do not label a window before the plant was built),
    clamped to [2016, 2021]; otherwise use the representative default year.
    """
    if commissioning_year is not None and not pd.isna(commissioning_year):
        cy = int(commissioning_year)
        if cy > DEFAULT_YEAR:
            return min(max(cy, MIN_YEAR), MAX_YEAR)
    return DEFAULT_YEAR


def load_records() -> list[dict[str, Any]]:
    csv_path = io.raw_dir(SLUG) / CSV_REL
    df = pd.read_csv(csv_path.path, low_memory=False)
    recs: list[dict[str, Any]] = []
    for row in df.itertuples(index=False):
        fuel = str(getattr(row, "primary_fuel", "")).strip().lower()
        cls = FUEL_MAP.get(fuel)
        if cls is None:
            continue
        lon = getattr(row, "longitude", None)
        lat = getattr(row, "latitude", None)
        if lon is None or lat is None or pd.isna(lon) or pd.isna(lat):
            continue
        recs.append(
            {
                "lon": float(lon),
                "lat": float(lat),
                "label": cls,
                "year": _year_for(getattr(row, "commissioning_year", None)),
                "source_id": str(getattr(row, "gppd_idnr", "")),
                "capacity_mw": None
                if pd.isna(getattr(row, "capacity_mw", None))
                else float(getattr(row, "capacity_mw")),
            }
        )
    return recs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=64)
    args = parser.parse_args()
    _ = args

    io.check_disk()
    manifest.write_registry_entry(SLUG, "in_progress")

    raw = io.raw_dir(SLUG)
    raw.mkdir(parents=True, exist_ok=True)
    with (raw / "SOURCE.txt").open("w") as f:
        f.write(
            "WRI Global Power Plant Database v1.3.0 (CC-BY-4.0)\n"
            f"portal: {SOURCE_URL}\n"
            "zip: https://wri-dataportal-prod.s3.amazonaws.com/manual/global_power_plant_database_v_1_3.zip\n"
        )

    recs = load_records()
    raw_counts = Counter(r["label"] for r in recs)
    print(
        f"loaded {len(recs)} georeferenced plants across {len(raw_counts)} fuel classes"
    )

    selected = balance_by_class(recs, "label", per_class=PER_CLASS)
    print(f"selected {len(selected)} (<= {PER_CLASS}/class)")

    points = []
    for i, r in enumerate(selected):
        pt = {
            "id": f"{i:06d}",
            "lon": r["lon"],
            "lat": r["lat"],
            "label": NAME_TO_ID[r["label"]],
            "time_range": io.year_range(r["year"]),
            "source_id": r["source_id"],
        }
        if r["capacity_mw"] is not None:
            pt["capacity_mw"] = r["capacity_mw"]
        points.append(pt)
    io.write_points_table(SLUG, "classification", points)

    sel_counts = Counter(r["label"] for r in selected)
    io.write_dataset_metadata(
        SLUG,
        {
            "dataset": SLUG,
            "name": "WRI Global Power Plant Database",
            "task_type": "classification",
            "source": "WRI",
            "license": "CC-BY-4.0",
            "provenance": {
                "url": SOURCE_URL,
                "have_locally": False,
                "annotation_method": "curated compilation",
            },
            "sensors_relevant": ["sentinel2", "sentinel1", "landsat"],
            "classes": [
                {"id": i, "name": name, "description": desc}
                for i, (name, desc) in enumerate(CLASSES)
            ],
            "nodata_value": io.CLASS_NODATA,
            "num_samples": len(selected),
            "class_counts": {name: sel_counts.get(name, 0) for name, _ in CLASSES},
            "raw_class_counts": {name: raw_counts.get(name, 0) for name, _ in CLASSES},
            "notes": (
                "Sparse-point fuel-type classification (spec 2a): one points.geojson, one "
                "1x1 point per plant. Plants with primary_fuel outside the 10 manifest "
                "classes (Storage, Other, Cogeneration, Petcoke, Wave and Tidal) are "
                "dropped. Static persistent structures -> 1-year Sentinel-era window "
                "(default 2019; anchored on commissioning_year when it is later, clamped "
                "to 2016-2021). Balanced to <=1000/class."
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
