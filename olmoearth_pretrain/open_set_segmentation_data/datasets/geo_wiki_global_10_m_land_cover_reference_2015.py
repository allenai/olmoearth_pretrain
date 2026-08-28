"""Process the Geo-Wiki Global 10 m Land Cover Reference (2015) into label patches.

Source: Zenodo record 14871659 (ESSD), file ``final_reference_data.csv`` — an
expert/crowd photointerpreted global reference set behind CGLS-LC100 and ESA WorldCover.
Each CSV row is one interpreted 10 m pixel with a land-cover ``class_name``, a WGS84
center (``center_x`` lon, ``center_y`` lat) and ``reference_year`` (all 2015). There are
~16.6M points over ~166k sample locations.

Recipe: sparse point segmentation -> one 1x1 uint8 label patch per point (value = class
id), balanced to <=1000 per class. The uncertain "Not sure" class is dropped (treated as
ignore, not a land-cover class), leaving the 12 land-cover classes.

Time range: the reference year is 2015, but global Sentinel-2 coverage in 2015 is sparse
(S2A launched mid-2015, S2B in 2017). Per the task guidance we anchor the 1-year window on
2016, the first full Sentinel-2 year, so labeled points can be co-located with imagery.
"""

import argparse
import multiprocessing
from typing import Any

import pandas as pd

from olmoearth_pretrain.open_set_segmentation_data import io, manifest
from olmoearth_pretrain.open_set_segmentation_data.sampling import balance_by_class

SLUG = "geo_wiki_global_10_m_land_cover_reference_2015"
NAME = "Geo-Wiki Global 10 m Land Cover Reference (2015)"
CSV_NAME = "final_reference_data.csv"
PER_CLASS = 1000
# Sentinel era: reference year is 2015 but S2 coverage is sparse; use first full S2 year.
TIME_YEAR = 2016
# Oversample per class from the CSV before balancing (keeps memory + I/O bounded).
PRESAMPLE_PER_CLASS = 5000

# Land-cover classes as they appear in the source (source class_id in comment), ordered by
# class_id. The uncertain "Not sure" (3034) class is intentionally excluded. Descriptions
# follow the Geo-Wiki / CGLS-LC100 / WorldCover discrete land-cover legend.
CLASSES = [
    (
        "tree",
        "Tree cover: land dominated by woody vegetation / trees (natural forest and plantations).",
    ),
    (
        "shrub",
        "Shrub cover: land dominated by woody shrubs generally less than ~5 m tall.",
    ),
    (
        "grassland",
        "Grassland: land dominated by natural herbaceous / grass vegetation.",
    ),
    (
        "crops",
        "Cultivated cropland: land used for annual/perennial cultivated crops and managed agriculture.",
    ),
    (
        "urban/built-up",
        "Urban / built-up: human-made impervious surfaces (buildings, roads, settlements).",
    ),
    (
        "bare",
        "Bare / sparsely vegetated: bare soil, sand, rock, or very sparse vegetation.",
    ),
    ("burnt", "Burnt: recently burned area with fire scars."),
    ("water", "Open water: permanent inland or coastal water bodies."),
    ("snow and ice", "Snow and ice: permanent snow, glaciers, and ice cover."),
    (
        "fallow/shifting cultivation",
        "Fallow / shifting cultivation: temporarily uncultivated or rotational agricultural land.",
    ),
    (
        "wetland (herbaceous)",
        "Herbaceous wetland: seasonally or permanently flooded herbaceous vegetation (marsh, swamp).",
    ),
    (
        "Lichen and moss",
        "Lichen and moss: land dominated by lichens and mosses (e.g. tundra).",
    ),
]
NAME_TO_ID = {name: i for i, (name, _desc) in enumerate(CLASSES)}


def scan_records() -> list[dict[str, Any]]:
    """Read the CSV and pre-sample up to PRESAMPLE_PER_CLASS records per land-cover class.

    Returns flat records with lon/lat/label/source_id. The full CSV has ~16.6M rows, so we
    read only the needed columns and draw a bounded random sample per class up front;
    ``balance_by_class`` then trims to PER_CLASS.
    """
    csv_path = io.raw_dir(SLUG) / CSV_NAME
    df = pd.read_csv(
        str(csv_path),
        usecols=["Unnamed: 0", "class_name", "center_x", "center_y"],
    )
    df = df.rename(columns={"Unnamed: 0": "row_idx"})
    df = df[df["class_name"].isin(NAME_TO_ID)]
    df = df.dropna(subset=["center_x", "center_y"])
    # Bounded random sample per class (reproducible): shuffle within class, then take the
    # first PRESAMPLE_PER_CLASS (fewer if the class has fewer rows). Retains all columns.
    df = df.groupby("class_name", group_keys=False).sample(frac=1.0, random_state=42)
    df = df.groupby("class_name", group_keys=False).head(PRESAMPLE_PER_CLASS)
    recs: list[dict[str, Any]] = []
    for row in df.itertuples(index=False):
        recs.append(
            {
                "lon": float(row.center_x),
                "lat": float(row.center_y),
                "label": row.class_name,
                "source_id": f"row_{int(row.row_idx)}",
            }
        )
    return recs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=64)
    args = parser.parse_args()

    io.check_disk()
    manifest.write_registry_entry(SLUG, "in_progress")

    raw = io.raw_dir(SLUG)
    raw.mkdir(parents=True, exist_ok=True)
    csv_path = raw / CSV_NAME
    if not csv_path.exists():
        from olmoearth_pretrain.open_set_segmentation_data import download

        download.download_zenodo("14871659", raw)
    io.check_disk()

    recs = scan_records()
    print(f"scanned {len(recs)} candidate points (pre-sampled)")
    selected = balance_by_class(recs, "label", per_class=PER_CLASS)
    print(f"selected {len(selected)} (<= {PER_CLASS}/class)")

    # Sparse point dataset -> one dataset-wide point table (spec 2a), not per-point tifs.
    points = [
        {
            "id": f"{i:06d}",
            "lon": r["lon"],
            "lat": r["lat"],
            "label": NAME_TO_ID[r["label"]],
            "time_range": io.year_range(TIME_YEAR),
            "source_id": r["source_id"],
        }
        for i, r in enumerate(selected)
    ]
    io.write_points_table(SLUG, "classification", points)

    from collections import Counter

    counts = Counter(r["label"] for r in selected)
    io.write_dataset_metadata(
        SLUG,
        {
            "dataset": SLUG,
            "name": NAME,
            "task_type": "classification",
            "source": "Zenodo / ESSD (record 14871659)",
            "license": "CC-BY-4.0",
            "provenance": {
                "url": "https://doi.org/10.5281/zenodo.14871659",
                "have_locally": False,
                "annotation_method": "manual photointerpretation (Geo-Wiki expert/crowd)",
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
                "Sparse point dataset -> points.json table (one row per interpreted 10 m "
                "reference point), <=1000/class. Reference year is 2015; time range anchored "
                "on 2016 (first full Sentinel-2 year) because 2015 S2 coverage is sparse. The "
                "source 'Not sure' class was dropped (uncertain, not a land-cover class)."
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
