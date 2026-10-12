"""Process the Annual NLCD Collection 1.0 Reference Data Product into open-set-segmentation
point labels.

Source: USGS ScienceBase, "Annual National Land Cover Database (NLCD) Collection 1.0
Reference Data Product" (DOI 10.5066/P13EDMAF; also mirrored on MRLC). An independent,
manually interpreted reference dataset of 8,360 30 m x 30 m plots across CONUS, each with
an annual land-cover label for every year 1984-2023. This is the manual *reference* data
behind Annual NLCD (preferred over the derived map, per the manifest note).

We keep only the Sentinel-era years (2016-2023) and treat each (plot, year) as one sparse
point-segmentation sample: the plot center lon/lat carries the plot's ``primary_landcover_code``
for that year, over a 1-year time range. Sparse 1x1 point labels -> one dataset-wide point
table (points.json, spec 2a), balanced to <=1000 samples per class (25k total cap).

The label is the standard NLCD level-2 legend (16 classes), remapped to contiguous 0-based
uint8 ids. Coordinates in the source are CONUS Albers (WGS84 datum); we reproject to WGS84
lon/lat and let pretraining snap the point onto the S2 grid.

Reproduce:
    python3 -m olmoearth_pretrain.open_set_segmentation_data.datasets.annual_nlcd_reference_data
"""

import argparse
import csv
import multiprocessing
import os
from collections import Counter
from typing import Any

from pyproj import CRS, Transformer

from olmoearth_pretrain.open_set_segmentation_data import io, manifest
from olmoearth_pretrain.open_set_segmentation_data.sampling import balance_by_class

SLUG = "annual_nlcd_reference_data"
SOURCE_URL = "https://doi.org/10.5066/P13EDMAF"
MIRROR_URL = (
    "https://www.mrlc.gov/downloads/sciweb1/shared/mrlc/data-bundles/"
    "Annual_NLCD_CONUSV1_Ref_Data_Release.zip"
)
PER_CLASS = 1000
YEAR_MIN, YEAR_MAX = 2016, 2023  # Sentinel-2 era

ATTR_CSV = "NLCD2023_Full8360_AnnualAttributes.csv"
COORD_CSV = "Plot_Coordinates_List_Simple_and_Stratified.csv"
PRJ_FILE = "lcnext_8360_final.prj"

# Standard NLCD level-2 legend. (NLCD source code, class name, definition) in ascending
# code order; list index is the 0-based uint8 class id we write. Names/definitions from the
# Annual NLCD reference-data metadata (primary_landcover_code domain) + NLCD legend.
NLCD_CLASSES = [
    (
        11,
        "Open Water",
        "Areas of open water, generally with less than 25% cover of vegetation or soil.",
    ),
    (
        12,
        "Perennial Ice/Snow",
        "Areas characterized by a majority cover of ice and/or snow that persists year-round.",
    ),
    (
        21,
        "Developed, Open Space",
        "Developed areas with mostly vegetation in the form of lawn grasses; impervious surfaces <20% of total cover (large-lot single-family housing, parks, golf courses).",
    ),
    (
        22,
        "Developed, Low Intensity",
        "Developed areas with a mix of constructed materials and vegetation; impervious surfaces 20-49% of total cover (single-family housing).",
    ),
    (
        23,
        "Developed, Medium Intensity",
        "Developed areas with a mix of constructed materials and vegetation; impervious surfaces 50-79% of total cover.",
    ),
    (
        24,
        "Developed, High Intensity",
        "Highly developed areas where people reside or work in high numbers; impervious surfaces 80-100% of total cover (apartments, commercial/industrial).",
    ),
    (
        31,
        "Barren Land",
        "Barren areas of bedrock, desert pavement, scarps, talus, slides, volcanic material, glacial debris, sand dunes, strip mines, gravel pits; vegetation <15% of cover.",
    ),
    (
        41,
        "Deciduous Forest",
        "Areas dominated by trees generally >5 m tall (>20% total vegetation cover) where more than 75% of the tree species shed foliage in response to seasonal change.",
    ),
    (
        42,
        "Evergreen Forest",
        "Areas dominated by trees generally >5 m tall (>20% total vegetation cover) where more than 75% of the tree species maintain their leaves all year.",
    ),
    (
        43,
        "Mixed Forest",
        "Areas dominated by trees generally >5 m tall (>20% total vegetation cover) where neither deciduous nor evergreen species are greater than 75% of tree cover.",
    ),
    (
        52,
        "Shrub/Scrub",
        "Areas dominated by shrubs <5 m tall with shrub canopy typically >20% of total vegetation; includes true shrubs, young trees, and stunted/environmentally-limited trees.",
    ),
    (
        71,
        "Grassland/Herbaceous",
        "Areas dominated by graminoid or herbaceous vegetation, generally >80% of total vegetation; not subject to intensive management but may be grazed.",
    ),
    (
        81,
        "Pasture/Hay",
        "Areas of grasses, legumes, or grass-legume mixtures planted for livestock grazing or the production of seed/hay crops, typically on a perennial cycle.",
    ),
    (
        82,
        "Cultivated Crops",
        "Areas used for the production of annual crops (corn, soybeans, vegetables, cotton) and perennial woody crops (orchards, vineyards); includes actively tilled land.",
    ),
    (
        90,
        "Woody Wetlands",
        "Areas where forest or shrubland vegetation accounts for >20% of vegetative cover and the soil or substrate is periodically saturated with or covered with water.",
    ),
    (
        95,
        "Emergent Herbaceous Wetlands",
        "Areas where perennial herbaceous vegetation accounts for >80% of vegetative cover and the soil or substrate is periodically saturated with or covered with water.",
    ),
]
CODE_TO_ID = {code: i for i, (code, _n, _d) in enumerate(NLCD_CLASSES)}


def _extracted_dir() -> str:
    return os.path.join(io.raw_dir(SLUG).path, "extracted")


def load_plot_lonlat() -> dict[str, tuple[float, float]]:
    """Read plot coordinates (CONUS Albers) and reproject each plot center to WGS84."""
    ed = _extracted_dir()
    crs = CRS.from_wkt(open(os.path.join(ed, PRJ_FILE)).read())
    transformer = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
    xs, ys, pids = [], [], []
    with open(os.path.join(ed, COORD_CSV)) as f:
        for row in csv.DictReader(f):
            pids.append(row["plotid"])
            xs.append(float(row["x"]))
            ys.append(float(row["y"]))
    lons, lats = transformer.transform(xs, ys)
    return {pid: (float(lon), float(lat)) for pid, lon, lat in zip(pids, lons, lats)}


def scan_records() -> list[dict[str, Any]]:
    """Build one record per (plot, year) in the Sentinel era with a known NLCD code."""
    plot_lonlat = load_plot_lonlat()
    recs: list[dict[str, Any]] = []
    with open(os.path.join(_extracted_dir(), ATTR_CSV)) as f:
        for row in csv.DictReader(f):
            year = int(row["image_year"])
            if not (YEAR_MIN <= year <= YEAR_MAX):
                continue
            code_str = row["primary_landcover_code"].strip()
            if not code_str:
                continue
            code = int(float(code_str))
            if code not in CODE_TO_ID:
                continue
            pid = row["plotid"]
            ll = plot_lonlat.get(pid)
            if ll is None:
                continue
            recs.append(
                {
                    "lon": ll[0],
                    "lat": ll[1],
                    "code": code,
                    "year": year,
                    "source_id": f"plot{pid}_{year}",
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

    ed = _extracted_dir()
    missing = [
        n
        for n in (ATTR_CSV, COORD_CSV, PRJ_FILE)
        if not os.path.exists(os.path.join(ed, n))
    ]
    if missing:
        raise FileNotFoundError(
            f"Missing extracted source files {missing} under {ed}. "
            f"Download {MIRROR_URL} into raw_dir and unzip the CSVs + .prj into 'extracted/'."
        )

    recs = scan_records()
    print(f"scanned {len(recs)} plot-year points ({YEAR_MIN}-{YEAR_MAX})")

    selected = balance_by_class(recs, "code", per_class=PER_CLASS)
    print(f"selected {len(selected)} (<= {PER_CLASS}/class, 25k cap)")

    points = []
    for i, r in enumerate(selected):
        points.append(
            {
                "id": f"{i:06d}",
                "lon": r["lon"],
                "lat": r["lat"],
                "label": CODE_TO_ID[r["code"]],
                "time_range": io.year_range(r["year"]),
                "change_time": None,
                "source_id": r["source_id"],
            }
        )
    io.write_points_table(SLUG, "classification", points)

    counts = Counter(r["code"] for r in selected)
    io.write_dataset_metadata(
        SLUG,
        {
            "dataset": SLUG,
            "name": "Annual NLCD Reference Data",
            "task_type": "classification",
            "source": "USGS ScienceBase",
            "license": "CC0-1.0",
            "provenance": {
                "url": SOURCE_URL,
                "mirror": MIRROR_URL,
                "have_locally": False,
                "annotation_method": "manual (analyst interpretation of reference plots)",
            },
            "sensors_relevant": ["sentinel2", "sentinel1", "landsat"],
            "classes": [
                {"id": i, "name": name, "description": desc, "nlcd_code": code}
                for i, (code, name, desc) in enumerate(NLCD_CLASSES)
            ],
            "nodata_value": io.CLASS_NODATA,
            "num_samples": len(selected),
            "class_counts": {
                name: counts.get(code, 0) for code, name, _ in NLCD_CLASSES
            },
            "notes": (
                "1x1 point-segmentation samples; one point per (plot, year). Only "
                f"{YEAR_MIN}-{YEAR_MAX} plot-years kept (Sentinel era). Label = "
                "primary_landcover_code remapped to contiguous 0-based ids. Coordinates "
                "reprojected from CONUS Albers (WGS84 datum) to WGS84 lon/lat. Balanced to "
                "<=1000 samples per class."
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
