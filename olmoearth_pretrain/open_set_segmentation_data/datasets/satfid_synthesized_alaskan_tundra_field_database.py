"""Process SATFiD (Synthesized Alaskan Tundra Field Database) into open-set-segmentation labels.

Source: ORNL DAAC / ABoVE, DOI 10.3334/ORNLDAAC/2177 -- "Field Data on Soils, Vegetation,
and Fire History for Alaska Tundra Sites, 1972-2020". A harmonized in-situ field database
compiled from 37 real field campaigns ("synthesized" = harmonized, NOT synthetic). Each row
of Tundra_field_database.csv is a georeferenced plot with decimal-degree lat/lon, a date,
and per-plant-functional-type (PFT) *percent cover* columns.

Label encoding: sparse-point CLASSIFICATION by *dominant PFT cover*. For each plot we take
argmax over the six manifest PFT cover columns (shrub, lichen, moss, graminoid, forb, litter)
-> class id. The cover columns are independent per-layer estimates (values may exceed 100 and
row sums vary), so argmax-dominant is the faithful single-scalar encoding for the §2a point
table (multi-target regression is not expressible in one dataset here). Sparse 1x1 points ->
one dataset-wide points.geojson (spec §2a), NOT per-point GeoTIFFs.

Access: the 3 CSVs live behind the NASA Earthdata / URS protected download. Provide creds in
~/.netrc (machine urs.earthdata.nasa.gov) OR pre-place the CSVs in raw/{slug}/, then run:
    python3 -m olmoearth_pretrain.open_set_segmentation_data.datasets.satfid_synthesized_alaskan_tundra_field_database
"""

import argparse
import subprocess
import zipfile
from collections import Counter
from typing import Any

import pandas as pd

from olmoearth_pretrain.open_set_segmentation_data import io, manifest

SLUG = "satfid_synthesized_alaskan_tundra_field_database"
NAME = "SATFiD (Synthesized Alaskan Tundra Field Database)"
DOI = "10.3334/ORNLDAAC/2177"
BUNDLE_URL = "https://data.ornldaac.earthdata.nasa.gov/protected/bundle/FieldData_Alaska_Tundra_2177.zip"
CSV_NAME = "Tundra_field_database.csv"
PER_CLASS = 1000
MIN_YEAR = 2016  # Sentinel era; DB spans 1972-2020 (manifest declares 2016-2020) -> keep post-2016 (§8.2)

# Class order == manifest order. The six PFT cover columns map 1:1 to these ids.
CLASSES = [
    (
        "shrubs",
        "shrub_cover",
        "Dwarf/low/tall woody shrubs (e.g. Betula, Salix, Vaccinium); dominant shrub-tundra cover.",
    ),
    (
        "lichens",
        "lichen_cover",
        "Terricolous lichens (e.g. Cladonia/Cetraria); dominant lichen mat cover.",
    ),
    (
        "mosses",
        "moss_cover",
        "Bryophytes / mosses (e.g. Sphagnum, feather mosses); dominant moss-layer cover.",
    ),
    (
        "graminoids",
        "graminoid_cover",
        "Grasses, sedges and rushes (e.g. Carex, Eriophorum); dominant graminoid/tussock cover.",
    ),
    (
        "forbs",
        "forb_cover",
        "Non-graminoid herbaceous plants (forbs); dominant forb cover.",
    ),
    ("litter", "litter_cover", "Dead plant material / litter; dominant litter cover."),
]
COVER_COLS = [c for _, c, _ in CLASSES]  # in manifest/class-id order
ALL_COVER = COVER_COLS + ["bare_cover"]
NODATA_COVER = -999


def _ensure_csv() -> str:
    """Return the local path to Tundra_field_database.csv, downloading the protected bundle
    via ~/.netrc Earthdata auth if the CSV is not already in raw/{slug}/.
    """
    raw = io.raw_dir(SLUG)
    raw.mkdir(parents=True, exist_ok=True)
    csv_path = raw / CSV_NAME
    if csv_path.exists():
        return csv_path.path
    zip_path = raw / "FieldData_Alaska_Tundra_2177.zip"
    if not zip_path.exists():
        print(f"downloading protected bundle via Earthdata netrc: {BUNDLE_URL}")
        tmp = raw / (zip_path.name + ".tmp")
        # curl handles the URS OAuth redirect + cookie jar with --netrc.
        subprocess.run(
            [
                "curl",
                "-sL",
                "--netrc",
                "--location-trusted",
                "--fail",
                "-o",
                tmp.path,
                BUNDLE_URL,
            ],
            check=True,
        )
        tmp.rename(zip_path)
    with zipfile.ZipFile(zip_path.path) as zf:
        member = next(n for n in zf.namelist() if n.endswith(CSV_NAME))
        with zf.open(member) as src:
            data = src.read()
    tmp = raw / (CSV_NAME + ".tmp")
    with tmp.open("wb") as f:
        f.write(data)
    tmp.rename(csv_path)
    # source note
    with (raw / "SOURCE.txt").open("w") as f:
        f.write(
            f"{NAME}\nORNL DAAC DOI {DOI}\nBundle: {BUNDLE_URL}\n"
            "Downloaded via NASA Earthdata (URS) login (~/.netrc machine urs.earthdata.nasa.gov).\n"
            "License: CC-BY-4.0.\n"
        )
    return csv_path.path


def build_records(csv_path: str) -> list[dict[str, Any]]:
    df = pd.read_csv(csv_path, low_memory=False)
    lat = pd.to_numeric(df["latitude"], errors="coerce")
    lon = pd.to_numeric(df["longitude"], errors="coerce")
    good_coord = (
        lat.notna()
        & lon.notna()
        & (lat != NODATA_COVER)
        & (lon != NODATA_COVER)
        & lat.between(-90, 90)
        & lon.between(-180, 180)
    )
    yr = pd.to_numeric(df["yr_data"], errors="coerce")
    df = df[good_coord & (yr >= MIN_YEAR)].copy()

    cov = (
        df[ALL_COVER]
        .apply(pd.to_numeric, errors="coerce")
        .mask(lambda x: x == NODATA_COVER)
    )
    pft = cov[COVER_COLS].fillna(-1.0).values  # NaN -> -1 so it never wins argmax
    bare = cov["bare_cover"].fillna(-1.0).values
    dom_idx = pft.argmax(axis=1)
    dom_val = pft.max(axis=1)
    # keep rows with a positive dominant PFT cover that is not beaten by bare soil
    keep = (dom_val > 0) & (bare <= dom_val)

    df = df[keep].reset_index(drop=True)
    dom_idx = dom_idx[keep]
    yrs = pd.to_numeric(df["yr_data"], errors="coerce").astype(int).values
    recs: list[dict[str, Any]] = []
    for i in range(len(df)):
        recs.append(
            {
                "lon": float(df["longitude"].iloc[i]),
                "lat": float(df["latitude"].iloc[i]),
                "label": int(dom_idx[i]),
                "year": int(yrs[i]),
                "source_id": f"{df['dataset_id'].iloc[i]}/{df['plot_id'].iloc[i]}",
            }
        )
    return recs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.parse_args()

    io.check_disk()
    manifest.write_registry_entry(SLUG, "in_progress")

    csv_path = _ensure_csv()
    recs = build_records(csv_path)
    print(f"usable 2016+ cover points: {len(recs)}")

    # Balance to <=1000/class (only ~350 points / 4 populated classes -> cap never binds).
    from olmoearth_pretrain.open_set_segmentation_data.sampling import balance_by_class

    selected = balance_by_class(recs, "label", per_class=PER_CLASS)
    print(f"selected {len(selected)}")

    points = []
    for i, r in enumerate(selected):
        points.append(
            {
                "id": f"{i:06d}",
                "lon": r["lon"],
                "lat": r["lat"],
                "label": r["label"],
                "time_range": io.year_range(r["year"]),
                "change_time": None,
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
            "source": "ORNL DAAC / ABoVE",
            "license": "CC-BY-4.0",
            "provenance": {
                "url": f"https://doi.org/{DOI}",
                "have_locally": False,
                "annotation_method": "in-situ field survey synthesis (37 campaigns), harmonized",
            },
            "sensors_relevant": ["sentinel2", "sentinel1", "landsat"],
            "classes": [
                {"id": i, "name": name, "description": desc}
                for i, (name, _col, desc) in enumerate(CLASSES)
            ],
            "nodata_value": io.CLASS_NODATA,
            "num_samples": len(selected),
            "class_counts": {
                name: counts.get(i, 0) for i, (name, _c, _d) in enumerate(CLASSES)
            },
            "notes": (
                "Sparse 1x1 point classification: label = dominant plant-functional-type by "
                "percent cover (argmax over the 6 PFT cover columns; plots dropped where "
                "bare_cover dominates or all PFT covers are nodata/0). DB spans 1972-2020; only "
                "the post-2016 (Sentinel-era) subset with vegetation-cover data is kept (§8.2). "
                "1-year time_range anchored on yr_data; change_time=null. forbs/litter classes "
                "retained in the class map but have 0 dominant samples in the 2016+ subset "
                "(kept per §5; downstream assembly filters too-small classes). Cover columns are "
                "independent per-layer estimates (may exceed 100%)."
            ),
        },
    )
    manifest.write_registry_entry(
        SLUG, "completed", task_type="classification", num_samples=len(selected)
    )
    print("done")


if __name__ == "__main__":
    main()
