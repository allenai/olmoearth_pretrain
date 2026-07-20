"""Process HydroWASTE (Global Wastewater Treatment Plants) into presence-only points.

Source: Ehalt Macedo, H., Lehner, B., Nicell, J. A., Grill, G., Li, J., Limtong, A.,
Shakya, R. "Distribution and characteristics of wastewater treatment plants within the
global river network." Earth Syst. Sci. Data 14, 559-577 (2022),
https://doi.org/10.5194/essd-14-559-2022. Data: HydroWASTE version 1.0 (HydroSHEDS),
figshare https://doi.org/10.6084/m9.figshare.14847786.v1, license CC-BY-4.0. One zip
containing HydroWASTE_v10.csv (58,502 WWTPs) + README.txt.

Task type: presence-only POINTS (spec section 2a), single class (wastewater treatment
plant). Each selected WWTP is emitted as one presence point in a dataset-wide
``points.geojson``; negatives are supplied by the downstream assembly (no fabricated
background tiles here). A WWTP's aeration/settling ponds and clarifier tanks are readily
discernible at 10-30 m.

Coordinate precision (spec note): HydroWASTE reports a geocoded plant location
(LAT_WWTP/LON_WWTP) with a per-record quality flag QUAL_LOC (1=high >80% of a
country/region's points accurate, 2=medium 50-80%, 3=low <50%, 4=not analysed). Most
records are 3-decimal (~110 m) precision. To keep positives reliable we place points only
on **well-located** plants (QUAL_LOC in {1,2}) whose STATUS implies a built plant (exclude
Projected/Proposed/Under Construction/Construction Completed). We use the reported PLANT
location, NOT the estimated river-outfall location (LAT_OUT/LON_OUT), since the physical
infrastructure sits at the plant.

Class scheme (single class):
  0 wastewater_treatment_plant

Time range: WWTPs are persistent structures (undated in the source). Per spec section 5
(static labels) each point gets a 1-year window at a representative Sentinel-era year,
spread pseudo-randomly across 2016-2022 for temporal diversity; change_time is null.

Sampling: up to 1000 points (sampling.balance_by_class, default 25k total cap).

Run (reuses cached raw CSV):
  python3 -m olmoearth_pretrain.open_set_segmentation_data.datasets.hydrowaste_global_wastewater_treatment_plants
"""

import argparse
import csv
import multiprocessing
import random
import zipfile
from typing import Any

from olmoearth_pretrain.open_set_segmentation_data import download, io, manifest
from olmoearth_pretrain.open_set_segmentation_data.sampling import balance_by_class

SLUG = "hydrowaste_global_wastewater_treatment_plants"
NAME = "HydroWASTE (Global Wastewater Treatment Plants)"
FIGSHARE_URL = "https://doi.org/10.6084/m9.figshare.14847786.v1"
DOWNLOAD_FILE_URL = "https://ndownloader.figshare.com/files/31910714"
ZIP_NAME = "HydroWASTE_v10.zip"
CSV_NAME = "HydroWASTE_v10.csv"

CID_WWTP = 0
CLASSES = [
    {
        "id": CID_WWTP,
        "name": "wastewater_treatment_plant",
        "description": "Wastewater/sewage treatment plant location from HydroWASTE (reported "
        "plant location LAT_WWTP/LON_WWTP, compiled from national/regional datasets). Physical "
        "infrastructure (aeration/settling ponds, clarifier tanks) discernible at 10-30 m.",
    },
]
CID_TO_NAME = {c["id"]: c["name"] for c in CLASSES}

# STATUS values that imply the plant is NOT (yet) a built, visible facility.
NOT_BUILT_STATUS = {
    "Projected",
    "Proposed",
    "Under Construction",
    "Construction Completed",
}
# Location-quality flags we trust for placing positive points (1=high, 2=medium).
GOOD_QUAL_LOC = {"1", "2"}

PER_CLASS = 1000  # WWTP presence points (single class, spec section 5)
YEARS = list(range(2016, 2023))
SEED = 42


def csv_path() -> Any:
    return io.raw_dir(SLUG) / CSV_NAME


def ensure_downloaded() -> None:
    """Download + extract HydroWASTE_v10.zip into raw_dir if the CSV is not present."""
    raw = io.raw_dir(SLUG)
    raw.mkdir(parents=True, exist_ok=True)
    if csv_path().exists():
        return
    zip_path = download.download_http(DOWNLOAD_FILE_URL, raw / ZIP_NAME)
    with zipfile.ZipFile(str(zip_path)) as z:
        z.extractall(str(raw))


def read_plants() -> list[dict[str, Any]]:
    """Read well-located, built HydroWASTE plants into presence records.

    Keeps rows with valid coords that are well-located (QUAL_LOC in {1,2}) and built
    (STATUS not in NOT_BUILT_STATUS). Each record has label/lon/lat/source_id.
    """
    good_plants: list[dict[str, Any]] = []
    with csv_path().open(encoding="latin-1") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                lat = float(row["LAT_WWTP"])
                lon = float(row["LON_WWTP"])
            except (TypeError, ValueError):
                continue
            if not (-90 <= lat <= 90 and -180 <= lon <= 180):
                continue
            if lat == 0.0 and lon == 0.0:
                continue
            if row["QUAL_LOC"] not in GOOD_QUAL_LOC or row["STATUS"] in NOT_BUILT_STATUS:
                continue
            good_plants.append(
                {
                    "label": CID_WWTP,
                    "lon": lon,
                    "lat": lat,
                    "source_id": f"WASTE_ID/{row['WASTE_ID']}",
                }
            )
    return good_plants


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=64)
    parser.parse_args()

    io.check_disk()
    manifest.write_registry_entry(SLUG, "in_progress")

    ensure_downloaded()

    raw = io.raw_dir(SLUG)
    with (raw / "SOURCE.txt").open("w") as f:
        f.write(
            "HydroWASTE version 1.0 (Global Wastewater Treatment Plants). Ehalt Macedo "
            "et al., Earth Syst. Sci. Data 14, 559-577 (2022). HydroSHEDS. License "
            "CC-BY-4.0.\n"
            f"figshare: {FIGSHARE_URL}\n"
            f"file: {DOWNLOAD_FILE_URL} -> {ZIP_NAME} -> {CSV_NAME} (58,502 WWTPs) + "
            "README.txt.\n"
            "Positives use reported plant location LAT_WWTP/LON_WWTP (not the estimated "
            "river outfall LAT_OUT/LON_OUT).\n"
        )

    print("reading WWTP points ...", flush=True)
    good_plants = read_plants()
    print(
        f"  {len(good_plants)} well-located & built plants "
        "(QUAL_LOC in {1,2}, STATUS built)",
        flush=True,
    )

    selected = balance_by_class(good_plants, "label", per_class=PER_CLASS, seed=SEED)
    print(f"selected {len(selected)} points (<= {PER_CLASS})", flush=True)

    # Persistent features -> 1-year window at a representative Sentinel-era year, spread
    # pseudo-randomly across 2016-2022 for temporal diversity.
    yrng = random.Random(123)
    points = []
    for i, r in enumerate(selected):
        year = YEARS[yrng.randrange(len(YEARS))]
        points.append(
            {
                "id": f"{i:06d}",
                "lon": r["lon"],
                "lat": r["lat"],
                "label": r["label"],
                "time_range": io.year_range(year),
                "change_time": None,
                "source_id": r["source_id"],
            }
        )
    io.write_points_table(SLUG, "classification", points)

    io.write_dataset_metadata(
        SLUG,
        {
            "dataset": SLUG,
            "name": NAME,
            "task_type": "classification",
            "source": "HydroSHEDS / HydroWASTE v1.0 (Ehalt Macedo et al. 2022, ESSD)",
            "license": "CC-BY-4.0",
            "provenance": {
                "url": FIGSHARE_URL,
                "paper": "https://doi.org/10.5194/essd-14-559-2022",
                "have_locally": False,
                "annotation_method": "authoritative + modeled (national/regional WWTP "
                "registries geocoded and completed with auxiliary data)",
            },
            "sensors_relevant": ["sentinel2", "sentinel1", "landsat"],
            "classes": CLASSES,
            "num_samples": len(selected),
            "class_counts": {CID_TO_NAME[CID_WWTP]: len(selected)},
            "notes": (
                "Presence-only POINTS converted from the former detection-tile encoding; "
                "negatives are supplied by the downstream assembly. Single class: "
                "0=wastewater_treatment_plant. Points placed at reported plant location "
                "(LAT_WWTP/LON_WWTP), restricted to QUAL_LOC in {1,2} (>50% located "
                "accurately) and built STATUS (Projected/Proposed/Under Construction/"
                "Construction Completed excluded). Up to 1000 points sampled from the "
                "well-located plants (balance_by_class). Persistent features -> 1-year "
                "window at a representative Sentinel-era year (2016-2022); change_time=null."
            ),
        },
    )
    manifest.write_registry_entry(
        SLUG, "completed", task_type="classification", num_samples=len(selected)
    )
    print("done:", len(selected), "points", flush=True)


if __name__ == "__main__":
    multiprocessing.set_start_method("forkserver", force=True)
    main()
