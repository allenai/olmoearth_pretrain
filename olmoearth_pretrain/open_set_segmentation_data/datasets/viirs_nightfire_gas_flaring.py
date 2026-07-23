"""Process the VIIRS Nightfire global gas-flaring catalog into a sparse point table.

Source: NASA ORNL DAAC "Global Gas Flare Survey by Infrared Imaging, VIIRS Nightfire,
2012-2019" (DOI 10.3334/ORNLDAAC/1874; produced by the Earth Observation Group / Elvidge
& Zhizhin). Each annual ``eog_global_flare_survey_{year}_flare_list.csv`` lists individual
gas-flaring sites detected that calendar year by the VIIRS Nightfire (VNF) algorithm on
Suomi-NPP, with per-site latitude/longitude, average flame temperature (K), estimated
flared gas volume (billion m^3/yr), detection frequency, and a flare-type tag.

This is a **positive-only, single-class** point dataset: every record marks the presence
of a gas flare (class 0 = "gas flare"); there is no background/negative class (the
assembly step supplies negatives from other datasets, spec §5). We therefore write ONE
dataset-wide GeoJSON point table (spec §2a) rather than per-point GeoTIFFs.

Annual catalog -> each point gets a 1-year time_range anchored on its detection year
(spec §5). We restrict to the Sentinel era (2016+) and, because the pooled 2016-2019
catalog exceeds the 25k per-dataset cap, sample down to 25,000 points (seeded).

Per-site temperature / volume / flare-type / country are carried as auxiliary feature
properties (informative provenance, not the classification label).

Run: python3 -m olmoearth_pretrain.open_set_segmentation_data.datasets.viirs_nightfire_gas_flaring
"""

import argparse
import csv
import multiprocessing
from collections import Counter
from typing import Any

from olmoearth_pretrain.open_set_segmentation_data import download, io, manifest
from olmoearth_pretrain.open_set_segmentation_data.sampling import (
    MAX_SAMPLES_PER_DATASET,
    balance_by_class,
)

SLUG = "viirs_nightfire_gas_flaring"
DOI = "https://doi.org/10.3334/ORNLDAAC/1874"
BASE_URL = "https://daac.ornl.gov/daacdata/cms/Methane_Flaring_Sites_VIIRS/"
DOC_FILE = "Methane_Flaring_Sites_VIIRS.pdf"

# Sentinel era only (spec §5 / §8: reject pre-2016 labels). ORNL DAAC 1874 spans
# 2012-2019; we keep 2016-2019.
YEARS = [2016, 2017, 2018, 2019]

CLASS_NAME = "gas flare"
CLASS_DESC = (
    "Location of an active natural-gas flare detected by the VIIRS Nightfire algorithm: "
    "a persistent high-temperature (~1600-2000 K) sub-pixel combustion source separated "
    "from biomass burning by temperature and persistence. Includes upstream production, "
    "refinery, gas-processing, and LNG flares."
)


def _flare_list_filename(year: int) -> str:
    return f"eog_global_flare_survey_{year}_flare_list.csv"


def download_raw() -> None:
    """Download the annual flare-list CSVs (2016-2019) + documentation to raw/ (idempotent)."""
    raw = io.raw_dir(SLUG)
    raw.mkdir(parents=True, exist_ok=True)
    for year in YEARS:
        fn = _flare_list_filename(year)
        download.download_earthdata(BASE_URL + "data/" + fn, raw / fn)
    download.download_earthdata(BASE_URL + "comp/" + DOC_FILE, raw / DOC_FILE)


def _valid_lonlat(lon: float, lat: float) -> bool:
    if not (-180.0 <= lon <= 180.0 and -90.0 <= lat <= 90.0):
        return False
    # Reject null-island / -9999 sentinels.
    if abs(lon) < 1e-6 and abs(lat) < 1e-6:
        return False
    return True


def _num_or_none(s: str | None) -> float | None:
    try:
        v = float(s)
    except (TypeError, ValueError):
        return None
    return None if v <= -999 else v


def scan_records() -> list[dict[str, Any]]:
    """Read all annual flare-list CSVs into flat per-flare records."""
    raw = io.raw_dir(SLUG)
    recs: list[dict[str, Any]] = []
    for year in YEARS:
        with (raw / _flare_list_filename(year)).open() as f:
            for r in csv.DictReader(f):
                try:
                    lon = float(r["longitude"])
                    lat = float(r["latitude"])
                except (KeyError, TypeError, ValueError):
                    continue
                if not _valid_lonlat(lon, lat):
                    continue
                catalog_id = (r.get("catalog_id") or "").strip()
                id_number = (r.get("id_number") or "").strip()
                sid = catalog_id if catalog_id and catalog_id != "-9999" else id_number
                recs.append(
                    {
                        "lon": lon,
                        "lat": lat,
                        "label": 0,  # single positive class: gas flare
                        "year": year,
                        "flr_volume_bcm": _num_or_none(r.get("flr_volume")),
                        "avg_temp_k": _num_or_none(r.get("avg_temp")),
                        "flr_type": (r.get("flr_type") or "").strip() or None,
                        "country_iso": (r.get("cntry_iso") or "").strip() or None,
                        "source_id": f"{year}/{sid}" if sid else f"{year}/idx",
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

    download_raw()

    recs = scan_records()
    print(f"scanned {len(recs)} flare records (years {YEARS})")

    # Single positive class -> balance_by_class caps the one class at the 25k total.
    selected = balance_by_class(recs, "label", per_class=MAX_SAMPLES_PER_DATASET)
    print(f"selected {len(selected)} points (<= {MAX_SAMPLES_PER_DATASET})")

    points: list[dict[str, Any]] = []
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
                # Auxiliary provenance (not the label):
                "detection_year": r["year"],
                "flr_volume_bcm": r["flr_volume_bcm"],
                "avg_temp_k": r["avg_temp_k"],
                "flr_type": r["flr_type"],
                "country_iso": r["country_iso"],
            }
        )
    io.write_points_table(SLUG, "classification", points)

    year_counts = Counter(r["year"] for r in selected)
    io.write_dataset_metadata(
        SLUG,
        {
            "dataset": SLUG,
            "name": "VIIRS Nightfire Gas Flaring",
            "task_type": "classification",
            "source": "NASA ORNL DAAC / EOG",
            "license": "open",
            "provenance": {
                "url": DOI,
                "have_locally": False,
                "annotation_method": "derived-product (VIIRS Nightfire IR detection)",
            },
            "sensors_relevant": ["sentinel2", "sentinel1", "landsat"],
            "classes": [{"id": 0, "name": CLASS_NAME, "description": CLASS_DESC}],
            "nodata_value": io.CLASS_NODATA,
            "num_samples": len(selected),
            "class_counts": {CLASS_NAME: len(selected)},
            "year_counts": {str(y): year_counts.get(y, 0) for y in YEARS},
            "notes": (
                "Positive-only, single-class (gas flare) sparse points; no background "
                "class (assembly supplies negatives, spec §5). 1x1 point labels -> one "
                "points.geojson (spec §2a). Each point carries a 1-year time_range "
                "anchored on its VIIRS detection year. ORNL DAAC 1874 spans 2012-2019; "
                "restricted to 2016-2019 (Sentinel era). Pooled catalog (31,358 valid "
                "records) sampled to the 25k cap. Flare-location precision ~VNF pixel "
                "(<= a few hundred m); pretraining snaps each point to one 10 m S2 pixel. "
                "flr_volume_bcm / avg_temp_k / flr_type / country_iso are auxiliary "
                "properties, not the label."
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
