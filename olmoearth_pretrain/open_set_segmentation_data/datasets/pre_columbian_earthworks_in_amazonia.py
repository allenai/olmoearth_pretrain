"""Pre-Columbian Earthworks in Amazonia -> open-set-segmentation point labels.

Source: Peripato et al., "More than 10,000 pre-Columbian earthworks are still hidden
throughout Amazonia", Science 382, 103-109 (2023). Data on Zenodo record 10214943
(DOI 10.5281/zenodo.7750985), the paper's code/data repo (Vperipato/ade2541). The
ground-truth layer is ``Database/Earthworks.rds`` inside the release zip: **961
confirmed, georeferenced pre-Columbian earthwork sites** (WGS84 lon/lat), compiled from
several archaeological databases (Amazon Arch, PAST, CNSA, INRAP & DAC, TREES/INPE, and
multi-source combinations). Each record carries only its source ``Database`` (provenance),
NOT an earthwork-type attribute -- the manifest's aspirational sub-classes (geoglyphs,
ring ditches, mound villages, ponds/wells, fortifications) are NOT present in the released
data, so we model a single presence phenomenon "pre-Columbian earthwork".

The record also ships two IPP-model probability rasters (linear/log10) at **1 km**
resolution. Those are a *model prediction* (a derived product), not reference ground
truth, and 1 km is far coarser than our 10 m label grid -- we do NOT use them; we use the
confirmed reference points.

Suitability at 10-30 m (spec 8): MIXED. Many Amazonian earthworks are small and/or lie
under closed forest canopy and are only detectable in LiDAR/VHR (e.g. the TREES/INPE
LiDAR-newly-detected sites). BUT a large fraction of the confirmed set are the big
deforested ditched enclosures ("geoglyphs")/ring ditches of Acre (Brazil), Bolivia and
Peru that are 100-300 m across and plainly resolvable in Sentinel-2/Landsat (the manifest
itself: "Geoglyphs/ring ditches 100-300 m; discernible at 10-30 m"). Per the task's
explicit guidance we KEEP the dataset as a WEAK single-phenomenon PRESENCE label at the
site point, with the caveat that some positives (under-canopy sites) will not be visible
in 10-30 m imagery -- these are noisy positives, documented in the summary.

Points carry a class -> dataset-wide point table (spec 2a), NOT per-point GeoTIFFs.

Class:
  0 pre-Columbian earthwork   (confirmed earthwork site: geoglyph / ditched enclosure /
                               ring ditch / mound village / pond-well / fortification)
Presence-only (no background/negative class): the assembly step supplies negatives from
other datasets (spec 5); we do NOT fabricate synthetic negatives.

Time range: persistent/static heritage sites -> a fixed representative 1-year Sentinel-era
window (2020).

Run (idempotent):
  python3 -m olmoearth_pretrain.open_set_segmentation_data.datasets.pre_columbian_earthworks_in_amazonia
"""

import argparse
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any

from olmoearth_pretrain.open_set_segmentation_data import download, io, manifest
from olmoearth_pretrain.open_set_segmentation_data.sampling import balance_by_class

SLUG = "pre_columbian_earthworks_in_amazonia"
NAME = "Pre-Columbian Earthworks in Amazonia"

ZENODO_RECORD = "10214943"
ZIP_NAME = "ade2541-v1.0.0.zip"
ZIP_URL = (
    f"https://zenodo.org/api/records/{ZENODO_RECORD}/files/Vperipato/{ZIP_NAME}/content"
)
RDS_RELPATH = "Vperipato-ade2541-78f685a/Database/Earthworks.rds"

CLASS_ID = 0
CLASS_NAME = "pre-Columbian earthwork"
CLASS_DESC = (
    "Confirmed, georeferenced pre-Columbian earthwork site in Amazonia (Brazil/Peru/"
    "Bolivia/Guianas): geoglyph/ditched enclosure, ring ditch, mound village, pond/well, "
    "or fortification. Compiled from multiple archaeological databases and LiDAR "
    "confirmation (Peripato et al., Science 2023). The released ground truth records only "
    "the site location and source database, not the earthwork sub-type, so all are treated "
    "as a single presence phenomenon."
)
PER_CLASS = 1000
STATIC_YEAR = 2020  # representative Sentinel-era year for these persistent sites


def ensure_data() -> str:
    """Download the Zenodo release zip and extract Earthworks.rds; return its local path."""
    raw = io.raw_dir(SLUG)
    raw.mkdir(parents=True, exist_ok=True)
    zip_path = raw / ZIP_NAME
    download.download_http(ZIP_URL, zip_path)
    unzip_root = Path(raw.path) / "unzip"
    rds = unzip_root / RDS_RELPATH
    if not rds.exists():
        unzip_root.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(Path(zip_path.path)) as zf:
            zf.extract(RDS_RELPATH, unzip_root)
    if not rds.exists():
        raise RuntimeError(f"expected {RDS_RELPATH} not found after unzip")
    with (raw / "SOURCE.txt").open("w") as f:
        f.write(
            "Pre-Columbian Earthworks in Amazonia.\n"
            "Peripato et al., Science 382, 103-109 (2023), doi:10.1126/science.ade2541.\n"
            f"Zenodo record {ZENODO_RECORD} (doi:10.5281/zenodo.7750985), file {ZIP_NAME}.\n"
            f"Ground truth: {RDS_RELPATH} -- 961 confirmed earthwork sites (lon/lat + source db).\n"
            "License: open (Zenodo 'other-open'); cite the paper + repository.\n"
            "NOT used: IPPModel_EarthworkProb-{linear,log10}.tif (1 km model prediction).\n"
        )
    return str(rds)


def load_records(rds_path: str) -> list[dict[str, Any]]:
    """Read the confirmed-earthwork points -> list of records (lon/lat/label/source_id)."""
    import pyreadr

    df = pyreadr.read_r(rds_path)[None]
    records: list[dict[str, Any]] = []
    for i, row in df.iterrows():
        lon, lat = row["Longitude"], row["Latitude"]
        if lon is None or lat is None:
            continue
        try:
            lon = float(lon)
            lat = float(lat)
        except (TypeError, ValueError):
            continue
        db = str(row.get("Database", "")).strip() or "unknown"
        records.append(
            {
                "lon": lon,
                "lat": lat,
                "label": CLASS_ID,
                "source_id": f"{db} #{i}",
            }
        )
    return records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.parse_args()

    io.check_disk()
    manifest.write_registry_entry(SLUG, "in_progress")

    rds_path = ensure_data()
    records = load_records(rds_path)
    print(f"loaded {len(records)} confirmed earthwork sites")

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
            "source": "Peripato et al., Science 2023 (Zenodo 10214943)",
            "license": "open (Zenodo other-open)",
            "provenance": {
                "url": "https://zenodo.org/records/10214943",
                "paper_doi": "10.1126/science.ade2541",
                "file": f"{ZIP_NAME}:{RDS_RELPATH}",
                "have_locally": False,
                "annotation_method": "expert-compiled from archaeological databases + LiDAR-confirmed",
            },
            "sensors_relevant": ["sentinel2", "sentinel1", "landsat"],
            "classes": [
                {"id": CLASS_ID, "name": CLASS_NAME, "description": CLASS_DESC}
            ],
            "nodata_value": io.CLASS_NODATA,
            "num_samples": len(selected),
            "class_counts": {CLASS_NAME: counts.get(CLASS_ID, 0)},
            "notes": (
                "Weak single-phenomenon presence label at confirmed pre-Columbian earthwork "
                "site points. 1x1 point segmentation via points.geojson (spec 2a). "
                "Released ground truth carries only lon/lat + source database (no earthwork "
                "sub-type), so the manifest's sub-classes are collapsed to one presence class. "
                f"Persistent static sites -> fixed {STATIC_YEAR} 1-year window. Presence-only: "
                "no explicit negative/background class (assembly supplies negatives). "
                "OBSERVABILITY CAVEAT (spec 8): MIXED at 10-30 m -- large deforested "
                "geoglyphs/ring ditches (100-300 m) are resolvable in S2/Landsat, but some "
                "sites lie under forest canopy and are LiDAR/VHR-only, i.e. noisy positives. "
                "The 1 km IPP probability raster (model prediction) is NOT used."
            ),
        },
    )
    manifest.write_registry_entry(
        SLUG, "completed", task_type="classification", num_samples=len(selected)
    )
    print("done:", dict(counts))


if __name__ == "__main__":
    main()
